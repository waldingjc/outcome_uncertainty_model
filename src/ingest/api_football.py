"""Ingest past fixture results from api-football (api-football.com / RapidAPI).

Usage:
    python -m src.ingest.api_football --league 39 --season 2023

Environment variables:
    API_FOOTBALL_KEY   Your api-football.com or RapidAPI key (required)
    API_FOOTBALL_HOST  Override host. Defaults to v3.football.api-sports.io
                       (direct api-football.com sign-up). Set to
                       api-football-v1.p.rapidapi.com if your key was
                       provisioned through RapidAPI.

Authentication header is selected automatically from the host:
  - Direct (v3.football.api-sports.io)        -> x-apisports-key: KEY
  - RapidAPI (api-football-v1.p.rapidapi.com) -> X-RapidAPI-Key + X-RapidAPI-Host
"""

from __future__ import annotations

import json
import logging
import os
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path

import requests

from src import config as _config  # noqa: F401  -- import for side effect: .env load
from src.db.schema import get_connection, init_db

# api-football retains odds for 7 days after the fixture date. Anything older
# than this at insert time is permanently odds-less.
ODDS_RETENTION_DAYS = 7

# Cache of the last seen `x-ratelimit-requests-remaining` value, so callers
# (e.g. the daily runner / backfill) can make budget decisions between calls
# without making an extra request just to check.
_last_remaining: int | None = None

logger = logging.getLogger(__name__)

_DIRECT_HOST = "v3.football.api-sports.io"  # api-football.com direct sign-up
_RAPIDAPI_HOST = "api-football-v1.p.rapidapi.com"  # RapidAPI marketplace
_DEFAULT_HOST = _DIRECT_HOST
_RAW_DIR = Path(__file__).parents[2] / "data" / "raw"
# Free tier: ~10 req/min, 100 req/day. 6.5s between calls keeps us comfortably
# under the per-minute limit. Override via API_FOOTBALL_DELAY_S env var on
# paid plans where this is overly cautious.
_REQUEST_DELAY_S = float(os.environ.get("API_FOOTBALL_DELAY_S", "6.5"))


def _host() -> str:
    return os.environ.get("API_FOOTBALL_HOST", _DEFAULT_HOST)


def _is_rapidapi(host: str) -> bool:
    return "rapidapi" in host.lower()


def _get_headers() -> dict:
    api_key = os.environ.get("API_FOOTBALL_KEY")
    if not api_key:
        raise EnvironmentError("API_FOOTBALL_KEY environment variable not set")
    host = _host()
    if _is_rapidapi(host):
        return {"X-RapidAPI-Key": api_key, "X-RapidAPI-Host": host}
    return {"x-apisports-key": api_key}


def _base_url() -> str:
    host = _host()
    # RapidAPI puts the v3 segment in the path; the direct host already
    # encodes it in the hostname (v3.football.api-sports.io).
    return f"https://{host}/v3" if _is_rapidapi(host) else f"https://{host}"


def request_v3(path: str, params: dict) -> tuple[dict, int | None]:
    """Make a GET against api-football v3.

    Returns (json_body, daily_calls_remaining). The remaining-call count comes
    from the `x-ratelimit-requests-remaining` response header (daily quota);
    `None` if the header is absent.

    Note: api-football returns HTTP 200 even on application-level errors
    (plan restrictions, invalid params, etc.). The error detail lives in the
    response body's `errors` field, which we log as a WARNING here so callers
    don't silently see an empty `response` array.
    """
    headers = _get_headers()
    url = f"{_base_url()}/{path.lstrip('/')}"
    response = requests.get(url, headers=headers, params=params, timeout=30)
    response.raise_for_status()
    raw_remaining = response.headers.get("x-ratelimit-requests-remaining")
    remaining = int(raw_remaining) if raw_remaining is not None else None
    if remaining is not None:
        global _last_remaining
        _last_remaining = remaining

    body = response.json()
    errors = body.get("errors")
    # api-football uses `errors: []` for "no errors" and `errors: {...}` for actual errors.
    if isinstance(errors, dict) and errors:
        logger.warning(
            "api-football %s returned errors: %s (params=%s)", path, errors, params,
        )
    return body, remaining


def _fetch_fixtures_page(
    league_id: int,
    season: int,
    page: int,
    status: str | None = "FT",
    from_date: str | None = None,
    to_date: str | None = None,
) -> tuple[dict, int | None]:
    """Fetch one page of fixtures from the API.

    Date filters use api-football's `from` / `to` params (YYYY-MM-DD). They are
    optional; supplying them narrows the response to that date range.

    NOTE: api-football.com's direct API rejects `page=1` with
    `errors: {page: "The Page field do not exist."}`. Pagination is only
    valid for page >= 2 — for page 1, omit the parameter entirely.
    """
    params: dict = {
        "league": league_id,
        "season": season,
    }
    if page >= 2:
        params["page"] = page
    if status is not None:
        params["status"] = status
    if from_date is not None:
        params["from"] = from_date
    if to_date is not None:
        params["to"] = to_date
    return request_v3("fixtures", params)


def _save_raw(data: dict, league_id: int, season: int, page: int) -> None:
    """Persist the raw API response as JSON for future reprocessing."""
    _RAW_DIR.mkdir(parents=True, exist_ok=True)
    path = _RAW_DIR / f"fixtures_l{league_id}_s{season}_p{page}.json"
    path.write_text(json.dumps(data, indent=2), encoding="utf-8")


def _is_odds_unavailable(fixture_date_str: str | None) -> int:
    """Return 1 if the fixture is older than the odds retention window at ingest time."""
    if not fixture_date_str:
        return 0
    try:
        fixture_dt = datetime.fromisoformat(fixture_date_str.replace("Z", "+00:00"))
    except ValueError:
        return 0
    if fixture_dt.tzinfo is None:
        fixture_dt = fixture_dt.replace(tzinfo=timezone.utc)
    age = datetime.now(timezone.utc) - fixture_dt
    return 1 if age > timedelta(days=ODDS_RETENTION_DAYS) else 0


def _parse_fixture(raw: dict) -> dict:
    """Flatten a single fixture response object into a row dict."""
    fix = raw["fixture"]
    league = raw["league"]
    teams = raw["teams"]
    goals = raw["goals"]
    score = raw["score"]
    fixture_date = fix.get("date")

    return {
        "fixture_id": fix["id"],
        "date": fixture_date,
        "league_id": league["id"],
        "league_name": league["name"],
        "season": league["season"],
        "round": league.get("round"),
        "home_team_id": teams["home"]["id"],
        "home_team_name": teams["home"]["name"],
        "away_team_id": teams["away"]["id"],
        "away_team_name": teams["away"]["name"],
        "home_goals": goals.get("home"),
        "away_goals": goals.get("away"),
        "home_goals_ht": (score.get("halftime") or {}).get("home"),
        "away_goals_ht": (score.get("halftime") or {}).get("away"),
        "status": fix["status"]["short"],
        "venue_name": (fix.get("venue") or {}).get("name"),
        "venue_city": (fix.get("venue") or {}).get("city"),
        "referee": fix.get("referee"),
        "odds_unavailable": _is_odds_unavailable(fixture_date),
        "ingested_at": datetime.now(timezone.utc).isoformat(),
    }


def _upsert_fixtures(rows: list[dict]) -> int:
    """Insert fixture rows, updating mutable fields on conflict.

    On conflict (fixture_id), `odds_unavailable` is preserved (never downgraded
    from 1 to 0) and `ingested_at` is left untouched so we keep the original
    insert time. Everything else (scores, status, etc.) is refreshed.
    """
    if not rows:
        return 0
    with get_connection() as conn:
        conn.executemany(
            """
            INSERT INTO fixtures (
                fixture_id, date, league_id, league_name, season, round,
                home_team_id, home_team_name, away_team_id, away_team_name,
                home_goals, away_goals, home_goals_ht, away_goals_ht,
                status, venue_name, venue_city, referee,
                odds_unavailable, ingested_at
            ) VALUES (
                :fixture_id, :date, :league_id, :league_name, :season, :round,
                :home_team_id, :home_team_name, :away_team_id, :away_team_name,
                :home_goals, :away_goals, :home_goals_ht, :away_goals_ht,
                :status, :venue_name, :venue_city, :referee,
                :odds_unavailable, :ingested_at
            )
            ON CONFLICT(fixture_id) DO UPDATE SET
                date           = excluded.date,
                league_name    = excluded.league_name,
                season         = excluded.season,
                round          = excluded.round,
                home_team_id   = excluded.home_team_id,
                home_team_name = excluded.home_team_name,
                away_team_id   = excluded.away_team_id,
                away_team_name = excluded.away_team_name,
                home_goals     = excluded.home_goals,
                away_goals     = excluded.away_goals,
                home_goals_ht  = excluded.home_goals_ht,
                away_goals_ht  = excluded.away_goals_ht,
                status         = excluded.status,
                venue_name     = excluded.venue_name,
                venue_city     = excluded.venue_city,
                referee        = excluded.referee,
                odds_unavailable = MAX(fixtures.odds_unavailable, excluded.odds_unavailable)
            """,
            rows,
        )
    return len(rows)


def _record_run(league_id: int, season: int, fixtures_saved: int) -> None:
    with get_connection() as conn:
        conn.execute(
            "INSERT INTO ingest_runs (league_id, season, fixtures_saved) VALUES (?, ?, ?)",
            (league_id, season, fixtures_saved),
        )


def ingest_results(
    league_id: int,
    season: int,
    save_raw: bool = True,
    status: str | None = "FT",
    from_date: str | None = None,
    to_date: str | None = None,
) -> int:
    """Fetch fixtures for a league/season and store them in SQLite.

    Args:
        league_id:  api-football league ID (e.g. 39 = English Premier League)
        season:     Four-digit season year (e.g. 2023 for 2023/24)
        save_raw:   If True, also write raw JSON responses to data/raw/
        status:     Filter by api-football status code (default "FT" = finished).
                    Pass None to fetch every status (useful for backfill where
                    we want everything that was ever scheduled).
        from_date:  Optional ISO date string (YYYY-MM-DD) — earliest fixture date.
        to_date:    Optional ISO date string (YYYY-MM-DD) — latest fixture date.

    Returns:
        Total number of fixtures saved.
    """
    init_db()
    total_saved = 0
    page = 1

    logger.info(
        "Starting ingest: league=%d season=%d status=%s from=%s to=%s",
        league_id, season, status, from_date, to_date,
    )

    while True:
        logger.debug("Fetching page %d", page)
        data, remaining = _fetch_fixtures_page(
            league_id, season, page,
            status=status, from_date=from_date, to_date=to_date,
        )

        if save_raw:
            _save_raw(data, league_id, season, page)

        fixtures_raw = data.get("response", [])
        if not fixtures_raw:
            break

        rows = [_parse_fixture(f) for f in fixtures_raw]
        saved = _upsert_fixtures(rows)
        total_saved += saved
        logger.info(
            "Page %d: saved %d fixtures (total %d, daily calls remaining: %s)",
            page, saved, total_saved, remaining,
        )

        paging = data.get("paging", {})
        if page >= paging.get("total", 1):
            break

        page += 1
        time.sleep(_REQUEST_DELAY_S)

    _record_run(league_id, season, total_saved)
    logger.info("Ingest complete: %d fixtures saved", total_saved)
    return total_saved


if __name__ == "__main__":
    import argparse

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    parser = argparse.ArgumentParser(description="Ingest past fixtures from api-football")
    parser.add_argument("--league", type=int, required=True, help="api-football league ID")
    parser.add_argument("--season", type=int, required=True, help="Season year (e.g. 2023)")
    parser.add_argument("--no-raw", action="store_true", help="Skip saving raw JSON responses")
    args = parser.parse_args()

    count = ingest_results(args.league, args.season, save_raw=not args.no_raw)
    print(f"Done — {count} fixtures saved.")
