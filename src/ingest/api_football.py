"""Ingest past fixture results and pre-match odds from api-football (api-football.com).

Usage:
    python -m src.ingest.api_football --league 39 --season 2024

Environment variables:
    API_FOOTBALL_KEY   Your api-football.com API key (required)
"""

import json
import logging
import os
import time
from datetime import datetime
from pathlib import Path

import requests

from src.db.schema import get_connection, init_db

logger = logging.getLogger(__name__)

_BASE_URL = "https://v3.football.api-sports.io"
_RAW_DIR = Path(__file__).parents[2] / "data" / "raw"
# api-football free tier: 100 req/day; paid tiers are higher.
_REQUEST_DELAY_S = 0.5
# api-football bet ID for the Match Winner (1X2) market
_MATCH_WINNER_BET_ID = 1


def _get_headers() -> dict:
    api_key = os.environ.get("API_FOOTBALL_KEY")
    if not api_key:
        raise EnvironmentError("API_FOOTBALL_KEY environment variable not set")
    return {"x-apisports-key": api_key}


def _get(endpoint: str, params: dict) -> dict:
    response = requests.get(
        f"{_BASE_URL}/{endpoint}",
        headers=_get_headers(),
        params=params,
        timeout=30,
    )
    response.raise_for_status()
    return response.json()


def _paginated(endpoint: str, base_params: dict, raw_prefix: str, save_raw: bool):
    """Yield response items across all pages for a given endpoint."""
    page = 1
    while True:
        params = {**base_params}
        if page > 1:
            params["page"] = page

        data = _get(endpoint, params)

        if save_raw:
            _RAW_DIR.mkdir(parents=True, exist_ok=True)
            path = _RAW_DIR / f"{raw_prefix}_p{page}.json"
            path.write_text(json.dumps(data, indent=2), encoding="utf-8")

        items = data.get("response", [])
        if not items:
            break

        yield from items

        paging = data.get("paging", {})
        if page >= paging.get("total", 1):
            break

        page += 1
        time.sleep(_REQUEST_DELAY_S)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def _parse_fixture(raw: dict) -> dict:
    fix = raw["fixture"]
    league = raw["league"]
    teams = raw["teams"]
    goals = raw["goals"]
    score = raw["score"]

    return {
        "fixture_id": fix["id"],
        "date": fix.get("date"),
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
        "ingested_at": datetime.utcnow().isoformat(),
    }


def _upsert_fixtures(rows: list[dict]) -> int:
    if not rows:
        return 0
    with get_connection() as conn:
        conn.executemany(
            """
            INSERT OR REPLACE INTO fixtures (
                fixture_id, date, league_id, league_name, season, round,
                home_team_id, home_team_name, away_team_id, away_team_name,
                home_goals, away_goals, home_goals_ht, away_goals_ht,
                status, venue_name, venue_city, referee, ingested_at
            ) VALUES (
                :fixture_id, :date, :league_id, :league_name, :season, :round,
                :home_team_id, :home_team_name, :away_team_id, :away_team_name,
                :home_goals, :away_goals, :home_goals_ht, :away_goals_ht,
                :status, :venue_name, :venue_city, :referee, :ingested_at
            )
            """,
            rows,
        )
    return len(rows)


def ingest_results(league_id: int, season: int, save_raw: bool = True) -> int:
    """Fetch all finished fixtures for a league/season and store them in SQLite."""
    rows = [
        _parse_fixture(f)
        for f in _paginated(
            "fixtures",
            {"league": league_id, "season": season, "status": "FT"},
            raw_prefix=f"fixtures_l{league_id}_s{season}",
            save_raw=save_raw,
        )
    ]
    saved = _upsert_fixtures(rows)
    logger.info("Fixtures: saved %d", saved)
    return saved


# ---------------------------------------------------------------------------
# Odds
# ---------------------------------------------------------------------------

def _parse_odds(raw: dict) -> list[dict]:
    """Extract Match Winner (1X2) odds for every bookmaker in one fixture response."""
    fixture_id = raw["fixture"]["id"]
    ingested_at = datetime.utcnow().isoformat()
    rows = []

    for bookmaker in raw.get("bookmakers", []):
        for bet in bookmaker.get("bets", []):
            if bet["id"] != _MATCH_WINNER_BET_ID:
                continue

            # Values list is ordered: Home, Draw, Away
            values = {v["value"]: v["odd"] for v in bet.get("values", [])}
            rows.append({
                "fixture_id": fixture_id,
                "bookmaker_id": bookmaker["id"],
                "bookmaker_name": bookmaker["name"],
                "home_odds": _to_float(values.get("Home")),
                "draw_odds": _to_float(values.get("Draw")),
                "away_odds": _to_float(values.get("Away")),
                "ingested_at": ingested_at,
            })

    return rows


def _to_float(value):
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _upsert_odds(rows: list[dict]) -> int:
    if not rows:
        return 0
    with get_connection() as conn:
        conn.executemany(
            """
            INSERT OR REPLACE INTO odds (
                fixture_id, bookmaker_id, bookmaker_name,
                home_odds, draw_odds, away_odds, ingested_at
            ) VALUES (
                :fixture_id, :bookmaker_id, :bookmaker_name,
                :home_odds, :draw_odds, :away_odds, :ingested_at
            )
            """,
            rows,
        )
    return len(rows)


def ingest_odds(league_id: int, season: int, save_raw: bool = True) -> int:
    """Fetch pre-match Match Winner odds for a league/season and store in SQLite."""
    all_rows = []
    for fixture_raw in _paginated(
        "odds",
        {"league": league_id, "season": season},
        raw_prefix=f"odds_l{league_id}_s{season}",
        save_raw=save_raw,
    ):
        all_rows.extend(_parse_odds(fixture_raw))

    saved = _upsert_odds(all_rows)
    if saved == 0:
        logger.warning(
            "Odds: no data returned for league=%d season=%d. "
            "api-football only retains pre-match odds for 7 days — "
            "run this ingest before fixtures are played to capture odds.",
            league_id, season,
        )
    else:
        logger.info("Odds: saved %d bookmaker rows", saved)
    return saved


# ---------------------------------------------------------------------------
# Orchestrator
# ---------------------------------------------------------------------------

def ingest_league_season(league_id: int, season: int, save_raw: bool = True) -> None:
    """Ingest fixtures and odds for a league/season in one call."""
    init_db()
    logger.info("Starting ingest: league=%d season=%d", league_id, season)

    fixtures_saved = ingest_results(league_id, season, save_raw)
    odds_saved = ingest_odds(league_id, season, save_raw)

    with get_connection() as conn:
        conn.execute(
            "INSERT INTO ingest_runs (league_id, season, fixtures_saved, odds_saved) VALUES (?, ?, ?, ?)",
            (league_id, season, fixtures_saved, odds_saved),
        )

    logger.info("Done — %d fixtures, %d odds rows", fixtures_saved, odds_saved)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    parser = argparse.ArgumentParser(description="Ingest fixtures and odds from api-football")
    parser.add_argument("--league", type=int, required=True, help="api-football league ID")
    parser.add_argument("--season", type=int, required=True, help="Season year (e.g. 2024)")
    parser.add_argument("--no-raw", action="store_true", help="Skip saving raw JSON responses")
    parser.add_argument("--odds-only", action="store_true", help="Only ingest odds, skip fixtures")
    parser.add_argument("--fixtures-only", action="store_true", help="Only ingest fixtures, skip odds")
    args = parser.parse_args()

    init_db()

    if args.odds_only:
        ingest_odds(args.league, args.season, save_raw=not args.no_raw)
    elif args.fixtures_only:
        ingest_results(args.league, args.season, save_raw=not args.no_raw)
    else:
        ingest_league_season(args.league, args.season, save_raw=not args.no_raw)
