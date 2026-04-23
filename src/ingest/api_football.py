"""Ingest past fixture results from api-football (api-football.com / RapidAPI).

Usage:
    python -m src.ingest.api_football --league 39 --season 2023

Environment variables:
    API_FOOTBALL_KEY   Your RapidAPI or api-football.com API key (required)
    API_FOOTBALL_HOST  Override host (default: api-football-v1.p.rapidapi.com)
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

_BASE_URL = "https://{host}/v3"
_DEFAULT_HOST = "api-football-v1.p.rapidapi.com"
_RAW_DIR = Path(__file__).parents[2] / "data" / "raw"
# api-football free tier: 100 req/day; paid tiers are higher.
_REQUEST_DELAY_S = 0.5


def _get_headers() -> dict:
    api_key = os.environ.get("API_FOOTBALL_KEY")
    if not api_key:
        raise EnvironmentError("API_FOOTBALL_KEY environment variable not set")
    host = os.environ.get("API_FOOTBALL_HOST", _DEFAULT_HOST)
    return {
        "X-RapidAPI-Key": api_key,
        "X-RapidAPI-Host": host,
    }


def _base_url() -> str:
    host = os.environ.get("API_FOOTBALL_HOST", _DEFAULT_HOST)
    return _BASE_URL.format(host=host)


def _fetch_fixtures_page(league_id: int, season: int, page: int) -> dict:
    """Fetch one page of finished fixtures from the API."""
    headers = _get_headers()
    params = {
        "league": league_id,
        "season": season,
        "status": "FT",  # full-time only
        "page": page,
    }
    url = f"{_base_url()}/fixtures"
    response = requests.get(url, headers=headers, params=params, timeout=30)
    response.raise_for_status()
    return response.json()


def _save_raw(data: dict, league_id: int, season: int, page: int) -> None:
    """Persist the raw API response as JSON for future reprocessing."""
    _RAW_DIR.mkdir(parents=True, exist_ok=True)
    path = _RAW_DIR / f"fixtures_l{league_id}_s{season}_p{page}.json"
    path.write_text(json.dumps(data, indent=2), encoding="utf-8")


def _parse_fixture(raw: dict) -> dict:
    """Flatten a single fixture response object into a row dict."""
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
    """Insert or replace fixture rows; returns count of rows written."""
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


def _record_run(league_id: int, season: int, fixtures_saved: int) -> None:
    with get_connection() as conn:
        conn.execute(
            "INSERT INTO ingest_runs (league_id, season, fixtures_saved) VALUES (?, ?, ?)",
            (league_id, season, fixtures_saved),
        )


def ingest_results(league_id: int, season: int, save_raw: bool = True) -> int:
    """Fetch all finished fixtures for a league/season and store them in SQLite.

    Args:
        league_id:  api-football league ID (e.g. 39 = English Premier League)
        season:     Four-digit season year (e.g. 2023 for 2023/24)
        save_raw:   If True, also write raw JSON responses to data/raw/

    Returns:
        Total number of fixtures saved.
    """
    init_db()
    total_saved = 0
    page = 1

    logger.info("Starting ingest: league=%d season=%d", league_id, season)

    while True:
        logger.debug("Fetching page %d", page)
        data = _fetch_fixtures_page(league_id, season, page)

        if save_raw:
            _save_raw(data, league_id, season, page)

        fixtures_raw = data.get("response", [])
        if not fixtures_raw:
            break

        rows = [_parse_fixture(f) for f in fixtures_raw]
        saved = _upsert_fixtures(rows)
        total_saved += saved
        logger.info("Page %d: saved %d fixtures (total so far: %d)", page, saved, total_saved)

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
