"""Daily ingestion job — runs end-to-end fixture + odds capture.

Two phases:
    1. Fixture refresh — for each tracked league, ingest fixtures dated within
       the last 7 days (current season). Cheap (~1-2 calls per league) and
       picks up newly-finished matches plus any updates to scores/referee.
    2. Odds backfill — for each FT fixture in the DB whose date falls within
       the last 7 days, has no odds yet, and isn't flagged as odds-unavailable,
       call /odds?fixture={id}. Stops when daily budget hits a configurable
       floor.

Designed to run once per day via Windows Task Scheduler. Idempotent: safe to
re-run without duplicating data.
"""

from __future__ import annotations

import argparse
import logging
import time
from datetime import date, timedelta

from src.db.schema import get_connection, init_db
from src.ingest import api_football as af
from src.ingest.api_football import ingest_results
from src.ingest.historical_backfill import current_season
from src.ingest.odds import fetch_odds_for_fixture

logger = logging.getLogger(__name__)

DEFAULT_LOOKBACK_DAYS = 7
# Reserve some headroom — never spend the last few calls of the day in case
# something else needs to call the API ad-hoc.
DEFAULT_MIN_BUDGET = 5


def _active_tracked_leagues() -> list[tuple[int, str]]:
    with get_connection() as conn:
        rows = conn.execute(
            "SELECT league_id, name FROM tracked_leagues WHERE active = 1 ORDER BY league_id"
        ).fetchall()
    return [(r["league_id"], r["name"]) for r in rows]


def _fixtures_needing_odds(lookback_days: int) -> list[int]:
    """FT fixtures from the last N days with no odds yet, not flagged unavailable."""
    cutoff = (date.today() - timedelta(days=lookback_days)).isoformat()
    sql = """
        SELECT f.fixture_id
        FROM fixtures f
        WHERE f.status = 'FT'
          AND f.odds_unavailable = 0
          AND date(f.date) >= date(?)
          AND f.fixture_id NOT IN (SELECT DISTINCT fixture_id FROM odds)
        ORDER BY f.date DESC
    """
    with get_connection() as conn:
        rows = conn.execute(sql, (cutoff,)).fetchall()
    return [r["fixture_id"] for r in rows]


def _budget_ok(min_budget: int) -> bool:
    """True if last-known daily budget is unknown or above the floor."""
    remaining = af._last_remaining
    if remaining is None:
        return True
    return remaining > min_budget


def run_daily(
    lookback_days: int = DEFAULT_LOOKBACK_DAYS,
    min_budget: int = DEFAULT_MIN_BUDGET,
    save_raw: bool = True,
) -> dict:
    """Run phase 1 (fixture refresh) then phase 2 (odds backfill). Returns summary."""
    init_db()
    leagues = _active_tracked_leagues()
    if not leagues:
        logger.warning("No active tracked leagues; nothing to do.")
        return {"leagues": 0, "fixtures_refreshed": 0, "odds_fixtures": 0, "odds_rows": 0}

    season = current_season()
    today = date.today()
    from_date = (today - timedelta(days=lookback_days)).isoformat()
    to_date = today.isoformat()

    # ---- Phase 1: fixture refresh ----------------------------------------
    fixtures_refreshed = 0
    logger.info(
        "Phase 1: refreshing fixtures for %d leagues, season=%d, %s..%s",
        len(leagues), season, from_date, to_date,
    )
    for league_id, league_name in leagues:
        if not _budget_ok(min_budget):
            logger.warning("Budget floor reached during phase 1; stopping.")
            break
        try:
            saved = ingest_results(
                league_id, season, save_raw=save_raw,
                status="FT", from_date=from_date, to_date=to_date,
            )
            fixtures_refreshed += saved
        except Exception:
            logger.exception("Phase 1 failed for %s (id=%d)", league_name, league_id)

    # ---- Phase 2: odds backfill ------------------------------------------
    pending = _fixtures_needing_odds(lookback_days)
    logger.info("Phase 2: %d fixtures need odds (lookback %d days)", len(pending), lookback_days)

    odds_fixtures = 0
    odds_rows = 0
    stopped_early = False
    for fixture_id in pending:
        if not _budget_ok(min_budget):
            logger.warning(
                "Budget floor (%d) reached after %d odds fetches; %d fixtures still pending.",
                min_budget, odds_fixtures, len(pending) - odds_fixtures,
            )
            stopped_early = True
            break
        try:
            written, _ = fetch_odds_for_fixture(fixture_id, save_raw=save_raw)
            odds_fixtures += 1
            odds_rows += written
        except Exception:
            logger.exception("Odds fetch failed for fixture %d", fixture_id)
        time.sleep(af._REQUEST_DELAY_S)

    summary = {
        "leagues": len(leagues),
        "fixtures_refreshed": fixtures_refreshed,
        "odds_fixtures": odds_fixtures,
        "odds_rows": odds_rows,
        "odds_pending_after_run": max(0, len(pending) - odds_fixtures),
        "stopped_early": stopped_early,
        "daily_calls_remaining": af._last_remaining,
    }
    logger.info("Daily run summary: %s", summary)
    return summary


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    parser = argparse.ArgumentParser(description="Daily fixture + odds ingestion run")
    parser.add_argument(
        "--lookback-days", type=int, default=DEFAULT_LOOKBACK_DAYS,
        help=f"How many days back to refresh / backfill (default: {DEFAULT_LOOKBACK_DAYS})",
    )
    parser.add_argument(
        "--min-budget", type=int, default=DEFAULT_MIN_BUDGET,
        help=f"Stop when daily calls remaining <= this (default: {DEFAULT_MIN_BUDGET})",
    )
    parser.add_argument("--no-raw", action="store_true", help="Skip saving raw JSON responses")
    args = parser.parse_args()

    summary = run_daily(
        lookback_days=args.lookback_days,
        min_budget=args.min_budget,
        save_raw=not args.no_raw,
    )
    print(
        f"Daily run complete — "
        f"{summary['fixtures_refreshed']} fixtures refreshed, "
        f"{summary['odds_fixtures']} odds fetched ({summary['odds_rows']} rows), "
        f"{summary['odds_pending_after_run']} still pending. "
        f"Remaining daily budget: {summary['daily_calls_remaining']}."
    )
