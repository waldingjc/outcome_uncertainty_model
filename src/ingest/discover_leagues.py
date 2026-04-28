"""One-shot discovery of every league + season combination api-football knows
about, seeded into `backfill_jobs` for the historical backfill runner to
consume.

We call /leagues with no filter — that returns every league across all
countries, with each league's full season list. For every league we enqueue
a (league_id, season) job for each season we want to ingest. The Free plan
covers 2022-2024, so by default we enqueue exactly those years; on a paid
plan you can pass `--from-season 2010` (or similar) to widen the range.

`backfill_jobs` uses INSERT OR IGNORE so re-running discovery is safe — it
just adds combos for any newly-introduced leagues / seasons without
disturbing existing job statuses.
"""

from __future__ import annotations

import argparse
import logging
from datetime import datetime, timezone

from src.db.schema import get_connection, init_db
from src.ingest.api_football import request_v3

logger = logging.getLogger(__name__)

# Free plan grants seasons 2022, 2023, 2024 only. Override on a paid plan.
DEFAULT_FROM_SEASON = 2022
DEFAULT_TO_SEASON = 2024


def _fetch_all_leagues() -> list[dict]:
    """Fetch every league from /leagues. The endpoint returns the full set in
    a single response (it does not support pagination), so this is one call.
    """
    data, remaining = request_v3("leagues", {})
    leagues = data.get("response", [])
    logger.info(
        "Discovery: collected %d leagues (daily calls remaining: %s)",
        len(leagues), remaining,
    )
    return leagues


def _enqueue_jobs(
    leagues_payload: list[dict], from_season: int, to_season: int
) -> tuple[int, int]:
    """Insert (league_id, season) combos into backfill_jobs.

    Returns (combos_considered, combos_inserted).
    """
    rows: list[tuple[int, int]] = []
    for entry in leagues_payload:
        league = entry.get("league") or {}
        league_id = league.get("id")
        if league_id is None:
            continue
        for season_meta in entry.get("seasons", []):
            year = season_meta.get("year")
            if year is None:
                continue
            if from_season <= year <= to_season:
                rows.append((league_id, year))

    if not rows:
        return 0, 0

    with get_connection() as conn:
        before = conn.execute("SELECT COUNT(*) AS n FROM backfill_jobs").fetchone()["n"]
        conn.executemany(
            "INSERT OR IGNORE INTO backfill_jobs (league_id, season) VALUES (?, ?)",
            rows,
        )
        after = conn.execute("SELECT COUNT(*) AS n FROM backfill_jobs").fetchone()["n"]
    return len(rows), after - before


def discover(from_season: int = DEFAULT_FROM_SEASON, to_season: int = DEFAULT_TO_SEASON) -> dict:
    """Run discovery end-to-end. Returns a summary dict."""
    init_db()
    started = datetime.now(timezone.utc).isoformat()

    leagues_payload = _fetch_all_leagues()
    considered, inserted = _enqueue_jobs(leagues_payload, from_season, to_season)

    summary = {
        "leagues_discovered": len(leagues_payload),
        "season_range": [from_season, to_season],
        "combos_considered": considered,
        "new_jobs_inserted": inserted,
        "started_at": started,
    }
    logger.info("Discovery summary: %s", summary)
    return summary


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    parser = argparse.ArgumentParser(description="Seed backfill_jobs from /leagues")
    parser.add_argument(
        "--from-season", type=int, default=DEFAULT_FROM_SEASON,
        help=f"Earliest season to enqueue (default: {DEFAULT_FROM_SEASON} — Free plan floor)",
    )
    parser.add_argument(
        "--to-season", type=int, default=DEFAULT_TO_SEASON,
        help=f"Latest season to enqueue (default: {DEFAULT_TO_SEASON} — Free plan ceiling)",
    )
    args = parser.parse_args()

    summary = discover(from_season=args.from_season, to_season=args.to_season)
    print(
        f"Discovery complete — {summary['leagues_discovered']} leagues seen, "
        f"{summary['combos_considered']} (league, season) combos in range, "
        f"{summary['new_jobs_inserted']} new jobs enqueued."
    )
