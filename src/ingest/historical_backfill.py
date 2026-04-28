"""Queue-driven historical backfill.

Consumes pending jobs from `backfill_jobs` (one row per league_id × season),
fetching every fixture for that combo and updating job status. Designed to
run daily — each invocation pops as many pending jobs as the daily API
budget allows, then exits cleanly. Re-run tomorrow and it picks up where it
left off.

Job ordering:
  1. Tracked leagues, by `tracked_leagues.priority` ASC (NULLs last).
  2. Then by league_id ASC.
  3. Within a league, season DESC (newest first — most relevant data lands
     in the DB earliest in case the run is interrupted).

Job statuses:
  - pending     — not yet attempted, eligible for processing.
  - completed   — fixtures fetched and stored.
  - no_access   — api-football returned a `errors.plan` message; we won't
                  retry this on the same plan.
  - failed      — transient error (HTTP failure, parse error, etc.). Stays
                  pending in spirit but is recorded so we can investigate;
                  reset to 'pending' manually to retry.

Usage:
    python -m src.ingest.historical_backfill
    python -m src.ingest.historical_backfill --min-budget 5
"""

from __future__ import annotations

import argparse
import logging
import time
from datetime import date, datetime, timezone

from src.db.schema import get_connection, init_db
from src.ingest import api_football as af
from src.ingest.api_football import (
    _REQUEST_DELAY_S, _parse_fixture, _save_raw, _upsert_fixtures,
    _fetch_fixtures_page, _record_run,
)

logger = logging.getLogger(__name__)

# Stop the daily run once the remaining budget hits this floor.
DEFAULT_MIN_BUDGET = 5


def current_season(today: date | None = None) -> int:
    """api-football season number for the season currently in progress.

    Football seasons run Aug -> May, so a given calendar year `Y` belongs to
    season `Y` from August onwards, and to season `Y-1` from January to July.
    """
    today = today or date.today()
    return today.year if today.month >= 8 else today.year - 1


def _next_pending_job() -> tuple[int, int] | None:
    """Pop the highest-priority pending job. Returns (league_id, season) or None."""
    sql = """
        SELECT bj.league_id, bj.season
        FROM backfill_jobs bj
        LEFT JOIN tracked_leagues tl ON tl.league_id = bj.league_id
        WHERE bj.status = 'pending'
        ORDER BY
            COALESCE(tl.priority, 9999) ASC,
            bj.league_id ASC,
            bj.season DESC
        LIMIT 1
    """
    with get_connection() as conn:
        row = conn.execute(sql).fetchone()
    return (row["league_id"], row["season"]) if row else None


def _mark_job(
    league_id: int, season: int, status: str,
    fixtures_saved: int | None = None, error: str | None = None,
) -> None:
    now = datetime.now(timezone.utc).isoformat()
    with get_connection() as conn:
        conn.execute(
            """
            UPDATE backfill_jobs
            SET status = ?, fixtures_saved = ?, last_attempt = ?, last_error = ?
            WHERE league_id = ? AND season = ?
            """,
            (status, fixtures_saved, now, error, league_id, season),
        )


def _propagate_no_access(league_id: int, error: str | None) -> int:
    """When one season of a league is plan-blocked, all other pending seasons
    of the same league will be too — Free-plan access is league-level, not
    season-level. Mark them all `no_access` without spending more API calls.
    Returns count of jobs auto-marked.
    """
    now = datetime.now(timezone.utc).isoformat()
    with get_connection() as conn:
        cur = conn.execute(
            """
            UPDATE backfill_jobs
            SET status = 'no_access', last_attempt = ?, last_error = ?
            WHERE league_id = ? AND status = 'pending'
            """,
            (now, error, league_id),
        )
        return cur.rowcount


def _process_job(league_id: int, season: int, save_raw: bool) -> tuple[str, int, str | None]:
    """Fetch all fixtures for one (league, season) and upsert.

    Returns (status, fixtures_saved, error_msg).
    Status will be one of 'completed', 'no_access', 'failed'.
    """
    total_saved = 0
    page = 1
    plan_blocked = False

    while True:
        # status=None pulls every match (including postponed/abandoned), giving
        # a complete picture of the season.
        data, remaining = _fetch_fixtures_page(
            league_id, season, page, status=None,
        )
        if save_raw:
            _save_raw(data, league_id, season, page)

        # api-football returns 200 with errors body on plan restrictions.
        errors = data.get("errors")
        if isinstance(errors, dict) and errors:
            if "plan" in errors:
                plan_blocked = True
                error_msg = errors["plan"]
            else:
                # Some other in-band error; treat as transient failure.
                return "failed", total_saved, str(errors)
            break

        fixtures_raw = data.get("response", [])
        if not fixtures_raw:
            break

        rows = [_parse_fixture(f) for f in fixtures_raw]
        saved = _upsert_fixtures(rows)
        total_saved += saved
        logger.info(
            "  page %d: saved %d fixtures (total %d, daily calls remaining: %s)",
            page, saved, total_saved, remaining,
        )

        paging = data.get("paging", {})
        if page >= paging.get("total", 1):
            break

        page += 1
        time.sleep(_REQUEST_DELAY_S)

    if plan_blocked:
        return "no_access", 0, error_msg

    _record_run(league_id, season, total_saved)
    return "completed", total_saved, None


def run_backfill(min_budget: int = DEFAULT_MIN_BUDGET, save_raw: bool = True) -> dict:
    """Loop pending jobs until queue empty or budget floor hit."""
    init_db()
    jobs_completed = 0
    jobs_no_access = 0
    jobs_failed = 0
    fixtures_total = 0
    stopped_for_budget = False

    while True:
        # Budget check first — bail before pulling another job if we're done for the day.
        if af._last_remaining is not None and af._last_remaining <= min_budget:
            logger.warning(
                "Daily budget at %d (<= floor %d); stopping for today.",
                af._last_remaining, min_budget,
            )
            stopped_for_budget = True
            break

        job = _next_pending_job()
        if job is None:
            logger.info("No pending jobs remaining — backfill caught up.")
            break

        league_id, season = job
        logger.info("Processing job: league=%d season=%d", league_id, season)
        try:
            status, saved, err = _process_job(league_id, season, save_raw=save_raw)
        except Exception as exc:  # network blip, parse failure, etc.
            logger.exception("Job failed: league=%d season=%d", league_id, season)
            _mark_job(league_id, season, "failed", error=str(exc))
            jobs_failed += 1
            time.sleep(_REQUEST_DELAY_S)
            continue

        _mark_job(league_id, season, status, fixtures_saved=saved, error=err)
        if status == "completed":
            jobs_completed += 1
            fixtures_total += saved
        elif status == "no_access":
            jobs_no_access += 1
            # Skip the per-call API cost on the rest of this league's seasons.
            propagated = _propagate_no_access(league_id, err)
            if propagated:
                logger.info(
                    "  propagated no_access to %d other pending seasons of league %d",
                    propagated, league_id,
                )
                jobs_no_access += propagated
        else:
            jobs_failed += 1

        time.sleep(_REQUEST_DELAY_S)

    summary = {
        "jobs_completed": jobs_completed,
        "jobs_no_access": jobs_no_access,
        "jobs_failed": jobs_failed,
        "fixtures_total": fixtures_total,
        "stopped_for_budget": stopped_for_budget,
        "daily_calls_remaining": af._last_remaining,
    }
    logger.info("Backfill run summary: %s", summary)
    return summary


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    parser = argparse.ArgumentParser(description="Process pending backfill_jobs")
    parser.add_argument(
        "--min-budget", type=int, default=DEFAULT_MIN_BUDGET,
        help=f"Stop when daily calls remaining <= this (default: {DEFAULT_MIN_BUDGET})",
    )
    parser.add_argument("--no-raw", action="store_true", help="Skip saving raw JSON responses")
    args = parser.parse_args()

    summary = run_backfill(min_budget=args.min_budget, save_raw=not args.no_raw)
    print(
        f"Backfill done — completed={summary['jobs_completed']}, "
        f"no_access={summary['jobs_no_access']}, failed={summary['jobs_failed']}, "
        f"fixtures saved this run={summary['fixtures_total']}, "
        f"remaining budget={summary['daily_calls_remaining']}."
    )
    if summary.get("stopped_for_budget"):
        print("Stopped for daily budget — re-run tomorrow to continue.")
