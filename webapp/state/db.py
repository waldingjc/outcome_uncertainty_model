"""Shared state for high-level DB stats (used on the home page and the
sidebar status indicator).

`DBState.load()` is wired up as an on-load handler in `webapp.py` — when
any page mounts, it pulls fresh stats from SQLite. Since the underlying
DB only grows once per day via the scheduled backfill, this is plenty
fresh for our purposes.
"""

from __future__ import annotations

import json
from pathlib import Path

import reflex as rx

from src.db.schema import get_connection

_LAST_RUN_JSON = Path(__file__).parents[2] / "data" / "logs" / "last-run.json"


class DBState(rx.State):
    fixture_count: int = 0
    league_count: int = 0
    team_count: int = 0
    date_min: str = ""
    date_max: str = ""

    queue_completed: int = 0
    queue_no_access: int = 0
    queue_pending: int = 0
    queue_failed: int = 0

    last_run_at: str = ""
    last_jobs_completed: int = 0
    last_fixtures_total: int = 0
    last_calls_remaining: int = 0
    last_was_skipped: bool = False

    # ---- Computed views ------------------------------------------------

    @rx.var
    def fixture_count_str(self) -> str:
        return f"{self.fixture_count:,}"

    @rx.var
    def league_count_str(self) -> str:
        return f"{self.league_count:,}"

    @rx.var
    def team_count_str(self) -> str:
        return f"{self.team_count:,}"

    @rx.var
    def date_range_str(self) -> str:
        if not self.date_min:
            return "—"
        return f"{self.date_min} → {self.date_max}"

    @rx.var
    def queue_total(self) -> int:
        return (
            self.queue_completed + self.queue_no_access
            + self.queue_pending + self.queue_failed
        )

    @rx.var
    def queue_done(self) -> int:
        return self.queue_completed + self.queue_no_access

    @rx.var
    def queue_pct_str(self) -> str:
        if self.queue_total == 0:
            return "—"
        return f"{100 * self.queue_done / self.queue_total:.1f}%"

    @rx.var
    def queue_progress_value(self) -> int:
        """0-100 integer for use as a progress-bar value."""
        if self.queue_total == 0:
            return 0
        return int(100 * self.queue_done / self.queue_total)

    @rx.var
    def last_run_summary(self) -> str:
        if not self.last_run_at:
            return "No runs recorded yet."
        if self.last_was_skipped:
            return (
                f"Last run ({self.last_run_at}): skipped (quota at "
                f"{self.last_calls_remaining})."
            )
        return (
            f"Last run ({self.last_run_at}): {self.last_jobs_completed} jobs, "
            f"{self.last_fixtures_total:,} fixtures saved."
        )

    # ---- Loaders -------------------------------------------------------

    def load(self):
        """Called on page mount via the App's on_load hook."""
        self._load_db_stats()
        self._load_queue()
        self._load_last_run()

    def _load_db_stats(self):
        with get_connection() as conn:
            self.fixture_count = conn.execute(
                "SELECT COUNT(*) FROM fixtures"
            ).fetchone()[0]
            self.league_count = conn.execute(
                "SELECT COUNT(DISTINCT league_id) FROM fixtures"
            ).fetchone()[0]
            self.team_count = conn.execute(
                "SELECT COUNT(*) FROM ("
                "SELECT home_team_id AS id FROM fixtures "
                "UNION SELECT away_team_id FROM fixtures)"
            ).fetchone()[0]
            rng = conn.execute(
                "SELECT MIN(DATE(date)), MAX(DATE(date)) FROM fixtures"
            ).fetchone()
            if rng and rng[0]:
                self.date_min, self.date_max = rng[0], rng[1]

    def _load_queue(self):
        with get_connection() as conn:
            rows = conn.execute(
                "SELECT status, COUNT(*) AS n FROM backfill_jobs GROUP BY status"
            ).fetchall()
        counts = {r["status"]: r["n"] for r in rows}
        self.queue_completed = int(counts.get("completed", 0))
        self.queue_no_access = int(counts.get("no_access", 0))
        self.queue_pending = int(counts.get("pending", 0))
        self.queue_failed = int(counts.get("failed", 0))

    def _load_last_run(self):
        if not _LAST_RUN_JSON.exists():
            return
        try:
            data = json.loads(_LAST_RUN_JSON.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return
        self.last_run_at = str(data.get("ran_at", ""))[:19].replace("T", " ")
        self.last_jobs_completed = int(data.get("jobs_completed", 0))
        self.last_fixtures_total = int(data.get("fixtures_total", 0))
        self.last_calls_remaining = int(data.get("daily_calls_remaining", 0) or 0)
        self.last_was_skipped = bool(data.get("skipped_pre_flight", False))
