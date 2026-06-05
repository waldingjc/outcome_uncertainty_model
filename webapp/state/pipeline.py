"""State for the pipeline status page.

Pulls:
  - queue status from `backfill_jobs`
  - last-run summary from `data/logs/last-run.json`
  - daily log presence from filenames in `data/logs/backfill-*.log`
"""

from __future__ import annotations

import json
import re
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import reflex as rx

from src.db.schema import get_connection

_REPO_ROOT = Path(__file__).parents[2]
_LOGS_DIR = _REPO_ROOT / "data" / "logs"
_LAST_RUN_JSON = _LOGS_DIR / "last-run.json"

# Filename pattern, e.g. backfill-2026-05-21.log
_LOG_PATTERN = re.compile(r"backfill-(\d{4}-\d{2}-\d{2})\.log$")


class PipelineState(rx.State):
    # Queue
    queue_completed: int = 0
    queue_no_access: int = 0
    queue_pending: int = 0
    queue_failed: int = 0

    # Last run
    last_run_at: str = ""
    last_jobs: int = 0
    last_fixtures: int = 0
    last_remaining: int = 0
    last_skipped: bool = False

    # Recent daily-log presence — list of {"date", "size_kb", "exists"}
    recent_days: list[dict[str, Any]] = []

    # ---- Computed -----------------------------------------------------

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
    def queue_pct_value(self) -> int:
        if self.queue_total == 0:
            return 0
        return int(100 * self.queue_done / self.queue_total)

    @rx.var
    def queue_done_str(self) -> str:
        return f"{self.queue_done:,}"

    @rx.var
    def queue_pending_str(self) -> str:
        return f"{self.queue_pending:,}"

    @rx.var
    def queue_total_str(self) -> str:
        return f"{self.queue_total:,}"

    @rx.var
    def last_summary_str(self) -> str:
        if not self.last_run_at:
            return "No last-run data."
        if self.last_skipped:
            return (
                f"{self.last_run_at} — skipped pre-flight "
                f"(quota at {self.last_remaining})"
            )
        return (
            f"{self.last_run_at} — {self.last_jobs} jobs, "
            f"{self.last_fixtures:,} fixtures saved "
            f"({self.last_remaining} calls remaining)"
        )

    # ---- Loaders -----------------------------------------------------

    def on_load(self):
        self._load_queue()
        self._load_last_run()
        self._scan_logs()

    def _load_queue(self):
        with get_connection() as conn:
            rows = conn.execute(
                "SELECT status, COUNT(*) AS n FROM backfill_jobs GROUP BY status"
            ).fetchall()
        c = {r["status"]: r["n"] for r in rows}
        self.queue_completed = int(c.get("completed", 0))
        self.queue_no_access = int(c.get("no_access", 0))
        self.queue_pending = int(c.get("pending", 0))
        self.queue_failed = int(c.get("failed", 0))

    def _load_last_run(self):
        if not _LAST_RUN_JSON.exists():
            return
        try:
            d = json.loads(_LAST_RUN_JSON.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return
        self.last_run_at = str(d.get("ran_at", ""))[:19].replace("T", " ")
        self.last_jobs = int(d.get("jobs_completed", 0))
        self.last_fixtures = int(d.get("fixtures_total", 0))
        self.last_remaining = int(d.get("daily_calls_remaining", 0) or 0)
        self.last_skipped = bool(d.get("skipped_pre_flight", False))

    def _scan_logs(self):
        """Build a 14-day rolling status — for each of the last 14 days,
        either an existing log file (= ran) or a gap (= didn't run)."""
        existing: dict[str, int] = {}
        if _LOGS_DIR.exists():
            for log in _LOGS_DIR.glob("backfill-*.log"):
                m = _LOG_PATTERN.search(log.name)
                if not m:
                    continue
                existing[m.group(1)] = log.stat().st_size

        # Anchor "today" to whatever the latest log we saw is — that's the
        # most recent activity, and using system clock could disagree with
        # the data's view of "today".
        if existing:
            anchor_date_str = max(existing.keys())
            anchor_date = datetime.strptime(anchor_date_str, "%Y-%m-%d").date()
        else:
            anchor_date = datetime.now(timezone.utc).date()

        rows: list[dict[str, Any]] = []
        for delta in range(13, -1, -1):
            d = anchor_date - timedelta(days=delta)
            key = d.isoformat()
            size = existing.get(key, 0)
            ran = size > 0
            size_kb = f"{size / 1024:.1f}" if size else "—"
            # Pre-render the tooltip here — Reflex can't concat Vars + str
            # inside the page render function, so we build the final text
            # eagerly while we still have native Python strings.
            tooltip = f"{key} — {size_kb} KB log" if ran else f"{key} — no log"
            rows.append({
                "date": key,
                "size_kb": size_kb,
                "ran": ran,
                "tooltip": tooltip,
            })
        self.recent_days = rows
