"""State for the head-to-head page.

Supports 2 to MAX_TEAMS (4) teams. The user types into per-slot search
boxes; we resolve each independently and, once at least two slots are
filled, filter the fixtures dataframe to matches where *both*
participants are in the resolved set. Aggregates and a comparison
figure are then computed off that filtered slice.

Caps the active team set at 4 to keep the comparison figure readable
and the resolved-name UI tractable. Empty slots are ignored — typing
into slot 3 doesn't require slot 4 to be filled.
"""

from __future__ import annotations

import logging
import re
from pathlib import Path
from typing import Any

import pandas as pd
import reflex as rx

from src.analysis.h2h import plot_h2h_breakdown
from src.analysis.team_breakdown import find_team
from webapp import _cache, _figures

logger = logging.getLogger(__name__)

# Cap N teams. 4 lets you compare e.g. the Manchester clubs + the two
# Merseyside ones without crowding the figure or the meetings table.
MAX_TEAMS = 4
INITIAL_SLOTS = 2

_REPO_ROOT = Path(__file__).parents[2]
_FIGURES_DIR = _REPO_ROOT / "data" / "figures"

# Bump this whenever the plot code changes in a way that should invalidate
# cached PNGs (palette, axis fixes, panel layout, etc.). The version is
# baked into the cache filename — old files with a different version
# simply won't be hit, so the next /h2h render produces a fresh figure
# with the current code. Cheap immortality for the cache.
_FIGURE_VERSION = "v2"


class H2HState(rx.State):
    # ---- Inputs --------------------------------------------------------
    # One query string per visible slot. We track the visible slot count
    # separately because Reflex needs concrete sized state vars — we
    # always keep MAX_TEAMS strings but only render the first
    # `visible_slots` of them.
    queries: list[str] = ["", ""] + [""] * (MAX_TEAMS - INITIAL_SLOTS)
    visible_slots: int = INITIAL_SLOTS

    # Parallel arrays of resolved ids/names/errors — same length as queries.
    team_ids: list[int] = [0] * MAX_TEAMS
    team_names: list[str] = [""] * MAX_TEAMS
    errors: list[str] = [""] * MAX_TEAMS

    # ---- Aggregates ----------------------------------------------------
    match_count: int = 0
    # Per-team record vs the rest of the set — list of dicts so we can
    # render with rx.foreach on the page.
    per_team_records: list[dict[str, Any]] = []
    # Top competitions across all meetings, dict[{league, n}]
    competitions: list[dict[str, Any]] = []
    # Recent meetings (capped at 50) — dict[{Date, Competition,
    # HomeTeam, AwayTeam, Score, Winner}]
    meetings: list[dict[str, Any]] = []

    # ---- Figure URL (generated on demand, cached) ----------------------
    figure_url: str = ""

    # Cache key for the most recently rendered figure — used to skip
    # `_ensure_figure` when the resolved team set hasn't changed since
    # the last call. Avoids re-rendering on every keystroke once the
    # second team has stabilised.
    last_figure_key: str = ""

    # ---- Computed display strings -------------------------------------

    @rx.var
    def active_team_count(self) -> int:
        return sum(1 for tid in self.team_ids[:self.visible_slots] if tid != 0)

    @rx.var
    def has_enough_teams(self) -> bool:
        return self.active_team_count >= 2

    @rx.var
    def can_add_slot(self) -> bool:
        return self.visible_slots < MAX_TEAMS

    @rx.var
    def can_remove_slot(self) -> bool:
        return self.visible_slots > INITIAL_SLOTS

    @rx.var
    def visible_indices(self) -> list[int]:
        """[0, 1, ..., visible_slots - 1] — used to drive rx.foreach
        over the search boxes since iterating the per-slot lists
        directly loses the index we need for set_query_at()."""
        return list(range(self.visible_slots))

    @rx.var
    def match_count_str(self) -> str:
        return f"{self.match_count:,}"

    @rx.var
    def header_str(self) -> str:
        names = [n for n in self.team_names[:self.visible_slots] if n]
        if len(names) >= 2:
            return "  vs  ".join(names)
        return "Head to head"

    # ---- Event handlers -----------------------------------------------

    def set_query_at(self, idx: int, q: str):
        """Update one slot's query, resolve that slot's team, and
        recompute the whole set."""
        if not (0 <= idx < MAX_TEAMS):
            return
        # Reflex passes list elements as copies — re-assign to trigger
        # reactivity on the queries list itself.
        new_queries = list(self.queries)
        new_queries[idx] = q
        self.queries = new_queries
        self._resolve_slot(idx)
        self._recompute()

    def add_slot(self):
        if self.visible_slots < MAX_TEAMS:
            self.visible_slots += 1

    def remove_slot(self):
        if self.visible_slots > INITIAL_SLOTS:
            i = self.visible_slots - 1
            # Clear out the slot we're dropping so it doesn't sneak
            # back in as a resolved team.
            new_queries  = list(self.queries);  new_queries[i] = ""
            new_ids      = list(self.team_ids); new_ids[i] = 0
            new_names    = list(self.team_names); new_names[i] = ""
            new_errors   = list(self.errors); new_errors[i] = ""
            self.queries = new_queries
            self.team_ids = new_ids
            self.team_names = new_names
            self.errors = new_errors
            self.visible_slots -= 1
            self._recompute()

    # ---- Internals ----------------------------------------------------

    def _resolve_slot(self, idx: int):
        q = self.queries[idx].strip()
        new_ids   = list(self.team_ids)
        new_names = list(self.team_names)
        new_err   = list(self.errors)
        if not q:
            new_ids[idx]   = 0
            new_names[idx] = ""
            new_err[idx]   = ""
        else:
            try:
                tid, name = find_team(q, _cache.fixtures())
                new_ids[idx]   = tid
                new_names[idx] = name
                new_err[idx]   = ""
            except ValueError as e:
                new_ids[idx]   = 0
                new_names[idx] = ""
                new_err[idx]   = str(e)
        self.team_ids   = new_ids
        self.team_names = new_names
        self.errors     = new_err

    def _active_set(self) -> tuple[list[int], list[str]]:
        """Return (ids, names) for currently-resolved slots, deduped
        (a user pasting the same name twice shouldn't cause double-counts)."""
        seen: set[int] = set()
        ids: list[int] = []
        names: list[str] = []
        for tid, name in zip(self.team_ids[:self.visible_slots],
                             self.team_names[:self.visible_slots]):
            if tid != 0 and tid not in seen:
                ids.append(tid)
                names.append(name)
                seen.add(tid)
        return ids, names

    def _recompute(self):
        ids, names = self._active_set()
        if len(ids) < 2:
            self._clear_aggregates()
            return

        df = _cache.fixtures()
        tset = set(ids)
        h2h = df[
            df["home_team_id"].isin(tset) & df["away_team_id"].isin(tset)
        ].sort_values("date", ascending=False).copy()

        self.match_count = int(len(h2h))
        if h2h.empty:
            self._clear_aggregates(keep_count=True)
            return

        # ---- Per-team record vs the rest of the set ------------------
        records: list[dict[str, Any]] = []
        for tid, name in zip(ids, names):
            # Restrict to matches involving this team against any other
            # team in the set.
            sub = h2h[(h2h["home_team_id"] == tid) | (h2h["away_team_id"] == tid)]
            home = sub["home_team_id"] == tid
            tg = sub["home_goals"].where(home, sub["away_goals"])
            og = sub["away_goals"].where(home, sub["home_goals"])
            w = int((tg > og).sum())
            d = int((tg == og).sum())
            l = int((tg < og).sum())
            records.append({
                "team":   name,
                "P":      int(len(sub)),
                "W":      w,
                "D":      d,
                "L":      l,
                "GF":     int(tg.sum()),
                "GA":     int(og.sum()),
                "Pts":    3 * w + d,
                "WinPct": f"{(100 * w / len(sub)):.1f}%" if len(sub) else "—",
            })
        # Sort by Pts desc, GD desc, so the "leader" of the head-to-head
        # league rises to the top.
        records.sort(key=lambda r: (r["Pts"], r["GF"] - r["GA"], r["GF"]),
                     reverse=True)
        self.per_team_records = records

        # ---- Competition mix -----------------------------------------
        comp_counts = h2h["league_name"].value_counts().head(6)
        self.competitions = [
            {"league": str(name), "n": int(n)} for name, n in comp_counts.items()
        ]

        # ---- Meetings table (cap 50) ---------------------------------
        rows = h2h.head(50).copy()
        rows["date_str"] = pd.to_datetime(rows["date"]).dt.strftime("%Y-%m-%d")
        rows["score"] = (
            rows["home_goals"].astype(int).astype(str)
            + "–"
            + rows["away_goals"].astype(int).astype(str)
        )
        # Winner name (or "Draw")
        name_by_id = dict(zip(ids, names))
        def _winner(r):
            if r["home_goals"] > r["away_goals"]:
                return name_by_id.get(int(r["home_team_id"]), str(r["home_team_name"]))
            if r["home_goals"] < r["away_goals"]:
                return name_by_id.get(int(r["away_team_id"]), str(r["away_team_name"]))
            return "Draw"
        rows["winner"] = rows.apply(_winner, axis=1)

        self.meetings = rows[[
            "date_str", "league_name", "home_team_name", "away_team_name",
            "score", "winner",
        ]].rename(columns={
            "date_str":        "Date",
            "league_name":     "Competition",
            "home_team_name":  "Home",
            "away_team_name":  "Away",
            "score":           "Score",
            "winner":          "Winner",
        }).to_dict("records")

        # ---- Comparison figure ---------------------------------------
        self.figure_url = self._ensure_figure(ids, names, df)

    def _clear_aggregates(self, keep_count: bool = False):
        if not keep_count:
            self.match_count = 0
        self.per_team_records = []
        self.competitions = []
        self.meetings = []
        self.figure_url = ""
        self.last_figure_key = ""

    def _ensure_figure(
        self, team_ids: list[int], team_names: list[str], df: pd.DataFrame,
    ) -> str:
        """Render (or fetch from cache) the multi-team comparison
        figure. Cache key is the sorted-name tuple so order-of-entry
        doesn't cause redundant renders."""
        key_parts = ["_".join(re.findall(r"\w+", n)) for n, _ in
                     sorted(zip(team_names, team_ids), key=lambda x: x[1])]
        cache_key = "|".join(key_parts)

        # Fast path — same team set as last render, reuse the URL.
        # Catches the common case where the user typed a few stray
        # characters into the search box but the resolved set never
        # actually changed.
        if cache_key == self.last_figure_key and self.figure_url:
            return self.figure_url

        png_name = (
            f"h2h_{_FIGURE_VERSION}_" + "_vs_".join(key_parts) + ".png"
        )
        # Filesystems aren't kind to extremely long names — cap it.
        if len(png_name) > 200:
            png_name = png_name[:196] + ".png"
        out_path = _FIGURES_DIR / png_name

        newly_rendered = False
        if not out_path.exists():
            try:
                plot_h2h_breakdown(team_ids, team_names, df, out_path)
                newly_rendered = True
            except Exception as e:
                logger.warning(
                    "Failed to render H2H figure for %s: %s", team_names, e,
                )
                return ""

        # Only sync assets when we just wrote a new figure — sync_figures
        # walks the data/figures/ tree and that adds up over many
        # keystrokes if we do it unconditionally.
        if newly_rendered:
            _figures.sync_figures()

        url = _figures.find_figure(png_name) or ""
        self.last_figure_key = cache_key
        return url
