"""State for the head-to-head page.

User types two team names; we resolve each via `find_team()`, filter
the fixtures dataframe down to matches where both teams played, and
expose:
  - aggregate record from team A's perspective (W/D/L, goals)
  - meeting frequency by competition
  - all historical meetings as a table (most recent first)

Both teams are resolved independently — typing into the box for team
B doesn't disturb team A. Errors per team are surfaced separately.
"""

from __future__ import annotations

import logging
from typing import Any

import pandas as pd
import reflex as rx

from src.analysis.team_breakdown import find_team, load_fixtures

logger = logging.getLogger(__name__)


# Reuse the same module-level fixtures cache as the team page — saves
# ~1s of SQLite read per session. We import it lazily so the h2h page
# doesn't drag a hard dependency on teams.py.
_FIXTURES: pd.DataFrame | None = None


def _get_fixtures() -> pd.DataFrame:
    global _FIXTURES
    if _FIXTURES is None:
        _FIXTURES = load_fixtures()
    return _FIXTURES


class H2HState(rx.State):
    # ---- Inputs --------------------------------------------------------
    query_a: str = ""
    query_b: str = ""

    # ---- Resolved teams ------------------------------------------------
    team_a_id: int = 0
    team_b_id: int = 0
    team_a_name: str = ""
    team_b_name: str = ""

    # ---- Aggregates (A perspective) -----------------------------------
    match_count: int = 0
    a_wins: int = 0
    draws: int = 0
    b_wins: int = 0
    a_goals: int = 0
    b_goals: int = 0

    # ---- Meeting history -----------------------------------------------
    meetings: list[dict[str, Any]] = []
    competitions: list[dict[str, Any]] = []   # [{league, n}], top 6

    # ---- Errors per slot (so each search bar can show its own) --------
    error_a: str = ""
    error_b: str = ""

    # ---- Computed display strings -------------------------------------

    @rx.var
    def has_both_teams(self) -> bool:
        return self.team_a_id != 0 and self.team_b_id != 0

    @rx.var
    def match_count_str(self) -> str:
        return f"{self.match_count:,}"

    @rx.var
    def record_str(self) -> str:
        if self.match_count == 0:
            return "—"
        return f"{self.a_wins}W · {self.draws}D · {self.b_wins}L"

    @rx.var
    def a_win_pct_str(self) -> str:
        if self.match_count == 0:
            return "—"
        return f"{100 * self.a_wins / self.match_count:.1f}%"

    @rx.var
    def b_win_pct_str(self) -> str:
        if self.match_count == 0:
            return "—"
        return f"{100 * self.b_wins / self.match_count:.1f}%"

    @rx.var
    def goals_str(self) -> str:
        """Goals tally — "A : B" totals across all meetings."""
        if self.match_count == 0:
            return "—"
        return f"{self.a_goals} : {self.b_goals}"

    @rx.var
    def avg_goals_str(self) -> str:
        if self.match_count == 0:
            return "—"
        return f"{(self.a_goals + self.b_goals) / self.match_count:.2f}"

    @rx.var
    def header_str(self) -> str:
        if self.has_both_teams:
            return f"{self.team_a_name}  vs  {self.team_b_name}"
        return "Head to head"

    # ---- Event handlers -----------------------------------------------

    def set_query_a(self, q: str):
        self.query_a = q
        self._resolve_a()
        self._recompute()

    def set_query_b(self, q: str):
        self.query_b = q
        self._resolve_b()
        self._recompute()

    def _resolve_a(self):
        if not self.query_a.strip():
            self.team_a_id = 0
            self.team_a_name = ""
            self.error_a = ""
            return
        try:
            tid, name = find_team(self.query_a.strip(), _get_fixtures())
            self.team_a_id = tid
            self.team_a_name = name
            self.error_a = ""
        except ValueError as e:
            self.team_a_id = 0
            self.team_a_name = ""
            self.error_a = str(e)

    def _resolve_b(self):
        if not self.query_b.strip():
            self.team_b_id = 0
            self.team_b_name = ""
            self.error_b = ""
            return
        try:
            tid, name = find_team(self.query_b.strip(), _get_fixtures())
            self.team_b_id = tid
            self.team_b_name = name
            self.error_b = ""
        except ValueError as e:
            self.team_b_id = 0
            self.team_b_name = ""
            self.error_b = str(e)

    def _recompute(self):
        if not (self.team_a_id and self.team_b_id):
            self.match_count = 0
            self.a_wins = self.draws = self.b_wins = 0
            self.a_goals = self.b_goals = 0
            self.meetings = []
            self.competitions = []
            return
        if self.team_a_id == self.team_b_id:
            # Both inputs resolved to the same team — likely a typo,
            # show 0 meetings rather than every match the team played.
            self.match_count = 0
            self.a_wins = self.draws = self.b_wins = 0
            self.a_goals = self.b_goals = 0
            self.meetings = []
            self.competitions = []
            return

        df = _get_fixtures()
        a, b = self.team_a_id, self.team_b_id
        mask = (
            ((df["home_team_id"] == a) & (df["away_team_id"] == b))
            | ((df["home_team_id"] == b) & (df["away_team_id"] == a))
        )
        h2h = df[mask].sort_values("date", ascending=False).copy()
        if h2h.empty:
            self.match_count = 0
            self.a_wins = self.draws = self.b_wins = 0
            self.a_goals = self.b_goals = 0
            self.meetings = []
            self.competitions = []
            return

        # Restate each row from team-A's perspective
        a_is_home = h2h["home_team_id"] == a
        h2h["a_goals"]    = h2h["home_goals"].where(a_is_home,  h2h["away_goals"])
        h2h["b_goals"]    = h2h["away_goals"].where(a_is_home,  h2h["home_goals"])
        h2h["venue_a"]    = a_is_home.map({True: "home", False: "away"})
        h2h["result_a"]   = "D"
        h2h.loc[h2h["a_goals"] > h2h["b_goals"], "result_a"] = "W"
        h2h.loc[h2h["a_goals"] < h2h["b_goals"], "result_a"] = "L"

        self.match_count = int(len(h2h))
        self.a_wins = int((h2h["result_a"] == "W").sum())
        self.draws  = int((h2h["result_a"] == "D").sum())
        self.b_wins = int((h2h["result_a"] == "L").sum())
        self.a_goals = int(h2h["a_goals"].sum())
        self.b_goals = int(h2h["b_goals"].sum())

        # Per-competition breakdown — top 6 by meetings, used for the
        # "where they've met" chip strip.
        comp_counts = h2h["league_name"].value_counts().head(6)
        self.competitions = [
            {"league": name, "n": int(n)} for name, n in comp_counts.items()
        ]

        # Recent meetings table (full history, capped at 50 for sanity).
        rows = h2h.head(50).assign(
            date_str=lambda d: pd.to_datetime(d["date"]).dt.strftime("%Y-%m-%d"),
            score=lambda d: d["a_goals"].astype(int).astype(str)
                + "–"
                + d["b_goals"].astype(int).astype(str),
        )[["date_str", "league_name", "venue_a", "score", "result_a"]].rename(
            columns={
                "date_str":    "Date",
                "league_name": "Competition",
                "venue_a":     "Venue",
                "score":       "Score",
                "result_a":    "R",
            }
        )
        self.meetings = rows.to_dict("records")
