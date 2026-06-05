"""State for the team-breakdown page.

When the user types in the search box, we resolve the query to a team_id
via the existing `find_team()` helper, then compute summary stats and a
list of recent matches. All operations are synchronous and run on the
backend; Reflex pushes the updated state to the browser over WebSocket.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
import reflex as rx

from src.analysis.team_breakdown import (
    find_team, load_fixtures, team_perspective,
)


# Cache the fixtures DataFrame at module level — loading 480K rows from
# SQLite takes ~1 second, and the data only changes once a day. Reflex's
# State is per-session, so a module-level cache survives across sessions.
_FIXTURES: pd.DataFrame | None = None


def _get_fixtures() -> pd.DataFrame:
    global _FIXTURES
    if _FIXTURES is None:
        _FIXTURES = load_fixtures()
    return _FIXTURES


class TeamState(rx.State):
    # ---- Inputs --------------------------------------------------------
    query: str = ""

    # ---- Resolved team -------------------------------------------------
    team_id: int = 0
    team_name: str = ""
    primary_league: str = ""

    # ---- Aggregates ----------------------------------------------------
    match_count: int = 0
    win_count: int = 0
    draw_count: int = 0
    loss_count: int = 0
    goals_for: int = 0
    goals_against: int = 0
    competition_count: int = 0

    # ---- Recent matches table -----------------------------------------
    recent_matches: list[dict[str, Any]] = []

    # ---- Error message -------------------------------------------------
    error: str = ""

    # ---- Computed display strings -------------------------------------

    @rx.var
    def has_team(self) -> bool:
        return self.team_id != 0

    @rx.var
    def has_query(self) -> bool:
        return bool(self.query.strip())

    @rx.var
    def match_count_str(self) -> str:
        return f"{self.match_count:,}"

    @rx.var
    def win_rate_str(self) -> str:
        if self.match_count == 0:
            return "—"
        return f"{100 * self.win_count / self.match_count:.1f}%"

    @rx.var
    def record_str(self) -> str:
        if self.match_count == 0:
            return "—"
        return f"{self.win_count}W · {self.draw_count}D · {self.loss_count}L"

    @rx.var
    def goal_diff_str(self) -> str:
        gd = self.goals_for - self.goals_against
        return f"{gd:+d}"

    @rx.var
    def goals_str(self) -> str:
        if self.match_count == 0:
            return "—"
        avg_for = self.goals_for / self.match_count
        avg_ag = self.goals_against / self.match_count
        return f"{avg_for:.2f} / {avg_ag:.2f}"

    @rx.var
    def competition_count_str(self) -> str:
        return f"{self.competition_count}"

    # ---- Event handlers -----------------------------------------------

    def set_query(self, q: str):
        """Triggered on every keystroke in the search input."""
        self.query = q
        self._resolve()

    def _resolve(self):
        """Look up the team and recompute aggregates."""
        if not self.query.strip():
            self._clear()
            return
        try:
            df = _get_fixtures()
            tid, name = find_team(self.query.strip(), df)
        except ValueError as e:
            self._clear()
            self.error = str(e)
            return

        dft = team_perspective(df, tid)
        if dft.empty:
            self._clear()
            self.error = f"No fixtures found for {name}"
            return

        self.error = ""
        self.team_id = tid
        self.team_name = name

        # Primary league = most-played league name across all matches
        league_counts = dft["league_name"].value_counts()
        self.primary_league = (
            str(league_counts.index[0]) if not league_counts.empty else "—"
        )

        self.match_count = int(len(dft))
        self.win_count   = int((dft["result"] == "W").sum())
        self.draw_count  = int((dft["result"] == "D").sum())
        self.loss_count  = int((dft["result"] == "L").sum())
        self.goals_for     = int(dft["team_goals"].sum())
        self.goals_against = int(dft["opp_goals"].sum())
        self.competition_count = int(dft["league_id"].nunique())

        # Recent 15 matches, oldest-newest reversed for display
        recent = (
            dft.sort_values("date", ascending=False).head(15)
            .assign(
                date_str=lambda d: pd.to_datetime(d["date"]).dt.strftime("%Y-%m-%d"),
                score=lambda d: d["team_goals"].astype(str) + "–" + d["opp_goals"].astype(str),
            )
            [["date_str", "venue", "opp_name", "score", "result", "league_name"]]
            .rename(columns={
                "date_str": "Date", "venue": "Venue", "opp_name": "Opponent",
                "score": "Score", "result": "R", "league_name": "Competition",
            })
        )
        self.recent_matches = recent.to_dict("records")

    def _clear(self):
        self.team_id = 0
        self.team_name = ""
        self.primary_league = ""
        self.match_count = 0
        self.win_count = self.draw_count = self.loss_count = 0
        self.goals_for = self.goals_against = 0
        self.competition_count = 0
        self.recent_matches = []
