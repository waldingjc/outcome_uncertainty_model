"""State for the league-ladders page.

User picks a league + season → we compute the final-position points table
from `fixtures` via SQL (fast — has indexes) and join in each team's
final Elo from the module-level cache.
"""

from __future__ import annotations

from typing import Any

import reflex as rx

from webapp import _cache


class LeagueState(rx.State):
    # Sentinel value for "nothing selected yet" — Reflex requires concrete
    # default types in State declarations.
    league_id: int = 39           # default to Premier League
    season: int = 2024            # default to most recent full season

    league_name: str = "Premier League"

    league_options: list[dict[str, str]] = []
    season_options: list[str] = []

    table_rows: list[dict[str, Any]] = []

    # ---- Computed -----------------------------------------------------

    @rx.var
    def has_rows(self) -> bool:
        return len(self.table_rows) > 0

    @rx.var
    def header_str(self) -> str:
        if not self.table_rows:
            return f"{self.league_name} · {self.season}-{(self.season + 1) % 100:02d}"
        return (
            f"{self.league_name} · {self.season}-{(self.season + 1) % 100:02d}  "
            f"({len(self.table_rows)} teams)"
        )

    # ---- Event handlers ----------------------------------------------

    def on_load(self):
        # Build dropdown options on first visit (cheap; cached behind LRU)
        self.league_options = [
            {"label": f"{name}  (id {lid})", "value": str(lid)}
            for lid, name in _cache.league_choices()[:80]  # top 80 by size
        ]
        self.season_options = [str(s) for s in _cache.season_choices()]
        self._recompute()

    def set_league(self, league_id_str: str):
        if not league_id_str:
            return
        try:
            self.league_id = int(league_id_str)
        except ValueError:
            return
        # Refresh league display name
        for lid, name in _cache.league_choices():
            if lid == self.league_id:
                self.league_name = name
                break
        self._recompute()

    def set_season(self, season_str: str):
        if not season_str:
            return
        try:
            self.season = int(season_str)
        except ValueError:
            return
        self._recompute()

    def _recompute(self):
        df = _cache.league_table(self.league_id, self.season)
        if df.empty:
            self.table_rows = []
            return

        ratings = _cache.elo_cache()["ratings"]
        rows: list[dict[str, Any]] = []
        for pos, r in enumerate(df.itertuples(index=False), start=1):
            elo = ratings.get(int(r.team_id))
            rows.append({
                "pos": pos,
                "team": r.team,
                "P": int(r.P),
                "W": int(r.W),
                "D": int(r.D),
                "L": int(r.L),
                "GF": int(r.GF),
                "GA": int(r.GA),
                "GD": int(r.GD),
                "Pts": int(r.Pts),
                "Elo": f"{elo:.0f}" if elo is not None else "—",
            })
        self.table_rows = rows
