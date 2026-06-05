"""State for the league-ladders page.

User picks a league + season → we compute the final-position points table
from `fixtures` via SQL (fast — has indexes) and join in each team's
final Elo from the module-level cache.

Sorting: column headers are clickable. Clicking a column once sorts
descending (high-to-low) — that's what makes sense for stats people
care about (Pts, W, Elo). Clicking again flips to ascending. Clicking
a different column resets direction to the default for that column.
"""

from __future__ import annotations

from typing import Any

import reflex as rx

from webapp import _cache


# Default sort direction per column. Pts/wins/etc. default to descending
# (you want the best at top); position and team name default to ascending.
_DEFAULT_DIR: dict[str, str] = {
    "pos":  "asc",
    "team": "asc",
    "P":    "desc",
    "W":    "desc",
    "D":    "desc",
    "L":    "desc",
    "GF":   "desc",
    "GA":   "asc",   # fewer goals against is better — ascending shows best at top
    "GD":   "desc",
    "Pts":  "desc",
    "Elo":  "desc",
}


class LeagueState(rx.State):
    # Sentinel value for "nothing selected yet" — Reflex requires concrete
    # default types in State declarations.
    league_id: int = 39           # default to Premier League
    season: int = 2024            # default to most recent full season

    league_name: str = "Premier League"

    league_options: list[dict[str, str]] = []
    season_options: list[str] = []

    # Source of truth — always stored in natural ladder order (pos asc).
    # `table_rows` is the rendered view, sorted by the user's chosen column.
    rows_data: list[dict[str, Any]] = []

    sort_key: str = "pos"
    sort_dir: str = "asc"   # "asc" or "desc"

    # ---- Computed -----------------------------------------------------

    @rx.var
    def has_rows(self) -> bool:
        return len(self.rows_data) > 0

    @rx.var
    def header_str(self) -> str:
        if not self.rows_data:
            return f"{self.league_name} · {self.season}-{(self.season + 1) % 100:02d}"
        return (
            f"{self.league_name} · {self.season}-{(self.season + 1) % 100:02d}  "
            f"({len(self.rows_data)} teams)"
        )

    @rx.var
    def table_rows(self) -> list[dict[str, Any]]:
        """Rows sorted by the active column. For Elo we sort on the
        numeric `elo_value` field even though the displayed column is
        the formatted string."""
        if not self.rows_data:
            return []
        key = self.sort_key
        reverse = self.sort_dir == "desc"

        if key == "Elo":
            # Missing Elo (None) goes to the bottom regardless of direction.
            def k(r):
                v = r.get("elo_value")
                return (v is None, v if v is not None else 0)
            return sorted(self.rows_data, key=k, reverse=reverse)

        return sorted(
            self.rows_data,
            key=lambda r: r.get(key, 0),
            reverse=reverse,
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

    def sort_by(self, key: str):
        """Toggle sort direction if clicking the active column; otherwise
        switch to the new column at its natural default direction."""
        if key == self.sort_key:
            self.sort_dir = "asc" if self.sort_dir == "desc" else "desc"
        else:
            self.sort_key = key
            self.sort_dir = _DEFAULT_DIR.get(key, "desc")

    def _recompute(self):
        df = _cache.league_table(self.league_id, self.season)
        if df.empty:
            self.rows_data = []
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
                # Display value (formatted) + numeric value for sorting.
                "Elo": f"{elo:.0f}" if elo is not None else "—",
                "elo_value": float(elo) if elo is not None else None,
            })
        # Reset to natural order whenever data is reloaded — but keep
        # whatever sort the user had chosen.
        self.rows_data = rows
