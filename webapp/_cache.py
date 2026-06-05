"""Module-level caches for expensive computations shared across pages.

Reflex's `State` is per-session, so heavy global data like the full Elo
walk lives here instead. Each helper is lazy — computed on first call,
cached forever (in practice, until the dev server restarts, which is
also when the underlying DB might have changed).
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

import pandas as pd

from src.analysis.strength import (
    DEFAULT_BASE_RATING,
    compute_elo_ratings,
    compute_league_elo,
    primary_league_map,
    team_name_map,
)
from src.analysis.team_breakdown import (
    load_fixtures as _load_fixtures_team_view,
    team_perspective as _team_perspective,
)
from src.db.schema import get_connection
from src.model.data import load_fixtures_for_elo


@lru_cache(maxsize=1)
def fixtures() -> pd.DataFrame:
    """Full fixtures dataframe (~480K rows, ~2.7s to load).

    Shared between /team, /h2h, and any future page that needs match-
    level data. Lives behind an lru_cache so we read it from SQLite
    once per process lifetime instead of once per page.
    """
    return _load_fixtures_team_view()


@lru_cache(maxsize=2048)
def team_perspective_for(team_id: int) -> pd.DataFrame:
    """Cached per-team perspective dataframe. Computing it is fast
    (~5ms), but `plot_h2h_breakdown` used to call it 10-20 times per
    figure render (once per panel per team) which added up. The cache
    is keyed on team_id only since the underlying fixtures cache also
    lives for the process lifetime."""
    return _team_perspective(fixtures(), team_id)


def prewarm():
    """Touch the heavy caches so the first page render doesn't pay
    cold-start. Called from `webapp.webapp` at module-import time so
    the user's first navigation is already warm."""
    fixtures()
    league_choices()
    season_choices()
    elo_cache()


@lru_cache(maxsize=1)
def league_choices() -> list[tuple[int, str]]:
    """All (league_id, league_name) pairs that have fixtures, sorted by
    number of fixtures descending. Used to populate the league dropdown
    on the ladders page."""
    with get_connection() as conn:
        rows = conn.execute(
            """
            SELECT league_id, league_name, COUNT(*) AS n
            FROM fixtures
            WHERE status = 'FT'
            GROUP BY league_id, league_name
            ORDER BY n DESC
            """
        ).fetchall()
    return [(int(r["league_id"]), str(r["league_name"])) for r in rows]


@lru_cache(maxsize=1)
def season_choices() -> list[int]:
    """All seasons present in the fixtures table."""
    with get_connection() as conn:
        rows = conn.execute(
            "SELECT DISTINCT season FROM fixtures ORDER BY season DESC"
        ).fetchall()
    return [int(r["season"]) for r in rows]


@lru_cache(maxsize=1)
def elo_cache() -> dict:
    """Compute the team-Elo walk once and reuse across pages.

    Returns dict with keys:
      - ratings:    {team_id: final_elo}
      - team_names: {team_id: name}
      - primary:    {team_id: (league_id, league_name)}
    """
    df = load_fixtures_for_elo()
    league_seeds, _ = compute_league_elo(df)
    pmap = primary_league_map(df)
    team_seeds = {
        tid: league_seeds.get(lid, DEFAULT_BASE_RATING)
        for tid, (lid, _) in pmap.items()
    }
    _, ratings = compute_elo_ratings(df, team_seeds=team_seeds)
    return {
        "ratings": ratings,
        "team_names": team_name_map(df),
        "primary": pmap,
    }


def league_table(league_id: int, season: int) -> pd.DataFrame:
    """Final-position points table for one (league, season). Computed
    directly from `fixtures` in SQL — fast even on 500K-row tables
    thanks to the indexes on date/league_id."""
    with get_connection() as conn:
        df = pd.read_sql(
            """
            SELECT home_team_id, home_team_name, away_team_id, away_team_name,
                   home_goals, away_goals
            FROM fixtures
            WHERE league_id = ? AND season = ? AND status = 'FT'
              AND home_goals IS NOT NULL AND away_goals IS NOT NULL
            """,
            conn, params=(league_id, season),
        )
    if df.empty:
        return pd.DataFrame(columns=[
            "team_id", "team", "P", "W", "D", "L", "GF", "GA", "GD", "Pts",
        ])

    # Long-form: one row per team per match
    h = df.rename(columns={
        "home_team_id": "team_id", "home_team_name": "team",
        "home_goals": "gf", "away_goals": "ga",
    })[["team_id", "team", "gf", "ga"]]
    h["pts"] = (h["gf"] > h["ga"]) * 3 + (h["gf"] == h["ga"]) * 1
    a = df.rename(columns={
        "away_team_id": "team_id", "away_team_name": "team",
        "away_goals": "gf", "home_goals": "ga",
    })[["team_id", "team", "gf", "ga"]]
    a["pts"] = (a["gf"] > a["ga"]) * 3 + (a["gf"] == a["ga"]) * 1
    stacked = pd.concat([h, a], ignore_index=True)
    stacked["w"] = (stacked["gf"] > stacked["ga"]).astype(int)
    stacked["d"] = (stacked["gf"] == stacked["ga"]).astype(int)
    stacked["l"] = (stacked["gf"] < stacked["ga"]).astype(int)

    table = stacked.groupby(["team_id", "team"]).agg(
        P=("pts", "size"),
        W=("w", "sum"),
        D=("d", "sum"),
        L=("l", "sum"),
        GF=("gf", "sum"),
        GA=("ga", "sum"),
        Pts=("pts", "sum"),
    ).reset_index()
    table["GD"] = table["GF"] - table["GA"]
    table = table[["team_id", "team", "P", "W", "D", "L", "GF", "GA", "GD", "Pts"]]
    return table.sort_values(
        ["Pts", "GD", "GF"], ascending=[False, False, False]
    ).reset_index(drop=True)
