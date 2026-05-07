"""Fixture loading, filtering, and train/test split for the modelling pipeline.

Two distinct loaders, because they answer different questions:

  * `load_fixtures_for_elo()` returns every fixture that should contribute to
    Elo ratings — that's basically everything except friendlies, youth
    competitions, and women's competitions, which would add noise to ratings
    without adding signal.

  * `load_training_fixtures()` returns the fixtures we actually train and
    test on. By default that's the Premier League only, with cups excluded
    (toggleable via `include_cups`).

The train/test split is time-based at `TRAIN_CUTOFF` (default
2024-07-01 = the start of the 2024-25 season). Anything before that date
is training; anything from that date forwards is test.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime
from typing import Iterable

import pandas as pd

from src.db.schema import get_connection

# Default test cutoff — start of the 2024-25 European football season. The
# 2024-25 season is held out as the test set. We use tz-naive UTC throughout
# the modelling layer for simplicity (no timezone arithmetic needed; matches
# happen at recorded UTC and the cutoff is also UTC).
TRAIN_CUTOFF: datetime = datetime(2024, 7, 1)

# Default target competition: English Premier League.
DEFAULT_TARGET_LEAGUE: int = 39

# Leagues entirely excluded from BOTH Elo computation and training.
# These don't represent normal competitive senior club football and would
# only add noise to ratings.
EXCLUDED_FROM_ELO: frozenset[int] = frozenset({
    10,   # Friendlies
    14,   # UEFA Youth League
    38,   # UEFA U21 Championship
    8,    # World Cup - Women
    44,   # FA WSL
    64,   # Feminine Division 1 (women's)
    74,   # Brasileiro Women
    82,   # Frauen Bundesliga
    91,   # Eredivisie Women
})


def is_cup_competition(league_name: str) -> bool:
    """Heuristic identification of knockout-cup competitions by name.

    Catches FA Cup, EFL Cup, Pokal (DE), Coupe (FR), Coppa (IT), Beker (NL),
    Taça (PT), and several Scandinavian / Welsh variants. Better than a
    hardcoded id list because it generalises automatically as new countries
    are added to the dataset.
    """
    name = league_name.lower()
    keywords = (
        "cup", "pokal", "pokalen", "coupe", "coppa", "beker",
        "trophy", "taça", "taca", "taça", "cupen", "kupa",
    )
    return any(kw in name for kw in keywords)


# ---------------- Public dataclasses ----------------

@dataclass(frozen=True)
class FilterConfig:
    """Filtering options for the training/test fixture loader."""
    target_league_id: int = DEFAULT_TARGET_LEAGUE
    include_cups: bool = False
    additional_excluded_leagues: frozenset[int] = frozenset()


@dataclass(frozen=True)
class SplitConfig:
    """Options for the time-based split."""
    cutoff: datetime = TRAIN_CUTOFF


# ---------------- Loaders ----------------

def _load_all(min_date: datetime | None = None) -> pd.DataFrame:
    """Pull every FT fixture from the DB. Internal helper."""
    sql = """
        SELECT fixture_id, date, league_id, league_name, season, round,
               home_team_id, home_team_name, away_team_id, away_team_name,
               home_goals, away_goals, home_goals_ht, away_goals_ht,
               status, venue_name, venue_city, referee
        FROM fixtures
        WHERE status = 'FT'
          AND home_goals IS NOT NULL
          AND away_goals IS NOT NULL
    """
    params: tuple = ()
    if min_date is not None:
        sql += " AND date >= ?"
        params = (min_date.isoformat(),)
    with get_connection() as conn:
        df = pd.read_sql(sql, conn, params=params, parse_dates=["date"])
    # Normalise dates to tz-naive UTC. pd.read_sql parses ISO strings like
    # "2024-08-16T15:00:00+00:00" as tz-aware UTC; we strip the tz info so
    # downstream code can compare against tz-naive cutoffs without juggling
    # timezones. All values are still wall-clock UTC after this conversion.
    df["date"] = pd.to_datetime(df["date"], utc=True).dt.tz_localize(None)
    df["is_cup"] = df["league_name"].apply(is_cup_competition)
    return df


def load_fixtures_for_elo(
    excluded: Iterable[int] = EXCLUDED_FROM_ELO,
) -> pd.DataFrame:
    """Fixtures used as the corpus for the Elo computation.

    Drops friendlies, youth competitions, women's competitions — anything
    that would add noise without signal. Keeps domestic cups, continental
    competitions, second/third tiers, etc. so Elo reflects performance
    across all competitive contexts a team plays in.
    """
    df = _load_all()
    return df[~df["league_id"].isin(set(excluded))].reset_index(drop=True)


def load_training_fixtures(
    cfg: FilterConfig | None = None,
) -> pd.DataFrame:
    """Fixtures we train and test on.

    By default: target league only (Premier League), with cups excluded.
    The user can pass a different `target_league_id` to widen scope, or set
    `include_cups=True` to include cup matches in the target league (PL has
    no native cups but other leagues might in future).
    """
    cfg = cfg or FilterConfig()
    df = _load_all()
    df = df[df["league_id"] == cfg.target_league_id]

    if not cfg.include_cups:
        df = df[~df["is_cup"]]

    if cfg.additional_excluded_leagues:
        df = df[~df["league_id"].isin(set(cfg.additional_excluded_leagues))]

    return df.sort_values("date").reset_index(drop=True)


# ---------------- Split ----------------

def time_split(
    df: pd.DataFrame, split_cfg: SplitConfig | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Split a fixture dataframe by `date < cutoff` (train) vs `>=` (test).

    Both parts are returned sorted by date, with index reset.
    """
    split_cfg = split_cfg or SplitConfig()
    cutoff = split_cfg.cutoff
    train = df[df["date"] < cutoff].sort_values("date").reset_index(drop=True)
    test = df[df["date"] >= cutoff].sort_values("date").reset_index(drop=True)
    return train, test


def summarize(df: pd.DataFrame) -> dict:
    """Quick stats for logging."""
    return {
        "n_matches": len(df),
        "date_min": str(df["date"].min().date()) if len(df) else None,
        "date_max": str(df["date"].max().date()) if len(df) else None,
        "n_leagues": int(df["league_id"].nunique()),
        "n_seasons": int(df["season"].nunique()),
        "n_teams": int(
            pd.concat([df["home_team_id"], df["away_team_id"]]).nunique()
        ),
        "result_distribution": {
            "H": int((df["home_goals"] > df["away_goals"]).sum()),
            "D": int((df["home_goals"] == df["away_goals"]).sum()),
            "A": int((df["home_goals"] < df["away_goals"]).sum()),
        },
    }
