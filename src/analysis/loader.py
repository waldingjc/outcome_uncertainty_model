"""
Load fixture and odds data from SQLite into pandas DataFrames.

PANDAS PRIMER
=============
A DataFrame is a 2-dimensional table — like a spreadsheet — where each column
has a name and a consistent data type. Every row has an integer index (0, 1, 2…)
by default, though we often replace that with a meaningful column like fixture_id.

Key concepts used in this module:

  pd.read_sql(query, con)
      Runs a SQL query against a database connection and returns the result
      as a DataFrame. Each column in the SELECT becomes a DataFrame column.

  df.set_index(col)
      Replaces the default integer row numbers with the values from `col`.
      Makes row lookup by that value fast (like a dict key).

  df.merge(other, on=col, how="left")
      Joins two DataFrames on a shared column — equivalent to SQL JOIN.
      how="left" keeps all rows from the left DataFrame even if there's no
      match on the right (unmatched right columns become NaN).

  df.groupby(col).agg(...)
      Groups rows that share the same value in `col`, then applies an
      aggregation function (mean, sum, etc.) to other columns within each group.
      Returns one row per unique group value.

  df[condition]
      Boolean indexing — filters rows where `condition` is True.
      e.g. df[df["season"] == 2024] returns only 2024 rows.
"""

import sqlite3

import pandas as pd

from src.db.schema import DB_PATH


def _connect():
    """Return a read-only SQLite connection to the results database."""
    # isolation_level=None puts SQLite in autocommit mode, which is fine
    # for read-only use and avoids leaving implicit transactions open.
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def load_fixtures(
    league_id: int = None,
    season: int = None,
) -> pd.DataFrame:
    """
    Load finished fixtures from the database into a DataFrame.

    Each row is one match. Columns mirror the fixtures table:
    fixture_id, date, league_id, league_name, season, round,
    home_team_id, home_team_name, away_team_id, away_team_name,
    home_goals, away_goals, home_goals_ht, away_goals_ht,
    status, venue_name, venue_city, referee.

    Parameters
    ----------
    league_id : int, optional
        Filter to a specific league. None returns all leagues.
    season : int, optional
        Filter to a specific season year. None returns all seasons.

    Returns
    -------
    pd.DataFrame
        One row per fixture, indexed by fixture_id.
    """
    # Build the WHERE clause dynamically based on which filters were supplied.
    # Using parameterised values (the ? placeholders) prevents SQL injection.
    conditions = []
    params = []

    if league_id is not None:
        conditions.append("league_id = ?")
        params.append(league_id)
    if season is not None:
        conditions.append("season = ?")
        params.append(season)

    where = f"WHERE {' AND '.join(conditions)}" if conditions else ""
    query = f"SELECT * FROM fixtures {where} ORDER BY date"

    with _connect() as conn:
        # pd.read_sql executes the query and returns a DataFrame.
        # The `params` list maps to the ? placeholders in order.
        df = pd.read_sql(query, conn, params=params)

    # parse_dates converts the ISO date strings stored in SQLite ("2024-08-17T…")
    # into proper pandas Timestamp objects, enabling date arithmetic and
    # time-based filtering later (e.g. df[df["date"].dt.month == 12]).
    df["date"] = pd.to_datetime(df["date"], utc=True)

    # set_index promotes fixture_id from a regular column to the row label.
    # This means df.loc[1035047] retrieves that fixture directly, and joins
    # on fixture_id are faster because pandas uses it as a lookup key.
    df = df.set_index("fixture_id")

    return df


def load_odds(
    league_id: int = None,
    season: int = None,
) -> pd.DataFrame:
    """
    Load pre-match Match Winner odds from the database.

    Each row is one bookmaker's odds for one fixture. Columns:
    fixture_id, bookmaker_id, bookmaker_name, home_odds, draw_odds, away_odds.

    Multiple rows can share the same fixture_id (one per bookmaker).

    Parameters
    ----------
    league_id : int, optional
        Filter by league. Requires a JOIN to the fixtures table.
    season : int, optional
        Filter by season. Requires a JOIN to the fixtures table.

    Returns
    -------
    pd.DataFrame
        One row per (fixture, bookmaker) pair.
    """
    conditions = []
    params = []

    # Odds don't store league/season directly — we join to fixtures to filter.
    if league_id is not None:
        conditions.append("f.league_id = ?")
        params.append(league_id)
    if season is not None:
        conditions.append("f.season = ?")
        params.append(season)

    where = f"WHERE {' AND '.join(conditions)}" if conditions else ""

    query = f"""
        SELECT
            o.fixture_id,
            o.bookmaker_id,
            o.bookmaker_name,
            o.home_odds,
            o.draw_odds,
            o.away_odds
        FROM odds o
        JOIN fixtures f ON o.fixture_id = f.fixture_id
        {where}
    """

    with _connect() as conn:
        df = pd.read_sql(query, conn, params=params)

    return df


def load_consensus_odds(
    league_id: int = None,
    season: int = None,
) -> pd.DataFrame:
    """
    Load odds averaged across all bookmakers, returning one row per fixture.

    Averaging across bookmakers smooths out individual pricing errors and
    gives a more reliable estimate of the true implied probability.

    Returns
    -------
    pd.DataFrame
        Indexed by fixture_id, with columns:
        home_odds_mean, draw_odds_mean, away_odds_mean,
        bookmaker_count (how many bookmakers contributed).
    """
    odds_df = load_odds(league_id=league_id, season=season)

    # groupby("fixture_id") splits the DataFrame into one group per fixture.
    # .agg() then applies named aggregations to each group:
    #   - mean() across home/draw/away_odds gives the consensus price
    #   - "count" on bookmaker_id tells us how many bookmakers contributed
    # The result has one row per fixture_id.
    consensus = (
        odds_df.groupby("fixture_id")
        .agg(
            home_odds_mean=("home_odds", "mean"),
            draw_odds_mean=("draw_odds", "mean"),
            away_odds_mean=("away_odds", "mean"),
            bookmaker_count=("bookmaker_id", "count"),
        )
    )

    # groupby can return nullable Float64 dtype (using pd.NA) rather than
    # numpy float64 (using np.nan), depending on the pandas version and input
    # dtypes. Force numpy float64 here so all downstream arithmetic and
    # matplotlib calls receive plain NaN rather than pd.NA.
    for col in ["home_odds_mean", "draw_odds_mean", "away_odds_mean"]:
        consensus[col] = consensus[col].astype("float64")

    return consensus


def load_fixtures_with_odds(
    league_id: int = None,
    season: int = None,
) -> pd.DataFrame:
    """
    Return a combined DataFrame: one row per fixture with consensus odds attached.

    This is the primary input for the metrics and visualisation modules.

    Columns include everything from load_fixtures() plus:
    home_odds_mean, draw_odds_mean, away_odds_mean, bookmaker_count.

    Fixtures with no odds data are retained (odds columns will be NaN),
    because we may still want to analyse their results even without price context.

    Returns
    -------
    pd.DataFrame
        One row per fixture, indexed by fixture_id.
    """
    fixtures = load_fixtures(league_id=league_id, season=season)
    consensus = load_consensus_odds(league_id=league_id, season=season)

    # merge joins two DataFrames on matching index/column values.
    # how="left" means: keep every row from `fixtures`, attach odds where
    # available, fill with NaN where not. This is equivalent to a SQL LEFT JOIN.
    # left_index=True / right_index=True tells pandas to join on the index
    # (fixture_id) of both DataFrames rather than a named column.
    combined = fixtures.merge(consensus, how="left", left_index=True, right_index=True)

    return combined
