"""
Compute outcome-expectedness metrics from fixtures and odds.

CORE CONCEPTS
=============

Implied Probability
-------------------
A decimal odd of 2.50 means the bookmaker is pricing the event at a 40%
chance of occurring: implied_prob = 1 / odd = 1 / 2.50 = 0.40.

Because bookmakers build in a margin (the "overround"), the three implied
probabilities for a match (home + draw + away) sum to slightly more than 1.0.
We normalise them so they sum to exactly 1.0, giving a cleaner probability
estimate that removes the bookmaker margin.

Upset Score
-----------
For each fixture we:
  1. Derive the normalised implied probability for each outcome.
  2. Determine the actual outcome (home win / draw / away win).
  3. The upset score is the implied probability of the FAVOURITE minus the
     implied probability of the ACTUAL outcome.

     - Score near 0: result matched the favourite (expected outcome).
     - Score near 1: a heavy favourite lost to a heavy underdog (big upset).
     - Negative score: the favourite won (actual > favourite probability
       is impossible by definition, but floating point can produce tiny
       negatives which we clip to 0).

This gives a continuous measure of "how surprising was this result?"
that we can use to filter, rank, and cluster events.
"""

import pandas as pd


# ---------------------------------------------------------------------------
# Implied probability
# ---------------------------------------------------------------------------

def add_implied_probabilities(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add normalised implied probability columns to a fixtures+odds DataFrame.

    Adds three new columns:
      prob_home, prob_draw, prob_away  — each in [0, 1], summing to 1.0.

    Rows with missing odds (NaN) produce NaN probabilities.

    Parameters
    ----------
    df : pd.DataFrame
        Output of loader.load_fixtures_with_odds(). Must contain
        home_odds_mean, draw_odds_mean, away_odds_mean columns.

    Returns
    -------
    pd.DataFrame
        Original DataFrame with three probability columns appended.
        The original is not modified (a copy is returned).
    """
    # Work on a copy so we never silently mutate the caller's DataFrame.
    # This is important in pandas — many operations return views, not copies,
    # and modifying a view can change the original unexpectedly.
    df = df.copy()

    # Raw implied probabilities: 1 / odds for each outcome.
    # pandas applies arithmetic operations column-wise automatically,
    # so this divides every value in the column by 1 in one vectorised step —
    # no loop needed and much faster than iterating row by row.
    raw_home = 1 / df["home_odds_mean"]
    raw_draw = 1 / df["draw_odds_mean"]
    raw_away = 1 / df["away_odds_mean"]

    # Total across the three outcomes (will be > 1.0 due to bookmaker margin).
    total = raw_home + raw_draw + raw_away

    # Normalise by dividing each raw probability by the total.
    # After this, prob_home + prob_draw + prob_away == 1.0 for every row.
    df["prob_home"] = raw_home / total
    df["prob_draw"] = raw_draw / total
    df["prob_away"] = raw_away / total

    return df


# ---------------------------------------------------------------------------
# Actual outcome
# ---------------------------------------------------------------------------

def add_actual_outcome(df: pd.DataFrame) -> pd.DataFrame:
    """
    Derive the actual match outcome from the scoreline.

    Adds one column:
      outcome — "home_win", "draw", or "away_win". NaN if goals are missing.

    Parameters
    ----------
    df : pd.DataFrame
        Must contain home_goals and away_goals columns.

    Returns
    -------
    pd.DataFrame
        Original DataFrame with outcome column appended.
    """
    df = df.copy()

    # pd.cut / np.select / apply are all valid here; we use numpy-style
    # vectorised selection via pd.Series methods for clarity.
    #
    # df["home_goals"] > df["away_goals"] produces a boolean Series —
    # a column of True/False values, one per row — which we then use
    # to build the outcome column conditionally with np.select-style logic.
    conditions = [
        df["home_goals"] > df["away_goals"],
        df["home_goals"] == df["away_goals"],
        df["home_goals"] < df["away_goals"],
    ]
    choices = ["home_win", "draw", "away_win"]

    # numpy's select picks from `choices` based on the first True condition
    # for each row. default=pd.NA handles rows where goals are NaN.
    import numpy as np
    df["outcome"] = np.select(conditions, choices, default=pd.NA)

    # Convert object dtype to pandas StringDtype so NaN is represented as
    # pd.NA (the pandas missing value) rather than the string "None".
    df["outcome"] = df["outcome"].astype("string")

    return df


# ---------------------------------------------------------------------------
# Upset score
# ---------------------------------------------------------------------------

def add_upset_score(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute a continuous measure of how surprising each result was.

    Upset score = P(favourite) − P(actual outcome), clipped to [0, 1].

    A score of 0 means the highest-probability outcome occurred.
    A score approaching 1 means a near-certainty favourite lost.

    Adds three columns:
      favourite     — "home_win", "draw", or "away_win"
      prob_actual   — implied probability assigned to what actually happened
      upset_score   — the surprise magnitude in [0, 1]

    Parameters
    ----------
    df : pd.DataFrame
        Must have prob_home, prob_draw, prob_away, and outcome columns.
        Run add_implied_probabilities() and add_actual_outcome() first.

    Returns
    -------
    pd.DataFrame
        Original DataFrame with three new columns appended.
    """
    df = df.copy()

    # Build a lookup from outcome label → its probability column.
    # We'll use this to extract the probability of whatever actually happened.
    prob_cols = {
        "home_win": "prob_home",
        "draw":     "prob_draw",
        "away_win": "prob_away",
    }

    # Identify the favourite: the outcome with the highest implied probability.
    # df[["prob_home", "prob_draw", "prob_away"]].idxmax(axis=1) returns a
    # Series where each value is the column name with the highest value in
    # that row. We then map those column names back to outcome labels.
    col_to_outcome = {v: k for k, v in prob_cols.items()}
    df["favourite"] = (
        df[["prob_home", "prob_draw", "prob_away"]]
        .idxmax(axis=1)
        .map(col_to_outcome)
    )

    # Extract the implied probability of the actual outcome for each row.
    # We can't simply do df[prob_cols[df["outcome"]]] because outcome varies
    # per row. Instead we use apply() which calls a function once per row,
    # passing the row as a Series so we can reference multiple columns together.
    #
    # apply(axis=1) is slower than vectorised operations for large DataFrames,
    # but the logic here genuinely requires per-row conditional column access,
    # which is the correct use case for apply.
    def _prob_of_actual(row):
        outcome = row["outcome"]
        if pd.isna(outcome):
            return float("nan")
        col = prob_cols.get(outcome)
        return row[col] if col else float("nan")

    # pd.to_numeric forces the column to float64, converting any pd.NA or
    # non-numeric values to NaN. This is necessary because apply() can return
    # an object-dtype column when values are mixed, leaving pd.NA values that
    # numpy and matplotlib cannot handle.
    df["prob_actual"] = pd.to_numeric(df.apply(_prob_of_actual, axis=1), errors="coerce").astype("float64")

    # Probability of the favourite outcome.
    def _prob_of_favourite(row):
        fav = row["favourite"]
        if pd.isna(fav):
            return float("nan")
        col = prob_cols.get(fav)
        return row[col] if col else float("nan")

    df["prob_favourite"] = pd.to_numeric(df.apply(_prob_of_favourite, axis=1), errors="coerce").astype("float64")

    # Upset score: how much more likely the favourite was than what happened.
    # clip(lower=0) removes tiny floating-point negatives that occur when
    # the favourite actually won (prob_favourite ≈ prob_actual).
    df["upset_score"] = (df["prob_favourite"] - df["prob_actual"]).clip(lower=0)

    return df


# ---------------------------------------------------------------------------
# Convenience wrapper
# ---------------------------------------------------------------------------

def compute_all(df: pd.DataFrame) -> pd.DataFrame:
    """
    Apply implied probabilities, actual outcome, and upset score in one call.

    Parameters
    ----------
    df : pd.DataFrame
        Output of loader.load_fixtures_with_odds().

    Returns
    -------
    pd.DataFrame
        Enriched DataFrame ready for analysis and visualisation.
    """
    df = add_implied_probabilities(df)
    df = add_actual_outcome(df)
    df = add_upset_score(df)
    return df
