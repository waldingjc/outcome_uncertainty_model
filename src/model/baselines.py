"""Baseline predictors for the H/D/A modelling task.

Two baselines, both deliberately stupid so that any "real" model can
clearly beat them:

  * `ClimatologyBaseline` — predicts the training-set H/D/A frequencies for
    every match. Captures only the unconditional outcome distribution; if a
    feature-based model can't beat this, our features are useless.

  * `EloBaseline` — closed-form three-way probability from the Elo gap
    (with home advantage). P(home win) comes from the standard Elo formula;
    P(draw) is fitted as a constant from the training set; P(away win) is
    the residual. This is the standard academic benchmark.

Both predictors return three columns ('p_H', 'p_D', 'p_A') summing to 1.0.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


CLASSES: tuple[str, str, str] = ("H", "D", "A")


def _train_distribution(y: pd.Series) -> tuple[float, float, float]:
    """Return (p_H, p_D, p_A) from a training set of result labels."""
    n = len(y)
    if n == 0:
        return (1 / 3, 1 / 3, 1 / 3)
    return (
        float((y == "H").sum() / n),
        float((y == "D").sum() / n),
        float((y == "A").sum() / n),
    )


class ClimatologyBaseline:
    """Predicts the training-set H/D/A rates for every match.

    When `X` contains a `league_id` column, fits per-league rates so each
    league gets its own H/D/A frequency. Falls back to the global rate
    for any league seen at predict-time but not at fit-time. Without the
    column, behaves as a single-league climatology.
    """

    def __init__(self) -> None:
        self._global: tuple[float, float, float] = (1 / 3, 1 / 3, 1 / 3)
        self._per_league: dict[int, tuple[float, float, float]] = {}

    def fit(self, X: pd.DataFrame, y: pd.Series) -> "ClimatologyBaseline":
        self._global = _train_distribution(y)
        if "league_id" in X.columns:
            self._per_league = {}
            for lid, group_y in y.groupby(X["league_id"]):
                self._per_league[int(lid)] = _train_distribution(group_y)
        return self

    def predict_proba(self, X: pd.DataFrame) -> pd.DataFrame:
        if "league_id" not in X.columns or not self._per_league:
            n = len(X)
            return pd.DataFrame({
                "p_H": np.full(n, self._global[0]),
                "p_D": np.full(n, self._global[1]),
                "p_A": np.full(n, self._global[2]),
            }, index=X.index)
        rows = []
        for lid in X["league_id"]:
            rows.append(self._per_league.get(int(lid), self._global))
        arr = np.array(rows)
        return pd.DataFrame({
            "p_H": arr[:, 0], "p_D": arr[:, 1], "p_A": arr[:, 2],
        }, index=X.index)


class EloBaseline:
    """Three-way Elo prediction with optional per-league draw rate.

    P(home win) from sigmoid of elo_gap (which already includes home
    advantage if it was added in feature engineering). P(draw) is a fitted
    constant — globally or per-league when `X` contains `league_id`.
    P(away win) is the residual.

    Reasoning for the constant draw rate: draws in football are roughly a
    structural property of the league (~22-26%), not strongly dependent on
    Elo gap in any simple closed form. Per-league constants capture cross-
    league variation (Italian draws ~28%, German draws ~22%) without
    over-engineering.
    """

    def __init__(self) -> None:
        self._global_draw_rate: float = 0.25
        self._per_league_draw_rate: dict[int, float] = {}

    def fit(self, X: pd.DataFrame, y: pd.Series) -> "EloBaseline":
        self._global_draw_rate = (
            float((y == "D").mean()) if len(y) else 0.25
        )
        if "league_id" in X.columns:
            self._per_league_draw_rate = {}
            for lid, group_y in y.groupby(X["league_id"]):
                if len(group_y) > 0:
                    self._per_league_draw_rate[int(lid)] = float((group_y == "D").mean())
        return self

    def _draw_rate_for(self, league_id: int | float | None) -> float:
        if league_id is None:
            return self._global_draw_rate
        return self._per_league_draw_rate.get(int(league_id), self._global_draw_rate)

    def predict_proba(self, X: pd.DataFrame) -> pd.DataFrame:
        gap = X["elo_gap"].values  # already includes home advantage
        if "league_id" in X.columns:
            draw_rates = X["league_id"].apply(self._draw_rate_for).values
        else:
            draw_rates = np.full(len(X), self._global_draw_rate)

        # Standard Elo expectation for the home team to win or draw
        # (treating draws as half-credit). We then split that into pure
        # win probability vs draw probability using the (per-league) draw rate.
        home_score_share = 1.0 / (1.0 + 10.0 ** (-gap / 400.0))

        p_H = home_score_share - draw_rates / 2
        p_A = (1 - home_score_share) - draw_rates / 2
        # Clamp to non-negative and renormalise (in extreme Elo gaps
        # p_A can fall below zero with the simple formula above).
        p_H = np.clip(p_H, 1e-6, None)
        p_A = np.clip(p_A, 1e-6, None)
        p_D = draw_rates
        total = p_H + p_D + p_A
        return pd.DataFrame({
            "p_H": p_H / total, "p_D": p_D / total, "p_A": p_A / total,
        }, index=X.index)
