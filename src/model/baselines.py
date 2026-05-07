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
    """Predicts the training-set H/D/A rates for every match."""

    def __init__(self) -> None:
        self.p_H: float = 1 / 3
        self.p_D: float = 1 / 3
        self.p_A: float = 1 / 3

    def fit(self, X: pd.DataFrame, y: pd.Series) -> "ClimatologyBaseline":
        self.p_H, self.p_D, self.p_A = _train_distribution(y)
        return self

    def predict_proba(self, X: pd.DataFrame) -> pd.DataFrame:
        n = len(X)
        return pd.DataFrame({
            "p_H": np.full(n, self.p_H),
            "p_D": np.full(n, self.p_D),
            "p_A": np.full(n, self.p_A),
        }, index=X.index)


class EloBaseline:
    """Three-way Elo prediction.

    P(home win) from sigmoid of elo_gap (which already includes home
    advantage if it was added in feature engineering). P(draw) is a fitted
    constant from the training set's draw rate. P(away win) is the residual.

    Reasoning for the constant draw rate: draws in football are roughly a
    structural property of the league (~22-26%), not strongly dependent on
    Elo gap in any simple closed form. A constant captures this without
    over-engineering.
    """

    def __init__(self) -> None:
        self.draw_rate: float = 0.25  # default; overridden by fit()

    def fit(self, X: pd.DataFrame, y: pd.Series) -> "EloBaseline":
        self.draw_rate = float((y == "D").mean()) if len(y) else 0.25
        return self

    def predict_proba(self, X: pd.DataFrame) -> pd.DataFrame:
        gap = X["elo_gap"].values  # already includes home advantage
        # Standard Elo expectation for the home team to win or draw
        # (treating draws as half-credit). We then split that into pure
        # win probability vs draw probability using the training draw rate.
        home_score_share = 1.0 / (1.0 + 10.0 ** (-gap / 400.0))

        # Home P(W) vs P(D) split: when the model says home_score_share
        # is e.g. 0.8 (favourite), the credit comes from the win in most
        # cases, with the draw rate held constant. Specifically:
        #   p_home_win = home_score_share - draw_rate / 2
        #   p_away_win = (1 - home_score_share) - draw_rate / 2
        # which simplifies to splitting the half-credit on draws evenly.
        p_H = home_score_share - self.draw_rate / 2
        p_A = (1 - home_score_share) - self.draw_rate / 2
        # Clamp to non-negative and renormalise (in extreme Elo gaps
        # p_A can fall below zero with the simple formula above).
        p_H = np.clip(p_H, 1e-6, None)
        p_A = np.clip(p_A, 1e-6, None)
        p_D = np.full_like(p_H, self.draw_rate)
        total = p_H + p_D + p_A
        return pd.DataFrame({
            "p_H": p_H / total, "p_D": p_D / total, "p_A": p_A / total,
        }, index=X.index)
