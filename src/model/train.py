"""Logistic regression v1 for the H/D/A modelling task.

A multinomial logistic regression with L2 regularisation, fitted on a
mean-imputed and z-score-standardised feature matrix. Deliberately simple
so it can serve as a calibrated, interpretable feature-driven baseline
before any GBM is brought in.

By default uses every numeric feature column except identifiers and dates.
Pass `feature_cols` explicitly to restrict.
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

logger = logging.getLogger(__name__)


# Columns always excluded from the feature matrix.
_EXCLUDED_BY_DEFAULT: frozenset[str] = frozenset({
    "fixture_id", "date", "season", "league_id",
    "home_team_id", "away_team_id", "result",
})


# Curated feature list, deliberately small to fit our 760-row training set
# without overfitting. Each feature is here for a defensible causal reason:
#
#   Strength:       elo_gap (with home advantage built in) is the single most
#                   informative feature; both absolute Elos kept so the model
#                   can learn league-level effects on draw rate.
#   Recent form:    gap features only (5 and 10 windows, winrate + GD), so
#                   the model sees the differential rather than two
#                   correlated absolute columns.
#   Form quality:   opponent-Elo gap captures whether form was earned
#                   against tough or weak opposition.
#   Venue:          home_venue_winrate − away_venue_winrate as a single
#                   "this is the home team's preferred surface" signal.
#   Streaks:        unbeaten-streak gap.
#   Fatigue:        rest-days gap and matches-in-last-14-days gap.
#   Context:        season match number for both sides (early-season uncertainty).
CURATED_FEATURES: tuple[str, ...] = (
    "home_pre_elo",
    "away_pre_elo",
    "elo_gap",
    "form_5_winrate_gap",
    "form_10_winrate_gap",
    "form_5_gd_avg_gap",
    "form_10_gd_avg_gap",
    "form_5_opp_elo_avg_gap",
    "venue_form_5_winrate_gap",
    "unbeaten_streak_gap",
    "rest_days_gap",
    "matches_last_14d_gap",
    "home_season_match_n",
    "away_season_match_n",
)


def _default_feature_cols(features_df: pd.DataFrame) -> list[str]:
    """Pick all numeric columns that aren't identifiers or the target.

    Used when no `feature_cols` is provided AND `use_curated=False`. The
    default behaviour of LogisticModel is to use the curated list above.
    """
    candidates = [c for c in features_df.columns if c not in _EXCLUDED_BY_DEFAULT]
    return [c for c in candidates if pd.api.types.is_numeric_dtype(features_df[c])]


class LogisticModel:
    """Multinomial logistic regression with mean imputation + standardisation.

    Calling convention mirrors the baselines in `src.model.baselines`:
      * `fit(X, y)` where X is the feature DataFrame and y is the result Series
      * `predict_proba(X)` returns DataFrame with columns p_H, p_D, p_A
    """

    def __init__(
        self,
        feature_cols: list[str] | None = None,
        C: float = 0.5,
        max_iter: int = 1000,
        use_curated: bool = True,
    ) -> None:
        """
        Args:
            feature_cols: Explicit list of columns to use as features. If
                None, behaviour depends on `use_curated`.
            C: Inverse L2 regularisation strength. Default 0.5 — heavier
                regularisation than sklearn's default 1.0 because our
                training set is small (~760 rows) and we want to prevent
                overfitting.
            max_iter: lbfgs iterations.
            use_curated: When `feature_cols` is None, use the curated list
                in `CURATED_FEATURES` (default). Set False to fall back to
                "every numeric column", which is the kitchen-sink default
                from before — useful for ablation comparisons.
        """
        self.feature_cols: list[str] | None = feature_cols
        self.C: float = C
        self.max_iter: int = max_iter
        self.use_curated: bool = use_curated

        # Fitted state
        self._impute_means: np.ndarray | None = None
        self._scale_mean: np.ndarray | None = None
        self._scale_std: np.ndarray | None = None
        self._model: LogisticRegression | None = None
        self._classes_: list[str] | None = None

    def fit(self, X: pd.DataFrame, y: pd.Series) -> "LogisticModel":
        if self.feature_cols is None:
            if self.use_curated:
                # Filter to columns that actually exist in X (defends
                # against version drift in the feature builder).
                self.feature_cols = [c for c in CURATED_FEATURES if c in X.columns]
                missing = set(CURATED_FEATURES) - set(self.feature_cols)
                if missing:
                    logger.warning(
                        "Curated feature list referenced columns not in X: %s",
                        missing,
                    )
                logger.info(
                    "LogisticModel: using %d curated features (C=%.2f)",
                    len(self.feature_cols), self.C,
                )
            else:
                self.feature_cols = _default_feature_cols(X)
                logger.info(
                    "LogisticModel: using %d default (kitchen-sink) features (C=%.2f)",
                    len(self.feature_cols), self.C,
                )

        Xf = X[self.feature_cols].astype(float).values

        # Impute NaN with column mean (computed from train only)
        means = np.nanmean(Xf, axis=0)
        # Guard against all-NaN columns
        means = np.where(np.isnan(means), 0.0, means)
        Xf = np.where(np.isnan(Xf), means, Xf)
        self._impute_means = means

        # Standardise to zero mean, unit variance
        col_mean = Xf.mean(axis=0)
        col_std = Xf.std(axis=0)
        col_std = np.where(col_std < 1e-9, 1.0, col_std)
        self._scale_mean = col_mean
        self._scale_std = col_std
        Xs = (Xf - col_mean) / col_std

        # Multinomial LR with L2 regularisation. (Don't pass multi_class
        # explicitly — sklearn 1.5+ deprecates it; the default is
        # multinomial when classes >= 3 with the lbfgs solver, which is
        # what we want.)
        self._model = LogisticRegression(
            solver="lbfgs",
            C=self.C,
            max_iter=self.max_iter,
        )
        self._model.fit(Xs, y.values)
        self._classes_ = list(self._model.classes_)
        logger.info(
            "Fit: %d rows, %d features, classes=%s",
            len(X), len(self.feature_cols), self._classes_,
        )
        return self

    def predict_proba(self, X: pd.DataFrame) -> pd.DataFrame:
        if self._model is None:
            raise RuntimeError("Call fit() first")
        assert self.feature_cols is not None
        assert self._impute_means is not None
        assert self._scale_mean is not None
        assert self._scale_std is not None
        assert self._classes_ is not None

        Xf = X[self.feature_cols].astype(float).values
        Xf = np.where(np.isnan(Xf), self._impute_means, Xf)
        Xs = (Xf - self._scale_mean) / self._scale_std
        proba = self._model.predict_proba(Xs)

        # Reorder to (p_H, p_D, p_A) regardless of sklearn class ordering
        idx_H = self._classes_.index("H")
        idx_D = self._classes_.index("D")
        idx_A = self._classes_.index("A")
        return pd.DataFrame({
            "p_H": proba[:, idx_H],
            "p_D": proba[:, idx_D],
            "p_A": proba[:, idx_A],
        }, index=X.index)

    @property
    def feature_importance_(self) -> pd.DataFrame | None:
        """Per-feature signed coefficients from the LR fit, one row per
        (feature, class). Useful for inspecting what drives predictions.
        Returns None until `fit()` is called.
        """
        if self._model is None or self.feature_cols is None or self._classes_ is None:
            return None
        coef = self._model.coef_  # shape (n_classes, n_features)
        return pd.DataFrame(
            coef.T,
            index=self.feature_cols,
            columns=[f"coef_{c}" for c in self._classes_],
        )
