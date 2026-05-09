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

# LightGBM is imported lazily inside GBMModel so the rest of the module
# loads cleanly even if lightgbm isn't installed (some dev / CI envs).
try:
    import lightgbm as lgb  # type: ignore
    _LGB_AVAILABLE = True
except ImportError:  # pragma: no cover
    _LGB_AVAILABLE = False

logger = logging.getLogger(__name__)


# Columns always excluded from the feature matrix.
# NOTE: league_id is intentionally NOT excluded here — when scope spans
# multiple leagues, it carries real signal (different leagues have
# different draw rates and home-advantage strengths). See _CATEGORICAL_COLS
# below for how it's handled.
_EXCLUDED_BY_DEFAULT: frozenset[str] = frozenset({
    "fixture_id", "date", "season",
    "home_team_id", "away_team_id", "result",
})

# Columns that should be treated as categorical, not numeric. For the LR
# we one-hot encode them; for the GBM we pass them through as native
# LightGBM categorical features. Currently just league_id.
_CATEGORICAL_COLS: tuple[str, ...] = ("league_id",)


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
    "league_id",                  # categorical — league-level effects
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
    # Promotion/relegation: did each team change tier this season?
    "home_league_changed",
    "away_league_changed",
    "league_strength_change_gap",
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
        # Final post-one-hot column list, captured at fit time so predict
        # can align (test sets may be missing or have extra category levels).
        self._encoded_columns: list[str] | None = None

    def _prepare_features(self, X: pd.DataFrame, fit_phase: bool) -> pd.DataFrame:
        """One-hot encode any categorical columns in `feature_cols`.

        At fit time this captures the resulting column list; at predict time
        it reindexes to that list so unseen category levels become all-zeros
        and missing levels are filled with zero. This keeps the column count
        / order consistent between fit and predict.
        """
        assert self.feature_cols is not None
        df = X[self.feature_cols].copy()
        cats = [c for c in _CATEGORICAL_COLS if c in df.columns]
        if cats:
            df = pd.get_dummies(df, columns=cats, dtype=float, dummy_na=False)
        if fit_phase:
            self._encoded_columns = df.columns.tolist()
        else:
            assert self._encoded_columns is not None
            df = df.reindex(columns=self._encoded_columns, fill_value=0.0)
        return df

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

        Xf_df = self._prepare_features(X, fit_phase=True)
        Xf = Xf_df.values.astype(float)

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

        Xf_df = self._prepare_features(X, fit_phase=False)
        Xf = Xf_df.values.astype(float)
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
        (encoded-feature, class). Returns None until `fit()` is called.

        Indexed by the post-one-hot column names (so categorical features
        get one row per level), not the original `feature_cols`.
        """
        if self._model is None or self._classes_ is None or self._encoded_columns is None:
            return None
        coef = self._model.coef_  # shape (n_classes, n_features)
        return pd.DataFrame(
            coef.T,
            index=self._encoded_columns,
            columns=[f"coef_{c}" for c in self._classes_],
        )


class GBMModel:
    """Gradient-boosted decision trees (LightGBM) for the H/D/A task.

    Compared to LogisticModel:
      * Handles NaN natively — no mean imputation needed.
      * Handles correlated features cleanly — splits on the most informative
        column at each node, doesn't get confused by multiple "form" columns
        encoding similar information.
      * Captures non-linear interactions automatically (e.g. "elo_gap
        matters less when the away team is on a hot streak").

    Default hyperparameters tuned for our small Premier League training
    set (~760 rows). The defaults below were the best of a sweep against
    the Elo baseline; relaxing any of them caused the GBM to overfit
    badly (kitchen-sink + standard GBM hyperparams gives log-loss ~1.20
    vs Elo's 1.00).

    Defaults:
      use_curated=True         use the 14-feature curated list, not all 64
      num_leaves=7             very shallow trees
      min_data_in_leaf=50      no leaf with fewer than 50 matches
      max_depth=3              hard cap at depth 3
      learning_rate=0.03       slow boosting
      n_estimators=100         fixed; can swap for early-stopping later

    With more data (e.g. once we widen scope past PL to top European
    leagues, ~5,500 matches), these can probably be relaxed.
    """

    def __init__(
        self,
        feature_cols: list[str] | None = None,
        n_estimators: int = 100,
        learning_rate: float = 0.03,
        num_leaves: int = 7,
        min_data_in_leaf: int = 50,
        max_depth: int = 3,
        reg_lambda: float = 0.5,
        random_state: int = 42,
        use_curated: bool = True,
    ) -> None:
        if not _LGB_AVAILABLE:
            raise RuntimeError(
                "lightgbm is not installed. Install with `pip install lightgbm`."
            )
        self.feature_cols = feature_cols
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.num_leaves = num_leaves
        self.min_data_in_leaf = min_data_in_leaf
        self.max_depth = max_depth
        self.reg_lambda = reg_lambda
        self.random_state = random_state
        # GBMs benefit from MORE features by default — they can ignore the
        # noise. The curated list is still available for ablation runs.
        self.use_curated = use_curated

        self._model: "lgb.LGBMClassifier | None" = None
        self._classes_: list[str] | None = None

    def fit(self, X: pd.DataFrame, y: pd.Series) -> "GBMModel":
        if self.feature_cols is None:
            if self.use_curated:
                self.feature_cols = [c for c in CURATED_FEATURES if c in X.columns]
            else:
                self.feature_cols = _default_feature_cols(X)
        logger.info(
            "GBMModel: %d features (n_estimators=%d, lr=%.2f, num_leaves=%d, "
            "min_data_in_leaf=%d, max_depth=%d)",
            len(self.feature_cols), self.n_estimators, self.learning_rate,
            self.num_leaves, self.min_data_in_leaf, self.max_depth,
        )

        # LightGBM accepts categoricals natively as int columns named in
        # `categorical_feature`. Don't cast those to float — keep them int.
        cat_cols = [c for c in _CATEGORICAL_COLS if c in self.feature_cols]
        Xf = X[self.feature_cols].copy()
        for c in self.feature_cols:
            if c in cat_cols:
                Xf[c] = Xf[c].astype("int32")
            else:
                Xf[c] = Xf[c].astype(float)

        self._model = lgb.LGBMClassifier(
            objective="multiclass",
            num_class=3,
            n_estimators=self.n_estimators,
            learning_rate=self.learning_rate,
            num_leaves=self.num_leaves,
            min_data_in_leaf=self.min_data_in_leaf,
            max_depth=self.max_depth,
            reg_lambda=self.reg_lambda,
            random_state=self.random_state,
            verbose=-1,
            force_col_wise=True,
        )
        self._model.fit(
            Xf, y.values,
            categorical_feature=cat_cols if cat_cols else "auto",
        )
        self._classes_ = list(self._model.classes_)
        self._cat_cols = cat_cols
        return self

    def predict_proba(self, X: pd.DataFrame) -> pd.DataFrame:
        if self._model is None or self.feature_cols is None or self._classes_ is None:
            raise RuntimeError("Call fit() first")
        Xf = X[self.feature_cols].copy()
        for c in self.feature_cols:
            if c in getattr(self, "_cat_cols", []):
                Xf[c] = Xf[c].astype("int32")
            else:
                Xf[c] = Xf[c].astype(float)
        proba = self._model.predict_proba(Xf)
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
        """Gain-based importance per feature, summed across all classes.

        LightGBM's "gain" is the total reduction in loss attributable to
        splits on that feature — high gain = informative feature. This is
        a single number per feature, unlike LR which has one per class.
        """
        if self._model is None or self.feature_cols is None:
            return None
        return (
            pd.DataFrame({
                "feature": self.feature_cols,
                "gain": self._model.booster_.feature_importance(importance_type="gain"),
                "splits": self._model.booster_.feature_importance(importance_type="split"),
            })
            .sort_values("gain", ascending=False)
            .reset_index(drop=True)
        )
