"""Evaluation utilities for H/D/A probabilistic predictions.

Headline metric: multiclass log-loss (proper scoring rule, sensitive to
calibration). Secondary: Brier score. Calibration diagnosis via reliability
diagram. All point estimates can be wrapped in bootstrap CIs to convey
sampling uncertainty — important given our 380-match PL test set.

The probability dataframes returned by predictors throughout the pipeline
have columns `p_H`, `p_D`, `p_A` and the labels are strings 'H', 'D', 'A'.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np
import pandas as pd


CLASSES: tuple[str, str, str] = ("H", "D", "A")
_EPS = 1e-12


def _label_to_onehot(y: pd.Series) -> np.ndarray:
    """Convert ['H', 'D', 'A'] labels to a (n, 3) one-hot matrix."""
    out = np.zeros((len(y), 3), dtype=float)
    for i, c in enumerate(CLASSES):
        out[:, i] = (y == c).astype(float).values
    return out


def _prob_matrix(probs: pd.DataFrame) -> np.ndarray:
    """Extract (n, 3) probability matrix from a p_H / p_D / p_A DataFrame."""
    return np.column_stack([probs["p_H"].values, probs["p_D"].values, probs["p_A"].values])


def log_loss(y_true: pd.Series, probs: pd.DataFrame) -> float:
    """Multiclass log-loss: -mean(log(p_actual))."""
    p = _prob_matrix(probs)
    onehot = _label_to_onehot(y_true)
    p_actual = (p * onehot).sum(axis=1)
    return float(-np.log(np.clip(p_actual, _EPS, 1.0)).mean())


def brier_score(y_true: pd.Series, probs: pd.DataFrame) -> float:
    """Multiclass Brier: mean over rows of sum_class (p_c - y_c)^2.

    Range [0, 2]. 0 = perfect, 2 = always confidently wrong.
    """
    p = _prob_matrix(probs)
    onehot = _label_to_onehot(y_true)
    return float(((p - onehot) ** 2).sum(axis=1).mean())


def accuracy(y_true: pd.Series, probs: pd.DataFrame) -> float:
    """Top-1 accuracy of the argmax class."""
    p = _prob_matrix(probs)
    pred_idx = p.argmax(axis=1)
    pred = np.array([CLASSES[i] for i in pred_idx])
    return float((pred == y_true.values).mean())


# ---------------- Tail-focused calibration ----------------

def tail_brier(y_true: pd.Series, probs: pd.DataFrame, threshold: float = 0.10) -> dict:
    """Brier restricted to (row, class) pairs where p_class <= threshold.

    Captures upset detection: are the events the model thinks are unlikely
    actually happening at the rate it predicts? Returns dict with the
    Brier value and the count of rows / class-cells that contributed.
    """
    p = _prob_matrix(probs)
    onehot = _label_to_onehot(y_true)
    mask = p <= threshold
    if not mask.any():
        return {"tail_brier": float("nan"), "n_cells": 0}
    sq_err = (p - onehot) ** 2
    return {
        "tail_brier": float(sq_err[mask].mean()),
        "n_cells": int(mask.sum()),
    }


# ---------------- Reliability data ----------------

@dataclass(frozen=True)
class ReliabilityBin:
    pred_mean: float
    obs_rate: float
    n: int


def reliability_data(
    y_true: pd.Series, probs: pd.DataFrame, n_bins: int = 10,
) -> list[ReliabilityBin]:
    """Pool ALL (row, class) prediction-outcome pairs and bin by predicted
    probability. Returns one bin's data per non-empty bin.

    Pooling across the three classes gives a single calibration curve
    that's more robust at small sample sizes (PL test = 380 matches × 3
    classes = 1140 prediction-outcome pairs).
    """
    p = _prob_matrix(probs).flatten()
    onehot = _label_to_onehot(y_true).flatten()
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    bin_idx = np.digitize(p, edges, right=False) - 1
    bin_idx = np.clip(bin_idx, 0, n_bins - 1)

    bins: list[ReliabilityBin] = []
    for b in range(n_bins):
        mask = bin_idx == b
        if not mask.any():
            continue
        bins.append(ReliabilityBin(
            pred_mean=float(p[mask].mean()),
            obs_rate=float(onehot[mask].mean()),
            n=int(mask.sum()),
        ))
    return bins


def expected_calibration_error(
    y_true: pd.Series, probs: pd.DataFrame, n_bins: int = 10,
) -> float:
    """ECE: weighted-by-bin-size average distance from the y=x diagonal."""
    bins = reliability_data(y_true, probs, n_bins=n_bins)
    if not bins:
        return float("nan")
    total = sum(b.n for b in bins)
    return float(sum(b.n * abs(b.pred_mean - b.obs_rate) for b in bins) / total)


# ---------------- Bootstrap CI ----------------

def bootstrap_metric(
    metric_fn: Callable[[pd.Series, pd.DataFrame], float],
    y_true: pd.Series,
    probs: pd.DataFrame,
    n_boot: int = 1000,
    rng_seed: int = 42,
    alpha: float = 0.05,
) -> dict:
    """Bootstrap CI for any metric of the form metric(y_true, probs) -> float.

    Returns dict with point estimate and a (lo, hi) CI at confidence
    1 - alpha. Default 95% CI from 1000 bootstrap resamples.
    """
    n = len(y_true)
    if n == 0:
        return {"point": float("nan"), "lo": float("nan"), "hi": float("nan"), "n_boot": 0}

    rng = np.random.default_rng(rng_seed)
    point = metric_fn(y_true, probs)

    samples = np.empty(n_boot, dtype=float)
    y_arr = y_true.reset_index(drop=True)
    p_arr = probs.reset_index(drop=True)
    for i in range(n_boot):
        idx = rng.integers(0, n, size=n)
        samples[i] = metric_fn(y_arr.iloc[idx], p_arr.iloc[idx])

    lo, hi = np.percentile(samples, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return {"point": float(point), "lo": float(lo), "hi": float(hi), "n_boot": n_boot}


# ---------------- Convenience: full report ----------------

def evaluation_report(
    y_true: pd.Series, probs: pd.DataFrame, n_boot: int = 1000,
) -> dict:
    """Single-call summary of all the metrics we care about."""
    return {
        "log_loss":    bootstrap_metric(log_loss, y_true, probs, n_boot=n_boot),
        "brier":       bootstrap_metric(brier_score, y_true, probs, n_boot=n_boot),
        "accuracy":    bootstrap_metric(accuracy, y_true, probs, n_boot=n_boot),
        "ece":         expected_calibration_error(y_true, probs),
        "tail_brier":  tail_brier(y_true, probs, threshold=0.10),
        "n_test":      int(len(y_true)),
    }
