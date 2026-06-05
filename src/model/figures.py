"""Visualisations for the modelling pipeline's evaluation output.

Produces a single 2x2 figure that tells the story of how each model performs:
  * Reliability diagram (per-model overlay) — calibration check
  * Log-loss bars with bootstrap 95% CI — headline metric
  * Brier-score bars with bootstrap 95% CI — secondary metric
  * Tail Brier (predictions <=10%) — upset detection quality

Use via run_and_plot() which threads the same dataset through baselines and
trained models, then writes the figure to data/figures/.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.analysis._style import apply_dark_style
from src.model.evaluate import (
    bootstrap_metric, brier_score, expected_calibration_error,
    log_loss, reliability_data, tail_brier,
)


_FIGURES_DIR = Path(__file__).parents[2] / "data" / "figures"

# Stable colour map across panels so each model is the same colour everywhere.
# Tuned for the dark theme — saturated enough to read on near-black panels.
MODEL_COLOURS: dict[str, str] = {
    "Climatology": "#c084fc",  # lilac
    "Elo":         "#62b6cb",  # cyan-blue (replaces dark teal that was invisible)
    "LR":          "#76c893",  # green
    "GBM":         "#f78737",  # orange — headline model gets the accent color
}


@dataclass
class ModelResult:
    name: str
    probs: pd.DataFrame
    y_true: pd.Series

    def metric(self, fn, n_boot: int = 1000) -> dict:
        return bootstrap_metric(fn, self.y_true, self.probs, n_boot=n_boot)


def _plot_reliability(ax, results: list[ModelResult], n_bins: int = 12) -> None:
    """Per-model overlaid reliability diagram (pooled across H/D/A classes)."""
    # Diagonal reference (perfect calibration)
    ax.plot([0, 1], [0, 1], color="grey", linestyle="--", alpha=0.6, label="perfect")

    for r in results:
        bins = reliability_data(r.y_true, r.probs, n_bins=n_bins)
        if not bins:
            continue
        x = [b.pred_mean for b in bins]
        y = [b.obs_rate for b in bins]
        sizes = [max(8, b.n / 10) for b in bins]  # marker size scales with bin n
        c = MODEL_COLOURS.get(r.name, "black")
        ax.plot(x, y, color=c, linewidth=1.4, alpha=0.85)
        ax.scatter(x, y, s=sizes, color=c, edgecolors="#161719",
                   linewidths=0.6, label=r.name, zorder=5)

    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_xlabel("Predicted probability")
    ax.set_ylabel("Observed frequency")
    ax.set_title("Reliability diagram (pooled H/D/A)\nclose to diagonal = well calibrated")
    ax.legend(loc="upper left", fontsize=9)
    ax.grid(alpha=0.3)


def _plot_metric_bars(
    ax, results: list[ModelResult], metric_fn, title: str, ylabel: str,
    n_boot: int = 1000, lower_is_better: bool = True,
) -> None:
    """Horizontal bar chart of a single metric across models, with 95% CI."""
    rows = []
    for r in results:
        ci = r.metric(metric_fn, n_boot=n_boot)
        rows.append((r.name, ci["point"], ci["lo"], ci["hi"]))
    df = pd.DataFrame(rows, columns=["name", "point", "lo", "hi"])
    df = df.sort_values("point", ascending=lower_is_better)

    y = np.arange(len(df))
    err_low  = (df["point"] - df["lo"]).values
    err_high = (df["hi"] - df["point"]).values
    bars = ax.barh(
        y, df["point"],
        xerr=[err_low, err_high],
        color=[MODEL_COLOURS.get(n, "grey") for n in df["name"]],
        ecolor="#e6e6e8", capsize=4, height=0.6,
    )
    ax.set_yticks(y, df["name"])
    ax.set_xlabel(ylabel)
    ax.set_title(title)
    ax.grid(axis="x", alpha=0.3)
    for bar, val in zip(bars, df["point"]):
        ax.text(val, bar.get_y() + bar.get_height() / 2,
                f"  {val:.4f}", va="center", fontsize=9)


def _plot_tail_brier(ax, results: list[ModelResult], threshold: float = 0.10) -> None:
    """Bar chart of tail-Brier (predictions <= threshold) per model."""
    rows = []
    for r in results:
        t = tail_brier(r.y_true, r.probs, threshold=threshold)
        ece = expected_calibration_error(r.y_true, r.probs)
        rows.append((r.name, t["tail_brier"], t["n_cells"], ece))
    df = pd.DataFrame(rows, columns=["name", "tail_brier", "n_cells", "ece"])
    df = df.sort_values("tail_brier", ascending=True)

    y = np.arange(len(df))
    bars = ax.barh(
        y, df["tail_brier"],
        color=[MODEL_COLOURS.get(n, "grey") for n in df["name"]],
        height=0.6,
    )
    ax.set_yticks(y, df["name"])
    ax.set_xlabel(f"Brier on predictions <= {threshold:.0%}")
    ax.set_title("Tail behaviour\n(low = honest about long-shots)")
    ax.grid(axis="x", alpha=0.3)
    for bar, row in zip(bars, df.itertuples()):
        ax.text(
            row.tail_brier, bar.get_y() + bar.get_height() / 2,
            f"  {row.tail_brier:.4f}  (n={row.n_cells}, ECE={row.ece:.3f})",
            va="center", fontsize=8,
        )


def plot_evaluation(
    results: list[ModelResult], out_path: Path,
    n_boot: int = 1000, suptitle: str | None = None,
) -> Path:
    """Produce the 2x2 evaluation figure."""
    apply_dark_style()
    fig, axes = plt.subplots(2, 2, figsize=(16, 11), constrained_layout=True)

    if suptitle:
        fig.suptitle(suptitle, fontsize=14, fontweight="bold")

    _plot_reliability(axes[0, 0], results)
    _plot_metric_bars(axes[0, 1], results, log_loss,
                      "Log-loss (bootstrap 95% CI)", "log-loss", n_boot=n_boot)
    _plot_metric_bars(axes[1, 0], results, brier_score,
                      "Brier score (bootstrap 95% CI)", "Brier", n_boot=n_boot)
    _plot_tail_brier(axes[1, 1], results)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=140, bbox_inches="tight")
    plt.close(fig)
    return out_path


# ---------------- End-to-end runner ----------------

def run_and_plot(
    target_league_ids: frozenset[int] | None = None,
    out_path: Path | None = None,
    n_boot: int = 1000,
    cutoff=None,
) -> Path:
    """Train climatology / Elo / LR / GBM, then write a 2x2 figure
    summarising the evaluation. Returns the output path.
    """
    # Local imports to keep the module light when used purely as a helper.
    from src.analysis.strength import (
        DEFAULT_BASE_RATING, compute_elo_ratings, compute_league_elo,
        primary_league_map,
    )
    from src.model.baselines import ClimatologyBaseline, EloBaseline
    from src.model.data import (
        DEFAULT_TARGET_LEAGUE, TRAIN_CUTOFF, SplitConfig,
        load_fixtures_for_elo, time_split,
    )
    from src.model.features import build_features
    from src.model.train import GBMModel, LogisticModel

    target_league_ids = target_league_ids or frozenset({DEFAULT_TARGET_LEAGUE})
    cutoff = cutoff or TRAIN_CUTOFF

    elo_corpus = load_fixtures_for_elo()
    pre = elo_corpus[elo_corpus["date"] < cutoff].reset_index(drop=True)
    seeds_by_league, _ = compute_league_elo(pre)
    pmap = primary_league_map(pre)
    team_seeds = {
        tid: seeds_by_league.get(lid, DEFAULT_BASE_RATING)
        for tid, (lid, _) in pmap.items()
    }
    elo_with_pre, _ = compute_elo_ratings(elo_corpus, team_seeds=team_seeds)

    target = elo_with_pre[
        elo_with_pre["league_id"].isin(set(target_league_ids))
    ].reset_index(drop=True)
    if "is_cup" in target.columns:
        target = target[~target["is_cup"]].reset_index(drop=True)

    features = build_features(target, elo_with_pre, league_strength=seeds_by_league)
    train, test = time_split(features, SplitConfig(cutoff=cutoff))
    y_train, y_test = train["result"], test["result"]

    clim = ClimatologyBaseline().fit(train, y_train)
    elo = EloBaseline().fit(train, y_train)
    lr = LogisticModel().fit(train, y_train)
    gbm = GBMModel().fit(train, y_train)

    results = [
        ModelResult("Climatology", clim.predict_proba(test), y_test),
        ModelResult("Elo",         elo.predict_proba(test),  y_test),
        ModelResult("LR",          lr.predict_proba(test),   y_test),
        ModelResult("GBM",         gbm.predict_proba(test),  y_test),
    ]

    if out_path is None:
        leagues_label = (
            "PL" if target_league_ids == frozenset({DEFAULT_TARGET_LEAGUE})
            else f"{len(target_league_ids)}leagues"
        )
        out_path = _FIGURES_DIR / f"model_evaluation_{leagues_label}.png"

    if len(target_league_ids) <= 8:
        league_str = ", ".join(str(x) for x in sorted(target_league_ids))
    else:
        league_str = f"{len(target_league_ids)} senior club domestic leagues"
    suptitle = (
        f"H/D/A model evaluation — {league_str}\n"
        f"train: {len(train):,} matches  ·  test: {len(test):,} matches"
    )
    plot_evaluation(results, out_path, n_boot=n_boot, suptitle=suptitle)

    # Sidecar JSON with the same stem, for non-figure consumers (e.g. the
    # webapp's model dashboard) to read the metrics without re-running the
    # whole pipeline. Contains per-model point estimates + bootstrap CIs.
    _write_sidecar(
        out_path, results, league_ids=target_league_ids,
        n_train=len(train), n_test=len(test),
        league_str=league_str, n_boot=n_boot,
    )
    return out_path


def _write_sidecar(
    out_path: Path,
    results: list[ModelResult],
    league_ids: frozenset[int],
    n_train: int,
    n_test: int,
    league_str: str,
    n_boot: int,
) -> None:
    """Write a `<figure_stem>.json` next to the PNG with metric values.

    The webapp's model dashboard reads this to populate its KPI cards
    without having to recompute Elo / fit models on every page view.
    """
    import json
    from datetime import datetime, timezone

    payload: dict = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "scope_label": league_str,
        "league_ids": sorted(int(x) for x in league_ids),
        "n_train": int(n_train),
        "n_test": int(n_test),
        "n_boot": int(n_boot),
        "models": {},
    }
    for r in results:
        ll = r.metric(log_loss, n_boot=n_boot)
        br = r.metric(brier_score, n_boot=n_boot)
        from src.model.evaluate import accuracy as _acc
        ac = r.metric(_acc, n_boot=n_boot)
        ece = expected_calibration_error(r.y_true, r.probs)
        tb  = tail_brier(r.y_true, r.probs, threshold=0.10)
        payload["models"][r.name] = {
            "log_loss":   ll["point"],
            "log_loss_ci": [ll["lo"], ll["hi"]],
            "brier":      br["point"],
            "brier_ci":   [br["lo"], br["hi"]],
            "accuracy":   ac["point"],
            "ece":        float(ece),
            "tail_brier": float(tb["tail_brier"]) if tb else None,
            "tail_n_cells": int(tb["n_cells"]) if tb else 0,
        }

    json_path = out_path.with_suffix(".json")
    json_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def main() -> None:
    import argparse, logging, sys

    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(levelname)s %(message)s",
                        stream=sys.stdout)
    parser = argparse.ArgumentParser(description="Generate model evaluation figure")
    parser.add_argument(
        "--target-leagues", type=str, default="top5",
        help="Comma-separated league IDs, or 'top5' for the top European "
             "first tiers (default: top5).",
    )
    parser.add_argument("--boot", type=int, default=1000)
    args = parser.parse_args()

    from src.model.data import (
        DEFAULT_TARGET_LEAGUE, TOP5_EUROPEAN, all_domestic_club_leagues,
    )

    target = args.target_leagues.strip().lower()
    if target == "top5":
        league_ids = TOP5_EUROPEAN
    elif target == "all":
        league_ids = all_domestic_club_leagues()
    else:
        league_ids = frozenset(int(s.strip()) for s in args.target_leagues.split(","))

    out = run_and_plot(target_league_ids=league_ids, n_boot=args.boot)
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
