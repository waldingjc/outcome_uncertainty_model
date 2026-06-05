"""Outcome-uncertainty analysis: refines the entropy view from `plots.py`
using Elo ratings to control for team-strength gaps.

Generates `data/figures/uncertainty.png` (2x2 panel):
  1. Upset rate by league — % of matches where the Elo favourite lost,
     conditional on a meaningful rating gap.
  2. Win-prob vs Elo gap — calibration: how often does the favourite actually
     win, bucketed by rating gap, with the theoretical Elo curve overlaid.
  3. Time-of-season form arc — home / draw / away rates by quintile of
     season elapsed, top leagues.
  4. Predictability evolution — Shannon entropy of H/D/A by quintile of
     season elapsed, top leagues.

Usage:
    python -m src.analysis.uncertainty
"""

from __future__ import annotations

import logging
import math
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.analysis._style import apply_dark_style
from src.analysis.strength import (
    compute_elo_ratings, load_fixtures, DEFAULT_HOME_ADVANTAGE,
)

logger = logging.getLogger(__name__)

_FIGURES_DIR = Path(__file__).parents[2] / "data" / "figures"

# An "upset" must clear this much Elo gap before we count it — otherwise the
# match is too close to call and "favourite lost" isn't really an upset.
DEFAULT_UPSET_GAP = 100.0
TOP_N_LEAGUES = 12
N_QUINTILES = 10  # actually deciles; named for historic reasons


def _add_season_quantile(df: pd.DataFrame, n_bins: int = N_QUINTILES) -> pd.DataFrame:
    """For each (league, season) bin matches into n_bins equal-time slices."""
    df = df.copy()
    df["season_q"] = np.nan
    for (lid, season), g in df.groupby(["league_id", "season"]):
        if len(g) < 2:
            continue
        t = (g["date"] - g["date"].min()) / (g["date"].max() - g["date"].min())
        # Bin into 0..n_bins-1
        df.loc[g.index, "season_q"] = np.minimum(
            (t * n_bins).astype(int), n_bins - 1
        )
    df["season_q"] = df["season_q"].astype("Int64")
    return df


def _favourite_outcome(df_elo: pd.DataFrame) -> pd.DataFrame:
    """Annotate each match with favourite/underdog status and outcome."""
    df = df_elo.copy()
    df["effective_home_elo"] = df["home_pre_elo"] + DEFAULT_HOME_ADVANTAGE
    df["elo_gap"] = (df["effective_home_elo"] - df["away_pre_elo"]).abs()
    df["home_is_fav"] = df["effective_home_elo"] >= df["away_pre_elo"]
    df["fav_won"] = (
        ((df["home_is_fav"]) & (df["result"] == "H"))
        | ((~df["home_is_fav"]) & (df["result"] == "A"))
    )
    df["fav_drew"] = df["result"] == "D"
    df["fav_lost"] = ~df["fav_won"] & ~df["fav_drew"]
    # Theoretical Elo expectation for the favourite winning (treats draw as
    # half-credit, which is the canonical Elo formulation for football).
    df["fav_expected"] = 1.0 / (1.0 + 10.0 ** (-df["elo_gap"] / 400.0))
    return df


def _top_leagues(df: pd.DataFrame, n: int = TOP_N_LEAGUES) -> list[str]:
    return (
        df.groupby("league_name").size()
        .sort_values(ascending=False)
        .head(n)
        .index.tolist()
    )


# ---------------- Plots ----------------

def _plot_upset_rate(ax, df, gap=DEFAULT_UPSET_GAP):
    """Per league, % of matches with gap >= threshold where favourite lost."""
    eligible = df[df["elo_gap"] >= gap].copy()
    by_lg = eligible.groupby("league_name").agg(
        n=("fav_lost", "size"),
        upsets=("fav_lost", "sum"),
    )
    by_lg["upset_rate"] = by_lg["upsets"] / by_lg["n"]
    # Filter to leagues with enough mismatched matches.
    by_lg = by_lg[by_lg["n"] >= 50].sort_values("upset_rate", ascending=True)
    by_lg = by_lg.tail(TOP_N_LEAGUES)

    y = np.arange(len(by_lg))
    bars = ax.barh(y, by_lg["upset_rate"] * 100, color="#9D4EDD")
    ax.set_yticks(y, by_lg.index)
    ax.set_xlabel(f"% of matches with Elo gap ≥ {gap:.0f} where favourite LOST")
    ax.set_title(f"Upset rate by league (gap ≥ {gap:.0f} Elo)")
    for bar, n, rate in zip(bars, by_lg["n"], by_lg["upset_rate"]):
        ax.text(rate * 100 + 0.3, bar.get_y() + bar.get_height() / 2,
                f"{rate*100:.1f}% (n={n})", va="center", fontsize=8)
    ax.grid(axis="x", alpha=0.3)


def _plot_winprob_calibration(ax, df, top_leagues):
    """Win-prob vs Elo gap, top leagues, with theoretical Elo curve."""
    bins = np.array([0, 25, 50, 75, 100, 150, 200, 300, 400, 600, 1500])
    centres = (bins[:-1] + bins[1:]) / 2

    cmap = plt.get_cmap("tab10")
    for i, lg in enumerate(top_leagues[:6]):
        sub = df[df["league_name"] == lg]
        if len(sub) < 200:
            continue
        idx = np.digitize(sub["elo_gap"].values, bins) - 1
        idx = np.clip(idx, 0, len(centres) - 1)
        # P(favourite gets the result) — wins counted full, draws as 0.5.
        outcome_credit = (
            sub["fav_won"].astype(float).values
            + 0.5 * sub["fav_drew"].astype(float).values
        )
        df_b = pd.DataFrame({"bin": idx, "credit": outcome_credit})
        agg = df_b.groupby("bin").agg(["mean", "size"])
        x = centres[agg.index]
        y = agg[("credit", "mean")].values
        ax.plot(x, y, marker="o", color=cmap(i), label=lg, linewidth=1.5, markersize=5)

    # Theoretical Elo curve.
    elo_x = np.linspace(0, 600, 100)
    elo_y = 1.0 / (1.0 + 10.0 ** (-elo_x / 400.0))
    ax.plot(elo_x, elo_y, color="black", linestyle="--", alpha=0.7,
            label="Elo theoretical")

    ax.set_xlabel("Elo gap (favourite − underdog)")
    ax.set_ylabel("P(favourite gets ≥ draw, with W=1, D=0.5)")
    ax.set_title("Calibration: actual vs theoretical Elo")
    ax.set_xlim(0, 600)
    ax.set_ylim(0.4, 1.0)
    ax.legend(fontsize=8, loc="lower right")
    ax.grid(alpha=0.3)


def _plot_form_arc(ax, df, top_leagues):
    """Home win % by quintile of season elapsed, top leagues."""
    df = df.dropna(subset=["season_q"])
    cmap = plt.get_cmap("tab10")
    for i, lg in enumerate(top_leagues[:6]):
        sub = df[df["league_name"] == lg]
        if len(sub) < 100:
            continue
        agg = sub.groupby("season_q")["result"].apply(
            lambda s: (s == "H").mean()
        )
        ax.plot(agg.index.astype(int), agg.values * 100,
                marker="o", color=cmap(i), label=lg, linewidth=1.5, markersize=5)

    ax.set_xlabel(f"Quantile of season elapsed (0 = start, {N_QUINTILES - 1} = end)")
    ax.set_ylabel("Home win rate (%)")
    ax.set_title("Home advantage across the season")
    ax.legend(fontsize=8, loc="best")
    ax.grid(alpha=0.3)


def _plot_predictability_evolution(ax, df, top_leagues):
    """Shannon entropy of H/D/A by quintile of season, top leagues."""
    df = df.dropna(subset=["season_q"])
    cmap = plt.get_cmap("tab10")
    for i, lg in enumerate(top_leagues[:6]):
        sub = df[df["league_name"] == lg]
        if len(sub) < 100:
            continue
        ents = []
        bins = sorted(sub["season_q"].dropna().unique())
        for b in bins:
            slice_ = sub[sub["season_q"] == b]
            counts = slice_["result"].value_counts(normalize=True)
            p = counts.reindex(["H", "D", "A"], fill_value=0).values
            p = p[p > 0]
            ents.append(-(p * np.log2(p)).sum())
        ax.plot(bins, ents, marker="o", color=cmap(i),
                label=lg, linewidth=1.5, markersize=5)

    ax.axhline(math.log2(3), color="grey", linestyle="--", alpha=0.5,
               label=f"max entropy ({math.log2(3):.3f})")
    ax.set_xlabel(f"Quantile of season elapsed (0 = start, {N_QUINTILES - 1} = end)")
    ax.set_ylabel("Shannon entropy of H/D/A (bits)")
    ax.set_title("Predictability across the season\n(higher = harder to predict)")
    ax.legend(fontsize=7, loc="best")
    ax.grid(alpha=0.3)


def main():
    import argparse

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    parser = argparse.ArgumentParser(description="Outcome-uncertainty figures")
    parser.add_argument(
        "--seeding", choices=["hardcoded", "league_elo", "uniform"],
        default="hardcoded",
        help="Elo seeding mode (passed through to compute_elo_ratings).",
    )
    parser.add_argument("--out", default=None, help="Output PNG path")
    args = parser.parse_args()

    df = load_fixtures()
    df_elo, _ = compute_elo_ratings(df, seeding=args.seeding)
    df_elo = _favourite_outcome(df_elo)
    df_elo = _add_season_quantile(df_elo)
    top = _top_leagues(df_elo)
    logger.info("Top leagues: %s", top[:6])

    apply_dark_style()
    fig, axes = plt.subplots(2, 2, figsize=(20, 14), constrained_layout=True)
    fig.suptitle(
        f"Outcome uncertainty (Elo-aware) — {len(df_elo):,} matches "
        f"(seeding={args.seeding})",
        fontsize=15, fontweight="bold",
    )
    _plot_upset_rate(axes[0, 0], df_elo)
    _plot_winprob_calibration(axes[0, 1], df_elo, top)
    _plot_form_arc(axes[1, 0], df_elo, top)
    _plot_predictability_evolution(axes[1, 1], df_elo, top)

    if args.out:
        out = Path(args.out)
    else:
        out = _FIGURES_DIR / f"uncertainty_{args.seeding}.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=140, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
