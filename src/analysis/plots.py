"""Quick exploratory plots over the ingested fixtures.

Generates a 2x2 overview figure saved to `data/figures/overview.png`:

  1. Outcome uncertainty by league (Shannon entropy of the H/D/A
     distribution — the headline metric for this project).
  2. Home / Draw / Away breakdown per league (stacked horizontal bars).
  3. Total goals per match distribution (boxplot per league).
  4. HT-leader flip rate — how often a team leading at half-time fails to
     win the match (a measure of "second-half drama").

Usage:
    python -m src.analysis.plots
"""

from __future__ import annotations

import logging
import math
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.db.schema import get_connection

logger = logging.getLogger(__name__)

_FIGURES_DIR = Path(__file__).parents[2] / "data" / "figures"

# Cap most plots at the top-N leagues by fixture count for readability.
TOP_N_LEAGUES = 15
GOALS_TOP_N = 10


def load_fixtures() -> pd.DataFrame:
    sql = """
        SELECT fixture_id, date, league_id, league_name, season,
               home_goals, away_goals, home_goals_ht, away_goals_ht
        FROM fixtures
        WHERE status = 'FT'
          AND home_goals IS NOT NULL
          AND away_goals IS NOT NULL
    """
    with get_connection() as conn:
        df = pd.read_sql(sql, conn)

    df["result"] = np.where(
        df["home_goals"] > df["away_goals"], "H",
        np.where(df["home_goals"] < df["away_goals"], "A", "D"),
    )
    df["total_goals"] = df["home_goals"] + df["away_goals"]

    ht_known = df["home_goals_ht"].notna() & df["away_goals_ht"].notna()
    df["ht_result"] = np.where(
        ht_known & (df["home_goals_ht"] > df["away_goals_ht"]), "H",
        np.where(ht_known & (df["home_goals_ht"] < df["away_goals_ht"]), "A",
        np.where(ht_known, "D", None)),
    )
    return df


def _entropy_bits(probs: np.ndarray) -> float:
    probs = probs[probs > 0]
    return float(-np.sum(probs * np.log2(probs)))


def _league_summary(df: pd.DataFrame) -> pd.DataFrame:
    """One row per league with H/D/A rates, fixture count, entropy, mean goals."""
    g = df.groupby(["league_id", "league_name"])
    total = g.size().rename("n")
    hda = (
        df.assign(one=1)
        .pivot_table(
            index=["league_id", "league_name"],
            columns="result", values="one", aggfunc="sum", fill_value=0,
        )
        .reindex(columns=["H", "D", "A"], fill_value=0)
    )
    hda_rate = hda.div(hda.sum(axis=1), axis=0)
    hda_rate.columns = ["home_rate", "draw_rate", "away_rate"]
    summary = pd.concat([total, hda_rate], axis=1).reset_index()
    summary["entropy_bits"] = summary[
        ["home_rate", "draw_rate", "away_rate"]
    ].apply(lambda row: _entropy_bits(row.values), axis=1)
    summary["mean_total_goals"] = g["total_goals"].mean().values
    return summary


def _ht_flip_rate(df: pd.DataFrame) -> pd.DataFrame:
    """% of HT-leading matches where the leading team failed to WIN at FT."""
    led = df[df["ht_result"].isin(["H", "A"])].copy()
    led["flipped"] = led["ht_result"] != led["result"]
    g = led.groupby(["league_id", "league_name"])
    out = pd.DataFrame({
        "n_led_at_ht": g.size(),
        "flip_rate": g["flipped"].mean(),
    }).reset_index()
    return out


def plot_overview(df: pd.DataFrame, out_path: Path) -> None:
    summary = _league_summary(df)
    flip = _ht_flip_rate(df)

    top = summary.nlargest(TOP_N_LEAGUES, "n").copy()
    top_goals = summary.nlargest(GOALS_TOP_N, "n").copy()
    top_flip = (
        flip.merge(summary[["league_id", "n"]], on="league_id")
            .nlargest(TOP_N_LEAGUES, "n")
            .copy()
    )

    fig, axes = plt.subplots(2, 2, figsize=(18, 12), constrained_layout=True)
    fig.suptitle(
        f"Fixture overview — {len(df):,} fixtures across "
        f"{summary['league_id'].nunique()} leagues, seasons 2022-2024",
        fontsize=15, fontweight="bold",
    )

    # ---- (1) Outcome uncertainty (Shannon entropy) -----------------------
    ax = axes[0, 0]
    s = top.sort_values("entropy_bits", ascending=True)
    bars = ax.barh(s["league_name"], s["entropy_bits"], color="#3C6E71")
    ax.axvline(math.log2(3), color="#E63946", linestyle="--", linewidth=1.2,
               label=f"max entropy (1/3, 1/3, 1/3) = {math.log2(3):.3f} bits")
    ax.set_xlabel("Shannon entropy of (Home, Draw, Away) outcomes (bits)")
    ax.set_title("Outcome uncertainty by league\n(higher = harder to predict from outcome alone)")
    ax.set_xlim(1.0, math.log2(3) * 1.02)
    ax.legend(loc="lower right", fontsize=9)
    for bar, val in zip(bars, s["entropy_bits"]):
        ax.text(val + 0.005, bar.get_y() + bar.get_height() / 2,
                f"{val:.3f}", va="center", fontsize=8)

    # ---- (2) Home / Draw / Away stacked bars -----------------------------
    ax = axes[0, 1]
    s = top.sort_values("home_rate", ascending=True)
    ax.barh(s["league_name"], s["home_rate"], label="Home win",
            color="#264653")
    ax.barh(s["league_name"], s["draw_rate"], left=s["home_rate"],
            label="Draw", color="#F4A261")
    ax.barh(s["league_name"], s["away_rate"],
            left=s["home_rate"] + s["draw_rate"],
            label="Away win", color="#A8DADC")
    ax.set_xlim(0, 1)
    ax.set_xlabel("Share of matches")
    ax.set_title("Home / Draw / Away split by league\n(sorted by home-win rate)")
    ax.legend(loc="lower right", fontsize=9)
    for i, (_, row) in enumerate(s.iterrows()):
        ax.text(row["home_rate"] / 2, i, f"{row['home_rate']*100:.0f}%",
                ha="center", va="center", fontsize=8, color="white")

    # ---- (3) Goals per match distribution --------------------------------
    ax = axes[1, 0]
    leagues_in_order = top_goals.sort_values("mean_total_goals", ascending=False)["league_id"]
    data = [
        df.loc[df["league_id"] == lid, "total_goals"].values
        for lid in leagues_in_order
    ]
    labels = [
        top_goals.set_index("league_id").loc[lid, "league_name"]
        for lid in leagues_in_order
    ]
    bp = ax.boxplot(
        data, labels=labels, vert=False, showfliers=False,
        patch_artist=True, medianprops={"color": "black"},
    )
    for patch in bp["boxes"]:
        patch.set_facecolor("#E9C46A")
    ax.set_xlabel("Total goals per match (home + away)")
    ax.set_title("Goals per match by league\n(box: IQR, whiskers: 1.5×IQR, outliers hidden)")
    for i, lid in enumerate(leagues_in_order, start=1):
        mean = top_goals.set_index("league_id").loc[lid, "mean_total_goals"]
        ax.text(mean, i, f"  μ={mean:.2f}", va="center", fontsize=8, color="#264653")

    # ---- (4) HT-leader flip rate -----------------------------------------
    ax = axes[1, 1]
    s = top_flip.sort_values("flip_rate", ascending=True)
    bars = ax.barh(s["league_name"], s["flip_rate"] * 100, color="#9D4EDD")
    ax.set_xlabel("% of HT-leading matches where the leading team didn't win FT")
    ax.set_title("Second-half drama by league\n(flip rate of HT-leading teams)")
    ax.set_xlim(0, max(s["flip_rate"]) * 110)
    for bar, val, n in zip(bars, s["flip_rate"], s["n_led_at_ht"]):
        ax.text(val * 100 + 0.3, bar.get_y() + bar.get_height() / 2,
                f"{val*100:.1f}% (n={n:,})", va="center", fontsize=8)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=140, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved overview figure to %s", out_path)


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    df = load_fixtures()
    logger.info("Loaded %d FT fixtures across %d leagues",
                len(df), df["league_id"].nunique())

    out = _FIGURES_DIR / "overview.png"
    plot_overview(df, out)
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
