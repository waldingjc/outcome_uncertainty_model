"""Multi-country pyramid analysis: tier-by-tier strength comparison.

Generates `data/figures/pyramid.png` — a small-multiples figure with one
panel per country, each panel showing the team-Elo distribution at each
tier of that country's pyramid (top tier at top of panel). Visualises the
strength gradient from elite to amateur across countries that have
multi-tier coverage in our dataset.

A second figure `data/figures/pyramid_gaps.png` summarises across-country
metrics: how steep the drop-off is from each tier to the next, mean goals
per match by tier, home-win rate by tier.

Usage:
    python -m src.analysis.pyramid
"""

from __future__ import annotations

import logging
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.analysis._style import apply_dark_style
from src.analysis.strength import (
    compute_elo_ratings, load_fixtures, primary_league_map,
)

logger = logging.getLogger(__name__)

_FIGURES_DIR = Path(__file__).parents[2] / "data" / "figures"

# Country → tier → [league_ids]. Tier 1 is the top flight; higher numbers go
# deeper down the pyramid. Where multiple regional leagues sit on the same
# tier (e.g. England's Non League Div One groups, Germany's Regionalliga),
# they're combined for the box-plot.
PYRAMID: dict[str, dict[int, list[int]]] = {
    "England": {
        1: [39],                  # Premier League
        2: [40],                  # Championship
        3: [41],                  # League One
        4: [42],                  # League Two
        5: [43],                  # National League
        6: [50, 51],              # National League N / S
        7: [58, 59, 60],          # Non League Premier (Isthmian / Northern / Southern South)
        8: [52, 53, 54, 55, 56, 57],  # Non League Division One (six regional groups)
    },
    "Germany": {
        1: [78],                  # Bundesliga
        2: [79],                  # 2. Bundesliga
        3: [80],                  # 3. Liga
        4: [83, 84, 85, 86, 87],  # Regionalliga (5 regional groups)
    },
    "Italy": {
        1: [135],                 # Serie A
        2: [72],                  # Serie B
        3: [75],                  # Serie C
        4: [76],                  # Serie D
    },
    "France": {
        1: [61],                  # Ligue 1
        2: [62],                  # Ligue 2
        3: [63],                  # National 1
        4: [67, 68, 69, 70],      # National 2 (4 groups)
    },
    "Spain": {
        1: [140],                 # La Liga
        2: [141],                 # Segunda División
    },
    "Netherlands": {
        1: [88],                  # Eredivisie
        2: [89],                  # Eerste Divisie
        3: [92, 93],              # Derde Divisie (Saturday / Sunday)
    },
    "Portugal": {
        1: [94],                  # Primeira Liga
        2: [95],                  # Segunda Liga
    },
    "Japan": {
        1: [98],                  # J1 League
        2: [99],                  # J2 League
        3: [100],                 # J3 League
    },
    "Norway": {
        1: [103],                 # Eliteserien
        2: [104],                 # 1. Division
    },
    "Poland": {
        1: [106],                 # Ekstraklasa
        2: [107],                 # I Liga
        3: [109],                 # II Liga
    },
    "Sweden": {
        1: [113],                 # Allsvenskan
        2: [114],                 # Superettan
    },
    "Wales": {
        1: [110],                 # Premier League (Wales)
        2: [111],                 # FAW Championship
    },
}


def _team_tiers(df: pd.DataFrame) -> dict[int, tuple[str, int]]:
    """Map team_id -> (country, tier) using each team's primary league.

    A team gets the country / tier of its most-played league across the
    dataset. Teams whose primary league isn't in PYRAMID are excluded.
    """
    pmap = primary_league_map(df)
    league_to_country_tier: dict[int, tuple[str, int]] = {}
    for country, tiers in PYRAMID.items():
        for tier, lids in tiers.items():
            for lid in lids:
                league_to_country_tier[lid] = (country, tier)

    out: dict[int, tuple[str, int]] = {}
    for tid, (lid, _) in pmap.items():
        if lid in league_to_country_tier:
            out[tid] = league_to_country_tier[lid]
    return out


def _team_metrics_by_tier(df: pd.DataFrame, ratings: dict[int, float]) -> pd.DataFrame:
    """Long-format: one row per team with country + tier + Elo + perf metrics."""
    team_ct = _team_tiers(df)

    # Per-team aggregate stats across all their fixtures.
    home = df[["home_team_id", "home_goals", "away_goals", "result"]].rename(
        columns={"home_team_id": "team_id"}
    )
    home["scored"] = home["home_goals"]
    home["conceded"] = home["away_goals"]
    home["won"] = home["result"] == "H"
    away = df[["away_team_id", "home_goals", "away_goals", "result"]].rename(
        columns={"away_team_id": "team_id"}
    )
    away["scored"] = away["away_goals"]
    away["conceded"] = away["home_goals"]
    away["won"] = away["result"] == "A"
    stacked = pd.concat(
        [home[["team_id", "scored", "conceded", "won"]],
         away[["team_id", "scored", "conceded", "won"]]],
        ignore_index=True,
    )
    perf = stacked.groupby("team_id").agg(
        n=("won", "size"),
        wins=("won", "sum"),
        gf=("scored", "mean"),
        ga=("conceded", "mean"),
    ).reset_index()
    perf["win_rate"] = perf["wins"] / perf["n"]

    rows = []
    for _, r in perf.iterrows():
        tid = int(r["team_id"])
        if tid not in team_ct:
            continue
        country, tier = team_ct[tid]
        rows.append({
            "team_id": tid,
            "country": country,
            "tier": tier,
            "elo": ratings.get(tid, np.nan),
            "win_rate": r["win_rate"],
            "gf": r["gf"],
            "ga": r["ga"],
            "n": int(r["n"]),
        })
    return pd.DataFrame(rows)


def _league_tier_metrics(df: pd.DataFrame) -> pd.DataFrame:
    """For each (country, tier), compute mean goals and home-win rate from
    the matches played in that tier's leagues. Useful for the across-tier
    line plots.
    """
    league_to_ct: dict[int, tuple[str, int]] = {}
    for country, tiers in PYRAMID.items():
        for tier, lids in tiers.items():
            for lid in lids:
                league_to_ct[lid] = (country, tier)

    df = df.copy()
    df["country_tier"] = df["league_id"].map(league_to_ct)
    df = df[df["country_tier"].notna()].copy()
    df[["country", "tier"]] = pd.DataFrame(
        df["country_tier"].tolist(), index=df.index
    )

    df["total_goals"] = df["home_goals"] + df["away_goals"]
    df["home_win"] = (df["result"] == "H").astype(int)
    df["draw"] = (df["result"] == "D").astype(int)

    g = df.groupby(["country", "tier"]).agg(
        n=("fixture_id", "size"),
        mean_goals=("total_goals", "mean"),
        home_win_rate=("home_win", "mean"),
        draw_rate=("draw", "mean"),
    ).reset_index()
    return g


# ---------------- Plotting ----------------

def plot_pyramid_silhouettes(team_df: pd.DataFrame, out_path: Path) -> None:
    """Per-country small multiples: tier-by-tier Elo box plots.

    Layout: 4 columns x 3 rows = 12 panels. Each panel shows that country's
    tiers stacked top-to-bottom (tier 1 at top), horizontal box per tier.
    """
    apply_dark_style()
    countries = list(PYRAMID.keys())
    n_cols = 4
    n_rows = (len(countries) + n_cols - 1) // n_cols

    fig, axes = plt.subplots(
        n_rows, n_cols, figsize=(n_cols * 5, n_rows * 4),
        sharex=True, constrained_layout=True,
    )
    axes = axes.flatten()

    # Common Elo range for visual comparison across countries.
    elo_min = team_df["elo"].quantile(0.01) - 30
    elo_max = team_df["elo"].quantile(0.99) + 30

    cmap = plt.colormaps.get_cmap("viridis")

    for i, country in enumerate(countries):
        ax = axes[i]
        country_df = team_df[team_df["country"] == country]
        max_tier = max(PYRAMID[country].keys())
        tiers = sorted(PYRAMID[country].keys())  # ascending: 1, 2, 3...

        data = []
        labels = []
        n_per_tier = []
        for t in tiers:
            sub = country_df[country_df["tier"] == t]["elo"].dropna().values
            data.append(sub)
            labels.append(f"Tier {t}")
            n_per_tier.append(len(sub))

        # Plot top tier at TOP of the panel.
        positions = np.arange(len(tiers), 0, -1)  # reversed
        bp = ax.boxplot(
            data, positions=positions, vert=False,
            widths=0.6, patch_artist=True,
            showfliers=False, medianprops={"color": "black"},
        )
        # Colour by tier (deeper = darker).
        for j, patch in enumerate(bp["boxes"]):
            colour = cmap((tiers[j] - 1) / max(max_tier, 1))
            patch.set_facecolor(colour)
            patch.set_alpha(0.85)

        ax.set_yticks(positions, labels)
        ax.set_xlim(elo_min, elo_max)
        ax.axvline(1500, color="grey", linestyle="--", alpha=0.4, linewidth=0.8)
        total_teams = sum(n_per_tier)
        ax.set_title(f"{country}  ({total_teams} teams across {len(tiers)} tiers)",
                     fontsize=11, fontweight="bold")
        # Annotate sample sizes at right edge.
        for pos, n in zip(positions, n_per_tier):
            ax.text(elo_max - 5, pos, f" n={n}",
                    va="center", ha="right", fontsize=7, color="dimgrey")
        ax.grid(axis="x", alpha=0.3)

    # Hide any leftover blank panels.
    for j in range(len(countries), len(axes)):
        axes[j].set_visible(False)

    fig.suptitle(
        "Country pyramids — team Elo distribution by tier "
        f"({len(team_df):,} teams across {team_df['country'].nunique()} countries)",
        fontsize=14, fontweight="bold",
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=140, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved %s", out_path)


def plot_pyramid_metrics(team_df: pd.DataFrame, league_df: pd.DataFrame, out_path: Path) -> None:
    """2x2: tier strength gap, mean goals per tier, home-win % per tier,
    draw rate per tier. Cross-country comparison.
    """
    apply_dark_style()
    fig, axes = plt.subplots(2, 2, figsize=(18, 12), constrained_layout=True)
    fig.suptitle(
        "Cross-country pyramid metrics — how steep is the drop, "
        "and how do play patterns shift down the pyramid?",
        fontsize=14, fontweight="bold",
    )

    # ---- (0,0) Tier 1 -> Tier 2 Elo gap -----------------------------
    ax = axes[0, 0]
    rows = []
    for country in PYRAMID:
        sub = team_df[team_df["country"] == country]
        t1 = sub[sub["tier"] == 1]["elo"]
        t2 = sub[sub["tier"] == 2]["elo"]
        if len(t1) >= 3 and len(t2) >= 3:
            rows.append({
                "country": country,
                "t1_median": t1.median(),
                "t2_median": t2.median(),
                "gap": t1.median() - t2.median(),
                "n_t1": len(t1),
                "n_t2": len(t2),
            })
    gap_df = pd.DataFrame(rows).sort_values("gap", ascending=True)
    y = np.arange(len(gap_df))
    ax.barh(y, gap_df["gap"], color="#264653")
    ax.set_yticks(y, gap_df["country"])
    ax.set_xlabel("Elo gap (Tier 1 median − Tier 2 median)")
    ax.set_title("Tier 1 → Tier 2 strength gap")
    for i, (_, row) in enumerate(gap_df.iterrows()):
        ax.text(row["gap"] + 4, i,
                f"{row['gap']:+.0f}  ({row['n_t1']} vs {row['n_t2']} teams)",
                va="center", fontsize=8)
    ax.grid(axis="x", alpha=0.3)

    # ---- (0,1) Mean goals per match across tiers ----------------------
    ax = axes[0, 1]
    cmap = plt.colormaps.get_cmap("tab10")
    countries_sorted = list(gap_df["country"]) + [c for c in PYRAMID if c not in set(gap_df["country"])]
    for i, country in enumerate(countries_sorted[:8]):  # top 8 to avoid clutter
        sub = league_df[league_df["country"] == country].sort_values("tier")
        if len(sub) < 2:
            continue
        ax.plot(sub["tier"], sub["mean_goals"], marker="o",
                color=cmap(i % 10), label=country, linewidth=1.6, markersize=6)
    ax.set_xlabel("Tier (1 = top flight)")
    ax.set_ylabel("Mean goals per match")
    ax.set_title("Goals per match by tier")
    ax.legend(fontsize=8, loc="best")
    ax.grid(alpha=0.3)
    ax.invert_xaxis()  # top tier on right? actually: tier 1 on left is fine, leave as is
    ax.invert_xaxis()

    # ---- (1,0) Home-win rate by tier ---------------------------------
    ax = axes[1, 0]
    for i, country in enumerate(countries_sorted[:8]):
        sub = league_df[league_df["country"] == country].sort_values("tier")
        if len(sub) < 2:
            continue
        ax.plot(sub["tier"], sub["home_win_rate"] * 100, marker="o",
                color=cmap(i % 10), label=country, linewidth=1.6, markersize=6)
    ax.set_xlabel("Tier")
    ax.set_ylabel("Home win rate (%)")
    ax.set_title("Home advantage by tier")
    ax.legend(fontsize=8, loc="best")
    ax.grid(alpha=0.3)

    # ---- (1,1) Draw rate by tier --------------------------------------
    ax = axes[1, 1]
    for i, country in enumerate(countries_sorted[:8]):
        sub = league_df[league_df["country"] == country].sort_values("tier")
        if len(sub) < 2:
            continue
        ax.plot(sub["tier"], sub["draw_rate"] * 100, marker="o",
                color=cmap(i % 10), label=country, linewidth=1.6, markersize=6)
    ax.set_xlabel("Tier")
    ax.set_ylabel("Draw rate (%)")
    ax.set_title("Draw rate by tier")
    ax.legend(fontsize=8, loc="best")
    ax.grid(alpha=0.3)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=140, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved %s", out_path)


def main():
    import sys

    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s",
        stream=sys.stdout,
    )
    df = load_fixtures()
    logger.info("Loaded %d fixtures", len(df))

    df_elo, ratings = compute_elo_ratings(df)
    logger.info("Computed Elo for %d teams", len(ratings))

    team_df = _team_metrics_by_tier(df, ratings)
    logger.info(
        "Pyramid teams: %d teams across %d countries",
        len(team_df), team_df["country"].nunique(),
    )
    league_df = _league_tier_metrics(df)
    logger.info("Pyramid match aggregates: %d (country, tier) rows", len(league_df))

    plot_pyramid_silhouettes(team_df, _FIGURES_DIR / "pyramid.png")
    plot_pyramid_metrics(team_df, league_df, _FIGURES_DIR / "pyramid_gaps.png")
    print(f"Wrote {_FIGURES_DIR / 'pyramid.png'}")
    print(f"Wrote {_FIGURES_DIR / 'pyramid_gaps.png'}")


if __name__ == "__main__":
    main()
