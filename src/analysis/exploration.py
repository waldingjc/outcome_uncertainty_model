"""Exploratory / records figure across the full dataset.

Generates `data/figures/exploration.png` (3x2 panel):
  1. Most-fixtured teams across the full dataset.
  2. Cross-competition win-rate matrix for top multi-competition teams.
  3. Highest-scoring single matches.
  4. Biggest Elo upsets — biggest rating-gap matches the underdog won.
  5. Most-played head-to-head pairs.
  6. Longest unbeaten runs (W or D, by team, anywhere in the data).

Usage:
    python -m src.analysis.exploration
"""

from __future__ import annotations

import logging
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.analysis.strength import (
    compute_elo_ratings, load_fixtures, primary_league_map, team_name_map,
    DEFAULT_HOME_ADVANTAGE,
)

logger = logging.getLogger(__name__)

_FIGURES_DIR = Path(__file__).parents[2] / "data" / "figures"


def _team_match_counts(df: pd.DataFrame) -> pd.DataFrame:
    """Per (team, competition) count of matches; one row per (team, league)."""
    home = df[["home_team_id", "home_team_name", "league_id", "league_name"]].rename(
        columns={"home_team_id": "team_id", "home_team_name": "team_name"}
    )
    away = df[["away_team_id", "away_team_name", "league_id", "league_name"]].rename(
        columns={"away_team_id": "team_id", "away_team_name": "team_name"}
    )
    stacked = pd.concat([home, away], ignore_index=True)
    return (
        stacked.groupby(["team_id", "team_name", "league_id", "league_name"])
        .size().rename("n").reset_index()
    )


def _team_winrates(df: pd.DataFrame) -> pd.DataFrame:
    """Per (team, league) win rate."""
    home = df.assign(
        team_id=df["home_team_id"], team_name=df["home_team_name"],
        won=(df["result"] == "H").astype(int),
    )[["team_id", "team_name", "league_id", "league_name", "won"]]
    away = df.assign(
        team_id=df["away_team_id"], team_name=df["away_team_name"],
        won=(df["result"] == "A").astype(int),
    )[["team_id", "team_name", "league_id", "league_name", "won"]]
    stacked = pd.concat([home, away], ignore_index=True)
    return (
        stacked.groupby(["team_id", "team_name", "league_id", "league_name"])
        .agg(n=("won", "size"), wins=("won", "sum"))
        .reset_index()
        .assign(win_rate=lambda d: d["wins"] / d["n"])
    )


def _longest_unbeaten_runs(df: pd.DataFrame, top_n: int = 12) -> pd.DataFrame:
    """For each team, find the longest streak of consecutive non-loss matches."""
    home = df[["home_team_id", "home_team_name", "date", "result"]].copy()
    home["team_id"] = home["home_team_id"]
    home["team_name"] = home["home_team_name"]
    home["non_loss"] = home["result"].isin(["H", "D"])
    away = df[["away_team_id", "away_team_name", "date", "result"]].copy()
    away["team_id"] = away["away_team_id"]
    away["team_name"] = away["away_team_name"]
    away["non_loss"] = away["result"].isin(["A", "D"])
    stacked = pd.concat(
        [home[["team_id", "team_name", "date", "non_loss"]],
         away[["team_id", "team_name", "date", "non_loss"]]],
        ignore_index=True,
    ).sort_values(["team_id", "date"])

    rows = []
    for (tid, tname), g in stacked.groupby(["team_id", "team_name"]):
        # Streak length: cumulative count that resets on each False.
        nl = g["non_loss"].values
        if not nl.any():
            continue
        # Compute run lengths via simple loop (fast enough at this size).
        best = cur = 0
        for v in nl:
            cur = cur + 1 if v else 0
            if cur > best:
                best = cur
        if best >= 5:
            rows.append({"team_id": tid, "team_name": tname, "longest": best})
    return pd.DataFrame(rows).nlargest(top_n, "longest").reset_index(drop=True)


def _biggest_upsets(df_elo: pd.DataFrame, top_n: int = 10) -> pd.DataFrame:
    """Matches with the biggest Elo gap where the underdog won."""
    df = df_elo.copy()
    df["effective_home_elo"] = df["home_pre_elo"] + DEFAULT_HOME_ADVANTAGE
    df["home_is_fav"] = df["effective_home_elo"] >= df["away_pre_elo"]
    df["fav_won"] = (
        ((df["home_is_fav"]) & (df["result"] == "H"))
        | ((~df["home_is_fav"]) & (df["result"] == "A"))
    )
    df["upset"] = (~df["fav_won"]) & (df["result"] != "D")
    df["gap"] = (df["effective_home_elo"] - df["away_pre_elo"]).abs()
    upsets = df[df["upset"]].nlargest(top_n, "gap")
    return upsets


def _highest_scoring(df: pd.DataFrame, top_n: int = 10) -> pd.DataFrame:
    df = df.copy()
    df["total_goals"] = df["home_goals"] + df["away_goals"]
    return df.nlargest(top_n, "total_goals")


def _most_played_h2h(df: pd.DataFrame, top_n: int = 12) -> pd.DataFrame:
    """Most-played team-pair head-to-heads (canonical pair ordering)."""
    df = df.copy()
    df["pair_a_id"] = np.minimum(df["home_team_id"], df["away_team_id"])
    df["pair_b_id"] = np.maximum(df["home_team_id"], df["away_team_id"])

    def _pair_name(row):
        a = row["pair_a_id"]
        if a == row["home_team_id"]:
            return row["home_team_name"], row["away_team_name"]
        return row["away_team_name"], row["home_team_name"]

    names = df.apply(_pair_name, axis=1, result_type="expand")
    df["pair_a_name"] = names[0]
    df["pair_b_name"] = names[1]

    counts = (
        df.groupby(["pair_a_id", "pair_b_id", "pair_a_name", "pair_b_name"])
        .size().rename("n").reset_index()
        .nlargest(top_n, "n")
    )
    return counts


# ---------------- Plots ----------------

def _plot_most_fixtured(ax, df, top_n=12):
    counts = _team_match_counts(df)
    totals = (
        counts.groupby(["team_id", "team_name"])["n"].sum()
        .nlargest(top_n).reset_index()
    )

    # Stack by competition.
    pivot = (
        counts[counts["team_id"].isin(totals["team_id"])]
        .pivot_table(index="team_id", columns="league_name", values="n", fill_value=0)
        .loc[totals.set_index("team_id").index]
    )
    pivot["__total__"] = pivot.sum(axis=1)
    pivot = pivot.sort_values("__total__", ascending=True).drop(columns="__total__")

    name_lookup = totals.set_index("team_id")["team_name"]
    cmap = plt.get_cmap("tab20")
    colours = [cmap(i % 20) for i in range(pivot.shape[1])]

    bottom = np.zeros(len(pivot))
    for col, c in zip(pivot.columns, colours):
        ax.barh(np.arange(len(pivot)), pivot[col], left=bottom,
                color=c, label=col)
        bottom += pivot[col].values

    ax.set_yticks(np.arange(len(pivot)),
                  [name_lookup.loc[tid] for tid in pivot.index])
    ax.set_xlabel("Total matches in dataset")
    ax.set_title(f"Top {top_n} most-fixtured teams (stacked by competition)")
    ax.legend(fontsize=6, loc="lower right", ncol=1)
    ax.grid(axis="x", alpha=0.3)


def _plot_cross_comp(
    ax, df, top_n_teams=10, top_n_comps=6,
    min_matches_in_top_comps=20, min_cell_matches=4,
):
    """Win-rate heatmap for club teams that play across many competitions.

    Filtering strategy: a team must have at least `min_matches_in_top_comps`
    matches in the heatmap's chosen columns (the busiest competitions).
    National teams are excluded automatically because they don't appear in
    domestic-club competitions like FA Cup / Championship.
    """
    counts = _team_match_counts(df)

    # 1. Pick top competitions by total fixtures (these are the heatmap columns).
    top_comps = (
        df.groupby(["league_id", "league_name"]).size()
        .sort_values(ascending=False).head(top_n_comps).reset_index()
    )

    # 2. Restrict counts to those competitions, then keep teams that have
    #    at least min_matches_in_top_comps total matches across them.
    counts_top = counts[counts["league_id"].isin(top_comps["league_id"])]
    matches_in_top = (
        counts_top.groupby(["team_id", "team_name"])["n"].sum()
    )
    qualified_teams = matches_in_top[matches_in_top >= min_matches_in_top_comps].index

    # 3. Of qualified teams, pick those with the most distinct competitions
    #    (now meaningfully a club-side multi-competition score).
    counts_q = counts_top[
        counts_top.set_index(["team_id", "team_name"]).index.isin(qualified_teams)
    ]
    n_comps = counts_q.groupby(["team_id", "team_name"])["league_id"].nunique().rename("ncomp")
    top_teams = n_comps.sort_values(ascending=False).head(top_n_teams)

    wr = _team_winrates(df)
    wr = wr[wr.set_index(["team_id", "team_name"]).index.isin(top_teams.index)]
    wr = wr[wr["league_id"].isin(top_comps["league_id"])]

    # 4. Hide cells with too-small sample.
    wr.loc[wr["n"] < min_cell_matches, "win_rate"] = np.nan

    pivot_rate = wr.pivot_table(index="team_name", columns="league_name",
                                values="win_rate", aggfunc="first")
    pivot_n = wr.pivot_table(index="team_name", columns="league_name",
                             values="n", aggfunc="first")

    team_order = [t for _, t in top_teams.index]
    comp_order = top_comps["league_name"].tolist()
    pivot_rate = pivot_rate.reindex(index=team_order, columns=comp_order)
    pivot_n = pivot_n.reindex(index=team_order, columns=comp_order)

    im = ax.imshow(pivot_rate.values, cmap="RdYlGn", aspect="auto", vmin=0, vmax=1)
    ax.set_xticks(np.arange(pivot_rate.shape[1]), pivot_rate.columns,
                  rotation=30, ha="right")
    ax.set_yticks(np.arange(pivot_rate.shape[0]), pivot_rate.index)
    ax.set_title(
        f"Win rate by competition (≥{min_matches_in_top_comps} matches in shown comps; "
        f"cells need ≥{min_cell_matches} match sample)"
    )
    plt.colorbar(im, ax=ax, label="win rate", shrink=0.8)
    for i in range(pivot_rate.shape[0]):
        for j in range(pivot_rate.shape[1]):
            v = pivot_rate.values[i, j]
            n = pivot_n.values[i, j]
            if np.isnan(v):
                continue
            ax.text(j, i, f"{v:.2f}\n(n={int(n)})", ha="center", va="center",
                    fontsize=7, color="black")


def _plot_high_scoring(ax, df, top_n=10):
    top = _highest_scoring(df, top_n).iloc[::-1].reset_index(drop=True)
    labels = [
        f"{r['home_team_name']} {int(r['home_goals'])}-{int(r['away_goals'])} {r['away_team_name']}"
        f"  ({r['league_name']}, {pd.to_datetime(r['date']).strftime('%Y-%m-%d')})"
        for _, r in top.iterrows()
    ]
    ax.barh(np.arange(len(top)), top["total_goals"], color="#F4A261")
    ax.set_yticks(np.arange(len(top)), labels)
    ax.set_xlabel("Total goals in match")
    ax.set_title(f"Top {top_n} highest-scoring matches")
    for i, v in enumerate(top["total_goals"]):
        ax.text(v + 0.1, i, f"{int(v)}", va="center", fontsize=8)
    ax.grid(axis="x", alpha=0.3)


def _plot_biggest_upsets(ax, df_elo, top_n=10):
    upsets = _biggest_upsets(df_elo, top_n).iloc[::-1].reset_index(drop=True)
    labels = [
        f"{r['home_team_name']} {int(r['home_goals'])}-{int(r['away_goals'])} {r['away_team_name']}"
        f"  ({r['league_name']}, {pd.to_datetime(r['date']).strftime('%Y-%m-%d')})"
        for _, r in upsets.iterrows()
    ]
    ax.barh(np.arange(len(upsets)), upsets["gap"], color="#9D4EDD")
    ax.set_yticks(np.arange(len(upsets)), labels)
    ax.set_xlabel("Elo gap at kickoff (favourite − underdog)")
    ax.set_title(f"Top {top_n} biggest Elo upsets")
    for i, v in enumerate(upsets["gap"]):
        ax.text(v + 4, i, f"{int(v)}", va="center", fontsize=8)
    ax.grid(axis="x", alpha=0.3)


def _plot_h2h(ax, df, top_n=12):
    pairs = _most_played_h2h(df, top_n).iloc[::-1].reset_index(drop=True)
    labels = [f"{r['pair_a_name']} vs {r['pair_b_name']}" for _, r in pairs.iterrows()]
    ax.barh(np.arange(len(pairs)), pairs["n"], color="#264653")
    ax.set_yticks(np.arange(len(pairs)), labels)
    ax.set_xlabel("Total head-to-head matches")
    ax.set_title(f"Top {top_n} most-played fixtures")
    for i, v in enumerate(pairs["n"]):
        ax.text(v + 0.1, i, f"{v}", va="center", fontsize=8)
    ax.grid(axis="x", alpha=0.3)


def _plot_unbeaten(ax, df, top_n=12):
    runs = _longest_unbeaten_runs(df, top_n).iloc[::-1].reset_index(drop=True)
    if runs.empty:
        ax.text(0.5, 0.5, "No streaks ≥ 5 matches found",
                ha="center", va="center", transform=ax.transAxes)
        ax.set_axis_off()
        return
    ax.barh(np.arange(len(runs)), runs["longest"], color="#2A9D8F")
    ax.set_yticks(np.arange(len(runs)), runs["team_name"])
    ax.set_xlabel("Longest unbeaten run (W + D, consecutive across all comps)")
    ax.set_title(f"Top {top_n} longest unbeaten runs")
    for i, v in enumerate(runs["longest"]):
        ax.text(v + 0.2, i, f"{v}", va="center", fontsize=8)
    ax.grid(axis="x", alpha=0.3)


def main():
    import argparse

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    parser = argparse.ArgumentParser(description="Exploration & records figures")
    parser.add_argument(
        "--seeding", choices=["hardcoded", "league_elo", "uniform"],
        default="hardcoded",
        help="Elo seeding mode (affects which matches show up as 'biggest upsets').",
    )
    parser.add_argument("--out", default=None, help="Output PNG path")
    args = parser.parse_args()

    df = load_fixtures()
    df_elo, _ = compute_elo_ratings(df, seeding=args.seeding)
    logger.info("Loaded %d fixtures, seeding=%s", len(df), args.seeding)

    fig, axes = plt.subplots(3, 2, figsize=(20, 18), constrained_layout=True)
    fig.suptitle(
        f"Exploration & records — {len(df):,} matches across "
        f"{df['league_id'].nunique()} competitions (seeding={args.seeding})",
        fontsize=15, fontweight="bold",
    )

    _plot_most_fixtured(axes[0, 0], df)
    _plot_cross_comp(axes[0, 1], df)
    _plot_high_scoring(axes[1, 0], df)
    _plot_biggest_upsets(axes[1, 1], df_elo)
    _plot_h2h(axes[2, 0], df)
    _plot_unbeaten(axes[2, 1], df)

    if args.out:
        out = Path(args.out)
    else:
        out = _FIGURES_DIR / f"exploration_{args.seeding}.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=140, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
