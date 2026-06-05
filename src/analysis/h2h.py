"""Multi-team head-to-head comparison figure.

Mirrors the structure of `team_breakdown.plot_team_breakdown` but
overlays N teams (2-4) on each panel so the user can compare them
directly. Where the single-team module shows "scored vs conceded
distribution", we show "scored distribution, one curve per team".

Panels (2x3):
  1. Pairwise W/D/L matrix between the teams in the set
  2. Goals scored per match — overlaid distributions, one per team
  3. Cumulative points by season — one line per team (current season only)
  4. Rolling win rate (50-match window) — one line per team
  5. Performance by competition — grouped horizontal bars
  6. H2H meetings by season — stacked bars (one stack per pairing winner)

Usage:
    from src.analysis.h2h import plot_h2h_breakdown
    plot_h2h_breakdown([39, 40], ["Arsenal", "Liverpool"], df, Path("h2h.png"))
"""

from __future__ import annotations

import logging
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.analysis._style import (
    CYCLE, D_COLOR, L_COLOR, W_COLOR, apply_dark_style,
)
from src.analysis.team_breakdown import team_perspective

logger = logging.getLogger(__name__)


# ---------------- Helpers ----------------

def _team_color(idx: int) -> str:
    """Per-team color, cycled. Used consistently across all panels so
    each team is always the same colour."""
    return CYCLE[idx % len(CYCLE)]


def _season_label(s: int) -> str:
    return f"{s}-{(s + 1) % 100:02d}"


# ---------------- Panel plotters ----------------

def _plot_pairwise_matrix(
    ax, df: pd.DataFrame, team_ids: list[int], team_names: list[str],
) -> None:
    """N×N grid showing the W-D-L record of each row team against
    each column team (W from the row team's perspective). Diagonal
    cells are left blank since a team doesn't play itself."""
    n = len(team_ids)
    # Compute W/D/L for each ordered pair (row team perspective).
    records = np.empty((n, n, 3), dtype=int)  # [W, D, L] per cell
    for i, a in enumerate(team_ids):
        for j, b in enumerate(team_ids):
            if i == j:
                records[i, j] = (0, 0, 0)
                continue
            mask = (
                ((df["home_team_id"] == a) & (df["away_team_id"] == b))
                | ((df["home_team_id"] == b) & (df["away_team_id"] == a))
            )
            sub = df[mask]
            a_home = sub["home_team_id"] == a
            a_goals = sub["home_goals"].where(a_home, sub["away_goals"])
            b_goals = sub["away_goals"].where(a_home, sub["home_goals"])
            w = int((a_goals > b_goals).sum())
            d = int((a_goals == b_goals).sum())
            l = int((a_goals < b_goals).sum())
            records[i, j] = (w, d, l)

    ax.set_xticks(range(n), team_names, rotation=20, ha="right")
    ax.set_yticks(range(n), team_names)
    ax.set_xlim(-0.5, n - 0.5)
    ax.set_ylim(n - 0.5, -0.5)
    ax.set_title("Pairwise record (row team vs column team)")
    ax.set_aspect("equal")

    for i in range(n):
        for j in range(n):
            if i == j:
                ax.add_patch(plt.Rectangle(
                    (j - 0.45, i - 0.45), 0.9, 0.9,
                    facecolor="#2a2c30", edgecolor="#161719",
                ))
                ax.text(j, i, "—", ha="center", va="center",
                        color="#9aa0a6", fontsize=12)
                continue
            w, d, l = records[i, j]
            total = w + d + l
            if total == 0:
                ax.add_patch(plt.Rectangle(
                    (j - 0.45, i - 0.45), 0.9, 0.9,
                    facecolor="#1c1d20", edgecolor="#161719",
                ))
                ax.text(j, i, "no\nmatches", ha="center", va="center",
                        color="#9aa0a6", fontsize=8)
                continue
            # Three horizontal stripes within the cell, widths proportional
            # to W/D/L counts. Reads at a glance.
            wr, dr, lr = w / total, d / total, l / total
            x0 = j - 0.45
            ax.add_patch(plt.Rectangle((x0, i - 0.45), 0.9 * wr, 0.9,
                                       facecolor=W_COLOR, edgecolor="none"))
            ax.add_patch(plt.Rectangle((x0 + 0.9 * wr, i - 0.45),
                                       0.9 * dr, 0.9,
                                       facecolor=D_COLOR, edgecolor="none"))
            ax.add_patch(plt.Rectangle((x0 + 0.9 * (wr + dr), i - 0.45),
                                       0.9 * lr, 0.9,
                                       facecolor=L_COLOR, edgecolor="none"))
            ax.text(j, i, f"{w}-{d}-{l}", ha="center", va="center",
                    color="#161719", fontsize=10, fontweight="bold")


def _plot_goals_overlay(
    ax, perspectives: list[pd.DataFrame], team_names: list[str],
) -> None:
    """Overlaid step histograms of goals scored per match, one curve
    per team. Includes a mean-marker for each team."""
    all_goals = [d["team_goals"].values for d in perspectives if not d.empty]
    if not all_goals:
        ax.set_title("Goals scored — no data")
        return

    max_g = int(max(g.max() for g in all_goals))
    bins = np.arange(-0.5, max_g + 1.5)
    for idx, (name, dft) in enumerate(zip(team_names, perspectives)):
        if dft.empty:
            continue
        goals = dft["team_goals"].values
        c = _team_color(idx)
        # Step histogram + filled-but-translucent body so overlap is readable.
        ax.hist(goals, bins=bins, color=c, alpha=0.25, label=None)
        ax.hist(goals, bins=bins, color=c, histtype="step", linewidth=2.0,
                label=f"{name} (μ={goals.mean():.2f})")
        ax.axvline(goals.mean(), color=c, linestyle="--",
                   linewidth=1.2, alpha=0.7)

    ax.set_xlabel("Goals scored per match")
    ax.set_ylabel("Match count")
    ax.set_title("Goals scored distribution (overlaid)")
    ax.set_xticks(range(max_g + 1))
    ax.legend(loc="upper right", fontsize=9)
    ax.grid(axis="y", alpha=0.3)


def _plot_points_by_season(
    ax, perspectives: list[pd.DataFrame], team_names: list[str],
) -> None:
    """Cumulative-points trajectory for each team's most recent season."""
    longest = 0
    plotted = 0
    for idx, (name, dft) in enumerate(zip(team_names, perspectives)):
        if dft.empty:
            continue
        latest_season = int(dft["season"].max())
        sub = dft[dft["season"] == latest_season].sort_values("date")
        if sub.empty:
            continue
        cum = np.concatenate([[0], sub["points"].cumsum().values])
        x = np.arange(0, len(cum))
        ax.plot(x, cum, color=_team_color(idx), linewidth=2.4,
                label=f"{name} ({_season_label(latest_season)}, {cum[-1]} pts)")
        longest = max(longest, len(sub))
        plotted += 1

    if plotted == 0:
        ax.set_title("Points trajectory — no data")
        return

    ax.plot([0, longest], [0, 3 * longest], color="grey", linestyle=":",
            alpha=0.5, label="3 PPG")
    ax.set_xlabel("Match number within season")
    ax.set_ylabel("Cumulative points")
    ax.set_title("Most-recent season points trajectory")
    ax.legend(loc="upper left", fontsize=9)
    ax.grid(alpha=0.3)


def _plot_rolling_winrate(
    ax, perspectives: list[pd.DataFrame], team_names: list[str],
    window: int = 50,
) -> None:
    """Rolling N-match win rate over time — one line per team."""
    plotted = 0
    for idx, (name, dft) in enumerate(zip(team_names, perspectives)):
        if dft.empty or len(dft) < window:
            continue
        sub = dft.sort_values("date")
        wins = (sub["result"] == "W").astype(int)
        roll = wins.rolling(window).mean() * 100
        ax.plot(sub["date"], roll, color=_team_color(idx), linewidth=2.0,
                label=name)
        plotted += 1

    if plotted == 0:
        ax.set_title("Rolling win rate — not enough matches")
        return

    ax.set_xlabel("Date")
    ax.set_ylabel(f"Win rate over rolling {window} matches (%)")
    ax.set_title(f"Form trajectory — {window}-match rolling win rate")
    ax.legend(loc="lower right", fontsize=9)
    ax.grid(alpha=0.3)
    ax.set_ylim(0, 100)


def _plot_by_competition(
    ax, perspectives: list[pd.DataFrame], team_names: list[str],
    top_n: int = 8,
) -> None:
    """Grouped horizontal bars: per team, matches played in each of
    the top N competitions across all teams in the set combined."""
    rows = []
    for idx, dft in enumerate(perspectives):
        if dft.empty:
            continue
        counts = dft.groupby("league_name").size()
        for league, n in counts.items():
            rows.append((idx, league, int(n)))
    if not rows:
        ax.set_title("By competition — no data")
        return
    df_counts = pd.DataFrame(rows, columns=["idx", "league", "n"])
    top_leagues = (
        df_counts.groupby("league")["n"].sum()
        .nlargest(top_n).index.tolist()
    )
    df_top = df_counts[df_counts["league"].isin(top_leagues)]

    y = np.arange(len(top_leagues))
    n_teams = len(perspectives)
    bar_h = 0.8 / n_teams
    for idx, name in enumerate(team_names):
        sub = df_top[df_top["idx"] == idx].set_index("league")["n"]
        vals = [int(sub.get(l, 0)) for l in top_leagues]
        offset = (idx - (n_teams - 1) / 2) * bar_h
        ax.barh(y + offset, vals, height=bar_h * 0.9,
                color=_team_color(idx), label=name)
    ax.set_yticks(y, top_leagues)
    ax.invert_yaxis()
    ax.set_xlabel("Matches played")
    ax.set_title(f"Top {len(top_leagues)} competitions (per team)")
    ax.legend(loc="lower right", fontsize=9)
    ax.grid(axis="x", alpha=0.3)


def _plot_meetings_by_season(
    ax, fixtures: pd.DataFrame, team_ids: list[int], team_names: list[str],
) -> None:
    """Head-to-head meetings between the team set, by season, stacked
    by which team in the set won. For N=2 this is the classic
    'meetings per season' chart; for N>2 it shows multilateral history."""
    # Build a meetings dataframe: matches where both home and away
    # are in the team set.
    tset = set(team_ids)
    mask = fixtures["home_team_id"].isin(tset) & fixtures["away_team_id"].isin(tset)
    h2h = fixtures[mask].copy()
    if h2h.empty:
        ax.set_title("H2H meetings — none in the dataset")
        return

    # Winner team_id (0 if draw)
    home_won = h2h["home_goals"] > h2h["away_goals"]
    away_won = h2h["home_goals"] < h2h["away_goals"]
    h2h["winner_id"] = np.where(home_won, h2h["home_team_id"],
                                np.where(away_won, h2h["away_team_id"], 0))

    seasons = sorted(h2h["season"].unique())
    name_by_id = dict(zip(team_ids, team_names))

    bottoms = np.zeros(len(seasons))
    for idx, tid in enumerate(team_ids):
        counts = np.array([
            int(((h2h["season"] == s) & (h2h["winner_id"] == tid)).sum())
            for s in seasons
        ])
        ax.bar(range(len(seasons)), counts, bottom=bottoms,
               color=_team_color(idx), label=f"{name_by_id[tid]} won",
               edgecolor="#161719", linewidth=0.6)
        bottoms += counts
    # Draws on top in neutral colour
    draws = np.array([
        int(((h2h["season"] == s) & (h2h["winner_id"] == 0)).sum())
        for s in seasons
    ])
    ax.bar(range(len(seasons)), draws, bottom=bottoms,
           color=D_COLOR, label="draw",
           edgecolor="#161719", linewidth=0.6)

    ax.set_xticks(range(len(seasons)),
                  [_season_label(s) for s in seasons], rotation=0)
    ax.set_ylabel("Meetings")
    ax.set_title("H2H meetings by season (stacked by winner)")
    ax.legend(loc="upper left", fontsize=9)
    ax.grid(axis="y", alpha=0.3)


# ---------------- Driver ----------------

def plot_h2h_breakdown(
    team_ids: list[int], team_names: list[str],
    fixtures: pd.DataFrame, out_path: Path,
) -> None:
    """Render the 2x3 multi-team comparison figure to `out_path`.

    `fixtures` is the full project fixtures dataframe (the same
    shape `load_fixtures()` returns from `team_breakdown`).

    Pre-computes `team_perspective(fixtures, tid)` once per team and
    passes the per-team dataframes into the panel functions. Earlier
    each panel re-computed perspectives independently, so a 4-team
    figure was doing ~20 redundant 480K-row dataframe scans.
    """
    if len(team_ids) < 2:
        raise ValueError("plot_h2h_breakdown needs at least 2 teams")
    apply_dark_style()

    # One-shot per-team perspective dataframes — reused by every panel
    # except the pairwise matrix and the season-meetings bars, which
    # need the raw cross-team fixture set.
    perspectives = [team_perspective(fixtures, tid) for tid in team_ids]

    fig, axes = plt.subplots(2, 3, figsize=(22, 12), constrained_layout=True)
    fig.suptitle("  vs  ".join(team_names), fontsize=15, fontweight="bold")

    _plot_pairwise_matrix(axes[0, 0], fixtures, team_ids, team_names)
    _plot_goals_overlay(axes[0, 1], perspectives, team_names)
    _plot_points_by_season(axes[0, 2], perspectives, team_names)
    _plot_rolling_winrate(axes[1, 0], perspectives, team_names)
    _plot_by_competition(axes[1, 1], perspectives, team_names)
    _plot_meetings_by_season(axes[1, 2], fixtures, team_ids, team_names)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=140, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved H2H figure to %s", out_path)
