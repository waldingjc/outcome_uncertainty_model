"""Team-level breakdown figure.

Generates a six-panel overview for any single team: season-by-season W/D/L,
goals distribution, cumulative points trajectory by season, top opponents,
scoreline frequency heatmap, and performance by competition.

Usage:
    python -m src.analysis.team_breakdown --team Fenerbahce
    python -m src.analysis.team_breakdown --team 611      # by team_id
    python -m src.analysis.team_breakdown --team "Real Madrid" --out custom.png
"""

from __future__ import annotations

import argparse
import logging
import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.db.schema import get_connection

logger = logging.getLogger(__name__)

_FIGURES_DIR = Path(__file__).parents[2] / "data" / "figures"

_COLORS = {
    "W": "#2A9D8F",   # green
    "D": "#E9C46A",   # yellow
    "L": "#E76F51",   # red
    "scored":   "#264653",
    "conceded": "#E76F51",
}


def load_fixtures() -> pd.DataFrame:
    sql = """
        SELECT fixture_id, date, league_id, league_name, season,
               home_team_id, home_team_name, away_team_id, away_team_name,
               home_goals, away_goals, home_goals_ht, away_goals_ht
        FROM fixtures
        WHERE status = 'FT'
          AND home_goals IS NOT NULL
          AND away_goals IS NOT NULL
    """
    with get_connection() as conn:
        df = pd.read_sql(sql, conn, parse_dates=["date"])
    return df


def find_team(query: str, df: pd.DataFrame) -> tuple[int, str]:
    """Resolve a team substring to (team_id, canonical_name).

    On multiple matches, picks the team with the most fixtures and logs
    the alternatives so the caller can disambiguate via team_id if needed.
    """
    home = df[["home_team_id", "home_team_name"]].rename(
        columns={"home_team_id": "id", "home_team_name": "name"}
    )
    away = df[["away_team_id", "away_team_name"]].rename(
        columns={"away_team_id": "id", "away_team_name": "name"}
    )
    teams = pd.concat([home, away]).drop_duplicates("id")

    pattern = re.escape(query)
    matches = teams[teams["name"].str.contains(pattern, case=False, na=False, regex=True)]
    if matches.empty:
        # Try a more lenient match — strip diacritics on both sides.
        try:
            import unicodedata
            def strip_diacritics(s: str) -> str:
                return "".join(
                    c for c in unicodedata.normalize("NFD", s)
                    if unicodedata.category(c) != "Mn"
                )
            q_stripped = strip_diacritics(query).lower()
            mask = teams["name"].apply(
                lambda n: q_stripped in strip_diacritics(str(n)).lower()
            )
            matches = teams[mask]
        except Exception:
            pass

    if matches.empty:
        raise ValueError(f"No team matching {query!r} in the dataset")

    counts = pd.concat([
        df.groupby("home_team_id").size().rename("n_home"),
        df.groupby("away_team_id").size().rename("n_away"),
    ], axis=1).fillna(0).sum(axis=1)
    matches = matches.assign(n=matches["id"].map(counts).fillna(0).astype(int))
    matches = matches.sort_values("n", ascending=False)

    if len(matches) > 1:
        alternatives = [
            f"{r['name']} (id={r['id']}, {r['n']} matches)"
            for _, r in matches.iloc[1:6].iterrows()
        ]
        logger.info(
            "Multiple matches for %r — picked %r (id=%d, %d matches). "
            "Other candidates: %s",
            query, matches.iloc[0]["name"], int(matches.iloc[0]["id"]),
            int(matches.iloc[0]["n"]), alternatives,
        )

    chosen = matches.iloc[0]
    return int(chosen["id"]), str(chosen["name"])


def team_perspective(df: pd.DataFrame, team_id: int) -> pd.DataFrame:
    """Reshape fixtures into team-perspective rows (one row per match the team played)."""
    home = df[df["home_team_id"] == team_id].copy()
    home["venue"] = "home"
    home["team_goals"] = home["home_goals"]
    home["opp_goals"]  = home["away_goals"]
    home["opp_id"]     = home["away_team_id"]
    home["opp_name"]   = home["away_team_name"]

    away = df[df["away_team_id"] == team_id].copy()
    away["venue"] = "away"
    away["team_goals"] = away["away_goals"]
    away["opp_goals"]  = away["home_goals"]
    away["opp_id"]     = away["home_team_id"]
    away["opp_name"]   = away["home_team_name"]

    out = (
        pd.concat([home, away], ignore_index=True)
        .sort_values("date")
        .reset_index(drop=True)
    )
    out["result"] = np.where(
        out["team_goals"] > out["opp_goals"], "W",
        np.where(out["team_goals"] < out["opp_goals"], "L", "D"),
    )
    out["points"] = out["result"].map({"W": 3, "D": 1, "L": 0})
    out["gd"] = out["team_goals"] - out["opp_goals"]
    return out


# ---------------- Plotting helpers ----------------

def _season_label(s: int) -> str:
    return f"{s}-{(s + 1) % 100:02d}"


def _plot_season_summary(ax, dft):
    seasons = sorted(dft["season"].unique())
    venues = ["home", "away"]
    width = 0.4
    x = np.arange(len(seasons))

    for i, venue in enumerate(venues):
        bottom = np.zeros(len(seasons))
        for outcome in ["W", "D", "L"]:
            counts = np.array([
                ((dft["season"] == s)
                 & (dft["venue"] == venue)
                 & (dft["result"] == outcome)).sum()
                for s in seasons
            ])
            offset = (i - 0.5) * width
            ax.bar(
                x + offset, counts, width=width, bottom=bottom,
                color=_COLORS[outcome],
                edgecolor="white", linewidth=1.0,
                hatch="" if venue == "home" else "//",
                label=f"{outcome}" if (i == 0) else None,
            )
            bottom += counts

    ax.set_xticks(x, [_season_label(s) for s in seasons])
    ax.set_ylabel("Matches")
    ax.set_title("Results by season\n(left = home, right = away with hatching)")
    ax.legend(loc="upper right", fontsize=9)
    ax.grid(axis="y", alpha=0.3)


def _plot_goals_distribution(ax, dft):
    max_g = int(max(dft["team_goals"].max(), dft["opp_goals"].max()))
    bins = np.arange(-0.5, max_g + 1.5)

    mu_s = dft["team_goals"].mean()
    mu_c = dft["opp_goals"].mean()
    ax.hist(dft["team_goals"], bins=bins, color=_COLORS["scored"],
            alpha=0.65, edgecolor="white",
            label=f"Scored (μ={mu_s:.2f})")
    ax.hist(dft["opp_goals"], bins=bins, color=_COLORS["conceded"],
            alpha=0.65, edgecolor="white",
            label=f"Conceded (μ={mu_c:.2f})")
    ax.axvline(mu_s, color=_COLORS["scored"], linestyle="--", linewidth=1.4)
    ax.axvline(mu_c, color=_COLORS["conceded"], linestyle="--", linewidth=1.4)
    ax.set_xlabel("Goals in match")
    ax.set_ylabel("Number of matches")
    ax.set_title("Goals scored vs conceded distribution")
    ax.set_xticks(range(max_g + 1))
    ax.legend()
    ax.grid(axis="y", alpha=0.3)


def _plot_points_trajectory(ax, dft):
    seasons = sorted(dft["season"].unique())
    cmap = plt.colormaps.get_cmap("viridis")
    colors = cmap(np.linspace(0.15, 0.85, max(1, len(seasons))))
    max_matches = 0
    for s, c in zip(seasons, colors):
        sub = dft[dft["season"] == s].copy()
        if sub.empty:
            continue
        # Prepend a (0, 0) anchor so each season starts visually at 0 points
        # before any match has been played.
        cum = np.concatenate([[0], sub["points"].cumsum().values])
        x = np.arange(0, len(cum))
        ax.plot(x, cum, color=c, linewidth=2.2,
                label=f"{_season_label(s)} ({len(sub)} matches, {cum[-1]} pts)")
        max_matches = max(max_matches, len(sub))
    ax.plot([0, max_matches], [0, 3 * max_matches],
            color="grey", linestyle=":", alpha=0.6, label="3 PPG (perfect)")
    ax.set_xlabel("Match number within season")
    ax.set_ylabel("Cumulative points")
    ax.set_title("Cumulative points by season")
    ax.legend(loc="upper left", fontsize=8)
    ax.grid(alpha=0.3)


def _plot_opponents(ax, dft, top_n: int = 12):
    counts = (
        dft.groupby(["opp_id", "opp_name", "result"]).size()
        .unstack("result", fill_value=0)
        .reindex(columns=["W", "D", "L"], fill_value=0)
    )
    counts["total"] = counts[["W", "D", "L"]].sum(axis=1)
    counts = counts.nlargest(top_n, "total").reset_index()
    counts = counts.sort_values("total", ascending=True)

    y = np.arange(len(counts))
    ax.barh(y, counts["W"], color=_COLORS["W"], label="W")
    ax.barh(y, counts["D"], left=counts["W"], color=_COLORS["D"], label="D")
    ax.barh(y, counts["L"], left=counts["W"] + counts["D"],
            color=_COLORS["L"], label="L")
    ax.set_yticks(y, counts["opp_name"])
    ax.set_xlabel("Matches played")
    ax.set_title(f"Top {len(counts)} most-played opponents")
    ax.legend(loc="lower right")
    ax.grid(axis="x", alpha=0.3)


def _plot_scoreline_heatmap(ax, dft, cap: int = 5):
    dft = dft.copy()
    dft["tg_cap"] = np.clip(dft["team_goals"], 0, cap)
    dft["og_cap"] = np.clip(dft["opp_goals"], 0, cap)
    grid = (
        pd.crosstab(dft["tg_cap"], dft["og_cap"])
        .reindex(index=range(cap + 1), columns=range(cap + 1), fill_value=0)
    )
    im = ax.imshow(grid.values, cmap="YlOrRd", origin="lower", aspect="auto")
    labels = [str(i) if i < cap else f"{cap}+" for i in range(cap + 1)]
    ax.set_xticks(range(cap + 1), labels)
    ax.set_yticks(range(cap + 1), labels)
    ax.set_xlabel("Opponent goals")
    ax.set_ylabel("Team goals")
    ax.set_title("Scoreline frequency")

    vmax = grid.values.max() or 1
    for i in range(cap + 1):
        for j in range(cap + 1):
            v = grid.values[i, j]
            if v == 0:
                continue
            color = "white" if v > vmax * 0.5 else "black"
            ax.text(j, i, str(v), ha="center", va="center",
                    color=color, fontsize=9)
    plt.colorbar(im, ax=ax, label="matches", shrink=0.85)


def _plot_by_competition(ax, dft):
    counts = (
        dft.groupby(["league_id", "league_name", "result"]).size()
        .unstack("result", fill_value=0)
        .reindex(columns=["W", "D", "L"], fill_value=0)
    )
    counts["total"] = counts[["W", "D", "L"]].sum(axis=1)
    counts = counts.sort_values("total", ascending=True).reset_index()

    y = np.arange(len(counts))
    ax.barh(y, counts["W"], color=_COLORS["W"], label="W")
    ax.barh(y, counts["D"], left=counts["W"], color=_COLORS["D"], label="D")
    ax.barh(y, counts["L"], left=counts["W"] + counts["D"],
            color=_COLORS["L"], label="L")
    ax.set_yticks(y, counts["league_name"])
    ax.set_xlabel("Matches played")
    ax.set_title("Performance by competition")
    ax.legend(loc="lower right")
    ax.grid(axis="x", alpha=0.3)


def plot_team_breakdown(name: str, dft: pd.DataFrame, out_path: Path) -> None:
    fig, axes = plt.subplots(2, 3, figsize=(20, 12), constrained_layout=True)

    record = dft["result"].value_counts().reindex(["W", "D", "L"]).fillna(0).astype(int)
    win_rate = record["W"] / record.sum() if record.sum() else 0.0
    fig.suptitle(
        f"{name}  —  {len(dft):,} matches across {dft['league_id'].nunique()} competitions  "
        f"(seasons {min(dft['season'])}–{max(dft['season'])})\n"
        f"Record: {record['W']}W / {record['D']}D / {record['L']}L  "
        f"({win_rate:.1%} win rate, "
        f"GF={dft['team_goals'].sum()}, GA={dft['opp_goals'].sum()}, "
        f"GD={int(dft['gd'].sum()):+d})",
        fontsize=14, fontweight="bold",
    )

    _plot_season_summary(axes[0, 0], dft)
    _plot_goals_distribution(axes[0, 1], dft)
    _plot_points_trajectory(axes[0, 2], dft)
    _plot_opponents(axes[1, 0], dft)
    _plot_scoreline_heatmap(axes[1, 1], dft)
    _plot_by_competition(axes[1, 2], dft)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=140, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved team breakdown to %s", out_path)


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    parser = argparse.ArgumentParser(description="Generate a team-level breakdown figure")
    parser.add_argument("--team", required=True,
                        help="Team name (substring, case- and diacritic-insensitive) or numeric team_id")
    parser.add_argument("--out", default=None,
                        help="Output PNG path (default: data/figures/team_<name>.png)")
    args = parser.parse_args()

    df = load_fixtures()

    if args.team.isdigit():
        team_id = int(args.team)
        sample = df[(df["home_team_id"] == team_id) | (df["away_team_id"] == team_id)]
        if sample.empty:
            raise SystemExit(f"No fixtures found for team_id={team_id}")
        first = sample.iloc[0]
        name = (first["home_team_name"] if first["home_team_id"] == team_id
                else first["away_team_name"])
    else:
        team_id, name = find_team(args.team, df)

    dft = team_perspective(df, team_id)
    if dft.empty:
        raise SystemExit(f"No fixtures found for {name} (id={team_id})")

    logger.info("Team: %s (id=%d), %d matches in dataset", name, team_id, len(dft))

    if args.out:
        out_path = Path(args.out)
    else:
        safe = re.sub(r"[^\w\-]+", "_", name).strip("_")
        out_path = _FIGURES_DIR / f"team_{safe}.png"

    plot_team_breakdown(name, dft, out_path)
    print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()
