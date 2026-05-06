"""Team-strength analysis: Elo ratings, league rating distributions, and
promotion/relegation step plots.

This module is the foundation for the modelling work — it produces a global
Elo rating per team and exposes that as a public function used by other
analyses (e.g. upset rate by league).

Generates `data/figures/strength.png` (2x2 panel):
  1. Top N teams by Elo, colour-coded by primary league.
  2. Elo distribution per league (boxplot, top leagues by team count).
  3. Promotion/relegation step plot — win rate before vs after a divisional move.
  4. League strength ladder — median team Elo per league.

Usage:
    python -m src.analysis.strength
"""

from __future__ import annotations

import logging
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.db.schema import get_connection

logger = logging.getLogger(__name__)

_FIGURES_DIR = Path(__file__).parents[2] / "data" / "figures"

# Standard Elo parameters tuned for football.
DEFAULT_K = 20
DEFAULT_HOME_ADVANTAGE = 70
DEFAULT_BASE_RATING = 1500

# Initial Elo per league (by league_id), used to seed each team based on their
# primary league before the Elo walk. Without this, the algorithm cannot tell
# that the Northern Premier League is weaker than Ligue 1, because matches
# between the two never happen — so dominant non-league teams end up with the
# same rating as PSG. Manually anchoring each league at a sensible baseline
# fixes the disconnected-graph problem.
#
# Anything not listed here defaults to DEFAULT_BASE_RATING (1500).
LEAGUE_INITIAL_RATING: dict[int, float] = {
    # ---- English pyramid (the main calibration target) ----
    39: 1700,   # Premier League
    40: 1500,   # Championship
    41: 1400,   # League One
    42: 1300,   # League Two
    43: 1200,   # National League
    50: 1100,   # National League - North
    51: 1100,   # National League - South
    58: 1050,   # Non League Premier - Isthmian
    59: 1050,   # Non League Premier - Northern
    60: 1050,   # Non League Premier - Southern South
    52: 1000,   # Non League Div One - Isthmian North
    53: 1000,   # Non League Div One - Isthmian South Central
    54: 1000,   # Non League Div One - Northern West
    55: 1000,   # Non League Div One - Northern Midlands
    56: 1000,   # Non League Div One - Southern South
    57: 1000,   # Non League Div One - Isthmian South East

    # ---- Top European leagues ----
    140: 1700,  # La Liga
    78: 1700,   # Bundesliga
    135: 1700,  # Serie A (Italy)
    61: 1700,   # Ligue 1
    203: 1600,  # Süper Lig

    # ---- 2nd tier European ----
    79: 1500,   # 2. Bundesliga
    72: 1500,   # Serie B (Italy)
    62: 1500,   # Ligue 2

    # ---- 3rd / 4th tier European ----
    75: 1400,   # Serie C (Italy)
    76: 1300,   # Serie D (Italy)
    63: 1300,   # National 1 (France)
    67: 1200, 68: 1200, 69: 1200, 70: 1200,  # National 2 (France) groups

    # ---- South America ----
    71: 1600,   # Serie A (Brazil)
    73: 1500,   # Copa do Brasil
    74: 1300,   # Brasileiro Women
    77: 1300,   # Alagoano (state league)

    # ---- Women's leagues ----
    44: 1500,   # FA WSL
    64: 1400,   # Feminine Division 1 (France)

    # ---- Major continental club competitions (entrants get domestic ratings) ----
    2: 1700,    # UEFA Champions League
    3: 1600,    # UEFA Europa League
    11: 1500,   # CONMEBOL Sudamericana
    12: 1500,   # CAF Champions League
    13: 1600,   # CONMEBOL Libertadores
    14: 1500,   # UEFA Youth League
    16: 1500,   # CONCACAF Champions League
    17: 1500,   # AFC Champions League
    18: 1400,   # AFC Champions League Two
    20: 1300,   # CAF Confederation Cup
    27: 1300,   # OFC Champions League
    15: 1700,   # FIFA Club World Cup

    # ---- Domestic cups (bridge tiers; default-y) ----
    45: 1500,   # FA Cup
    46: 1300,   # EFL Trophy
    47: 1100,   # FA Trophy
    48: 1500,   # League Cup (EFL Cup)
    66: 1500,   # Coupe de France

    # ---- International / national-team competitions ----
    1: 1500,    # World Cup
    4: 1500,    # Euro Championship
    5: 1500,    # UEFA Nations League
    6: 1500,    # AFCON
    7: 1500,    # Asian Cup
    8: 1300,    # World Cup - Women
    9: 1500,    # Copa America
    10: 1500,   # Friendlies
    19: 1300,   # African Nations Championship
    22: 1500,   # CONCACAF Gold Cup
    24: 1300,   # ASEAN Championship
    25: 1300,   # Gulf Cup of Nations
    28: 1200,   # SAFF Championship
    29: 1500, 30: 1500, 31: 1500, 32: 1500, 33: 1500, 34: 1500,  # WC qualifying
    35: 1300, 36: 1300, 37: 1500,
    38: 1300,   # UEFA U21 Championship
}


def load_fixtures() -> pd.DataFrame:
    sql = """
        SELECT fixture_id, date, league_id, league_name, season,
               home_team_id, home_team_name, away_team_id, away_team_name,
               home_goals, away_goals
        FROM fixtures
        WHERE status = 'FT'
          AND home_goals IS NOT NULL
          AND away_goals IS NOT NULL
    """
    with get_connection() as conn:
        df = pd.read_sql(sql, conn, parse_dates=["date"])
    df["result"] = np.where(
        df["home_goals"] > df["away_goals"], "H",
        np.where(df["home_goals"] < df["away_goals"], "A", "D"),
    )
    return df


def compute_league_elo(
    df: pd.DataFrame,
    k_init: float = 40.0,
    k_final: float = 5.0,
    max_passes: int = 10,
    convergence_threshold: float = 1.0,
    balance_sigma: float = 200.0,
    base_rating: float = DEFAULT_BASE_RATING,
) -> tuple[dict[int, float], dict[int, int]]:
    """Compute Elo per LEAGUE using only inter-league matches.

    Strategy: walk through matches chronologically. If the two teams have
    different primary leagues, we treat the result as a contest between
    the two leagues and update each league's rating. Intra-league matches
    are skipped — they tell us about within-league spread, not absolute
    league strength.

    Three improvements over a naive single-pass walk:

    1. **Multi-pass with K decay** — iterate up to `max_passes` (default 10),
       linearly decaying K from `k_init` (default 40, big swings to
       establish ordering) down to `k_final` (default 5, fine-tuning).

    2. **Balanced-match weighting** — from pass 2 onwards, scale each
       match's update by a Gaussian of the league-Elo gap:
            weight = exp(-(gap / sigma)**2)
       Matches between similarly-rated leagues (gap small) get full weight;
       matches between mismatched leagues (gap large) get near-zero
       weight. This is the key fix for the "top tiers crush their
       domestic cup opponents and drift up" artifact: once ratings are
       roughly correct, the cup-stomp matches contribute almost nothing,
       and ratings settle based on contests between comparable leagues.

    3. **Early stopping on convergence** — after each pass, measure the
       max per-league rating change. If below `convergence_threshold`
       (default 1.0 Elo points), declare converged and stop. Otherwise
       run all `max_passes`.

    Returns:
        league_ratings:    {league_id: final Elo}
        n_inter_matches:   {league_id: count of inter-league matches
                            that updated this league across pass 1}
    """
    import math

    df = df.sort_values("date").reset_index(drop=True)
    pmap = primary_league_map(df)

    home_ids = df["home_team_id"].values
    away_ids = df["away_team_id"].values
    home_goals = df["home_goals"].values
    away_goals = df["away_goals"].values

    league_ratings: dict[int, float] = {}
    n_matches: dict[int, int] = {}

    # Pre-compute league IDs for each match (one allocation, not n × max_passes).
    h_lids = np.empty(len(df), dtype=np.int64)
    a_lids = np.empty(len(df), dtype=np.int64)
    for i in range(len(df)):
        h_lids[i] = pmap.get(int(home_ids[i]), (-1, ""))[0]
        a_lids[i] = pmap.get(int(away_ids[i]), (-1, ""))[0]
    valid = (h_lids != a_lids) & (h_lids >= 0) & (a_lids >= 0)

    actual_h_arr = np.where(
        home_goals > away_goals, 1.0,
        np.where(home_goals < away_goals, 0.0, 0.5),
    )

    for pass_idx in range(max_passes):
        # Linearly decay K from k_init (pass 0) to k_final (pass max-1).
        if max_passes <= 1:
            k = k_final
        else:
            k = k_init + (k_final - k_init) * (pass_idx / (max_passes - 1))

        prev_ratings = dict(league_ratings)

        for i in range(len(df)):
            if not valid[i]:
                continue
            h_lid = int(h_lids[i])
            a_lid = int(a_lids[i])

            h = league_ratings.get(h_lid, base_rating)
            a = league_ratings.get(a_lid, base_rating)
            expected_h = 1.0 / (1.0 + 10.0 ** ((a - h) / 400.0))

            # Balance weight: only kicks in from pass 2 onwards (need rough
            # ratings first). Matches between mismatched leagues contribute
            # less. Pass 1 is unweighted to establish initial ordering.
            if pass_idx == 0:
                weight = 1.0
            else:
                gap = abs(h - a)
                weight = math.exp(-((gap / balance_sigma) ** 2))

            delta = k * weight * (actual_h_arr[i] - expected_h)
            league_ratings[h_lid] = h + delta
            league_ratings[a_lid] = a - delta

            # Count samples only on pass 1.
            if pass_idx == 0:
                n_matches[h_lid] = n_matches.get(h_lid, 0) + 1
                n_matches[a_lid] = n_matches.get(a_lid, 0) + 1

        # Convergence: max change since the snapshot at start of pass.
        if pass_idx > 0:
            keys = set(league_ratings) | set(prev_ratings)
            max_change = max(
                abs(league_ratings.get(lid, base_rating)
                    - prev_ratings.get(lid, base_rating))
                for lid in keys
            ) if keys else 0.0
            logger.info(
                "Pass %d/%d (k=%.1f): max league rating change = %.3f",
                pass_idx + 1, max_passes, k, max_change,
            )
            if max_change < convergence_threshold:
                logger.info(
                    "Converged after %d passes (max change %.3f < %.3f)",
                    pass_idx + 1, max_change, convergence_threshold,
                )
                break
        else:
            logger.info("Pass 1/%d (k=%.1f, unweighted)", max_passes, k)

    return league_ratings, n_matches


def _seed_team_ratings(
    df: pd.DataFrame, base_rating: float, mode: str = "hardcoded",
) -> dict[int, float]:
    """Initial Elo per team based on their primary league.

    Modes:
      - "hardcoded":  use the LEAGUE_INITIAL_RATING dict (curated by hand).
      - "league_elo": run `compute_league_elo` on inter-league matches and
                     use those data-driven league ratings as seeds. Falls
                     back to `base_rating` for leagues with zero
                     inter-league play.

    Either way, every team starts at the rating of their primary (most-
    played) league. This calibrates the disconnected-graph problem: without
    it, a dominant Tier-7 non-league side ends up at the same Elo ceiling
    as Premier League teams because their two networks never interact.
    """
    pmap = primary_league_map(df)
    if mode == "hardcoded":
        rating_map = LEAGUE_INITIAL_RATING
    elif mode == "league_elo":
        rating_map, _ = compute_league_elo(df, base_rating=base_rating)
    else:
        raise ValueError(f"Unknown seeding mode: {mode!r}")
    return {
        tid: rating_map.get(lid, base_rating)
        for tid, (lid, _) in pmap.items()
    }


def compute_elo_ratings(
    df: pd.DataFrame,
    k: float = DEFAULT_K,
    home_advantage: float = DEFAULT_HOME_ADVANTAGE,
    base_rating: float = DEFAULT_BASE_RATING,
    seeding: str = "hardcoded",
) -> tuple[pd.DataFrame, dict[int, float]]:
    """Walk matches chronologically, updating Elo ratings.

    Args:
        seeding: how to initialise team ratings before the walk. One of:
            - "hardcoded": LEAGUE_INITIAL_RATING dict (default; my curated
              guesses by tier).
            - "league_elo": run `compute_league_elo` first to produce
              data-driven league ratings, then seed teams from those.
            - "uniform":  every team starts at base_rating (1500).

    Returns:
        df_out:  copy of `df` sorted by date, with two new columns
                 `home_pre_elo` and `away_pre_elo` — each team's rating BEFORE
                 the match (used downstream for upset detection).
        ratings: final rating per team_id after all matches.
    """
    df = df.sort_values("date").reset_index(drop=True).copy()
    if seeding == "uniform":
        ratings: dict[int, float] = {}
    else:
        ratings = _seed_team_ratings(df, base_rating, mode=seeding)
    pre_home = np.empty(len(df), dtype=float)
    pre_away = np.empty(len(df), dtype=float)

    home_ids = df["home_team_id"].values
    away_ids = df["away_team_id"].values
    home_goals = df["home_goals"].values
    away_goals = df["away_goals"].values

    for i in range(len(df)):
        h_id = int(home_ids[i])
        a_id = int(away_ids[i])
        h = ratings.get(h_id, base_rating)
        a = ratings.get(a_id, base_rating)
        pre_home[i] = h
        pre_away[i] = a

        expected_h = 1.0 / (1.0 + 10.0 ** ((a - (h + home_advantage)) / 400.0))
        if home_goals[i] > away_goals[i]:
            actual_h = 1.0
        elif home_goals[i] < away_goals[i]:
            actual_h = 0.0
        else:
            actual_h = 0.5

        delta = k * (actual_h - expected_h)
        ratings[h_id] = h + delta
        ratings[a_id] = a - delta

    df["home_pre_elo"] = pre_home
    df["away_pre_elo"] = pre_away
    return df, ratings


def primary_league_map(df: pd.DataFrame) -> dict[int, tuple[int, str]]:
    """For each team, find the league they played the most matches in."""
    home = df[["home_team_id", "league_id", "league_name"]].rename(
        columns={"home_team_id": "team_id"}
    )
    away = df[["away_team_id", "league_id", "league_name"]].rename(
        columns={"away_team_id": "team_id"}
    )
    stacked = pd.concat([home, away])
    counts = stacked.groupby(["team_id", "league_id", "league_name"]).size().rename("n").reset_index()
    counts = counts.sort_values(["team_id", "n"], ascending=[True, False])
    primary = counts.drop_duplicates("team_id", keep="first").set_index("team_id")
    return {int(tid): (int(r["league_id"]), str(r["league_name"]))
            for tid, r in primary.iterrows()}


def team_name_map(df: pd.DataFrame) -> dict[int, str]:
    home = df[["home_team_id", "home_team_name"]].rename(
        columns={"home_team_id": "id", "home_team_name": "name"}
    )
    away = df[["away_team_id", "away_team_name"]].rename(
        columns={"away_team_id": "id", "away_team_name": "name"}
    )
    return dict(pd.concat([home, away]).drop_duplicates("id").set_index("id")["name"])


def promotion_relegation_pairs(df: pd.DataFrame) -> pd.DataFrame:
    """Find teams whose primary league changed between consecutive seasons.

    Returns one row per (team_id, season_pair) where the team's most-played
    league differs between season N and N+1. Useful for slope plots showing
    win-rate before vs after a divisional move.
    """
    # For each (team, season) compute the team's primary league for that season
    # and their win rate in it.
    home = df[["home_team_id", "season", "league_id", "league_name", "result"]].copy()
    home["team_id"] = home["home_team_id"]
    home["team_won"] = home["result"] == "H"
    home["was_home"] = True

    away = df[["away_team_id", "season", "league_id", "league_name", "result"]].copy()
    away["team_id"] = away["away_team_id"]
    away["team_won"] = away["result"] == "A"
    away["was_home"] = False

    stacked = pd.concat(
        [home[["team_id", "season", "league_id", "league_name", "team_won"]],
         away[["team_id", "season", "league_id", "league_name", "team_won"]]],
        ignore_index=True,
    )

    by_team_season_league = (
        stacked.groupby(["team_id", "season", "league_id", "league_name"])
        .agg(n=("team_won", "size"), wins=("team_won", "sum"))
        .reset_index()
    )
    by_team_season_league["win_rate"] = by_team_season_league["wins"] / by_team_season_league["n"]

    # Pick the team's primary (most-played) league for each season — that's
    # what we call "the team's league this season".
    by_team_season_league = by_team_season_league.sort_values(
        ["team_id", "season", "n"], ascending=[True, True, False]
    )
    primary_per_season = by_team_season_league.drop_duplicates(
        ["team_id", "season"], keep="first"
    )

    # Pair consecutive seasons (N, N+1) for each team where leagues differ.
    rows = []
    for tid, g in primary_per_season.groupby("team_id"):
        g = g.sort_values("season")
        for (_, prev), (_, curr) in zip(g.iterrows(), g.iloc[1:].iterrows()):
            if prev["league_id"] == curr["league_id"]:
                continue
            rows.append({
                "team_id": tid,
                "season_from": int(prev["season"]),
                "season_to": int(curr["season"]),
                "from_league_id": int(prev["league_id"]),
                "from_league_name": prev["league_name"],
                "from_win_rate": prev["win_rate"],
                "from_n": int(prev["n"]),
                "to_league_id": int(curr["league_id"]),
                "to_league_name": curr["league_name"],
                "to_win_rate": curr["win_rate"],
                "to_n": int(curr["n"]),
            })

    return pd.DataFrame(rows)


# ---------------- Plotting helpers ----------------

def _plot_top_teams(ax, ratings, names, primary_map, top_n=30):
    """Top N teams by Elo, colour-coded by primary league."""
    elo = pd.DataFrame(
        [(tid, r) for tid, r in ratings.items()],
        columns=["team_id", "elo"],
    )
    elo["name"] = elo["team_id"].map(names)
    elo["primary_league"] = elo["team_id"].map(
        lambda t: primary_map.get(t, (0, "?"))[1]
    )
    top = elo.nlargest(top_n, "elo").iloc[::-1].reset_index(drop=True)

    # Distinct colours per league appearing in this top list.
    leagues_present = top["primary_league"].unique().tolist()
    cmap = plt.get_cmap("tab20")
    colour_map = {lg: cmap(i % 20) for i, lg in enumerate(leagues_present)}

    bars = ax.barh(
        np.arange(len(top)), top["elo"],
        color=[colour_map[lg] for lg in top["primary_league"]],
    )
    ax.set_yticks(np.arange(len(top)), top["name"])
    ax.set_xlabel("Elo rating")
    ax.set_title(f"Top {top_n} teams by Elo (across all leagues)")
    ax.set_xlim(top["elo"].min() - 30, top["elo"].max() + 30)

    # Per-league legend.
    handles = [plt.Rectangle((0, 0), 1, 1, color=c) for c in colour_map.values()]
    ax.legend(handles, list(colour_map.keys()), fontsize=7,
              loc="lower right", ncol=2, framealpha=0.9)
    for bar, val in zip(bars, top["elo"]):
        ax.text(val + 4, bar.get_y() + bar.get_height() / 2,
                f"{val:.0f}", va="center", fontsize=7)
    ax.grid(axis="x", alpha=0.3)


def _plot_elo_by_league(ax, ratings, primary_map, top_n_leagues=15, min_teams=5):
    """Boxplot of Elo ratings per league for leagues with enough teams."""
    rows = []
    for tid, r in ratings.items():
        lid, lname = primary_map.get(tid, (0, "?"))
        rows.append({"team_id": tid, "elo": r, "league_id": lid, "league_name": lname})
    df = pd.DataFrame(rows)
    counts = df.groupby("league_name").size()
    eligible = counts[counts >= min_teams].index
    df = df[df["league_name"].isin(eligible)]
    medians = df.groupby("league_name")["elo"].median().sort_values(ascending=False)
    medians = medians.head(top_n_leagues)
    df = df[df["league_name"].isin(medians.index)]

    data = [df.loc[df["league_name"] == lg, "elo"].values for lg in medians.index]
    bp = ax.boxplot(data, tick_labels=medians.index, vert=False,
                    showfliers=False, patch_artist=True,
                    medianprops={"color": "black"})
    for patch in bp["boxes"]:
        patch.set_facecolor("#3C6E71")
    ax.invert_yaxis()
    ax.set_xlabel("Elo rating")
    ax.set_title(f"Elo distribution per league (top {top_n_leagues} by median)")
    ax.axvline(DEFAULT_BASE_RATING, color="grey", linestyle="--",
               alpha=0.5, label="base rating")
    ax.legend(fontsize=8)
    ax.grid(axis="x", alpha=0.3)


def _plot_promotion_relegation(ax, pr_df, names):
    """2D density of (from_win_rate, to_win_rate) for teams who switched leagues.

    Below the y=x diagonal: win rate dropped after the move (likely a step up).
    Above the y=x diagonal: win rate rose (likely a step down).
    """
    if pr_df.empty:
        ax.text(0.5, 0.5, "No promotion/relegation pairs found",
                ha="center", va="center", transform=ax.transAxes)
        ax.set_axis_off()
        return

    pr_df = pr_df.copy()
    pr_df["delta"] = pr_df["to_win_rate"] - pr_df["from_win_rate"]
    promoted = (pr_df["delta"] < -0.05).sum()
    relegated = (pr_df["delta"] > 0.05).sum()
    flat = len(pr_df) - promoted - relegated

    hb = ax.hexbin(
        pr_df["from_win_rate"], pr_df["to_win_rate"],
        gridsize=22, cmap="viridis", mincnt=1, edgecolors="none",
    )
    cb = plt.colorbar(hb, ax=ax, label="teams in bin", shrink=0.85)
    cb.outline.set_visible(False)

    # y=x reference line (no win-rate change).
    ax.plot([0, 1], [0, 1], color="white", linestyle="--", alpha=0.85, linewidth=1.5)

    # Mean shift (centroid of all moves) — single data point summary.
    mean_from = pr_df["from_win_rate"].mean()
    mean_to = pr_df["to_win_rate"].mean()
    ax.scatter([mean_from], [mean_to], s=140, color="#E63946",
               edgecolors="white", linewidths=2, zorder=5,
               label=f"Mean shift: {mean_from:.2f} → {mean_to:.2f} "
                     f"(Δ = {mean_to - mean_from:+.2f})")

    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_xlabel("Win rate in old league (season N)")
    ax.set_ylabel("Win rate in new league (season N+1)")
    ax.set_title(
        f"Promotion / relegation density — {len(pr_df)} pairs\n"
        f"(below dashed = harder league after move, above = easier; "
        f"{promoted} stepped up, {relegated} stepped down, {flat} flat)"
    )
    ax.legend(loc="upper left", fontsize=8, framealpha=0.85)


def _plot_league_strength_ladder(
    ax, ratings, primary_map, league_elo=None, top_n_leagues=20, min_teams=5,
):
    """Median team Elo per league + (optionally) overlay computed league Elo.

    Median team Elo: aggregate of the team-level walk (one rating per team).
    League Elo: derived purely from inter-league matches (separate metric).
    Plotting both reveals where they agree (good) and where they diverge
    (interesting — usually a sample-size / connectivity issue).
    """
    rows = []
    for tid, r in ratings.items():
        lid, lname = primary_map.get(tid, (0, "?"))
        rows.append({"elo": r, "league_id": lid, "league_name": lname})
    df = pd.DataFrame(rows)
    counts = df.groupby("league_name").size().rename("n_teams")
    eligible = counts[counts >= min_teams].index
    df = df[df["league_name"].isin(eligible)]
    summary = (
        df.groupby(["league_name", "league_id"])["elo"]
        .agg(["median", "min", "max", "count"])
        .reset_index()
        .sort_values("median", ascending=True)
    ).tail(top_n_leagues).reset_index(drop=True)

    y = np.arange(len(summary))
    ax.errorbar(
        summary["median"], y,
        xerr=[summary["median"] - summary["min"], summary["max"] - summary["median"]],
        fmt="o", color="#264653", ecolor="#A8DADC", elinewidth=3, capsize=4,
        markersize=8, label="median team Elo (with min↔max range)",
    )

    if league_elo is not None:
        le_vals = [league_elo.get(int(row["league_id"]), np.nan)
                   for _, row in summary.iterrows()]
        ax.scatter(le_vals, y, marker="D", s=70, color="#E63946",
                   edgecolors="white", linewidths=1.2, zorder=5,
                   label="league Elo (from inter-league matches)")

    ax.set_yticks(y, summary["league_name"])
    ax.set_xlabel("Elo rating")
    ax.set_title(f"League strength ladder (top {top_n_leagues} by median team Elo)")
    ax.axvline(DEFAULT_BASE_RATING, color="grey", linestyle="--", alpha=0.5)
    for i, row in summary.iterrows():
        ax.text(row["max"] + 10, i, f" n={int(row['count'])}",
                va="center", fontsize=7, color="grey")
    ax.legend(loc="lower right", fontsize=8, framealpha=0.85)
    ax.grid(axis="x", alpha=0.3)


def _plot_league_elo_bar(ax, league_elo, n_matches, league_name_lookup, top_n=20):
    """Standalone bar chart of computed league Elo, top N by rating."""
    rows = []
    for lid, elo in league_elo.items():
        n = n_matches.get(lid, 0)
        if n < 30:  # filter low-sample leagues for stability
            continue
        rows.append({
            "league_id": lid, "league_name": league_name_lookup.get(lid, f"#{lid}"),
            "elo": elo, "n_matches": n,
        })
    df = pd.DataFrame(rows).nlargest(top_n, "elo").iloc[::-1].reset_index(drop=True)

    y = np.arange(len(df))
    cmap = plt.colormaps.get_cmap("plasma")
    norm = plt.Normalize(df["n_matches"].min(), df["n_matches"].max())
    colours = cmap(norm(df["n_matches"].values))
    ax.barh(y, df["elo"], color=colours)
    ax.set_yticks(y, df["league_name"])
    ax.set_xlabel("League Elo (inter-league matches only)")
    ax.set_title(
        f"League Elo from inter-league play\n"
        f"(top {top_n}, ≥30 inter-league matches; bar colour = sample size)"
    )
    ax.axvline(DEFAULT_BASE_RATING, color="grey", linestyle="--", alpha=0.5)
    for i, row in df.iterrows():
        ax.text(row["elo"] + 4, i, f"{row['elo']:.0f}  (n={row['n_matches']})",
                va="center", fontsize=7)
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    plt.colorbar(sm, ax=ax, label="# inter-league matches", shrink=0.7)
    ax.grid(axis="x", alpha=0.3)


def _plot_seed_vs_computed(ax, league_elo, n_matches, league_name_lookup):
    """Scatter: hardcoded LEAGUE_INITIAL_RATING vs computed league Elo.

    Point size scales with inter-league match count (= our confidence in the
    computed value). Far-from-diagonal points are where my hardcoded prior
    disagrees most with the data.
    """
    rows = []
    for lid, computed in league_elo.items():
        hardcoded = LEAGUE_INITIAL_RATING.get(lid)
        n = n_matches.get(lid, 0)
        if hardcoded is None or n < 30:
            continue
        rows.append({
            "league_id": lid, "league_name": league_name_lookup.get(lid, f"#{lid}"),
            "hardcoded": hardcoded, "computed": computed, "n_matches": n,
        })
    df = pd.DataFrame(rows)
    if df.empty:
        ax.text(0.5, 0.5, "No leagues with both hardcoded & sufficient data",
                ha="center", va="center", transform=ax.transAxes)
        ax.set_axis_off()
        return

    sizes = np.clip(df["n_matches"] / df["n_matches"].max() * 250, 30, 250)
    sc = ax.scatter(
        df["hardcoded"], df["computed"], s=sizes, alpha=0.7,
        c=df["n_matches"], cmap="plasma",
        edgecolors="black", linewidths=0.5,
    )

    lo = min(df["hardcoded"].min(), df["computed"].min()) - 50
    hi = max(df["hardcoded"].max(), df["computed"].max()) + 50
    ax.plot([lo, hi], [lo, hi], color="grey", linestyle="--", alpha=0.7,
            label="hardcoded = computed")

    # Label the largest disagreements (top 8 by abs difference)
    df["delta"] = (df["computed"] - df["hardcoded"]).abs()
    for _, row in df.nlargest(8, "delta").iterrows():
        ax.annotate(
            row["league_name"],
            (row["hardcoded"], row["computed"]),
            xytext=(6, 6), textcoords="offset points",
            fontsize=7, alpha=0.85,
        )

    ax.set_xlabel("Hardcoded LEAGUE_INITIAL_RATING (my priors)")
    ax.set_ylabel("Computed league Elo (data-driven)")
    ax.set_title(
        f"Hardcoded prior vs data-driven league Elo\n"
        f"({len(df)} leagues with ≥30 inter-league matches)"
    )
    plt.colorbar(sc, ax=ax, label="# inter-league matches", shrink=0.8)
    ax.legend(loc="upper left", fontsize=9)
    ax.grid(alpha=0.3)


def main():
    import argparse

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    parser = argparse.ArgumentParser(description="Team strength figures")
    parser.add_argument(
        "--seeding", choices=["hardcoded", "league_elo", "uniform"],
        default="hardcoded",
        help="How to seed initial team Elos before the walk. "
             "'hardcoded' uses my LEAGUE_INITIAL_RATING dict (default); "
             "'league_elo' uses data-driven league Elo from inter-league matches; "
             "'uniform' starts every team at 1500.",
    )
    parser.add_argument(
        "--out", default=None,
        help=f"Output PNG path (default: {_FIGURES_DIR}/strength_<seeding>.png)",
    )
    args = parser.parse_args()

    df = load_fixtures()
    logger.info("Loaded %d fixtures", len(df))

    df_elo, ratings = compute_elo_ratings(df, seeding=args.seeding)
    logger.info("Computed team Elo for %d teams (seeding=%s)", len(ratings), args.seeding)

    league_elo, n_matches = compute_league_elo(df)
    logger.info("Computed league Elo for %d leagues", len(league_elo))

    primary_map = primary_league_map(df)
    names = team_name_map(df)
    league_name_lookup = dict(df.groupby("league_id")["league_name"].first().items())
    pr_df = promotion_relegation_pairs(df)
    logger.info("Found %d promotion/relegation pairs", len(pr_df))

    fig, axes = plt.subplots(2, 3, figsize=(28, 14), constrained_layout=True)
    fig.suptitle(
        f"Team strength — {len(df):,} matches, {len(ratings):,} teams, "
        f"Elo (k={DEFAULT_K}, home advantage={DEFAULT_HOME_ADVANTAGE}, "
        f"seeding={args.seeding})",
        fontsize=15, fontweight="bold",
    )
    _plot_top_teams(axes[0, 0], ratings, names, primary_map)
    _plot_elo_by_league(axes[0, 1], ratings, primary_map)
    _plot_league_elo_bar(axes[0, 2], league_elo, n_matches, league_name_lookup)
    _plot_promotion_relegation(axes[1, 0], pr_df, names)
    _plot_league_strength_ladder(axes[1, 1], ratings, primary_map, league_elo=league_elo)
    _plot_seed_vs_computed(axes[1, 2], league_elo, n_matches, league_name_lookup)

    if args.out:
        out = Path(args.out)
    else:
        out = _FIGURES_DIR / f"strength_{args.seeding}.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=140, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
