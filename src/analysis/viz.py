"""
Visualisation helpers for outcome-uncertainty analysis.

Each function takes a pre-computed DataFrame (output of metrics.compute_all())
and returns a matplotlib Figure, which can be shown interactively with
plt.show() or saved with fig.savefig("path.png").

All functions degrade gracefully when odds data is absent — they render an
informative placeholder rather than crashing, so the example script works
even before odds have been collected.

MATPLOTLIB PRIMER
=================
matplotlib works in two layers:

  Figure  — the overall canvas. Created with plt.figure() or plt.subplots().
  Axes    — an individual plot panel within a Figure. A Figure can hold many.

We always create explicit Figure/Axes objects (the "object-oriented" style)
rather than using pyplot's implicit "current axes" state. This makes it easy
to embed plots in notebooks, save them programmatically, or arrange multiple
panels side by side without surprises.

  fig, ax = plt.subplots()      # one panel
  fig, axes = plt.subplots(1, 2)  # two panels side by side
"""

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import pandas as pd


def _no_data_figure(message: str) -> plt.Figure:
    """Return a blank figure with an explanatory message instead of crashing."""
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.text(0.5, 0.5, message, ha="center", va="center", fontsize=12, color="#888888",
            transform=ax.transAxes, wrap=True)
    ax.axis("off")
    fig.tight_layout()
    return fig


def plot_results_breakdown(df: pd.DataFrame, title: str = None) -> plt.Figure:
    """
    Bar chart showing the split of home wins, draws, and away wins.

    Requires only fixture data — no odds needed.

    Parameters
    ----------
    df : pd.DataFrame
        Must contain home_goals and away_goals columns.

    Returns
    -------
    matplotlib.figure.Figure
    """
    valid = df.dropna(subset=["home_goals", "away_goals"])

    if valid.empty:
        return _no_data_figure("No fixture data available.")

    home_wins = (valid["home_goals"] > valid["away_goals"]).sum()
    draws = (valid["home_goals"] == valid["away_goals"]).sum()
    away_wins = (valid["home_goals"] < valid["away_goals"]).sum()
    total = len(valid)

    labels = ["Home Win", "Draw", "Away Win"]
    counts = [home_wins, draws, away_wins]
    colours = ["#4C72B0", "#8172B2", "#C44E52"]

    fig, ax = plt.subplots(figsize=(7, 5))
    bars = ax.bar(labels, counts, color=colours, edgecolor="white", linewidth=0.6, width=0.5)

    for bar, count in zip(bars, counts):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.5,
            f"{count} ({count/total*100:.1f}%)",
            ha="center", fontsize=11,
        )

    ax.set_ylabel("Number of Fixtures", fontsize=11)
    ax.set_title(title or "Result Breakdown", fontsize=13)
    ax.set_ylim(0, max(counts) * 1.15)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    return fig


def plot_goals_distribution(df: pd.DataFrame, title: str = None) -> plt.Figure:
    """
    Histogram of total goals per match.

    Requires only fixture data — no odds needed.

    Parameters
    ----------
    df : pd.DataFrame
        Must contain home_goals and away_goals columns.

    Returns
    -------
    matplotlib.figure.Figure
    """
    valid = df.dropna(subset=["home_goals", "away_goals"])

    if valid.empty:
        return _no_data_figure("No fixture data available.")

    total_goals = valid["home_goals"] + valid["away_goals"]
    max_goals = int(total_goals.max())

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.hist(
        total_goals,
        bins=range(0, max_goals + 2),
        color="#4C72B0", edgecolor="white", linewidth=0.6, align="left",
    )

    mean_goals = total_goals.mean()
    ax.axvline(mean_goals, color="#C44E52", linestyle="--", linewidth=1.2,
               label=f"Mean: {mean_goals:.2f}")

    ax.set_xlabel("Total Goals in Match", fontsize=11)
    ax.set_ylabel("Number of Fixtures", fontsize=11)
    ax.set_title(title or "Goals per Match Distribution", fontsize=13)
    ax.xaxis.set_major_locator(mticker.MultipleLocator(1))
    ax.legend(fontsize=10)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    return fig


def plot_upset_distribution(df: pd.DataFrame, title: str = None) -> plt.Figure:
    """
    Histogram of upset scores across all fixtures.

    Shows how often results matched expectations (score near 0) vs.
    how often genuine upsets occurred (score near 1).

    Parameters
    ----------
    df : pd.DataFrame
        Must contain an upset_score column (from metrics.compute_all()).
    title : str, optional
        Override the default plot title.

    Returns
    -------
    matplotlib.figure.Figure
    """
    # Drop rows where upset_score couldn't be computed (no odds data).
    # dropna() removes rows with NaN in the specified column.
    scores = df["upset_score"].dropna()

    if scores.empty:
        return _no_data_figure("No odds data available yet.\nRun the ingest before fixtures are played to capture pre-match odds.")

    fig, ax = plt.subplots(figsize=(9, 5))

    # bins=20 divides the [0, 1] range into 20 equal buckets.
    # edgecolor draws a thin border around each bar, making them easier to
    # distinguish when colours are similar.
    ax.hist(scores, bins=20, range=(0, 1), color="#4C72B0", edgecolor="white", linewidth=0.6)

    ax.set_xlabel("Upset Score  (0 = expected result, 1 = maximum surprise)", fontsize=11)
    ax.set_ylabel("Number of Fixtures", fontsize=11)
    ax.set_title(title or "Distribution of Upset Scores", fontsize=13)

    # Add a subtle vertical line at the mean so the "average surprise level"
    # is immediately visible on the chart.
    mean_score = scores.mean()
    ax.axvline(mean_score, color="#C44E52", linestyle="--", linewidth=1.2,
               label=f"Mean: {mean_score:.2f}")
    ax.legend(fontsize=10)

    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    return fig


def plot_upsets_over_time(df: pd.DataFrame, title: str = None) -> plt.Figure:
    """
    Rolling average of upset scores across matchdays, showing whether
    surprise results cluster in particular parts of the season.

    Parameters
    ----------
    df : pd.DataFrame
        Must contain date and upset_score columns.

    Returns
    -------
    matplotlib.figure.Figure
    """
    # Sort by date so the rolling window moves forward in time.
    # sort_values returns a new DataFrame; it doesn't modify df in place
    # unless you pass inplace=True (which we deliberately avoid).
    sorted_df = df.dropna(subset=["upset_score"]).sort_values("date")

    if sorted_df.empty:
        return _no_data_figure("No odds data available yet.\nRun the ingest before fixtures are played to capture pre-match odds.")

    fig, ax = plt.subplots(figsize=(11, 5))

    # Scatter every individual fixture as a semi-transparent dot.
    # alpha controls opacity (0 = invisible, 1 = opaque). Using a low alpha
    # lets overlapping points show density — a cluster of dots appears darker.
    ax.scatter(
        sorted_df["date"],
        sorted_df["upset_score"],
        alpha=0.25,
        s=18,
        color="#4C72B0",
        label="Individual fixture",
    )

    # rolling(window=10) creates a sliding window of 10 rows.
    # .mean() then computes the average within each window position.
    # The first 9 rows produce NaN (not enough preceding values), which
    # matplotlib simply skips when drawing the line.
    rolling_mean = sorted_df["upset_score"].rolling(window=10).mean()
    ax.plot(sorted_df["date"], rolling_mean, color="#C44E52", linewidth=2,
            label="10-match rolling mean")

    ax.set_xlabel("Date", fontsize=11)
    ax.set_ylabel("Upset Score", fontsize=11)
    ax.set_title(title or "Upset Score Over Time", fontsize=13)
    ax.legend(fontsize=10)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    return fig


def plot_upsets_by_venue(df: pd.DataFrame, top_n: int = 20, title: str = None) -> plt.Figure:
    """
    Horizontal bar chart of mean upset score by venue.

    Useful for spotting whether certain grounds produce disproportionately
    surprising results (poor pitch, unusual atmosphere, travel difficulty, etc.).

    Parameters
    ----------
    df : pd.DataFrame
        Must contain venue_name and upset_score columns.
    top_n : int
        How many venues to show (those with the highest mean upset score).

    Returns
    -------
    matplotlib.figure.Figure
    """
    # Filter out rows with no venue or no upset score.
    venue_df = df.dropna(subset=["venue_name", "upset_score"])

    if venue_df.empty:
        return _no_data_figure("No odds data available yet.\nRun the ingest before fixtures are played to capture pre-match odds.")

    # groupby + agg to get mean upset score and match count per venue.
    venue_stats = (
        venue_df.groupby("venue_name")
        .agg(
            mean_upset=("upset_score", "mean"),
            match_count=("upset_score", "count"),
        )
        # Only include venues with enough matches to be statistically meaningful.
        .query("match_count >= 5")
        .sort_values("mean_upset", ascending=False)
        .head(top_n)
        # Reverse so the highest bar appears at the top of the chart.
        .iloc[::-1]
    )

    if venue_stats.empty:
        return _no_data_figure("Not enough data per venue yet (minimum 5 matches required).")

    fig, ax = plt.subplots(figsize=(9, max(4, len(venue_stats) * 0.4)))

    bars = ax.barh(
        venue_stats.index,
        venue_stats["mean_upset"],
        color="#4C72B0",
        edgecolor="white",
        linewidth=0.5,
    )

    # Annotate each bar with the match count so the reader can judge reliability.
    for bar, count in zip(bars, venue_stats["match_count"]):
        ax.text(
            bar.get_width() + 0.002,
            bar.get_y() + bar.get_height() / 2,
            f"n={count}",
            va="center",
            fontsize=8,
            color="#555555",
        )

    ax.set_xlabel("Mean Upset Score", fontsize=11)
    ax.set_title(title or f"Top {top_n} Venues by Mean Upset Score", fontsize=13)
    ax.xaxis.set_major_formatter(mticker.FormatStrFormatter("%.2f"))
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    return fig


def plot_favourite_win_rate(df: pd.DataFrame, title: str = None) -> plt.Figure:
    """
    Bar chart showing what percentage of matches were won by the favourite,
    split by whether the favourite was the home team, draw, or away team.

    Parameters
    ----------
    df : pd.DataFrame
        Must contain favourite and outcome columns.

    Returns
    -------
    matplotlib.figure.Figure
    """
    valid = df.dropna(subset=["favourite", "outcome"])

    if valid.empty:
        return _no_data_figure("No odds data available yet.\nRun the ingest before fixtures are played to capture pre-match odds.")

    # A result is "as expected" when the favourite outcome actually occurred.
    # This creates a boolean column, then we compute the mean — because in
    # Python/pandas, True == 1 and False == 0, so mean() gives the proportion.
    valid = valid.copy()
    valid["favourite_won"] = valid["favourite"] == valid["outcome"]

    # Win rate per favourite type (home / draw / away).
    win_rates = (
        valid.groupby("favourite")["favourite_won"]
        .mean()
        .reindex(["home_win", "draw", "away_win"])  # consistent left-to-right order
        * 100  # convert proportion to percentage
    )

    fig, ax = plt.subplots(figsize=(7, 5))

    colours = ["#4C72B0", "#8172B2", "#C44E52"]
    bars = ax.bar(
        ["Home favourite", "Draw favourite", "Away favourite"],
        win_rates.fillna(0),
        color=colours,
        edgecolor="white",
        linewidth=0.6,
        width=0.5,
    )

    # Label each bar with its percentage value, skipping bars with no data.
    for bar, rate in zip(bars, win_rates):
        if pd.notna(rate):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.8,
                f"{rate:.1f}%",
                ha="center",
                fontsize=11,
            )

    ax.set_ylabel("Favourite Win Rate (%)", fontsize=11)
    ax.set_ylim(0, 100)
    ax.set_title(title or "How Often Does the Favourite Win?", fontsize=13)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    return fig
