"""
Quick example: load 2024/25 Premier League data and produce all four plots.

Run from the project root:
    python -m src.analysis.example
"""

import matplotlib.pyplot as plt

from src.analysis import loader, metrics, viz

# Load fixtures joined with consensus odds (one row per match).
df = loader.load_fixtures_with_odds(league_id=39, season=2024)

# Enrich with implied probabilities, actual outcome, and upset score.
df = metrics.compute_all(df)

print(f"Loaded {len(df)} fixtures, {df['upset_score'].notna().sum()} with odds data.")
print(df[["home_team_name", "away_team_name", "home_goals", "away_goals", "upset_score"]]
      .sort_values("upset_score", ascending=False)
      .head(10)
      .to_string())

# Fixture-only plots — work without odds data.
viz.plot_results_breakdown(df)
viz.plot_goals_distribution(df)

# Odds-dependent plots — show placeholders until odds are collected.
viz.plot_upset_distribution(df)
viz.plot_upsets_over_time(df)
viz.plot_upsets_by_venue(df)
viz.plot_favourite_win_rate(df)

plt.show()
