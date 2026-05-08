"""End-to-end H/D/A modelling pipeline.

Loads fixtures, computes Elo with a leakage-free seeding (league seeds and
primary-league map come from PRE-cutoff data only), builds features for
the target league's matches, time-splits into train/test, fits the
climatology baseline + Elo baseline + logistic regression, then evaluates
all three on the test set with bootstrap CIs.

Usage:
    python -m src.model.run
    python -m src.model.run --target-league 39 --no-cups
    python -m src.model.run --boot 5000  # tighter CIs (slower)
"""

from __future__ import annotations

import argparse
import logging
import sys
from datetime import datetime

import pandas as pd

from src.analysis.strength import (
    DEFAULT_BASE_RATING,
    compute_elo_ratings,
    compute_league_elo,
    primary_league_map,
)
from src.model.baselines import ClimatologyBaseline, EloBaseline
from src.model.data import (
    DEFAULT_TARGET_LEAGUE,
    TOP5_EUROPEAN,
    TRAIN_CUTOFF,
    FilterConfig,
    SplitConfig,
    load_fixtures_for_elo,
    summarize,
    time_split,
)
from src.model.evaluate import evaluation_report
from src.model.features import build_features
from src.model.train import GBMModel, LogisticModel

logger = logging.getLogger(__name__)


def _format_metric(label: str, ci: dict) -> str:
    if "point" not in ci:
        return f"  {label:<14}{ci}"
    point = ci["point"]
    lo, hi = ci.get("lo", float("nan")), ci.get("hi", float("nan"))
    return f"  {label:<14}{point:7.4f}   95% CI [{lo:.4f}, {hi:.4f}]"


def _print_report(name: str, report: dict) -> None:
    print(f"\n=== {name} (n_test={report['n_test']}) ===")
    print(_format_metric("log-loss", report["log_loss"]))
    print(_format_metric("Brier",    report["brier"]))
    print(_format_metric("accuracy", report["accuracy"]))
    print(f"  ECE           {report['ece']:.4f}")
    tail = report["tail_brier"]
    print(f"  tail Brier    {tail['tail_brier']:.4f}   "
          f"(over {tail['n_cells']} prob<=0.10 cells)")


def run(
    target_league_ids: frozenset[int] = frozenset({DEFAULT_TARGET_LEAGUE}),
    include_cups: bool = False,
    cutoff: datetime = TRAIN_CUTOFF,
    n_boot: int = 1000,
) -> None:
    # ---- 1. Load Elo corpus -------------------------------------------------
    elo_corpus = load_fixtures_for_elo()
    logger.info("Elo corpus: %s", summarize(elo_corpus))

    # ---- 2. Leakage-free seeding from PRE-cutoff slice ---------------------
    pre = elo_corpus[elo_corpus["date"] < cutoff].reset_index(drop=True)
    logger.info("Pre-cutoff slice for seeding: %d matches", len(pre))
    league_seeds, _ = compute_league_elo(pre)
    pmap = primary_league_map(pre)
    team_seeds = {
        tid: league_seeds.get(lid, DEFAULT_BASE_RATING)
        for tid, (lid, _) in pmap.items()
    }
    logger.info("Computed %d team seeds from pre-cutoff data", len(team_seeds))

    # ---- 3. Walk team Elo through the FULL corpus using those seeds --------
    elo_with_pre, _ = compute_elo_ratings(elo_corpus, team_seeds=team_seeds)

    # ---- 4. Filter to target league(s) for feature building ----------------
    target = elo_with_pre[
        elo_with_pre["league_id"].isin(set(target_league_ids))
    ].reset_index(drop=True)
    if not include_cups and "is_cup" in target.columns:
        target = target[~target["is_cup"]].reset_index(drop=True)
    logger.info(
        "Target rows: %d across %d league(s) [%s]",
        len(target), target["league_id"].nunique(),
        ", ".join(str(x) for x in sorted(target["league_id"].unique())),
    )

    # ---- 5. Build features (leakage-safe by construction) ------------------
    features = build_features(target, elo_with_pre)
    logger.info("Features built: %d rows × %d cols", *features.shape)

    # ---- 6. Time-based train/test split ------------------------------------
    train, test = time_split(features, SplitConfig(cutoff=cutoff))
    print(f"\nTrain: {len(train)} matches "
          f"({train['date'].min().date()} -> {train['date'].max().date()})")
    print(f"Test:  {len(test)} matches "
          f"({test['date'].min().date()} -> {test['date'].max().date()})")
    print(f"Train H/D/A rate: "
          f"{(train['result']=='H').mean():.2%} / "
          f"{(train['result']=='D').mean():.2%} / "
          f"{(train['result']=='A').mean():.2%}")
    print(f"Test  H/D/A rate: "
          f"{(test['result']=='H').mean():.2%} / "
          f"{(test['result']=='D').mean():.2%} / "
          f"{(test['result']=='A').mean():.2%}")

    y_train = train["result"]
    y_test = test["result"]

    # ---- 7. Fit + evaluate baselines ---------------------------------------
    clim = ClimatologyBaseline().fit(train, y_train)
    clim_probs = clim.predict_proba(test)
    _print_report("Climatology baseline", evaluation_report(y_test, clim_probs, n_boot=n_boot))

    elo = EloBaseline().fit(train, y_train)
    elo_probs = elo.predict_proba(test)
    _print_report("Elo baseline", evaluation_report(y_test, elo_probs, n_boot=n_boot))

    # ---- 8. Fit + evaluate logistic regression -----------------------------
    lr = LogisticModel().fit(train, y_train)
    lr_probs = lr.predict_proba(test)
    _print_report("Logistic regression v2", evaluation_report(y_test, lr_probs, n_boot=n_boot))

    # ---- 9. Fit + evaluate GBM (LightGBM) ---------------------------------
    gbm = GBMModel().fit(train, y_train)
    gbm_probs = gbm.predict_proba(test)
    _print_report("Gradient-boosted trees", evaluation_report(y_test, gbm_probs, n_boot=n_boot))

    # ---- Diagnostics ------------------------------------------------------
    fi_lr = lr.feature_importance_
    if fi_lr is not None:
        print("\nLR — top 10 features by |coef_H| (positive = pushes prob toward home win):")
        top = fi_lr.assign(abs_H=fi_lr["coef_H"].abs()).sort_values("abs_H", ascending=False).head(10)
        print(top[["coef_H", "coef_D", "coef_A"]].to_string(float_format=lambda x: f"{x:+.3f}"))

    fi_gbm = gbm.feature_importance_
    if fi_gbm is not None:
        print("\nGBM — top 10 features by gain (total loss reduction across splits):")
        print(fi_gbm.head(10).to_string(index=False))


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        stream=sys.stdout,
    )
    parser = argparse.ArgumentParser(description="Run the H/D/A modelling pipeline")
    parser.add_argument(
        "--target-leagues", type=str, default=str(DEFAULT_TARGET_LEAGUE),
        help="Comma-separated league IDs, or the literal 'top5' for the "
             "top-5 European first tiers (PL, La Liga, Serie A, Bundesliga, "
             "Ligue 1). Default: 39 (Premier League only).",
    )
    parser.add_argument("--include-cups", action="store_true",
                        help="Include cup matches in target (default: exclude)")
    parser.add_argument("--cutoff", type=str, default=TRAIN_CUTOFF.date().isoformat(),
                        help=f"Train/test cutoff date YYYY-MM-DD (default: {TRAIN_CUTOFF.date()})")
    parser.add_argument("--boot", type=int, default=1000,
                        help="Bootstrap iterations for CI (default: 1000)")
    args = parser.parse_args()

    cutoff = datetime.fromisoformat(args.cutoff)
    if args.target_leagues.strip().lower() == "top5":
        league_ids = TOP5_EUROPEAN
    else:
        league_ids = frozenset(int(s.strip()) for s in args.target_leagues.split(","))

    run(
        target_league_ids=league_ids,
        include_cups=args.include_cups,
        cutoff=cutoff,
        n_boot=args.boot,
    )


if __name__ == "__main__":
    main()
