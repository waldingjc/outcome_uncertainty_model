# outcome_uncertainty_model

Using historic sporting event data to identify and analyse potential low-probability events and their shared characteristics.

## What this is

A pipeline that ingests historical football fixture data (from [api-football.com](https://www.api-football.com/)) into a local SQLite database, generates exploratory analyses on top of it, and — eventually — feeds a model that quantifies *outcome uncertainty*: how confidently a result can be predicted given everything we know going in.

The medium-term goal is to surface matches where the favourite is meaningfully *more* uncertain than the bookmakers' price implies, by building a model with calibration that beats the H/D/A market on its own terms across a controlled basket of leagues.

The short-term goal — and where the project currently sits — is **finishing the historical-fixture sweep** so the model has a clean training corpus to work with.

## Current state

| | |
|---|---|
| Plan | api-football Free (100 calls/day, ~10 calls/min, no odds, no current season) |
| Coverage so far | ~100K fixtures, 144 leagues, 5K+ teams, seasons 2022 / 2023 / 2024 (= 2022-23, 2023-24, 2024-25 football seasons) |
| Backfill queue | ~390 / 2,863 (league × season) combos done; remainder draining at ~95/day |
| Live ingestion (current season) | **Not yet** — Free plan blocks it. Awaits a Pro upgrade. |
| Odds | **Not yet** — same reason; closing odds also have a 7-day retention window on api-football, so live ingestion is a prerequisite. |

The Free plan, surprisingly, has granted *every league we've tested so far* — no `no_access` rejections in 388 jobs. The drag is purely the daily 100-call cap, which the scheduled task (below) drains incrementally.

## Architecture

### Data flow

```
api-football.com  →  src/ingest/historical_backfill.py  →  data/results.db (SQLite)
                                                              ↓
                                                       src/analysis/*.py
                                                              ↓
                                                       data/figures/*.png
```

### Layout

| Path | Purpose |
|---|---|
| `src/db/schema.py` | SQLite schema + idempotent migrations |
| `src/config.py` | Zero-dependency `.env` loader (shell env wins on conflict) |
| `src/ingest/api_football.py` | Low-level HTTP client (auto-detects RapidAPI vs direct host, handles rate-limit headers, surfaces api-football's `errors` body) |
| `src/ingest/discover_leagues.py` | One-shot `/leagues` call → seeds the `backfill_jobs` work queue |
| `src/ingest/historical_backfill.py` | Queue-driven runner; pre-flight `/status` check; budget-aware |
| `src/ingest/odds.py` | Per-fixture odds fetch + flatten + UPSERT (idle until paid plan) |
| `src/ingest/daily_runner.py` | Recent-fixture refresh + odds backfill (idle until paid plan) |
| `src/analysis/plots.py` | Cross-league overview (entropy, H/D/A, goals, HT flips) |
| `src/analysis/team_breakdown.py` | Per-team 6-panel figure (substring or `team_id` lookup) |
| `src/analysis/strength.py` | Elo ratings, league strength ladders, promotion/relegation hexbin |
| `src/analysis/uncertainty.py` | Upset rate by league, Elo calibration curves, in-season form arcs |
| `src/analysis/exploration.py` | Records (top scores, biggest upsets, longest unbeaten runs, etc.) |
| `src/analysis/pyramid.py` | Country pyramid silhouettes + tier-gap metrics |
| `scripts/run-backfill.ps1` | Wrapper invoked by Task Scheduler; logs + Windows-toast summary |
| `scripts/install-backfill-task.ps1` | One-shot scheduler registration (10-min post-logon trigger) |
| `scripts/uninstall-backfill-task.ps1` | Removes the scheduled task |

### Schema (high level)

- `fixtures` — one row per match (date, league, teams, scores, HT scores, venue, referee, `odds_unavailable` flag, `ingested_at`)
- `odds` — per-bookmaker × per-bet-type × per-value odds, FK'd to fixtures (currently unused on Free plan)
- `backfill_jobs` — work queue: `(league_id, season)` combos with status `pending|completed|no_access|failed`
- `tracked_leagues` — priority-ordered league list for the daily runner (Premier, Championship, La Liga, Bundesliga, Serie A, Ligue 1, Süper Lig)
- `ingest_runs` — per-run audit log

## Setup

```powershell
# 1. Install Python deps
python -m pip install -r requirements.txt

# 2. Create .env with your api-football key (gitignored)
#    NOTE: direct sign-up at api-football.com → host = v3.football.api-sports.io
#    NOTE: RapidAPI sign-up → host = api-football-v1.p.rapidapi.com
#    The host is auto-detected; you only need to override via API_FOOTBALL_HOST if using RapidAPI.
echo API_FOOTBALL_KEY=your-key-here > .env

# 3. Initialise the DB (creates results.db + tables; idempotent)
python -c "from src.db.schema import init_db; init_db()"

# 4. Seed the backfill queue (one /leagues call; safe to re-run)
python -m src.ingest.discover_leagues

# 5. Run the backfill until daily quota is exhausted
python -m src.ingest.historical_backfill

# 6. (Optional) install the scheduler so step 5 fires automatically
scripts\install-backfill-task.ps1
```

## Console cheat sheet

All commands run from the repo root. Output figures land in `data/figures/`.

### Top-level overviews

```powershell
# 4-panel cross-league overview (entropy / H-D-A / goals / HT flips)
python -m src.analysis.plots
```

### Team strength (Elo + ladders + pyramid hexbin)

```powershell
# Default: hardcoded league seeds (curated priors)
python -m src.analysis.strength

# Data-driven seeds (computed league Elo from inter-league matches)
python -m src.analysis.strength --seeding league_elo

# Sanity check — every team starts at 1500 (you'll see Macclesfield-style pathologies)
python -m src.analysis.strength --seeding uniform

# Custom output filename
python -m src.analysis.strength --seeding league_elo --out data/figures/my_strength.png
```

### Outcome uncertainty (Elo-aware upsets / form / predictability)

```powershell
python -m src.analysis.uncertainty
python -m src.analysis.uncertainty --seeding league_elo
```

### Exploration & records (top scorers, biggest upsets, longest streaks, etc.)

```powershell
python -m src.analysis.exploration
python -m src.analysis.exploration --seeding league_elo
```

### Country pyramid analysis

```powershell
# Generates pyramid.png (per-country silhouettes) + pyramid_gaps.png (cross-country metrics)
python -m src.analysis.pyramid
```

### Per-team breakdown

```powershell
# Substring match (case- and diacritic-insensitive; picks most-played team if ambiguous)
python -m src.analysis.team_breakdown --team Fenerbahce
python -m src.analysis.team_breakdown --team "Real Madrid"
python -m src.analysis.team_breakdown --team Galatasaray

# Disambiguate by team_id (api-football's id, persistent across runs)
python -m src.analysis.team_breakdown --team 611    # Fenerbahce
python -m src.analysis.team_breakdown --team 541    # Real Madrid CF
python -m src.analysis.team_breakdown --team 50     # Manchester City
python -m src.analysis.team_breakdown --team 47     # Tottenham

# Custom output path
python -m src.analysis.team_breakdown --team Liverpool --out data/figures/lfc.png
```

### Ingestion (manual)

```powershell
# One-shot discovery (only needed once; idempotent)
python -m src.ingest.discover_leagues

# Daily backfill — chews through up to ~95 calls of pending jobs, then exits
python -m src.ingest.historical_backfill

# Pull odds for a single fixture (post-upgrade; currently blocked)
python -m src.ingest.odds --fixture 1208021

# Per-day refresh + odds backfill (post-upgrade; currently blocked)
python -m src.ingest.daily_runner
```

### Scheduler (Windows Task Scheduler)

```powershell
# Install: registers OutcomeUncertaintyBackfill task, fires 10 min after logon
scripts\install-backfill-task.ps1

# Run it manually once (smoke test)
Start-ScheduledTask -TaskName OutcomeUncertaintyBackfill

# Inspect last run
Get-ScheduledTaskInfo -TaskName OutcomeUncertaintyBackfill

# Logs and notifications
#   data\logs\backfill-<date>.log    full per-day log
#   data\logs\last-run.json          machine-readable last-run summary
#   data\logs\last-run.txt           human-readable summary (toast fallback)

# Remove
scripts\uninstall-backfill-task.ps1
```

### Useful one-liners

```powershell
# Backfill queue status
python -c "from src.db.schema import get_connection; conn = get_connection(); print({r['status']: r['n'] for r in conn.execute('SELECT status, COUNT(*) AS n FROM backfill_jobs GROUP BY status')})"

# Top 10 teams by current Elo (hardcoded seeding)
python -c "from src.analysis.strength import *; df=load_fixtures(); _, r=compute_elo_ratings(df); n=team_name_map(df); [print(f'{n[t]:<28}{v:7.1f}') for t,v in sorted(r.items(),key=lambda kv:-kv[1])[:10]]"

# Same with data-driven seeding
python -c "from src.analysis.strength import *; df=load_fixtures(); _, r=compute_elo_ratings(df, seeding='league_elo'); n=team_name_map(df); [print(f'{n[t]:<28}{v:7.1f}') for t,v in sorted(r.items(),key=lambda kv:-kv[1])[:10]]"

# Cumulative DB stats
python -c "from src.db.schema import get_connection; c=get_connection(); print('fixtures:',c.execute('SELECT COUNT(*) FROM fixtures').fetchone()[0]); print('leagues:',c.execute('SELECT COUNT(DISTINCT league_id) FROM fixtures').fetchone()[0])"
```

## Plan moving forward

Roughly in order of when each unlocks:

1. **Drain the backfill queue** — ~25 more days of scheduled runs before the queue's empty. No model work blocked on this; the data we already have (Premier League, Championship, all top-5 European tiers) is enough to start prototyping.

2. **Prototype an outcome model on existing data.** Logistic regression / ordinal model first; gradient-boosted second. Inputs = team Elo (both seedings), recent form, home advantage, league prior. Output = (P(home), P(draw), P(away)) per fixture. Calibration via reliability diagrams. *No odds needed for this step* — we're just trying to predict outcomes accurately.

3. **Sanity-check the model on holdout matches** — predict 2024-25 from a model trained on 2022-23 + 2023-24. Compare log-loss against a simple baseline (home rate / draw rate / away rate per league). The whole project is moot if we can't beat that.

4. **Upgrade to a paid api-football plan** once the model shape feels promising. Pro tier unlocks current-season fixtures + odds + much higher daily quota. At that point:
   - `daily_runner.py` activates: refreshes recent fixtures, captures closing odds within the 7-day retention window
   - The `odds` table starts filling
   - We can compare model probabilities against bookmaker prices

5. **The actual project goal:** identify low-probability outcomes the market underweights. With historical odds in hand, run the trained model on every fixture, compute model-implied probability vs market-implied probability, look for systematic discrepancies. Specifically interested in cases where the model is *less confident* than the market in obvious-favourite scenarios — markets tend to underprice draws and underdog wins in lopsided fixtures, and outcome uncertainty isn't a single number; it's a shape.

6. **Out of scope for now (but tempting):** in-play / live data, xG / shot-level data (api-football has it on higher tiers), expanding to other sports. None of these unlock until the football model proves itself.

## Notes / known wrinkles

- api-football's Free plan **does not return odds** and **does not return current-season data**. Both are gated to paid tiers. The full ingestion infrastructure for both is built and idle.
- api-football returns **HTTP 200 with embedded `errors` bodies** on plan/parameter issues — the client surfaces these as `WARNING` log lines but doesn't raise.
- **Disconnected league sub-graphs** (e.g. Tier 7 non-league teams who never play above Tier 5) caused early-pass Elo pathologies. Mitigated via league-aware initial ratings (`LEAGUE_INITIAL_RATING` in `strength.py`) — see commit history for details.
- The local SQLite (`data/results.db`) and raw JSON dumps (`data/raw/`) are gitignored. To rebuild from scratch on a fresh checkout: run `init_db()` → `discover_leagues` → `historical_backfill` (will take ~25 days on Free).
