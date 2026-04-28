"""SQLite schema initialisation for sporting results."""

import sqlite3
from pathlib import Path

DB_PATH = Path(__file__).parents[2] / "data" / "results.db"


def get_connection(db_path: Path = DB_PATH) -> sqlite3.Connection:
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    return conn


# Priority leagues. Lower priority number = backfilled first. Leagues not
# listed here are still discovered and ingested, just after these have all
# completed (with a fallback ORDER BY league_id).
_DEFAULT_TRACKED_LEAGUES = [
    # (league_id, name, priority)
    (39,  "Premier League", 1),
    (40,  "Championship",   2),
    (140, "La Liga",        3),
    (78,  "Bundesliga",     4),
    (135, "Serie A",        5),
    (61,  "Ligue 1",        6),
    (203, "Süper Lig",      7),
]


def _column_exists(conn: sqlite3.Connection, table: str, column: str) -> bool:
    rows = conn.execute(f"PRAGMA table_info({table})").fetchall()
    return any(r["name"] == column for r in rows)


def init_db(db_path: Path = DB_PATH) -> None:
    db_path.parent.mkdir(parents=True, exist_ok=True)
    with get_connection(db_path) as conn:
        conn.executescript("""
            CREATE TABLE IF NOT EXISTS fixtures (
                fixture_id        INTEGER PRIMARY KEY,
                date              TEXT NOT NULL,
                league_id         INTEGER NOT NULL,
                league_name       TEXT NOT NULL,
                season            INTEGER NOT NULL,
                round             TEXT,
                home_team_id      INTEGER NOT NULL,
                home_team_name    TEXT NOT NULL,
                away_team_id      INTEGER NOT NULL,
                away_team_name    TEXT NOT NULL,
                home_goals        INTEGER,
                away_goals        INTEGER,
                home_goals_ht     INTEGER,
                away_goals_ht     INTEGER,
                status            TEXT NOT NULL,
                venue_name        TEXT,
                venue_city        TEXT,
                referee           TEXT,
                odds_unavailable  INTEGER NOT NULL DEFAULT 0,
                ingested_at       TEXT NOT NULL DEFAULT (datetime('now'))
            );

            CREATE TABLE IF NOT EXISTS ingest_runs (
                id              INTEGER PRIMARY KEY AUTOINCREMENT,
                league_id       INTEGER NOT NULL,
                season          INTEGER NOT NULL,
                fixtures_saved  INTEGER NOT NULL,
                ran_at          TEXT NOT NULL DEFAULT (datetime('now'))
            );

            CREATE TABLE IF NOT EXISTS tracked_leagues (
                league_id  INTEGER PRIMARY KEY,
                name       TEXT NOT NULL,
                priority   INTEGER,
                active     INTEGER NOT NULL DEFAULT 1
            );

            CREATE TABLE IF NOT EXISTS backfill_jobs (
                league_id      INTEGER NOT NULL,
                season         INTEGER NOT NULL,
                status         TEXT NOT NULL DEFAULT 'pending',
                fixtures_saved INTEGER,
                last_attempt   TEXT,
                last_error     TEXT,
                PRIMARY KEY (league_id, season)
            );

            CREATE INDEX IF NOT EXISTS idx_backfill_jobs_status ON backfill_jobs(status);

            CREATE TABLE IF NOT EXISTS odds (
                id             INTEGER PRIMARY KEY AUTOINCREMENT,
                fixture_id     INTEGER NOT NULL REFERENCES fixtures(fixture_id),
                bookmaker_id   INTEGER NOT NULL,
                bookmaker_name TEXT NOT NULL,
                bet_id         INTEGER NOT NULL,
                bet_name       TEXT NOT NULL,
                value_label    TEXT NOT NULL,
                odd            REAL NOT NULL,
                ingested_at    TEXT NOT NULL DEFAULT (datetime('now')),
                UNIQUE(fixture_id, bookmaker_id, bet_id, value_label)
            );

            CREATE INDEX IF NOT EXISTS idx_fixtures_date ON fixtures(date);
            CREATE INDEX IF NOT EXISTS idx_fixtures_status_date ON fixtures(status, date);
            CREATE INDEX IF NOT EXISTS idx_odds_fixture ON odds(fixture_id);
        """)

        # Migration: add odds_unavailable column to pre-existing fixtures table.
        if not _column_exists(conn, "fixtures", "odds_unavailable"):
            conn.execute(
                "ALTER TABLE fixtures ADD COLUMN odds_unavailable INTEGER NOT NULL DEFAULT 0"
            )
        # Migration: add priority column to pre-existing tracked_leagues table.
        if not _column_exists(conn, "tracked_leagues", "priority"):
            conn.execute("ALTER TABLE tracked_leagues ADD COLUMN priority INTEGER")

        # Seed tracked leagues idempotently. Use UPSERT so priorities update
        # if the seed list changes between runs (e.g. user reorders leagues).
        conn.executemany(
            """
            INSERT INTO tracked_leagues (league_id, name, priority)
            VALUES (?, ?, ?)
            ON CONFLICT(league_id) DO UPDATE SET
                name     = excluded.name,
                priority = excluded.priority
            """,
            _DEFAULT_TRACKED_LEAGUES,
        )
