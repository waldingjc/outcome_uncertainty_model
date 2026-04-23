"""SQLite schema initialisation for sporting results."""

import sqlite3
from pathlib import Path

DB_PATH = Path(__file__).parents[2] / "data" / "results.db"


def get_connection(db_path: Path = DB_PATH) -> sqlite3.Connection:
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    return conn


def init_db(db_path: Path = DB_PATH) -> None:
    db_path.parent.mkdir(parents=True, exist_ok=True)
    with get_connection(db_path) as conn:
        conn.executescript("""
            CREATE TABLE IF NOT EXISTS fixtures (
                fixture_id      INTEGER PRIMARY KEY,
                date            TEXT NOT NULL,
                league_id       INTEGER NOT NULL,
                league_name     TEXT NOT NULL,
                season          INTEGER NOT NULL,
                round           TEXT,
                home_team_id    INTEGER NOT NULL,
                home_team_name  TEXT NOT NULL,
                away_team_id    INTEGER NOT NULL,
                away_team_name  TEXT NOT NULL,
                home_goals      INTEGER,
                away_goals      INTEGER,
                home_goals_ht   INTEGER,
                away_goals_ht   INTEGER,
                status          TEXT NOT NULL,
                venue_name      TEXT,
                venue_city      TEXT,
                referee         TEXT,
                ingested_at     TEXT NOT NULL DEFAULT (datetime('now'))
            );

            CREATE TABLE IF NOT EXISTS odds (
                fixture_id      INTEGER NOT NULL,
                bookmaker_id    INTEGER NOT NULL,
                bookmaker_name  TEXT NOT NULL,
                home_odds       REAL,
                draw_odds       REAL,
                away_odds       REAL,
                ingested_at     TEXT NOT NULL DEFAULT (datetime('now')),
                PRIMARY KEY (fixture_id, bookmaker_id),
                FOREIGN KEY (fixture_id) REFERENCES fixtures(fixture_id)
            );

            CREATE TABLE IF NOT EXISTS ingest_runs (
                id              INTEGER PRIMARY KEY AUTOINCREMENT,
                league_id       INTEGER NOT NULL,
                season          INTEGER NOT NULL,
                fixtures_saved  INTEGER NOT NULL,
                odds_saved      INTEGER NOT NULL DEFAULT 0,
                ran_at          TEXT NOT NULL DEFAULT (datetime('now'))
            );
        """)
