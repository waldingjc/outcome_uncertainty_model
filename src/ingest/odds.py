"""Fetch and store pre-match odds from api-football.

api-football retains odds for **7 days** after the fixture date, so this is a
retroactive fetch — we look back at recently-finished fixtures and grab their
closing odds (the last quoted price before kick-off, the most predictive for
modelling). One `/odds?fixture={id}` call returns every bookmaker x bet type
combination for that fixture in a single response.

Module is exposed as `fetch_odds_for_fixture(fixture_id)`; the daily runner
loops over this for fixtures missing odds.
"""

from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from pathlib import Path

from src.db.schema import get_connection
from src.ingest.api_football import request_v3

logger = logging.getLogger(__name__)

_RAW_DIR = Path(__file__).parents[2] / "data" / "raw"


def _save_raw(data: dict, fixture_id: int) -> None:
    _RAW_DIR.mkdir(parents=True, exist_ok=True)
    path = _RAW_DIR / f"odds_f{fixture_id}.json"
    path.write_text(json.dumps(data, indent=2), encoding="utf-8")


def _flatten_odds(fixture_id: int, response: list[dict]) -> list[dict]:
    """Flatten api-football's nested odds response into per-row odds entries.

    Response shape:
        response[0].bookmakers[].bets[].values[]   (one fixture per response)
    Each value gives a price for one (bookmaker, bet, label) combination.
    """
    rows: list[dict] = []
    if not response:
        return rows

    now = datetime.now(timezone.utc).isoformat()
    # The /odds endpoint returns one entry per fixture; we filter explicitly
    # in case api-football ever returns multiple.
    for entry in response:
        if (entry.get("fixture") or {}).get("id") != fixture_id:
            continue
        for bookmaker in entry.get("bookmakers", []):
            bookmaker_id = bookmaker.get("id")
            bookmaker_name = bookmaker.get("name")
            if bookmaker_id is None or bookmaker_name is None:
                continue
            for bet in bookmaker.get("bets", []):
                bet_id = bet.get("id")
                bet_name = bet.get("name")
                if bet_id is None or bet_name is None:
                    continue
                for value in bet.get("values", []):
                    label = value.get("value")
                    odd_str = value.get("odd")
                    if label is None or odd_str is None:
                        continue
                    try:
                        odd = float(odd_str)
                    except (TypeError, ValueError):
                        continue
                    rows.append({
                        "fixture_id": fixture_id,
                        "bookmaker_id": bookmaker_id,
                        "bookmaker_name": bookmaker_name,
                        "bet_id": bet_id,
                        "bet_name": bet_name,
                        "value_label": str(label),
                        "odd": odd,
                        "ingested_at": now,
                    })
    return rows


def _upsert_odds(rows: list[dict]) -> int:
    if not rows:
        return 0
    with get_connection() as conn:
        conn.executemany(
            """
            INSERT INTO odds (
                fixture_id, bookmaker_id, bookmaker_name,
                bet_id, bet_name, value_label, odd, ingested_at
            ) VALUES (
                :fixture_id, :bookmaker_id, :bookmaker_name,
                :bet_id, :bet_name, :value_label, :odd, :ingested_at
            )
            ON CONFLICT(fixture_id, bookmaker_id, bet_id, value_label) DO UPDATE SET
                bookmaker_name = excluded.bookmaker_name,
                bet_name       = excluded.bet_name,
                odd            = excluded.odd,
                ingested_at    = excluded.ingested_at
            """,
            rows,
        )
    return len(rows)


def fetch_odds_for_fixture(
    fixture_id: int, save_raw: bool = True
) -> tuple[int, int | None]:
    """Fetch odds for one fixture and upsert them into the DB.

    Returns:
        (rows_written, daily_calls_remaining)
    """
    data, remaining = request_v3("odds", {"fixture": fixture_id})
    if save_raw:
        _save_raw(data, fixture_id)

    rows = _flatten_odds(fixture_id, data.get("response", []))
    written = _upsert_odds(rows)
    logger.info(
        "Fixture %d: wrote %d odds rows (daily calls remaining: %s)",
        fixture_id, written, remaining,
    )
    return written, remaining


if __name__ == "__main__":
    import argparse

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    parser = argparse.ArgumentParser(description="Fetch odds for a single fixture")
    parser.add_argument("--fixture", type=int, required=True, help="api-football fixture ID")
    parser.add_argument("--no-raw", action="store_true", help="Skip saving raw JSON responses")
    args = parser.parse_args()

    written, remaining = fetch_odds_for_fixture(args.fixture, save_raw=not args.no_raw)
    print(f"Done — {written} odds rows. Daily calls remaining: {remaining}.")
