"""Environment configuration helpers.

Loads `.env` from the current working directory at import time. Existing
process env vars are NOT overridden — `.env` only fills in what's missing,
so an explicit shell `export` (or `setx` / Windows User env var) always wins.

Format (one per line):
    KEY=value
    QUOTED_KEY="value with spaces"
    # comment lines and blank lines are ignored

Used for things like API_FOOTBALL_KEY so contributors don't need to set
machine-wide env vars just to run the ingestion scripts.
"""

from __future__ import annotations

import os
from pathlib import Path


def load_dotenv(path: Path | None = None) -> int:
    """Load KEY=VALUE pairs from a .env file. Returns count of vars set."""
    env_path = path or Path.cwd() / ".env"
    if not env_path.is_file():
        return 0

    loaded = 0
    for raw_line in env_path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        if "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip()
        if (value.startswith('"') and value.endswith('"')) or (
            value.startswith("'") and value.endswith("'")
        ):
            value = value[1:-1]
        if key and key not in os.environ:
            os.environ[key] = value
            loaded += 1
    return loaded


# Auto-load on import so any module that does `from src.config import *`
# or just imports anything that transitively imports this gets .env applied.
load_dotenv()
