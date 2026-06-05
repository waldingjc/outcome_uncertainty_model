"""Static-asset bridge for the matplotlib figures we generate elsewhere.

Reflex serves files placed under `assets/` at the root URL of the dev
server (so `assets/figures/foo.png` is reachable at `/figures/foo.png`).
Our analysis modules write to `data/figures/` — gitignored, sometimes
regenerated. This module copies (or symlinks where possible) those PNGs
into `assets/figures/` at app startup so they're servable.

Called from `webapp.webapp` at module import time. Cheap: just stats and
copies a handful of files.
"""

from __future__ import annotations

import shutil
from pathlib import Path

_REPO_ROOT = Path(__file__).parents[1]
_DATA_FIGURES = _REPO_ROOT / "data" / "figures"
_ASSETS_FIGURES = _REPO_ROOT / "assets" / "figures"


def sync_figures() -> dict[str, str]:
    """Copy every PNG from data/figures to assets/figures.

    Returns a {filename: relative_url} map of available figures so pages
    can iterate over what's actually present without re-listing the
    directory.
    """
    _ASSETS_FIGURES.mkdir(parents=True, exist_ok=True)
    available: dict[str, str] = {}

    if not _DATA_FIGURES.exists():
        return available

    for png in sorted(_DATA_FIGURES.glob("*.png")):
        target = _ASSETS_FIGURES / png.name
        # Skip if already up to date — avoid pointless writes on hot reloads.
        if target.exists() and target.stat().st_mtime >= png.stat().st_mtime:
            available[png.name] = f"/figures/{png.name}"
            continue
        try:
            shutil.copy2(png, target)
        except OSError:
            continue
        available[png.name] = f"/figures/{png.name}"

    return available


def find_figure(*candidates: str) -> str | None:
    """Return the first /figures/<name>.png URL that exists.

    Pages call this with a list of preferred names — e.g. the model
    dashboard prefers `model_evaluation_143leagues.png` but falls back
    to whatever's available. Returns None if nothing matches.
    """
    for name in candidates:
        if (_ASSETS_FIGURES / name).exists():
            return f"/figures/{name}"
    return None


def list_figures() -> list[str]:
    """Sorted list of available figure URLs after sync."""
    if not _ASSETS_FIGURES.exists():
        return []
    return sorted(f"/figures/{p.name}" for p in _ASSETS_FIGURES.glob("*.png"))
