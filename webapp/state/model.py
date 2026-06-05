"""State for the model dashboard page.

Reads:
  - the latest model_evaluation_*.png from data/figures/
  - the matching model_evaluation_*.json sidecar (when present) for the
    actual metric values, otherwise falls back to a "no data" view.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import reflex as rx

from webapp import _figures

_REPO_ROOT = Path(__file__).parents[2]
_FIGURES_DIR = _REPO_ROOT / "data" / "figures"

# Match the league-count suffix on figure names like
# `model_evaluation_671leagues.png` so we can pick the widest-scope
# regen automatically as the dataset grows.
_SCOPE_PATTERN = re.compile(r"model_evaluation_(\d+)leagues\.png$")


class ModelState(rx.State):
    figure_url: str = ""
    has_figure: bool = False

    log_loss: str = "—"
    brier: str = "—"
    ece: str = "—"
    tail_brier: str = "—"
    accuracy: str = "—"

    n_train: int = 0
    n_test: int = 0
    scope_label: str = ""
    generated_at: str = ""
    metrics_available: bool = False

    def on_load(self):
        # Ensure figures are synced
        _figures.sync_figures()

        # Pick the model_evaluation figure with the widest scope (highest
        # league count). Falls back to the PL-only figure if no
        # multi-league ones exist. Sorting numerically by the
        # `<N>leagues` suffix means we don't need a code change every
        # time the dataset grows another tier.
        scope_files: list[tuple[int, str]] = []
        if _FIGURES_DIR.exists():
            for png in _FIGURES_DIR.glob("model_evaluation_*leagues.png"):
                m = _SCOPE_PATTERN.search(png.name)
                if m:
                    scope_files.append((int(m.group(1)), png.name))
        scope_files.sort(reverse=True)

        candidates = [name for _, name in scope_files]
        candidates.append("model_evaluation_PL.png")  # last-ditch fallback

        url = _figures.find_figure(*candidates)
        if url:
            self.figure_url = url
            self.has_figure = True
            self._load_metrics(url)
        else:
            self.has_figure = False

    def _load_metrics(self, figure_url: str):
        """Look for a sidecar JSON next to the figure (same stem)."""
        png_name = figure_url.split("/")[-1]
        json_path = _FIGURES_DIR / png_name.replace(".png", ".json")
        if not json_path.exists():
            self.metrics_available = False
            self.scope_label = png_name.replace("model_evaluation_", "").replace(".png", "")
            return
        try:
            data = json.loads(json_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            self.metrics_available = False
            return

        # Headline metrics — for the GBM specifically (the project's best model)
        gbm = (data.get("models", {}).get("GBM")
               or data.get("gbm")
               or {})
        self.log_loss   = self._fmt(gbm.get("log_loss",   gbm.get("log_loss_point")))
        self.brier      = self._fmt(gbm.get("brier",      gbm.get("brier_point")))
        self.ece        = self._fmt(gbm.get("ece"),       precision=4)
        self.tail_brier = self._fmt(gbm.get("tail_brier"), precision=4)
        self.accuracy   = self._fmt_pct(gbm.get("accuracy"))

        self.n_train = int(data.get("n_train", 0) or 0)
        self.n_test  = int(data.get("n_test",  0) or 0)
        self.scope_label = str(data.get("scope_label",
                              png_name.replace("model_evaluation_", "").replace(".png", "")))
        self.generated_at = str(data.get("generated_at", ""))[:19].replace("T", " ")
        self.metrics_available = True

    @staticmethod
    def _fmt(x, precision: int = 4) -> str:
        if x is None:
            return "—"
        try:
            return f"{float(x):.{precision}f}"
        except (TypeError, ValueError):
            return "—"

    @staticmethod
    def _fmt_pct(x) -> str:
        if x is None:
            return "—"
        try:
            return f"{100 * float(x):.1f}%"
        except (TypeError, ValueError):
            return "—"
