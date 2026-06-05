"""State for the model dashboard page.

Reads:
  - the latest model_evaluation_*.png from data/figures/
  - the matching model_evaluation_*.json sidecar (when present) for the
    actual metric values, otherwise falls back to a "no data" view.
"""

from __future__ import annotations

import json
from pathlib import Path

import reflex as rx

from webapp import _figures

_REPO_ROOT = Path(__file__).parents[2]
_FIGURES_DIR = _REPO_ROOT / "data" / "figures"


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

        # Find the best available model_evaluation figure (largest scope first).
        # Reflex's static-file URLs are served at the assets root.
        candidates = [
            "model_evaluation_143leagues.png",
            "model_evaluation_115leagues.png",
            "model_evaluation_5leagues.png",
            "model_evaluation_PL.png",
        ]
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
