"""Shared matplotlib styling for all project figures.

We render every figure in dark mode to match the Reflex webapp's dark
theme (orange accent over slate gray). Importing this module applies the
style at import time as a side effect — call `apply_dark_style()`
explicitly from each figure-generating module's `main()` so the choice
is visible at the call site too.

Why not just `plt.style.use("dark_background")`?
  - The built-in dark style picks fully-saturated colors that clash with
    the warmer slate/orange of the webapp.
  - We want consistent panel backgrounds across figures (not pure black —
    a slate that matches the Reflex panels).
  - The webapp serves these PNGs over a dark background, so transparent
    figure backgrounds would let the page bleed through; we keep a solid
    panel color so the figure reads as one self-contained object.
"""

from __future__ import annotations

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator

# Slate-ish background mirroring the Radix `slate` palette the webapp uses,
# with orange-tinted accents that match THEME_ACCENT_COLOR = "orange".
_FIGURE_BG = "#161719"   # near-black slate (Radix slate-1 in dark)
_PANEL_BG  = "#1c1d20"   # axes face — slightly lighter than figure
_FG        = "#e6e6e8"   # text + ticks
_MUTED     = "#9aa0a6"   # secondary text
_GRID      = "#2a2c30"   # grid lines
_ACCENT    = "#f78737"   # orange (Radix orange-9)

# Categorical palette used by figures that pick colors from prop_cycle.
_CYCLE = [
    "#f78737",  # orange
    "#62b6cb",  # cyan-blue
    "#b8c480",  # lime
    "#c084fc",  # lilac
    "#f4a261",  # warm peach
    "#ef6f6c",  # coral
    "#76c893",  # green
    "#e9c46a",  # mustard
]


def apply_dark_style() -> None:
    """Apply project-wide dark theme to matplotlib's rcParams.

    Idempotent. Safe to call multiple times — just sets rcParams.
    """
    mpl.rcParams.update({
        "figure.facecolor":  _FIGURE_BG,
        "savefig.facecolor": _FIGURE_BG,
        "axes.facecolor":    _PANEL_BG,
        "axes.edgecolor":    _GRID,
        "axes.labelcolor":   _FG,
        "axes.titlecolor":   _FG,
        "xtick.color":       _MUTED,
        "ytick.color":       _MUTED,
        "text.color":        _FG,
        "grid.color":        _GRID,
        "grid.alpha":        0.6,
        "legend.facecolor":  _PANEL_BG,
        "legend.edgecolor":  _GRID,
        "legend.labelcolor": _FG,
        "axes.prop_cycle":   plt.cycler(color=_CYCLE),
        "savefig.bbox":      "tight",
        "savefig.dpi":       140,
    })


# Exported color constants so individual figures can pull project-consistent
# hues without re-defining their own palettes.
FIGURE_BG = _FIGURE_BG
PANEL_BG  = _PANEL_BG
FG        = _FG
MUTED     = _MUTED
GRID      = _GRID
ACCENT    = _ACCENT
CYCLE     = list(_CYCLE)

# Semantic colors — used by team-breakdown, model-eval, etc.
W_COLOR = "#5fbf75"   # win — slightly muted green so it stays readable on dark
D_COLOR = "#e9c46a"   # draw — mustard
L_COLOR = "#ef6f6c"   # loss — coral (lighter than pure red, plays nicer on dark)
SCORED_COLOR = "#62b6cb"   # cyan-blue for "goals scored"
CONCEDED_COLOR = "#ef6f6c" # coral for "goals conceded"


def integer_axis(ax, axis: str = "y") -> None:
    """Force the given axis to use whole-number ticks only.

    Matplotlib's default autoscale picks fractional ticks (0.5, 1.5, ...)
    when the data range is small, even when the values are intrinsically
    integer-valued (matches played, goals scored, points, etc.). Apply
    this to any axis where 0.5 of something is meaningless.

    `axis` is "y" (default), "x", or "both". Use this AFTER plotting,
    since MaxNLocator inspects the axis limits.
    """
    locator = MaxNLocator(integer=True, min_n_ticks=1)
    if axis in ("y", "both"):
        ax.yaxis.set_major_locator(locator)
    if axis in ("x", "both"):
        # MaxNLocator can't be shared between axes — make a fresh one.
        ax.xaxis.set_major_locator(MaxNLocator(integer=True, min_n_ticks=1))
