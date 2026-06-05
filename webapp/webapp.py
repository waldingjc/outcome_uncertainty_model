"""Reflex app entry point.

Defines the theme, the routes, and the on-load hooks that pull DB stats
into state when each page mounts.
"""

from __future__ import annotations

import reflex as rx

from webapp import _figures, style
from webapp.pages.h2h import h2h
from webapp.pages.home import home
from webapp.pages.leagues import leagues
from webapp.pages.model import model_dashboard
from webapp.pages.pipeline import pipeline
from webapp.pages.team import team
from webapp.state.db import DBState
from webapp.state.leagues import LeagueState
from webapp.state.model import ModelState
from webapp.state.pipeline import PipelineState


# Make matplotlib figures available as static assets before the first
# page render. sync_figures() copies any new PNGs from data/figures/
# into assets/figures/, where Reflex serves them at /figures/<name>.
_figures.sync_figures()


# Theme — passed once at App construction time. Reflex applies it
# globally; pages don't need to wrap themselves in `rx.theme(...)`.
app = rx.App(
    theme=rx.theme(
        appearance=style.THEME_APPEARANCE,
        accent_color=style.THEME_ACCENT_COLOR,
        gray_color=style.THEME_GRAY_COLOR,
        radius=style.THEME_RADIUS,
        panel_background=style.THEME_PANEL_BACKGROUND,
        has_background=True,
    ),
)


# Pages — each `app.add_page(...)` registers a route. The `on_load` arg
# fires when the page mounts; we use it to refresh data from disk/SQLite
# so the page reflects the current state of the project.

app.add_page(
    home,
    route="/",
    title="Outcome Uncertainty Model",
    on_load=DBState.load,
)

app.add_page(
    team,
    route="/team",
    title="Team breakdown · Outcome Uncertainty Model",
)

app.add_page(
    h2h,
    route="/h2h",
    title="Head to head · Outcome Uncertainty Model",
)

app.add_page(
    leagues,
    route="/leagues",
    title="League ladders · Outcome Uncertainty Model",
    on_load=LeagueState.on_load,
)

app.add_page(
    model_dashboard,
    route="/model",
    title="Model dashboard · Outcome Uncertainty Model",
    on_load=ModelState.on_load,
)

app.add_page(
    pipeline,
    route="/pipeline",
    title="Pipeline status · Outcome Uncertainty Model",
    on_load=PipelineState.on_load,
)
