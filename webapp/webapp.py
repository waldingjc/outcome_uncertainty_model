"""Reflex app entry point.

Defines the theme, the routes, and the on-load hooks that pull DB stats
into state when each page mounts.
"""

from __future__ import annotations

import reflex as rx

from webapp import style
from webapp.pages.home import home
from webapp.pages.team import team
from webapp.state.db import DBState


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
# fires when the page mounts; we use it to refresh DB stats so the home
# page's KPIs always reflect the latest data.

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

# Other pages (leagues, model, pipeline) are placeholders we'll wire up
# once their state modules exist. For now they 404 — easier than
# half-built pages distracting from the working ones.
