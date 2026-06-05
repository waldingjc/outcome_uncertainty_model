"""The standard page layout — sidebar on the left, content area on the right."""

from __future__ import annotations

import reflex as rx

from webapp import style
from webapp.components.sidebar import sidebar


def page(*content: rx.Component) -> rx.Component:
    """Wrap a page's content in the standard sidebar + content layout."""
    return rx.hstack(
        sidebar(),
        rx.box(
            rx.vstack(
                *content,
                spacing="6",
                align="start",
                width="100%",
                max_width=style.CONTENT_MAX_WIDTH,
            ),
            flex="1",
            padding="2em",
            padding_top="1.5em",
            overflow_y="auto",
        ),
        spacing="0",
        width="100%",
        min_height="100vh",
        align="start",
    )
