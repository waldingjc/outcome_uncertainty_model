"""Persistent left-hand navigation."""

from __future__ import annotations

import reflex as rx

from webapp import style


def _nav_item(label: str, href: str, icon: str) -> rx.Component:
    """One row in the sidebar nav. Active route gets accent highlighting."""
    return rx.link(
        rx.hstack(
            rx.icon(icon, size=18),
            rx.text(label, size="2"),
            spacing="3",
            align="center",
            padding_x="3",
            padding_y="2",
            border_radius="6px",
            width="100%",
            _hover={"background": rx.color("gray", 4)},
        ),
        href=href,
        underline="none",
        color=rx.color("gray", 12),
        width="100%",
    )


def sidebar() -> rx.Component:
    """Sidebar shown on every page. Branding + nav."""
    return rx.vstack(
        # Branding
        rx.hstack(
            rx.icon("activity", size=24, color=rx.color("teal", 9)),
            rx.heading("Outcome", size="5", weight="bold"),
            spacing="2",
            align="center",
            padding_x="3",
            padding_y="4",
        ),

        rx.separator(),

        # Nav items
        rx.vstack(
            _nav_item("Home",              "/",          "house"),
            _nav_item("Team breakdown",    "/team",      "user"),
            _nav_item("League ladders",    "/leagues",   "trophy"),
            _nav_item("Model dashboard",   "/model",     "chart-line"),
            _nav_item("Pipeline status",   "/pipeline",  "activity"),
            spacing="1",
            padding_x="2",
            padding_y="2",
            width="100%",
        ),

        rx.spacer(),

        # Footer
        rx.vstack(
            rx.text(
                "Free api-football tier",
                size="1",
                color=rx.color("gray", 10),
            ),
            rx.text(
                "v0.1",
                size="1",
                color=rx.color("gray", 9),
            ),
            spacing="0",
            padding_x="3",
            padding_y="3",
        ),

        spacing="0",
        width=style.SIDEBAR_WIDTH,
        height="100vh",
        position="sticky",
        top="0",
        background=rx.color("gray", 2),
        border_right=f"1px solid {rx.color('gray', 5)}",
    )
