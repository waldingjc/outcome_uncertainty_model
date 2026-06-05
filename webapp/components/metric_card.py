"""KPI tile — used across the home page and team breakdown."""

from __future__ import annotations

import reflex as rx


def metric_card(
    label: str,
    value: rx.Var | str,
    icon: str | None = None,
    accent: bool = False,
) -> rx.Component:
    """A single stat tile.

    Args:
        label:  small caption above the value
        value:  the main number (str or rx.Var pointing at one)
        icon:   optional Lucide icon name (e.g. "trophy", "users", "database")
        accent: if True, value uses the theme accent color instead of default
    """
    value_color = rx.color("orange", 11) if accent else rx.color("gray", 12)

    return rx.card(
        rx.vstack(
            rx.hstack(
                rx.cond(
                    icon is not None,
                    rx.icon(icon or "circle", size=14, color=rx.color("gray", 10)),
                    rx.fragment(),
                ),
                rx.text(
                    label,
                    size="1",
                    color=rx.color("gray", 11),
                    weight="medium",
                    style={"text_transform": "uppercase", "letter_spacing": "0.05em"},
                ),
                spacing="2",
                align="center",
            ),
            rx.heading(
                value,
                size="7",
                weight="bold",
                color=value_color,
            ),
            spacing="2",
            align="start",
        ),
        size="2",
        variant="surface",
        width="100%",
    )
