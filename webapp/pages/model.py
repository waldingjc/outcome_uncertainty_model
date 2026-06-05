"""Model dashboard — shows the latest H/D/A model evaluation."""

from __future__ import annotations

import reflex as rx

from webapp.components.layout import page
from webapp.components.metric_card import metric_card
from webapp.state.model import ModelState


def _header() -> rx.Component:
    return rx.vstack(
        rx.hstack(
            rx.heading("Model dashboard", size="7", weight="bold"),
            rx.cond(
                ModelState.has_figure,
                rx.badge(
                    ModelState.scope_label,
                    color_scheme="orange",
                    variant="surface",
                    size="2",
                ),
                rx.fragment(),
            ),
            spacing="3",
            align="center",
        ),
        rx.text(
            "Latest H/D/A model evaluation. The GBM metrics below come from "
            "the most recent `python -m src.model.figures` run.",
            color=rx.color("gray", 11),
            size="2",
        ),
        spacing="2",
        align="start",
    )


def _metric_row() -> rx.Component:
    return rx.cond(
        ModelState.metrics_available,
        rx.vstack(
            rx.grid(
                metric_card("Log-loss",   ModelState.log_loss,   icon="ruler", accent=True),
                metric_card("Brier",      ModelState.brier,      icon="target"),
                metric_card("ECE",        ModelState.ece,        icon="trending-down", accent=True),
                metric_card("Tail Brier", ModelState.tail_brier, icon="dices"),
                metric_card("Accuracy",   ModelState.accuracy,   icon="check"),
                columns="5",
                spacing="3",
                width="100%",
            ),
            rx.text(
                "Trained on ",
                rx.text.strong(ModelState.n_train.to_string()),
                " matches  ·  evaluated on ",
                rx.text.strong(ModelState.n_test.to_string()),
                " matches  ·  generated ",
                ModelState.generated_at,
                size="1",
                color=rx.color("gray", 11),
            ),
            spacing="3",
            align="start",
            width="100%",
        ),
        rx.callout(
            "No metric sidecar found for the latest figure. Run "
            "`py -3.14 -m src.model.figures --target-leagues all` to generate "
            "the figure with metrics.",
            icon="info",
            color_scheme="gray",
            variant="surface",
            size="1",
        ),
    )


def _figure_card() -> rx.Component:
    return rx.cond(
        ModelState.has_figure,
        rx.card(
            rx.vstack(
                rx.hstack(
                    rx.icon("chart-line", size=18, color=rx.color("orange", 10)),
                    rx.text(
                        "Reliability + scoring breakdown",
                        size="3",
                        weight="medium",
                        color=rx.color("gray", 12),
                    ),
                    spacing="2",
                    align="center",
                    width="100%",
                ),
                rx.divider(),
                rx.image(
                    src=ModelState.figure_url,
                    width="100%",
                    height="auto",
                    style={
                        "border_radius": "6px",
                        "background": rx.color("gray", 1),
                    },
                ),
                spacing="3",
                align="start",
                width="100%",
            ),
            size="3",
            variant="surface",
            width="100%",
        ),
        rx.callout(
            "No model evaluation figure found in data/figures/. "
            "Generate one with `py -3.14 -m src.model.figures --target-leagues all`.",
            icon="image-off",
            color_scheme="gray",
            variant="surface",
            size="1",
        ),
    )


def model_dashboard() -> rx.Component:
    return page(
        _header(),
        _metric_row(),
        _figure_card(),
    )
