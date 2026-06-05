"""Home page — landing dashboard with cumulative DB stats and recent activity."""

from __future__ import annotations

import reflex as rx

from webapp.components.layout import page
from webapp.components.metric_card import metric_card
from webapp.state.db import DBState


def _hero() -> rx.Component:
    """Top section: project name, one-line description."""
    return rx.vstack(
        rx.hstack(
            rx.heading(
                "Outcome Uncertainty Model",
                size="8",
                weight="bold",
            ),
            rx.badge("Free tier", color_scheme="orange", variant="surface", size="2"),
            spacing="3",
            align="center",
        ),
        rx.text(
            "Calibrated H/D/A predictions across the world's senior club football leagues. "
            "GBM at ECE 0.003, beats Elo by ~2% log-loss.",
            color=rx.color("gray", 11),
            size="3",
        ),
        spacing="2",
        align="start",
    )


def _kpi_grid() -> rx.Component:
    """The 4-up KPI row."""
    return rx.grid(
        metric_card("Fixtures",   DBState.fixture_count_str, icon="database", accent=True),
        metric_card("Leagues",    DBState.league_count_str,  icon="trophy"),
        metric_card("Teams",      DBState.team_count_str,    icon="users"),
        metric_card("Date range", DBState.date_range_str,    icon="calendar"),
        columns="4",
        spacing="3",
        width="100%",
    )


def _queue_card() -> rx.Component:
    """Backfill queue progress."""
    return rx.card(
        rx.vstack(
            rx.hstack(
                rx.icon("list-checks", size=18, color=rx.color("orange", 10)),
                rx.text("Ingestion queue",
                        size="2", weight="medium",
                        color=rx.color("gray", 12)),
                rx.spacer(),
                rx.text(
                    DBState.queue_pct_str,
                    size="2",
                    weight="bold",
                    color=rx.color("orange", 11),
                ),
                align="center",
                width="100%",
            ),
            rx.progress(
                value=DBState.queue_progress_value,
                color_scheme="orange",
                size="2",
                width="100%",
            ),
            rx.hstack(
                rx.text(
                    f"Done: ",
                    rx.text.strong(DBState.queue_done.to_string()),
                    size="1",
                    color=rx.color("gray", 11),
                ),
                rx.text(
                    f"Pending: ",
                    rx.text.strong(DBState.queue_pending.to_string()),
                    size="1",
                    color=rx.color("gray", 11),
                ),
                rx.text(
                    f"Failed: ",
                    rx.text.strong(DBState.queue_failed.to_string()),
                    size="1",
                    color=rx.color("gray", 11),
                ),
                spacing="4",
                width="100%",
            ),
            spacing="3",
            width="100%",
        ),
        size="3",
        variant="surface",
        width="100%",
    )


def _last_run_card() -> rx.Component:
    """The last scheduled-run status line."""
    return rx.card(
        rx.hstack(
            rx.icon("clock", size=18, color=rx.color("orange", 10)),
            rx.vstack(
                rx.text(
                    "Most recent backfill",
                    size="1",
                    weight="medium",
                    color=rx.color("gray", 11),
                    style={"text_transform": "uppercase", "letter_spacing": "0.05em"},
                ),
                rx.text(
                    DBState.last_run_summary,
                    size="2",
                    color=rx.color("gray", 12),
                ),
                spacing="1",
                align="start",
            ),
            spacing="3",
            align="center",
            width="100%",
        ),
        size="3",
        variant="surface",
        width="100%",
    )


def home() -> rx.Component:
    return page(
        _hero(),
        _kpi_grid(),
        rx.grid(
            _queue_card(),
            _last_run_card(),
            columns="2",
            spacing="3",
            width="100%",
        ),
    )
