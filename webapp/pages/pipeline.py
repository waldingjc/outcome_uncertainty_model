"""Pipeline status page — queue health, last-run summary, and a 14-day
log-presence calendar."""

from __future__ import annotations

import reflex as rx

from webapp.components.layout import page
from webapp.components.metric_card import metric_card
from webapp.state.pipeline import PipelineState


def _header() -> rx.Component:
    return rx.vstack(
        rx.heading("Pipeline status", size="7", weight="bold"),
        rx.text(
            "Backfill queue progress, the most recent scheduled run, and a "
            "rolling 14-day view of which days produced a log file.",
            color=rx.color("gray", 11),
            size="2",
        ),
        spacing="2",
        align="start",
    )


def _queue_card() -> rx.Component:
    """Queue progress — done / pending / total with a progress bar."""
    return rx.card(
        rx.vstack(
            rx.hstack(
                rx.icon("list-checks", size=18, color=rx.color("orange", 10)),
                rx.text(
                    "Backfill queue",
                    size="3",
                    weight="medium",
                    color=rx.color("gray", 12),
                ),
                spacing="2",
                align="center",
                width="100%",
            ),
            rx.divider(),
            rx.grid(
                metric_card("Done",    PipelineState.queue_done_str,    icon="check",    accent=True),
                metric_card("Pending", PipelineState.queue_pending_str, icon="hourglass"),
                metric_card("Total",   PipelineState.queue_total_str,   icon="layers"),
                metric_card("Progress", PipelineState.queue_pct_str,    icon="percent",  accent=True),
                columns="4",
                spacing="3",
                width="100%",
            ),
            rx.progress(
                value=PipelineState.queue_pct_value,
                max=100,
                color_scheme="orange",
                size="2",
                width="100%",
            ),
            spacing="3",
            align="start",
            width="100%",
        ),
        size="3",
        variant="surface",
        width="100%",
    )


def _last_run_card() -> rx.Component:
    """A one-liner summary of the most recent backfill run."""
    return rx.card(
        rx.vstack(
            rx.hstack(
                rx.icon("clock", size=18, color=rx.color("orange", 10)),
                rx.text(
                    "Last scheduled run",
                    size="3",
                    weight="medium",
                    color=rx.color("gray", 12),
                ),
                spacing="2",
                align="center",
                width="100%",
            ),
            rx.divider(),
            rx.text(
                PipelineState.last_summary_str,
                size="2",
                color=rx.color("gray", 12),
                style={"font_family": "monospace"},
            ),
            spacing="3",
            align="start",
            width="100%",
        ),
        size="3",
        variant="surface",
        width="100%",
    )


def _day_cell(row) -> rx.Component:
    """One cell of the 14-day calendar — green-ish if a log exists,
    muted if the day was skipped."""
    return rx.tooltip(
        rx.box(
            rx.vstack(
                rx.text(
                    row["date"],
                    size="1",
                    style={"font_family": "monospace"},
                    color=rx.cond(row["ran"], rx.color("orange", 11), rx.color("gray", 10)),
                ),
                rx.text(
                    row["size_kb"],
                    size="1",
                    color=rx.color("gray", 11),
                ),
                spacing="1",
                align="center",
            ),
            padding="0.5em 0.6em",
            border_radius="6px",
            background=rx.cond(
                row["ran"],
                rx.color("orange", 3),
                rx.color("gray", 3),
            ),
            border=rx.cond(
                row["ran"],
                f"1px solid {rx.color('orange', 6)}",
                f"1px solid {rx.color('gray', 5)}",
            ),
            min_width="86px",
        ),
        content=rx.cond(
            row["ran"],
            row["date"] + " — " + row["size_kb"] + " KB log",
            row["date"] + " — no log",
        ),
    )


def _log_calendar_card() -> rx.Component:
    return rx.card(
        rx.vstack(
            rx.hstack(
                rx.icon("calendar-days", size=18, color=rx.color("orange", 10)),
                rx.text(
                    "Recent daily logs (14d)",
                    size="3",
                    weight="medium",
                    color=rx.color("gray", 12),
                ),
                spacing="2",
                align="center",
                width="100%",
            ),
            rx.divider(),
            rx.grid(
                rx.foreach(PipelineState.recent_days, _day_cell),
                columns="7",
                spacing="2",
                width="100%",
            ),
            rx.text(
                "Each tile is one day — orange if a backfill log was written, "
                "muted if the run didn't happen. Size shown in KB.",
                size="1",
                color=rx.color("gray", 11),
            ),
            spacing="3",
            align="start",
            width="100%",
        ),
        size="3",
        variant="surface",
        width="100%",
    )


def pipeline() -> rx.Component:
    return page(
        _header(),
        _queue_card(),
        _last_run_card(),
        _log_calendar_card(),
    )
