"""Head-to-head page — pick two teams, see their historical record."""

from __future__ import annotations

import reflex as rx

from webapp.components.layout import page
from webapp.components.metric_card import metric_card
from webapp.state.h2h import H2HState


def _search_row() -> rx.Component:
    """Two search bars side-by-side — one per team."""
    return rx.vstack(
        rx.heading("Head to head", size="7", weight="bold"),
        rx.text(
            "Pick two teams and see their entire shared history — record, "
            "goal tally, where they've met, and every meeting in the dataset.",
            color=rx.color("gray", 11),
            size="2",
        ),
        rx.grid(
            _team_search("Team A", H2HState.query_a, H2HState.set_query_a,
                         H2HState.team_a_name, H2HState.error_a),
            _team_search("Team B", H2HState.query_b, H2HState.set_query_b,
                         H2HState.team_b_name, H2HState.error_b),
            columns="2",
            spacing="4",
            width="100%",
        ),
        spacing="3",
        align="start",
        width="100%",
    )


def _team_search(
    label: str, value, on_change, resolved_name, error,
) -> rx.Component:
    """One search slot — input + resolved name confirmation + per-slot error."""
    return rx.vstack(
        rx.text(
            label,
            size="1",
            weight="medium",
            color=rx.color("gray", 11),
            style={"text_transform": "uppercase", "letter_spacing": "0.05em"},
        ),
        rx.hstack(
            rx.icon("search", size=18, color=rx.color("gray", 10)),
            rx.input(
                placeholder="e.g. Liverpool",
                value=value,
                on_change=on_change,
                size="3",
                width="100%",
                variant="surface",
            ),
            spacing="2",
            align="center",
            width="100%",
        ),
        rx.cond(
            resolved_name != "",
            rx.badge(
                resolved_name,
                color_scheme="orange",
                variant="surface",
                size="1",
            ),
            rx.fragment(),
        ),
        rx.cond(
            error != "",
            rx.callout(error, icon="triangle-alert", color_scheme="amber",
                       variant="surface", size="1", width="100%"),
            rx.fragment(),
        ),
        spacing="2",
        align="start",
        width="100%",
    )


def _kpis() -> rx.Component:
    return rx.grid(
        metric_card("Meetings",
                    H2HState.match_count_str, icon="hash"),
        metric_card("Record (A perspective)",
                    H2HState.record_str,      icon="list-checks"),
        metric_card("A win rate",
                    H2HState.a_win_pct_str,   icon="trophy",  accent=True),
        metric_card("B win rate",
                    H2HState.b_win_pct_str,   icon="trophy"),
        metric_card("Goals (A : B)",
                    H2HState.goals_str,       icon="target"),
        metric_card("Avg goals/match",
                    H2HState.avg_goals_str,   icon="trending-up", accent=True),
        columns="3",
        spacing="3",
        width="100%",
    )


def _competition_chips() -> rx.Component:
    """Shows the leagues/cups where the two teams have met, with counts."""
    return rx.cond(
        H2HState.competitions.length() > 0,
        rx.hstack(
            rx.text(
                "Met in:",
                size="1",
                weight="medium",
                color=rx.color("gray", 11),
                style={"text_transform": "uppercase", "letter_spacing": "0.05em"},
            ),
            rx.foreach(
                H2HState.competitions,
                lambda c: rx.badge(
                    rx.fragment(c["league"], " · ", c["n"].to_string()),
                    color_scheme="gray",
                    variant="surface",
                    size="2",
                ),
            ),
            spacing="3",
            align="center",
            wrap="wrap",
            width="100%",
        ),
        rx.fragment(),
    )


def _result_badge(result) -> rx.Component:
    """W/D/L pill from A's perspective."""
    return rx.match(
        result,
        ("W", rx.badge("W", color_scheme="green", variant="solid", size="1")),
        ("D", rx.badge("D", color_scheme="amber", variant="solid", size="1")),
        ("L", rx.badge("L", color_scheme="red",   variant="solid", size="1")),
        rx.badge(result, color_scheme="gray", variant="surface", size="1"),
    )


def _meetings_table() -> rx.Component:
    return rx.card(
        rx.vstack(
            rx.hstack(
                rx.icon("calendar-days", size=18, color=rx.color("orange", 10)),
                rx.text(
                    "All meetings (most recent first, up to 50)",
                    size="2",
                    weight="medium",
                    color=rx.color("gray", 12),
                ),
                spacing="2",
                align="center",
                width="100%",
            ),
            rx.divider(),
            rx.table.root(
                rx.table.header(
                    rx.table.row(
                        rx.table.column_header_cell("Date"),
                        rx.table.column_header_cell("Competition"),
                        rx.table.column_header_cell("Venue (A)"),
                        rx.table.column_header_cell("Score (A–B)"),
                        rx.table.column_header_cell("Result (A)"),
                    ),
                ),
                rx.table.body(
                    rx.foreach(
                        H2HState.meetings,
                        lambda m: rx.table.row(
                            rx.table.cell(m["Date"]),
                            rx.table.cell(
                                rx.text(
                                    m["Competition"],
                                    size="1",
                                    color=rx.color("gray", 11),
                                ),
                            ),
                            rx.table.cell(
                                rx.badge(
                                    m["Venue"],
                                    color_scheme=rx.cond(
                                        m["Venue"] == "home", "blue", "gray",
                                    ),
                                    variant="surface",
                                    size="1",
                                ),
                            ),
                            rx.table.cell(
                                m["Score"],
                                style={"font_family": "monospace"},
                            ),
                            rx.table.cell(_result_badge(m["R"])),
                        ),
                    ),
                ),
                variant="surface",
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


def _matchup_header() -> rx.Component:
    return rx.hstack(
        rx.heading(H2HState.header_str, size="6", weight="bold"),
        rx.text(
            "match_count: ",
            H2HState.match_count.to_string(),
            size="1",
            color=rx.color("gray", 10),
            style={"font_family": "monospace"},
        ),
        spacing="4",
        align="end",
        wrap="wrap",
    )


def h2h() -> rx.Component:
    return page(
        _search_row(),
        rx.cond(
            H2HState.has_both_teams,
            rx.cond(
                H2HState.match_count > 0,
                rx.vstack(
                    _matchup_header(),
                    _competition_chips(),
                    _kpis(),
                    _meetings_table(),
                    spacing="5",
                    align="start",
                    width="100%",
                ),
                rx.callout(
                    "No meetings between these two teams in the dataset.",
                    icon="info",
                    color_scheme="gray",
                    variant="surface",
                    size="1",
                ),
            ),
            rx.callout(
                "Type a name into both search boxes above to load the "
                "head-to-head history.",
                icon="info",
                color_scheme="gray",
                variant="surface",
                size="1",
            ),
        ),
    )
