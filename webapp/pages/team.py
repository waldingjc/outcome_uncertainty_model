"""Team breakdown page — text search → KPIs + recent matches table."""

from __future__ import annotations

import reflex as rx

from webapp.components.layout import page
from webapp.components.metric_card import metric_card
from webapp.state.teams import TeamState


def _search_bar() -> rx.Component:
    return rx.vstack(
        rx.heading("Team breakdown", size="7", weight="bold"),
        rx.text(
            "Search by name (case- and diacritic-insensitive). "
            "Picks the team with the most fixtures if ambiguous.",
            color=rx.color("gray", 11),
            size="2",
        ),
        rx.hstack(
            rx.icon("search", size=18, color=rx.color("gray", 10)),
            rx.input(
                placeholder="e.g. Fenerbahce, Real Madrid, Wrexham",
                value=TeamState.query,
                on_change=TeamState.set_query,
                size="3",
                width="100%",
                variant="surface",
            ),
            spacing="2",
            align="center",
            width="100%",
        ),
        rx.cond(
            TeamState.error != "",
            rx.callout(
                TeamState.error,
                icon="triangle-alert",
                color_scheme="amber",
                variant="surface",
                size="1",
                width="100%",
            ),
            rx.fragment(),
        ),
        spacing="3",
        align="start",
        width="100%",
    )


def _team_header() -> rx.Component:
    return rx.vstack(
        rx.hstack(
            rx.heading(TeamState.team_name, size="6", weight="bold"),
            rx.badge(
                TeamState.primary_league,
                color_scheme="orange",
                variant="surface",
                size="2",
            ),
            spacing="3",
            align="center",
        ),
        rx.text(
            "team_id ", TeamState.team_id.to_string(),
            "  ·  ", TeamState.competition_count_str,
            " competitions",
            size="1",
            color=rx.color("gray", 10),
            style={"font_family": "monospace"},
        ),
        spacing="1",
        align="start",
    )


def _kpis() -> rx.Component:
    return rx.grid(
        metric_card("Matches",   TeamState.match_count_str, icon="hash"),
        metric_card("Record",    TeamState.record_str,      icon="list-checks"),
        metric_card("Win rate",  TeamState.win_rate_str,    icon="trophy", accent=True),
        metric_card("GF / GA (avg)", TeamState.goals_str,   icon="target"),
        metric_card("Goal diff", TeamState.goal_diff_str,   icon="trending-up"),
        columns="5",
        spacing="3",
        width="100%",
    )


def _result_badge(result: rx.Var) -> rx.Component:
    """Colour-coded W/D/L pill for the recent-matches table."""
    return rx.match(
        result,
        ("W", rx.badge("W", color_scheme="green", variant="solid", size="1")),
        ("D", rx.badge("D", color_scheme="amber", variant="solid", size="1")),
        ("L", rx.badge("L", color_scheme="red",   variant="solid", size="1")),
        rx.badge(result, color_scheme="gray", variant="surface", size="1"),
    )


def _recent_matches_table() -> rx.Component:
    return rx.card(
        rx.vstack(
            rx.hstack(
                rx.icon("calendar-days", size=18, color=rx.color("orange", 10)),
                rx.text(
                    "Recent matches (last 15)",
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
                        rx.table.column_header_cell("Venue"),
                        rx.table.column_header_cell("Opponent"),
                        rx.table.column_header_cell("Score"),
                        rx.table.column_header_cell("Result"),
                        rx.table.column_header_cell("Competition"),
                    ),
                ),
                rx.table.body(
                    rx.foreach(
                        TeamState.recent_matches,
                        lambda m: rx.table.row(
                            rx.table.cell(m["Date"]),
                            rx.table.cell(
                                rx.badge(
                                    m["Venue"],
                                    color_scheme=rx.cond(
                                        m["Venue"] == "home", "blue", "gray"
                                    ),
                                    variant="surface",
                                    size="1",
                                ),
                            ),
                            rx.table.cell(m["Opponent"]),
                            rx.table.cell(
                                m["Score"],
                                style={"font_family": "monospace"},
                            ),
                            rx.table.cell(_result_badge(m["R"])),
                            rx.table.cell(
                                rx.text(
                                    m["Competition"],
                                    size="1",
                                    color=rx.color("gray", 11),
                                ),
                            ),
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


def _figure_card() -> rx.Component:
    """The 6-panel matplotlib breakdown — generated server-side on first
    search of each team, then cached on disk."""
    return rx.cond(
        TeamState.figure_url != "",
        rx.card(
            rx.vstack(
                rx.hstack(
                    rx.icon("chart-line", size=18, color=rx.color("orange", 10)),
                    rx.text(
                        "Visual breakdown",
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
                    src=TeamState.figure_url,
                    width="100%",
                    height="auto",
                    style={
                        "border_radius": "6px",
                        "background": rx.color("gray", 1),
                    },
                ),
                rx.text(
                    "Season-by-season W/D/L, goal distributions, points "
                    "trajectory, top opponents, scoreline heatmap, and "
                    "performance by competition.",
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
        ),
        rx.fragment(),
    )


def team() -> rx.Component:
    return page(
        _search_bar(),
        rx.cond(
            TeamState.has_team,
            rx.vstack(
                _team_header(),
                _kpis(),
                _recent_matches_table(),
                _figure_card(),
                spacing="5",
                align="start",
                width="100%",
            ),
            rx.cond(
                TeamState.has_query,
                rx.text(""),  # error already shown above via callout
                rx.callout(
                    "Type a team name above to load their breakdown.",
                    icon="info",
                    color_scheme="gray",
                    variant="surface",
                    size="1",
                ),
            ),
        ),
    )
