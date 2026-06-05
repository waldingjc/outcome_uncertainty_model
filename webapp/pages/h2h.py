"""Head-to-head page — pick 2 to 4 teams, see their pairwise history."""

from __future__ import annotations

import reflex as rx

from webapp.components.layout import page
from webapp.state.h2h import H2HState, MAX_TEAMS


def _header() -> rx.Component:
    return rx.vstack(
        rx.heading("Head to head", size="7", weight="bold"),
        rx.text(
            "Pick at least two teams (up to ",
            str(MAX_TEAMS),
            ") and see their combined head-to-head record, the leagues "
            "they have met in, every meeting between them, and a six-panel "
            "comparison figure overlaying each team's broader form.",
            color=rx.color("gray", 11),
            size="2",
        ),
        spacing="2",
        align="start",
    )


def _team_slot(idx) -> rx.Component:
    """One search box. The index drives `set_query_at(idx, value)` so
    every visible slot can be edited independently."""
    return rx.vstack(
        rx.text(
            "Team ",
            (idx + 1).to_string(),
            size="1",
            weight="medium",
            color=rx.color("gray", 11),
            style={"text_transform": "uppercase", "letter_spacing": "0.05em"},
        ),
        rx.hstack(
            rx.icon("search", size=18, color=rx.color("gray", 10)),
            rx.input(
                placeholder="e.g. Liverpool",
                value=H2HState.queries[idx],
                on_change=lambda val: H2HState.set_query_at(idx, val),
                size="3",
                width="100%",
                variant="surface",
            ),
            spacing="2",
            align="center",
            width="100%",
        ),
        rx.cond(
            H2HState.team_names[idx] != "",
            rx.badge(
                H2HState.team_names[idx],
                color_scheme="orange",
                variant="surface",
                size="1",
            ),
            rx.fragment(),
        ),
        rx.cond(
            H2HState.errors[idx] != "",
            rx.callout(
                H2HState.errors[idx],
                icon="triangle-alert",
                color_scheme="amber",
                variant="surface",
                size="1",
                width="100%",
            ),
            rx.fragment(),
        ),
        spacing="2",
        align="start",
        width="100%",
    )


def _slot_controls() -> rx.Component:
    """Add / remove buttons. Disabled at boundary conditions so the
    user can't go below 2 slots or above MAX_TEAMS."""
    return rx.hstack(
        rx.button(
            rx.icon("plus", size=14),
            "Add team",
            on_click=H2HState.add_slot,
            disabled=~H2HState.can_add_slot,
            variant="surface",
            size="2",
        ),
        rx.button(
            rx.icon("minus", size=14),
            "Remove last",
            on_click=H2HState.remove_slot,
            disabled=~H2HState.can_remove_slot,
            variant="surface",
            color_scheme="gray",
            size="2",
        ),
        spacing="2",
    )


def _search_grid() -> rx.Component:
    return rx.vstack(
        rx.grid(
            rx.foreach(H2HState.visible_indices, _team_slot),
            columns="2",
            spacing="4",
            width="100%",
        ),
        _slot_controls(),
        spacing="3",
        align="start",
        width="100%",
    )


def _competition_chips() -> rx.Component:
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


def _record_table() -> rx.Component:
    """Mini-league standing: each team's record vs the rest of the set."""
    return rx.card(
        rx.vstack(
            rx.hstack(
                rx.icon("trophy", size=18, color=rx.color("orange", 10)),
                rx.text(
                    "Pairwise standings",
                    size="3",
                    weight="medium",
                    color=rx.color("gray", 12),
                ),
                rx.text(
                    "(",
                    H2HState.match_count.to_string(),
                    " meetings)",
                    size="1",
                    color=rx.color("gray", 11),
                ),
                spacing="2",
                align="center",
                width="100%",
            ),
            rx.divider(),
            rx.table.root(
                rx.table.header(
                    rx.table.row(
                        rx.table.column_header_cell("Team"),
                        rx.table.column_header_cell("P", width="48px"),
                        rx.table.column_header_cell("W", width="48px"),
                        rx.table.column_header_cell("D", width="48px"),
                        rx.table.column_header_cell("L", width="48px"),
                        rx.table.column_header_cell("GF", width="56px"),
                        rx.table.column_header_cell("GA", width="56px"),
                        rx.table.column_header_cell("Pts", width="64px"),
                        rx.table.column_header_cell("Win %", width="80px"),
                    ),
                ),
                rx.table.body(
                    rx.foreach(
                        H2HState.per_team_records,
                        lambda r: rx.table.row(
                            rx.table.cell(rx.text(r["team"], weight="medium")),
                            rx.table.cell(r["P"].to_string()),
                            rx.table.cell(r["W"].to_string()),
                            rx.table.cell(r["D"].to_string()),
                            rx.table.cell(r["L"].to_string()),
                            rx.table.cell(r["GF"].to_string()),
                            rx.table.cell(r["GA"].to_string()),
                            rx.table.cell(
                                rx.text(r["Pts"].to_string(), weight="bold"),
                            ),
                            rx.table.cell(
                                rx.text(
                                    r["WinPct"],
                                    color=rx.color("orange", 11),
                                    style={"font_family": "monospace"},
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
    """Six-panel comparison figure — overlaid distributions, points
    trajectories, rolling win-rate, and a pairwise W/D/L matrix."""
    return rx.cond(
        H2HState.figure_url != "",
        rx.card(
            rx.vstack(
                rx.hstack(
                    rx.icon("chart-line", size=18, color=rx.color("orange", 10)),
                    rx.text(
                        "Comparison figure",
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
                    src=H2HState.figure_url,
                    width="100%",
                    height="auto",
                    style={
                        "border_radius": "6px",
                        "background": rx.color("gray", 1),
                    },
                ),
                rx.text(
                    "Pairwise W-D-L matrix · overlaid goals-per-match · "
                    "most-recent-season points · rolling 50-match win rate · "
                    "per-competition match counts · H2H meetings by season.",
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
                        rx.table.column_header_cell("Home"),
                        rx.table.column_header_cell("Score"),
                        rx.table.column_header_cell("Away"),
                        rx.table.column_header_cell("Winner"),
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
                            rx.table.cell(rx.text(m["Home"], weight="medium")),
                            rx.table.cell(
                                m["Score"],
                                style={"font_family": "monospace"},
                            ),
                            rx.table.cell(rx.text(m["Away"], weight="medium")),
                            rx.table.cell(
                                rx.badge(
                                    m["Winner"],
                                    color_scheme=rx.cond(
                                        m["Winner"] == "Draw", "amber", "orange",
                                    ),
                                    variant="surface",
                                    size="1",
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


def _matchup_header() -> rx.Component:
    return rx.heading(H2HState.header_str, size="6", weight="bold")


def h2h() -> rx.Component:
    return page(
        _header(),
        _search_grid(),
        rx.cond(
            H2HState.has_enough_teams,
            rx.cond(
                H2HState.match_count > 0,
                rx.vstack(
                    _matchup_header(),
                    _competition_chips(),
                    _record_table(),
                    _figure_card(),
                    _meetings_table(),
                    spacing="5",
                    align="start",
                    width="100%",
                ),
                rx.callout(
                    "These teams have not met in the dataset.",
                    icon="info",
                    color_scheme="gray",
                    variant="surface",
                    size="1",
                ),
            ),
            rx.callout(
                "Type at least two team names above to load the head-to-head.",
                icon="info",
                color_scheme="gray",
                variant="surface",
                size="1",
            ),
        ),
    )
