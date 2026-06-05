"""League ladders page — pick a league + season, see the points table
with Elo overlaid."""

from __future__ import annotations

import reflex as rx

from webapp.components.layout import page
from webapp.state.leagues import LeagueState


def _controls() -> rx.Component:
    """League + season dropdowns."""
    return rx.hstack(
        rx.vstack(
            rx.text(
                "League",
                size="1",
                weight="medium",
                color=rx.color("gray", 11),
                style={"text_transform": "uppercase", "letter_spacing": "0.05em"},
            ),
            rx.select.root(
                rx.select.trigger(placeholder="Select a league…", variant="surface"),
                rx.select.content(
                    rx.foreach(
                        LeagueState.league_options,
                        lambda opt: rx.select.item(opt["label"], value=opt["value"]),
                    ),
                ),
                value=LeagueState.league_id.to_string(),
                on_change=LeagueState.set_league,
                size="2",
            ),
            spacing="1",
            align="start",
            min_width="300px",
        ),
        rx.vstack(
            rx.text(
                "Season",
                size="1",
                weight="medium",
                color=rx.color("gray", 11),
                style={"text_transform": "uppercase", "letter_spacing": "0.05em"},
            ),
            rx.select.root(
                rx.select.trigger(placeholder="Season…", variant="surface"),
                rx.select.content(
                    rx.foreach(
                        LeagueState.season_options,
                        lambda s: rx.select.item(
                            rx.fragment(s, "–", s),  # crude "2024–25"
                            value=s,
                        ),
                    ),
                ),
                value=LeagueState.season.to_string(),
                on_change=LeagueState.set_season,
                size="2",
            ),
            spacing="1",
            align="start",
            min_width="160px",
        ),
        spacing="4",
        align="end",
        width="100%",
    )


def _sortable_header(label: str, key: str, width: str | None = None) -> rx.Component:
    """A clickable column header that wires into LeagueState.sort_by(key).
    Shows an arrow indicator (↑/↓) when this column is the active sort."""
    is_active = LeagueState.sort_key == key
    arrow = rx.cond(
        is_active,
        rx.cond(LeagueState.sort_dir == "asc", " ▲", " ▼"),
        "",
    )
    return rx.table.column_header_cell(
        rx.hstack(
            rx.text(
                label,
                weight=rx.cond(is_active, "bold", "medium"),
                color=rx.cond(
                    is_active,
                    rx.color("orange", 11),
                    rx.color("gray", 12),
                ),
            ),
            rx.text(
                arrow,
                size="1",
                color=rx.color("orange", 11),
            ),
            spacing="1",
            align="center",
            style={
                "cursor": "pointer",
                "user_select": "none",
                "_hover": {"color": rx.color("orange", 11)},
            },
            on_click=LeagueState.sort_by(key),
        ),
        width=width,
    )


def _table_header() -> rx.Component:
    return rx.table.header(
        rx.table.row(
            _sortable_header("#",    "pos",  "40px"),
            _sortable_header("Team", "team"),
            _sortable_header("P",    "P",   "48px"),
            _sortable_header("W",    "W",   "48px"),
            _sortable_header("D",    "D",   "48px"),
            _sortable_header("L",    "L",   "48px"),
            _sortable_header("GF",   "GF",  "56px"),
            _sortable_header("GA",   "GA",  "56px"),
            _sortable_header("GD",   "GD",  "64px"),
            _sortable_header("Pts",  "Pts", "64px"),
            _sortable_header("Elo",  "Elo", "80px"),
        ),
    )


def _table_row(row) -> rx.Component:
    """One row of the points table. Top 4 highlighted with accent;
    bottom 3 with a subtle red tint for the relegation feel."""
    return rx.table.row(
        rx.table.cell(
            rx.text(
                row["pos"].to_string(),
                style={"font_family": "monospace"},
                color=rx.color("gray", 11),
            ),
        ),
        rx.table.cell(rx.text(row["team"], weight="medium")),
        rx.table.cell(row["P"].to_string()),
        rx.table.cell(row["W"].to_string()),
        rx.table.cell(row["D"].to_string()),
        rx.table.cell(row["L"].to_string()),
        rx.table.cell(row["GF"].to_string()),
        rx.table.cell(row["GA"].to_string()),
        rx.table.cell(row["GD"].to_string(), style={"font_family": "monospace"}),
        rx.table.cell(
            rx.text(row["Pts"].to_string(), weight="bold"),
        ),
        rx.table.cell(
            rx.text(
                row["Elo"],
                style={"font_family": "monospace"},
                color=rx.color("orange", 11),
            ),
        ),
    )


def _table() -> rx.Component:
    return rx.card(
        rx.vstack(
            rx.hstack(
                rx.icon("trophy", size=18, color=rx.color("orange", 10)),
                rx.text(
                    LeagueState.header_str,
                    size="3",
                    weight="medium",
                    color=rx.color("gray", 12),
                ),
                spacing="2",
                align="center",
                width="100%",
            ),
            rx.divider(),
            rx.cond(
                LeagueState.has_rows,
                rx.table.root(
                    _table_header(),
                    rx.table.body(
                        rx.foreach(LeagueState.table_rows, _table_row),
                    ),
                    variant="surface",
                    size="2",
                    width="100%",
                ),
                rx.callout(
                    "No fixtures in this league/season. Try another combination.",
                    icon="info",
                    color_scheme="gray",
                    variant="surface",
                    size="1",
                ),
            ),
            spacing="3",
            align="start",
            width="100%",
        ),
        size="3",
        variant="surface",
        width="100%",
    )


def leagues() -> rx.Component:
    return page(
        rx.vstack(
            rx.heading("League ladders", size="7", weight="bold"),
            rx.text(
                "Final-season points table for any league we've ingested, "
                "with each team's current Elo overlaid.",
                color=rx.color("gray", 11),
                size="2",
            ),
            spacing="2",
            align="start",
        ),
        _controls(),
        _table(),
    )
