"""League ladders page — pick a league + season, see the points table
with Elo overlaid."""

from __future__ import annotations

import reflex as rx

from webapp import _qualifications as _quals
from webapp.components.layout import page
from webapp.state.leagues import LeagueState


def _qual_color(qual_var, shade: int) -> rx.Var:
    """Map a row's `qual` field (string Var) onto the project color for
    that qualification kind. Returns gray for "" (no qualification) so
    we can use this colour as a transparent-ish accent on every row.
    """
    return rx.match(
        qual_var,
        (_quals.UCL,     rx.color(_quals.COLORS[_quals.UCL],     shade)),
        (_quals.UEL,     rx.color(_quals.COLORS[_quals.UEL],     shade)),
        (_quals.UECL,    rx.color(_quals.COLORS[_quals.UECL],    shade)),
        (_quals.PROMO,   rx.color(_quals.COLORS[_quals.PROMO],   shade)),
        (_quals.PLAYOFF, rx.color(_quals.COLORS[_quals.PLAYOFF], shade)),
        (_quals.RELEG,   rx.color(_quals.COLORS[_quals.RELEG],   shade)),
        rx.color("gray", shade),  # default: no qual
    )


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
    """One row of the points table.

    The position cell carries a coloured left-border stripe whose hue
    indicates the qualification slot (UCL/UEL/UECL/promotion/playoff/
    relegation). Rows with no qualification get a neutral gray stripe
    so the column alignment stays consistent.
    """
    stripe = _qual_color(row["qual"], 9)
    return rx.table.row(
        rx.table.cell(
            rx.text(
                row["pos"].to_string(),
                style={"font_family": "monospace"},
                color=rx.color("gray", 11),
            ),
            style={
                "border_left": "4px solid",
                "border_left_color": stripe,
                "padding_left": "10px",
            },
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
        # Subtle row tint matching the qual colour — only visible for
        # rows with a real slot, since gray-3 ~= the table background.
        style={"background": _qual_color(row["qual"], 2)},
    )


def _qual_label(kind_var) -> rx.Var:
    """Map a Reflex Var holding a qual kind to its human label."""
    return rx.match(
        kind_var,
        (_quals.UCL,     _quals.LABELS[_quals.UCL]),
        (_quals.UEL,     _quals.LABELS[_quals.UEL]),
        (_quals.UECL,    _quals.LABELS[_quals.UECL]),
        (_quals.PROMO,   _quals.LABELS[_quals.PROMO]),
        (_quals.PLAYOFF, _quals.LABELS[_quals.PLAYOFF]),
        (_quals.RELEG,   _quals.LABELS[_quals.RELEG]),
        "",
    )


def _legend_chip(kind) -> rx.Component:
    """A small swatch + label for one qualification kind, used in the
    legend strip above the table. `kind` is a Reflex Var here because
    we're inside an rx.foreach — both the color and the label have to
    be expressed as rx.match() Vars, not Python dict lookups."""
    return rx.hstack(
        rx.box(
            width="10px",
            height="10px",
            border_radius="2px",
            background=_qual_color(kind, 9),
        ),
        rx.text(
            _qual_label(kind),
            size="1",
            color=rx.color("gray", 11),
        ),
        spacing="2",
        align="center",
    )


def _legend() -> rx.Component:
    """Legend strip — only shows when the current league has known
    qualification rules. Drives off `LeagueState.active_quals`."""
    return rx.cond(
        LeagueState.active_quals.length() > 0,
        rx.hstack(
            rx.text(
                "Slots:",
                size="1",
                weight="medium",
                color=rx.color("gray", 11),
                style={"text_transform": "uppercase", "letter_spacing": "0.05em"},
            ),
            rx.foreach(LeagueState.active_quals, _legend_chip),
            spacing="4",
            align="center",
            wrap="wrap",
        ),
        rx.fragment(),
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
            _legend(),
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
