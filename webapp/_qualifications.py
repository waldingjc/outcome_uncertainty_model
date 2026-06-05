"""Per-league qualification + relegation slot definitions.

Tells the league-ladders page which finishing positions correspond to
which European competition, promotion, playoff, or relegation. The
rules differ wildly by country and even by season — we encode a
sensible "current" rule set (post-2024 reforms where the Champions
League expanded to 36 teams) for the leagues we cover most heavily.

Leagues not listed here just don't get any row highlighting. That's
intentional — better to omit highlighting than to display a wrong rule.

The slot kinds (`UCL`, `UEL`, `UECL`, `PROMO`, `PLAYOFF`, `RELEG`) are
shared across leagues so the legend on the page only needs one row per
kind.
"""

from __future__ import annotations

from dataclasses import dataclass


# Slot kinds — string constants so they can flow into Reflex state as
# plain dict values without needing a custom serialiser.
UCL     = "UCL"      # Champions League group stage / league phase
UEL     = "UEL"      # Europa League
UECL    = "UECL"     # Conference League
PROMO   = "PROMO"    # Direct promotion to the tier above
PLAYOFF = "PLAYOFF"  # Promotion playoff (and relegation playoff in some leagues)
RELEG   = "RELEG"    # Direct relegation


@dataclass(frozen=True)
class _Slots:
    """How positions translate into qualification labels for a league.

    `top`/`mid`/`bottom` are evaluated in order — `top` covers the
    best finishes (UCL/UEL/promotion), `mid` covers playoffs which
    sit between top finishers and the safe pack, and `bottom` covers
    relegation. Positions count from 1.

    Each entry is `(count, kind)` — e.g. `(5, UCL)` means positions
    1..5 inclusive get the UCL slot.
    """
    top:    list[tuple[int, str]]      # from position 1 downward
    mid:    list[tuple[int, str]]      # consecutive after top
    bottom: list[tuple[int, str]]      # from last position upward


# League IDs come from API-Football. The big-5 entries reflect the
# post-2024 European competition reforms: 5 UCL slots from the top
# associations (rank-bonus included), 1 UEL, 1 UECL.
_RULES: dict[int, _Slots] = {
    # ---- Big-5 ------------------------------------------------------
    # Premier League — England
    39: _Slots(
        top=[(5, UCL), (1, UEL), (1, UECL)],
        mid=[],
        bottom=[(3, RELEG)],
    ),
    # La Liga — Spain
    140: _Slots(
        top=[(5, UCL), (1, UEL), (1, UECL)],
        mid=[],
        bottom=[(3, RELEG)],
    ),
    # Bundesliga — Germany (18 teams; 4 UCL, 1 UEL, 1 UECL,
    # 16 relegation playoff, 17-18 direct relegation)
    78: _Slots(
        top=[(4, UCL), (1, UEL), (1, UECL)],
        mid=[],
        bottom=[(2, RELEG), (1, PLAYOFF)],
    ),
    # Serie A — Italy
    135: _Slots(
        top=[(5, UCL), (1, UEL), (1, UECL)],
        mid=[],
        bottom=[(3, RELEG)],
    ),
    # Ligue 1 — France (18 teams; UCL 1-3, UEL 4, UECL 5,
    # 16 relegation playoff, 17-18 direct relegation)
    61: _Slots(
        top=[(3, UCL), (1, UEL), (1, UECL)],
        mid=[],
        bottom=[(2, RELEG), (1, PLAYOFF)],
    ),

    # ---- English pyramid -------------------------------------------
    # Championship — England (tier 2; 1-2 promo, 3-6 playoff, 22-24 releg)
    40: _Slots(
        top=[(2, PROMO)],
        mid=[(4, PLAYOFF)],
        bottom=[(3, RELEG)],
    ),
    # League One — England (tier 3; 1-2 promo, 3-6 playoff, 21-24 releg)
    41: _Slots(
        top=[(2, PROMO)],
        mid=[(4, PLAYOFF)],
        bottom=[(4, RELEG)],
    ),
    # League Two — England (tier 4; 1-3 promo, 4-7 playoff, 24 releg)
    42: _Slots(
        top=[(3, PROMO)],
        mid=[(4, PLAYOFF)],
        bottom=[(2, RELEG)],
    ),

    # ---- Other top-flights we cover --------------------------------
    # Eredivisie — Netherlands
    88: _Slots(
        top=[(1, UCL), (1, UEL), (2, UECL)],
        mid=[],
        bottom=[(1, RELEG), (1, PLAYOFF)],
    ),
    # Primeira Liga — Portugal
    94: _Slots(
        top=[(3, UCL), (1, UEL), (1, UECL)],
        mid=[],
        bottom=[(2, RELEG)],
    ),
    # Belgian Pro League — Belgium
    144: _Slots(
        top=[(1, UCL), (1, UEL), (1, UECL)],
        mid=[],
        bottom=[(1, RELEG)],
    ),
    # Scottish Premiership
    179: _Slots(
        top=[(1, UCL), (1, UEL), (1, UECL)],
        mid=[],
        bottom=[(1, RELEG), (1, PLAYOFF)],
    ),
    # Turkish Super Lig
    203: _Slots(
        top=[(1, UCL), (2, UEL), (1, UECL)],
        mid=[],
        bottom=[(2, RELEG)],
    ),
}


# Human-readable display labels + Radix color schemes for each kind.
# The colour scheme strings are what `rx.color(scheme, shade)` accepts.
LABELS: dict[str, str] = {
    UCL:     "UCL",
    UEL:     "UEL",
    UECL:    "UECL",
    PROMO:   "Promotion",
    PLAYOFF: "Playoff",
    RELEG:   "Relegation",
}

COLORS: dict[str, str] = {
    UCL:     "blue",      # the famous starry blue of the UCL logo
    UEL:     "orange",    # UEL keeps the project accent
    UECL:    "green",     # green = Conference
    PROMO:   "cyan",      # cyan promotions stand out next to UCL blue
    PLAYOFF: "amber",     # amber playoff — "in contention"
    RELEG:   "red",
}

# Legend order — top-to-bottom, matching how positions read down the table.
ORDER: list[str] = [UCL, UEL, UECL, PROMO, PLAYOFF, RELEG]


def qualification_for(league_id: int, pos: int, total: int) -> str | None:
    """Return the qualification kind for `pos` in `league_id`, or None
    if no slot applies (or the league isn't in `_RULES`)."""
    rules = _RULES.get(league_id)
    if rules is None:
        return None

    # Top slots — count downward from position 1.
    cursor = 0
    for count, kind in rules.top:
        if cursor < pos <= cursor + count:
            return kind
        cursor += count

    # Mid slots — continue from where top finished.
    for count, kind in rules.mid:
        if cursor < pos <= cursor + count:
            return kind
        cursor += count

    # Bottom slots — count upward from the last position.
    cursor_bot = total
    for count, kind in rules.bottom:
        if cursor_bot - count < pos <= cursor_bot:
            return kind
        cursor_bot -= count

    return None
