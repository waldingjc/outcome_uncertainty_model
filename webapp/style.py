"""Theme constants for the webapp.

We use Reflex's Radix-based theme system. The accent + grey are chosen to
echo the matplotlib figure palette (teal #264653 + green #2A9D8F) so the
web app and the static figures feel like one product.
"""

# Theme — passed to rx.theme() in webapp.py
THEME_APPEARANCE = "dark"
THEME_ACCENT_COLOR = "teal"
THEME_GRAY_COLOR = "slate"
THEME_RADIUS = "medium"
THEME_PANEL_BACKGROUND = "solid"

# Layout constants
SIDEBAR_WIDTH = "240px"
CONTENT_MAX_WIDTH = "1400px"

# Reusable styling dicts. Reflex accepts standard CSS-in-Python.
PAGE_PADDING = {"padding": "2em", "padding_top": "1.5em"}
SECTION_GAP = "1.5em"
