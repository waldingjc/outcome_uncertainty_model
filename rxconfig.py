"""Reflex configuration for the outcome-uncertainty webapp.

Run with:
    py -3.14 -m reflex run

The first run will download a small JavaScript runtime (Bun) and build a
Next.js bundle in `.web/`. Subsequent runs are fast.

# Sharing the running app
Reflex has a two-port architecture — the Next.js frontend (3000) renders
in the visitor's browser and tries to reach the FastAPI backend (8000)
over WebSocket. The frontend needs to know the backend URL *from the
visitor's perspective*, not the host's. We read that URL from
`REFLEX_API_URL` so you can flip between localhost / LAN IP / a tunnel
domain without editing this file.

Two ready-made wrappers in `scripts/`:

  scripts/share-lan.ps1       — auto-detects your LAN IP, runs reflex
                                bound to 0.0.0.0 so friends on the same
                                Wi-Fi can hit http://<your-ip>:3000.

  scripts/share-tunnel.ps1    — paste a Cloudflare quick-tunnel URL,
                                runs reflex against it. Works for
                                viewers anywhere on the internet.
"""

import os

import reflex as rx

# Default to localhost — anything else is a sharing scenario, opted into
# via environment variable so the file stays clean for git.
_API_URL = os.environ.get("REFLEX_API_URL", "http://localhost:8000")

config = rx.Config(
    app_name="webapp",
    # All output (compiled JS, build cache) goes under .web/, which is
    # gitignored.
    frontend_port=3000,
    backend_port=8000,
    api_url=_API_URL,
)
