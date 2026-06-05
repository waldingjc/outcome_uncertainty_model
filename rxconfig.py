"""Reflex configuration for the outcome-uncertainty webapp.

Run with:
    py -3.14 -m reflex run

The first run will download a small JavaScript runtime (Bun) and build a
Next.js bundle in `.web/`. Subsequent runs are fast.
"""

import reflex as rx

config = rx.Config(
    app_name="webapp",
    # All output (compiled JS, build cache) goes under .web/, which is
    # gitignored.
    frontend_port=3000,
    backend_port=8000,
)
