"""Shared pytest setup. Importing bot.py constructs API clients at module scope (no network until
called), so we just ensure the env flag the codebase expects is present before that import."""
import os

os.environ.setdefault("USE_YAHOO_BARS", "1")
