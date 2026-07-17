"""Shared pytest setup. config.py reads ALPACA_API_KEY/SECRET at import time (raises if
missing) and market.py constructs an Alpaca TradingClient at module scope (no network call
until a method is invoked) — set dummy credentials before anything imports those modules."""
import os

os.environ.setdefault("ALPACA_API_KEY", "test-key")
os.environ.setdefault("ALPACA_SECRET_KEY", "test-secret")
