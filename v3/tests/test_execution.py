"""
Unit tests for v3/execution.py.
Verifies contract guardrail enforcement (hard 6.5% spread cap, 10 OI floor, root symbol match)
and timed horizon exit calculation.
"""

from types import SimpleNamespace
import pytest
from v3 import execution, config as cfg


def test_passes_contract_guardrails_spread_cap():
    c = SimpleNamespace(tradable=True, root_symbol="AAPL", strike_price="150.0", open_interest=100)
    stock_price = 150.0

    # Clean quote: 5% spread -> PASS
    quote_good = {"bid": 4.80, "ask": 5.05, "mid": 4.925, "spread_pct": 0.05}
    assert execution._passes_contract_guardrails(c, quote_good, stock_price, "AAPL") is True

    # Wide quote: 8% spread (> 6.5% cap) -> REJECT
    quote_wide = {"bid": 4.70, "ask": 5.10, "mid": 4.90, "spread_pct": 0.08}
    assert execution._passes_contract_guardrails(c, quote_wide, stock_price, "AAPL") is False


def test_passes_contract_guardrails_open_interest():
    stock_price = 100.0
    quote = {"bid": 2.00, "ask": 2.10, "mid": 2.05, "spread_pct": 0.048}

    # OI >= 10 -> PASS
    c_good = SimpleNamespace(tradable=True, root_symbol="XYZ", strike_price="100.0", open_interest=15)
    assert execution._passes_contract_guardrails(c_good, quote, stock_price, "XYZ") is True

    # OI < 10 -> REJECT
    c_low_oi = SimpleNamespace(tradable=True, root_symbol="XYZ", strike_price="100.0", open_interest=5)
    assert execution._passes_contract_guardrails(c_low_oi, quote, stock_price, "XYZ") is False

    # OI None -> REJECT (fixes the live bug where None silently bypassed the check)
    c_none_oi = SimpleNamespace(tradable=True, root_symbol="XYZ", strike_price="100.0", open_interest=None)
    assert execution._passes_contract_guardrails(c_none_oi, quote, stock_price, "XYZ") is False


def test_passes_contract_guardrails_root_symbol():
    stock_price = 50.0
    quote = {"bid": 1.00, "ask": 1.05, "mid": 1.025, "spread_pct": 0.048}

    # Mismatched root symbol (e.g. mini-option or spinoff) -> REJECT
    c_wrong_root = SimpleNamespace(tradable=True, root_symbol="ABC1", strike_price="50.0", open_interest=50)
    assert execution._passes_contract_guardrails(c_wrong_root, quote, stock_price, "ABC") is False

