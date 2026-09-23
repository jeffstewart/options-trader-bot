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


def test_monitor_open_positions_triggers_horizon_exit(monkeypatch, tmp_path):
    """Verify monitor triggers timed exit when holding period exceeds horizon."""
    from datetime import datetime, timedelta, timezone
    from v3 import market as mkt

    closed_signals = []
    monkeypatch.setattr(execution, "close_option_position",
                        lambda sym, reason: closed_signals.append((sym, reason)))

    # Position entered 35 minutes ago (horizon is 30m)
    now = datetime.now(timezone.utc)
    entry_35m_ago = (now - timedelta(minutes=35)).isoformat()
    mkt._monitored_positions = {
        "TEST_SYM": {
            "symbol": "TEST_SYM",
            "underlying": "TEST",
            "qty": 1,
            "entry_price": 2.00,
            "entry_time": entry_35m_ago,
            "peak_price": 2.00,
        }
    }

    execution.monitor_open_positions()
    assert len(closed_signals) == 1
    assert closed_signals[0] == ("TEST_SYM", "horizon_30m")


def test_monitor_open_positions_holds_under_horizon(monkeypatch):
    """Verify monitor leaves position open when holding period is under 30 minutes."""
    from datetime import datetime, timedelta, timezone
    from v3 import market as mkt

    closed_signals = []
    monkeypatch.setattr(execution, "close_option_position",
                        lambda sym, reason: closed_signals.append((sym, reason)))
    monkeypatch.setattr(mkt, "get_option_quote",
                        lambda sym: {"bid": 2.50, "ask": 2.60, "mid": 2.55, "spread_pct": 0.04})

    # Position entered 10 minutes ago
    now = datetime.now(timezone.utc)
    entry_10m_ago = (now - timedelta(minutes=10)).isoformat()
    mkt._monitored_positions = {
        "HOLD_SYM": {
            "symbol": "HOLD_SYM",
            "underlying": "HOLD",
            "qty": 1,
            "entry_price": 2.00,
            "entry_time": entry_10m_ago,
            "peak_price": 2.00,
        }
    }

    execution.monitor_open_positions()
    assert len(closed_signals) == 0
    # Peak price updated to 2.50
    assert mkt._monitored_positions["HOLD_SYM"]["peak_price"] == 2.50


def test_save_and_load_state_roundtrip(tmp_path, monkeypatch):
    """Verify state file serialization and deserialization."""
    from v3 import market as mkt

    state_file = tmp_path / "bot_state.json"
    monkeypatch.setattr(execution, "STATE_FILE", state_file)

    test_pos = {
        "TEST_ROUNDTRIP": {
            "symbol": "TEST_ROUNDTRIP",
            "underlying": "TRIP",
            "qty": 2,
            "entry_price": 1.50,
            "entry_time": "2026-09-22T18:00:00+00:00",
            "peak_price": 1.80,
        }
    }
    mkt._monitored_positions = test_pos
    execution.save_state()

    # Clear and reload
    mkt._monitored_positions = {}
    execution.load_state()
    assert "TEST_ROUNDTRIP" in mkt._monitored_positions
    assert mkt._monitored_positions["TEST_ROUNDTRIP"]["entry_price"] == 1.50


