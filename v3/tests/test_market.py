"""
test_market.py — unit tests for v3 market data calculations, regime checking, and risk management.
"""

from datetime import date, datetime, timezone
from unittest.mock import MagicMock
import pytest
from v3 import config as cfg
from v3 import market as mkt


def test_daily_loss_breaker_trips_at_threshold(monkeypatch):
    """Verify daily loss circuit breaker trips when realized losses cross limit."""
    # Reset state
    mkt._daily_loss_usd = 0.0
    mkt._trading_halted = False
    mkt._loss_day = datetime.now(timezone.utc).date()

    # Mock equity to $10,000. Limit is 2% = $200.
    monkeypatch.setattr(mkt, "get_account_equity", lambda: 10000.0)

    # 1. Under limit: $100 loss -> no breaker
    mkt.record_realized_loss(100.0)
    assert not mkt.check_daily_loss_breaker()
    assert not mkt._trading_halted

    # 2. Reaching limit: +$100 -> $200 total loss -> breaker trips!
    mkt.record_realized_loss(100.0)
    assert mkt.check_daily_loss_breaker()
    assert mkt._trading_halted

    # 3. New calendar date resets the breaker
    mkt._loss_day = date(2025, 1, 1)  # prior date
    assert not mkt.check_daily_loss_breaker()
    assert mkt._daily_loss_usd == 0.0
    assert not mkt._trading_halted


def test_option_quote_spread_and_mid(monkeypatch):
    """Verify mid price and bid/ask spread percentage calculations."""
    mock_quote = MagicMock()
    mock_quote.bid_price = 2.00
    mock_quote.ask_price = 2.10

    mock_client = MagicMock()
    mock_client.get_option_latest_quote.return_value = {"AAPL260925C00230000": mock_quote}
    monkeypatch.setattr(mkt, "option_data_client", mock_client)

    q = mkt.get_option_quote("AAPL260925C00230000")
    assert q is not None
    assert q["bid"] == 2.00
    assert q["ask"] == 2.10
    assert q["mid"] == 2.05
    # spread = (2.10 - 2.00) / 2.05 = 0.0488 (4.88%)
    assert q["spread_pct"] == pytest.approx(0.0488, abs=1e-4)


def test_option_quote_rejects_inverted_or_zero_ask(monkeypatch):
    """Verify quote fetch returns None if ask is zero or non-positive."""
    mock_quote = MagicMock()
    mock_quote.bid_price = 0.00
    mock_quote.ask_price = 0.00

    mock_client = MagicMock()
    mock_client.get_option_latest_quote.return_value = {"TEST": mock_quote}
    monkeypatch.setattr(mkt, "option_data_client", mock_client)

    assert mkt.get_option_quote("TEST") is None


def test_regime_gate_passes_in_healthy_uptrend(monkeypatch):
    """Verify SPY regime gate passes when above 200 SMA, positive momentum, and low chop."""
    # Reset regime cache
    mkt._regime_cache = {"date": None, "uptrend_ok": False, "mom_ok": False, "chop_ok": False, "details": {}}
    monkeypatch.setattr(cfg, "REGIME_GATE_ENABLED", True)

    # 210 days of steady upward closes: 400.0, 401.0, ..., 610.0
    mock_bars = [MagicMock(close=float(400 + i)) for i in range(210)]
    mock_client = MagicMock()
    mock_client.get_stock_bars.return_value = {"SPY": mock_bars}
    monkeypatch.setattr(mkt, "stock_data_client", mock_client)

    res = mkt.evaluate_regime()
    assert res["passes_regime"] is True
    assert res["uptrend_ok"] is True
    assert res["mom_ok"] is True
    assert res["chop_ok"] is True


def test_regime_gate_fails_below_200_sma(monkeypatch):
    """Verify SPY regime gate fails when current price is below 200d SMA."""
    mkt._regime_cache = {"date": None, "uptrend_ok": False, "mom_ok": False, "chop_ok": False, "details": {}}
    monkeypatch.setattr(cfg, "REGIME_GATE_ENABLED", True)

    # 200 days at 500.0, then sudden drop to 400.0
    mock_bars = [MagicMock(close=500.0) for _ in range(200)] + [MagicMock(close=400.0)]
    mock_client = MagicMock()
    mock_client.get_stock_bars.return_value = {"SPY": mock_bars}
    monkeypatch.setattr(mkt, "stock_data_client", mock_client)

    res = mkt.evaluate_regime()
    assert res["passes_regime"] is False
    assert res["uptrend_ok"] is False


def test_regime_gate_fails_on_excessive_chop(monkeypatch):
    """Verify SPY regime gate fails when down-day density exceeds 60% over 10 sessions."""
    mkt._regime_cache = {"date": None, "uptrend_ok": False, "mom_ok": False, "chop_ok": False, "details": {}}
    monkeypatch.setattr(cfg, "REGIME_GATE_ENABLED", True)

    # 200 steady bars, followed by 10 days with 7 down days
    base_bars = [MagicMock(close=500.0 + i) for i in range(195)]
    # Alternating downward steps: 9 intervals, 7 down
    chop_closes = [700, 690, 695, 685, 680, 675, 680, 670, 665, 660]
    all_bars = base_bars + [MagicMock(close=float(c)) for c in chop_closes]

    mock_client = MagicMock()
    mock_client.get_stock_bars.return_value = {"SPY": all_bars}
    monkeypatch.setattr(mkt, "stock_data_client", mock_client)

    res = mkt.evaluate_regime()
    assert res["chop_ok"] is False
    assert res["passes_regime"] is False

