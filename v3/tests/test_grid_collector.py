"""
Unit tests for v3/grid_collector.py.
Verifies normalized two-file schema (signals metadata + snapshot time series),
same-day collection bounds, contract selection, and Call/Put support.
"""

import asyncio
import csv
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from alpaca.trading.enums import ContractType

from v3 import config as cfg
from v3 import grid_collector


def test_should_collect():
    # Above threshold 0.20
    assert grid_collector.should_collect({"magnitude": 0.5, "confidence": 0.8}) is True
    assert grid_collector.should_collect({"magnitude": 0.45, "confidence": 0.5}) is True
    # Below threshold 0.20
    assert grid_collector.should_collect({"magnitude": 0.2, "confidence": 0.5}) is False
    assert grid_collector.should_collect({"magnitude": 0.1, "confidence": 0.9}) is False
    # Invalid / missing values
    assert grid_collector.should_collect({}) is False
    assert grid_collector.should_collect({"magnitude": "invalid"}) is False


def test_make_signal_id():
    ts = datetime(2026, 9, 29, 14, 30, 45, 123456, tzinfo=timezone.utc)
    sig_id = grid_collector.make_signal_id("NVDA", ts)
    assert sig_id == "NVDA_20260929T143045123456"


def test_interval_for():
    # 0-15 min -> 30s
    assert grid_collector._interval_for(5.0) == 30
    assert grid_collector._interval_for(15.0) == 30
    # 15-60 min -> 180s
    assert grid_collector._interval_for(25.0) == 180
    assert grid_collector._interval_for(60.0) == 180
    # 1-6.5 hrs -> 900s
    assert grid_collector._interval_for(120.0) == 900
    assert grid_collector._interval_for(300.0) == 900


def test_gate_eligibility():
    # Bullish with high mag/conf clears call gate
    call_ok, lotto_ok = grid_collector._gate_eligibility("bullish", 0.50, 0.80)
    assert call_ok is True
    assert lotto_ok is False

    # Lotto thresholds
    call_ok, lotto_ok = grid_collector._gate_eligibility("bullish", 0.75, 0.90)
    assert call_ok is True
    assert lotto_ok is True

    # Bearish signals do not clear the long call trade gate
    call_ok, lotto_ok = grid_collector._gate_eligibility("bearish", 0.75, 0.90)
    assert call_ok is False
    assert lotto_ok is False


def test_write_signal_row(tmp_path, monkeypatch):
    """Verify single-row metadata write to contract_grid_signals.csv."""
    sig_csv = tmp_path / "contract_grid_signals.csv"
    monkeypatch.setattr(cfg, "GRID_SIGNALS_CSV", sig_csv)

    state = {
        "signal_ts": datetime(2026, 9, 29, 14, 0, 0, tzinfo=timezone.utc),
        "ticker": "AAPL",
        "headline": "Apple Beats Q3",
        "source": "Benzinga",
        "scorer_model": "llama3.2",
        "sentiment": "bullish",
        "magnitude": 0.75,
        "confidence": 0.85,
        "catalyst": "Earnings",
        "stock_price_at_signal": 225.0,
        "spy_price": 560.0,
        "spy_sma": 550.0,
        "regime_uptrend": True,
        "regime_mom_ok": True,
        "regime_chop_ok": True,
        "regime_dd_density": 0.2,
        "vol_proxy_symbol": "VIXY",
        "vol_proxy_price": 14.5,
        "vol_proxy_pctile": 0.45,
        "passes_news_call_gate": True,
        "passes_lotto_gate": False,
    }

    grid_collector._write_signal_row("sig_123", state)

    assert sig_csv.exists()
    with open(sig_csv, "r") as f:
        reader = list(csv.reader(f))
        assert len(reader) == 2  # Header + 1 row
        header = reader[0]
        assert header == grid_collector._SIGNAL_HEADER
        row_dict = dict(zip(header, reader[1]))
        assert row_dict["signal_id"] == "sig_123"
        assert row_dict["ticker"] == "AAPL"
        assert row_dict["sentiment"] == "bullish"
        assert float(row_dict["magnitude"]) == 0.75


def test_write_snapshot_rows(tmp_path, monkeypatch):
    """Verify recurring snapshots write compact, non-redundant rows to contract_grid_snapshots.csv."""
    snap_csv = tmp_path / "contract_grid_snapshots.csv"
    monkeypatch.setattr(cfg, "GRID_SNAPSHOT_CSV", snap_csv)

    row1 = ["sig_1", "2026-09-29T14:00:05+00:00", 0.08, 220.5, "AAPL261016C00220000",
            "call", 220.0, "2026-10-16", 17, 1500, 4.20, 4.30, 4.25, 0.0235, 0.28,
            0.52, 0.03, -0.05, 0.15, 0.08]

    row2 = ["sig_1", "2026-09-29T14:00:05+00:00", 0.08, 220.5, "AAPL261016C00225000",
            "call", 225.0, "2026-10-16", 17, 2100, 2.10, 2.20, 2.15, 0.0465, 0.27,
            0.42, 0.04, -0.04, 0.14, 0.06]

    grid_collector._write_snapshot_rows([row1, row2])

    assert snap_csv.exists()
    with open(snap_csv, "r") as f:
        reader = list(csv.reader(f))
        assert len(reader) == 3  # Header + 2 rows
        header = reader[0]
        assert header == grid_collector._SNAPSHOT_HEADER
        # Notice redundant columns are completely absent:
        assert "headline" not in header
        assert "source" not in header
        assert "scorer_model" not in header
        assert "regime_uptrend" not in header

        row1_dict = dict(zip(header, reader[1]))
        assert row1_dict["signal_id"] == "sig_1"
        assert row1_dict["symbol"] == "AAPL261016C00220000"
        assert row1_dict["contract_type"] == "call"
        assert float(row1_dict["bid"]) == 4.20


def test_fetch_contracts_meta():
    mock_contract_call = MagicMock()
    mock_contract_call.symbol = "AAPL261016C00220000"
    mock_contract_call.strike_price = "220.0"
    mock_contract_call.expiration_date = date(2026, 10, 16)
    mock_contract_call.open_interest = 500
    mock_contract_call.tradable = True
    mock_contract_call.root_symbol = "AAPL"

    mock_resp = MagicMock()
    mock_resp.option_contracts = [mock_contract_call]
    mock_resp.next_page_token = None

    with patch.object(grid_collector.trading_client, "get_option_contracts", return_value=mock_resp) as mock_get:
        # 1. Test CALL fetch
        meta_calls = grid_collector._fetch_contracts_meta("AAPL", 220.0, ContractType.CALL)
        assert "AAPL261016C00220000" in meta_calls
        strike, exp, dte, oi = meta_calls["AAPL261016C00220000"]
        assert strike == 220.0
        assert exp == "2026-10-16"
        assert oi == 500
        req_call = mock_get.call_args[0][0]
        assert req_call.type == ContractType.CALL

        # 2. Test PUT fetch
        mock_contract_put = MagicMock()
        mock_contract_put.symbol = "AAPL261016P00220000"
        mock_contract_put.strike_price = "220.0"
        mock_contract_put.expiration_date = date(2026, 10, 16)
        mock_contract_put.open_interest = 650
        mock_contract_put.tradable = True
        mock_contract_put.root_symbol = "AAPL"
        mock_resp.option_contracts = [mock_contract_put]

        meta_puts = grid_collector._fetch_contracts_meta("AAPL", 220.0, ContractType.PUT)
        assert "AAPL261016P00220000" in meta_puts
        req_put = mock_get.call_args[0][0]
        assert req_put.type == ContractType.PUT


def test_snapshot_rows_call_and_put():
    state_call = {
        "ticker": "AAPL",
        "headline": "Apple Beats Q3",
        "source": "Benzinga",
        "scorer_model": "llama3.2",
        "sentiment": "bullish",
        "contract_type": ContractType.CALL,
        "magnitude": 0.6,
        "confidence": 0.8,
        "catalyst": "Earnings beat",
        "stock_price_at_signal": 220.0,
        "spy_price": 560.0,
        "spy_sma": 550.0,
        "regime_uptrend": True,
        "regime_mom_ok": True,
        "regime_chop_ok": True,
        "regime_dd_density": 0.2,
        "vol_proxy_symbol": "VIXY",
        "vol_proxy_price": 14.5,
        "vol_proxy_pctile": 0.45,
        "passes_news_call_gate": True,
        "passes_lotto_gate": False,
        "signal_ts": datetime.now(timezone.utc),
        "contracts_meta": {
            "AAPL261016C00220000": (220.0, "2026-10-16", 17, 1000)
        },
        "get_stock_price": lambda t: 221.0,
    }

    mock_quote = MagicMock()
    mock_quote.bid_price = 4.20
    mock_quote.ask_price = 4.30

    mock_greeks = MagicMock()
    mock_greeks.delta = 0.52
    mock_greeks.gamma = 0.03
    mock_greeks.theta = -0.05
    mock_greeks.vega = 0.15
    mock_greeks.rho = 0.08

    mock_snap_obj = MagicMock()
    mock_snap_obj.latest_quote = mock_quote
    mock_snap_obj.greeks = mock_greeks
    mock_snap_obj.implied_volatility = 0.28

    mock_snap_resp = {"AAPL261016C00220000": mock_snap_obj}

    with patch.object(grid_collector.option_data_client, "get_option_snapshot", return_value=mock_snap_resp):
        rows = grid_collector._snapshot_rows("sig_call_1", state_call)
        assert len(rows) == 1
        row_dict = dict(zip(grid_collector._SNAPSHOT_HEADER, rows[0]))
        # Verify contract_type is recorded as 'call'
        assert row_dict["signal_id"] == "sig_call_1"
        assert row_dict["contract_type"] == "call"
        assert row_dict["symbol"] == "AAPL261016C00220000"
        assert row_dict["bid"] == 4.20
        assert row_dict["ask"] == 4.30
        assert row_dict["mid"] == 4.25

    # Now verify PUT
    state_put = dict(state_call)
    state_put["sentiment"] = "bearish"
    state_put["contract_type"] = ContractType.PUT
    state_put["contracts_meta"] = {
        "AAPL261016P00220000": (220.0, "2026-10-16", 17, 1200)
    }
    mock_greeks.delta = -0.48
    mock_snap_resp_put = {"AAPL261016P00220000": mock_snap_obj}

    with patch.object(grid_collector.option_data_client, "get_option_snapshot", return_value=mock_snap_resp_put):
        rows_put = grid_collector._snapshot_rows("sig_put_1", state_put)
        assert len(rows_put) == 1
        row_p_dict = dict(zip(grid_collector._SNAPSHOT_HEADER, rows_put[0]))
        assert row_p_dict["signal_id"] == "sig_put_1"
        assert row_p_dict["contract_type"] == "put"
        assert row_p_dict["symbol"] == "AAPL261016P00220000"


def test_run_poll_loop_same_day_pruning():
    """Verify signals are pruned when trading day finishes or session minutes exceed bound."""
    async def _run():
        grid_collector._tracked.clear()

        now = datetime.now(timezone.utc)
        # Signal 1: Created yesterday -> must be pruned
        yesterday = now - timedelta(days=1)
        grid_collector._tracked["sig_old"] = {
            "ticker": "OLD",
            "signal_ts": yesterday,
            "last_poll": yesterday,
            "last_expiry": (now + timedelta(days=10)).date(),
        }

        # Signal 2: Created 400 minutes ago today -> exceeds 390m trading session limit -> must be pruned
        grid_collector._tracked["sig_session_over"] = {
            "ticker": "SESSION",
            "signal_ts": now - timedelta(minutes=400),
            "last_poll": now - timedelta(minutes=20),
            "last_expiry": (now + timedelta(days=10)).date(),
        }

        # Signal 3: Created 10 minutes ago -> active
        grid_collector._tracked["sig_active"] = {
            "ticker": "ACTIVE",
            "signal_ts": now - timedelta(minutes=10),
            "last_poll": now,
            "last_expiry": (now + timedelta(days=10)).date(),
        }

        # Run one pass of poll loop
        task = asyncio.create_task(grid_collector.run_poll_loop(market_is_open=lambda: True, poll_sleep_s=0.01))
        await asyncio.sleep(0.05)
        task.cancel()
        try:
            await task
        except asyncio.CancelledError:
            pass

        assert "sig_old" not in grid_collector._tracked
        assert "sig_session_over" not in grid_collector._tracked
        assert "sig_active" in grid_collector._tracked
        grid_collector._tracked.clear()

    asyncio.run(_run())

