"""
test_contract_strategy_tuner.py

Unit tests for research/contract_strategy_tuner.py covering:
  - Two-pass memory-efficient loading (load_entry_snapshot_and_stats, load_selected_paths)
  - Contract path preparation (prep_contract_paths)
  - Selection rules and exit strategies
  - Gate evaluation and notable trades extraction
"""

import os
import tempfile
import numpy as np
import pandas as pd
import pytest

from research.contract_strategy_tuner import (
    load_entry_snapshot_and_stats,
    load_selected_paths,
    prep_contract_paths,
    get_signal_meta,
    chronological_split,
    SELECTION_RULES,
    EXIT_STRATEGIES,
    simulate,
    evaluate_split,
    build_broad_feature_table,
    apply_rule_gate,
    apply_model_gate,
    notable_trades,
)


@pytest.fixture
def sample_csv():
    """Create a small temporary CSV matching contract_grid_snapshots schema."""
    rows = [
        # Signal 1, Contract A (entry + 2 path updates)
        {
            "signal_id": "SIG_1", "signal_ts": "2026-08-17T13:30:00+00:00", "ticker": "AAPL",
            "headline": "Apple announcement", "source": "Benzinga", "scorer_model": "llama3.2",
            "magnitude": 0.85, "confidence": 0.90, "catalyst": 0.80, "stock_price_at_signal": 150.0,
            "spy_price": 450.0, "spy_sma": 445.0, "regime_uptrend": True, "regime_mom_ok": True,
            "regime_chop_ok": False, "regime_dd_density": 0.1, "vol_proxy_symbol": "VIXY",
            "vol_proxy_price": 15.0, "vol_proxy_pctile": 0.2, "passes_news_call_gate": True,
            "passes_lotto_gate": False, "snapshot_ts": "2026-08-17T13:30:01+00:00",
            "minutes_since_signal": 0.02, "stock_price": 150.0, "symbol": "AAPL260821C00150000",
            "strike": 150.0, "expiry": "2026-08-21", "dte_at_signal": 4, "oi_at_signal": 100,
            "bid": 2.0, "ask": 2.2, "mid": 2.1, "spread_pct": 0.09, "iv": 0.25,
            "delta": 0.52, "gamma": 0.05, "theta": -0.05, "vega": 0.1, "rho": 0.02,
        },
        {
            "signal_id": "SIG_1", "signal_ts": "2026-08-17T13:30:00+00:00", "ticker": "AAPL",
            "headline": "Apple announcement", "source": "Benzinga", "scorer_model": "llama3.2",
            "magnitude": 0.85, "confidence": 0.90, "catalyst": 0.80, "stock_price_at_signal": 150.0,
            "spy_price": 450.0, "spy_sma": 445.0, "regime_uptrend": True, "regime_mom_ok": True,
            "regime_chop_ok": False, "regime_dd_density": 0.1, "vol_proxy_symbol": "VIXY",
            "vol_proxy_price": 15.0, "vol_proxy_pctile": 0.2, "passes_news_call_gate": True,
            "passes_lotto_gate": False, "snapshot_ts": "2026-08-17T13:45:00+00:00",
            "minutes_since_signal": 15.0, "stock_price": 151.0, "symbol": "AAPL260821C00150000",
            "strike": 150.0, "expiry": "2026-08-21", "dte_at_signal": 4, "oi_at_signal": 100,
            "bid": 2.6, "ask": 2.8, "mid": 2.7, "spread_pct": 0.07, "iv": 0.26,
            "delta": 0.55, "gamma": 0.05, "theta": -0.05, "vega": 0.1, "rho": 0.02,
        },
        {
            "signal_id": "SIG_1", "signal_ts": "2026-08-17T13:30:00+00:00", "ticker": "AAPL",
            "headline": "Apple announcement", "source": "Benzinga", "scorer_model": "llama3.2",
            "magnitude": 0.85, "confidence": 0.90, "catalyst": 0.80, "stock_price_at_signal": 150.0,
            "spy_price": 450.0, "spy_sma": 445.0, "regime_uptrend": True, "regime_mom_ok": True,
            "regime_chop_ok": False, "regime_dd_density": 0.1, "vol_proxy_symbol": "VIXY",
            "vol_proxy_price": 15.0, "vol_proxy_pctile": 0.2, "passes_news_call_gate": True,
            "passes_lotto_gate": False, "snapshot_ts": "2026-08-17T14:00:00+00:00",
            "minutes_since_signal": 30.0, "stock_price": 152.0, "symbol": "AAPL260821C00150000",
            "strike": 150.0, "expiry": "2026-08-21", "dte_at_signal": 4, "oi_at_signal": 100,
            "bid": 3.1, "ask": 3.3, "mid": 3.2, "spread_pct": 0.06, "iv": 0.27,
            "delta": 0.60, "gamma": 0.05, "theta": -0.05, "vega": 0.1, "rho": 0.02,
        },
        # Signal 2, Contract B (entry + 1 path update)
        {
            "signal_id": "SIG_2", "signal_ts": "2026-08-18T14:00:00+00:00", "ticker": "MSFT",
            "headline": "Microsoft update", "source": "Benzinga", "scorer_model": "llama3.2",
            "magnitude": 0.75, "confidence": 0.85, "catalyst": 0.70, "stock_price_at_signal": 300.0,
            "spy_price": 451.0, "spy_sma": 446.0, "regime_uptrend": True, "regime_mom_ok": True,
            "regime_chop_ok": False, "regime_dd_density": 0.15, "vol_proxy_symbol": "VIXY",
            "vol_proxy_price": 16.0, "vol_proxy_pctile": 0.3, "passes_news_call_gate": True,
            "passes_lotto_gate": False, "snapshot_ts": "2026-08-18T14:00:01+00:00",
            "minutes_since_signal": 0.01, "stock_price": 300.0, "symbol": "MSFT260821C00300000",
            "strike": 300.0, "expiry": "2026-08-21", "dte_at_signal": 3, "oi_at_signal": 250,
            "bid": 4.0, "ask": 4.3, "mid": 4.15, "spread_pct": 0.07, "iv": 0.22,
            "delta": 0.49, "gamma": 0.03, "theta": -0.08, "vega": 0.15, "rho": 0.03,
        },
        {
            "signal_id": "SIG_2", "signal_ts": "2026-08-18T14:00:00+00:00", "ticker": "MSFT",
            "headline": "Microsoft update", "source": "Benzinga", "scorer_model": "llama3.2",
            "magnitude": 0.75, "confidence": 0.85, "catalyst": 0.70, "stock_price_at_signal": 300.0,
            "spy_price": 451.0, "spy_sma": 446.0, "regime_uptrend": True, "regime_mom_ok": True,
            "regime_chop_ok": False, "regime_dd_density": 0.15, "vol_proxy_symbol": "VIXY",
            "vol_proxy_price": 16.0, "vol_proxy_pctile": 0.3, "passes_news_call_gate": True,
            "passes_lotto_gate": False, "snapshot_ts": "2026-08-18T14:30:00+00:00",
            "minutes_since_signal": 30.0, "stock_price": 298.0, "symbol": "MSFT260821C00300000",
            "strike": 300.0, "expiry": "2026-08-21", "dte_at_signal": 3, "oi_at_signal": 250,
            "bid": 3.2, "ask": 3.5, "mid": 3.35, "spread_pct": 0.09, "iv": 0.23,
            "delta": 0.45, "gamma": 0.03, "theta": -0.08, "vega": 0.14, "rho": 0.02,
        },
    ]
    df = pd.DataFrame(rows)
    with tempfile.NamedTemporaryFile(suffix=".csv", mode="w", delete=False) as f:
        df.to_csv(f.name, index=False)
        temp_path = f.name
    yield temp_path
    if os.path.exists(temp_path):
        os.remove(temp_path)


def test_two_pass_loading(sample_csv):
    """Verify Pass 1 and Pass 2 stream correctly and maintain expected row counts and dtypes."""
    entry_df, total_rows, rows_per_signal = load_entry_snapshot_and_stats(sample_csv, chunksize=2)
    assert total_rows == 5
    assert len(entry_df) == 2  # exactly 2 unique (signal_id, symbol) pairs
    assert len(rows_per_signal) == 2
    assert rows_per_signal["SIG_1"] == 3
    assert rows_per_signal["SIG_2"] == 2
    assert entry_df["bid"].dtype == np.float32

    # Selection rules on entry
    picked = SELECTION_RULES["delta_0.50_liquid"](entry_df)
    assert len(picked) == 2
    selected_pairs = set(zip(picked.index, picked.values))

    # Pass 2: load paths
    paths_df = load_selected_paths(sample_csv, selected_pairs, chunksize=2)
    assert len(paths_df) == 5
    assert "entry_price" in paths_df.columns
    assert "ret_from_entry" in paths_df.columns
    assert "drawdown_from_peak" in paths_df.columns


def test_selection_rules(sample_csv):
    """Verify selection rules pick expected contracts."""
    entry_df, _, _ = load_entry_snapshot_and_stats(sample_csv)
    for rule_name, rule_fn in SELECTION_RULES.items():
        picked = rule_fn(entry_df)
        assert len(picked) == 2
        assert "SIG_1" in picked.index
        assert "SIG_2" in picked.index


def test_exit_simulation(sample_csv):
    """Verify exit strategies calculate valid exit prices and PnL."""
    entry_df, _, _ = load_entry_snapshot_and_stats(sample_csv)
    picked = SELECTION_RULES["delta_0.50_liquid"](entry_df)
    selected_pairs = set(zip(picked.index, picked.values))
    paths_df = load_selected_paths(sample_csv, selected_pairs)

    for exit_name in EXIT_STRATEGIES:
        exits = simulate(paths_df, exit_name)
        assert len(exits) == 2
        assert "pnl_pct" in exits.columns
        assert not exits["pnl_pct"].isna().any()


def test_chronological_split(sample_csv):
    """Verify train/test splitting maintains chronological order."""
    entry_df, _, _ = load_entry_snapshot_and_stats(sample_csv)
    signal_meta = get_signal_meta(entry_df)
    train_ids, test_ids = chronological_split(signal_meta, test_frac=0.5)
    assert len(train_ids) == 1
    assert len(test_ids) == 1
    assert "SIG_1" in train_ids  # earlier signal
    assert "SIG_2" in test_ids   # later signal


def test_feature_table_and_gates(sample_csv):
    """Verify broad feature table creation and gate application."""
    entry_df, _, _ = load_entry_snapshot_and_stats(sample_csv)
    picked = SELECTION_RULES["delta_0.50_liquid"](entry_df)
    feats, dropped = build_broad_feature_table(entry_df, picked)

    assert "magnitude" in feats.columns
    assert "spread_pct" in feats.columns

    # Test applying a rule gate
    rule = {
        "thresholds": {"spread_pct": ("max", 0.10)},
        "avg_pnl": 0.05,
    }
    kept = apply_rule_gate(feats, rule)
    assert kept.sum() == 2


def test_notable_trades_defensive():
    """Verify notable_trades works even with missing metadata columns."""
    exits = pd.DataFrame([
        {"signal_id": "S1", "pnl_pct": 0.25, "exit_reason": "take_profit", "minutes_held": 15.0},
        {"signal_id": "S2", "pnl_pct": -0.10, "exit_reason": "stop_loss", "minutes_held": 10.0},
    ])
    meta = pd.DataFrame([
        {"signal_id": "S1", "ticker": "AAPL", "signal_ts": "2026-08-17T13:30:00+00:00"},
        {"signal_id": "S2", "ticker": "MSFT", "signal_ts": "2026-08-18T14:00:00+00:00"},
    ])
    keep_rule = pd.Series([True, False], index=["S1", "S2"])
    keep_model = pd.Series([True, True], index=["S1", "S2"])

    best, worst = notable_trades(meta, exits, ids={"S1", "S2"}, keep_rule=keep_rule, keep_model=keep_model, n=1)
    assert len(best) == 1
    assert best.iloc[0]["signal_id"] == "S1"
    assert len(worst) == 1
    assert worst.iloc[0]["signal_id"] == "S2"


def test_load_normalized_schema(tmp_path):
    """Verify loading compact snapshots file merges with contract_grid_signals.csv."""
    sig_csv = tmp_path / "contract_grid_signals.csv"
    snap_csv = tmp_path / "contract_grid_snapshots.csv"

    # 1. Write signals metadata
    sig_df = pd.DataFrame([{
        "signal_id": "NORM_SIG_1",
        "signal_ts": "2026-09-29T14:30:00+00:00",
        "ticker": "AAPL",
        "headline": "Apple Q4 Beats",
        "source": "Benzinga",
        "scorer_model": "llama3.2",
        "sentiment": "bullish",
        "magnitude": 0.85,
        "confidence": 0.90,
        "catalyst": "Record Services Revenue",
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
    }])
    sig_df.to_csv(sig_csv, index=False)

    # 2. Write compact snapshots referencing only signal_id
    snap_df = pd.DataFrame([
        {
            "signal_id": "NORM_SIG_1",
            "snapshot_ts": "2026-09-29T14:30:05+00:00",
            "minutes_since_signal": 0.08,
            "stock_price": 225.0,
            "symbol": "AAPL261016C00225000",
            "contract_type": "call",
            "strike": 225.0,
            "expiry": "2026-10-16",
            "dte_at_signal": 17,
            "oi_at_signal": 2500,
            "bid": 3.40,
            "ask": 3.50,
            "mid": 3.45,
            "spread_pct": 0.029,
            "iv": 0.28,
            "delta": 0.51,
            "gamma": 0.04,
            "theta": -0.05,
            "vega": 0.16,
            "rho": 0.07,
        },
        {
            "signal_id": "NORM_SIG_1",
            "snapshot_ts": "2026-09-29T14:30:35+00:00",
            "minutes_since_signal": 0.58,
            "stock_price": 225.2,
            "symbol": "AAPL261016C00225000",
            "contract_type": "call",
            "strike": 225.0,
            "expiry": "2026-10-16",
            "dte_at_signal": 17,
            "oi_at_signal": 2500,
            "bid": 3.50,
            "ask": 3.60,
            "mid": 3.55,
            "spread_pct": 0.028,
            "iv": 0.28,
            "delta": 0.52,
            "gamma": 0.04,
            "theta": -0.05,
            "vega": 0.16,
            "rho": 0.07,
        }
    ])
    snap_df.to_csv(snap_csv, index=False)

    entry_df, total_rows, counts = load_entry_snapshot_and_stats(str(snap_csv))
    assert total_rows == 2
    assert len(entry_df) == 1
    row = entry_df.iloc[0]
    # Check that signal metadata was merged cleanly from contract_grid_signals.csv:
    assert row["signal_id"] == "NORM_SIG_1"
    assert row["ticker"] == "AAPL"
    assert row["headline"] == "Apple Q4 Beats"
    assert row["magnitude"] == 0.85
    assert row["confidence"] == 0.90
    assert row["catalyst"] == "Record Services Revenue"
    assert row["delta"] == 0.51

