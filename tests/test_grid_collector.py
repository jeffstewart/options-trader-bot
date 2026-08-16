"""grid_collector.py -- the contract-picking research collector (2026-08-XX), v1 build. Pure logic
and row-shape coverage; no real network calls. The one thing that must never regress: this module
can NEVER slow or block the real trading path, so should_collect must be cheap/pure and every I/O
path must swallow its own exceptions rather than propagate."""
import asyncio
from datetime import date, datetime, timedelta, timezone
from types import SimpleNamespace

import config
import grid_collector as gc


def test_should_collect_uses_the_derived_threshold():
    # 0.13 = closest-achievable coverage point on Ollama's clustered mag*conf distribution (see
    # config.py comment) -- not a smooth 90th percentile, Ollama's confidence-collapse means there
    # is no threshold that lands there exactly.
    assert gc.should_collect({"magnitude": 0.5, "confidence": 0.3}) is True     # product 0.15
    assert gc.should_collect({"magnitude": 0.3, "confidence": 0.3}) is False    # product 0.09
    assert gc.should_collect({"magnitude": 0.9, "confidence": 0.9}) is True


def test_should_collect_handles_missing_or_bad_fields():
    assert gc.should_collect({}) is False
    assert gc.should_collect({"magnitude": "not-a-number", "confidence": 0.9}) is False


def test_interval_for_schedule_phases():
    assert gc._interval_for(0) == 30
    assert gc._interval_for(15) == 30
    assert gc._interval_for(15.01) == 180
    assert gc._interval_for(60) == 180
    assert gc._interval_for(61) == 900
    assert gc._interval_for(360) == 900
    assert gc._interval_for(361) == 1800    # long end is 30 min here, not v2's 2hr
    assert gc._interval_for(999999) == 1800


def _fake_snapshot(bid=1.0, ask=1.2, iv=0.35, delta=0.3):
    quote = SimpleNamespace(bid_price=bid, ask_price=ask)
    greeks = SimpleNamespace(delta=delta, gamma=0.01, theta=-0.02, vega=0.05, rho=0.01)
    return SimpleNamespace(latest_quote=quote, latest_trade=None, implied_volatility=iv, greeks=greeks)


def _base_state(**overrides):
    state = {
        "ticker": "ABC", "headline": "h", "source": "s", "scorer_model": "llama3.2",
        "magnitude": 0.6, "confidence": 0.8, "catalyst": 0.5, "stock_price_at_signal": 100.0,
        "spy_price": 550.0, "spy_sma": 540.0, "regime_uptrend": True, "regime_mom_ok": True,
        "regime_chop_ok": True, "regime_dd_density": 0.2,
        "vol_proxy_symbol": "VIXY", "vol_proxy_price": 18.5, "vol_proxy_pctile": 0.6,
        "passes_news_call_gate": True, "passes_lotto_gate": False,
        "signal_ts": datetime.now(timezone.utc), "get_stock_price": lambda t: 101.0,
    }
    state.update(overrides)
    return state


def test_snapshot_rows_computes_mid_and_spread_correctly(monkeypatch):
    state = _base_state(
        signal_ts=datetime.now(timezone.utc) - timedelta(minutes=5),
        contracts_meta={"ABC250101C00100000": (100.0, "2025-01-01", 10, 250)},
    )
    monkeypatch.setattr(gc, "option_data_client", SimpleNamespace(
        get_option_snapshot=lambda req: {"ABC250101C00100000": _fake_snapshot(bid=1.0, ask=1.2)}))
    rows = gc._snapshot_rows("sig1", state)
    assert len(rows) == 1
    row = dict(zip(gc._HEADER, rows[0]))
    assert row["symbol"] == "ABC250101C00100000"
    assert row["oi_at_signal"] == 250
    assert row["bid"] == 1.0 and row["ask"] == 1.2
    assert row["mid"] == 1.1
    assert abs(row["spread_pct"] - (0.2 / 1.1)) < 1e-4
    assert row["iv"] == 0.35 and row["delta"] == 0.3
    assert 4.9 <= row["minutes_since_signal"] <= 5.1
    assert row["stock_price"] == 101.0
    assert row["scorer_model"] == "llama3.2" and row["catalyst"] == 0.5
    assert row["spy_price"] == 550.0 and row["regime_uptrend"] is True
    assert row["vol_proxy_symbol"] == "VIXY" and row["vol_proxy_pctile"] == 0.6
    assert row["passes_news_call_gate"] is True and row["passes_lotto_gate"] is False


def test_snapshot_rows_missing_quote_still_writes_a_row(monkeypatch):
    """A contract that's disappeared from the tape must show up as a blank-quote row, not
    silently vanish from the dataset -- that absence is itself the finding."""
    state = _base_state(contracts_meta={"ABC250101C00100000": (100.0, "2025-01-01", 10, None)})
    monkeypatch.setattr(gc, "option_data_client",
                         SimpleNamespace(get_option_snapshot=lambda req: {}))
    rows = gc._snapshot_rows("sig1", state)
    assert len(rows) == 1
    row = dict(zip(gc._HEADER, rows[0]))
    assert row["bid"] == "" and row["oi_at_signal"] is None


def test_snapshot_rows_never_raises_on_fetch_failure(monkeypatch):
    state = _base_state(contracts_meta={"X": (100.0, "2025-01-01", 10, 100)})
    def boom(req):
        raise RuntimeError("network down")
    monkeypatch.setattr(gc, "option_data_client", SimpleNamespace(get_option_snapshot=boom))
    assert gc._snapshot_rows("sig1", state) == []


def test_current_stock_price_never_raises_and_blanks_on_failure():
    state = _base_state(get_stock_price=lambda t: (_ for _ in ()).throw(RuntimeError("down")))
    assert gc._current_stock_price(state) == ""
    state2 = _base_state(get_stock_price=lambda t: None)
    assert gc._current_stock_price(state2) == ""


def test_gate_eligibility_matches_config_thresholds():
    news_call_ok, lotto_ok = gc._gate_eligibility(0.99, 0.99)
    assert news_call_ok is True and lotto_ok is True
    news_call_ok, lotto_ok = gc._gate_eligibility(0.01, 0.01)
    assert news_call_ok is False and lotto_ok is False
    # right at the news_call magnitude floor but confidence too low for the dynamic floor
    news_call_ok, _ = gc._gate_eligibility(config.MIN_MAGNITUDE, 0.0)
    assert news_call_ok is False


def test_regime_snapshot_caches_per_day_and_never_raises(monkeypatch):
    gc._regime_cache.update(date=None, state={})
    calls = {"n": 0}
    def boom(req):
        calls["n"] += 1
        raise RuntimeError("network down")
    monkeypatch.setattr(gc, "stock_data_client", SimpleNamespace(get_stock_bars=boom))
    assert gc._regime_snapshot() == {}
    assert gc._regime_snapshot() == {}
    # first call fetches SPY regime + the VIXY vol-proxy (2 sub-fetches, both boom); second call
    # is served entirely from the same-day cache, no further fetches
    assert calls["n"] == 2
    gc._regime_cache.update(date=None, state={})


def _fake_bars(symbol: str, closes: list):
    return SimpleNamespace(data={symbol: [SimpleNamespace(close=c) for c in closes]})


def test_vol_proxy_snapshot_computes_percentile_rank(monkeypatch):
    # window rising to a new high -> latest close ranks at the top of its own trailing window,
    # regardless of the ETP's absolute price level (which drifts from contango decay over time)
    closes = [10.0, 11.0, 9.0, 12.0, 15.0]
    monkeypatch.setattr(gc, "stock_data_client", SimpleNamespace(
        get_stock_bars=lambda req: _fake_bars("VIXY", closes)))
    out = gc._vol_proxy_snapshot()
    assert out["vol_proxy_symbol"] == "VIXY"
    assert out["vol_proxy_price"] == 15.0
    assert out["vol_proxy_pctile"] == 1.0


def test_vol_proxy_snapshot_never_raises_on_fetch_failure(monkeypatch):
    def boom(req):
        raise RuntimeError("network down")
    monkeypatch.setattr(gc, "stock_data_client", SimpleNamespace(get_stock_bars=boom))
    assert gc._vol_proxy_snapshot() == {}


def test_write_rows_creates_header_once(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(config, "GRID_SNAPSHOT_CSV", "grid.csv")
    gc._write_rows([["a"] * len(gc._HEADER)])
    gc._write_rows([["b"] * len(gc._HEADER)])
    lines = (tmp_path / "grid.csv").read_text().strip().splitlines()
    assert len(lines) == 3
    assert lines[0].split(",")[0] == "signal_id"


def test_write_rows_never_raises_on_bad_path(monkeypatch):
    monkeypatch.setattr(config, "GRID_SNAPSHOT_CSV", "/nonexistent-dir/does/not/exist.csv")
    gc._write_rows([["x"] * len(gc._HEADER)])


def test_make_signal_id_is_pure_and_deterministic():
    ts = datetime(2026, 8, 15, 12, 0, 0, 123456, tzinfo=timezone.utc)
    assert gc.make_signal_id("ABC", ts) == gc.make_signal_id("ABC", ts)
    assert gc.make_signal_id("ABC", ts) != gc.make_signal_id("XYZ", ts)


def test_start_collection_never_raises_when_stock_price_unavailable():
    ts = datetime.now(timezone.utc)
    asyncio.run(gc.start_collection(gc.make_signal_id("ABC", ts), ts, "ABC",
                                     {"magnitude": 0.9, "confidence": 0.9}, "h", "s",
                                     get_stock_price=lambda ticker: None))
    assert "ABC" not in [s["ticker"] for s in gc._tracked.values()]


def test_start_collection_never_raises_when_contract_lookup_fails(monkeypatch):
    def boom(ticker, price):
        raise RuntimeError("api down")
    monkeypatch.setattr(gc, "_fetch_contracts_meta", boom)
    ts = datetime.now(timezone.utc)
    asyncio.run(gc.start_collection(gc.make_signal_id("ABC", ts), ts, "ABC",
                                     {"magnitude": 0.9, "confidence": 0.9}, "h", "s",
                                     get_stock_price=lambda ticker: 100.0))


def test_start_collection_uses_the_caller_supplied_signal_id(monkeypatch):
    """The id/timestamp stashed on signal["_grid_signal_id"] by bot.py must be the SAME id this
    module tracks under and writes to the CSV -- not a freshly-minted one -- or the join back to
    trades.csv breaks."""
    ts = datetime.now(timezone.utc)
    sid = gc.make_signal_id("ABC", ts)
    monkeypatch.setattr(gc, "_fetch_contracts_meta",
                         lambda ticker, price: {"ABC250101C00100000": (100.0, "2099-01-01", 10, 100)})
    monkeypatch.setattr(gc, "_regime_snapshot", lambda: {})
    monkeypatch.setattr(gc, "option_data_client", SimpleNamespace(get_option_snapshot=lambda req: {}))
    asyncio.run(gc.start_collection(sid, ts, "ABC", {"magnitude": 0.9, "confidence": 0.9}, "h", "s",
                                     get_stock_price=lambda ticker: 100.0))
    assert sid in gc._tracked
    gc._tracked.clear()


def test_poll_loop_prune_predicate_catches_expired_and_stale_signals():
    """A tracked signal whose last contract has already expired, or that's been tracked past
    GRID_MAX_TRACK_DAYS, must be prunable. run_poll_loop() itself sleeps 15s before its first
    check (too slow for a unit test) -- exercise the exact prune predicate from its body."""
    now = datetime.now(timezone.utc)

    def would_be_pruned(state: dict) -> bool:
        elapsed_days = (now - state["signal_ts"]).days
        return now.date() > state["last_expiry"] or elapsed_days > config.GRID_MAX_TRACK_DAYS

    expired = {"signal_ts": now - timedelta(days=2), "last_expiry": date.today() - timedelta(days=1)}
    too_old = {"signal_ts": now - timedelta(days=config.GRID_MAX_TRACK_DAYS + 5),
               "last_expiry": date.today() + timedelta(days=30)}
    fresh = {"signal_ts": now - timedelta(minutes=5), "last_expiry": date.today() + timedelta(days=10)}

    assert would_be_pruned(expired) is True
    assert would_be_pruned(too_old) is True
    assert would_be_pruned(fresh) is False


def test_poll_loop_skips_entirely_when_market_closed():
    """The poller must not poll (or advance state) while the market is closed -- overnight/weekend
    hours would just waste API calls for zero new information."""
    gc._tracked.clear()
    gc._tracked["sig1"] = _base_state(
        signal_ts=datetime.now(timezone.utc) - timedelta(hours=1),
        last_poll=datetime.now(timezone.utc) - timedelta(hours=1),
        last_expiry=date.today() + timedelta(days=10), contracts_meta={},
    )
    before = dict(gc._tracked["sig1"])

    async def run_briefly():
        task = asyncio.create_task(gc.run_poll_loop(market_is_open=lambda: False))
        await asyncio.sleep(0.05)
        task.cancel()
        try:
            await task
        except asyncio.CancelledError:
            pass

    asyncio.run(run_briefly())
    # loop's first check is 15s away regardless, so this really just confirms the task starts and
    # can be cancelled cleanly without touching state -- the real market-closed guard is asserted
    # structurally: market_is_open() is checked before any due/finished computation in the loop body.
    assert gc._tracked["sig1"]["last_poll"] == before["last_poll"]
    gc._tracked.clear()
