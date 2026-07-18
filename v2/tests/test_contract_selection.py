"""Contract-mispricing switch (2026-07-18) -- jeff's idea, validated with a small
promising-but-unvalidated historical signal (research/lotto_mispricing_test.py, n=15) and an
accumulating live shadow in v1. Wired in LIVE for v2 (unlike v1's shadow-only version), so these
pure-logic pieces (no network) need solid coverage: the quadratic fit, the liquidity gate, and
above all the switch decision -- it must never switch into an illiquid contract just because it
looks cheap on paper, and must never switch on a gap too small to distinguish from fit noise."""
from types import SimpleNamespace

import config as cfg
import execution as ex
import market as mkt


def fake_contract(symbol, strike, expiry="2026-08-07", tradable=True, root=None, open_interest=500):
    return SimpleNamespace(symbol=symbol, strike_price=strike, expiration_date=expiry,
                            tradable=tradable, root_symbol=root, open_interest=open_interest)


def test_fit_quadratic_recovers_exact_parabola():
    xs = [10.0, 20.0, 30.0, 40.0, 50.0]
    ys = [x * x for x in xs]
    a, b, c = ex._fit_quadratic(xs, ys)
    for x, y in zip(xs, ys):
        assert abs((a * x * x + b * x + c) - y) < 1e-6


def test_fit_quadratic_degenerate_same_x_returns_none():
    assert ex._fit_quadratic([10.0, 10.0, 10.0, 10.0], [1.0, 2.0, 3.0, 4.0]) is None


def test_passes_contract_checks_rejects_non_tradable():
    c = fake_contract("X250101C00100000", 100.0, tradable=False)
    quote = {"mid": 1.0, "bid": 0.9, "ask": 1.1, "spread_pct": 0.05}
    assert not ex._passes_contract_checks(c, quote, stock_price=95.0, ticker="X")


def test_passes_contract_checks_rejects_root_symbol_mismatch():
    """Adjusted/non-standard contracts (e.g. post-corporate-action) carry a different root and
    get rejected by Alpaca on order submission -- must be filtered before ever being considered."""
    c = fake_contract("FDX1250101C00100000", 100.0, root="FDX1")
    quote = {"mid": 1.0, "bid": 0.9, "ask": 1.1, "spread_pct": 0.05}
    assert not ex._passes_contract_checks(c, quote, stock_price=95.0, ticker="FDX")


def test_passes_contract_checks_rejects_low_open_interest(monkeypatch):
    monkeypatch.setattr(mkt, "passes_guardrails", lambda q, **k: True)
    c = fake_contract("X250101C00100000", 100.0, open_interest=10)
    quote = {"mid": 1.0, "bid": 0.9, "ask": 1.1, "spread_pct": 0.05}
    assert not ex._passes_contract_checks(c, quote, stock_price=95.0, ticker="X")


def test_passes_contract_checks_rejects_quote_disconnected_from_underlying():
    """Bad-data guard: a call quote can't sit below 90% of intrinsic or above the stock price."""
    c = fake_contract("X250101C00050000", 50.0, open_interest=500)
    bad_quote = {"mid": 200.0, "bid": 199.0, "ask": 201.0, "spread_pct": 0.01}   # way above stock_price
    assert not ex._passes_contract_checks(c, bad_quote, stock_price=95.0, ticker="X")


def test_maybe_switch_disabled_always_returns_target(monkeypatch):
    monkeypatch.setattr(cfg, "MISPRICING_SWITCH_ENABLED", False)
    target = {"symbol": "X250101C00100000", "strike": 100.0, "expiry": "2026-08-07"}
    result = ex._maybe_switch_to_cheaper_contract("X", target, 95.0, all_contracts=[])
    assert result is target


def test_maybe_switch_stays_on_target_when_gap_below_threshold(monkeypatch):
    """A tiny residual gap must NOT trigger a switch -- this is the guard against acting on
    quadratic-fit noise instead of real mispricing (see config.py's MISPRICING_MIN_RESIDUAL_GAP)."""
    monkeypatch.setattr(cfg, "MISPRICING_SWITCH_ENABLED", True)
    monkeypatch.setattr(cfg, "MISPRICING_MIN_RESIDUAL_GAP", 0.02)
    monkeypatch.setattr(mkt, "passes_guardrails", lambda q, **k: True)

    target = {"symbol": "X250101C00100000", "strike": 100.0, "expiry": "2026-08-07"}
    strikes = [80.0, 90.0, 100.0, 110.0, 120.0]
    # near-perfect smile (IV rises smoothly with strike) plus a sub-threshold wobble on strike 110
    ivs = {80.0: 0.30, 90.0: 0.33, 100.0: 0.36, 110.0: 0.385, 120.0: 0.42}
    contracts = [fake_contract(f"X250101C{int(s*1000):08d}", s) for s in strikes]

    def fake_quote(symbol):
        return {"mid": 1.0, "bid": 0.9, "ask": 1.1, "spread_pct": 0.02}

    monkeypatch.setattr(mkt, "get_option_quote", fake_quote)
    monkeypatch.setattr(ex, "implied_vol_call", lambda price, S, K, T: ivs[K])

    result = ex._maybe_switch_to_cheaper_contract("X", target, 95.0, contracts)
    assert result is target


def test_maybe_switch_switches_when_gap_exceeds_threshold(monkeypatch):
    """A same-expiry neighbor sitting clearly below the local smile should win."""
    monkeypatch.setattr(cfg, "MISPRICING_SWITCH_ENABLED", True)
    monkeypatch.setattr(cfg, "MISPRICING_MIN_RESIDUAL_GAP", 0.02)
    monkeypatch.setattr(mkt, "passes_guardrails", lambda q, **k: True)

    target = {"symbol": "X250101C00100000", "strike": 100.0, "expiry": "2026-08-07"}
    strikes = [80.0, 90.0, 100.0, 110.0, 120.0]
    # strike 110 sits well BELOW the smile implied by its neighbors -- a real dislocation
    ivs = {80.0: 0.30, 90.0: 0.33, 100.0: 0.36, 110.0: 0.20, 120.0: 0.42}
    contracts = [fake_contract(f"X250101C{int(s*1000):08d}", s) for s in strikes]

    def fake_quote(symbol):
        return {"mid": 1.0, "bid": 0.9, "ask": 1.1, "spread_pct": 0.02}

    monkeypatch.setattr(mkt, "get_option_quote", fake_quote)
    monkeypatch.setattr(ex, "implied_vol_call", lambda price, S, K, T: ivs[K])

    result = ex._maybe_switch_to_cheaper_contract("X", target, 95.0, contracts)
    assert result["strike"] == 110.0
    assert result["symbol"] != target["symbol"]


def test_maybe_switch_never_switches_into_an_illiquid_contract(monkeypatch):
    """The cheapest-by-residual contract must still pass the SAME liquidity checks as the target
    -- a thin/stale quote looking cheap on paper is not a real opportunity."""
    monkeypatch.setattr(cfg, "MISPRICING_SWITCH_ENABLED", True)
    monkeypatch.setattr(cfg, "MISPRICING_MIN_RESIDUAL_GAP", 0.02)
    monkeypatch.setattr(mkt, "passes_guardrails", lambda q, **k: True)

    target = {"symbol": "X250101C00100000", "strike": 100.0, "expiry": "2026-08-07"}
    strikes = [80.0, 90.0, 100.0, 110.0, 120.0]
    ivs = {80.0: 0.30, 90.0: 0.33, 100.0: 0.36, 110.0: 0.20, 120.0: 0.42}   # 110 looks cheapest
    # strike 110 is the illiquid one (open_interest below MIN_OPEN_INTEREST)
    contracts = [fake_contract(f"X250101C{int(s*1000):08d}", s,
                                open_interest=(10 if s == 110.0 else 500)) for s in strikes]

    def fake_quote(symbol):
        return {"mid": 1.0, "bid": 0.9, "ask": 1.1, "spread_pct": 0.02}

    monkeypatch.setattr(mkt, "get_option_quote", fake_quote)
    monkeypatch.setattr(ex, "implied_vol_call", lambda price, S, K, T: ivs[K])

    result = ex._maybe_switch_to_cheaper_contract("X", target, 95.0, contracts)
    assert result is target   # 110 was cheapest but illiquid -- must stay on target
