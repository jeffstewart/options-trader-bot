"""
lotto_scaleout_test.py — does SCALING OUT of lotto winners beat all-or-nothing?

For a convex bet the standard worry is "I had a 5x and trailed it back to 2x." Scaling out
(sell part at a profit multiple, let the rest ride the tiered trail) books some gain while
keeping tail exposure. This tests several scale-out specs at the live config (Δ0.25, 14 DTE,
7d hold, tiered exit) vs the all-or-nothing baseline.

Uses the opt-in `scale_out` param just added to simulate_option_pnl (default None = unchanged).
The BASELINE row (scale_out=None) must reproduce the known lotto P&L (~$9,938) — that validates
the sim is byte-identical with the feature off before we trust the scale-out variants.

CAVEAT: lotto positions are tiny (qty often 1–3 contracts at the $250 cap), so a "sell half"
is only realizable when qty≥2. This models fractional contracts to measure the STRATEGY effect;
at qty=1 a scale-out degenerates to a hard profit-target (see lotto_exit_sweep.py).

Usage:  USE_YAHOO_BARS=1 .venv/bin/python lotto_scaleout_test.py
"""
import os
os.environ.setdefault("USE_YAHOO_BARS", "1")
from datetime import datetime, timezone, timedelta
from pathlib import Path

import backtest as _bt
import tune_v2, config as cfg
from benchmark import compute_stats
from regime_filter import build_regime

DELTA, DTE, HOLD, POS = cfg.LOTTO_TARGET_DELTA, 14, cfg.LOTTO_MAX_HOLD_DAYS, cfg.LOTTO_POSITION_USD
GATE_MAG, GATE_CONF = cfg.LOTTO_MIN_MAGNITUDE, cfg.LOTTO_MIN_CONFIDENCE
TIERED = {"tiers": [(1.0, 0.40), (3.0, 0.30), (float("inf"), 0.20)]}
WINDOWS = [
    ("bull-meltup", datetime.now(timezone.utc) - timedelta(days=5), 180, "dual_score_cache.json"),
    ("2022-bear",   datetime(2022, 6, 30, tzinfo=timezone.utc), 90, "bear_dual_cache.json"),
]
SPECS = [
    ("ALL-OR-NOTHING (baseline)", None),
    ("sell 50% at 1.5x",          (1.5, 0.50)),
    ("sell 50% at 2x",            (2.0, 0.50)),
    ("sell 50% at 3x",            (3.0, 0.50)),
    ("sell 33% at 2x",            (2.0, 0.33)),
    ("sell 33% at 2x + tail runs (already)", (2.0, 0.34)),
]


def run(rows, scale_out, regime):
    save = {k: getattr(_bt, k) for k in
            ("TARGET_DELTA", "DTE_TARGET", "MAX_HOLD_DAYS", "TRAILING_STOP_PCT", "EXIT_PARAMS")}
    _bt.TARGET_DELTA, _bt.DTE_TARGET, _bt.MAX_HOLD_DAYS = DELTA, DTE, HOLD
    _bt.TRAILING_STOP_PCT, _bt.EXIT_PARAMS = 0.30, TIERED
    trades, seen = [], set()
    try:
        for r in rows:
            d = r["created_at"].date()
            if regime and not regime(d):           # lotto is regime-gated (uptrend only)
                continue
            for tk in r["tickers"][:1]:
                if tk in ("BTC", "ETH") or not _bt.is_valid_stock_ticker(tk):
                    continue
                k = f"{d}_{tk}"
                if k in seen:
                    continue
                seen.add(k)
                sp = _bt.get_price_at(tk, r["created_at"])
                if not sp:
                    continue
                t = _bt.simulate_option_pnl(
                    tk, r["created_at"], sp, max(POS * 0.5, POS * r["magnitude"] * r["confidence"]),
                    {"magnitude": r["magnitude"], "confidence": r["confidence"]},
                    option_type="call", exit_rule="tiered_profit", spread_mult=1.0, scale_out=scale_out)
                if t:
                    trades.append(t)
    finally:
        for kk, vv in save.items():
            setattr(_bt, kk, vv)
    return trades


def line(tag, trades):
    if not trades:
        return f"  {tag:40} n=   0"
    s = compute_stats(trades)
    x2 = sum(1 for t in trades if t["pnl_pct"] >= 100)
    x4 = sum(1 for t in trades if t["pnl_pct"] >= 300)
    return (f"  {tag:40} n={len(trades):>4}  P&L=${s['total_pnl']:>8,.0f}  Sharpe={s['sharpe']:>5.2f}  "
            f"win={s['win_rate']:>4.1f}%  2x+={x2:>3} 4x+={x4:>3}")


def main():
    print(f"═══ LOTTO SCALE-OUT TEST — Δ{DELTA}, {DTE}DTE, {HOLD}d hold ═══")
    print("(baseline must ≈ $9,938 to confirm the sim is unchanged with scale_out off)\n")
    for label, end_dt, days, cache in WINDOWS:
        if not Path(cache).exists():
            print(f"[{label}] cache missing — skip"); continue
        scored = tune_v2.load_scored_from_dual_cache(Path(cache), "bull", end_dt, days)
        gated = [r for r in scored if r["magnitude"] >= GATE_MAG and r["confidence"] >= GATE_CONF]
        reg = build_regime(end_dt, days, 200)
        print(f"═══ {label} ═══  gated={len(gated)}")
        for tag, spec in SPECS[:5]:
            print(line(tag, run(gated, spec, reg)))
        print()
    print("Read: scale-out helps only if total P&L/Sharpe RISES vs baseline. Expect it to lift the")
    print("floor (book some 2x) but clip the 4x+ tail — net is what matters for a convex bet.")


if __name__ == "__main__":
    main()
