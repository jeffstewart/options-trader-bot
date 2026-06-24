"""
lotto_exit_sweep.py — re-tune the LOTTO exit rule for the NEW short-DTE config.

The live lotto leg moved to 8-21 DTE / 7-day hold today, but LOTTO_EXIT_TIERS (the wide
"let it run" tiered trail) was validated at the OLD ~21-DTE / longer-hold setup. Exit tuning
matters as much as entry, so this sweeps several exit rules at the CURRENT config (Δ0.25,
14 DTE, 7d hold) and compares P&L / Sharpe / win / multibagger retention vs the live rule.

Reuses lotto_backtest.run_variant (which patches the backtest globals + prices the option),
so the sim is identical to lotto_backtest except for the exit rule/params we pass in.

Usage:  USE_YAHOO_BARS=1 .venv/bin/python lotto_exit_sweep.py
"""
import os
os.environ.setdefault("USE_YAHOO_BARS", "1")
from pathlib import Path

import lotto_backtest as lb
import config as cfg
from benchmark import compute_stats

INF = float("inf")
DELTA = cfg.LOTTO_TARGET_DELTA   # 0.25
LIVE = [(1.0, 0.40), (3.0, 0.30), (INF, 0.20)]   # current live tiered trail

# (label, exit_rule, exit_params). tiers = (gain_threshold, trail_width) — tighten the trail
# as the gain grows. hard_target = sell 100% at that profit multiple (a cap).
CONFIGS = [
    ("LIVE .40/.30/.20 (no cap)",          "tiered_profit", {"tiers": LIVE}),
    # ── HARD-CAP LEVEL sweep (live tiers + cap) — finish #3 ──────────────
    ("LIVE + hard cap 3x",                 "tiered_profit", {"tiers": LIVE, "hard_target": 3.0}),
    ("LIVE + hard cap 4x",                 "tiered_profit", {"tiers": LIVE, "hard_target": 4.0}),
    ("LIVE + hard cap 5x",                 "tiered_profit", {"tiers": LIVE, "hard_target": 5.0}),
    ("LIVE + hard cap 6x",                 "tiered_profit", {"tiers": LIVE, "hard_target": 6.0}),
    # ── RATCHET: tighten the trail at higher tiers, NO cap (let runners run) ──
    ("ratchet top  .40/.25/.12",           "tiered_profit", {"tiers": [(1.0, 0.40), (3.0, 0.25), (INF, 0.12)]}),
    ("ratchet 4-tier .35/.25/.15/.10",     "tiered_profit", {"tiers": [(1.0, 0.35), (2.0, 0.25), (4.0, 0.15), (INF, 0.10)]}),
    ("ratchet tight  .30/.20/.15/.10",     "tiered_profit", {"tiers": [(1.0, 0.30), (2.0, 0.20), (3.0, 0.15), (INF, 0.10)]}),
    ("ratchet fine 5-tier",                "tiered_profit", {"tiers": [(0.5, 0.40), (1.0, 0.30), (2.0, 0.20), (4.0, 0.15), (INF, 0.10)]}),
    # ── COMBO: tight ratchet + a HIGH safety cap (best of both?) ──────────
    ("ratchet .35/.25/.15/.10 + cap 6x",   "tiered_profit", {"tiers": [(1.0, 0.35), (2.0, 0.25), (4.0, 0.15), (INF, 0.10)], "hard_target": 6.0}),
    ("ratchet .35/.25/.15/.10 + cap 8x",   "tiered_profit", {"tiers": [(1.0, 0.35), (2.0, 0.25), (4.0, 0.15), (INF, 0.10)], "hard_target": 8.0}),
]


def line(tag, trades):
    if not trades:
        return f"  {tag:34} n=   0  (no data)"
    s = compute_stats(trades)
    x2 = sum(1 for t in trades if t["pnl_pct"] >= 100)
    x4 = sum(1 for t in trades if t["pnl_pct"] >= 300)
    x5 = sum(1 for t in trades if t["pnl_pct"] >= 400)
    mx = max(t["pnl_pct"] for t in trades)
    return (f"  {tag:34} n={len(trades):>4}  P&L=${s['total_pnl']:>8,.0f}  Sharpe={s['sharpe']:>5.2f}  "
            f"win={s['win_rate']:>4.1f}%  2x+={x2:>3} 4x+={x4:>3} 5x+={x5:>3}  max={mx:>+5.0f}%")


def main():
    print(f"═══ LOTTO EXIT SWEEP — Δ{DELTA}, DTE {lb.DTE}, {lb.MAX_HOLD}d hold ═══")
    print("(re-tuning the exit rule for the NEW short-DTE window; LIVE rule = first row)\n")
    for label, end_dt, days, cache in lb.WINDOWS:
        if not Path(cache).exists():
            print(f"[{label}] cache missing — skip"); continue
        scored = lb.tune_v2.load_scored_from_dual_cache(Path(cache), "bull", end_dt, days)
        gated = [r for r in scored if r["magnitude"] >= lb.LOTTO_MIN_MAG and r["confidence"] >= lb.LOTTO_MIN_CONF]
        reg = lb.build_regime(end_dt, days, 200)
        print(f"═══ {label} ═══  (gated high-conviction signals: {len(gated)})")
        for tag, rule, params in CONFIGS:
            trades = lb.run_variant(gated, reg, DELTA, rule, params, spread_mult=1.0)
            print(line(tag, trades))
        print()
    print("Read: prefer the rule that RAISES P&L/Sharpe while keeping 4x+ winners. For a convex")
    print("bet, a higher win-rate that sacrifices the 4x+ tail is usually WORSE, not better.")


if __name__ == "__main__":
    main()
