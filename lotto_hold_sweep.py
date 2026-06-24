"""
lotto_hold_sweep.py — find the sweet-spot MAX-HOLD for the NEW short-DTE lotto.

We moved lotto to 8-21 DTE (target ~14) and shipped a 3× profit cap, but never re-swept the
time-stop (still LOTTO_MAX_HOLD_DAYS=7). Shorter DTE → faster theta on non-movers, so a shorter
hold may cut dead money before it bleeds. Sweeps MAX_HOLD at DTE 14 with the shipped exit
(tiered trail + 3× hard cap). Winners mostly exit at the cap before the time-stop, so the hold
mainly governs how fast we cut the LOSERS/non-movers.

Usage:  USE_YAHOO_BARS=1 .venv/bin/python lotto_hold_sweep.py
"""
import os
os.environ.setdefault("USE_YAHOO_BARS", "1")
from pathlib import Path

import lotto_backtest as lb
import config as cfg
from benchmark import compute_stats

INF = float("inf")
DELTA = cfg.LOTTO_TARGET_DELTA           # 0.25
DTE = 14                                  # new live target (8-21 window)
# shipped live exit: tiered "let it run" trail + 3× hard profit cap
EXIT_RULE = "tiered_profit"
EXIT_PARAMS = {"tiers": [(1.0, 0.40), (3.0, 0.30), (INF, 0.20)], "hard_target": 3.0}
HOLDS = [2, 3, 4, 5, 7, 10]


def line(tag, trades):
    if not trades:
        return f"  {tag:14} n=   0"
    s = compute_stats(trades)
    x2 = sum(1 for t in trades if t["pnl_pct"] >= 100)
    x4 = sum(1 for t in trades if t["pnl_pct"] >= 300)
    return (f"  {tag:14} n={len(trades):>4}  P&L=${s['total_pnl']:>8,.0f}  Sharpe={s['sharpe']:>5.2f}  "
            f"win={s['win_rate']:>4.1f}%  2x+={x2:>3} 4x+={x4:>3}")


def main():
    print(f"═══ LOTTO MAX-HOLD SWEEP — Δ{DELTA}, DTE {DTE}, tiered+3× cap (shipped exit) ═══")
    print("(winners exit at the 3× cap; hold governs how fast non-movers are cut)\n")
    lb.DTE = DTE
    for label, end_dt, days, cache in lb.WINDOWS:
        if not Path(cache).exists():
            print(f"[{label}] cache missing — skip"); continue
        scored = lb.tune_v2.load_scored_from_dual_cache(Path(cache), "bull", end_dt, days)
        gated = [r for r in scored if r["magnitude"] >= lb.LOTTO_MIN_MAG and r["confidence"] >= lb.LOTTO_MIN_CONF]
        reg = lb.build_regime(end_dt, days, 200)
        print(f"═══ {label} ═══  (gated: {len(gated)})")
        for h in HOLDS:
            lb.MAX_HOLD = h
            trades = lb.run_variant(gated, reg, DELTA, EXIT_RULE, EXIT_PARAMS, spread_mult=1.0)
            print(line(f"hold {h}d", trades))
        print()
    print("Read: prefer the shortest hold that keeps P&L/Sharpe + the 4x+ tail. If a shorter hold")
    print("matches or beats 7d, cut LOTTO_MAX_HOLD_DAYS — less theta bleed on non-movers.")


if __name__ == "__main__":
    main()
