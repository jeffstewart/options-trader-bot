"""
qqq_macro_geom_sweep.py — sweep QQQ-macro option GEOMETRY (delta × DTE), NET of costs, to test
whether ANY strike/expiry makes the macro-bullish→QQQ-call leg net-positive. The QQQ STOCK
version is mildly positive (+$184/Sh2.85), so a small directional signal exists; the live
Δ0.60/DTE10 call can't capture it (theta/IV-crush/spread). Deeper-ITM (tighter spread, less
crush) + longer-DTE (slower macro drift) might. Hold = DTE (ride to ~expiry), tiered exit.

QQQ is PRE-WARMED over the full window first (the per-date get_price_at flakes under Yahoo
rate-limiting otherwise — that artifact made the exit sweep show a bogus n=5). Single leg →
single process is safe. Frictionless row included: if gross edge is ~0 everywhere, no geometry
saves it.

Usage:  USE_YAHOO_BARS=1 .venv/bin/python -u qqq_macro_geom_sweep.py
"""
import os, random
os.environ.setdefault("USE_YAHOO_BARS", "1")
from datetime import timedelta

import backtest as _bt
import config as cfg
from regime_filter import build_regime
import exit_sweep_unified as es
import qqq_macro_eval as qm

DELTAS = [0.40, 0.50, 0.60, 0.70, 0.80]
DTES   = [10, 14, 21, 30, 45]
N_BOOT = 10000
random.seed(20260614)


def sim(picks, reg, delta, dte, spread_mult=1.0):
    save = {k: getattr(_bt, k) for k in ("TARGET_DELTA", "DTE_TARGET", "MAX_HOLD_DAYS", "TRAILING_STOP_PCT", "EXIT_PARAMS")}
    _bt.TARGET_DELTA, _bt.DTE_TARGET, _bt.MAX_HOLD_DAYS = delta, dte, dte
    _bt.EXIT_PARAMS = {"tiers": list(cfg.EXIT_TIERS)}
    trades = []
    try:
        for r in picks:
            if reg and not reg(r["dt"].date()):
                continue
            sp = _bt.get_price_at("QQQ", r["dt"])
            if not sp:
                continue
            t = _bt.simulate_option_pnl("QQQ", r["dt"], sp, es._scale(r["magnitude"], r["confidence"], cfg.MAX_POSITION_USD),
                                        {"magnitude": r["magnitude"], "confidence": r["confidence"]},
                                        option_type="call", exit_rule="tiered_trail", spread_mult=spread_mult)
            if t:
                trades.append(t)
    finally:
        for k, v in save.items():
            setattr(_bt, k, v)
    return trades


def main():
    _, end, days, cache = es.BULL
    _, bend, bdays, bcache = es.BEAR
    start = end - timedelta(days=days)
    print("Pre-warming QQQ (bull window + longest DTE + bear)…")
    qm.prewarm(start, end + timedelta(days=50))
    qm.prewarm(bend - timedelta(days=bdays), bend + timedelta(days=50))

    rows = es.load_signals(cache, end, days, "unified_v1", want_ticker=False)
    picks = qm.macro_days(rows)
    reg = build_regime(end, days, 200)
    brows = es.load_signals(bcache, bend, bdays, "unified_v1", want_ticker=False)
    bpicks = qm.macro_days(brows)
    breg = build_regime(bend, bdays, 200)
    print(f"\nqqq_macro GEOMETRY sweep — {len(picks)} macro-days, NET of costs, hold=DTE, tiered exit\n")
    print("  total net P&L / Sharpe  (rows=Δ, cols=DTE):")
    print(f"  {'Δ \\ DTE':>8}" + "".join(f"{d:>13}" for d in DTES))
    res = {}
    for delta in DELTAS:
        cells = []
        for dte in DTES:
            s = es.stat(sim(picks, reg, delta, dte))
            res[(delta, dte)] = s
            cells.append(f"${s['total']/1000:>5.1f}k/{s['sharpe']:>5.2f}")
        print(f"  Δ{delta:>0.2f}  " + "".join(f"{c:>13}" for c in cells), flush=True)

    pos = [(k, s) for k, s in res.items() if s["n"] >= 20 and s["total"] > 0]
    if not pos:
        print("\n  VERDICT: NO geometry is net-positive (≥20 trades) → confirms qqq_macro has no "
              "tradeable edge as an option. The stock version (+$184) is the ceiling.")
        return
    (bd, bt), best = max(pos, key=lambda kv: kv[1]["sharpe"])
    ff = es.stat(sim(picks, reg, bd, bt, spread_mult=0.0))
    pnls = best["pnls"]; B = sorted(sum(random.choices(pnls, k=len(pnls))) for _ in range(N_BOOT))
    lo, hi = B[int(.025*N_BOOT)], B[int(.975*N_BOOT)]
    big = max(pnls)
    bs = es.stat(sim(bpicks, breg, bd, bt))
    print(f"\n  {len(pos)} net-positive cell(s). BEST: Δ{bd}/DTE{bt} → n={best['n']} ${best['total']:,.0f} "
          f"${best['mean']:+,.0f}/tr Sh{best['sharpe']:.2f} win{best['win']:.0f}% 2x+={best['x2']}")
    print(f"    frictionless ${ff['total']:,.0f} (gross edge before spread; cost drag ${best['total']-ff['total']:,.0f})")
    print(f"    bootstrap CI [${lo:,.0f}, ${hi:,.0f}] P>0={sum(1 for b in B if b>0)/N_BOOT*100:.0f}%  "
          f"jackknife drop-biggest(${big:,.0f})=${best['total']-big:,.0f}")
    print(f"    bear guard (2022): n={bs['n']} ${bs['total']:,.0f} Sh{bs['sharpe']:.2f}")
    print(f"\n  NOTE: weigh robustness (CI>0, not one-trade) + bear + whether it beats the +$184 QQQ-stock "
          "version by enough to justify the option's cost/complexity.")


if __name__ == "__main__":
    main()
