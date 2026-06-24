"""
qqq_macro_eval.py — proper evaluation of the qqq_macro leg on unified_v1.

qqq_macro routes BULLISH signals with NO specific ticker (macro/market news) into a QQQ call.
The exit sweep showed only n=5 / negative, but that was a DATA ARTIFACT: get_price_at("QQQ")
flaked out under Yahoo rate-limiting (only 13/81 signal-days priced). Fix: PRE-WARM QQQ bars
over the whole window once, then evaluate the real ~81-trade sample.

Compares, net of realistic costs:
  • qqq_macro OPTION  (QQQ Δ0.60/DTE10 call, current tiered exit) — the live leg
  • QQQ STOCK         (buy QQQ, 10% trail, 5d) — does the option add value over just owning QQQ?
  • frictionless option — how much spread eats
Bull window + 2022-bear guard. ≤1 highest-conviction macro signal/day (matches qqq_routing).

Usage:  USE_YAHOO_BARS=1 .venv/bin/python -u qqq_macro_eval.py
"""
import os, random
os.environ.setdefault("USE_YAHOO_BARS", "1")
from datetime import datetime, timezone, timedelta

import backtest as _bt
import stock_backtest as _sb
import config as cfg
from benchmark import compute_stats
from regime_filter import build_regime
import exit_sweep_unified as es

N_BOOT = 10000
random.seed(20260614)


def prewarm(window_start, window_end):
    """Fetch QQQ bars over the full window ONCE so per-date get_price_at hits cache (no 429 flake)."""
    _bt.get_stock_bars("QQQ", window_start - timedelta(days=20), window_end + timedelta(days=40))


def macro_days(rows):
    """Pass dynamic floor, then keep the single highest-confidence macro signal per calendar day."""
    g = [r for r in rows if es.dyn_gate(r["magnitude"], r["confidence"])]
    by_day = {}
    for r in g:
        d = r["dt"].date()
        if d not in by_day or r["confidence"] > by_day[d]["confidence"]:
            by_day[d] = r
    return sorted(by_day.values(), key=lambda r: r["dt"])


def sim_qqq_option(picks, reg, spread_mult=1.0):
    save = {k: getattr(_bt, k) for k in ("TARGET_DELTA", "DTE_TARGET", "MAX_HOLD_DAYS", "TRAILING_STOP_PCT", "EXIT_PARAMS")}
    _bt.TARGET_DELTA, _bt.DTE_TARGET, _bt.MAX_HOLD_DAYS = cfg.QQQ_MACRO_TARGET_DELTA, 10, 10
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


def sim_qqq_stock(picks, reg):
    save = (_sb.STOCK_TRAIL, _sb.MAX_HOLD)
    _sb.STOCK_TRAIL, _sb.MAX_HOLD = 0.10, 5
    trades = []
    try:
        for r in picks:
            if reg and not reg(r["dt"].date()):
                continue
            t = _sb.simulate_stock("QQQ", r["dt"], es._scale(r["magnitude"], r["confidence"], cfg.MAX_POSITION_USD))
            if t:
                trades.append(t)
    finally:
        _sb.STOCK_TRAIL, _sb.MAX_HOLD = save
    return trades


def line(tag, s):
    if s["n"] == 0:
        return f"  {tag:28} n=0 (no data)"
    return (f"  {tag:28} n={s['n']:>3}  total=${s['total']:>8,.0f}  ${s['mean']:+,.0f}/tr  "
            f"Sh{s['sharpe']:>5.2f}  win{s['win']:>3.0f}%  2x+={s['x2']}")


def main():
    _, end, days, cache = es.BULL
    _, bend, bdays, bcache = es.BEAR
    start = end - timedelta(days=days)
    print("Pre-warming QQQ bars (bull + bear windows)…")
    prewarm(start, end)
    prewarm(bend - timedelta(days=bdays), bend)

    rows = es.load_signals(cache, end, days, "unified_v1", want_ticker=False)
    picks = macro_days(rows)
    reg = build_regime(end, days, 200)
    inreg = [p for p in picks if reg(p["dt"].date())]
    print(f"\nqqq_macro — bull window: {len(rows)} no-ticker bullish → {len(picks)} macro-days (≤1/day) "
          f"→ {len(inreg)} pass SPY>200d regime\n")

    opt   = es.stat(sim_qqq_option(picks, reg, spread_mult=1.0))
    optff = es.stat(sim_qqq_option(picks, reg, spread_mult=0.0))
    stk   = es.stat(sim_qqq_stock(picks, reg))
    print(line("qqq_macro OPTION (live)", opt))
    print(line("  frictionless option", optff) + f"   (cost drag ${opt['total']-optff['total']:,.0f})")
    print(line("QQQ STOCK (just own QQQ)", stk))

    if opt["n"] >= 20 and opt["pnls"]:
        pnls = opt["pnls"]; B = sorted(sum(random.choices(pnls, k=len(pnls))) for _ in range(N_BOOT))
        lo, hi = B[int(.025*N_BOOT)], B[int(.975*N_BOOT)]
        big = max(pnls)
        print(f"\n  OPTION bootstrap total P&L 95% CI [${lo:,.0f}, ${hi:,.0f}]  P>0={sum(1 for b in B if b>0)/N_BOOT*100:.0f}%")
        print(f"  OPTION jackknife: biggest ${big:,.0f} ({big/opt['total']*100:.0f}%) → drop = ${opt['total']-big:,.0f}")

    # bear guard
    brows = es.load_signals(bcache, bend, bdays, "unified_v1", want_ticker=False)
    bpicks = macro_days(brows)
    breg = build_regime(bend, bdays, 200)
    bopt = es.stat(sim_qqq_option(bpicks, breg, spread_mult=1.0))
    print(f"\n  BEAR guard (2022, regime-gated): " + line("", bopt).strip())


if __name__ == "__main__":
    main()
