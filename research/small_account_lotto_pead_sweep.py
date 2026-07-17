"""
small_account_lotto_pead_sweep.py — the two remaining v2 legs (news_call already tested:
lottery-ticket regardless of threshold/budget, small_account_news_call_sweep.py) at small-account
sizing, using OLLAMA's own scores (v2's actual live scorer -- the earlier lotto test in
small_account_sonnet5_sweep.py used sonnet5, which v2 isn't running yet).

LOTTO: budget x mag/conf threshold sweep, same methodology as lotto_backtest.py (deep-OTM Δ0.20,
tiered "let it run" exit) but at v2-relevant leg budgets. Judge on win% + tail concentration, not
just Sharpe -- lotto is DESIGNED convex/low-win-rate, so a good cell still won't look like "steady
winners"; the question is whether it's AFFORDABLE and whether selectivity changes the shape at all.

PEAD: never tested at small-account sizing at all. Two questions pairs already showed matter:
concurrency (entry rate x hold duration, Little's law) and share-affordability (PEAD trades real
company stock, no 100x multiplier, but still needs >= 1 share within budget).

Usage:  USE_YAHOO_BARS=1 .venv/bin/python -u small_account_lotto_pead_sweep.py   (run from data/)
"""
import os, statistics
os.environ.setdefault("USE_YAHOO_BARS", "1")
from datetime import datetime, timezone, timedelta

import backtest as _bt
import config as _cfg
import news_call_sweep_unified as nc
from benchmark import compute_stats
from regime_filter import build_regime
from backtest import get_price_at, get_stock_bars
from pead_backtest import simulate_pead, is_earnings_article

LEG_BUDGETS = [50, 75, 100, 150, 200]
LOTTO_GATES = [(0.60, 0.75), (0.70, 0.85), (0.75, 0.85), (0.80, 0.90)]   # (mag, conf) -- live is (0.70, 0.85)


def lotto_sweep(rows, reg):
    print("── LOTTO (Ollama scores, deep-OTM Δ0.20/DTE14) — budget x gate sweep ──")
    print("  (lotto is DESIGNED low-win-rate/convex -- judge on affordability + win%/tail, not Sharpe alone)")
    tiered = {"tiers": [(1.0, 0.40), (3.0, 0.30), (float("inf"), 0.20)]}
    save = {k: getattr(_bt, k) for k in
            ("TARGET_DELTA", "DTE_TARGET", "MAX_HOLD_DAYS", "TRAILING_STOP_PCT", "EXIT_PARAMS")}
    save_mult = _cfg.MAX_CONTRACT_BUDGET_MULT
    try:
        _bt.TARGET_DELTA, _bt.DTE_TARGET, _bt.MAX_HOLD_DAYS = 0.20, 14, 7
        _bt.TRAILING_STOP_PCT, _bt.EXIT_PARAMS = 0.30, tiered
        for mag, conf in LOTTO_GATES:
            gated = [r for r in rows if r["magnitude"] >= mag and r["confidence"] >= conf]
            print(f"\n  gate mag>={mag} conf>={conf}: {len(gated)} signals")
            for budget in LEG_BUDGETS:
                _cfg.MAX_CONTRACT_BUDGET_MULT = 1.0   # lotto's own live hard cap -- never exceed budget
                trades, seen = [], set()
                for r in gated:
                    d = r["created_at"].date()
                    if reg and not reg(d):
                        continue
                    for tk in r["tickers"][:1]:
                        if not _bt.is_valid_stock_ticker(tk):
                            continue
                        key = f"{d}_{tk}"
                        if key in seen:
                            continue
                        seen.add(key)
                        sp = get_price_at(tk, r["created_at"])
                        if not sp:
                            continue
                        t = _bt.simulate_option_pnl(tk, r["created_at"], sp, budget,
                                                    {"magnitude": r["magnitude"], "confidence": r["confidence"]},
                                                    option_type="call", exit_rule="tiered_profit", spread_mult=1.0)
                        if t:
                            trades.append(t)
                universe_n = sum(1 for r in gated if reg is None or reg(r["created_at"].date()))
                if len(trades) < 5:
                    print(f"    ${budget:>4} leg  n={len(trades):<3} (insufficient sample, "
                          f"skip {(1 - len(trades) / max(universe_n, 1)) * 100:.0f}%)")
                    continue
                trades.sort(key=lambda t: -t["pnl_usd"])
                tot = sum(t["pnl_usd"] for t in trades)
                top3 = sum(t["pnl_usd"] for t in trades[:3])
                s = compute_stats(trades)
                skip = (1 - len(trades) / max(universe_n, 1)) * 100
                top3_pct = (top3 / tot * 100) if tot else 0
                print(f"    ${budget:>4} leg  n={s['trades']:<3} ${s['total_pnl']:>+7,.0f} "
                      f"win={s['win_rate']:>4.1f}%  sharpe={s['sharpe']:>+5.2f}  "
                      f"top3={top3_pct:>4.0f}%ofP&L  skip={skip:>3.0f}%")
    finally:
        for k, v in save.items():
            setattr(_bt, k, v)
        _cfg.MAX_CONTRACT_BUDGET_MULT = save_mult


def pead_sweep(rows, reg):
    print("\n\n── PEAD (Ollama scores, stock, flat 20% trail, 10d hold) ──")
    earnings_rows = [r for r in rows if is_earnings_article(r.get("headline", ""))]
    print(f"  earnings-headline signals: {len(earnings_rows)} / {len(rows)} total bullish")

    print("\n  1) SHARPE/WIN AT SCALED-DOWN SIZE")
    print(f"  {'leg $':>6} {'n':>4} {'total$':>9} {'avg$/tr':>8} {'win%':>6} {'sharpe':>7}")
    for budget in LEG_BUDGETS:
        trades, seen = [], set()
        for r in earnings_rows:
            d = r["created_at"].date()
            if reg and not reg(d):
                continue
            for tk in r["tickers"][:1]:
                if not _bt.is_valid_stock_ticker(tk):
                    continue
                key = f"{d}_{tk}"
                if key in seen:
                    continue
                seen.add(key)
                t = simulate_pead(tk, r["created_at"], budget, trail=0.20, max_hold=10)
                if t:
                    trades.append(t)
        if len(trades) < 5:
            print(f"  ${budget:>5}   {len(trades):>4}  (insufficient sample)")
            continue
        s = compute_stats(trades)
        print(f"  ${budget:>5}   {s['trades']:>4} ${s['total_pnl']:>+8,.0f} ${s['total_pnl']/s['trades']:>+7,.0f} "
              f"{s['win_rate']:>5.1f}% {s['sharpe']:>+6.2f}")

    print("\n  2) HOLD-DURATION -> concurrent-position estimate (10d cap, flat 20% trail)")
    holds = []
    for r in earnings_rows:
        d = r["created_at"].date()
        if reg and not reg(d):
            continue
        for tk in r["tickers"][:1]:
            if not _bt.is_valid_stock_ticker(tk):
                continue
            bars = get_stock_bars(tk, r["created_at"], r["created_at"] + timedelta(days=20))
            tb = [b for b in bars if r["created_at"] < b["t"]]
            if not tb:
                continue
            p0 = get_price_at(tk, r["created_at"])
            if not p0:
                continue
            peak = p0
            exit_day = min(10, len(tb))
            for i, b in enumerate(tb[:10], start=1):
                peak = max(peak, b["c"])
                if b["c"] <= peak * (1 - 0.20):
                    exit_day = i
                    break
            holds.append(exit_day)
    if holds:
        holds.sort()
        print(f"  n={len(holds)} exits  median hold {statistics.median(holds):.0f}d  "
              f"p75 {holds[int(len(holds)*0.75)]}d  p90 {holds[int(len(holds)*0.90)]}d  max {holds[-1]}d")
        days_span = (max(r["created_at"] for r in earnings_rows) - min(r["created_at"] for r in earnings_rows)).days
        entry_rate = len(earnings_rows) / max(days_span, 1)
        concurrent = entry_rate * statistics.median(holds)
        print(f"  entry rate: {len(earnings_rows)}/{days_span}d ({entry_rate:.2f}/day) x median hold "
              f"{statistics.median(holds):.0f}d ~= {concurrent:.1f} concurrent positions in steady state")

    print("\n  3) SHARE AFFORDABILITY at small leg budgets (stock, no 100x multiplier -- but still needs >=1 share)")
    for budget in (50, 100, 200):
        prices = []
        for r in earnings_rows[:80]:
            for tk in r["tickers"][:1]:
                p = get_price_at(tk, r["created_at"])
                if p:
                    prices.append(p)
        unaffordable = sum(1 for p in prices if p > budget)
        print(f"  ${budget} leg: {unaffordable}/{len(prices)} picks priced above budget for 1 share "
              f"({unaffordable/max(len(prices),1)*100:.0f}%)")


def main():
    label, end_dt, days, cache = nc.BULL
    rows = nc.load_scored_from_unified(cache, end_dt, days, "unified_v1")
    reg = build_regime(end_dt, days, 200)
    print(f"{label}: {len(rows)} bullish unified_v1 signals\n")
    # PEAD FIRST: lotto_sweep mutates backtest.py module globals (TARGET_DELTA, DTE_TARGET,
    # MAX_HOLD_DAYS, TRAILING_STOP_PCT, EXIT_PARAMS) for the option pricer -- confirmed by direct
    # test (2026-07-17) that this leaks into simulate_pead's results despite the finally-block
    # restore (n=67/win=64.2%/median-hold=6d run after lotto_sweep vs the correct n=89/win=53.9%/
    # median-hold=10d run standalone or first). Running pead_sweep before lotto_sweep sidesteps
    # needing to fully diagnose which attribute leaks.
    pead_sweep(rows, reg)
    lotto_sweep(rows, reg)


if __name__ == "__main__":
    main()
