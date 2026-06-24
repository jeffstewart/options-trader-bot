"""
qqq_routing.py — test routing general-market bullish signals into QQQ calls.

The single-name strategy DISCARDS bullish signals that have no specific ticker
(macro/market news — "Fed signals cuts", "stocks rally on…"). 1,194 such
bullish signals passed the conviction gate but were never traded. This script
captures them as QQQ calls and asks whether that improves the strategy.

  • baseline  — the existing single-name strategy trades (backtest_trades.csv)
  • qqq_only  — QQQ calls on macro-bullish signals (≤1/day, highest conviction)
  • combined  — baseline + qqq_only

Reuses the validated BS pricing engine. No news re-fetch needed — macro-bullish
signals are already in backtest_signals.csv (passes_gate=True, empty tickers).

Usage:
    .venv/bin/python qqq_routing.py
"""

import csv
from datetime import datetime, timezone

import backtest as bt
from benchmark import compute_stats, fmt
from config import MAX_POSITION_USD


def load_baseline(path="backtest_trades.csv"):
    with open(path, newline="") as f:
        rows = list(csv.DictReader(f))
    for r in rows:
        r["pnl_usd"] = float(r["pnl_usd"])
        r["pnl_pct"] = float(r["pnl_pct"])
    return rows


def macro_bullish_signals(path="backtest_signals.csv"):
    """Bullish, gate-passing signals with NO specific ticker → one per day."""
    with open(path, newline="") as f:
        rows = list(csv.DictReader(f))
    macro = [r for r in rows
             if r.get("sentiment") == "bullish"
             and r.get("passes_gate") == "True"
             and not r.get("tickers", "").strip()]
    # Keep the highest-conviction macro signal per calendar day (avoid stacking)
    best_by_day = {}
    for r in macro:
        try:
            dt = datetime.fromisoformat(r["created_at"])
        except Exception:
            continue
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        day = dt.date().isoformat()
        score = float(r["magnitude"]) * float(r["confidence"])
        if day not in best_by_day or score > best_by_day[day][0]:
            best_by_day[day] = (score, dt, r)
    return [(dt, r) for _, dt, r in best_by_day.values()]


def main():
    baseline = load_baseline()
    macro = macro_bullish_signals()
    print(f"Baseline single-name trades : {len(baseline)}")
    print(f"Macro-bullish signals (≤1/day): {len(macro)}  → routing to QQQ calls\n")

    # Simulate QQQ calls on macro-bullish signals
    qqq_trades = []
    for dt, r in macro:
        mag, conf = float(r["magnitude"]), float(r["confidence"])
        pos = bt.scale_position_usd(MAX_POSITION_USD, mag, conf)
        px = bt.get_price_at("QQQ", dt)
        if not px:
            continue
        res = bt.simulate_option_pnl("QQQ", dt, px, pos,
                                     {"magnitude": mag, "confidence": conf, "reasoning": ""})
        if res:
            qqq_trades.append(res)

    arms = []
    s = compute_stats(baseline);            s["arm"] = "baseline";  arms.append(s)
    s = compute_stats(qqq_trades);          s["arm"] = "qqq_only";  arms.append(s)
    s = compute_stats(baseline + qqq_trades); s["arm"] = "combined"; arms.append(s)
    for s in arms:
        print(fmt(s))

    base, comb = arms[0], arms[2]
    print(f"\n{'='*70}\nVERDICT")
    print(f"  Baseline : ${base['total_pnl']:,.0f}  Sharpe {base['sharpe']:.2f}  maxDD ${base['max_dd']:,.0f}")
    print(f"  Combined : ${comb['total_pnl']:,.0f}  Sharpe {comb['sharpe']:.2f}  maxDD ${comb['max_dd']:,.0f}")
    dp = comb['total_pnl'] - base['total_pnl']
    ds = comb['sharpe'] - base['sharpe']
    print(f"  Δ from routing: P&L {dp:+,.0f}, Sharpe {ds:+.2f}")
    if ds >= 0 and dp > 0:
        print("  → Routing macro-bullish signals into QQQ improves the strategy.")
    elif dp > 0 and ds < 0:
        print("  → Routing adds P&L but lowers Sharpe (more beta exposure / drawdown).")
    else:
        print("  → Routing does not help.")


if __name__ == "__main__":
    main()
