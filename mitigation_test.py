"""
mitigation_test.py — is there a tighter-spread SUBSET that keeps a positive
NET-of-cost edge? Base case (no filter) keeps only 35% of P&L after spreads.

Sweeps tradeability filters (min option premium / max bid-ask spread), each run
NET of BASE (1.0×) spreads, on the bull-melt-up window. Reports surviving net
P&L, trade count, win%, and what fraction of trades pass the filter. The goal:
find a filter where net P&L stays solidly positive with enough trades — that's
the genuinely tradeable version. If none do, the edge is a frictionless mirage.

Usage:  USE_YAHOO_BARS=1 .venv/bin/python mitigation_test.py
"""
from datetime import datetime, timezone, timedelta
from pathlib import Path

import backtest as _bt
import tune_v2
from backtest import get_price_at, is_valid_stock_ticker
import config as _cfg

_bt.EXIT_PARAMS = {**_bt.EXIT_PARAMS, "tiers": _cfg.EXIT_TIERS}
END_DT = datetime.now(timezone.utc) - timedelta(days=5)

# (name, min_premium, max_entry_spread)
FILTERS = [
    ("no filter",            0.0, 1.00),
    ("premium ≥ $1",         1.0, 1.00),
    ("premium ≥ $2",         2.0, 1.00),
    ("spread ≤ 12%",         0.0, 0.12),
    ("spread ≤ 8%",          0.0, 0.08),
    ("prem≥$2 & spread≤10%", 2.0, 0.10),
]


def run(min_prem, max_spread, spread_mult=1.0):
    scored = tune_v2.load_scored_from_dual_cache(Path("dual_score_cache.json"), "bull", END_DT, 180)

    def scale(mag, conf):
        return max(_cfg.MAX_POSITION_USD * 0.10, _cfg.MAX_POSITION_USD * mag * conf)

    trades, seen, considered = [], {}, 0
    for row in scored:
        mag, conf = row["magnitude"], row["confidence"]
        if mag < _cfg.MIN_MAGNITUDE or conf < _cfg.BASE_CONFIDENCE + (1 - mag) * _cfg.CONFIDENCE_SLOPE:
            continue
        ds = seen.setdefault(row["created_at"].date().isoformat(), set())
        for tk in row["tickers"][:2]:
            if tk in ds or tk in ("BTC", "ETH") or not is_valid_stock_ticker(tk):
                continue
            ds.add(tk)
            px = get_price_at(tk, row["created_at"])
            if not px or px < _cfg.MIN_STOCK_PRICE:
                continue
            considered += 1
            t = _bt.simulate_option_pnl(tk, row["created_at"], px, scale(mag, conf),
                                        {"magnitude": mag, "confidence": conf, "reasoning": ""},
                                        option_type="call", exit_rule="tiered_profit",
                                        spread_mult=spread_mult, min_premium=min_prem,
                                        max_entry_spread=max_spread)
            if t:
                trades.append(t)
    return trades, considered


def main():
    print("Mitigation test — net of BASE (1.0×) spreads, bull-melt-up window.")
    print("Goal: a filter that keeps net P&L solidly positive with enough trades.\n")
    print(f"  {'filter':22} {'trades':>7} {'kept%':>6} {'NET P&L':>11} {'win%':>6} {'avg':>7}")
    # frictionless reference of unfiltered set for context
    fr, _ = run(0.0, 1.0, spread_mult=0.0)
    print(f"  {'(frictionless, all)':22} {len(fr):>7} {'':>6} ${sum(t['pnl_usd'] for t in fr):>10,.0f}  (gross reference)\n")
    for name, mp, ms in FILTERS:
        trades, considered = run(mp, ms, spread_mult=1.0)
        n = len(trades); pnl = sum(t["pnl_usd"] for t in trades)
        win = sum(1 for t in trades if t["pnl_usd"] > 0) / n * 100 if n else 0
        avg = pnl / n if n else 0
        kept = n / considered * 100 if considered else 0
        print(f"  {name:22} {n:>7} {kept:>5.0f}% ${pnl:>10,.0f} {win:>5.1f}% ${avg:>6,.0f}")


if __name__ == "__main__":
    main()
