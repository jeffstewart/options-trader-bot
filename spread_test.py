"""
spread_test.py — how much P&L survives realistic option bid/ask spreads + slippage?

The strategy's edge lives in small/mid-cap names (99% of P&L per pnl_breakdown),
which have WIDE option spreads. This runs the current live-config backtest across
spread scenarios (frictionless → tight → base → wide) and reports surviving P&L.
A sensitivity, not a single guess, since real spreads are uncertain.

  spread_mult: 0 = frictionless · 0.5 = optimistic · 1.0 = base · 1.5 = pessimistic
  (base spreads: liquid ETF/mega ~1.5-3%, mid/small-cap ~7-30% by premium, + ~1% slippage)

Usage:  USE_YAHOO_BARS=1 .venv/bin/python spread_test.py
"""
from datetime import datetime, timezone, timedelta
from pathlib import Path

import backtest as _bt
import tune_v2
from backtest import get_price_at, is_valid_stock_ticker, LIQUID_UNDERLYINGS
import config as _cfg

_bt.EXIT_PARAMS = {**_bt.EXIT_PARAMS, "tiers": _cfg.EXIT_TIERS}
END_DT = datetime.now(timezone.utc) - timedelta(days=5)
SCENARIOS = [("frictionless", 0.0), ("tight (0.5×)", 0.5), ("base (1.0×)", 1.0), ("wide (1.5×)", 1.5)]


def run(spread_mult):
    scored = tune_v2.load_scored_from_dual_cache(Path("dual_score_cache.json"), "bull", END_DT, 180)

    def scale(mag, conf):
        return max(_cfg.MAX_POSITION_USD * 0.10, _cfg.MAX_POSITION_USD * mag * conf)

    trades, seen = [], {}
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
            t = _bt.simulate_option_pnl(tk, row["created_at"], px, scale(mag, conf),
                                        {"magnitude": mag, "confidence": conf, "reasoning": ""},
                                        option_type="call", exit_rule="tiered_profit",
                                        spread_mult=spread_mult)
            if t:
                trades.append(t)
    return trades


def stats(trades):
    n = len(trades); pnl = sum(t["pnl_usd"] for t in trades)
    win = sum(1 for t in trades if t["pnl_usd"] > 0) / n * 100 if n else 0
    return n, pnl, win


def main():
    print("Spread/slippage sensitivity — bull-melt-up window, current live config + tiered exit\n")
    print(f"  {'scenario':16} {'trades':>7} {'P&L':>12} {'win%':>6} {'vs frictionless':>16}")
    base_pnl = None
    base_trades = None
    for name, m in SCENARIOS:
        trades = run(m)
        n, pnl, win = stats(trades)
        if m == 0.0:
            base_pnl = pnl
        if m == 1.0:
            base_trades = trades
        pct = f"{pnl/base_pnl*100:.0f}% kept" if base_pnl else ""
        print(f"  {name:16} {n:>7} ${pnl:>11,.0f} {win:>5.1f}% {pct:>16}")

    # Under base spread: liquid vs other erosion
    if base_trades:
        liq = [t for t in base_trades if t["ticker"] in LIQUID_UNDERLYINGS]
        oth = [t for t in base_trades if t["ticker"] not in LIQUID_UNDERLYINGS]
        print(f"\n  Under base spreads — liquid: ${sum(t['pnl_usd'] for t in liq):,.0f} ({len(liq)} tr)  |  "
              f"other: ${sum(t['pnl_usd'] for t in oth):,.0f} ({len(oth)} tr)")


if __name__ == "__main__":
    main()
