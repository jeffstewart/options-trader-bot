"""
pnl_breakdown.py — where does the P&L come from: liquid large-caps/ETFs vs the
model's other (smaller-cap) picks? Informs whether restricting the universe to a
liquid handful keeps the edge while minimizing spread/slippage cost.

Runs the current live-config backtest (DTE/Δ/conf from config + tiered_profit
exit) on the bull-melt-up window (real Yahoo data), dumps per-trade P&L, and
buckets by ticker liquidity.

Usage:  USE_YAHOO_BARS=1 .venv/bin/python pnl_breakdown.py
"""
from datetime import datetime, timezone, timedelta
from pathlib import Path
from collections import defaultdict

import backtest as _bt
import tune_v2
from backtest import get_price_at, is_valid_stock_ticker
import config as _cfg

# The "liquid handful" = tradeable ETFs + mega-caps (tight option spreads)
MEGACAPS = {"AAPL", "MSFT", "NVDA", "AMD", "TSLA", "META", "AMZN", "GOOGL", "GOOG",
            "AVGO", "NFLX", "COST", "ADBE", "CRM", "ORCL", "QCOM", "INTC"}
LIQUID = set(_cfg.TRADEABLE_ETFS) | MEGACAPS

# Use the tiered exit with the live tiers
_bt.EXIT_PARAMS = {**_bt.EXIT_PARAMS, "tiers": _cfg.EXIT_TIERS}
END_DT = datetime.now(timezone.utc) - timedelta(days=5)


def run():
    scored = tune_v2.load_scored_from_dual_cache(Path("dual_score_cache.json"), "bull", END_DT, 180)

    def scale(mag, conf):
        return max(_cfg.MAX_POSITION_USD * 0.10, _cfg.MAX_POSITION_USD * mag * conf)

    trades, seen = [], {}
    for row in scored:
        mag, conf = row["magnitude"], row["confidence"]
        if mag < _cfg.MIN_MAGNITUDE:
            continue
        if conf < _cfg.BASE_CONFIDENCE + (1 - mag) * _cfg.CONFIDENCE_SLOPE:
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
                                        option_type="call", exit_rule="tiered_profit")
            if t:
                trades.append(t)
    return trades


def main():
    trades = run()
    total = sum(t["pnl_usd"] for t in trades)
    print(f"\nTotal: ${total:,.0f} over {len(trades)} trades\n")

    # bucket by liquidity
    buckets = {"liquid (ETF/mega)": [], "other (model picks)": []}
    for t in trades:
        key = "liquid (ETF/mega)" if t["ticker"] in LIQUID else "other (model picks)"
        buckets[key].append(t)
    print(f"{'bucket':22} {'trades':>7} {'P&L':>12} {'%P&L':>6} {'win%':>6} {'avg':>8}")
    for k, ts in buckets.items():
        pnl = sum(x["pnl_usd"] for x in ts)
        win = sum(1 for x in ts if x["pnl_usd"] > 0) / len(ts) * 100 if ts else 0
        avg = pnl / len(ts) if ts else 0
        print(f"  {k:20} {len(ts):>7} ${pnl:>11,.0f} {pnl/total*100 if total else 0:>5.0f}% {win:>5.1f}% ${avg:>7,.0f}")

    # per-ticker top contributors
    by_tk = defaultdict(lambda: [0.0, 0])
    for t in trades:
        by_tk[t["ticker"]][0] += t["pnl_usd"]; by_tk[t["ticker"]][1] += 1
    top = sorted(by_tk.items(), key=lambda kv: kv[1][0], reverse=True)[:15]
    print(f"\nTop 15 tickers by P&L (L=liquid):")
    for tk, (pnl, n) in top:
        tag = "L" if tk in LIQUID else " "
        print(f"  {tag} {tk:6} ${pnl:>10,.0f}  ({n} trades)")
    # distinct ticker counts
    distinct = len(by_tk); liq_distinct = sum(1 for tk in by_tk if tk in LIQUID)
    print(f"\nDistinct tickers: {distinct} ({liq_distinct} liquid, {distinct-liq_distinct} other)")


if __name__ == "__main__":
    main()
