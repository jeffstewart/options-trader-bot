"""
stock_backtest.py — trade the UNDERLYING STOCK on the news signal instead of options.

Almost everything that destroyed the options edge was option-specific (IV crush,
wide option spreads, theta). Stocks have penny-wide spreads and none of that. This
tests whether the signal's binary name-selection edge survives once those frictions
are gone — same signals, same regime gate, just a stock position.

Per trade: buy stock at the news-day price (+ tiny slippage), trailing-stop on the
STOCK price, hold ≤ MAX_HOLD_DAYS. Cross-regime (bull-melt-up + 2022-bear), with and
without the 200-day SPY regime gate.

Usage:  USE_YAHOO_BARS=1 .venv/bin/python stock_backtest.py
"""
from datetime import datetime, timezone, timedelta
from pathlib import Path

import backtest as _bt
import tune_v2
from backtest import get_price_at, get_stock_bars, is_valid_stock_ticker
from benchmark import compute_stats
from regime_filter import build_regime
import config as _cfg

STOCK_SLIPPAGE = 0.001   # 0.10% round-trip (generous for liquid stocks)
STOCK_TRAIL    = 0.10    # 10% trailing stop on the stock price
MAX_HOLD       = 30      # trading days
WINDOWS = [
    ("bull-meltup", datetime.now(timezone.utc) - timedelta(days=5), 180, "dual_score_cache.json"),
    ("2022-bear",   datetime(2022, 6, 30, tzinfo=timezone.utc),     90,  "bear_dual_cache.json"),
]


def simulate_stock(ticker, entry_dt, position_usd):
    bars = get_stock_bars(ticker, entry_dt, entry_dt + timedelta(days=int(MAX_HOLD * 1.5) + 7))
    tb = [b for b in bars if entry_dt < b["t"]]
    if not tb or (tb[0]["t"] - entry_dt).days > 7:
        return None
    p0 = get_price_at(ticker, entry_dt)
    if not p0 or p0 < _cfg.MIN_STOCK_PRICE or p0 > 10000:
        return None
    # data-sanity: skip implausible single-day jumps
    prev = p0
    for b in tb:
        if prev > 0 and abs(b["c"] / prev - 1) > 0.5:
            return None
        prev = b["c"]
    entry_fill = p0 * (1 + STOCK_SLIPPAGE / 2)
    shares = position_usd / entry_fill
    peak = p0
    exit_price = tb[min(MAX_HOLD, len(tb)) - 1]["c"]
    for b in tb[:MAX_HOLD]:
        peak = max(peak, b["c"])
        if b["c"] <= peak * (1 - STOCK_TRAIL):
            exit_price = b["c"]
            break
    exit_fill = exit_price * (1 - STOCK_SLIPPAGE / 2)
    pnl = (exit_fill - entry_fill) * shares
    return {"ticker": ticker, "entry_dt": entry_dt.isoformat(),
            "pnl_usd": pnl, "pnl_pct": (exit_fill / entry_fill - 1) * 100}


def run(end_dt, days, cache, regime=None):
    scored = tune_v2.load_scored_from_dual_cache(Path(cache), "bull", end_dt, days)

    def scale(mag, conf):
        return max(_cfg.MAX_POSITION_USD * 0.10, _cfg.MAX_POSITION_USD * mag * conf)

    trades, seen = [], {}
    for row in scored:
        mag, conf = row["magnitude"], row["confidence"]
        if mag < _cfg.MIN_MAGNITUDE or conf < _cfg.BASE_CONFIDENCE + (1 - mag) * _cfg.CONFIDENCE_SLOPE:
            continue
        d = row["created_at"].date()
        if regime is not None and not regime(d):
            continue
        ds = seen.setdefault(d.isoformat(), set())
        for tk in row["tickers"][:2]:
            if tk in ds or tk in ("BTC", "ETH") or not is_valid_stock_ticker(tk):
                continue
            ds.add(tk)
            t = simulate_stock(tk, row["created_at"], scale(mag, conf))
            if t:
                trades.append(t)
    return trades


def line(tag, s):
    return (f"  {tag:22} trades={s['trades']:>4}  P&L=${s['total_pnl']:>9,.0f}  "
            f"Sharpe={s['sharpe']:>5.2f}  win={s['win_rate']:>4.1f}%  maxDD=${s['max_dd']:>8,.0f}")


def main():
    print(f"STOCK strategy backtest (slippage {STOCK_SLIPPAGE*100:.2f}% RT, {STOCK_TRAIL*100:.0f}% trail, "
          f"≤{MAX_HOLD}d, sized like options)\n")
    for label, end_dt, days, cache in WINDOWS:
        print(f"═══ {label} ═══")
        nofilt = run(end_dt, days, cache, regime=None)
        print(line("no regime gate", compute_stats(nofilt)))
        reg = build_regime(end_dt, days, 200)
        gated = run(end_dt, days, cache, regime=reg)
        print(line("SPY>200d gated", compute_stats(gated)))
        print()


if __name__ == "__main__":
    main()
