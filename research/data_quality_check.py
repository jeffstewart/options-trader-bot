"""
data_quality_check.py — Compare Alpaca IEX historical prices vs Yahoo Finance
for a sample of tickers from the bull-window backtest period.

Verdict: if correlation is high and errors are small, the bull-window data is real
and all backtests on it are trustworthy. If prices diverge materially, the data is
synthetic/simulated and we should weight the 2022 Yahoo results much more heavily.

Usage:  python data_quality_check.py
"""
import os
import json
import statistics
from datetime import datetime, timezone, timedelta
from pathlib import Path

os.environ["USE_YAHOO_BARS"] = "1"

from backtest import get_stock_bars
from yahoo_data import get_yahoo_bars

# ── Sample: liquid names that appear often in the bull-window backtest ────────
TICKERS = [
    "AAPL", "MSFT", "NVDA", "AMZN", "GOOGL",   # mega-caps
    "JPM", "GS", "BAC",                           # financials
    "AMD", "INTC", "MU",                          # semis
    "NFLX", "META", "TSLA",                       # growth
    "SPY", "QQQ",                                 # ETFs
    "SNOW", "CRM", "ORCL",                        # enterprise tech
    "XOM", "CVX",                                 # energy (held up in 2022)
]

# Test window: pick a month inside our "bull melt-up" period
TEST_START = datetime(2026, 1, 15, tzinfo=timezone.utc)
TEST_END   = datetime(2026, 2, 15, tzinfo=timezone.utc)


def get_alpaca_prices(ticker, start, end):
    """Fetch via Alpaca IEX (same path backtest uses, USE_YAHOO_BARS=0)."""
    old = os.environ.get("USE_YAHOO_BARS")
    os.environ["USE_YAHOO_BARS"] = "0"
    try:
        bars = get_stock_bars(ticker, start, end)
    finally:
        if old is not None:
            os.environ["USE_YAHOO_BARS"] = old
        else:
            del os.environ["USE_YAHOO_BARS"]
    return {b["t"].date(): b["c"] for b in (bars or [])}


def get_yahoo_prices(ticker, start, end):
    bars = get_yahoo_bars(ticker, start, end)
    return {b["t"].date(): b["c"] for b in (bars or [])}


def compare(ticker):
    alpaca = get_alpaca_prices(ticker, TEST_START, TEST_END)
    yahoo  = get_yahoo_prices(ticker, TEST_START, TEST_END)
    common = sorted(set(alpaca) & set(yahoo))
    if len(common) < 5:
        return None
    a_vals = [alpaca[d] for d in common]
    y_vals = [yahoo[d]  for d in common]
    diffs  = [abs(a - y) / y * 100 for a, y in zip(a_vals, y_vals)]
    mean_a = statistics.mean(a_vals)
    mean_y = statistics.mean(y_vals)
    pct_diff_means = abs(mean_a - mean_y) / mean_y * 100
    return {
        "n":             len(common),
        "alpaca_mean":   round(mean_a, 2),
        "yahoo_mean":    round(mean_y, 2),
        "mean_abs_err%": round(statistics.mean(diffs), 2),
        "max_abs_err%":  round(max(diffs), 2),
        "price_level_match": pct_diff_means < 5.0,   # within 5% on average
    }


def main():
    print(f"Data quality check: Alpaca IEX vs Yahoo Finance")
    print(f"Window: {TEST_START.date()} → {TEST_END.date()}\n")

    results = {}
    mismatches = []
    matches    = []

    for tk in TICKERS:
        r = compare(tk)
        if r is None:
            print(f"  {tk:8}  insufficient data")
            continue
        results[tk] = r
        status = "✅" if r["price_level_match"] else "❌"
        print(f"  {tk:8}  Alpaca=${r['alpaca_mean']:>8.2f}  Yahoo=${r['yahoo_mean']:>8.2f}"
              f"  err={r['mean_abs_err%']:>5.2f}%  max={r['max_abs_err%']:>5.2f}%  {status}")
        if r["price_level_match"]:
            matches.append(tk)
        else:
            mismatches.append(tk)

    print(f"\n{'─'*65}")
    match_pct = len(matches) / max(len(results), 1) * 100
    print(f"  Match rate: {len(matches)}/{len(results)} tickers ({match_pct:.0f}%)")

    if match_pct >= 80:
        print("\n  ✅ VERDICT: Bull-window data appears REAL.")
        print("     Alpaca IEX prices align with Yahoo for most names.")
        print("     Backtest results on the bull window are trustworthy.")
    elif match_pct >= 50:
        print("\n  ⚠️  VERDICT: MIXED quality.")
        print("     Some names match, some don't. Apply caution to bull-window results.")
        print("     Weight 2022 Yahoo results more heavily for cross-regime conclusions.")
    else:
        print("\n  ❌ VERDICT: Bull-window data appears SYNTHETIC/SIMULATED.")
        print("     Alpaca IEX prices diverge significantly from Yahoo for most names.")
        print("     DO NOT trust bull-window backtest P&L figures.")
        print("     2022 Yahoo results are the only trustworthy cross-regime evidence.")

    if mismatches:
        print(f"\n  Mismatched tickers: {', '.join(mismatches)}")

    return match_pct


if __name__ == "__main__":
    main()
