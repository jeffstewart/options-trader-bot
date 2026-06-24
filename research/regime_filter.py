"""
regime_filter.py — does a market-regime gate fix the bull strategy's bear-market losses?

Cross-regime backtest showed the news_call (long-call) strategy makes money in a
bull melt-up (+$51K) but LOSES in the 2022 bear (-$1.9K) — it's leveraged long
beta. Hypothesis: only take call trades when the broad market is in an UPTREND
(SPY above its N-day SMA), so the strategy sits out downturns.

This runs the winning bull combo (DTE 14-21, Δ0.50, Conf0.65, Trail15%) WITH and
WITHOUT the regime gate, across BOTH windows (bull-melt-up + 2022-bear), on real
Yahoo data. The filter is a fixed rule (not tuned), so we evaluate on the full
window. Tests 50-day and 200-day SMA.

Usage:  USE_YAHOO_BARS=1 .venv/bin/python regime_filter.py
"""
from datetime import datetime, timezone, timedelta
from pathlib import Path

import backtest as _bt
import tune_v2
from backtest import get_price_at, is_valid_stock_ticker
from benchmark import compute_stats
import config as _cfg
from yahoo_data import get_yahoo_bars

PARAMS = {"TARGET_DELTA": 0.50, "MIN_MAGNITUDE": 0.25, "BASE_CONFIDENCE": 0.65,
          "TRAILING_STOP_PCT": 0.15}
MIN_DTE, MAX_DTE = 14, 21
REGIME_TICKER = "SPY"

WINDOWS = [
    ("bull-melt-up", datetime.now(timezone.utc) - timedelta(days=5), 180, "dual_score_cache.json"),
    ("2022-bear",    datetime(2022, 6, 30, tzinfo=timezone.utc),     90,  "bear_dual_cache.json"),
]


def apply_combo():
    _bt.MIN_DAYS_TO_EXPIRY = MIN_DTE
    _bt.MAX_DAYS_TO_EXPIRY = MAX_DTE
    _bt.DTE_TARGET = round((MIN_DTE + MAX_DTE) / 2)
    _bt.TARGET_DELTA = PARAMS["TARGET_DELTA"]
    _bt.TRAILING_STOP_PCT = PARAMS["TRAILING_STOP_PCT"]


import yahoo_data

def _spy_series(end_dt, lookback_days):
    """Fetch SPY daily closes directly via Yahoo (curl_cffi + retries), bypassing
    the shared on-disk bar cache (which concurrent runs can corrupt). Returns
    (dates[], closes[]) sorted ascending."""
    import sys
    p1 = int((end_dt - timedelta(days=lookback_days)).timestamp())
    p2 = int(end_dt.timestamp())
    payload = yahoo_data._fetch_yahoo(REGIME_TICKER, p1, p2, tries=6)
    parsed = yahoo_data._parse(payload) if payload else {}
    items = sorted(parsed.items())
    dates = [datetime.fromisoformat(d).date() for d, _ in items]
    closes = [v[3] for _, v in items]   # close
    print(f"[build_regime] {REGIME_TICKER} end={end_dt.date()} lookback={lookback_days}d → {len(dates)} bars", file=sys.stderr)
    return dates, closes


def build_regime(end_dt, days, ma):
    """Return uptrend(date)->bool: SPY close >= its trailing ma-day SMA."""
    # Lookback must cover the window PLUS ma trading days of prior history.
    dates, vals = _spy_series(end_dt, int((days + ma) * 1.7) + 60)

    def uptrend(day):
        import bisect
        i = bisect.bisect_right(dates, day) - 1
        if i < ma:                       # not enough history → don't trade
            return False
        sma = sum(vals[i - ma + 1:i + 1]) / ma
        return vals[i] >= sma
    return uptrend


def run_window(label, end_dt, days, cache, uptrend=None):
    apply_combo()
    scored = tune_v2.load_scored_from_dual_cache(Path(cache), "bull", end_dt, days)

    def scale(mag, conf):
        return max(_cfg.MAX_POSITION_USD * 0.10, _cfg.MAX_POSITION_USD * mag * conf)

    trades, seen, skipped_regime = [], {}, 0
    for row in scored:
        mag, conf = row["magnitude"], row["confidence"]
        if mag < PARAMS["MIN_MAGNITUDE"]:
            continue
        if conf < PARAMS["BASE_CONFIDENCE"] + (1 - mag) * _cfg.CONFIDENCE_SLOPE:
            continue
        d = row["created_at"].date()
        if uptrend is not None and not uptrend(d):
            skipped_regime += 1
            continue
        ds = seen.setdefault(d.isoformat(), set())
        for tk in row["tickers"][:2]:
            if tk in ds or tk in ("BTC", "ETH") or not is_valid_stock_ticker(tk):
                continue
            ds.add(tk)
            px = get_price_at(tk, row["created_at"])
            if not px or px < _cfg.MIN_STOCK_PRICE:
                continue
            t = _bt.simulate_option_pnl(tk, row["created_at"], px, scale(mag, conf),
                                        {"magnitude": mag, "confidence": conf, "reasoning": ""},
                                        option_type="call")
            if t:
                trades.append(t)
    s = compute_stats(trades)
    s["skipped_regime"] = skipped_regime
    return s


def fmt(label, s):
    return (f"  {label:28} trades={s['trades']:>4}  P&L=${s['total_pnl']:>10,.0f}  "
            f"Sharpe={s['sharpe']:>6.2f}  win={s['win_rate']:>4.1f}%")


def main():
    print(f"Regime filter test — {REGIME_TICKER} SMA gate on the news_call strategy\n")
    for label, end_dt, days, cache in WINDOWS:
        print(f"═══ {label} window ═══")
        base = run_window(label, end_dt, days, cache, uptrend=None)
        print(fmt("no filter", base))
        for ma in (50, 200):
            up = build_regime(end_dt, days, ma)
            f = run_window(label, end_dt, days, cache, uptrend=up)
            print(fmt(f"filter: SPY>{ma}d SMA", f) + f"   (skipped {f['skipped_regime']} down-regime signals)")
        print()


if __name__ == "__main__":
    main()
