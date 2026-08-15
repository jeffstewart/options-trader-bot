"""
backfill_stock_trend.py — research-only (2026-07-23): does the UNDERLYING STOCK's price trend
leading into entry predict trade P&L better than the option contract's own (thin, gappy) trend
did? backfill_option_trend.py found no signal in the option-bar version, but 58 of 109 trades
didn't even have enough option trade prints to compute a trend -- the underlying is dense and
has no data-availability restriction (confirmed 2026-07-22: stock bars return 200 in real time,
option bars 403 under 15min old), so this re-tests the same hypothesis with a cleaner instrument
before touching anything live. Covers ALL 198 trades.csv rows (source_ticker exists regardless
of strategy/asset type), not just the 98 with a real option contract.

Writes to stock_trend.csv (separate from contract_trend.csv -- this is not wired into bot.py).

Usage:  .venv/bin/python research/backfill_stock_trend.py
"""
import csv
import os
import time
from datetime import datetime, timedelta, timezone

import requests
from dotenv import load_dotenv

load_dotenv()

ALPACA_KEY = os.environ["ALPACA_API_KEY"]
ALPACA_SECRET = os.environ["ALPACA_SECRET_KEY"]
DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data")
TRADES_CSV = os.path.join(DATA_DIR, "trades.csv")
OUT_CSV = os.path.join(DATA_DIR, "stock_trend.csv")


def stock_recent_trend(ticker: str, end_dt: datetime, lookback_min: int = 15, timeout_s: float = 10.0) -> dict:
    try:
        start = end_dt - timedelta(minutes=lookback_min)
        r = requests.get(
            "https://data.alpaca.markets/v2/stocks/bars",
            headers={"APCA-API-KEY-ID": ALPACA_KEY, "APCA-API-SECRET-KEY": ALPACA_SECRET},
            params={"symbols": ticker, "timeframe": "1Min", "feed": "iex",
                    "start": start.strftime("%Y-%m-%dT%H:%M:%SZ"),
                    "end": end_dt.strftime("%Y-%m-%dT%H:%M:%SZ"), "limit": 50},
            timeout=timeout_s,
        )
        if r.status_code != 200:
            return {"n_bars": 0, "window_trend_pct": None, "recent_trend_pct": None, "error": f"HTTP {r.status_code}"}
        bars = sorted((r.json().get("bars") or {}).get(ticker, []), key=lambda b: b["t"])
    except Exception as e:
        return {"n_bars": 0, "window_trend_pct": None, "recent_trend_pct": None, "error": str(e)}

    if len(bars) < 2:
        return {"n_bars": len(bars), "window_trend_pct": None, "recent_trend_pct": None}

    first_c, last_c, prev_c = bars[0]["c"], bars[-1]["c"], bars[-2]["c"]
    window_trend = (last_c - first_c) / first_c * 100 if first_c else None
    recent_trend = (last_c - prev_c) / prev_c * 100 if prev_c else None
    return {"n_bars": len(bars), "window_trend_pct": window_trend, "recent_trend_pct": recent_trend}


def main():
    with open(TRADES_CSV, newline="") as f:
        trades = list(csv.DictReader(f))

    already = set()
    if os.path.exists(OUT_CSV):
        with open(OUT_CSV, newline="") as f:
            for r in csv.DictReader(f):
                already.add((r["symbol"], r["entry_dt"]))

    write_header = not os.path.exists(OUT_CSV)
    n_done = n_skip = n_err = 0
    with open(OUT_CSV, "a", newline="") as f:
        w = csv.writer(f)
        if write_header:
            w.writerow(["logged_at", "ticker", "symbol", "strategy", "entry_dt",
                        "trend_n_bars", "trend_window_pct", "trend_recent_pct"])
        for t in trades:
            ticker = t["source_ticker"]
            opt_sym = t.get("option_symbol") or "-"
            key_sym = opt_sym if opt_sym != "-" else ticker   # unique key per trade row
            entry_dt = datetime.fromisoformat(t["timestamp"])
            key = (key_sym, entry_dt.isoformat())
            if key in already:
                n_skip += 1
                continue
            trend = stock_recent_trend(ticker, entry_dt)
            if trend.get("error"):
                n_err += 1
                print(f"  ERROR {ticker} @ {entry_dt.isoformat()}: {trend['error']}")
            w.writerow([
                datetime.now(timezone.utc).isoformat(), ticker, key_sym, t.get("strategy", ""), entry_dt.isoformat(),
                trend["n_bars"],
                f"{trend['window_trend_pct']:.4f}" if trend.get("window_trend_pct") is not None else "",
                f"{trend['recent_trend_pct']:.4f}" if trend.get("recent_trend_pct") is not None else "",
            ])
            f.flush()
            already.add(key)
            n_done += 1
            time.sleep(0.15)

    print(f"\nbackfilled {n_done} trades, skipped {n_skip} already-present, {n_err} errors")


if __name__ == "__main__":
    main()
