"""
backfill_option_trend.py — retroactively compute the pre-entry option-price trend
(_option_recent_trend's exact logic from core/bot.py) for every historical option trade in
trades.csv, and append it to contract_trend.csv alongside the live data trailing_stop_monitor
has been logging since 2026-07-22. Historical bars are NOT subject to the OPRA <15min
real-time restriction (only very-recent windows 403), so this works for anything already in
the past -- which is all of trades.csv.

Skips (symbol, entry_dt) pairs already present in contract_trend.csv so it's safe to re-run.

Usage:  .venv/bin/python research/backfill_option_trend.py   (run from anywhere; writes
        relative to data/ via the hardcoded path below)
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
TREND_CSV = os.path.join(DATA_DIR, "contract_trend.csv")


def option_recent_trend(symbol: str, end_dt: datetime, lookback_min: int = 15, timeout_s: float = 10.0) -> dict:
    try:
        start = end_dt - timedelta(minutes=lookback_min)
        r = requests.get(
            "https://data.alpaca.markets/v1beta1/options/bars",
            headers={"APCA-API-KEY-ID": ALPACA_KEY, "APCA-API-SECRET-KEY": ALPACA_SECRET},
            params={"symbols": symbol, "timeframe": "1Min",
                    "start": start.strftime("%Y-%m-%dT%H:%M:%SZ"),
                    "end": end_dt.strftime("%Y-%m-%dT%H:%M:%SZ"), "limit": 50},
            timeout=timeout_s,
        )
        if r.status_code != 200:
            return {"n_bars": 0, "window_trend_pct": None, "recent_trend_pct": None, "error": f"HTTP {r.status_code}"}
        bars = sorted((r.json().get("bars") or {}).get(symbol, []), key=lambda b: b["t"])
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
        trades = [r for r in csv.DictReader(f) if r.get("option_symbol") not in ("-", "", None)]

    already = set()
    if os.path.exists(TREND_CSV):
        with open(TREND_CSV, newline="") as f:
            for r in csv.DictReader(f):
                already.add((r["symbol"], r["entry_dt"]))

    write_header = not os.path.exists(TREND_CSV)
    n_done = n_skip = n_err = 0
    with open(TREND_CSV, "a", newline="") as f:
        w = csv.writer(f)
        if write_header:
            w.writerow(["logged_at", "symbol", "strategy", "entry_dt",
                        "trend_n_bars", "trend_window_pct", "trend_recent_pct"])
        for t in trades:
            sym = t["option_symbol"]
            entry_dt = datetime.fromisoformat(t["timestamp"])
            key = (sym, entry_dt.isoformat())
            if key in already:
                n_skip += 1
                continue
            trend = option_recent_trend(sym, entry_dt)
            if trend.get("error"):
                n_err += 1
                print(f"  ERROR {sym} @ {entry_dt.isoformat()}: {trend['error']}")
            w.writerow([
                datetime.now(timezone.utc).isoformat(), sym, t.get("strategy", ""), entry_dt.isoformat(),
                trend["n_bars"],
                f"{trend['window_trend_pct']:.4f}" if trend.get("window_trend_pct") is not None else "",
                f"{trend['recent_trend_pct']:.4f}" if trend.get("recent_trend_pct") is not None else "",
            ])
            f.flush()
            already.add(key)
            n_done += 1
            time.sleep(0.2)

    print(f"\nbackfilled {n_done} trades, skipped {n_skip} already-present, {n_err} errors")


if __name__ == "__main__":
    main()
