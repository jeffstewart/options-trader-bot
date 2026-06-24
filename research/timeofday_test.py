"""
timeofday_test.py — does the news's TIME OF DAY (ET) change the edge?

Buckets each gated bullish signal by ET session and reports the 2-day stock return + win rate.
Two questions:
  • Intra-RTH variation — do open / midday / close signals differ (→ size or gate by phase)?
  • The bot only trades RTH, so PRE-MARKET / OVERNIGHT / AFTER-HOURS news is currently IGNORED.
    Do those off-hours signals carry edge we're leaving on the table (→ act at the next open)?

Usage:  USE_YAHOO_BARS=1 .venv/bin/python timeofday_test.py
"""
import os, statistics
os.environ.setdefault("USE_YAHOO_BARS", "1")
from datetime import datetime, timezone, timedelta
from zoneinfo import ZoneInfo
from pathlib import Path

import stock_backtest as sb
import tune_v2, config as cfg
from regime_filter import build_regime

sb.MAX_HOLD = cfg.NEWS_STOCK_MAX_HOLD_DAYS
BASE = 1000.0
ET = ZoneInfo("America/New_York")
WINDOWS = [
    ("bull-meltup", datetime.now(timezone.utc) - timedelta(days=5), 180, "dual_score_cache.json"),
    ("2022-bear",   datetime(2022, 6, 30, tzinfo=timezone.utc), 90, "bear_dual_cache.json"),
]
# (label, start_min, end_min) in ET minutes-from-midnight; RTH = open/midday/close
SESSIONS = [
    ("overnight  20:00-04:00", 20 * 60, 28 * 60),     # wraps; handled below
    ("pre-market 04:00-09:30", 4 * 60, 9 * 60 + 30),
    ("OPEN  09:30-11:00 [RTH]", 9 * 60 + 30, 11 * 60),
    ("MIDDAY 11:00-14:00 [RTH]", 11 * 60, 14 * 60),
    ("CLOSE 14:00-16:00 [RTH]", 14 * 60, 16 * 60),
    ("after-hrs 16:00-20:00", 16 * 60, 20 * 60),
]


def gate(m, c):
    return m >= cfg.MIN_MAGNITUDE and c >= cfg.BASE_CONFIDENCE + (1 - m) * cfg.CONFIDENCE_SLOPE


def session_of(dt_utc):
    et = dt_utc.astimezone(ET)
    mins = et.hour * 60 + et.minute
    for name, a, b in SESSIONS:
        if name.startswith("overnight"):
            if mins >= 20 * 60 or mins < 4 * 60:
                return name
        elif a <= mins < b:
            return name
    return "overnight  20:00-04:00"


def collect(rows, regime):
    out, seen = {s[0]: [] for s in SESSIONS}, {}
    for r in rows:
        m, c = r["magnitude"], r["confidence"]
        if not gate(m, c):
            continue
        d = r["created_at"].date()
        if regime and not regime(d):
            continue
        ds = seen.setdefault(d.isoformat(), set())
        for tk in r["tickers"][:2]:
            if tk in ds or tk in ("BTC", "ETH") or not sb.is_valid_stock_ticker(tk):
                continue
            ds.add(tk)
            try:
                t = sb.simulate_stock(tk, r["created_at"], BASE)
            except Exception:
                continue
            if t:
                out[session_of(r["created_at"])].append(t["pnl_pct"])
    return out


def main():
    print(f"═══ TIME-OF-DAY TEST — STOCK leg, {sb.MAX_HOLD}d hold, by ET session ═══")
    print("(the live bot only trades RTH → pre-market/overnight/after-hrs are currently IGNORED)\n")
    for label, end_dt, days, cache in WINDOWS:
        if not Path(cache).exists():
            print(f"[{label}] cache missing — skip"); continue
        scored = tune_v2.load_scored_from_dual_cache(Path(cache), "bull", end_dt, days)
        reg = build_regime(end_dt, days, 200)
        buckets = collect(scored, reg)
        total = sum(len(v) for v in buckets.values())
        print(f"═══ {label} ═══  total trades={total}")
        print(f"  {'session (ET)':26} {'n':>5} {'mean_ret':>9} {'median':>8} {'win%':>6}")
        for name, *_ in SESSIONS:
            v = buckets[name]
            if not v:
                print(f"  {name:26} {0:>5}"); continue
            win = sum(1 for r in v if r > 0) / len(v) * 100
            print(f"  {name:26} {len(v):>5} {statistics.mean(v):>+8.2f}% "
                  f"{sorted(v)[len(v)//2]:>+7.2f}% {win:>5.1f}%")
        print()
    print("Read: big edge differences across sessions → gate/size by phase. If pre-market/overnight")
    print("(currently ignored) shows strong edge, acting on it at the next open is opportunity.")


if __name__ == "__main__":
    main()
