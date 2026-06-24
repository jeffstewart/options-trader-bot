"""
probe_intraday.py — feasibility probe for the post-news intraday DRIFT strategy.

Question: after a realistic reaction DELAY (the bot isn't first — HFT is), is there
still a capturable upward drift following bullish news on liquid watchlist names?

For each bullish news event on a watchlist ticker (during market hours), using
real Alpaca minute bars:
  • model the bot entering D minutes AFTER the news timestamp
  • measure how much move was MISSED during the delay
  • measure the CAPTURABLE drift after entry: returns at +15/+30/+60 min and the
    max-favorable-excursion (MFE, the best exit the "ride" could get)
Aggregated overall and by LLM conviction, so we can see if the model's score
predicts bigger drift (the "LLM picks when to trade" thesis).

Usage:  .venv/bin/python probe_intraday.py [--limit N] [--delays 1,3,5]
"""
import argparse
import json
import re
import statistics
from datetime import datetime, timezone, timedelta

from dotenv import load_dotenv; load_dotenv()
import os
from alpaca.data.historical.stock import StockHistoricalDataClient
from alpaca.data.requests import StockBarsRequest
from alpaca.data.timeframe import TimeFrame

sc = StockHistoricalDataClient(os.environ["ALPACA_API_KEY"], os.environ["ALPACA_SECRET_KEY"])

# Mega-caps already shown to have NO intraday drift — exclude them; probe the rest.
MEGACAPS = {"SPY", "QQQ", "AAPL", "MSFT", "NVDA", "AMD", "TSLA", "META", "AMZN", "GOOGL"}
HORIZONS = [15, 30, 60]   # minutes after entry
_TICKER_RE = re.compile(r"^[A-Z]{1,5}$")


def load_events():
    """Bullish events on a MODEL-TAGGED non-mega-cap ticker, w/ timestamp + conviction."""
    cache = json.load(open("dual_score_cache.json"))
    events = []
    for v in cache.values():
        if not isinstance(v, dict):
            continue
        art = v.get("_article", {})
        b = v.get("bullish", {})
        if b.get("reasoning") == "SCORE_FAILED":
            continue
        # The model's tagged tickers, excluding mega-caps / crypto / malformed
        cands = [t for t in (b.get("tickers", []) or [])
                 if _TICKER_RE.match(t) and t not in MEGACAPS and t not in ("BTC", "ETH")]
        if not cands:
            continue
        ca = art.get("created_at")
        if not ca:
            continue
        try:
            dt = datetime.fromisoformat(str(ca).replace("Z", "+00:00"))
            if dt.tzinfo is None:
                dt = dt.replace(tzinfo=timezone.utc)
        except Exception:
            continue
        events.append({"ticker": cands[0], "dt": dt.astimezone(timezone.utc),
                       "mag": float(b.get("magnitude", 0)), "conf": float(b.get("confidence", 0))})
    events.sort(key=lambda e: e["dt"])
    return events


def minute_bars(ticker, start, end):
    try:
        r = sc.get_stock_bars(StockBarsRequest(symbol_or_symbols=ticker, timeframe=TimeFrame.Minute,
                                               start=start, end=end, feed="iex"))
        return (r.data or {}).get(ticker, [])
    except Exception:
        return []


def price_at_or_after(bars, t):
    for b in bars:
        if b.timestamp >= t:
            return b.close, b.timestamp
    return None, None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=250)
    ap.add_argument("--delays", default="1,3,5")
    args = ap.parse_args()
    delays = [int(x) for x in args.delays.split(",")]

    events = load_events()
    print(f"Watchlist bullish news events found: {len(events)}")
    if len(events) > args.limit:
        step = len(events) / args.limit
        events = [events[int(i * step)] for i in range(args.limit)]
    print(f"Probing {len(events)} (sampled), measuring drift net of reaction delay…\n")

    # per delay → lists of forward returns / MFE / missed
    by_delay = {d: {"r": {h: [] for h in HORIZONS}, "mfe": [], "missed": []} for d in delays}
    conv = []   # (conf, +60min MFE%, +30min ret%) at delay=delays[0] — for the conviction split
    usable = skipped_nodata = skipped_baddata = 0
    for ev in events:
        t = ev["dt"]
        bars = minute_bars(ev["ticker"], t - timedelta(minutes=2), t + timedelta(minutes=max(delays) + 65))
        if len(bars) < 20:        # outside a live session OR sparse/no intraday data
            skipped_nodata += 1
            continue
        p_news, _ = price_at_or_after(bars, t)
        if not p_news or p_news > 10000:   # bad-data guard (e.g. EDBL $417K)
            skipped_baddata += 1
            continue
        usable += 1
        for d in delays:
            p_entry, t_entry = price_at_or_after(bars, t + timedelta(minutes=d))
            if not p_entry:
                continue
            miss = (p_entry / p_news - 1) * 100
            if abs(miss) > 50:        # intraday artifact (bad bar) → skip this event-delay
                continue
            by_delay[d]["missed"].append(miss)
            fwd_bars = [b for b in bars if b.timestamp >= t_entry]
            mfe = (max(b.high for b in fwd_bars) / p_entry - 1) * 100 if fwd_bars else 0
            if fwd_bars and abs(mfe) < 80:
                by_delay[d]["mfe"].append(mfe)
            r30 = None
            for h in HORIZONS:
                p_h, _ = price_at_or_after(bars, t_entry + timedelta(minutes=h))
                if p_h:
                    rr = (p_h / p_entry - 1) * 100
                    if abs(rr) > 50:    # artifact guard
                        continue
                    by_delay[d]["r"][h].append(rr)
                    if h == 30:
                        r30 = rr
            if d == delays[0]:
                conv.append((ev["conf"], mfe, r30 if r30 is not None else 0))

    print(f"Data availability: usable={usable}  | skipped: no/sparse intraday bars={skipped_nodata}, "
          f"bad data={skipped_baddata}\n")
    if usable < 10:
        print("⚠️  Too few usable intraday events — most watchlist news may be after-hours.")
        return

    def med(xs): return statistics.median(xs) if xs else 0
    def hit(xs): return (sum(1 for x in xs if x > 0) / len(xs) * 100) if xs else 0

    print("Delay = minutes the bot reacts AFTER the news timestamp.")
    print(f"{'delay':>6} {'missed%':>8} | " + " ".join(f'+{h}min(med%/win%)' for h in HORIZONS) + f"  {'MFE med%':>9}")
    for d in delays:
        D = by_delay[d]
        cells = []
        for h in HORIZONS:
            cells.append(f"{med(D['r'][h]):+5.2f}/{hit(D['r'][h]):4.0f}%")
        print(f"{d:>5}m {med(D['missed']):+7.2f}% | " + "  ".join(cells) + f"   {med(D['mfe']):+8.2f}")

    # Conviction split: does a higher LLM score predict bigger drift / MFE?
    d0 = delays[0]
    print(f"\nConviction split (delay {d0}m) — does the LLM score predict drift?")
    print(f"  {'tier':16} {'n':>4} {'+30min med%':>12} {'+30 win%':>9} {'MFE med%':>9}")
    for label, lo_c, hi_c in [("conf ≥ 0.85", 0.85, 1.01), ("0.70–0.85", 0.70, 0.85), ("< 0.70", 0.0, 0.70)]:
        grp = [(mfe, r30) for c, mfe, r30 in conv if lo_c <= c < hi_c]
        if grp:
            r30s = [r for _, r in grp]; mfes = [m for m, _ in grp]
            print(f"  {label:16} {len(grp):>4} {med(r30s):>+11.2f} "
                  f"{(sum(1 for r in r30s if r>0)/len(r30s)*100):>8.0f}% {med(mfes):>+8.2f}")


if __name__ == "__main__":
    main()
