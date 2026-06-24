"""
index_event_backtest.py — backtest the index-inclusion filter on the RIGHT data: past index-
reconstitution windows (Russell late-June, S&P/Nasdaq quarterly 3rd-Fri Mar/Jun/Sep/Dec). The bot's
own caches miss these (quarterly bursts outside the backtest windows), so we pull historical Alpaca
news directly for each rebalance window, isolate the index add/join/replace headlines, and measure
the forward STOCK return of the added ticker (the catalyst the bot would trade bullish). Weak/negative
forward returns ⇒ the filter is dropping losers.

Usage:  USE_YAHOO_BARS=1 .venv/bin/python -u index_event_backtest.py
"""
import os, re, json, random
from datetime import datetime, timezone, timedelta
os.environ.setdefault("USE_YAHOO_BARS", "1")
import numpy as np
from dotenv import load_dotenv
load_dotenv()
from alpaca.data.historical import NewsClient
from alpaca.data.requests import NewsRequest, StockBarsRequest
from alpaca.data.timeframe import TimeFrame
import bot

NEWS = NewsClient(os.environ["ALPACA_API_KEY"], os.environ["ALPACA_SECRET_KEY"])
CACHE = "index_event_cache.json"

# Rebalance windows (effective dates → headlines cluster the days around): label, start, end
WINDOWS = [
    ("Russell-2024", "2024-06-22", "2024-06-30"),
    ("Russell-2025", "2025-06-20", "2025-06-30"),
    ("S&P-2025-09",  "2025-09-13", "2025-09-22"),
    ("S&P/Nasdaq-2025-12", "2025-12-12", "2025-12-22"),
    ("S&P-2026-03",  "2026-03-14", "2026-03-23"),
]
IDX = r"(s&amp;p|s&p|dow jones|nasdaq-?100|russell|midcap|smallcap)"
# ADD-side only (the bullish ticker the bot would trade): "X to join/added to/set to join/to replace .. in"
ADD = re.compile(r"(\bto join (?:the )?.{0,25}" + IDX + r"|set to join .{0,20}" + IDX + r"|added to .{0,25}" + IDX +
                 r"|\bjoins?\b.{0,30}(russell|s&amp;p|s&p|nasdaq|dow jones)|to replace .{0,40}? in (?:the )?.{0,25}" + IDX + r")", re.I)


def fetch_index_headlines():
    if os.path.exists(CACHE):
        return json.load(open(CACHE))
    out = []
    for label, s, e in WINDOWS:
        try:
            req = NewsRequest(start=datetime.fromisoformat(s).replace(tzinfo=timezone.utc),
                              end=datetime.fromisoformat(e).replace(tzinfo=timezone.utc), limit=20000)
            arts = NEWS.get_news(req).data["news"]
        except Exception as ex:
            print(f"  {label}: fetch failed {repr(ex)[:60]}"); continue
        hits = [a for a in arts if ADD.search(a.headline or "")]
        for a in hits:
            syms = [s for s in (a.symbols or []) if s.isalpha() and len(s) <= 5]
            out.append({"date": a.created_at.date().isoformat(), "tk": syms[0] if syms else None,
                        "h": a.headline, "win": label})
        print(f"  {label}: {len(arts)} articles → {len(hits)} index-ADD headlines", flush=True)
    json.dump(out, open(CACHE, "w"))
    return out


def fwd_returns(tk, day):
    """r1 / r3 / r5 close-to-close from the headline's trading day."""
    try:
        start = datetime.fromisoformat(day).replace(tzinfo=timezone.utc) - timedelta(days=4)
        end = datetime.fromisoformat(day).replace(tzinfo=timezone.utc) + timedelta(days=14)
        bars = (bot.stock_data_client.get_stock_bars(StockBarsRequest(
            symbol_or_symbols=tk, timeframe=TimeFrame.Day, start=start, end=end, feed="iex")).data or {}).get(tk, [])
        cl = [(b.timestamp.date().isoformat(), float(b.close)) for b in bars]
        i = next((j for j, (d, _) in enumerate(cl) if d >= day), None)
        if i is None:
            return None
        e = cl[i][1]
        out = {"px": e}
        for n, k in ((1, "r1"), (3, "r3"), (5, "r5")):
            out[k] = (cl[i + n][1] / e - 1) * 100 if i + n < len(cl) else None
        return out
    except Exception:
        return None


def boot(xs, n=5000):
    rng = random.Random(0)
    d = [sum(rng.choices(xs, k=len(xs))) / len(xs) for _ in range(n)]
    return float(np.percentile(d, 5)), float(np.percentile(d, 95))


def main():
    print("Fetching index-reconstitution headlines from past rebalance windows…", flush=True)
    rows = fetch_index_headlines()
    rows = [r for r in rows if r["tk"]]
    print(f"\n  total index-ADD headlines with a ticker: {len(rows)}", flush=True)
    print("  computing forward returns of the added stocks…", flush=True)
    for r in rows:
        r["fr"] = fwd_returns(r["tk"], r["date"])
    json.dump(rows, open(CACHE, "w"))
    good = [r for r in rows if r.get("fr") and r["fr"].get("px", 0) >= 5]
    print(f"  with valid forward returns (px≥$5): {len(good)}\n")

    print("  ══ FORWARD RETURN of index-ADD stocks (the catalyst the bot trades bullish) ══")
    for k, lab in (("r1", "next-day"), ("r3", "3-day"), ("r5", "5-day")):
        xs = [r["fr"][k] for r in good if r["fr"].get(k) is not None]
        if not xs:
            continue
        avg = sum(xs) / len(xs); pos = sum(1 for x in xs if x > 0) / len(xs) * 100
        lo, hi = boot(xs)
        sig = "SIG<0 ✓" if hi < 0 else ("SIG>0" if lo > 0 else "~0 (spans)")
        print(f"    {lab:9} n={len(xs):<4} avg {avg:+5.2f}% · {pos:3.0f}% positive · boot mean CI [{lo:+.2f},{hi:+.2f}] {sig}")
    print("\n  worst/best individual moves (3-day):")
    g3 = sorted([r for r in good if r['fr'].get('r3') is not None], key=lambda r: r['fr']['r3'])
    for r in g3[:4] + g3[-3:]:
        print(f"    {r['fr']['r3']:+6.1f}%  {r['tk']:<5} {r['win']:<14} {r['h'][:54]}")
    print("\n  Read: index-ADD catalysts with avg ≤0 and ~50% positive = no edge → calls on them lose to")
    print("  theta. That validates dropping them pre-score (esp. since they over-score to 0.85 → trip bypass).")


if __name__ == "__main__":
    main()
