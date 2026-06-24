"""
signal_quality.py — does the LLM's score predict realized return? (the metric for
evaluating any prompt/model change, before committing to a full re-score.)

For a sample of bullish events, compares the LLM score (magnitude × confidence)
to the REALIZED N-day stock return after the news (real Yahoo data). Reports:
  • rank correlation (score vs return) — is there ANY predictive signal?
  • decile table — does a higher score monotonically mean a bigger move?
  • top-vs-bottom-decile return spread + hit rates — the practical edge.

This is the baseline to beat. A better prompt/model should raise the correlation
and the top-decile return (ideally on bigger / more-liquid names → cost-survivable).

Usage:  USE_YAHOO_BARS=1 .venv/bin/python signal_quality.py [--fwd 5] [--limit 500]
"""
import argparse
import json
import re
import statistics
from datetime import datetime, timezone, timedelta

import backtest as _bt

_TICKER_RE = re.compile(r"^[A-Z]{1,5}$")
MEGACAPS = {"SPY", "QQQ", "IWM", "GLD", "TLT", "AAPL", "MSFT", "NVDA", "AMD", "TSLA",
            "META", "AMZN", "GOOGL", "GOOG", "AVGO", "NFLX", "INTC", "ORCL", "QCOM", "CRM"}


def load_events():
    cache = json.load(open("dual_score_cache.json"))
    out = []
    for v in cache.values():
        if not isinstance(v, dict):
            continue
        b = v.get("bullish", {})
        if b.get("reasoning") == "SCORE_FAILED":
            continue
        cands = [t for t in (b.get("tickers", []) or []) if _TICKER_RE.match(t) and t not in ("BTC", "ETH")]
        if not cands:
            continue
        ca = (v.get("_article", {}) or {}).get("created_at")
        if not ca:
            continue
        try:
            dt = datetime.fromisoformat(str(ca).replace("Z", "+00:00")).astimezone(timezone.utc)
        except Exception:
            continue
        out.append({"ticker": cands[0], "dt": dt,
                    "mag": float(b.get("magnitude", 0)), "conf": float(b.get("confidence", 0))})
    out.sort(key=lambda e: e["dt"])
    return out


def fwd_return(ticker, dt, fwd_days):
    """Stock return from the news-day close to +fwd_days trading-day close."""
    bars = _bt.get_stock_bars(ticker, dt - timedelta(days=4), dt + timedelta(days=fwd_days * 2 + 8))
    if not bars:
        return None
    on_or_before = [b for b in bars if b["t"].date() <= dt.date()]
    after = [b for b in bars if b["t"].date() > dt.date()]
    if not on_or_before or len(after) < fwd_days:
        return None
    p0 = on_or_before[-1]["c"]
    p1 = after[fwd_days - 1]["c"]
    if p0 <= 0 or p0 > 10000:
        return None
    r = (p1 / p0 - 1) * 100
    return r if abs(r) < 80 else None   # artifact guard


def spearman(xs, ys):
    n = len(xs)
    rx = {v: i for i, v in enumerate(sorted(range(n), key=lambda i: xs[i]))}
    ry = {v: i for i, v in enumerate(sorted(range(n), key=lambda i: ys[i]))}
    d2 = sum((rx[i] - ry[i]) ** 2 for i in range(n))
    return 1 - 6 * d2 / (n * (n * n - 1)) if n > 1 else 0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fwd", type=int, default=5, help="forward trading days")
    ap.add_argument("--limit", type=int, default=500)
    args = ap.parse_args()

    events = load_events()
    print(f"Bullish tagged events: {len(events)}")
    if len(events) > args.limit:
        step = len(events) / args.limit
        events = [events[int(i * step)] for i in range(args.limit)]

    rows = []
    for e in events:
        r = fwd_return(e["ticker"], e["dt"], args.fwd)
        if r is not None:
            rows.append((e["mag"] * e["conf"], e["mag"], e["conf"], r, e["ticker"]))
    print(f"Usable (have {args.fwd}d forward price): {len(rows)}\n")
    if len(rows) < 30:
        print("Too few usable events."); return

    scores = [x[0] for x in rows]; rets = [x[3] for x in rows]
    print(f"=== Score (mag×conf) vs realized {args.fwd}d return ===")
    print(f"  Spearman rank corr: {spearman(scores, rets):+.3f}   (0 = no signal)")
    print(f"  overall mean {args.fwd}d return: {statistics.mean(rets):+.2f}%  "
          f"(bull regime → baseline drift up)")

    # Decile table by score
    order = sorted(rows, key=lambda x: x[0])
    q = len(order) // 5
    print(f"\n  score quintile → mean {args.fwd}d return / win% / n:")
    for i, name in enumerate(["Q1 (lowest)", "Q2", "Q3", "Q4", "Q5 (highest)"]):
        grp = order[i * q:(i + 1) * q] if i < 4 else order[i * q:]
        gr = [g[3] for g in grp]
        win = sum(1 for x in gr if x > 0) / len(gr) * 100 if gr else 0
        print(f"    {name:14} {statistics.mean(gr):+6.2f}%  {win:4.0f}%  ({len(gr)})")

    # liquidity of the winners (cost relevance)
    big = sorted(rows, key=lambda x: x[3], reverse=True)[:max(10, len(rows)//10)]
    liq = sum(1 for x in big if x[4] in MEGACAPS)
    print(f"\n  Top-decile movers: {liq}/{len(big)} are liquid (mega/ETF) "
          f"→ cost-survivable share")


if __name__ == "__main__":
    main()
