"""
news_source_attribution.py — was trade quality different by news source (Alpaca/Benzinga vs
SEC EDGAR 8-K vs NewsAPI)? Answered 2026-07-18, with an important coverage caveat.

trades.csv/closed_trades.csv never recorded which feed produced a signal (core/bot.py's `source`
param was log-only). This reconstructs it retroactively by joining each trade's stored `reasoning`
text to the "📰 [source] ... -> 🤖 ... conf=/mag=... -> <reasoning line>" block in logs/bot.log --
scoring is a synchronous (blocking) call inside process_signal, so those three log lines are always
atomic/adjacent for a given article, making the join safe without needing article-level IDs.

CAVEAT (don't skip this when rerunning): logs/bot.log auto-rotates to the last ~50k lines
(manage.sh:35), so on 2026-07-18 it only covered ~July 6-18 of the full June 9-July 16 trade
history. Of 191 total trades, only 43 fell in the surviving window and got a source tag; of
those, only 27 had a matched close -- and EVERY one of those 27 was Alpaca/Benzinga. Zero
SEC-EDGAR- or NewsAPI-sourced signals converted into a completed trade in the surviving window,
so no real P&L comparison across sources was possible from history alone.

What WAS learned (signal-level stats, same ~10-day window, all three sources present):
  source            n_scored  bullish%  avg_conf/mag (bullish)  no_ticker%
  Alpaca/Benzinga    1481      48.8%     0.82 / 0.64             58.6%
  NewsAPI             230      23.5%     0.82 / 0.53             87.4%
  SEC EDGAR 8-K       312      ~0%       --                      99.7%

The real finding: the SEC EDGAR feed is essentially non-functional as wired -- it ingests only the
RSS filer-index title (e.g. "8-K - AppTech Payments Corp. (0001070050) (Filer)"), boilerplate with
no actual disclosure content, so almost nothing resolves to a ticker or scores meaningfully. Fixing
it would mean fetching the actual 8-K item text/exhibit, not just the index entry. NewsAPI skews
toward generic macro/political headlines (87% no-ticker) -- lower yield than Alpaca/Benzinga, but
not necessarily lower quality on the rare hit.

Fix applied the same day: core/bot.py and v2/{bot,execution}.py now stamp signal["_source"] and
persist it as a "news_source" column in trades.csv, so this reconstruction won't be needed again --
future analysis can just group trades.csv by that column directly. v1's trades.csv header was
updated in place (data/trades.csv, gitignored) to add the column name; historical rows are simply
blank for it. The code change takes effect on v1's NEXT restart, not retroactively.

Usage: .venv/bin/python -u news_source_attribution.py   (run from data/, needs logs/bot.log + trades.csv/closed_trades.csv)
"""
import csv
import re
from collections import Counter, defaultdict
from datetime import datetime

BOT_LOG = "../logs/bot.log"
TRADES_CSV = "trades.csv"
CLOSED_TRADES_CSV = "closed_trades.csv"

SOURCE_RE = re.compile(r"📰 \[([^\]]+)\]")
SIG_RE = re.compile(r"conf=([\d.]+)\s+mag=([\d.]+)")
FULL_SIG_RE = re.compile(r"(\w[\w.]*):\s+(bullish|bearish|neutral)\s+conf=([\d.]+)\s+mag=([\d.]+)\s+tickers=(\[.*\])")
PREFIX_RE = re.compile(r"^\d{2}:\d{2}:\d{2}\s+\S+\s+")


def build_reasoning_to_source(path=BOT_LOG):
    """The 📰 line sets 'current source'; the very next 'conf=/mag=' line is the score; the line
    right after THAT is always the reasoning text (score_with_ollama logs it unconditionally,
    right after the conf/mag line, with no await in between -- so no other article's block can
    interleave here even though multiple pollers run concurrently)."""
    reasoning_to_source = {}
    current_source = None
    pending = False
    with open(path, encoding="utf-8", errors="replace") as f:
        for line in f:
            body = PREFIX_RE.sub("", line, count=1)
            m = SOURCE_RE.search(body)
            if m:
                current_source = m.group(1)
                pending = False
                continue
            if SIG_RE.search(body):
                pending = True
                continue
            if pending:
                text = body.strip()
                if text and current_source:
                    reasoning_to_source[text] = current_source
                pending = False
    return reasoning_to_source


def source_signal_stats(path=BOT_LOG):
    """Pre-trade stats: volume/bullish-rate/avg-conf-mag/no-ticker-rate per source, over
    whatever window logs/bot.log currently covers (all three sources are present in that
    window, unlike the trade-level join below)."""
    by_source = defaultdict(lambda: {"n": 0, "bullish": 0, "sum_conf": 0.0, "sum_mag": 0.0, "no_ticker": 0})
    current_source = None
    with open(path, encoding="utf-8", errors="replace") as f:
        for line in f:
            body = PREFIX_RE.sub("", line, count=1)
            m = SOURCE_RE.search(body)
            if m:
                current_source = m.group(1)
                continue
            sm = FULL_SIG_RE.search(body)
            if sm and current_source:
                _model, sentiment, conf, mag, tickers = sm.groups()
                d = by_source[current_source]
                d["n"] += 1
                if sentiment == "bullish":
                    d["bullish"] += 1
                    d["sum_conf"] += float(conf)
                    d["sum_mag"] += float(mag)
                if tickers.strip() == "[]":
                    d["no_ticker"] += 1
    return by_source


def join_trades_to_source(reasoning_to_source, trades_csv=TRADES_CSV, closed_csv=CLOSED_TRADES_CSV):
    with open(trades_csv) as f:
        opens = list(csv.DictReader(f))
    with open(closed_csv) as f:
        closes = list(csv.DictReader(f))
    for c in closes:
        c["_ts"] = datetime.fromisoformat(c["timestamp"])
    closes.sort(key=lambda c: c["_ts"])
    for o in opens:
        o["_ts"] = datetime.fromisoformat(o["timestamp"])
    opens.sort(key=lambda o: o["_ts"])

    used = set()
    matched = []
    for o in opens:
        src = reasoning_to_source.get(o["reasoning"].strip())
        if not src:
            continue
        ticker, strat = o["source_ticker"], o["strategy"]
        best = None
        for i, c in enumerate(closes):
            if i in used or c["underlying"] != ticker or c["strategy"] != strat or c["_ts"] < o["_ts"]:
                continue
            best = i
            break
        if best is not None:
            used.add(best)
            c = closes[best]
            matched.append({"source": src, "ticker": ticker, "strategy": strat,
                             "pnl_usd": float(c["pnl_usd"]), "reason": c["reason"]})
    return matched


def main():
    r2s = build_reasoning_to_source()
    print(f"reasoning->source entries recovered from {BOT_LOG}: {len(r2s)}")
    print(Counter(r2s.values()))

    print("\n-- signal-level stats (pre-trade, same log window) --")
    by_source = source_signal_stats()
    print(f"{'source':<10} {'n_scored':>8} {'bullish%':>9} {'avg_conf':>9} {'avg_mag':>8} {'no_ticker%':>10}")
    for src, d in sorted(by_source.items(), key=lambda x: -x[1]["n"]):
        bull = d["bullish"]
        avgc = d["sum_conf"] / bull if bull else 0
        avgm = d["sum_mag"] / bull if bull else 0
        print(f"{src:<10} {d['n']:>8} {bull/d['n']*100:>8.1f}% {avgc:>9.3f} {avgm:>8.3f} {d['no_ticker']/d['n']*100:>9.1f}%")

    print("\n-- trade-level P&L by source (only trades whose reasoning survived log rotation) --")
    matched = join_trades_to_source(r2s)
    print(f"{len(matched)} closed trades matched to a source")
    by_src = defaultdict(list)
    for t in matched:
        by_src[t["source"]].append(t)
    print(f"{'source':<12} {'n':>4} {'win%':>6} {'total$':>9} {'avg$':>8}")
    for src, trades in sorted(by_src.items(), key=lambda x: -len(x[1])):
        n = len(trades)
        wins = sum(1 for t in trades if t["pnl_usd"] > 0)
        tot = sum(t["pnl_usd"] for t in trades)
        print(f"{src:<12} {n:>4} {wins/n*100:>5.1f}% ${tot:>+8,.0f} ${tot/n:>+7,.0f}")


if __name__ == "__main__":
    main()
