"""
trade_review_202608.py — full trade-level review across BOTH bots (v1 core/, v2), all
strategies, looking for systematic patterns worth turning into a live filter: what
characterizes losers we could block, and what characterizes winners we could size up.

jeff's ask (2026-08-08): review all trades made by both bots and see if anything stands out
that we could systematically act on.

Joins each bot's trades.csv (open: confidence/magnitude/reasoning/spread/source) to its
closed_trades.csv (close: pnl/exit reason) via a FIFO pair per (strategy, symbol) -- option
strategies (news_call, lotto) key on option_symbol; stock-shaped strategies (stock, pead,
pairs_long, pairs_short, qqq_macro) key on source_ticker, since log_trade_stock writes
option_symbol="-" and log_closed_trade's `symbol` arg is the ticker itself for those. FIFO
(not a naive symbol join) because a symbol CAN be re-entered same-day (e.g. the 2026-08-03
BA double-entry) -- naive join would double-count or mismatch pairs.

Every breakdown below prints n so a small-sample "pattern" doesn't get mistaken for a real
one -- see memory project_surprise_signal_ruled_out and project_overnight_8k_backtest for two
prior mirages that only showed up because nobody looked at n.

Usage (run from repo root):
    .venv/bin/python research/trade_review_202608.py
    .venv/bin/python research/trade_review_202608.py --since 2026-07-16
"""
import argparse
import csv
import re
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

STOCK_SHAPED = {"stock", "pead", "pairs_long", "pairs_short", "qqq_macro"}


def _read_csv(path: Path) -> list[dict]:
    if not path.exists():
        return []
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


def load_bot(name: str, data_dir: Path) -> list[dict]:
    """Pair opens to closes FIFO per (strategy, symbol-key). Returns one dict per closed trade
    merging both sides' fields, or None fields for the open-side if no matching close exists yet
    (still open -- excluded from P&L stats but left in for completeness/inspection)."""
    opens = _read_csv(data_dir / "trades.csv")
    closes = _read_csv(data_dir / "closed_trades.csv")

    def open_key(r):
        strat = r["strategy"]
        sym = r["source_ticker"] if strat in STOCK_SHAPED else r["option_symbol"]
        return (strat, sym)

    def close_key(r):
        # For stock-shaped strategies, closed_trades.csv's `symbol` is a COMPOSITE key like
        # "STAK__stock" (v1's _monitored_positions dict key, so the same ticker can be tracked
        # under multiple strategies at once without colliding) -- the bare ticker is in
        # `underlying` instead. Matching on `symbol` here silently orphaned every stock/pead/pairs
        # close from its open (confirmed: closed_trades.csv had 64 "stock" rows whose `symbol`
        # never once matched an opened `source_ticker`, while total open/close COUNTS matched --
        # a silent 100% mismatch hiding behind a correct total).
        strat = r["strategy"]
        sym = r["underlying"] if strat in STOCK_SHAPED else r["symbol"]
        return (strat, sym)

    queues = defaultdict(list)
    for r in opens:
        queues[open_key(r)].append(r)

    trades = []
    for c in closes:
        k = close_key(c)
        o = queues[k].pop(0) if queues.get(k) else {}
        try:
            pnl_usd = float(c["pnl_usd"])
            pnl_pct = float(c["pnl_pct"])
        except (KeyError, ValueError):
            continue
        try:
            conf = float(o.get("confidence") or "nan")
            mag = float(o.get("magnitude") or "nan")
        except ValueError:
            conf = mag = float("nan")
        trades.append({
            "bot": name,
            "strategy": c["strategy"],
            "symbol": c["symbol"],
            "underlying": c.get("underlying", c["symbol"]),
            "ts_open": o.get("timestamp", ""),
            "ts_close": c["timestamp"],
            "confidence": conf,
            "magnitude": mag,
            "spread_pct": o.get("spread_pct", ""),
            "news_source": o.get("news_source", ""),
            "scorer": o.get("scorer", ""),
            "reasoning": (o.get("reasoning") or "").lower(),
            "pnl_usd": pnl_usd,
            "pnl_pct": pnl_pct,
            "exit_reason": c.get("reason", ""),
        })
    return trades


def fmt_row(label, n, total, avg, win_rate):
    return f"  {label:<28} n={n:<4} total=${total:>9,.0f}  avg=${avg:>+7.1f}  win%={win_rate:>5.1f}%"


def summarize(trades: list[dict], label_key, min_n=1, sort_by="total"):
    buckets = defaultdict(list)
    for t in trades:
        buckets[t[label_key]].append(t)
    rows = []
    for label, ts in buckets.items():
        n = len(ts)
        if n < min_n:
            continue
        total = sum(t["pnl_usd"] for t in ts)
        avg = total / n
        win = 100 * sum(1 for t in ts if t["pnl_usd"] > 0) / n
        rows.append((label, n, total, avg, win))
    key = {"total": lambda r: r[2], "avg": lambda r: r[3], "n": lambda r: r[1]}[sort_by]
    rows.sort(key=key)
    for label, n, total, avg, win in rows:
        print(fmt_row(str(label)[:28], n, total, avg, win))
    return rows


def bucket_conf_mag(trades, field, width=0.1):
    def b(t):
        v = t[field]
        if v != v:  # nan
            return "unknown"
        lo = int(v / width) * width
        return f"{lo:.2f}-{lo + width:.2f}"
    return [{**t, f"{field}_bucket": b(t)} for t in trades]


KEYWORDS = [
    "earnings", "beat", "miss", "guidance", "price target", "analyst", "upgrade", "downgrade",
    "acquisition", "acquire", "merger", "buyout", "fda", "approval", "patent", "lawsuit",
    "recall", "partnership", "contract", "buyback", "ipo", "ceo", "resign", "layoff",
    "bankruptcy", "investigation", "recall", "delisted", "split", "dividend", "short",
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--since", default=None,
                     help="YYYY-MM-DD -- only include trades OPENED on/after this date "
                          "(filters on ts_open, i.e. when the bot decided to trade under "
                          "whatever config was live then, not when the position happened to "
                          "close). Trades with no matched open (ts_open missing) are dropped "
                          "when --since is set, since we can't confirm which side of the cutoff "
                          "they fall on.")
    args = ap.parse_args()

    v1 = load_bot("v1", ROOT / "data")
    v2 = load_bot("v2", ROOT / "v2" / "data")
    all_trades = v1 + v2

    if args.since:
        before = len(all_trades)
        all_trades = [t for t in all_trades if t["ts_open"] and t["ts_open"][:10] >= args.since]
        print(f"--since {args.since}: kept {len(all_trades)} of {before} closed trades "
              f"(dropped {before - len(all_trades)} opened earlier or with no matched open)\n")
        v1 = [t for t in v1 if t["ts_open"] and t["ts_open"][:10] >= args.since]
        v2 = [t for t in v2 if t["ts_open"] and t["ts_open"][:10] >= args.since]

    print(f"loaded {len(v1)} closed v1 trades, {len(v2)} closed v2 trades, {len(all_trades)} total\n")

    print("=" * 100)
    print("1. BY STRATEGY (both bots combined) -- sorted worst total$ first")
    print("=" * 100)
    summarize(all_trades, "strategy", sort_by="total")

    print("\n" + "=" * 100)
    print("2. BY STRATEGY x BOT")
    print("=" * 100)
    tagged = [{**t, "strategy_bot": f"{t['strategy']} [{t['bot']}]"} for t in all_trades]
    summarize(tagged, "strategy_bot", sort_by="total")

    print("\n" + "=" * 100)
    print("3. BY EXIT REASON (within each strategy)")
    print("=" * 100)
    for strat in sorted({t["strategy"] for t in all_trades}):
        sub = [t for t in all_trades if t["strategy"] == strat]
        if len(sub) < 3:
            continue
        print(f"\n {strat} (n={len(sub)}):")
        summarize(sub, "exit_reason", sort_by="total")

    print("\n" + "=" * 100)
    print("4. BY NEWS SOURCE")
    print("=" * 100)
    summarize(all_trades, "news_source", min_n=3, sort_by="total")

    print("\n" + "=" * 100)
    print("5. BY CONFIDENCE BUCKET (option/lotto/news_call strategies only, min n=3)")
    print("=" * 100)
    scored = [t for t in all_trades if t["strategy"] in ("lotto", "news_call") and t["confidence"] == t["confidence"]]
    scored_c = bucket_conf_mag(scored, "confidence")
    summarize(scored_c, "confidence_bucket", min_n=3, sort_by="n")

    print("\n" + "=" * 100)
    print("6. BY MAGNITUDE BUCKET (option/lotto/news_call strategies only, min n=3)")
    print("=" * 100)
    scored_m = bucket_conf_mag(scored, "magnitude")
    summarize(scored_m, "magnitude_bucket", min_n=3, sort_by="n")

    print("\n" + "=" * 100)
    print("7. BY TICKER (repeat offenders/winners, min n=2)")
    print("=" * 100)
    summarize(all_trades, "underlying", min_n=2, sort_by="total")

    print("\n" + "=" * 100)
    print("8. BY SPREAD_PCT BUCKET AT ENTRY (option strategies, min n=3)")
    print("=" * 100)
    def spread_bucket(t):
        s = t["spread_pct"]
        if not s or s == "0.0%":
            return "n/a (stock)"
        try:
            v = float(s.strip("%"))
        except ValueError:
            return "n/a"
        if v < 10: return "0-10%"
        if v < 20: return "10-20%"
        if v < 30: return "20-30%"
        if v < 50: return "30-50%"
        return "50%+"
    spread_tagged = [{**t, "spread_bucket": spread_bucket(t)} for t in all_trades]
    summarize(spread_tagged, "spread_bucket", min_n=3, sort_by="n")

    print("\n" + "=" * 100)
    print("9. KEYWORD SCAN ON REASONING TEXT (min n=4 mentions)")
    print("=" * 100)
    kw_trades = []
    for t in all_trades:
        for kw in KEYWORDS:
            if kw in t["reasoning"]:
                kw_trades.append({**t, "keyword": kw})
    summarize(kw_trades, "keyword", min_n=4, sort_by="avg")

    print("\n" + "=" * 100)
    print("10. BY DAY OF WEEK / HOUR (entry timestamp, UTC)")
    print("=" * 100)
    from datetime import datetime
    dow_tagged = []
    for t in all_trades:
        if not t["ts_open"]:
            continue
        try:
            dt = datetime.fromisoformat(t["ts_open"].replace("Z", "+00:00"))
        except ValueError:
            continue
        dow_tagged.append({**t, "dow": dt.strftime("%a"), "hour_utc": dt.hour})
    print(" -- by weekday --")
    summarize(dow_tagged, "dow", min_n=3, sort_by="n")
    print(" -- by hour (UTC) --")
    summarize(dow_tagged, "hour_utc", min_n=3, sort_by="n")

    print("\n" + "=" * 100)
    print("11. BIGGEST SINGLE LOSERS AND WINNERS (top 10 each)")
    print("=" * 100)
    srt = sorted(all_trades, key=lambda t: t["pnl_usd"])
    print(" worst 10:")
    for t in srt[:10]:
        print(f"   {t['ts_close'][:16]} {t['bot']:<3} {t['strategy']:<12} {t['symbol']:<22} "
              f"${t['pnl_usd']:>+8.0f} ({t['pnl_pct']:>+6.1f}%) reason={t['exit_reason']:<16} "
              f"conf={t['confidence']:.2f} mag={t['magnitude']:.2f} src={t['news_source']}")
    print(" best 10:")
    for t in srt[-10:][::-1]:
        print(f"   {t['ts_close'][:16]} {t['bot']:<3} {t['strategy']:<12} {t['symbol']:<22} "
              f"${t['pnl_usd']:>+8.0f} ({t['pnl_pct']:>+6.1f}%) reason={t['exit_reason']:<16} "
              f"conf={t['confidence']:.2f} mag={t['magnitude']:.2f} src={t['news_source']}")


if __name__ == "__main__":
    main()
