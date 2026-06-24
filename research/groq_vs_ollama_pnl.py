"""
groq_vs_ollama_pnl.py — would PAID GROQ score live signals as well as (or better than) local Ollama?
Decision tool for moving the bot's PRIMARY scorer to Groq in the cloud.

Reads the ticker-instrumented scorer_ab.csv (Ollama vs Groq scored the SAME article), fetches each
article's forward STOCK return via Alpaca (point-in-time, no lookahead), and compares the two scorers
AS PRIMARY: a scorer "trades" an article when it calls bullish & magnitude ≥ MAG. The P&L of each
scorer = the forward returns of the trades IT would open. Also breaks out the disagreements (the
trades unique to each scorer) — that's where one scorer is actually better than the other.

Run it any time as data accrues; it uses only rows old enough to have a settled forward return.
Forward returns are cached so re-runs are instant.

Usage:  USE_YAHOO_BARS=1 .venv/bin/python -u groq_vs_ollama_pnl.py
"""
import os, csv, json, statistics
from datetime import datetime, timezone, timedelta
os.environ.setdefault("USE_YAHOO_BARS", "1")
import numpy as np
import bot
from alpaca.data.requests import StockBarsRequest
from alpaca.data.timeframe import TimeFrame

AB_CSV = "scorer_ab.csv"
FR_CACHE = "scorer_ab_fwd_returns.json"
MAG = 0.75              # live news_call trade gate (bullish & mag ≥ this → "would trade")
HOLD_DAYS = 3          # settle window: only score rows whose forward return has had time to realize
HORIZON = "r3"


def fwd_return(ticker, ts_iso, cache):
    """Forward stock return for `ticker` from the article timestamp: r1 / r3 (close-to-close).
    Entry ref = close of the article's trading day. Cached by (ticker, date)."""
    day = ts_iso[:10]
    key = f"{ticker}_{day}"
    if key in cache:
        return cache[key]
    try:
        start = datetime.fromisoformat(ts_iso) - timedelta(days=4)
        end = datetime.fromisoformat(ts_iso) + timedelta(days=12)
        resp = bot.stock_data_client.get_stock_bars(StockBarsRequest(
            symbol_or_symbols=ticker, timeframe=TimeFrame.Day, start=start, end=end, feed="iex"))
        bars = (resp.data or {}).get(ticker, [])
        closes = [(b.timestamp.date().isoformat(), float(b.close)) for b in bars]
        idx = next((i for i, (d, _) in enumerate(closes) if d >= day), None)
        out = None
        if idx is not None and idx + HOLD_DAYS < len(closes):
            e = closes[idx][1]
            out = {"px": e,
                   "r1": (closes[idx + 1][1] / e - 1) * 100,
                   "r3": (closes[idx + HOLD_DAYS][1] / e - 1) * 100}
        cache[key] = out
        return out
    except Exception:
        cache[key] = None
        return None


def trade(sent, mag):
    try:
        return sent == "bullish" and float(mag or 0) >= MAG
    except Exception:
        return False


def stat(name, xs):
    if not xs:
        print(f"  {name:36} n=0"); return
    n = len(xs); tot = sum(xs); avg = tot / n
    sd = statistics.pstdev(xs) if n > 1 else 0
    sharpe = avg / sd if sd else 0
    win = sum(1 for x in xs if x > 0) / n * 100
    print(f"  {name:36} n={n:<4} Σ{tot:>+8.1f}% avg{avg:>+6.2f}% win{win:>4.0f}% sharpe{sharpe:>+5.2f}")


def main():
    if not os.path.exists(AB_CSV):
        print("no scorer_ab.csv yet — let the bot run to accrue ticker-instrumented rows."); return
    rows = list(csv.DictReader(open(AB_CSV)))
    if rows and "ollama_tickers" not in rows[0]:
        print("scorer_ab.csv is the OLD schema (no tickers). New rows accrue going forward."); return
    cache = json.load(open(FR_CACHE)) if os.path.exists(FR_CACHE) else {}
    cutoff = (datetime.now(timezone.utc) - timedelta(days=HOLD_DAYS + 1)).isoformat()

    usable, both_scored = [], 0     # (row, ticker, return) for groq-ok, settled, ticker+return rows
    for r in rows:
        if r["groq_status"] != "ok" or not r["ollama_sentiment"] or not r["groq_sentiment"]:
            continue
        both_scored += 1
        if r["timestamp"] > cutoff:
            continue                                  # not settled yet
        tk = (r.get("ollama_tickers") or r.get("groq_tickers") or "").split(",")[0].strip()
        if not tk:
            continue
        fr = fwd_return(tk, r["timestamp"], cache)
        if fr:
            usable.append((r, tk, fr[HORIZON]))
    json.dump(cache, open(FR_CACHE, "w"))

    models = sorted(set(r.get("groq_model", "") for r, _, _ in usable))
    print(f"A/B rows: {len(rows)} · groq-ok: {both_scored} · settled+ticker+return: {len(usable)} "
          f"· groq models: {', '.join(m.split('/')[-1] for m in models) or '(none)'}")
    if len(usable) < 20:
        print(f"\n  Only {len(usable)} usable rows — need more data. Re-run after the bot accrues a few "
              f"sessions (each row needs {HOLD_DAYS}+ trading days to settle).")
        return

    # OLLAMA-primary baseline: one decision per ARTICLE (dedupe — Ollama is logged once per groq model)
    seen, ollama_rows = set(), []
    for r, tk, ret in usable:
        k = (r["timestamp"], tk)
        if k not in seen:
            seen.add(k); ollama_rows.append((r, ret))
    o_pnl = [ret for r, ret in ollama_rows if trade(r["ollama_sentiment"], r["ollama_mag"])]

    print(f"\n  ══ AS PRIMARY SCORER — forward {HORIZON} P&L of the trades each would open ══")
    stat("OLLAMA-primary (baseline)", o_pnl)
    for model in models:
        mrows = [(r, ret) for r, tk, ret in usable if r.get("groq_model") == model]
        g_pnl = [ret for r, ret in mrows if trade(r["groq_sentiment"], r["groq_mag"])]
        o_only = [ret for r, ret in mrows if trade(r["ollama_sentiment"], r["ollama_mag"]) and not trade(r["groq_sentiment"], r["groq_mag"])]
        g_only = [ret for r, ret in mrows if trade(r["groq_sentiment"], r["groq_mag"]) and not trade(r["ollama_sentiment"], r["ollama_mag"])]
        print()
        stat(f"GROQ-primary [{model.split('/')[-1]}]", g_pnl)
        stat(f"  └ {model.split('/')[-1]}-only (Ollama skips)", g_only)
        stat(f"  └ Ollama-only ({model.split('/')[-1]} skips)", o_only)

    print(f"\n  Read: compare each GROQ-primary line to OLLAMA-primary for the headline (who picks more")
    print(f"  profitable trades). The -only lines isolate where they differ: Groq-only POSITIVE = catches")
    print(f"  winners Ollama misses; Ollama-only NEGATIVE = Groq's caution avoids losers. Agreement +")
    print(f"  latency come from the A/B log directly; this adds the P&L the old schema couldn't.")


if __name__ == "__main__":
    main()
