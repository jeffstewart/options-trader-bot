"""
bear_backtest.py — test a bearish (long-put) strategy on the news flow.

The scorer was prompted to only flag bullish signals, so the model assigns
tickers to just 5 of 1,298 bearish articles. We therefore keep the model's
bearish DETECTION (sentiment + magnitude + confidence gate — the actual signal)
but resolve the ticker from the news item's own symbols. Caveat: ticker
selection here is news-metadata, not model-driven; a faithful "model picks the
short" test would require re-scoring with a bear-aware prompt.

Long puts are priced with the same validated BS + IV-crush engine
(simulate_option_pnl(..., option_type="put")).

Benchmarks (same dates & sizing, only the underlying changes):
  • bear      — puts on bearish-flagged news names
  • QQQ puts  — puts on QQQ (short-market proxy; should BLEED in a bull regime)
  • random    — puts on random names from the same universe

Usage:
    .venv/bin/python bear_backtest.py [--seeds 3]
"""

import argparse
import random
import re
from datetime import datetime, timedelta, timezone

import backtest as bt
from benchmark import compute_stats, fmt
from config import MIN_MAGNITUDE, MAX_POSITION_USD

_TICKER_RE = re.compile(r"^[A-Z]{1,5}$")


def gather_bearish_trades():
    """Re-fetch the news window, join cached scores, simulate long puts."""
    end_dt   = datetime.now(timezone.utc) - timedelta(days=5)
    start_dt = end_dt - timedelta(days=180)

    print(f"Fetching news {start_dt.date()} → {end_dt.date()} and joining cached scores…")
    bt.load_cache()

    bear_signals = []
    day = start_dt
    days_scanned = 0
    while day <= end_dt:
        for art in bt.fetch_alpaca_news_day(day):
            headline = art.get("headline", "")
            body     = art.get("summary", "")
            # Cache-only: look up by key and skip misses. Never call the scorer
            # (Ollama) — these articles were all scored in the main backtest.
            key = bt.cache_key(headline, body)
            if key not in bt._score_cache:
                continue
            score = bt._score_cache[key]
            if score.get("sentiment") != "bearish":
                continue
            passes, mag, _ = bt.signal_checks(score)
            if not passes:
                continue
            # Resolve ticker from the news item's own symbols
            syms = [s for s in art.get("symbols", []) if _TICKER_RE.match(s)]
            if not syms:
                continue
            bear_signals.append({
                "created_at": art.get("created_at"),
                "ticker": syms[0],
                "magnitude": float(score.get("magnitude", 0)),
                "confidence": float(score.get("confidence", 0)),
            })
        days_scanned += 1
        if days_scanned % 30 == 0:
            print(f"  …{days_scanned}/180 days, {len(bear_signals)} bearish-with-ticker signals")
        day += timedelta(days=1)

    # Dedup: one trade per ticker per day (cooldown proxy, matches bullish side)
    seen = set()
    uniq = []
    for s in bear_signals:
        ca = s["created_at"]
        if isinstance(ca, str):
            ca = datetime.fromisoformat(ca)
        if ca.tzinfo is None:
            ca = ca.replace(tzinfo=timezone.utc)
        k = (s["ticker"], ca.date())
        if k in seen:
            continue
        seen.add(k)
        s["_dt"] = ca
        uniq.append(s)
    print(f"\n{len(uniq)} unique bearish signals with a tradeable ticker.\n")
    return uniq


def sim_arm(signals, ticker_fn, option_type="put"):
    out = []
    for s in signals:
        tk = ticker_fn(s)
        if not tk:
            continue
        pos = bt.scale_position_usd(MAX_POSITION_USD, s["magnitude"], s["confidence"])
        px = bt.get_price_at(tk, s["_dt"])
        if not px:
            continue
        res = bt.simulate_option_pnl(tk, s["_dt"], px, pos,
                                     {"magnitude": s["magnitude"], "confidence": s["confidence"],
                                      "reasoning": ""},
                                     option_type=option_type)
        if res:
            out.append(res)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=3)
    args = ap.parse_args()

    signals = gather_bearish_trades()
    if len(signals) < 10:
        print(f"⚠️  Only {len(signals)} tradeable bearish signals — too few to conclude.")
        return

    universe = sorted({s["ticker"] for s in signals})

    arms = []
    s = compute_stats(sim_arm(signals, lambda x: x["ticker"]))
    s["arm"] = "bear (puts)"; arms.append(s); print(fmt(s))

    s = compute_stats(sim_arm(signals, lambda x: "QQQ"))
    s["arm"] = "QQQ puts"; arms.append(s); print(fmt(s))

    rand_stats = []
    for seed in range(args.seeds):
        rnd = random.Random(seed)
        rand_stats.append(compute_stats(sim_arm(signals, lambda x: rnd.choice(universe))))
    avg = {"arm": f"random puts (×{args.seeds})",
           "trades": round(sum(s["trades"] for s in rand_stats) / len(rand_stats)),
           "total_pnl": sum(s["total_pnl"] for s in rand_stats) / len(rand_stats),
           "win_rate": sum(s["win_rate"] for s in rand_stats) / len(rand_stats),
           "avg_win": 0, "avg_loss": 0,
           "profit_factor": sum(s["profit_factor"] for s in rand_stats) / len(rand_stats),
           "max_dd": sum(s["max_dd"] for s in rand_stats) / len(rand_stats),
           "sharpe": sum(s["sharpe"] for s in rand_stats) / len(rand_stats)}
    arms.append(avg); print(fmt(avg))

    bear, qqq, rnd = arms[0], arms[1], arms[2]
    print(f"\n{'='*70}\nVERDICT")
    print(f"  Bear strategy : ${bear['total_pnl']:,.0f}  Sharpe {bear['sharpe']:.2f}  win {bear['win_rate']:.1f}%")
    print(f"  QQQ puts      : ${qqq['total_pnl']:,.0f}  (short-market proxy)")
    print(f"  Random puts   : ${rnd['total_pnl']:,.0f}")
    if bear["total_pnl"] > 0 and bear["total_pnl"] > rnd["total_pnl"]:
        print("  → Bearish signal has put-side value even in a bull regime "
              "(beats random; positive P&L).")
    elif bear["total_pnl"] > rnd["total_pnl"]:
        print("  → Bear strategy loses money (bull regime) but still beats random "
              "shorting → the signal has relative skill; revisit in a bear/flat regime.")
    else:
        print("  → No demonstrated bearish edge in this regime.")


if __name__ == "__main__":
    main()
