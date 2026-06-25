"""
already_moved_backtest.py — NOVELTY prototype: does a stock that ALREADY moved up into the entry have
weaker forward returns (the news is already priced → we'd be chasing)? If so, a "skip if pre-moved"
pre-score filter has edge — and it attacks the root cause behind the index-inclusion / analyst-PT /
ETF filters (the model over-scores positive-but-already-priced news).

For each bullish-tradeable signal: pre_move = entry px vs the PRIOR trading day's close (the gap/pop
into our entry) and runup_5d = the 5-day run-up ending the day before the headline (multi-day
anticipation). Compare forward r3 by pre-move bucket, then sweep a "drop if pre_move ≥ X" policy.

Usage:  USE_YAHOO_BARS=1 .venv/bin/python -u research/already_moved_backtest.py
"""
import os, json, random
from datetime import timedelta
os.environ.setdefault("USE_YAHOO_BARS", "1")
import config
os.chdir(config.DATA_DIR)                      # bare cache refs resolve into data/
import numpy as np
import gemini_lotto_pnl as gl
import bot
from alpaca.data.requests import StockBarsRequest
from alpaca.data.timeframe import TimeFrame

CACHE = "already_moved_cache.json"
HORIZON = "r3"


def universe():
    uni = json.load(open("unified_scores.json")); cp = json.load(open("yahoo_candpool_cache.json"))
    gl.SAMPLE = 100000
    out = []
    for c in gl.candidates():
        u = uni.get("unified_v1:" + c["ck"]); fr = cp.get(f"{c['tk']}_{c['ck']}")
        if (isinstance(u, dict) and u.get("sentiment") == "bullish" and float(u.get("magnitude", 0) or 0) >= 0.75
                and fr and fr.get("px", 0) >= 5):
            out.append((c, u, fr))
    return out


def pre_moves(tk, day, entry_px, cache):
    """(pre_move_1d %, runup_5d %): the gap into entry, and the 5-day run-up ending the prior close."""
    key = f"{tk}_{day}"
    if key in cache:
        return cache[key]
    try:
        from datetime import datetime, timezone
        d0 = datetime.fromisoformat(day).replace(tzinfo=timezone.utc)
        bars = (bot.stock_data_client.get_stock_bars(StockBarsRequest(
            symbol_or_symbols=tk, timeframe=TimeFrame.Day, start=d0 - timedelta(days=16),
            end=d0 + timedelta(days=2), feed="iex")).data or {}).get(tk, [])
        cl = [(b.timestamp.date().isoformat(), float(b.close)) for b in bars]
        # last close strictly BEFORE the headline day
        prior = [px for d, px in cl if d < day]
        out = None
        if len(prior) >= 6 and entry_px:
            prior_close = prior[-1]; ref = prior[-6]
            out = {"pre1d": (entry_px / prior_close - 1) * 100, "run5d": (prior_close / ref - 1) * 100}
        cache[key] = out
        return out
    except Exception:
        cache[key] = None
        return None


def boot(xs, n=5000):
    if not xs:
        return (0, 0)
    rng = random.Random(0)
    m = [sum(rng.choices(xs, k=len(xs))) / len(xs) for _ in range(n)]
    return float(np.percentile(m, 5)), float(np.percentile(m, 95))


def cell(rs):
    if not rs:
        return "n=0"
    a = sum(rs) / len(rs); pos = sum(1 for x in rs if x > 0) / len(rs) * 100
    return f"n={len(rs):<4} avg{a:>+6.2f}% {pos:>3.0f}%pos"


def main():
    univ = universe()
    print(f"bullish-tradeable signals: {len(univ)} · computing pre-move into entry…", flush=True)
    cache = json.load(open(CACHE)) if os.path.exists(CACHE) else {}
    rows = []
    for i, (c, u, fr) in enumerate(univ):
        pm = pre_moves(c["tk"], c["dt"].date().isoformat(), fr.get("px"), cache)
        if pm:
            rows.append((pm["pre1d"], pm["run5d"], fr[HORIZON]))
        if i % 100 == 99:
            json.dump(cache, open(CACHE, "w")); print(f"  {i+1}/{len(univ)}", flush=True)
    json.dump(cache, open(CACHE, "w"))
    print(f"  with pre-move data: {len(rows)}\n", flush=True)

    for label, idx, tiers in (
        ("PRE-MOVE into entry (gap vs prior close)", 0, [("≤0% (flat/down)", -1e9, 0), ("0–3%", 0, 3),
                                                          ("3–6%", 3, 6), ("6–10%", 6, 10), (">10% (popped)", 10, 1e9)]),
        ("5-DAY RUN-UP before the news", 1, [("≤0%", -1e9, 0), ("0–5%", 0, 5), ("5–10%", 5, 10), (">10%", 10, 1e9)]),
    ):
        print(f"  ══ forward {HORIZON} by {label} ══")
        for name, lo, hi in tiers:
            rs = [r[2] for r in rows if lo <= r[idx] < hi]
            print(f"    {name:18} {cell(rs)}")
        print()

    print(f"  ── policy: DROP signals already moved ≥ X into entry (forward P&L of the REMOVED set) ──")
    allr = [r[2] for r in rows]
    print(f"    keep ALL (current)        n={len(allr)} Σ{sum(allr):>+7.0f}% avg{sum(allr)/len(allr):>+5.2f}%")
    for X in (3, 5, 8, 10):
        removed = [r[2] for r in rows if r[0] >= X]
        kept = [r[2] for r in rows if r[0] < X]
        if not removed:
            continue
        lo, hi = boot(removed)
        sig = "✓removes losers" if hi < 0 else ("removes WINNERS!" if lo > 0 else "~neutral")
        print(f"    drop pre-move ≥{X}%  removes {len(removed):>3} (Σ{sum(removed):>+6.0f}%, avg{sum(removed)/len(removed):>+5.2f}%, "
              f"CI[{lo:>+4.1f},{hi:>+4.1f}] {sig}) → kept {len(kept)} avg{sum(kept)/len(kept):>+5.2f}%")
    print("\n  If the removed set is net-negative/zero (CI≤0) while kept avg rises, the filter helps:")
    print("  it's dropping already-priced chases. If removed is POSITIVE, we'd be cutting real winners.")


if __name__ == "__main__":
    main()
