"""
signal_lab.py — the YARDSTICK for prompt/model experiments (Stage-1 stat screen).

The agreed methodology: fix the measurement before iterating prompts, or every
"improvement" is just overfitting. This module supplies the three things
signal_quality.py lacked:

  1. MARKET-EXCESS labels — forward return MINUS SPY's forward return over the same
     window (beta≈1 residual). Strips the melt-up beta so we measure SELECTION skill,
     not "rode the index." This is the label that defines a "real winner."
  2. FROZEN dev/test splits — deterministic by hash(ticker|date), so a prompt is
     always judged on the same articles and the test set stays untouched while we
     iterate on dev.
  3. BOOTSTRAP confidence intervals + a LIFT metric — so we only "promote" a prompt
     when it beats baseline by more than the noise band (the 8B-+0.166 lesson).

Stage-1 metrics (cheap screen) per split:
  • rank-IC : Spearman(score, excess_return)  [95% bootstrap CI]
  • lift    : P(winner | top-score quintile) / base-rate   where winner =
              excess_return ≥ WIN_THRESH  [95% bootstrap CI]   (the "pick winners")
  • noise   : P(winner | bottom quintile)   (should be ≤ base-rate → "exclude noise")
  • liquidity share of the top movers (cost-survivability)

A prompt/model's scores are read from ANY cache file via --cache/--score-field, so
the same yardstick grades the baseline and every experiment identically. Stage-3
(net-of-cost P&L gate) reuses the existing sims — run after a prompt clears Stage 1.

Usage:
  USE_YAHOO_BARS=1 .venv/bin/python signal_lab.py                 # baseline (dual cache)
  USE_YAHOO_BARS=1 .venv/bin/python signal_lab.py --split test    # confirm on held-out
  USE_YAHOO_BARS=1 .venv/bin/python signal_lab.py --fwd 5 --win-thresh 5 --limit 800
"""
import argparse, hashlib, json, random, statistics
from datetime import datetime, timezone, timedelta

import backtest as _bt
from signal_quality import spearman, MEGACAPS, _TICKER_RE

TEST_BUCKETS = {0, 1}   # 20% held out as TEST; the rest is DEV


def split_of(ticker, dt):
    h = int(hashlib.md5(f"{ticker}|{dt.date()}".encode()).hexdigest(), 16) % 10
    return "test" if h in TEST_BUCKETS else "dev"


def load_scored(cache_file, score_field):
    """Load (ticker, dt, score) from a cache. score_field:
       'magcONF' = dual-cache bullish magnitude×confidence (baseline);
       otherwise read that float field from each entry (e.g. a prompt_lab cache)."""
    raw = json.load(open(cache_file))
    out = []
    for v in raw.values():
        if not isinstance(v, dict):
            continue
        if score_field == "magconf":
            b = v.get("bullish", {})
            if b.get("reasoning") == "SCORE_FAILED":
                continue
            tks = [t for t in (b.get("tickers", []) or []) if _TICKER_RE.match(t) and t not in ("BTC", "ETH")]
            if not tks:
                continue
            score = float(b.get("magnitude", 0)) * float(b.get("confidence", 0))
            tk, art = tks[0], v.get("_article", {})
        else:
            tks = v.get("tickers") or ([v.get("ticker")] if v.get("ticker") else [])
            tks = [t for t in tks if t and _TICKER_RE.match(t) and t not in ("BTC", "ETH")]
            if not tks:
                continue
            try:
                score = float(v.get(score_field, 0))
            except Exception:
                continue
            tk, art = tks[0], v.get("_article", {})
        ca = art.get("created_at")
        if not ca:
            continue
        try:
            dt = datetime.fromisoformat(str(ca).replace("Z", "+00:00")).astimezone(timezone.utc)
        except Exception:
            continue
        out.append({"ticker": tk, "dt": dt, "score": score})
    out.sort(key=lambda e: e["dt"])
    return out


_spy_cache = {}
def spy_fwd(dt, fwd):
    """SPY forward return over the same window (cached)."""
    key = dt.date()
    if key in _spy_cache:
        return _spy_cache[key]
    r = _stock_fwd("SPY", dt, fwd)
    _spy_cache[key] = r
    return r


def _stock_fwd(ticker, dt, fwd):
    bars = _bt.get_stock_bars(ticker, dt - timedelta(days=4), dt + timedelta(days=fwd * 2 + 8))
    if not bars:
        return None
    before = [b for b in bars if b["t"].date() <= dt.date()]
    after  = [b for b in bars if b["t"].date() > dt.date()]
    if not before or len(after) < fwd:
        return None
    p0, p1 = before[-1]["c"], after[fwd - 1]["c"]
    if p0 <= 0 or p0 > 10000:
        return None
    r = (p1 / p0 - 1) * 100
    return r if abs(r) < 80 else None


def excess_return(ticker, dt, fwd):
    rs = _stock_fwd(ticker, dt, fwd)
    rm = spy_fwd(dt, fwd)
    if rs is None or rm is None:
        return None
    return rs - rm   # market-excess (beta≈1 residual)


def boot_ci(fn, data, n=1000, seed=1):
    rng = random.Random(seed)
    vals = []
    for _ in range(n):
        sample = [data[rng.randrange(len(data))] for _ in range(len(data))]
        v = fn(sample)
        if v is not None:
            vals.append(v)
    vals.sort()
    return (vals[int(0.025 * len(vals))], vals[int(0.975 * len(vals))]) if vals else (float("nan"), float("nan"))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", default="dual_score_cache.json")
    ap.add_argument("--score-field", default="magconf",
                    help="'magconf' for dual cache, else a float field name in the cache entries")
    ap.add_argument("--split", choices=["dev", "test", "all"], default="dev")
    ap.add_argument("--fwd", type=int, default=5)
    ap.add_argument("--win-thresh", type=float, default=5.0, help="excess %% that counts as a 'winner'")
    ap.add_argument("--limit", type=int, default=800)
    args = ap.parse_args()

    events = load_scored(args.cache, args.score_field)
    events = [e for e in events if args.split == "all" or split_of(e["ticker"], e["dt"]) == args.split]
    if len(events) > args.limit:
        step = len(events) / args.limit
        events = [events[int(i * step)] for i in range(args.limit)]

    rows = []
    for e in events:
        ex = excess_return(e["ticker"], e["dt"], args.fwd)
        if ex is not None:
            rows.append({"score": e["score"], "ex": ex, "ticker": e["ticker"]})
    print(f"=== signal_lab  cache={args.cache}  field={args.score_field}  "
          f"split={args.split}  fwd={args.fwd}d ===")
    print(f"  events={len(events)}  usable(excess return)={len(rows)}  win-thresh=+{args.win_thresh}% excess\n")
    if len(rows) < 50:
        print("  too few usable events for a stable read."); return

    scores = [r["score"] for r in rows]
    exs    = [r["ex"] for r in rows]
    base   = sum(1 for r in rows if r["ex"] >= args.win_thresh) / len(rows)

    # rank-IC vs market-excess
    ic = spearman(scores, exs)
    ic_ci = boot_ci(lambda s: spearman([x["score"] for x in s], [x["ex"] for x in s]), rows)
    print(f"  rank-IC (score vs EXCESS return): {ic:+.3f}   95% CI [{ic_ci[0]:+.3f}, {ic_ci[1]:+.3f}]")
    print(f"    → CI excludes 0? {'YES (real signal)' if ic_ci[0] > 0 or ic_ci[1] < 0 else 'no (indistinguishable from noise)'}")

    # quintiles by score
    order = sorted(rows, key=lambda r: r["score"])
    q = len(order) // 5
    print(f"\n  score quintile → mean EXCESS {args.fwd}d return / winner% / n:")
    for i, nm in enumerate(["Q1 (low)", "Q2", "Q3", "Q4", "Q5 (high)"]):
        grp = order[i*q:(i+1)*q] if i < 4 else order[i*q:]
        m = statistics.mean(x["ex"] for x in grp)
        w = sum(1 for x in grp if x["ex"] >= args.win_thresh) / len(grp) * 100
        print(f"    {nm:10} {m:+6.2f}%   {w:4.0f}%   ({len(grp)})")

    # lift: top-quintile winner-rate vs base; noise: bottom-quintile winner-rate
    topq = order[4*q:]; botq = order[:q]
    def winrate(s): return sum(1 for x in s if x["ex"] >= args.win_thresh) / len(s) if s else 0
    lift = winrate(topq) / base if base else float("nan")
    lift_ci = boot_ci(lambda s: (winrate(s) / base) if base else None, topq)
    print(f"\n  base winner-rate: {base*100:.1f}%")
    print(f"  LIFT (top-quintile winner-rate / base): {lift:.2f}×   95% CI [{lift_ci[0]:.2f}, {lift_ci[1]:.2f}]")
    print(f"  noise (bottom-quintile winner-rate): {winrate(botq)*100:.1f}%  (want ≤ base → excludes noise)")

    big = sorted(rows, key=lambda r: r["ex"], reverse=True)[:max(10, len(rows)//10)]
    liq = sum(1 for x in big if x["ticker"] in MEGACAPS)
    print(f"\n  top excess movers liquid (mega/ETF): {liq}/{len(big)} → cost-survivable share")
    print("\n  PROMOTE a prompt only if rank-IC CI clears baseline's AND lift CI > 1 on DEV,")
    print("  then confirm on --split test + the 2022 set + the Stage-3 net-of-cost P&L gate.")


if __name__ == "__main__":
    main()
