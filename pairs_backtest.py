"""
pairs_backtest.py — Market-neutral long/short pairs backtest.

The core alpha question: does the LLM's name-selection skill produce regime-independent
returns, or is everything just market beta?

Method:
  Each trading day, collect all bullish signals (passing the gate) and all bearish
  signals (passing the gate) from dual_score_cache.json. If we have at least one of
  each, go long the top bullish name and short the top bearish name in equal $ amounts.
  The market beta largely cancels; residual return = pure selection skill.

Two variants:
  1. Any-pair: top bull vs top bear, regardless of sector (broad test)
  2. Same-day: only pair signals that appeared within the same calendar day

Cross-regime: bull-meltup + 2022-bear (Yahoo).

Benchmark: compare net L/S Sharpe vs:
  - SPY buy-and-hold (market beta)
  - Long-only stock strategy (our existing baseline)
  - Random L/S pairs (random bull × random bear same day)

Usage:  USE_YAHOO_BARS=1 .venv/bin/python pairs_backtest.py
"""
import os, random, statistics
from collections import defaultdict
from datetime import datetime, timezone, timedelta
from pathlib import Path

os.environ.setdefault("USE_YAHOO_BARS", "1")

import tune_v2
from pead_backtest import simulate_pead   # reuse long sim
from pead_bear import sim_short           # reuse short sim
from backtest import is_valid_stock_ticker, get_price_at
from benchmark import compute_stats
from regime_filter import build_regime
import config as _cfg

WINDOWS = [
    ("bull-meltup", datetime.now(timezone.utc) - timedelta(days=5), 180,
     "dual_score_cache.json",  "dual_score_cache.json"),
    ("2022-bear",   datetime(2022, 6, 30, tzinfo=timezone.utc), 90,
     "bear_dual_cache.json",   "bear_dual_cache.json"),
]

LONG_TRAIL  = 0.20   # PEAD-width trail for long leg
SHORT_TRAIL = 0.15   # tuned bear-short trail
MAX_HOLD    = 45
RANDOM_SEED = 42


def _scale(mag, conf):
    return max(_cfg.MAX_POSITION_USD * 0.10, _cfg.MAX_POSITION_USD * mag * conf)


def _passes(row, min_mag=0.35, min_conf=None):
    mag  = row["magnitude"]
    conf = row["confidence"]
    if mag < min_mag:
        return False
    floor = (min_conf or _cfg.BASE_CONFIDENCE) + (1 - mag) * _cfg.CONFIDENCE_SLOPE
    return conf >= floor


def load_both_sides(cache_path, end_dt, days):
    """Load bullish and bearish signals from the same cache file."""
    import json
    raw = json.loads(Path(cache_path).read_text())
    start_dt = end_dt - timedelta(days=days)

    bull_rows, bear_rows = [], []
    for entry in raw.values():
        article = entry.get("_article", {})
        ts_str  = article.get("created_at", "")
        if not ts_str:
            continue
        try:
            ts = datetime.fromisoformat(str(ts_str).replace("Z", "+00:00"))
            if ts.tzinfo is None:
                ts = ts.replace(tzinfo=timezone.utc)
        except Exception:
            continue
        if not (start_dt <= ts <= end_dt):
            continue

        for side, bucket in [("bullish", bull_rows), ("bearish", bear_rows)]:
            sig = entry.get(side, {})
            if sig.get("reasoning") == "SCORE_FAILED" or not sig.get("tickers"):
                continue
            bucket.append({
                "created_at": ts,
                "tickers":    sig.get("tickers", []),
                "magnitude":  float(sig.get("magnitude", 0)),
                "confidence": float(sig.get("confidence", 0)),
                "reasoning":  sig.get("reasoning", ""),
            })

    bull_rows.sort(key=lambda r: r["created_at"])
    bear_rows.sort(key=lambda r: r["created_at"])
    return bull_rows, bear_rows


def build_daily_signals(rows, side):
    """Group signals by date, keeping the highest-magnitude ticker per day."""
    by_day = defaultdict(list)
    for row in rows:
        if not _passes(row):
            continue
        d = row["created_at"].date()
        for tk in row["tickers"][:2]:
            if tk in ("BTC", "ETH") or not is_valid_stock_ticker(tk):
                continue
            by_day[d].append({
                "ticker":     tk,
                "magnitude":  row["magnitude"],
                "confidence": row["confidence"],
                "created_at": row["created_at"],
                "position_usd": _scale(row["magnitude"], row["confidence"]),
            })
    # Keep top-mag signal per day
    best = {}
    for d, signals in by_day.items():
        best[d] = max(signals, key=lambda s: s["magnitude"] * s["confidence"])
    return best


def run_pairs(bull_daily, bear_daily, regime=None, randomize=False, rng=None):
    """
    For each day with both bull and bear signals, simulate L/S pair.
    Returns list of trade dicts with combined pnl_usd.
    """
    common_days = sorted(set(bull_daily) & set(bear_daily))
    trades = []
    for d in common_days:
        if regime and not regime(d):
            continue
        bull_sig = bull_daily[d]
        bear_sig = bear_daily[d]

        if randomize and rng:
            # Swap tickers randomly for the benchmark
            bull_sig = dict(bull_sig, ticker=rng.choice(list(bull_daily[d]["ticker"]
                            if isinstance(bull_daily[d]["ticker"], list)
                            else [bull_daily[d]["ticker"]])))
            bear_sig = dict(bear_sig, ticker=rng.choice(list(bear_daily[d]["ticker"]
                            if isinstance(bear_daily[d]["ticker"], list)
                            else [bear_daily[d]["ticker"]])))

        # Skip if same ticker on both sides (degenerate pair)
        if bull_sig["ticker"] == bear_sig["ticker"]:
            continue

        pos = bull_sig["position_usd"]

        long_t  = simulate_pead(bull_sig["ticker"], bull_sig["created_at"],
                                pos, trail=LONG_TRAIL, max_hold=MAX_HOLD)
        short_t = sim_short(bear_sig["ticker"], bear_sig["created_at"],
                            pos, trail=SHORT_TRAIL, max_hold=MAX_HOLD)

        if long_t is None and short_t is None:
            continue

        long_pnl  = long_t["pnl_usd"]  if long_t  else 0
        short_pnl = short_t["pnl_usd"] if short_t else 0
        net_pnl   = long_pnl + short_pnl

        # For compute_stats compatibility
        trades.append({
            "ticker":    f"{bull_sig['ticker']}L/{bear_sig['ticker']}S",
            "entry_dt":  bull_sig["created_at"].isoformat(),
            "pnl_usd":   net_pnl,
            "pnl_pct":   (net_pnl / pos) * 100 if pos else 0,
            "long_pnl":  long_pnl,
            "short_pnl": short_pnl,
        })
    return trades


def spy_return(end_dt, days):
    """Approximate SPY buy-and-hold return over the window."""
    start = end_dt - timedelta(days=days)
    p0 = get_price_at("SPY", start)
    p1 = get_price_at("SPY", end_dt - timedelta(days=3))
    if p0 and p1:
        return (p1 / p0 - 1) * 100
    return None


def line(tag, s, extra=""):
    return (f"  {tag:38}  n={s['trades']:>4}  Sharpe={s['sharpe']:>6.2f}"
            f"  P&L=${s['total_pnl']:>9,.0f}  win={s['win_rate']:>4.1f}%"
            f"  maxDD=${s['max_dd']:>7,.0f}{extra}")


def main():
    print("═══ MARKET-NEUTRAL PAIRS BACKTEST ═══\n")
    print("Method: long top-bull + short top-bear on same day, equal $ size.\n")

    rng = random.Random(RANDOM_SEED)

    for label, end_dt, days, bull_cache, bear_cache in WINDOWS:
        print(f"{'═'*60}")
        print(f"  {label}")
        print(f"{'═'*60}")

        bull_rows, bear_rows = load_both_sides(bull_cache, end_dt, days)
        print(f"  Bull signals: {len(bull_rows)}  |  Bear signals: {len(bear_rows)}")

        bull_daily = build_daily_signals(bull_rows, "bull")
        bear_daily = build_daily_signals(bear_rows, "bear")
        print(f"  Days with bull: {len(bull_daily)}  |  Days with bear: {len(bear_daily)}")
        print(f"  Overlap days:   {len(set(bull_daily) & set(bear_daily))}\n")

        regime = build_regime(end_dt, days, 200)

        # ── Pairs (no regime gate — key test) ────────────────────────────────
        pairs_ng = run_pairs(bull_daily, bear_daily, regime=None)
        print(line("L/S pairs — NO regime gate", compute_stats(pairs_ng)))

        # ── Pairs (with regime gate) ──────────────────────────────────────────
        pairs_rg = run_pairs(bull_daily, bear_daily, regime=regime)
        print(line("L/S pairs — regime gated",  compute_stats(pairs_rg)))

        # ── Long-only baseline ────────────────────────────────────────────────
        long_only = [{"ticker": d["ticker"], "entry_dt": d["created_at"].isoformat(),
                       "pnl_usd": (simulate_pead(d["ticker"], d["created_at"],
                                                  d["position_usd"], LONG_TRAIL, MAX_HOLD)
                                   or {"pnl_usd": 0})["pnl_usd"],
                       "pnl_pct": 0}
                     for d in bull_daily.values() if regime is None or regime(d["created_at"].date())]
        long_only = [t for t in long_only if t["pnl_usd"] != 0]
        for t in long_only:
            pos = _scale(0.5, 0.7)  # approximate
            t["pnl_pct"] = t["pnl_usd"] / pos * 100 if pos else 0
        print(line("Long-only (bull signals)",   compute_stats(long_only),
                   " ← BASELINE (has market beta)"))

        # ── Random L/S pairs (benchmark) ─────────────────────────────────────
        random_results = []
        for _ in range(3):
            shuffled_bear = dict(bear_daily)
            shuffled_keys = list(shuffled_bear.keys())
            rng.shuffle(shuffled_keys)
            shuffled_bear = dict(zip(bull_daily.keys(), [bear_daily.get(k, v)
                                     for k, v in zip(shuffled_keys, shuffled_bear.values())]))
            t = run_pairs(bull_daily, shuffled_bear, regime=None)
            random_results.extend(t)
        if random_results:
            for t in random_results:
                t["pnl_usd"] /= 3
                t["pnl_pct"] /= 3
        print(line("Random L/S (shuffle benchmark)", compute_stats(random_results)))

        spy_ret = spy_return(end_dt, days)
        if spy_ret is not None:
            print(f"\n  SPY buy-and-hold: {spy_ret:+.1f}% over {days}d")

        # ── SPY correlation ───────────────────────────────────────────────────
        if pairs_ng:
            from backtest import get_stock_bars
            spy_bars = get_stock_bars("SPY", end_dt - timedelta(days=days+5), end_dt)
            if spy_bars:
                spy_by_date = {b["t"].date(): b["c"] for b in spy_bars}
                corr_data = []
                for t in pairs_ng:
                    d = datetime.fromisoformat(t["entry_dt"]).date()
                    if d in spy_by_date:
                        spy_d = list(spy_by_date.keys())
                        idx = spy_d.index(d) if d in spy_d else -1
                        if idx > 0:
                            spy_ret_d = spy_by_date[spy_d[idx]] / spy_by_date[spy_d[idx-1]] - 1
                            corr_data.append((t["pnl_pct"] / 100, spy_ret_d))
                if len(corr_data) >= 10:
                    xs, ys = zip(*corr_data)
                    mean_x = sum(xs)/len(xs); mean_y = sum(ys)/len(ys)
                    cov = sum((x-mean_x)*(y-mean_y) for x,y in zip(xs,ys))/len(xs)
                    std_x = (sum((x-mean_x)**2 for x in xs)/len(xs))**0.5 or 1
                    std_y = (sum((y-mean_y)**2 for y in ys)/len(ys))**0.5 or 1
                    corr  = cov/(std_x*std_y)
                    print(f"  L/S pairs SPY correlation: {corr:+.3f}  "
                          f"(0 = market-neutral, +1 = pure beta)")

        print()

    print("═══ INTERPRETATION ═══")
    print("  Positive L/S Sharpe > random L/S → signal has real alpha")
    print("  L/S Sharpe > long-only Sharpe → regime-independence (key test)")
    print("  Low SPY correlation → market-neutral (not just leveraged beta)")


if __name__ == "__main__":
    main()
