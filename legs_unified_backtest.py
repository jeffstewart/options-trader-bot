"""
legs_unified_backtest.py — re-backtest the three DORMANT legs (bear_short, pairs, pead) on the
LIVE unified_v1 prompt scores, to decide cut-vs-keep. The legs' sim functions are proven; only the
SIGNAL SOURCE changes: instead of the old dual_score_cache bull/bear sides, we join unified_v1's
scores (unified_scores.json) to the same article metadata via cache_key — exactly like
news_call's unified sweep already does for the bullish long legs.

Windows: bull-meltup (180d) + 2022-bear (90d), Yahoo bars. NET of slippage (stock legs are
slippage-only — no IV crush). Each leg keeps its OWN proven exit + a control baseline:
  bear_short : MODEL bearish shorts vs RANDOM shorts (same dates/universe) — selection edge?
  pead       : earnings-filtered drift vs same-signal no-earnings baseline — does PEAD add value?
  pairs      : L/S market-neutral vs long-only vs SPY buy-hold — regime-independent alpha?

Usage:  USE_YAHOO_BARS=1 .venv/bin/python -u legs_unified_backtest.py
"""
import os, json, statistics, random
os.environ.setdefault("USE_YAHOO_BARS", "1")
from datetime import datetime, timezone, timedelta
from pathlib import Path

import config as cfg
from stock_selectivity_sweep import cache_key
from backtest import is_valid_stock_ticker
from benchmark import compute_stats
from regime_filter import build_regime
import bear_short_backtest as bs
import pead_backtest as pead
import pairs_backtest as pairs

WINDOWS = [
    ("bull-meltup", datetime.now(timezone.utc) - timedelta(days=5), 180, "dual_score_cache.json"),
    ("2022-bear",   datetime(2022, 6, 30, tzinfo=timezone.utc),     90,  "bear_dual_cache.json"),
]


def load_unified(cache_file, end_dt, days, sentiment):
    """unified_v1 scores joined to article metadata, filtered to one sentiment + has-ticker +
    in-window. Same row shape the leg sims already consume."""
    sc = json.loads(Path("unified_scores.json").read_text())
    raw = json.loads(Path(cache_file).read_text())
    start_dt = end_dt - timedelta(days=days)
    rows, seen = [], set()
    for entry in raw.values():
        if not isinstance(entry, dict):
            continue
        a = entry.get("_article", {}) or {}
        h, ca, body = a.get("headline"), a.get("created_at"), a.get("summary", "")
        if not h or not ca:
            continue
        try:
            dt = datetime.fromisoformat(str(ca).replace("Z", "+00:00"))
            dt = dt.replace(tzinfo=timezone.utc) if dt.tzinfo is None else dt.astimezone(timezone.utc)
        except Exception:
            continue
        if not (start_dt <= dt <= end_dt):
            continue
        ck = cache_key(h, body)
        if ck in seen:
            continue
        u = sc.get(f"unified_v1:{ck}")
        if not isinstance(u, dict) or u.get("sentiment") != sentiment:
            continue
        tickers = [t for t in (u.get("tickers") or []) if isinstance(t, str)]
        if not tickers:
            continue
        seen.add(ck)
        rows.append({"headline": h, "created_at": dt,
                     "magnitude": float(u.get("magnitude", 0.0) or 0),
                     "confidence": float(u.get("confidence", 0.0) or 0),
                     "tickers": tickers})
    rows.sort(key=lambda x: x["created_at"])
    return rows


def pl(s):
    return f"P&L=${s['total_pnl']:>9,.0f}  Sharpe={s['sharpe']:>5.2f}  win={s['win_rate']:>4.1f}%  n={s['trades']:>4}"


def run_bear_short(label, end_dt, days, cache):
    bear = load_unified(cache, end_dt, days, "bearish")
    universe = sorted({tk for r in bear for tk in r["tickers"][:2] if is_valid_stock_ticker(tk)})
    print(f"\n── bear_short · {label} ──  ({len(bear)} bearish unified_v1 signals, {len(universe)} names)")
    model = bs.run(bear, None)
    print("  " + bs.line("MODEL shorts", model).strip())
    rand_pl, rand_sh = [], []
    for s in range(3):
        rt = bs.run(bear, None, universe=universe, seed=s)
        if rt:
            st = compute_stats(rt); rand_pl.append(st["total_pnl"]); rand_sh.append(st["sharpe"])
    if rand_pl:
        print(f"  RANDOM shorts (control)         P&L=${statistics.mean(rand_pl):>8,.0f}  "
              f"Sharpe={statistics.mean(rand_sh):>5.2f}  (3 seeds)")


def run_pead(label, end_dt, days, cache):
    bull = load_unified(cache, end_dt, days, "bullish")
    reg = build_regime(end_dt, days, 200)
    print(f"\n── pead · {label} ──  ({len(bull)} bullish unified_v1 signals)")
    # LIVE config: flat 20% trail, 10-day hold, earnings-filtered, regime-gated
    live = pead.run_scored(bull, regime=reg, trail=0.20, max_hold=cfg.PEAD_MAX_HOLD_DAYS,
                           earnings_filter=True)
    base = pead.run_scored(bull, regime=reg, trail=0.20, max_hold=cfg.PEAD_MAX_HOLD_DAYS,
                           earnings_filter=False)
    opt = pead.run_scored(bull, regime=reg, trail=0.15, max_hold=45, earnings_filter=True)
    print(f"  LIVE (earnings, 20%/10d)        {pl(compute_stats(live)) if live else 'n=   0'}")
    print(f"  baseline (NO earnings, 20%/10d) {pl(compute_stats(base)) if base else 'n=   0'}")
    print(f"  bt-optimal (earnings, 15%/45d)  {pl(compute_stats(opt)) if opt else 'n=   0'}")


def run_pairs(label, end_dt, days, cache):
    bull = load_unified(cache, end_dt, days, "bullish")
    bear = load_unified(cache, end_dt, days, "bearish")
    bd = pairs.build_daily_signals(bull, "bull")
    rd = pairs.build_daily_signals(bear, "bear")
    reg = build_regime(end_dt, days, 200)
    overlap = sorted(set(bd) & set(rd))
    print(f"\n── pairs · {label} ──  (bull-days={len(bd)} bear-days={len(rd)} overlap={len(overlap)})")
    ng = pairs.run_pairs(bd, rd, regime=None)
    rg = pairs.run_pairs(bd, rd, regime=reg)
    print(f"  L/S pairs — no gate             {pl(compute_stats(ng)) if ng else 'n=   0'}")
    print(f"  L/S pairs — regime gated        {pl(compute_stats(rg)) if rg else 'n=   0'}")
    # long-only baseline on the same bull picks (the alternative use of that capital)
    lo = []
    for d in bd.values():
        if not reg(d["created_at"].date()):
            continue
        t = pairs.simulate_pead(d["ticker"], d["created_at"], d["position_usd"],
                                pairs.LONG_TRAIL, pairs.MAX_HOLD)
        if t:
            lo.append(t)
    print(f"  long-only baseline (bull picks) {pl(compute_stats(lo)) if lo else 'n=   0'}")
    sr = pairs.spy_return(end_dt, days)
    if sr is not None:
        print(f"  SPY buy-hold over window:       {sr:+.1f}%")


def main():
    print("═══ DORMANT-LEG BACKTEST on the LIVE unified_v1 prompt ═══")
    print("Stock legs · NET of slippage · bull-meltup + 2022-bear (Yahoo)")
    for label, end_dt, days, cache in WINDOWS:
        if not Path(cache).exists():
            print(f"\n[{label}] cache {cache} missing — skip"); continue
        print(f"\n{'█'*64}\n  {label}\n{'█'*64}")
        run_bear_short(label, end_dt, days, cache)
        run_pead(label, end_dt, days, cache)
        run_pairs(label, end_dt, days, cache)
    print("\nDecision: a leg is worth KEEPING only if it's net-positive AND beats its control")
    print("(bear_short>random · pead>no-earnings · pairs>0 & justifies the capital vs long-only).")


if __name__ == "__main__":
    main()
