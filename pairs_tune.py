"""
pairs_tune.py — parameter sweep for the L/S pairs strategy.

Sweeps LONG_TRAIL × SHORT_TRAIL × MAX_HOLD cross-regime.
Bull window: 80/20 train/holdout split.
Bear window: full window (small sample, no split).
Ranks by combined score: holdout bull Sharpe + bear Sharpe.

Uses bars already cached in yahoo_bars_cache.json — runs in ~5 min.

Usage:  USE_YAHOO_BARS=1 .venv/bin/python pairs_tune.py
"""
import os, itertools, statistics
from collections import defaultdict
from datetime import datetime, timezone, timedelta
from pathlib import Path

os.environ.setdefault("USE_YAHOO_BARS", "1")

from pairs_backtest import (load_both_sides, build_daily_signals, run_pairs,
                             WINDOWS, RANDOM_SEED)
from benchmark import compute_stats
from regime_filter import build_regime
import config as _cfg

LONG_TRAIL_GRID  = [0.10, 0.15, 0.20, 0.25]
SHORT_TRAIL_GRID = [0.10, 0.15, 0.20]
MAX_HOLD_GRID    = [20, 30, 45]

HOLDOUT_PCT = 0.20


def split_daily(daily_dict, holdout_pct):
    """Split daily signals into train/holdout by date order."""
    dates = sorted(daily_dict.keys())
    split = int(len(dates) * (1 - holdout_pct))
    train_dates = set(dates[:split])
    hold_dates  = set(dates[split:])
    train = {d: v for d, v in daily_dict.items() if d in train_dates}
    hold  = {d: v for d, v in daily_dict.items() if d in hold_dates}
    return train, hold


def run_combo(bull_daily, bear_daily, long_trail, short_trail, max_hold, regime=None):
    """Run pairs with specific params; returns list of trade dicts."""
    common = sorted(set(bull_daily) & set(bear_daily))
    from pead_backtest import simulate_pead
    from pead_bear import sim_short
    trades = []
    for d in common:
        if regime and not regime(d):
            continue
        bull_sig = bull_daily[d]
        bear_sig = bear_daily[d]
        if bull_sig["ticker"] == bear_sig["ticker"]:
            continue
        pos = bull_sig["position_usd"]
        long_t  = simulate_pead(bull_sig["ticker"], bull_sig["created_at"],
                                pos, trail=long_trail, max_hold=max_hold)
        short_t = sim_short(bear_sig["ticker"], bear_sig["created_at"],
                            pos, trail=short_trail, max_hold=max_hold)
        if long_t is None and short_t is None:
            continue
        net = (long_t["pnl_usd"] if long_t else 0) + (short_t["pnl_usd"] if short_t else 0)
        trades.append({
            "ticker":   f"{bull_sig['ticker']}L/{bear_sig['ticker']}S",
            "entry_dt": bull_sig["created_at"].isoformat(),
            "pnl_usd":  net,
            "pnl_pct":  net / pos * 100 if pos else 0,
        })
    return trades


def main():
    print("═══ PAIRS STRATEGY PARAMETER TUNE ═══")
    print(f"Grid: long_trail={LONG_TRAIL_GRID}  short_trail={SHORT_TRAIL_GRID}"
          f"  max_hold={MAX_HOLD_GRID}\n")

    results = []

    for label, end_dt, days, bull_cache, bear_cache in WINDOWS:
        if not Path(bull_cache).exists():
            print(f"  [{label}] cache not found — skipping"); continue

        bull_rows, bear_rows = load_both_sides(bull_cache, end_dt, days)
        bull_daily = build_daily_signals(bull_rows, "bull")
        bear_daily = build_daily_signals(bear_rows, "bear")
        regime     = build_regime(end_dt, days, 200)

        if label == "bull-meltup":
            bull_train, bull_hold = split_daily(bull_daily, HOLDOUT_PCT)
            bear_train, bear_hold = split_daily(bear_daily, HOLDOUT_PCT)
            print(f"  {label}: {len(bull_daily)} bull-days "
                  f"({len(bull_train)} train / {len(bull_hold)} holdout), "
                  f"{len(set(bull_daily)&set(bear_daily))} overlap days\n")

            combos = list(itertools.product(LONG_TRAIL_GRID, SHORT_TRAIL_GRID, MAX_HOLD_GRID))
            combo_results = []
            for lt, st, mh in combos:
                tr = run_combo(bull_train, bear_train, lt, st, mh, regime)
                ho = run_combo(bull_hold,  bear_hold,  lt, st, mh, regime)
                s_tr = compute_stats(tr)
                s_ho = compute_stats(ho)
                combo_results.append({
                    "lt": lt, "st": st, "mh": mh,
                    "tr_sharpe": s_tr["sharpe"], "tr_pnl": s_tr["total_pnl"],
                    "ho_sharpe": s_ho["sharpe"], "ho_pnl": s_ho["total_pnl"],
                    "ho_n": s_ho["trades"], "ho_win": s_ho["win_rate"],
                })

            # Sort by holdout Sharpe
            combo_results.sort(key=lambda r: r["ho_sharpe"], reverse=True)

            print(f"  ── Bull-meltup top 10 (by holdout Sharpe) ──")
            print(f"  {'LT':>5} {'ST':>5} {'Hold':>5}  "
                  f"{'Tr.Sh':>7} {'Tr.P&L':>9}  "
                  f"{'Ho.Sh':>7} {'Ho.P&L':>9} {'Ho.n':>5} {'Ho.win':>7}")
            for r in combo_results[:10]:
                print(f"  {r['lt']:>5.2f} {r['st']:>5.2f} {r['mh']:>5}  "
                      f"{r['tr_sharpe']:>7.2f} ${r['tr_pnl']:>8,.0f}  "
                      f"{r['ho_sharpe']:>7.2f} ${r['ho_pnl']:>8,.0f} "
                      f"{r['ho_n']:>5} {r['ho_win']:>6.1f}%")

            # Store for cross-regime join
            results.append(("bull", combo_results))

        else:  # 2022-bear — full window, no holdout
            bear_combos = []
            combos = list(itertools.product(LONG_TRAIL_GRID, SHORT_TRAIL_GRID, MAX_HOLD_GRID))
            for lt, st, mh in combos:
                tr = run_combo(bull_daily, bear_daily, lt, st, mh, regime=None)
                s  = compute_stats(tr)
                bear_combos.append({
                    "lt": lt, "st": st, "mh": mh,
                    "sharpe": s["sharpe"], "pnl": s["total_pnl"],
                    "n": s["trades"], "win": s["win_rate"],
                })
            results.append(("bear", bear_combos))

            bear_combos.sort(key=lambda r: r["sharpe"], reverse=True)
            print(f"\n  ── 2022-bear top 10 (by Sharpe, no regime gate) ──")
            print(f"  {'LT':>5} {'ST':>5} {'Hold':>5}  "
                  f"{'Sharpe':>7} {'P&L':>9} {'n':>5} {'win':>6}")
            for r in bear_combos[:10]:
                print(f"  {r['lt']:>5.2f} {r['st']:>5.2f} {r['mh']:>5}  "
                      f"{r['sharpe']:>7.2f} ${r['pnl']:>8,.0f} "
                      f"{r['n']:>5} {r['win']:>5.1f}%")

    # Cross-regime ranking
    if len(results) == 2:
        bull_res = {(r["lt"], r["st"], r["mh"]): r for r in results[0][1]}
        bear_res = {(r["lt"], r["st"], r["mh"]): r for r in results[1][1]}
        cross = []
        for key, br in bull_res.items():
            be = bear_res.get(key, {})
            bull_ho = br["ho_sharpe"]
            bear_s  = be.get("sharpe", 0)
            # Combined score: bull holdout Sharpe + bear Sharpe (equal weight)
            combined = bull_ho + bear_s
            cross.append({
                "key": key, "bull_ho": bull_ho, "bear_s": bear_s,
                "combined": combined,
                "bull_ho_pnl": br["ho_pnl"], "bear_pnl": be.get("pnl", 0),
            })
        cross.sort(key=lambda r: r["combined"], reverse=True)

        print(f"\n  ══ CROSS-REGIME RANKING (bull holdout + bear, combined score) ══")
        print(f"  {'LT':>5} {'ST':>5} {'Hold':>5}  "
              f"{'Bull.Ho.Sh':>11} {'Bear.Sh':>8}  "
              f"{'Combined':>9}  Bull.P&L  Bear.P&L")
        for r in cross[:10]:
            lt, st, mh = r["key"]
            print(f"  {lt:>5.2f} {st:>5.2f} {mh:>5}  "
                  f"{r['bull_ho']:>11.2f} {r['bear_s']:>8.2f}  "
                  f"{r['combined']:>9.2f}  "
                  f"${r['bull_ho_pnl']:>7,.0f}  ${r['bear_pnl']:>7,.0f}")

        best = cross[0]
        lt, st, mh = best["key"]
        print(f"\n  ★ RECOMMENDED PARAMS: long_trail={lt}  short_trail={st}  max_hold={mh}")
        print(f"    Bull holdout Sharpe: {best['bull_ho']:.2f} | Bear Sharpe: {best['bear_s']:.2f}")

    print("\n  Note: bear window is 90d (small) — bear Sharpe has high variance.")
    print("  Trust bull holdout Sharpe as primary. Bear = directional signal only.")


if __name__ == "__main__":
    main()
