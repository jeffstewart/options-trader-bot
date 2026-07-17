"""
small_account_news_call_sweep.py — news_call sizing/selectivity for a $500-1000 real account.

The live sizing formula (`scale(mag,conf) = max(0.10, mag*conf) x MAX_POSITION_USD`) targets
~$500-750/trade at MAX_POSITION_USD=$1000 (confirmed empirically: 70% of live trades spent
$500-750, per trades.csv cost_basis x qty) -- that's the ENTIRE small account in one trade. Fixing
this needs two independent levers, swept together:
  1. LEG BUDGET -- position_usd passed to simulate_option_pnl (small accounts can't dollar-scale up
     qty like the $83k book does; qty floors to 1 naturally at small budgets)
  2. AFFORDABILITY GATE -- MAX_CONTRACT_BUDGET_MULT tightened from the live 3.0x down toward 1.0x,
     so a contract genuinely unaffordable at this budget gets SKIPPED, not bought oversized.
  3. SELECTIVITY -- mag/conf floor, since jeff wants fewer/better trades over many marginal ones,
     which the live-loss review also supports empirically (sub-threshold shadow book was ~wash).

Same unified_v1 sim as news_call_sweep_unified.py (NET of costs -- spread, IV crush). Judge on
avg $/trade, win%, and how much of the universe survives the affordability gate, not totals (this
session's other sweeps all showed totals are tail-skewed).

Usage:  USE_YAHOO_BARS=1 .venv/bin/python -u small_account_news_call_sweep.py   (run from data/)
"""
import os, statistics
os.environ.setdefault("USE_YAHOO_BARS", "1")

import config as _cfg
import news_call_sweep_unified as nc
from benchmark import compute_stats
from regime_filter import build_regime

LEG_BUDGETS = [75, 100, 150, 200, 250]
MAG_FLOORS  = [0.75, 0.80, 0.85, 0.90]
BUDGET_MULT = 1.15   # small tolerance over the exact leg budget (matches "qty floors to 1 anyway")

# GEOMETRY x BUDGET cross: ATM (live default, Delta 0.50/DTE17) vs the ITM/longer-DTE geometry
# news_call_options_test.py already showed improves median (-$286->-$90) and win% (36->43) --
# does that survive at small-account budgets, given ITM options cost MORE per contract (more
# intrinsic value baked in)? Fixed mag>=0.80 (best cell from the first sweep), sweep geometry x budget.
GEOMETRIES = [("ATM  d0.50/17d (live)", 0.50, 17), ("ITM  d0.70/30d", 0.70, 30), ("ITM  d0.80/45d", 0.80, 45)]


def scan():
    label, end_dt, days, cache = nc.BULL
    rows = nc.load_scored_from_unified(cache, end_dt, days, "unified_v1")
    reg  = build_regime(end_dt, days, 200)
    print(f"{label}: {len(rows)} bullish unified_v1 signals\n")

    save_max_mult = _cfg.MAX_CONTRACT_BUDGET_MULT
    _cfg.MAX_CONTRACT_BUDGET_MULT = BUDGET_MULT

    try:
        print(f"{'leg $':>6} {'mag≥':>5} {'n':>4} {'avg$/tr':>8} {'med%':>6} {'win%':>5} {'sharpe':>7} {'skip-unafford%':>14}")
        print("-" * 62)
        # nc.scale() closes over module-level BASE_USD; patch it directly for the sweep.
        for budget in LEG_BUDGETS:
            nc.BASE_USD = budget
            for mag_floor in MAG_FLOORS:
                trades = nc.run_gate(rows, reg, mag_floor, 0.70, spread_mult=1.0,
                                     delta=nc.DELTA, dte=nc.DTE)
                # universe of candidates that PASS mag/conf+regime (denominator for skip-rate),
                # independent of affordability -- inlines run_gate's own filter (no shared helper)
                universe_n = sum(1 for r in rows
                                 if r["magnitude"] >= mag_floor and r["confidence"] >= 0.70
                                 and reg(r["created_at"].date()))
                s = compute_stats(trades) if trades else None
                skip_pct = (1 - len(trades) / universe_n) * 100 if universe_n else 0
                if s and s["trades"] >= 5:
                    med = statistics.median(t["pnl_pct"] for t in trades)
                    print(f"${budget:>5} {mag_floor:>5.2f} {s['trades']:>4} ${s['total_pnl']/s['trades']:>+7,.0f} "
                          f"{med:>+5.1f}% {s['win_rate']:>4.0f}% {s['sharpe']:>+6.2f} {skip_pct:>13.0f}%")
                else:
                    n = s["trades"] if s else 0
                    print(f"${budget:>5} {mag_floor:>5.2f} {n:>4}  (insufficient sample)")
        print(f"\n\n══ GEOMETRY x BUDGET cross (fixed mag>=0.80 -- best selectivity cell above) ══")
        print(f"{'geometry':>24} {'leg $':>6} {'n':>4} {'avg$/tr':>8} {'med%':>6} {'win%':>5} {'sharpe':>7} {'skip%':>6}")
        print("-" * 68)
        for gname, delta, dte in GEOMETRIES:
            for budget in LEG_BUDGETS:
                nc.BASE_USD = budget
                trades = nc.run_gate(rows, reg, 0.80, 0.70, spread_mult=1.0, delta=delta, dte=dte)
                universe_n = sum(1 for r in rows
                                 if r["magnitude"] >= 0.80 and r["confidence"] >= 0.70
                                 and reg(r["created_at"].date()))
                skip_pct = (1 - len(trades) / universe_n) * 100 if universe_n else 0
                s = compute_stats(trades) if trades else None
                if s and s["trades"] >= 5:
                    med = statistics.median(t["pnl_pct"] for t in trades)
                    print(f"{gname:>24} ${budget:>5} {s['trades']:>4} ${s['total_pnl']/s['trades']:>+7,.0f} "
                          f"{med:>+5.1f}% {s['win_rate']:>4.0f}% {s['sharpe']:>+6.2f} {skip_pct:>5.0f}%")
                else:
                    n = s["trades"] if s else 0
                    print(f"{gname:>24} ${budget:>5} {n:>4}  (insufficient sample, skip {skip_pct:.0f}%)")
    finally:
        _cfg.MAX_CONTRACT_BUDGET_MULT = save_max_mult


if __name__ == "__main__":
    scan()
