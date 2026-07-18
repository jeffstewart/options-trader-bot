"""
lotto_sonnet5_selectivity_grid.py — fine mag x conf grid for LOTTO on sonnet5 scores, looking for
a cell that pushes win% meaningfully above what Ollama's own gates found
(lotto_sonnet5/small_account_lotto_pead_sweep.py: Ollama tight gate ~35-40% win, top3 60-82% of
P&L). Same deep-OTM Δ0.20/DTE14 tiered-exit sim, fixed $100 budget for comparability across cells
and against the Ollama grid (rerun alongside at the same budget for a clean side-by-side).

Usage:  USE_YAHOO_BARS=1 .venv/bin/python -u lotto_sonnet5_selectivity_grid.py   (run from data/)
"""
import os, statistics
os.environ.setdefault("USE_YAHOO_BARS", "1")

import backtest as _bt
import config as _cfg
import news_call_sweep_unified as nc
import small_account_sonnet5_sweep as s5
from benchmark import compute_stats
from regime_filter import build_regime

BUDGET = 100
MAG_GRID  = [0.30, 0.40, 0.50, 0.60, 0.65, 0.75]
CONF_GRID = [0.60, 0.65, 0.70, 0.75, 0.80, 0.85]
MIN_N = 15   # below this, don't trust the cell


def run_grid(rows, reg, label):
    print(f"\n══ {label} — mag x conf grid, ${BUDGET} budget ══")
    print(f"  {'mag>=':>6} {'conf>=':>7} {'n':>4} {'total$':>9} {'win%':>6} {'sharpe':>7} {'top3%':>6}")
    results = []
    for mag in MAG_GRID:
        for conf in CONF_GRID:
            gated = [r for r in rows if r["magnitude"] >= mag and r["confidence"] >= conf]
            trades, seen = [], set()
            for r in gated:
                d = r["created_at"].date()
                if reg and not reg(d):
                    continue
                for tk in r["tickers"][:1]:
                    if not _bt.is_valid_stock_ticker(tk):
                        continue
                    key = f"{d}_{tk}"
                    if key in seen:
                        continue
                    seen.add(key)
                    sp = _bt.get_price_at(tk, r["created_at"])
                    if not sp:
                        continue
                    t = _bt.simulate_option_pnl(tk, r["created_at"], sp, BUDGET,
                                                {"magnitude": r["magnitude"], "confidence": r["confidence"]},
                                                option_type="call", exit_rule="tiered_profit",
                                                spread_mult=1.0)
                    if t:
                        trades.append(t)
            if len(trades) < MIN_N:
                continue
            trades.sort(key=lambda t: -t["pnl_usd"])
            tot = sum(t["pnl_usd"] for t in trades)
            top3 = sum(t["pnl_usd"] for t in trades[:3])
            s = compute_stats(trades)
            top3_pct = (top3 / tot * 100) if tot else 0
            print(f"  {mag:>6.2f} {conf:>7.2f} {s['trades']:>4} ${s['total_pnl']:>+8,.0f} "
                  f"{s['win_rate']:>5.1f}% {s['sharpe']:>+6.2f} {top3_pct:>5.0f}%")
            results.append((mag, conf, s["trades"], s["win_rate"], s["sharpe"], top3_pct))
    if results:
        best = max(results, key=lambda r: r[3])
        print(f"\n  best win%% cell (n>={MIN_N}): mag>={best[0]} conf>={best[1]}  "
              f"n={best[2]} win={best[3]:.1f}% sharpe={best[4]:+.2f} top3={best[5]:.0f}%")
    return results


def main():
    tiered = {"tiers": [(1.0, 0.40), (3.0, 0.30), (float("inf"), 0.20)]}
    save = {k: getattr(_bt, k) for k in
            ("TARGET_DELTA", "DTE_TARGET", "MAX_HOLD_DAYS", "TRAILING_STOP_PCT", "EXIT_PARAMS")}
    save_mult = _cfg.MAX_CONTRACT_BUDGET_MULT
    try:
        _bt.TARGET_DELTA, _bt.DTE_TARGET, _bt.MAX_HOLD_DAYS = 0.20, 14, 7
        _bt.TRAILING_STOP_PCT, _bt.EXIT_PARAMS = 0.30, tiered
        _cfg.MAX_CONTRACT_BUDGET_MULT = 1.0

        s5_rows = s5.load_sonnet5_rows()
        reg = build_regime(s5.END_DT, s5.DAYS, 200)
        run_grid(s5_rows, reg, "SONNET5")

        label, end_dt, days, cache = nc.BULL
        ollama_rows = nc.load_scored_from_unified(cache, end_dt, days, "unified_v1")
        reg2 = build_regime(end_dt, days, 200)
        run_grid(ollama_rows, reg2, "OLLAMA (reference, same budget/sim)")
    finally:
        for k, v in save.items():
            setattr(_bt, k, v)
        _cfg.MAX_CONTRACT_BUDGET_MULT = save_mult


if __name__ == "__main__":
    main()
