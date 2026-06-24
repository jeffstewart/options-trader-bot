"""
qqq_macro_sweep.py — sweep DTE × delta for the qqq_macro strategy (long QQQ calls on
macro-bullish, no-ticker signals). The strategy's contract geometry (14-21 DTE → mid 18,
delta 0.50) was inherited from the generic equity-option defaults and NEVER swept — only the
ROUTING decision was validated (qqq_routing.py). This asks: is ATM/~18-DTE actually optimal,
or does a macro/index thesis prefer different DTE/delta?

Same 157 macro-bullish signals + the validated BS pricing engine. Holds everything else at the
LIVE config (trail 10%, hold-cap 30, trail_premium exit) and varies ONLY DTE_TARGET & TARGET_DELTA.
The (DTE 18, Δ0.50) cell reproduces qqq_routing.py's qqq_only arm (+$100,036 / Sharpe 3.62) as a
harness check. Best combo is bootstrap/jackknifed vs the current config before any change.

Usage:  .venv/bin/python qqq_macro_sweep.py
"""
import random, statistics
from datetime import timezone

import backtest as bt
from benchmark import compute_stats
from config import MAX_POSITION_USD
from qqq_routing import macro_bullish_signals

DTES   = [10, 14, 18, 21, 30, 45, 60]
DELTAS = [0.30, 0.40, 0.50, 0.60, 0.70]
CUR_DTE, CUR_DELTA = 18, 0.50          # current live qqq_macro
N_BOOT = 10000
random.seed(20260612)


def simulate_combo(macro, prices, dte, delta):
    """Returns (trades list, {date: pnl_usd}). Overrides only DTE_TARGET & TARGET_DELTA;
    everything else stays at the live/backtest default (trail 10%, hold-cap 30)."""
    save = (bt.DTE_TARGET, bt.TARGET_DELTA)
    bt.DTE_TARGET, bt.TARGET_DELTA = dte, delta
    trades, by_date = [], {}
    try:
        for dt, r in macro:
            px = prices.get(dt)
            if not px:
                continue
            mag, conf = float(r["magnitude"]), float(r["confidence"])
            pos = bt.scale_position_usd(MAX_POSITION_USD, mag, conf)
            res = bt.simulate_option_pnl("QQQ", dt, px, pos,
                                         {"magnitude": mag, "confidence": conf, "reasoning": ""})
            if res:
                trades.append(res)
                by_date[dt.date()] = res["pnl_usd"]
    finally:
        bt.DTE_TARGET, bt.TARGET_DELTA = save
    return trades, by_date


def pctl(xs, p):
    xs = sorted(xs); i = max(0, min(len(xs) - 1, int(round(p / 100 * (len(xs) - 1)))))
    return xs[i]


def main():
    macro = macro_bullish_signals()
    prices = {dt: bt.get_price_at("QQQ", dt) for dt, _ in macro}      # fetch once, reuse
    n_px = sum(1 for v in prices.values() if v)
    print(f"qqq_macro DTE×delta sweep — {len(macro)} macro signals ({n_px} priced), "
          f"live exit (trail {bt.TRAILING_STOP_PCT:.0%}, hold-cap {bt.MAX_HOLD_DAYS})\n")

    results = {}   # (dte,delta) -> (stats, by_date, trades)
    total = len(DTES) * len(DELTAS); done = 0
    for dte in DTES:
        for d in DELTAS:
            tr, bd = simulate_combo(macro, prices, dte, d)
            results[(dte, d)] = (compute_stats(tr), bd, tr)
            done += 1
            print(f"    [{done:>2}/{total}] DTE{dte:>2} Δ{d:.2f}: {len(tr):>3} trades  "
                  f"${compute_stats(tr)['total_pnl']:>+10,.0f}", flush=True)
    print()

    def grid(metric, fmt):
        print(f"  ── {metric} ──")
        print("   DTE\\Δ " + "".join(f"{d:>11.2f}" for d in DELTAS))
        for dte in DTES:
            cells = []
            for d in DELTAS:
                s = results[(dte, d)][0]
                v = s.get(metric, 0)
                mark = "*" if (dte, d) == (CUR_DTE, CUR_DELTA) else " "
                cells.append(f"{fmt(v):>10}{mark}")
            print(f"  {dte:>4}  " + "".join(cells))
        print()

    grid("total_pnl", lambda v: f"${v:,.0f}")
    grid("sharpe",    lambda v: f"{v:.2f}")

    cur_s = results[(CUR_DTE, CUR_DELTA)][0]
    print(f"  * = current live (DTE {CUR_DTE}, Δ{CUR_DELTA}):  "
          f"${cur_s['total_pnl']:,.0f}  Sharpe {cur_s['sharpe']:.2f}  "
          f"(qqq_routing ref: $100,036 / 3.62 — harness check)\n")

    # Rank: require BEAT current on BOTH P&L and Sharpe (don't trade $ for vol blindly)
    cur_pnl, cur_sh = cur_s["total_pnl"], cur_s["sharpe"]
    better = sorted([(k, s) for k, (s, _, _) in results.items()
                     if s["total_pnl"] > cur_pnl and s["sharpe"] >= cur_sh and k != (CUR_DTE, CUR_DELTA)],
                    key=lambda x: x[1]["total_pnl"], reverse=True)
    if not better:
        print("  No combo beats current on BOTH P&L and Sharpe → current geometry holds. Done.")
        best_pnl = max(results.items(), key=lambda kv: kv[1][0]["total_pnl"])
        best_sh  = max(results.items(), key=lambda kv: kv[1][0]["sharpe"])
        print(f"  (best P&L cell: {best_pnl[0]} ${best_pnl[1][0]['total_pnl']:,.0f}/Sh{best_pnl[1][0]['sharpe']:.2f}; "
              f"best Sharpe cell: {best_sh[0]} ${best_sh[1][0]['total_pnl']:,.0f}/Sh{best_sh[1][0]['sharpe']:.2f})")
        return

    cand_key = better[0][0]
    cs = results[cand_key][0]
    print(f"  Candidate (beats current on both): DTE {cand_key[0]}, Δ{cand_key[1]} → "
          f"${cs['total_pnl']:,.0f}/Sh{cs['sharpe']:.2f}  vs current ${cur_pnl:,.0f}/Sh{cur_sh:.2f}\n")

    # ── PAIRED bootstrap/jackknife: candidate − current per signal (same dates) ──
    cur_bd, cand_bd = results[(CUR_DTE, CUR_DELTA)][1], results[cand_key][1]
    common = sorted(set(cur_bd) & set(cand_bd))
    diffs = [cand_bd[d] - cur_bd[d] for d in common]
    tot = sum(diffs)
    boots = [sum(random.choices(diffs, k=len(diffs))) for _ in range(N_BOOT)]
    lo, hi = pctl(boots, 2.5), pctl(boots, 97.5)
    frac = sum(1 for b in boots if b > 0) / N_BOOT
    print(f"  Paired Δ (candidate − current) over {len(common)} signals: total ${tot:+,.0f}")
    print(f"    95% CI [${lo:+,.0f}, ${hi:+,.0f}]   P(candidate > current) = {frac*100:.1f}%")
    big = max(diffs, key=abs)
    print(f"    jackknife: drop biggest single Δ (${big:+,.0f}) → total ${tot-big:+,.0f} "
          f"({'still favors candidate' if (tot-big) > 0 else 'FLIPS — edge was one signal'})")
    verdict = ("ROBUST → worth switching" if lo > 0 and (tot - big) > 0
               else "NOT robust (CI straddles 0 or one-signal-driven) → keep current")
    print(f"\n  VERDICT: {verdict}")


if __name__ == "__main__":
    main()
