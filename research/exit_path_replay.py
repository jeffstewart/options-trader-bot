"""
exit_path_replay.py — replay counterfactual exit rules over REAL logged premium paths.

data/position_paths.csv (core/bot.py `_log_position_path`, since 2026-06-26) samples every live
position's real option mid at ~30s cadence — phase=open while held, and phase=postclose AFTER the
live exit (post-exit tracking to the position's natural max-hold horizon was designed in
precisely so hold-longer counterfactuals are observable; see memory project_exit_strategy_intraday).

So each position gives: the LIVE exit outcome (last phase=open sample — the ratchet 40/25/15
@+15/+35 policy deployed 2026-06-26) and a FULL real path (open + postclose) any alternate rule
can replay over. This is the validation the synthetic sweeps kept deferring: flat-40 vs flat-25
vs ratchet vs take-profits vs EOD, on real quotes, no BS reconstruction.

All trail percentages are fractions of PEAK PREMIUM (mid <= peak*(1-x)), matching live
semantics. Exits fill at the sample that triggered — same stop-fills-at-stop optimism for every
rule including the baseline, so comparisons are apples-to-apples even though absolute numbers
are slightly kind. Positions not observed from birth (|first pnl| > 10%) are dropped.

Usage:  .venv/bin/python research/exit_path_replay.py [--paths data/position_paths.csv]
"""
import argparse
from collections import defaultdict

import numpy as np
import pandas as pd

RULES = [
    # name, fn(entry, ts_arr, mid_arr, first_day) -> exit pnl fraction
    ("flat trail 40%",        {"kind": "trail", "tiers": [(0.0, 0.40)]}),
    ("flat trail 25%",        {"kind": "trail", "tiers": [(0.0, 0.25)]}),
    ("flat trail 15%",        {"kind": "trail", "tiers": [(0.0, 0.15)]}),
    ("ratchet 40/25/15 @15/35", {"kind": "trail", "tiers": [(0.0, 0.40), (0.15, 0.25), (0.35, 0.15)]}),
    ("TP +15%",               {"kind": "tp", "tp": 0.15}),
    ("TP +25%",               {"kind": "tp", "tp": 0.25}),
    ("TP +50%",               {"kind": "tp", "tp": 0.50}),
    ("TP25 + flat trail 40%", {"kind": "tp_trail", "tp": 0.25, "tiers": [(0.0, 0.40)]}),
    ("close EOD entry day",   {"kind": "eod"}),
    ("hold to horizon",       {"kind": "hold"}),
]


def load_positions(paths_csv: str):
    df = pd.read_csv(paths_csv, parse_dates=["ts"])
    df = df[(df.asset_type == "option") & (df.entry > 0)].dropna(subset=["mid"])
    positions = []
    for (sym, entry), g in df.groupby(["symbol", "entry"]):
        g = g.sort_values("ts")
        if abs(g.iloc[0].pnl_pct) > 10:      # opened before logging began -> biased, drop
            continue
        opened = g[g.phase == "open"]
        if opened.empty:
            continue
        positions.append({
            "symbol": sym, "strategy": g.iloc[0].strategy, "entry": entry,
            "ts": list(g.ts), "mid": g.mid.to_numpy(),
            "first_day": g.iloc[0].ts.date(),
            # the live (ratchet) exit realized this mark:
            "live_exit_pnl": opened.iloc[-1].mid / entry - 1,
            "has_postclose": (g.phase == "postclose").any(),
        })
    return positions


def trail_pct(tiers, peak_pnl):
    pct = tiers[0][1]
    for arm, p in tiers:
        if peak_pnl >= arm:
            pct = p
    return pct


def replay(pos, rule):
    entry = pos["entry"]
    peak_mid = 0.0
    for ts, mid in zip(pos["ts"], pos["mid"]):
        peak_mid = max(peak_mid, mid)
        pnl = mid / entry - 1
        if rule["kind"] in ("tp", "tp_trail") and pnl >= rule["tp"]:
            return pnl
        if rule["kind"] in ("trail", "tp_trail") and "tiers" in rule:
            if mid <= peak_mid * (1 - trail_pct(rule["tiers"], peak_mid / entry - 1)):
                return pnl
        if rule["kind"] == "eod" and ts.date() == pos["first_day"] \
                and (ts.hour, ts.minute) >= (19, 55):     # UTC; 19:55Z ~= 15:55 ET
            return pnl
    return pos["mid"][-1] / entry - 1


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--paths", default="/Users/jeff/Claude/Trader/data/position_paths.csv")
    args = ap.parse_args()
    by_strat = defaultdict(list)
    for p in load_positions(args.paths):
        by_strat[p["strategy"]].append(p)

    for strat, group in sorted(by_strat.items()):
        n_pc = sum(p["has_postclose"] for p in group)
        print(f"\n── {strat} (n={len(group)}, {n_pc} with post-exit tracking) " + "─" * 30)
        live = np.array([p["live_exit_pnl"] for p in group])
        print(f"{'rule':<28} {'mean%':>7} {'med%':>7} {'win%':>5} {'P(≥+25%)':>9} "
              f"{'worst%':>7} {'Δlive':>7} {'t':>5}")
        print(f"{'LIVE exits (ratchet)':<28} {100 * live.mean():>+7.1f} "
              f"{100 * np.median(live):>+7.1f} {100 * (live > 0).mean():>5.0f} "
              f"{100 * (live >= 0.25).mean():>9.0f} {100 * live.min():>+7.1f}")
        for name, rule in RULES:
            r = np.array([replay(p, rule) for p in group])
            d = r - live
            t = d.mean() / (d.std(ddof=1) / np.sqrt(len(d))) if d.std(ddof=1) > 0 else 0.0
            print(f"{name:<28} {100 * r.mean():>+7.1f} {100 * np.median(r):>+7.1f} "
                  f"{100 * (r > 0).mean():>5.0f} {100 * (r >= 0.25).mean():>9.0f} "
                  f"{100 * r.min():>+7.1f} {100 * d.mean():>+7.1f} {t:>5.1f}")


if __name__ == "__main__":
    main()
