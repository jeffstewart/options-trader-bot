"""
news_call_geom_sweep.py — sweep news_call GEOMETRY (target delta × DTE) at the already-chosen
sweet-spot gate (mag≥0.75 & conf≥dynamic) on the unified_v1 score distribution, NET of realistic
option costs. The gate sweep (news_call_sweep_unified.py) fixed mag≥0.75; this asks whether a
different strike (delta) or expiry (DTE) survives spreads better than the live Δ0.50 / DTE17.

Reuses news_call_sweep_unified.run_gate (same BS+IV-crush sim, tiered bid/ask spread model,
live sizing, SPY>200d regime gate, tiered_trail exit). Cached scores + Yahoo bars only.

Usage:  USE_YAHOO_BARS=1 .venv/bin/python -u news_call_geom_sweep.py
"""
import os, random
os.environ.setdefault("USE_YAHOO_BARS", "1")

import news_call_sweep_unified as nc
from regime_filter import build_regime

MAG, CONF = 0.75, 0.70                      # the chosen sweet-spot gate
DELTAS = [0.40, 0.50, 0.60, 0.70]
DTES   = [10, 14, 17, 21, 30, 45]           # 17 = current live (mid of 14-21)
LIVE   = (0.50, 17)
N_BOOT = 10000
random.seed(20260614)


def main():
    label, end_dt, days, cache = nc.BULL
    rows = nc.load_scored_from_unified(cache, end_dt, days, "unified_v1")
    reg = build_regime(end_dt, days, 200)
    blabel, bend, bdays, bcache = nc.BEAR
    brows = nc.load_scored_from_unified(bcache, bend, bdays, "unified_v1")
    breg = build_regime(bend, bdays, 200)

    print(f"news_call GEOMETRY sweep @ sweet-spot gate (mag≥{MAG} & conf≥dyn), "
          f"{len(rows)} bullish unified_v1 signals, NET of realistic spreads\n")
    print("  total net P&L / Sharpe by Δ (rows) × DTE (cols):")
    print(f"  {'Δ \\ DTE':>8}" + "".join(f"{d:>13}" for d in DTES))
    res = {}
    for delta in DELTAS:
        cells = []
        for dte in DTES:
            s = nc.stats(nc.run_gate(rows, reg, MAG, CONF, spread_mult=1.0, delta=delta, dte=dte))
            res[(delta, dte)] = s
            cells.append(f"${s['total']/1000:>4.0f}k/{s['sharpe']:>4.2f}")
        print(f"  Δ{delta:>0.2f}  " + "".join(f"{c:>13}" for c in cells), flush=True)

    print("\n  Top geometries by Sharpe (n≥100, net-positive):")
    print(f"    {'Δ':>4} {'DTE':>4} {'n':>4} {'total':>9} {'$/tr':>7} {'Sharpe':>7} {'win%':>5} {'2x+':>4} {'4x+':>4}")
    elig = [(k, s) for k, s in res.items() if s["n"] >= 100 and s["total"] > 0]
    for (delta, dte), s in sorted(elig, key=lambda kv: -kv[1]["sharpe"])[:8]:
        live = "  ← live" if (delta, dte) == LIVE else ""
        print(f"    {delta:>4.2f} {dte:>4} {s['n']:>4} {('${:,.0f}'.format(s['total'])):>9} "
              f"{('${:+,.0f}'.format(s['mean'])):>7} {s['sharpe']:>7.2f} {s['win']:>5.0f} "
              f"{s['x2']:>4} {s['x4']:>4}{live}")

    cur = res[LIVE]
    print(f"\n  CURRENT live geometry Δ0.50/DTE17: n={cur['n']} ${cur['total']:,.0f} "
          f"${cur['mean']:+,.0f}/tr Sh{cur['sharpe']:.2f} win{cur['win']:.0f}%")

    if not elig:
        print("\n  VERDICT: no geometry net-positive with ≥100 trades — keep live Δ0.50/DTE17.")
        return
    (bd, bt), best = max(elig, key=lambda kv: kv[1]["sharpe"])
    pnls = best["pnls"]
    B = sorted(sum(random.choices(pnls, k=len(pnls))) for _ in range(N_BOOT))
    lo, hi = B[int(0.025 * N_BOOT)], B[int(0.975 * N_BOOT)]
    big = max(pnls)
    bb = nc.stats(nc.run_gate(brows, breg, MAG, CONF, spread_mult=1.0, delta=bd, dte=bt))
    print(f"\n  BEST-Sharpe geometry: Δ{bd}/DTE{bt} → n={best['n']} ${best['total']:,.0f} "
          f"${best['mean']:+,.0f}/tr Sh{best['sharpe']:.2f} win{best['win']:.0f}%")
    print(f"    BOOTSTRAP total P&L 95% CI [${lo:,.0f}, ${hi:,.0f}]  "
          f"P>0={sum(1 for b in B if b > 0)/N_BOOT*100:.0f}%")
    print(f"    JACKKNIFE biggest ${big:,.0f} ({big/best['total']*100:.0f}%) → drop = ${best['total']-big:,.0f}")
    print(f"    BEAR guard (2022, regime-gated): n={bb['n']} ${bb['total']:,.0f} Sh{bb['sharpe']:.2f}")
    print(f"\n  vs live Δ0.50/DTE17 (${cur['total']:,.0f}/Sh{cur['sharpe']:.2f}): "
          f"Δ improvement {best['total']-cur['total']:+,.0f} / Sharpe {best['sharpe']-cur['sharpe']:+.2f}")
    print("  NOTE: prefer a robust PLATEAU over the single best cell; longer-DTE/higher-Δ = more "
          "bull beta, so weight the bear guard + bootstrap, not just peak bull Sharpe.")


if __name__ == "__main__":
    main()
