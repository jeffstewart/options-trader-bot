"""
news_call_sweep_unified.py — evaluate + sweep the news_call strategy (standard ~Δ0.50 ATM
calls on bullish signals) on the LIVE unified_v1 score distribution, NET of realistic option
costs.

Context: news_call was DISABLED 2026-06-05 after 38 live trades printed 13% win / -$6,601 —
realistic option costs ate the edge at its gate width (the backtested cost-fragility finding).
The standing note: "re-enable only after identifying a cost-survivable improvement." Q now: does
unified_v1's better-calibrated magnitude/confidence yield a gate where news_call is net-positive?

Reuses backtest.simulate_option_pnl (BS + IV-crush pricing + tiered bid/ask spread model). The
spread is the whole point — every number is NET of spread_mult=1.0 (realistic), with a
frictionless (spread_mult=0.0) reference so the cost drag is explicit. Cached scores + Yahoo
bars only (NO ollama / Alpaca). Bull window + 2022-bear out-of-regime guard.

Faithful to LIVE news_call geometry: Δ0.50 ATM, DTE≈17 (mid of the 14-21 window), tiered_trail
exit on the config EXIT_TIERS, sized scale_position_usd(MAX_POSITION_USD, mag, conf), 2 tickers
per signal, SPY>200d regime gate.

Usage:  USE_YAHOO_BARS=1 .venv/bin/python -u news_call_sweep_unified.py
        NC_DELTA=0.60 NC_DTE=21 ... .venv/bin/python -u news_call_sweep_unified.py   # geometry sweep
"""
import os, random
os.environ.setdefault("USE_YAHOO_BARS", "1")
from datetime import datetime, timezone, timedelta

import backtest as _bt
import config as _cfg
from benchmark import compute_stats
from regime_filter import build_regime
from stock_selectivity_sweep import load_scored_from_unified  # unified_scores join loader

PROMPT = os.environ.get("UNI_PROMPT", "unified_v1")
BULL = ("bull-meltup", datetime.now(timezone.utc) - timedelta(days=5), 180, "dual_score_cache.json")
BEAR = ("2022-bear",   datetime(2022, 6, 30, tzinfo=timezone.utc),     90,  "bear_dual_cache.json")

# news_call live geometry (env-overridable for a geometry sweep)
DELTA = float(os.environ.get("NC_DELTA", "0.50"))   # ATM
DTE   = int(os.environ.get("NC_DTE", "17"))          # mid of the live 14-21 window
BASE_USD  = _cfg.MAX_POSITION_USD                     # 1000, scaled by mag×conf (floor 10%)
EXIT_RULE = "tiered_trail"
EXIT = {**_bt.EXIT_PARAMS, "tiers": list(_cfg.EXIT_TIERS)}   # live tiered trail

MAG_FLOORS  = [0.35, 0.45, 0.55, 0.65, 0.75]
CONF_FLOORS = [0.70, 0.80, 0.85, 0.90]
MIN_TRADES  = 30
N_BOOT = 10000
random.seed(20260614)


def scale(mag, conf):
    return max(BASE_USD * 0.10, BASE_USD * mag * conf)


def run_gate(rows, regime, mag_floor, conf_floor, spread_mult=1.0, delta=DELTA, dte=DTE):
    """Simulate news_call (Δ ATM) on rows passing (mag≥mag_floor & conf≥conf_floor),
    regime-gated, 1 trade per ticker/day, ≤2 tickers/signal, NET of spread_mult."""
    save = {k: getattr(_bt, k) for k in
            ("TARGET_DELTA", "DTE_TARGET", "MAX_HOLD_DAYS", "TRAILING_STOP_PCT", "EXIT_PARAMS")}
    _bt.TARGET_DELTA      = delta
    _bt.DTE_TARGET        = dte
    _bt.MAX_HOLD_DAYS     = dte                 # no live time cap → ride toward expiry
    _bt.TRAILING_STOP_PCT = _cfg.TRAILING_STOP_PCT
    _bt.EXIT_PARAMS       = EXIT
    trades, seen = [], set()
    try:
        for r in rows:
            if r["magnitude"] < mag_floor or r["confidence"] < conf_floor:
                continue
            d = r["created_at"].date()
            if regime and not regime(d):
                continue
            for tk in r["tickers"][:2]:
                if tk in ("BTC", "ETH") or not _bt.is_valid_stock_ticker(tk):
                    continue
                key = f"{d}_{tk}"
                if key in seen:
                    continue
                seen.add(key)
                sp = _bt.get_price_at(tk, r["created_at"])
                if not sp:
                    continue
                t = _bt.simulate_option_pnl(
                    tk, r["created_at"], sp, scale(r["magnitude"], r["confidence"]),
                    {"magnitude": r["magnitude"], "confidence": r["confidence"]},
                    option_type="call", exit_rule=EXIT_RULE, spread_mult=spread_mult)
                if t:
                    trades.append(t)
    finally:
        for k, v in save.items():
            setattr(_bt, k, v)
    return trades


def stats(trades):
    if not trades:
        return dict(n=0, total=0.0, mean=0.0, sharpe=0.0, win=0.0, x2=0, x4=0, pnls=[])
    s = compute_stats(trades)
    return dict(n=s["trades"], total=s["total_pnl"], mean=s["total_pnl"] / s["trades"],
                sharpe=s["sharpe"], win=s["win_rate"],
                x2=sum(1 for t in trades if t["pnl_pct"] >= 100),
                x4=sum(1 for t in trades if t["pnl_pct"] >= 300),
                pnls=[t["pnl_usd"] for t in trades])


def pctl(xs, p):
    xs = sorted(xs); i = max(0, min(len(xs) - 1, int(round(p / 100 * (len(xs) - 1)))))
    return xs[i]


def dyn_gate(r):
    return (r["magnitude"] >= _cfg.MIN_MAGNITUDE
            and r["confidence"] >= _cfg.BASE_CONFIDENCE + (1 - r["magnitude"]) * _cfg.CONFIDENCE_SLOPE)


def main():
    label, end_dt, days, cache = BULL
    rows = load_scored_from_unified(cache, end_dt, days, PROMPT)
    reg = build_regime(end_dt, days, 200)
    print(f"news_call sweep — {label}, {len(rows)} bullish {PROMPT} signals, Δ{DELTA} ATM, DTE{DTE}, "
          f"tiered_trail exit, sized like live, SPY>200d regime-gated, NET of realistic spreads\n")

    # LIVE dynamic-floor gate = what news_call WOULD fire on today (if re-enabled)
    dyn_rows = [r for r in rows if dyn_gate(r)]
    cur   = stats(run_gate(dyn_rows, reg, 0.0, 0.0))
    curff = stats(run_gate(dyn_rows, reg, 0.0, 0.0, spread_mult=0.0))
    print("  LIVE dynamic-floor gate (mag≥0.35 & conf≥dyn) — the historically money-losing setting:")
    print(f"    NET costs   : n={cur['n']:>4}  total=${cur['total']:>9,.0f}  ${cur['mean']:+,.0f}/tr  "
          f"Sh{cur['sharpe']:>5.2f}  win{cur['win']:>4.0f}%  2x+={cur['x2']} 4x+={cur['x4']}")
    print(f"    frictionless: n={curff['n']:>4}  total=${curff['total']:>9,.0f}  ${curff['mean']:+,.0f}/tr   "
          f"→ spread/cost drag = ${cur['total']-curff['total']:,.0f}\n")

    print(f"  {'gate':>16} {'n':>5} {'total':>10} {'mean/tr':>8} {'Sharpe':>7} {'win%':>6} {'2x+':>4} {'4x+':>4}")
    grid = {}
    for mf in MAG_FLOORS:
        for cf in CONF_FLOORS:
            r = stats(run_gate(rows, reg, mf, cf))
            grid[(mf, cf)] = r
            flag = "  <thin" if 0 < r["n"] < MIN_TRADES else ("  ✓net+" if r["total"] > 0 else "")
            print(f"  mag≥{mf} c≥{cf:.2f} {r['n']:>5} {('${:,.0f}'.format(r['total'])):>10} "
                  f"{('${:+,.0f}'.format(r['mean'])):>8} {r['sharpe']:>7.2f} {r['win']:>6.0f} "
                  f"{r['x2']:>4} {r['x4']:>4}{flag}", flush=True)

    # Best net-positive gate with enough volume, ranked by Sharpe
    elig = [(k, r) for k, r in grid.items() if r["n"] >= MIN_TRADES and r["total"] > 0]
    if not elig:
        print(f"\n  VERDICT: NO gate is net-positive with ≥{MIN_TRADES} trades after realistic costs "
              f"→ news_call STILL cost-fragile on {PROMPT}. Keep DISABLED.")
        return
    best_key, best = max(elig, key=lambda kv: kv[1]["sharpe"])
    print(f"\n  BEST net-positive gate: mag≥{best_key[0]} & conf≥{best_key[1]:.2f} → "
          f"n={best['n']}  total=${best['total']:,.0f}  ${best['mean']:+,.0f}/tr  "
          f"Sh{best['sharpe']:.2f}  win{best['win']:.0f}%  2x+={best['x2']} 4x+={best['x4']}")
    pnls = best["pnls"]
    boots = [sum(random.choices(pnls, k=len(pnls))) for _ in range(N_BOOT)]
    big = max(pnls)
    print(f"    BOOTSTRAP total P&L 95% CI [${pctl(boots,2.5):,.0f}, ${pctl(boots,97.5):,.0f}]  "
          f"P>0={sum(1 for b in boots if b > 0)/N_BOOT*100:.0f}%")
    print(f"    JACKKNIFE biggest single trade ${big:,.0f} ({big/best['total']*100:.0f}% of total) "
          f"→ drop it = ${best['total']-big:,.0f} {'(still +)' if best['total']-big > 0 else '(FLIPS NEG)'}")

    # Bear-window out-of-regime guard on the best gate
    blabel, bend, bdays, bcache = BEAR
    brows = load_scored_from_unified(bcache, bend, bdays, PROMPT)
    breg = build_regime(bend, bdays, 200)
    bb = stats(run_gate(brows, breg, best_key[0], best_key[1]))
    print(f"\n  Bear-window guard ({blabel}, regime-gated, best gate): "
          f"n={bb['n']}  total=${bb['total']:,.0f}  Sh{bb['sharpe']:.2f}  win{bb['win']:.0f}%")

    robust = pctl(boots, 2.5) > 0 and (best['total'] - big) > 0
    print(f"\n  VERDICT: {'net-positive AND robust (CI>0, not one-trade) → candidate to RE-ENABLE at this gate' if robust else 'net-positive but NOT robust (CI spans 0 or one-trade-driven) → more data before re-enabling'}")


if __name__ == "__main__":
    main()
