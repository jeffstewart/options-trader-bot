"""
regime_bypass_option_pnl.py — confirm the regime-bypass finding on ACTUAL option P&L, not stock
returns. Simulates REAL news_call long-call trades (BS + IV-crush + spreads + the live exit logic,
Δ0.40/DTE10/hold3/trail0.40, $650 sizing) for every bullish news_call-eligible signal, tags each
with the point-in-time live regime (SPY 200d SMA AND 3d momentum) + magnitude, and compares option
P&L by regime × magnitude. Then simulates policies: current (calls only in up-regime) vs a
high-conviction bypass (allow down-regime calls when mag ≥ threshold), with bootstrap CIs on the
ADDED down-regime set — the rigorous test of whether those bypass trades make money on options.

Unlike prefilter_pnl.py (which skips down-regime), this simulates BOTH regimes. Cached.

Usage:  USE_YAHOO_BARS=1 .venv/bin/python -u regime_bypass_option_pnl.py
"""
import os, json, random
os.environ.setdefault("USE_YAHOO_BARS", "1")
import numpy as np
import backtest as _bt
import news_call_sweep_unified as nc
from stock_selectivity_sweep import load_scored_from_unified
from regime_bypass_backtest import build_regime, regime_for

CACHE = "regime_bypass_option_trades.json"
MAG, CONF = 0.75, 0.70                      # live news_call gate
DELTA, DTE, HOLD, TRAIL = 0.40, 10, 3, 0.40
POS = 650.0


def simulate(window, sdays, reg):
    label, end_dt, days, cache = window
    rows = load_scored_from_unified(cache, end_dt, days, "unified_v1")
    save = {k: getattr(_bt, k) for k in ("TARGET_DELTA", "DTE_TARGET", "MAX_HOLD_DAYS", "TRAILING_STOP_PCT")}
    _bt.TARGET_DELTA, _bt.DTE_TARGET, _bt.MAX_HOLD_DAYS, _bt.TRAILING_STOP_PCT = DELTA, DTE, HOLD, TRAIL
    out, seen = [], set()
    n_seen = 0
    try:
        for r in rows:
            if r["magnitude"] < MAG or r["confidence"] < CONF:
                continue
            up = regime_for(r["created_at"].date(), sdays, reg)
            if up is None:
                continue
            for tk in r["tickers"][:2]:
                key = f"{r['created_at'].date()}_{tk}"
                if key in seen or tk in ("BTC", "ETH") or not _bt.is_valid_stock_ticker(tk):
                    continue
                seen.add(key)
                sp = _bt.get_price_at(tk, r["created_at"])
                if not sp:
                    continue
                t = _bt.simulate_option_pnl(tk, r["created_at"], sp, POS,
                                            {"magnitude": r["magnitude"], "confidence": r["confidence"]},
                                            option_type="call", exit_rule="trail_premium", spread_mult=1.0)
                if t:
                    out.append({"pnl": t["pnl_usd"], "mag": r["magnitude"], "up": bool(up), "win": window[0]})
                    n_seen += 1
                    if n_seen % 25 == 0:
                        print(f"    {label}: {n_seen} simulated…", flush=True)
    finally:
        for k, v in save.items():
            setattr(_bt, k, v)
    return out


def boot(pnls, n=5000):
    if not pnls:
        return (0, 0, 0)
    rng = random.Random(0)
    tot = [sum(rng.choices(pnls, k=len(pnls))) for _ in range(n)]
    return sum(pnls), float(np.percentile(tot, 5)), float(np.percentile(tot, 95))


def cell(pnls):
    if not pnls:
        return "        —         "
    n = len(pnls); tot = sum(pnls); avg = tot / n
    win = sum(1 for p in pnls if p > 0) / n * 100
    return f"${tot:>+7.0f} avg${avg:>+5.0f} {win:>3.0f}%w n={n:<3}"


def main():
    sdays, reg = build_regime()
    print(f"regime mapped: {len(sdays)} days ({sum(reg.values())} up / {len(reg)-sum(reg.values())} down)\n", flush=True)

    if os.path.exists(CACHE):
        trades = json.load(open(CACHE))
        print(f"loaded {len(trades)} cached simulated option trades\n")
    else:
        print("Simulating news_call option P&L over ALL bullish news_call signals (both regimes)…", flush=True)
        trades = simulate(nc.BULL, sdays, reg)
        try:
            trades += simulate(nc.BEAR, sdays, reg)
        except Exception as e:
            print(f"  (bear window skipped: {repr(e)[:80]})", flush=True)
        json.dump(trades, open(CACHE, "w"))
        print(f"  {len(trades)} simulated option trades (cached)\n", flush=True)

    tiers = [("0.75–0.85", 0.75, 0.85), ("0.85–1.00", 0.85, 1.01)]
    windows = sorted(set(t["win"] for t in trades))
    for win in windows + ["ALL"]:
        sub = trades if win == "ALL" else [t for t in trades if t["win"] == win]
        nup = sum(1 for t in sub if t["up"]); ndn = len(sub) - nup
        print(f"\n  ══ {win}  ({len(sub)} trades · {nup} up-regime / {ndn} down-regime) ══")
        print(f"  news_call OPTION P&L by regime × magnitude  (total · avg/trade · win% · n)")
        print(f"  {'magnitude':12} {'UP-regime':>26}   {'DOWN-regime':>26}")
        for name, lo, h in tiers:
            u = [t["pnl"] for t in sub if t["up"] and lo <= t["mag"] < h]
            d = [t["pnl"] for t in sub if not t["up"] and lo <= t["mag"] < h]
            print(f"  {name:12} {cell(u):>26}   {cell(d):>26}")
        # bear-market acid test: down-regime ≥0.85 calls (what the bypass would allow)
        add = [t["pnl"] for t in sub if (not t["up"]) and t["mag"] >= 0.85]
        if add:
            s, lo2, hi2 = boot(add)
            print(f"  → bypass-eligible (down ≥0.85): n={len(add)} total ${s:+.0f} avg ${s/len(add):+.0f} "
                  f"CI[${lo2:+.0f},${hi2:+.0f}] {'SIG>0' if lo2>0 else 'spans0'}")

    print(f"\n  ── policy comparison (real option P&L, $ {POS:.0f}/trade) ──")
    print(f"  {'policy':40} {'trades':>6} {'total P&L':>11} {'avg':>7} {'win%':>5}")
    pol = [
        ("Current (calls only in UP-regime)",  lambda m, up: up),
        ("No regime filter (all calls)",       lambda m, up: True),
        ("Bypass: UP all + DOWN if mag≥0.85",  lambda m, up: up or m >= 0.85),
        ("Bypass: UP all + DOWN if mag≥0.90",  lambda m, up: up or m >= 0.90),
    ]
    base = None
    for label, keep in pol:
        pnls = [t["pnl"] for t in trades if keep(t["mag"], t["up"])]
        tot = sum(pnls); avg = tot / len(pnls) if pnls else 0
        win = sum(1 for p in pnls if p > 0) / len(pnls) * 100 if pnls else 0
        if base is None:
            base = tot
        d = f"  (Δ {tot-base:+.0f})" if label != pol[0][0] else ""
        print(f"  {label:40} {len(pnls):>6} ${tot:>+8.0f}{d:>11} ${avg:>+5.0f} {win:>4.0f}%")

    # rigorous test: is the ADDED down-regime ≥0.85 option P&L significantly > 0?
    add = [t["pnl"] for t in trades if (not t["up"]) and t["mag"] >= 0.85]
    s, lo, hi = boot(add)
    print(f"\n  ADDED by ≥0.85 bypass (down-regime ≥0.85 calls): n={len(add)} · total ${s:+.0f} · "
          f"boot 90% CI [${lo:+.0f}, ${hi:+.0f}]  → {'SIGNIFICANT >0' if lo > 0 else 'CI spans 0'}")
    print("  Positive CI entirely above 0 = the bypass trades make money on real option contracts, not just stock.")


if __name__ == "__main__":
    main()
