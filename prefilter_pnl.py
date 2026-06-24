"""
prefilter_pnl.py — robust P&L comparison: pre-score noise filter (and the soft-catalyst gate) ON
vs OFF, simulated as REAL news_call option trades (BS + IV-crush + spreads) over the full historical
bullish-tradeable signal set, bull + 2022-bear windows. Each signal is simulated ONCE, then we sum
over subsets — so the filter's effect = the P&L of the trades it removes. Bootstrap CIs on the
removed sets tell us, with confidence, whether dropping them helps.

Usage:  USE_YAHOO_BARS=1 .venv/bin/python -u prefilter_pnl.py
"""
import os, json, random, statistics
os.environ.setdefault("USE_YAHOO_BARS", "1")
import numpy as np
import backtest as _bt
import bot
import news_call_sweep_unified as nc
from stock_selectivity_sweep import load_scored_from_unified
from regime_filter import build_regime

TRADE_CACHE = "prefilter_pnl_trades.json"   # simulated trades, so re-runs / re-cuts are instant

MAG, CONF = 0.75, 0.70                      # live news_call gate
DELTA, DTE, HOLD, TRAIL = 0.40, 10, 3, 0.40
POS = 650.0                                 # flat live sizing (NONMAG_SIZE_FRAC×MAX)


def simulate_all(window):
    label, end_dt, days, cache = window
    rows = load_scored_from_unified(cache, end_dt, days, "unified_v1")
    reg = build_regime(end_dt, days, 200)
    save = {k: getattr(_bt, k) for k in ("TARGET_DELTA", "DTE_TARGET", "MAX_HOLD_DAYS", "TRAILING_STOP_PCT")}
    _bt.TARGET_DELTA, _bt.DTE_TARGET, _bt.MAX_HOLD_DAYS, _bt.TRAILING_STOP_PCT = DELTA, DTE, HOLD, TRAIL
    out, seen = [], set()
    try:
        for r in rows:
            if r["magnitude"] < MAG or r["confidence"] < CONF or not reg(r["created_at"].date()):
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
                    h = r["headline"]
                    out.append({"pnl": t["pnl_usd"], "pre": bool(bot._prescore_noise(h)),
                                "soft": bool(bot._soft_catalyst_hit(h, "")), "h": h})
    finally:
        for k, v in save.items():
            setattr(_bt, k, v)
    return out


def boot_ci(pnls, n=5000):
    if not pnls:
        return (0, 0, 0)
    rng = random.Random(0)
    tot = [sum(rng.choices(pnls, k=len(pnls))) for _ in range(n)]
    return sum(pnls), np.percentile(tot, 5), np.percentile(tot, 95)


def stat(pnls):
    n = len(pnls)
    if not n:
        return 0, 0, 0, 0
    mean = sum(pnls) / n
    sd = statistics.pstdev(pnls) if n > 1 else 0
    sharpe = mean / sd if sd else 0          # per-trade Sharpe (comparable across variants)
    win = sum(1 for p in pnls if p > 0) / n * 100
    return n, sum(pnls), sharpe, win


def main():
    if os.path.exists(TRADE_CACHE):
        trades = json.load(open(TRADE_CACHE))
        print(f"loaded {len(trades)} cached simulated trades\n")
    else:
        print("Simulating news_call option P&L over all bullish-tradeable signals (bull + 2022-bear)…", flush=True)
        trades = simulate_all(nc.BULL) + simulate_all(nc.BEAR)
        json.dump(trades, open(TRADE_CACHE, "w"))
        print(f"  {len(trades)} simulated trades (cached)\n", flush=True)

    def show(name, keep):
        pnls = [t["pnl"] for t in trades if keep(t)]
        if not pnls:
            print(f"  {name:34} n=0"); return
        n, _, sh, win = stat(pnls); tot, lo, hi = boot_ci(pnls)
        print(f"  {name:34} n={n:>4}  P&L ${tot:>+8,.0f}  [boot ${lo:>+8,.0f}, ${hi:>+8,.0f}]  "
              f"Sharpe {sh:>5.2f}  win {win:>4.1f}%")

    print("══ VARIANTS (same trades, different inclusion) ══")
    show("BASELINE (current: trade all)", lambda t: True)
    show("PRE-FILTER on (drop noise)",    lambda t: not t["pre"])
    show("PRE-FILTER + SOFT-CATALYST",    lambda t: not t["pre"] and not t["soft"])

    print("\n══ WHAT EACH FILTER REMOVES (the decision-relevant P&L) ══")
    for name, keep in (("removed by PRE-FILTER", lambda t: t["pre"]),
                       ("removed by SOFT-CATALYST", lambda t: t["soft"] and not t["pre"]),
                       ("removed by EITHER", lambda t: t["pre"] or t["soft"])):
        pnls = [t["pnl"] for t in trades if keep(t)]
        if not pnls:
            print(f"  {name:26} n=0"); continue
        tot, lo, hi = boot_ci(pnls)
        verdict = "✓ clearly LOSES (drop helps)" if hi < 0 else ("~ ambiguous (CI spans 0)" if lo < 0 else "✗ profitable (don't drop)")
        print(f"  {name:26} n={len(pnls):>4}  P&L ${tot:>+8,.0f}  [boot ${lo:>+8,.0f}, ${hi:>+8,.0f}]  {verdict}")
    print("\n  A removed set 'clearly LOSES' (95% CI < $0) ⇒ dropping it is a statistically-supported win.")


if __name__ == "__main__":
    main()
