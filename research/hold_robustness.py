"""
hold_robustness.py — is lotto hold 3d > 7d ROBUST, or a small-sample artifact?

Paired reanalysis of the same bull-meltup lotto trades (DTE14, Δ0.25, tiered+3× cap), aligned
per signal so hold3 vs hold7 is a paired comparison. Same Tier-1 discipline we gave the 3× cap:
  1. BOOTSTRAP 20k paired resamples — fraction where hold3 beats hold7 on P&L + 95% CI.
  2. JACKKNIFE — drop the top-K winners; does the ranking survive?

Usage:  USE_YAHOO_BARS=1 .venv/bin/python hold_robustness.py
"""
import os, statistics, random
os.environ.setdefault("USE_YAHOO_BARS", "1")
from datetime import datetime, timezone, timedelta
from pathlib import Path

import backtest as _bt
import tune_v2, config as cfg
from regime_filter import build_regime

INF = float("inf")
DELTA, DTE, POS = 0.25, 14, cfg.LOTTO_POSITION_USD
GATE_MAG, GATE_CONF = cfg.LOTTO_MIN_MAGNITUDE, cfg.LOTTO_MIN_CONFIDENCE
EXIT = {"tiers": [(1.0, 0.40), (3.0, 0.30), (INF, 0.20)], "hard_target": 3.0}
END = datetime.now(timezone.utc) - timedelta(days=5)
HOLDS = (3, 7)   # A vs B


def collect():
    scored = tune_v2.load_scored_from_dual_cache(Path("dual_score_cache.json"), "bull", END, 180)
    gated = [r for r in scored if r["magnitude"] >= GATE_MAG and r["confidence"] >= GATE_CONF]
    reg = build_regime(END, 180, 200)
    save = {k: getattr(_bt, k) for k in ("TARGET_DELTA", "DTE_TARGET", "MAX_HOLD_DAYS", "TRAILING_STOP_PCT", "EXIT_PARAMS")}
    _bt.TARGET_DELTA, _bt.DTE_TARGET = DELTA, DTE
    _bt.TRAILING_STOP_PCT, _bt.EXIT_PARAMS = 0.30, EXIT
    rows, seen = [], set()
    try:
        for r in gated:
            d = r["created_at"].date()
            if reg and not reg(d):
                continue
            for tk in r["tickers"][:1]:
                if tk in ("BTC", "ETH") or not _bt.is_valid_stock_ticker(tk):
                    continue
                k = f"{d}_{tk}"
                if k in seen:
                    continue
                seen.add(k)
                sp = _bt.get_price_at(tk, r["created_at"])
                if not sp:
                    continue
                pos_usd = max(POS * 0.5, POS * r["magnitude"] * r["confidence"])
                sig = {"magnitude": r["magnitude"], "confidence": r["confidence"]}
                res, ok = {}, True
                for h in HOLDS:
                    _bt.MAX_HOLD_DAYS = h
                    t = _bt.simulate_option_pnl(tk, r["created_at"], sp, pos_usd, sig,
                                                option_type="call", exit_rule="tiered_profit", spread_mult=1.0)
                    if not t:
                        ok = False
                        break
                    res[h] = t["pnl_usd"]
                if ok:
                    rows.append(res)
    finally:
        for kk, vv in save.items():
            setattr(_bt, kk, vv)
    return rows


def main():
    rows = collect()
    n = len(rows)
    A, B = HOLDS
    tA, tB = sum(r[A] for r in rows), sum(r[B] for r in rows)
    print(f"═══ HOLD {A}d vs {B}d ROBUSTNESS  (n={n} paired lotto trades, DTE{DTE}/3×cap) ═══\n")
    print(f"  full sample: hold{A}d = ${tA:+,.0f}   hold{B}d = ${tB:+,.0f}   diff = ${tA-tB:+,.0f}\n")

    # bootstrap
    diffs = [r[A] - r[B] for r in rows]
    rng = random.Random(42)
    boot, wins = [], 0
    for _ in range(20000):
        idx = [rng.randrange(n) for _ in range(n)]
        d = sum(diffs[i] for i in idx)
        boot.append(d)
        if d > 0:
            wins += 1
    boot.sort()
    lo, hi = boot[500], boot[19500]
    pct = wins / 20000 * 100
    print(f"  [BOOTSTRAP] hold{A} beats hold{B} in {pct:.1f}% of 20k resamples")
    print(f"    P&L diff 95% CI: [${lo:+,.0f}, ${hi:+,.0f}]")
    print(f"    → {'ROBUST' if pct > 90 else 'WEAK/suggestive' if pct > 75 else 'NOISE'}\n")

    # jackknife — drop the top-K winners (by best outcome across the two holds)
    order = sorted(range(n), key=lambda i: max(rows[i][A], rows[i][B]), reverse=True)
    print(f"  [JACKKNIFE] drop top-K winners:")
    for K in (0, 1, 2, 3, 5):
        keep = set(order[K:])
        sa = sum(rows[i][A] for i in range(n) if i in keep)
        sb = sum(rows[i][B] for i in range(n) if i in keep)
        print(f"    drop top {K}:  hold{A}=${sa:>8,.0f}  hold{B}=${sb:>8,.0f}  diff=${sa-sb:>+8,.0f}  → hold{A if sa>sb else B} wins")


if __name__ == "__main__":
    main()
