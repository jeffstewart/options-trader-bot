"""
tier1_robustness.py — is the ratchet+8x-cap edge over the simple 3x-cap REAL or noise?

Three reanalyses of the SAME 114 bull-meltup lotto trades (no new data):
  1. BOOTSTRAP — resample trades w/ replacement 20k×; in what fraction does ratchet+8x beat
     3x-cap on total P&L? (paired: same resampled indices for both). + 95% CI on the diff.
  2. JACKKNIFE — drop the top-K biggest winners; does the ranking survive, or does it hinge on
     one or two monster trades?
  3. PARAMETER STABILITY — sweep the cap level around each choice (simple cap 2/3/4x; ratchet
     cap 6/8/10x/none). A smooth PLATEAU = robust; a sharp SPIKE = overfit to noise.

Per-trade P&L is collected ALIGNED per signal (same entry, only the exit rule differs) so the
comparisons are paired.

Usage:  USE_YAHOO_BARS=1 .venv/bin/python tier1_robustness.py
"""
import os, statistics, random
os.environ.setdefault("USE_YAHOO_BARS", "1")
from datetime import datetime, timezone, timedelta
from pathlib import Path

import backtest as _bt
import tune_v2, config as cfg
from regime_filter import build_regime

INF = float("inf")
DELTA, DTE, HOLD, POS = 0.25, 14, 7, cfg.LOTTO_POSITION_USD
GATE_MAG, GATE_CONF = cfg.LOTTO_MIN_MAGNITUDE, cfg.LOTTO_MIN_CONFIDENCE
END = datetime.now(timezone.utc) - timedelta(days=5)
RAT = [(1.0, 0.35), (2.0, 0.25), (4.0, 0.15), (INF, 0.10)]          # ratchet tiers
LIV = [(1.0, 0.40), (3.0, 0.30), (INF, 0.20)]                       # live tiers

CONFIGS = {                                                          # name -> exit params
    "LIVE":      {"tiers": LIV},
    "cap2x":     {"tiers": LIV, "hard_target": 2.0},
    "cap3x":     {"tiers": LIV, "hard_target": 3.0},
    "cap4x":     {"tiers": LIV, "hard_target": 4.0},
    "rat_cap6":  {"tiers": RAT, "hard_target": 6.0},
    "rat_cap8":  {"tiers": RAT, "hard_target": 8.0},                # the candidate
    "rat_cap10": {"tiers": RAT, "hard_target": 10.0},
    "rat_nocap": {"tiers": RAT},
}
A, B = "cap3x", "LIVE"   # the actual change we'd ship: 3x cap vs current live exit


def collect():
    scored = tune_v2.load_scored_from_dual_cache(Path("dual_score_cache.json"), "bull", END, 180)
    gated = [r for r in scored if r["magnitude"] >= GATE_MAG and r["confidence"] >= GATE_CONF]
    reg = build_regime(END, 180, 200)
    save = {k: getattr(_bt, k) for k in
            ("TARGET_DELTA", "DTE_TARGET", "MAX_HOLD_DAYS", "TRAILING_STOP_PCT", "EXIT_PARAMS")}
    _bt.TARGET_DELTA, _bt.DTE_TARGET, _bt.MAX_HOLD_DAYS = DELTA, DTE, HOLD
    _bt.TRAILING_STOP_PCT = 0.30
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
                for name, params in CONFIGS.items():
                    _bt.EXIT_PARAMS = params
                    t = _bt.simulate_option_pnl(tk, r["created_at"], sp, pos_usd, sig,
                                                option_type="call", exit_rule="tiered_profit", spread_mult=1.0)
                    if not t:
                        ok = False
                        break
                    res[name] = t["pnl_usd"]
                if ok:
                    res["_peak_pct"] = max(0.0, sp)  # placeholder; tail id below uses max pnl
                    rows.append(res)
    finally:
        for kk, vv in save.items():
            setattr(_bt, kk, vv)
    return rows


def total(rows, name):
    return sum(r[name] for r in rows)


def sharpe(vals):
    sd = statistics.pstdev(vals)
    return statistics.mean(vals) / sd if sd else 0.0


def main():
    rows = collect()
    n = len(rows)
    print(f"═══ TIER-1 ROBUSTNESS — ratchet+8x vs 3x-cap  (n={n} bull-meltup lotto trades) ═══\n")

    print("  config totals (full sample):")
    for name in CONFIGS:
        v = [r[name] for r in rows]
        print(f"    {name:10} P&L=${total(rows, name):>8,.0f}  Sharpe={sharpe(v):>5.2f}")

    # ── 1. BOOTSTRAP (paired) ──────────────────────────────────────────────
    print(f"\n  [1] BOOTSTRAP — {A} vs {B}, 20,000 paired resamples:")
    diffs = [rows[i][A] - rows[i][B] for i in range(n)]
    rng = random.Random(42)
    boot_diff, a_wins, a_sh_wins = [], 0, 0
    for _ in range(20000):
        idx = [rng.randrange(n) for _ in range(n)]
        dsum = sum(diffs[i] for i in idx)
        boot_diff.append(dsum)
        if dsum > 0:
            a_wins += 1
        if sharpe([rows[i][A] for i in idx]) > sharpe([rows[i][B] for i in idx]):
            a_sh_wins += 1
    boot_diff.sort()
    lo, hi = boot_diff[int(0.025 * 20000)], boot_diff[int(0.975 * 20000)]
    print(f"    P&L diff ({A}−{B}): observed ${total(rows,A)-total(rows,B):>+,.0f}  "
          f"95% CI [${lo:>+,.0f}, ${hi:>+,.0f}]")
    print(f"    {A} beats {B} on P&L  in {a_wins/20000*100:>5.1f}% of resamples")
    print(f"    {A} beats {B} on Sharpe in {a_sh_wins/20000*100:>5.1f}% of resamples")
    print(f"    → {'ROBUST' if a_wins/20000 > 0.90 else 'NOISE / not distinguishable' if a_wins/20000 < 0.75 else 'WEAK / suggestive'}")

    # ── 2. JACKKNIFE — drop the top-K winners (by best outcome across configs) ──
    print(f"\n  [2] JACKKNIFE — drop the top-K biggest winners, recompute {A} vs {B}:")
    order = sorted(range(n), key=lambda i: max(rows[i][c] for c in CONFIGS), reverse=True)
    for K in (0, 1, 2, 3, 5):
        keep = set(order[K:])
        sub = [rows[i] for i in range(n) if i in keep]
        ta, tb = total(sub, A), total(sub, B)
        print(f"    drop top {K}:  {A}=${ta:>8,.0f}  {B}=${tb:>8,.0f}  "
              f"diff=${ta-tb:>+8,.0f}  → {A if ta>tb else B} wins")

    # ── 3. PARAMETER STABILITY ─────────────────────────────────────────────
    print(f"\n  [3] PARAMETER STABILITY (plateau = robust, spike = overfit):")
    print(f"    simple cap level:  " + "  ".join(
        f"{c.replace('cap',''):>4}=${total(rows,c):>7,.0f}" for c in ("cap2x", "cap3x", "cap4x")))
    print(f"    ratchet cap level: " + "  ".join(
        f"{c.replace('rat_',''):>5}=${total(rows,c):>7,.0f}" for c in ("rat_cap6", "rat_cap8", "rat_cap10", "rat_nocap")))


if __name__ == "__main__":
    main()
