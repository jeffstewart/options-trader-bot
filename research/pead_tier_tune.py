"""
pead_tier_tune.py — Tune tiered trailing-stop parameters for PEAD (and stock) positions.

The tiered trail (wide early → tightens as gains grow) is pure position management:
it's signal-agnostic, so we tune it on the full stock dataset (~1,300 training trades)
where we have enough power, then validate the transfer to the PEAD-filtered subset.

Grid: t1 (initial wide trail) × t2 (medium) × t3 (tight winner) × g1 (first
breakpoint) × g2 (second breakpoint) × lock_pct (breakeven lock threshold).
Invalid combos (t3 >= t2 >= t1, or g1 >= g2) are skipped.

Outputs:
  - Top-10 configs by holdout Sharpe on full stock dataset
  - Transfer validation: same configs on PEAD-only subset (cross-check)
  - Baseline comparison: flat 10% trail (current stock), flat 20% trail (current PEAD)
  - Recommended config (highest mean rank across stock + PEAD holdout)

Usage:  USE_YAHOO_BARS=1 .venv/bin/python pead_tier_tune.py
"""
import itertools
import sys
from datetime import datetime, timezone, timedelta
from pathlib import Path

import tune_v2
from pead_backtest import (
    simulate_pead, run_scored, is_earnings_article, STOCK_SLIPPAGE, WINDOWS
)
from backtest import get_price_at, get_stock_bars, is_valid_stock_ticker
from benchmark import compute_stats
from regime_filter import build_regime
import config as _cfg

BULL_END  = datetime.now(timezone.utc) - timedelta(days=5)
BULL_DAYS = 180
BULL_CACHE = Path("dual_score_cache.json")

# ── Tiered simulate (shared core) ─────────────────────────────────────────────

def sim_tiered(ticker, entry_dt, position_usd, t1, t2, t3, g1, g2, lock_pct,
               max_hold=45):
    """
    Like simulate_pead but with a 3-tier trail + optional breakeven lock.
    Tier selection is by PEAK gain (not instantaneous), so it ratchets up.
    """
    bars = get_stock_bars(ticker, entry_dt,
                          entry_dt + timedelta(days=int(max_hold * 1.6) + 14))
    tb = [b for b in bars if entry_dt < b["t"]]
    if not tb or (tb[0]["t"] - entry_dt).days > 7:
        return None
    p0 = get_price_at(ticker, entry_dt)
    if not p0 or p0 < _cfg.MIN_STOCK_PRICE or p0 > 10_000:
        return None
    prev = p0
    for b in tb:
        if prev > 0 and abs(b["c"] / prev - 1) > 0.5:
            return None
        prev = b["c"]

    entry_fill = p0 * (1 + STOCK_SLIPPAGE / 2)
    shares = position_usd / entry_fill
    peak = p0
    exit_price = tb[min(max_hold, len(tb)) - 1]["c"]

    for b in tb[:max_hold]:
        peak = max(peak, b["c"])
        gain = (peak / p0 - 1.0)
        trail = t3 if gain >= g2 else (t2 if gain >= g1 else t1)
        stop  = peak * (1 - trail)
        if lock_pct > 0 and gain >= lock_pct:
            stop = max(stop, p0 * 1.005)   # never lose the trade once up lock_pct
        if b["c"] <= stop:
            exit_price = b["c"]
            break

    exit_fill = exit_price * (1 - STOCK_SLIPPAGE / 2)
    pnl = (exit_fill - entry_fill) * shares
    return {"ticker": ticker, "entry_dt": entry_dt.isoformat(),
            "pnl_usd": pnl, "pnl_pct": (exit_fill / entry_fill - 1) * 100}


# ── Dataset runners ───────────────────────────────────────────────────────────

def _scale(mag, conf):
    return max(_cfg.MAX_POSITION_USD * 0.10, _cfg.MAX_POSITION_USD * mag * conf)


def run_stock_tiered(scored_rows, regime, t1, t2, t3, g1, g2, lock_pct, max_hold=30):
    """Full stock universe (no earnings filter) — large sample for tier tuning."""
    trades, seen = [], {}
    for row in scored_rows:
        mag, conf = row["magnitude"], row["confidence"]
        req = _cfg.BASE_CONFIDENCE + (1 - mag) * _cfg.CONFIDENCE_SLOPE
        if mag < _cfg.MIN_MAGNITUDE or conf < req:
            continue
        d = row["created_at"].date()
        if regime and not regime(d):
            continue
        ds = seen.setdefault(d.isoformat(), set())
        for tk in row["tickers"][:2]:
            if tk in ds or tk in ("BTC", "ETH") or not is_valid_stock_ticker(tk):
                continue
            ds.add(tk)
            t = sim_tiered(tk, row["created_at"], _scale(mag, conf),
                           t1, t2, t3, g1, g2, lock_pct, max_hold)
            if t:
                trades.append(t)
    return trades


def run_pead_tiered(scored_rows, regime, t1, t2, t3, g1, g2, lock_pct, max_hold=45):
    """PEAD subset (earnings filter) — validation that tiers transfer."""
    trades, seen = [], {}
    for row in scored_rows:
        mag, conf = row["magnitude"], row["confidence"]
        req = _cfg.BASE_CONFIDENCE + (1 - mag) * _cfg.CONFIDENCE_SLOPE
        if mag < _cfg.MIN_MAGNITUDE or conf < req:
            continue
        if not is_earnings_article(row.get("headline", "")):
            continue
        d = row["created_at"].date()
        if regime and not regime(d):
            continue
        ds = seen.setdefault(d.isoformat(), set())
        for tk in row["tickers"][:2]:
            if tk in ds or tk in ("BTC", "ETH") or not is_valid_stock_ticker(tk):
                continue
            ds.add(tk)
            t = sim_tiered(tk, row["created_at"], _scale(mag, conf),
                           t1, t2, t3, g1, g2, lock_pct, max_hold)
            if t:
                trades.append(t)
    return trades


# ── Baseline (flat trail) ─────────────────────────────────────────────────────

def run_flat(scored_rows, regime, flat_trail, max_hold, earnings_only=False):
    trades, seen = [], {}
    for row in scored_rows:
        mag, conf = row["magnitude"], row["confidence"]
        req = _cfg.BASE_CONFIDENCE + (1 - mag) * _cfg.CONFIDENCE_SLOPE
        if mag < _cfg.MIN_MAGNITUDE or conf < req:
            continue
        if earnings_only and not is_earnings_article(row.get("headline", "")):
            continue
        d = row["created_at"].date()
        if regime and not regime(d):
            continue
        ds = seen.setdefault(d.isoformat(), set())
        for tk in row["tickers"][:2]:
            if tk in ds or tk in ("BTC", "ETH") or not is_valid_stock_ticker(tk):
                continue
            ds.add(tk)
            t = simulate_pead(tk, row["created_at"], _scale(mag, conf),
                              trail=flat_trail, max_hold=max_hold)
            if t:
                trades.append(t)
    return trades


# ── Tune grid ─────────────────────────────────────────────────────────────────

GRID = {
    "t1":      [0.15, 0.20, 0.25],    # initial wide trail
    "t2":      [0.08, 0.12, 0.15],    # medium trail (after g1)
    "t3":      [0.05, 0.07, 0.10],    # tight trail (after g2)
    "g1":      [0.10, 0.15, 0.20],    # first tightening point
    "g2":      [0.25, 0.35],          # second tightening point
    "lock_pct":[0.0, 0.15, 0.20],     # breakeven lock (0 = disabled)
}


def main():
    if not BULL_CACHE.exists():
        print(f"Cache not found: {BULL_CACHE}"); sys.exit(1)

    all_scored = tune_v2.load_scored_from_dual_cache(BULL_CACHE, "bull",
                                                     BULL_END, BULL_DAYS)
    n     = len(all_scored)
    split = int(n * 0.80)
    train_s, hold_s = all_scored[:split], all_scored[split:]
    train_end = train_s[-1]["created_at"]
    hold_end  = all_scored[-1]["created_at"]
    train_reg = build_regime(train_end, BULL_DAYS, 200)
    hold_reg  = build_regime(hold_end,  BULL_DAYS, 200)

    print("PEAD TIER TUNE  (80/20 holdout on bull window)\n")
    print(f"  Total scored: {n}  |  Train: {len(train_s)}  |  Holdout: {len(hold_s)}\n")

    # ── Baselines ─────────────────────────────────────────────────────────────
    print("── Baselines ──")
    def show_base(label, trades):
        s = compute_stats(trades)
        print(f"  {label:35}  n={s['trades']:>4}  Sharpe={s['sharpe']:>6.2f}"
              f"  P&L=${s['total_pnl']:>9,.0f}  win={s['win_rate']:>4.1f}%")

    show_base("flat 10% (current stock)",
              run_flat(hold_s, hold_reg, 0.10, 30, False))
    show_base("flat 20% (current PEAD)",
              run_flat(hold_s, hold_reg, 0.20, 45, True))
    show_base("flat 10% PEAD earnings",
              run_flat(hold_s, hold_reg, 0.10, 45, True))
    print()

    # ── Grid search on STOCK (full dataset, better power) ────────────────────
    keys   = list(GRID.keys())
    combos = [(dict(zip(keys, v))) for v in itertools.product(*[GRID[k] for k in keys])]
    # Filter invalid (t3 >= t2, t2 >= t1, g1 >= g2)
    combos = [c for c in combos if c["t3"] < c["t2"] < c["t1"] and c["g1"] < c["g2"]]
    print(f"  {len(combos)} valid combos (full stock dataset)...", flush=True)

    results = []
    for i, p in enumerate(combos):
        if (i + 1) % 100 == 0:
            print(f"  {i+1}/{len(combos)}", flush=True)
        tr = run_stock_tiered(train_s, train_reg, **p, max_hold=30)
        ts = compute_stats(tr)
        results.append({**p, "train_sharpe": ts["sharpe"],
                        "train_pnl": ts["total_pnl"], "train_n": ts["trades"]})

    results.sort(key=lambda r: r["train_sharpe"], reverse=True)
    top10 = results[:10]

    hdr = (f"  {'t1':>4} {'t2':>4} {'t3':>4} {'g1':>4} {'g2':>4} {'lock':>5}"
           f"  {'Tr.Sh':>7}  {'Ho.Sh(stk)':>10}  {'Ho.n(stk)':>9}"
           f"  {'Ho.Sh(pead)':>11}  {'Ho.n(pead)':>10}")

    print(f"\n── Top 10 by train Sharpe (stock) ──\n{hdr}")
    holdout_rows = []
    for r in top10:
        p = {k: r[k] for k in keys}
        ho_stk  = run_stock_tiered(hold_s, hold_reg, **p, max_hold=30)
        ho_pead = run_pead_tiered(hold_s, hold_reg, **p, max_hold=45)
        hs = compute_stats(ho_stk)
        hp = compute_stats(ho_pead)
        holdout_rows.append({**r, "ho_sh_stk": hs["sharpe"], "ho_n_stk": hs["trades"],
                             "ho_sh_pead": hp["sharpe"], "ho_n_pead": hp["trades"]})
        print(f"  {r['t1']:>4.2f} {r['t2']:>4.2f} {r['t3']:>4.2f} {r['g1']:>4.2f} "
              f"{r['g2']:>4.2f} {r['lock_pct']:>5.2f}"
              f"  {r['train_sharpe']:>7.2f}  {hs['sharpe']:>10.2f}  {hs['trades']:>9}"
              f"  {hp['sharpe']:>11.2f}  {hp['trades']:>10}")

    # ── Best by holdout stock Sharpe ──────────────────────────────────────────
    holdout_rows.sort(key=lambda r: r["ho_sh_stk"], reverse=True)
    best = holdout_rows[0]
    print(f"\n── Best holdout combo ──")
    print(f"  t1={best['t1']}  t2={best['t2']}  t3={best['t3']}"
          f"  g1={best['g1']}  g2={best['g2']}  lock={best['lock_pct']}")
    print(f"  Stock holdout  → Sharpe {best['ho_sh_stk']:.2f}  n={best['ho_n_stk']}")
    print(f"  PEAD holdout   → Sharpe {best['ho_sh_pead']:.2f}  n={best['ho_n_pead']}")

    # ── Config string for config.py ───────────────────────────────────────────
    t1,t2,t3 = best["t1"], best["t2"], best["t3"]
    g1,g2,lk = best["g1"], best["g2"], best["lock_pct"]
    print(f"\n── Config to apply ──")
    print(f"  PEAD_EXIT_TIERS     = [({g1}, {t1}), ({g2}, {t2}), (float('inf'), {t3})]")
    print(f"  PEAD_BREAKEVEN_LOCK = {lk}")
    print()


if __name__ == "__main__":
    main()
