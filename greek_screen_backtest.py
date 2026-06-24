"""
greek_screen_backtest.py — does screening lotto picks on ENTRY GREEKS improve P&L?

For every high-conviction lotto signal (mag≥0.70 & conf≥0.85, regime-gated), we recompute
the entry option the live bot would buy (Δ0.25, ~14 DTE, 7-day hold) and its entry Greeks
from the SAME Black-Scholes model the sim prices with — so the Greeks are self-consistent
with the simulated P&L. Then we (A) compare each Greek's distribution for winners vs losers,
and (B) sweep threshold screens and measure P&L/Sharpe/multibaggers of the survivors vs the
unscreened baseline.

HONEST CAVEATS (read before trusting):
  • Backtest IV = realized_vol × NEWS_IV_MULTIPLIER (a FLAT news premium). So an "IV screen"
    here ≈ a realized-vol screen; it can't see the real, skewed, about-to-crush news premium
    that lives in MARKET IV (now logged live → forward calibration).
  • With Δ and DTE FIXED, gamma/theta/vega are deterministic in (S, IV) → the three screens
    are highly COLLINEAR (all proxy the name's vol). Treat them as one question, not three.
  • Delta-band screening is NOT testable here: we pick the strike to hit Δ0.25 exactly, so
    actual≈target with no variation. That screen needs LIVE data (real delta vs target).

Usage:  USE_YAHOO_BARS=1 .venv/bin/python greek_screen_backtest.py
"""
import os, statistics
os.environ.setdefault("USE_YAHOO_BARS", "1")
from datetime import datetime, timezone, timedelta
from pathlib import Path

import backtest as _bt
import tune_v2
import config as cfg
from pricing import strike_for_delta, bs_call_gamma, bs_call_theta, bs_call_vega
from benchmark import compute_stats
from regime_filter import build_regime

WINDOWS = [
    ("bull-meltup", datetime.now(timezone.utc) - timedelta(days=5), 180, "dual_score_cache.json"),
    ("2022-bear",   datetime(2022, 6, 30, tzinfo=timezone.utc), 90,  "bear_dual_cache.json"),
]
GATE_MAG, GATE_CONF = cfg.LOTTO_MIN_MAGNITUDE, cfg.LOTTO_MIN_CONFIDENCE   # 0.70 / 0.85
DELTA   = cfg.LOTTO_TARGET_DELTA                                          # 0.25
DTE     = 14                                                             # live ~14 (8-21 window)
HOLD    = cfg.LOTTO_MAX_HOLD_DAYS                                        # 7
POS_USD = cfg.LOTTO_POSITION_USD                                        # 250
TIERED  = {"tiers": [(1.0, 0.40), (3.0, 0.30), (float("inf"), 0.20)]}


def _scale(mag, conf):
    return max(POS_USD * 0.5, POS_USD * mag * conf)


def collect(rows, regime):
    """Run each gated signal through the live-equivalent option sim, attaching entry Greeks."""
    save = {k: getattr(_bt, k) for k in
            ("TARGET_DELTA", "DTE_TARGET", "MAX_HOLD_DAYS", "TRAILING_STOP_PCT", "EXIT_PARAMS")}
    _bt.TARGET_DELTA, _bt.DTE_TARGET, _bt.MAX_HOLD_DAYS = DELTA, DTE, HOLD
    _bt.TRAILING_STOP_PCT, _bt.EXIT_PARAMS = 0.30, TIERED
    T = DTE / 365.0
    trades, seen = [], set()
    try:
        for r in rows:
            d = r["created_at"].date()
            if regime and not regime(d):
                continue
            for tk in r["tickers"][:1]:
                if tk in ("BTC", "ETH") or not _bt.is_valid_stock_ticker(tk):
                    continue
                key = f"{d}_{tk}"
                if key in seen:
                    continue
                seen.add(key)
                sp = _bt.get_price_at(tk, r["created_at"])
                if not sp:
                    continue
                base_iv = _bt.base_iv_for(tk, r["created_at"])
                entry_iv = base_iv * _bt.NEWS_IV_MULTIPLIER
                strike = strike_for_delta(sp, T, entry_iv, DELTA)
                gamma = bs_call_gamma(sp, strike, T, entry_iv)
                theta = bs_call_theta(sp, strike, T, entry_iv)
                vega  = bs_call_vega(sp, strike, T, entry_iv)
                t = _bt.simulate_option_pnl(
                    tk, r["created_at"], sp, _scale(r["magnitude"], r["confidence"]),
                    {"magnitude": r["magnitude"], "confidence": r["confidence"]},
                    option_type="call", exit_rule="tiered_profit", spread_mult=1.0)
                if not t:
                    continue
                t.update(entry_iv=entry_iv, gamma=gamma, theta=theta, vega=vega,
                         gt_ratio=(gamma / abs(theta) if theta else 0.0))
                trades.append(t)
    finally:
        for k, v in save.items():
            setattr(_bt, k, v)
    return trades


def _pct(xs, q):
    xs = sorted(xs)
    if not xs:
        return 0.0
    i = min(len(xs) - 1, max(0, int(q / 100.0 * (len(xs) - 1))))
    return xs[i]


def winner_loser_table(trades):
    """Mean of each Greek for losers / small winners / 2x+ multibaggers — does any separate?"""
    buckets = {"loss (≤0%)":   [t for t in trades if t["pnl_pct"] <= 0],
               "win (0-100%)":  [t for t in trades if 0 < t["pnl_pct"] < 100],
               "2x+ (≥100%)":   [t for t in trades if t["pnl_pct"] >= 100]}
    print(f"  {'bucket':14} {'n':>4} {'entry_IV':>9} {'gamma':>9} {'theta/d':>9} {'vega':>8} {'γ/|θ|':>8}")
    for name, ts in buckets.items():
        if not ts:
            print(f"  {name:14} {0:>4}"); continue
        m = lambda k: statistics.mean(t[k] for t in ts)
        print(f"  {name:14} {len(ts):>4} {m('entry_iv'):>9.3f} {m('gamma'):>9.4f} "
              f"{m('theta'):>9.4f} {m('vega'):>8.3f} {m('gt_ratio'):>8.4f}")


def screen_sweep(trades, key, keep_low, label):
    """keep_low=True → keep trades with key ≤ cutoff (drop the high tail); else keep ≥ cutoff."""
    base = compute_stats(trades)
    print(f"\n  ── screen: {label} ──   (baseline n={len(trades)} "
          f"P&L=${base['total_pnl']:,.0f} Sharpe={base['sharpe']:.2f})")
    cuts = [100, 80, 60, 40, 20] if keep_low else [0, 20, 40, 60, 80]
    for q in cuts:
        thr = _pct([t[key] for t in trades], q)
        kept = [t for t in trades if (t[key] <= thr if keep_low else t[key] >= thr)]
        if len(kept) < 10:
            continue
        s = compute_stats(kept)
        x2 = sum(1 for t in kept if t["pnl_pct"] >= 100)
        x4 = sum(1 for t in kept if t["pnl_pct"] >= 300)
        side = "≤" if keep_low else "≥"
        print(f"    keep {key}{side}p{q:<3} (={thr:>8.3f})  n={len(kept):>4}  "
              f"P&L=${s['total_pnl']:>8,.0f}  Sharpe={s['sharpe']:>5.2f}  "
              f"win={s['win_rate']:>4.1f}%  2x+={x2:>3} 4x+={x4:>3}")


def main():
    print("═══ GREEK-SCREEN BACKTEST (lotto Δ%.2f, %dDTE, %dd hold) ═══" % (DELTA, DTE, HOLD))
    print("Caveat: backtest IV = realized_vol×news_mult (flat) → IV/vega/gamma screens are "
          "COLLINEAR proxies for the name's vol; true IV-crush needs forward real-IV data.\n")
    for label, end_dt, days, cache in WINDOWS:
        if not Path(cache).exists():
            print(f"[{label}] cache missing — skip"); continue
        scored = tune_v2.load_scored_from_dual_cache(Path(cache), "bull", end_dt, days)
        gated = [r for r in scored if r["magnitude"] >= GATE_MAG and r["confidence"] >= GATE_CONF]
        reg = build_regime(end_dt, days, 200)
        trades = collect(gated, reg)
        print(f"═══ {label} ═══  gated={len(gated)}  simulated_trades={len(trades)}")
        if len(trades) < 15:
            print("  (too few trades for a meaningful screen)\n"); continue
        winner_loser_table(trades)
        screen_sweep(trades, "entry_iv", True,  "ENTRY IV — drop high-IV picks")
        screen_sweep(trades, "vega",     True,  "VEGA — drop high vol-exposure picks")
        screen_sweep(trades, "gt_ratio", False, "GAMMA/|THETA| — keep high convexity-per-decay")
        print()
    print("Read: a screen 'helps' only if it RAISES P&L/Sharpe while keeping the 4x+ winners.")
    print("Dropping winners to raise win-rate is NOT an improvement for a convex bet.")


if __name__ == "__main__":
    main()
