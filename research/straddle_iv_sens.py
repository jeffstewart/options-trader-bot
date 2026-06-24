"""
straddle_iv_sens.py — Sensitivity of earnings straddle P&L to IV assumptions.

Two key unknowns for the straddle strategy:
  1. NEWS_IV_MULTIPLIER: how much IV is elevated at entry vs base realized vol
     - Our calibration used all news days → 1.10×
     - Real earnings IV typically spikes more: 1.3–2.0×
  2. IV_CRUSH_HALFLIFE_DAYS: how fast IV reverts post-announcement
     - Our default is 3 days
     - Earnings IV crushes faster — often same-day (halflife 0.5–1 day)

This script sweeps both axes and shows Sharpe/P&L as a 2D table.
Find where Sharpe goes negative → that's the break-even assumption.
If break-even IV mult is well above what real earnings options trade at,
strategy is robust. If it's marginal, don't deploy without real data.

Usage:  USE_YAHOO_BARS=1 .venv/bin/python straddle_iv_sens.py
"""
import os
from datetime import datetime, timezone, timedelta
from pathlib import Path

os.environ.setdefault("USE_YAHOO_BARS", "1")

import backtest as _bt
import straddle_backtest as _sb
from benchmark import compute_stats
from regime_filter import build_regime
import tune_v2

IV_MULT_GRID  = [1.10, 1.25, 1.40, 1.60, 1.80, 2.00]
HALFLIFE_GRID = [0.5, 1.0, 2.0, 3.0]

WINDOWS = [
    ("bull-meltup", datetime.now(timezone.utc) - timedelta(days=5), 180,
     "dual_score_cache.json"),
    ("2022-bear",   datetime(2022, 6, 30, tzinfo=timezone.utc), 90,
     "bear_dual_cache.json"),
]


def run_at(scored_rows, iv_mult, halflife, regime=None):
    """Run earnings straddle sim with patched IV params."""
    orig_mult     = _sb.NEWS_IV_MULTIPLIER
    orig_halflife = _sb.IV_CRUSH_HALFLIFE_DAYS

    # Patch straddle_backtest's module-level vars (imported from backtest)
    _sb.NEWS_IV_MULTIPLIER      = iv_mult
    _sb.IV_CRUSH_HALFLIFE_DAYS  = halflife

    try:
        trades = _sb.run(scored_rows, regime=regime, earnings_filter=True)
    finally:
        _sb.NEWS_IV_MULTIPLIER      = orig_mult
        _sb.IV_CRUSH_HALFLIFE_DAYS  = orig_halflife

    return trades


def fmt(sharpe, pnl, n):
    if n == 0:
        return "  n/a  "
    return f"{sharpe:+5.1f}"


def main():
    print("═══ EARNINGS STRADDLE — IV SENSITIVITY ═══\n")
    print("Each cell = Sharpe (n=trades).  + = profitable, - = losing.")
    print("Break-even row = where Sharpe flips from + to -.\n")

    for label, end_dt, days, cache in WINDOWS:
        if not Path(cache).exists():
            print(f"  [{label}] cache not found — skipping"); continue

        scored = tune_v2.load_scored_from_dual_cache(Path(cache), "bull", end_dt, days)
        regime = build_regime(end_dt, days, 200) if label == "bull-meltup" else None

        print(f"{'═'*60}")
        print(f"  {label}  (regime gate: {'yes' if regime else 'no'})")
        print(f"{'═'*60}")

        # Header row: halflife values
        header = f"  {'IV mult':>8}  " + "".join(f"  HL={h:.1f}d" for h in HALFLIFE_GRID)
        print(header)
        print("  " + "─" * (len(header) - 2))

        for iv_mult in IV_MULT_GRID:
            row = f"  {iv_mult:>7.2f}×  "
            for halflife in HALFLIFE_GRID:
                trades = run_at(scored, iv_mult, halflife, regime)
                s = compute_stats(trades)
                cell = fmt(s["sharpe"], s["total_pnl"], s["trades"])
                row += f"  {cell:>7}"
            print(row)

        print()

        # Also print P&L table
        print(f"  P&L table ($):")
        print(header)
        print("  " + "─" * (len(header) - 2))
        for iv_mult in IV_MULT_GRID:
            row = f"  {iv_mult:>7.2f}×  "
            for halflife in HALFLIFE_GRID:
                trades = run_at(scored, iv_mult, halflife, regime)
                s = compute_stats(trades)
                if s["trades"] == 0:
                    cell = "  n/a  "
                else:
                    cell = f"${s['total_pnl']:>6,.0f}"
                row += f"  {cell:>8}"
            print(row)

        print()

    print("═══ INTERPRETATION ═══")
    print("  Current model assumption: IV_MULT=1.10, HALFLIFE=3.0")
    print("  Real earnings IV: typically 1.3–2.0× base (varies by name/event)")
    print("  Real earnings crush: faster than news-day average (halflife ~0.5–1d)")
    print("  Strategy viable if positive Sharpe at realistic (IV_MULT, HALFLIFE)")
    print()
    print("  To check real IV: run calibrate_iv.py on earnings-only trades")
    print("  (filter backtest_trades.csv rows where headline matches earnings keywords)")


if __name__ == "__main__":
    main()
