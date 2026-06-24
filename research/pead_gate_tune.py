"""
pead_gate_tune.py — Tune regime gate conditions for PEAD long trades in bear markets.

The 200d-SMA gate is blunt: it blocks ALL long trades in 2022 even if an individual
earnings beat is strong enough to move the stock against the market.  This script
tests whether more nuanced gate conditions allow profitable PEAD longs in a downturn.

Gate types tested:
  1. SPY MA-period variants:  no gate | 50d | 100d | 150d | 200d
  2. Magnitude-only gate:     no MA, but require mag ≥ threshold (signal quality gate)
  3. Combined:                lighter SPY MA + higher mag threshold
  4. Stock-own-momentum gate: stock above its own N-day MA at signal time
     (individual stock in uptrend, regardless of broad market)

The stock-momentum gate is the most novel: an earnings beat on a stock that's
already trending up is more likely to keep running than one that's been falling.
Tested at 20d and 50d lookback.

Cross-regime: best gate combos from 2022-bear validated on bull-meltup (sanity check).

Usage:  USE_YAHOO_BARS=1 .venv/bin/python pead_gate_tune.py
"""
import itertools
import sys
from datetime import datetime, timezone, timedelta
from pathlib import Path

import tune_v2
from pead_backtest import is_earnings_article, STOCK_SLIPPAGE, simulate_pead
from backtest import get_price_at, get_stock_bars, is_valid_stock_ticker
from benchmark import compute_stats
from regime_filter import build_regime
import config as _cfg

BEAR_END   = datetime(2022, 6, 30, tzinfo=timezone.utc)
BEAR_DAYS  = 90
BEAR_CACHE = Path("bear_dual_cache.json")
BULL_END   = datetime.now(timezone.utc) - timedelta(days=5)
BULL_DAYS  = 180
BULL_CACHE = Path("dual_score_cache.json")

PEAD_TRAIL   = 0.20
PEAD_HOLD    = 45


# ── SPY MA-series (pre-built once) ────────────────────────────────────────────

import os
os.environ.setdefault("USE_YAHOO_BARS", "1")

def _spy_ma_series(end_dt, days, ma_period):
    """Return {date: bool} — True if SPY was above its ma_period-day SMA on that date."""
    lookback_days = days + ma_period * 2 + 30
    start = end_dt - timedelta(days=lookback_days)
    bars = get_stock_bars("SPY", start, end_dt)
    if not bars:
        return {}
    prices = [(b["t"].date(), b["c"]) for b in bars]
    result = {}
    for i, (d, price) in enumerate(prices):
        if i < ma_period - 1:
            continue
        ma = sum(p for _, p in prices[i - ma_period + 1: i + 1]) / ma_period
        result[d] = price > ma
    return result


def _stock_ma_series(ticker, entry_dt, ma_period):
    """Return True if ticker was above its ma_period-day SMA on entry_dt date."""
    start = entry_dt - timedelta(days=ma_period * 2 + 14)
    bars = get_stock_bars(ticker, start, entry_dt + timedelta(days=2))
    bars = [b for b in bars if b["t"] <= entry_dt + timedelta(hours=24)]
    if len(bars) < ma_period:
        return None  # insufficient data — treat as no signal
    prices = [b["c"] for b in bars]
    ma = sum(prices[-ma_period:]) / ma_period
    current = prices[-1]
    return current > ma


# ── Core runner (gate-flexible) ───────────────────────────────────────────────

def _scale(mag, conf):
    return max(_cfg.MAX_POSITION_USD * 0.10, _cfg.MAX_POSITION_USD * mag * conf)


def run_gated(scored_rows, spy_mask=None, min_mag=0.35, min_conf=0.55,
              stock_ma=None, trail=PEAD_TRAIL, max_hold=PEAD_HOLD):
    """
    Run PEAD longs with a flexible gate.

    spy_mask : dict {date: bool} from _spy_ma_series, or None (no SPY gate).
    stock_ma : int — if set, also require the individual stock to be above
               its own N-day MA at signal time. None = no stock-level gate.
    """
    trades, seen = [], {}
    for row in scored_rows:
        mag, conf = row["magnitude"], row["confidence"]
        req = min_conf + (1 - mag) * _cfg.CONFIDENCE_SLOPE
        if mag < min_mag or conf < req:
            continue
        if not is_earnings_article(row.get("headline", "")):
            continue
        d = row["created_at"].date()

        # SPY gate
        if spy_mask is not None and not spy_mask.get(d, False):
            continue

        ds = seen.setdefault(d.isoformat(), set())
        for tk in row["tickers"][:2]:
            if tk in ds or tk in ("BTC", "ETH") or not is_valid_stock_ticker(tk):
                continue

            # Stock-own-momentum gate
            if stock_ma is not None:
                above = _stock_ma_series(tk, row["created_at"], stock_ma)
                if above is False:   # explicitly below MA — skip
                    continue
                # above is None (insufficient data) → let through (conservative)

            ds.add(tk)
            t = simulate_pead(tk, row["created_at"], _scale(mag, conf),
                              trail=trail, max_hold=max_hold)
            if t:
                trades.append(t)
    return trades


# ── Reporting ─────────────────────────────────────────────────────────────────

def line(tag, trades, note=""):
    s = compute_stats(trades)
    suffix = f"  {note}" if note else ""
    return (f"  {tag:40}  n={s['trades']:>4}  Sharpe={s['sharpe']:>6.2f}"
            f"  P&L=${s['total_pnl']:>9,.0f}  win={s['win_rate']:>4.1f}%"
            f"  maxDD=${s['max_dd']:>7,.0f}{suffix}")


# ── Tune ──────────────────────────────────────────────────────────────────────

def tune_gate(scored_rows, end_dt, days, label):
    """Grid search all gate combos on the given scored dataset."""
    n = len(scored_rows)
    split = int(n * 0.80)
    train_s, hold_s = scored_rows[:split], scored_rows[split:]
    train_end = train_s[-1]["created_at"]

    # Pre-build SPY MA series for both train and holdout periods
    spy_masks = {}
    for ma_days in [50, 100, 150, 200]:
        spy_masks[ma_days] = _spy_ma_series(end_dt, days + 30, ma_days)

    grid = {
        "spy_ma":   [0, 50, 100, 150, 200],  # 0 = no SPY gate
        "min_mag":  [0.35, 0.45, 0.55, 0.65],
        "stock_ma": [None, 20, 50],            # None = no individual-stock gate
    }
    keys   = list(grid.keys())
    combos = list(itertools.product(*[grid[k] for k in keys]))
    print(f"  {label}: {len(combos)} combos  (train={len(train_s)} | holdout={len(hold_s)} articles)",
          flush=True)

    results = []
    for vals in combos:
        p = dict(zip(keys, vals))
        mask = spy_masks.get(p["spy_ma"]) if p["spy_ma"] else None
        tr = run_gated(train_s, spy_mask=mask,
                       min_mag=p["min_mag"], stock_ma=p["stock_ma"])
        ts = compute_stats(tr)
        results.append({**p, "train_sharpe": ts["sharpe"],
                        "train_pnl": ts["total_pnl"], "train_n": ts["trades"],
                        "train_win": ts["win_rate"]})

    results.sort(key=lambda r: r["train_sharpe"], reverse=True)
    top8 = results[:8]

    hdr = (f"  {'SPY-MA':>6} {'mag':>4} {'stk-MA':>6}"
           f"  {'Tr.Sh':>7} {'Tr.n':>5}"
           f"  {'Ho.Sh':>7} {'Ho.P&L':>9} {'Ho.n':>5} {'Ho.win':>6}")
    print(f"\n  ── Top 8 combos ──\n{hdr}")

    holdout_results = []
    for r in top8:
        mask = spy_masks.get(r["spy_ma"]) if r["spy_ma"] else None
        ho = run_gated(hold_s, spy_mask=mask,
                       min_mag=r["min_mag"], stock_ma=r["stock_ma"])
        hs = compute_stats(ho)
        holdout_results.append({**r,
            "ho_sharpe": hs["sharpe"], "ho_pnl": hs["total_pnl"],
            "ho_n": hs["trades"], "ho_win": hs["win_rate"]})
        spy_label = f"SPY>{r['spy_ma']}d" if r["spy_ma"] else "no-SPY"
        stk_label = f"stk>{r['stock_ma']}d" if r["stock_ma"] else "no-stk"
        print(f"  {spy_label:>6} {r['min_mag']:>4.2f} {stk_label:>6}"
              f"  {r['train_sharpe']:>7.2f} {r['train_n']:>5}"
              f"  {hs['sharpe']:>7.2f} ${hs['total_pnl']:>8,.0f} {hs['trades']:>5}"
              f" {hs['win_rate']:>5.1f}%")

    holdout_results.sort(key=lambda r: r["ho_sharpe"], reverse=True)
    return holdout_results


def main():
    if not BEAR_CACHE.exists():
        print(f"Bear cache not found: {BEAR_CACHE}"); sys.exit(1)

    bear_scored = tune_v2.load_scored_from_dual_cache(BEAR_CACHE, "bull",
                                                      BEAR_END, BEAR_DAYS)
    bull_scored = tune_v2.load_scored_from_dual_cache(BULL_CACHE, "bull",
                                                      BULL_END, BULL_DAYS)

    print("═══ PEAD GATE TUNE ═══\n")

    # ── Section 1: SPY MA period comparison (no stock-level gate) ─────────────
    print("═ SPY MA-period comparison in 2022 bear (no stock-level gate) ═")
    spy_masks_bear = {}
    for ma in [50, 100, 150, 200]:
        spy_masks_bear[ma] = _spy_ma_series(BEAR_END, BEAR_DAYS + 30, ma)

    for ma in [0, 50, 100, 150, 200]:
        mask = spy_masks_bear.get(ma) if ma else None
        gate_name = f"SPY > {ma}d SMA" if ma else "no gate     "
        t = run_gated(bear_scored, spy_mask=mask, min_mag=0.35)
        print(line(gate_name, t))
    print()

    # ── Section 2: Magnitude-only gate (no SPY gate) ──────────────────────────
    print("═ Magnitude-only gate in 2022 bear (no SPY filter) ═")
    for mag_thresh in [0.35, 0.45, 0.55, 0.65, 0.70]:
        t = run_gated(bear_scored, spy_mask=None, min_mag=mag_thresh)
        print(line(f"mag ≥ {mag_thresh:.2f} only", t))
    print()

    # ── Section 3: Stock-own-momentum gate ────────────────────────────────────
    print("═ Stock-own-momentum gate in 2022 bear (stock above its own SMA) ═")
    for stock_ma in [20, 50]:
        for mag in [0.35, 0.55]:
            t = run_gated(bear_scored, spy_mask=None, min_mag=mag, stock_ma=stock_ma)
            print(line(f"stk>{stock_ma}d MA + mag≥{mag:.2f} (no SPY gate)", t))
    print()

    # ── Section 4: Combined gates ─────────────────────────────────────────────
    print("═ Combined: SPY 50d + stock momentum + high mag ═")
    mask_50 = spy_masks_bear.get(50)
    for mag in [0.35, 0.45, 0.55]:
        for stock_ma in [None, 20, 50]:
            t = run_gated(bear_scored, spy_mask=mask_50, min_mag=mag, stock_ma=stock_ma)
            stk_lbl = f"stk>{stock_ma}d" if stock_ma else "no-stk"
            print(line(f"SPY>50d + mag≥{mag:.2f} + {stk_lbl}", t))
    print()

    # ── Section 5: Full tune with holdout ─────────────────────────────────────
    print("═ Full gate tune with 80/20 holdout (2022 bear) ═")
    best_bear = tune_gate(bear_scored, BEAR_END, BEAR_DAYS, "2022-bear")
    if best_bear:
        b = best_bear[0]
        spy_lbl = f"SPY>{b['spy_ma']}d" if b['spy_ma'] else "no-SPY"
        stk_lbl = f"stk>{b['stock_ma']}d" if b['stock_ma'] else "no-stk"
        print(f"\n  Best holdout combo: {spy_lbl}, mag≥{b['min_mag']:.2f}, {stk_lbl}")
        print(f"    Bear holdout  → Sharpe {b['ho_sharpe']:.2f}  "
              f"P&L=${b['ho_pnl']:,.0f}  n={b['ho_n']}  win={b['ho_win']:.1f}%")

        # Cross-validate on bull melt-up: same gate should still allow most trades
        print("\n  ── Cross-validate best gate on bull melt-up ──")
        spy_masks_bull = {}
        for ma in [50, 100, 150, 200]:
            spy_masks_bull[ma] = _spy_ma_series(BULL_END, BULL_DAYS + 30, ma)
        mask_bull = spy_masks_bull.get(b['spy_ma']) if b['spy_ma'] else None
        bull_t = run_gated(bull_scored, spy_mask=mask_bull,
                           min_mag=b['min_mag'], stock_ma=b['stock_ma'])
        # Compare vs current gate on bull
        current_bull = run_gated(bull_scored,
                                 spy_mask=_spy_ma_series(BULL_END, BULL_DAYS+30, 200),
                                 min_mag=0.35)
        print(line("Current gate (SPY>200d, mag≥0.35) — bull", current_bull))
        print(line(f"Best bear gate applied to bull", bull_t,
                   note=f"({len(bull_t)}/{max(len(current_bull),1)*100//max(len(current_bull),1)}% of baseline trades)"))


if __name__ == "__main__":
    main()
