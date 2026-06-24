"""
pead_bear.py — Bear-market PEAD: can earnings signals make money in a downturn?

Two tests:
  1. BULL signals, no regime gate, 2022 bear window — do earnings stocks still
     drift up after a bullish catalyst even when the market is falling?
  2. BEAR signals (earnings misses/warnings), short the stock, 2022 bear window —
     can we profit from post-negative-earnings drift DOWN?

Includes a tune for the bear-short trail parameters (same grid as bull PEAD tune
but with the inverted simulation).

Usage:  USE_YAHOO_BARS=1 .venv/bin/python pead_bear.py
"""
import itertools
import sys
from datetime import datetime, timezone, timedelta
from pathlib import Path

import tune_v2
from pead_backtest import is_earnings_article, STOCK_SLIPPAGE
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


# ── Long simulation (bull signals, no regime gate) ────────────────────────────

def sim_long(ticker, entry_dt, position_usd, trail=0.15, max_hold=45):
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
        if b["c"] <= peak * (1 - trail):
            exit_price = b["c"]
            break
    exit_fill = exit_price * (1 - STOCK_SLIPPAGE / 2)
    pnl = (exit_fill - entry_fill) * shares
    return {"ticker": ticker, "entry_dt": entry_dt.isoformat(),
            "pnl_usd": pnl, "pnl_pct": (exit_fill / entry_fill - 1) * 100}


# ── Short simulation (bear signals, earnings misses) ─────────────────────────

def sim_short(ticker, entry_dt, position_usd, trail=0.15, max_hold=45):
    """
    Simulate shorting the stock at entry. Profit when price falls.
    Stop: price bounces 'trail' % above its lowest point since entry.
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
    entry_fill = p0  # short at market — no slippage adjustment on entry
    shares = position_usd / entry_fill
    trough = p0
    exit_price = tb[min(max_hold, len(tb)) - 1]["c"]
    for b in tb[:max_hold]:
        trough = min(trough, b["c"])
        # Stop: price recovered trail% above the trough (lock in the move)
        if b["c"] >= trough * (1 + trail):
            exit_price = b["c"]
            break
    exit_fill = exit_price * (1 + STOCK_SLIPPAGE / 2)  # pay up to cover
    pnl = (entry_fill - exit_fill) * shares  # profit = entry - exit
    pnl_pct = (entry_fill - exit_fill) / entry_fill * 100
    return {"ticker": ticker, "entry_dt": entry_dt.isoformat(),
            "pnl_usd": pnl, "pnl_pct": pnl_pct}


# ── Dataset runners ───────────────────────────────────────────────────────────

def _scale(mag, conf):
    return max(_cfg.MAX_POSITION_USD * 0.10, _cfg.MAX_POSITION_USD * mag * conf)


def run_side(scored_rows, regime, fn, earnings_only=True,
             trail=0.15, max_hold=45, min_mag=0.35, min_conf=0.55):
    trades, seen = [], {}
    for row in scored_rows:
        mag, conf = row["magnitude"], row["confidence"]
        req = min_conf + (1 - mag) * _cfg.CONFIDENCE_SLOPE
        if mag < min_mag or conf < req:
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
            t = fn(tk, row["created_at"], _scale(mag, conf), trail, max_hold)
            if t:
                trades.append(t)
    return trades


# ── Bear tune ─────────────────────────────────────────────────────────────────

BEAR_TUNE_GRID = {
    "max_hold": [21, 30, 45, 60],
    "trail":    [0.10, 0.15, 0.20, 0.25],
    "min_mag":  [0.35, 0.45, 0.55],
    "min_conf": [0.55, 0.65, 0.70],
}


def tune_bear(scored_rows):
    """Grid search bear-short params; 80/20 holdout on the bear cache."""
    n = len(scored_rows)
    split = int(n * 0.80)
    train_s, hold_s = scored_rows[:split], scored_rows[split:]
    # No regime gate for bear — we're explicitly testing the bear window
    regime = None

    keys   = list(BEAR_TUNE_GRID.keys())
    combos = list(itertools.product(*[BEAR_TUNE_GRID[k] for k in keys]))
    print(f"  Bear tune: {len(combos)} combos  "
          f"(train={len(train_s)} | holdout={len(hold_s)} articles)", flush=True)

    results = []
    for vals in combos:
        p = dict(zip(keys, vals))
        tr = run_side(train_s, regime, sim_short, earnings_only=True,
                      trail=p["trail"], max_hold=p["max_hold"],
                      min_mag=p["min_mag"], min_conf=p["min_conf"])
        ts = compute_stats(tr)
        results.append({**p, "train_sharpe": ts["sharpe"],
                        "train_pnl": ts["total_pnl"], "train_n": ts["trades"]})

    results.sort(key=lambda r: r["train_sharpe"], reverse=True)
    top5 = results[:5]

    hdr = (f"  {'hold':>4} {'trail':>5} {'mag':>4} {'conf':>4}"
           f"  {'Tr.Sh':>7}  {'Tr.P&L':>9}  {'Tr.n':>5}"
           f"  {'Ho.Sh':>7}  {'Ho.P&L':>9}  {'Ho.n':>5}")
    print(f"\n  ── Top 5 (bear-short, earnings filtered) ──\n{hdr}")
    for r in top5:
        p = {k: r[k] for k in keys}
        ho = run_side(hold_s, regime, sim_short, earnings_only=True,
                      trail=p["trail"], max_hold=p["max_hold"],
                      min_mag=p["min_mag"], min_conf=p["min_conf"])
        hs = compute_stats(ho)
        print(f"  {r['max_hold']:>4} {r['trail']:>5.2f} {r['min_mag']:>4.2f} {r['min_conf']:>4.2f}"
              f"  {r['train_sharpe']:>7.2f}  ${r['train_pnl']:>8,.0f}  {r['train_n']:>5}"
              f"  {hs['sharpe']:>7.2f}  ${hs['total_pnl']:>8,.0f}  {hs['trades']:>5}")
    return top5


# ── Main ──────────────────────────────────────────────────────────────────────

def line(tag, s):
    return (f"  {tag:35}  n={s['trades']:>4}  Sharpe={s['sharpe']:>6.2f}"
            f"  P&L=${s['total_pnl']:>9,.0f}  win={s['win_rate']:>4.1f}%"
            f"  maxDD=${s['max_dd']:>7,.0f}")


def main():
    if not BEAR_CACHE.exists():
        print(f"Bear cache not found: {BEAR_CACHE}"); sys.exit(1)

    bear_scored = tune_v2.load_scored_from_dual_cache(BEAR_CACHE, "bear",
                                                      BEAR_END, BEAR_DAYS)
    bull_bear_scored = tune_v2.load_scored_from_dual_cache(BEAR_CACHE, "bull",
                                                           BEAR_END, BEAR_DAYS)
    bull_meltup = tune_v2.load_scored_from_dual_cache(BULL_CACHE, "bull",
                                                      BULL_END, BULL_DAYS)

    bear_reg = build_regime(BEAR_END, BEAR_DAYS, 200)  # mostly blocked in 2022

    print("═══ BEAR-MARKET PEAD TESTS ═══\n")

    # ── 1. Bull PEAD without regime gate in 2022 ─────────────────────────────
    print("═ Bull signals, 2022 bear window (sanity check) ═")
    b_nogated = run_side(bull_bear_scored, None,     sim_long, True,  0.15, 45)
    b_gated   = run_side(bull_bear_scored, bear_reg, sim_long, True,  0.15, 45)
    b_all_ng  = run_side(bull_bear_scored, None,     sim_long, False, 0.10, 30)
    print(line("PEAD earnings, NO regime gate",  compute_stats(b_nogated)))
    print(line("PEAD earnings, regime gated",    compute_stats(b_gated)))
    print(line("All bull signals, no gate 30d",  compute_stats(b_all_ng)))
    print()

    # ── 2. Bear-short PEAD in 2022 ────────────────────────────────────────────
    print("═ Bear signals, 2022 bear window (earnings-miss shorts) ═")
    sh_earn  = run_side(bear_scored, None, sim_short, True,  0.15, 45)
    sh_all   = run_side(bear_scored, None, sim_short, False, 0.15, 30)
    sh_earn_b= run_side(bear_scored, None, sim_short, True,  0.15, 45, 0.45, 0.65)
    print(line("Bear-short, earnings only",           compute_stats(sh_earn)))
    print(line("Bear-short, all bearish signals",     compute_stats(sh_all)))
    print(line("Bear-short earnings (mag≥0.45)",      compute_stats(sh_earn_b)))
    print()

    # ── 3. Bear-short PEAD in bull melt-up (sanity check) ────────────────────
    print("═ Bear signals, bull melt-up window (sanity check) ═")
    bull_bear_sc = tune_v2.load_scored_from_dual_cache(BULL_CACHE, "bear",
                                                       BULL_END, BULL_DAYS)
    sh_bull = run_side(bull_bear_sc, None, sim_short, True, 0.15, 45)
    print(line("Bear-short earnings in bull melt-up", compute_stats(sh_bull)))
    print("  (expect loss — shorting in a +18% melt-up)\n")

    # ── 4. Tune bear-short params ─────────────────────────────────────────────
    print("═ Tune bear-short trail parameters ═")
    tune_bear(bear_scored)
    print()


if __name__ == "__main__":
    main()
