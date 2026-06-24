"""
overnight_gap_test.py — does exiting at the CLOSE on the max-hold day beat carrying
the position overnight and exiting at the next morning's OPEN?

Theory (user, 2026-06-08): the news edge is largely a first-day effect, so holding a
position overnight just to close it the next morning adds uncompensated gap risk — the
stock may not open where it closed. Test it directly on the STOCK strategy.

Method: reuse the exact stock-strategy entry universe + cost model (stock_backtest). For
each trade, simulate two exit policies that differ ONLY for trades that reach the time-stop
(max-hold) without first hitting the trailing stop:
   • EOD-CLOSE  : exit at the CLOSE of the max-hold day   (current backtest behavior)
   • NEXT-OPEN  : exit at the OPEN of the following day    (carry one night)
Trailing-stopped trades are identical in both arms (they exit intraday at a breach close),
so any P&L difference is purely the overnight gap on time-stopped exits.

Reports, per window (regime-gated like live): trade counts, P&L/Sharpe/win for each arm,
and the overnight-gap distribution on the affected trades.

Usage:  USE_YAHOO_BARS=1 .venv/bin/python overnight_gap_test.py
"""
import os, statistics
os.environ.setdefault("USE_YAHOO_BARS", "1")
from datetime import timedelta
from pathlib import Path

import tune_v2
import config as _cfg
import stock_backtest as sb
from backtest import get_stock_bars, get_price_at, is_valid_stock_ticker
from benchmark import compute_stats
from regime_filter import build_regime

SLIP, TRAIL = sb.STOCK_SLIPPAGE, sb.STOCK_TRAIL
MAX_HOLD = _cfg.NEWS_STOCK_MAX_HOLD_DAYS   # live value (2d); override with --max-hold


def simulate_both(ticker, entry_dt, position_usd):
    """Return (eod_trade, nextopen_trade, exit_mode, gap_pct) or None.
    Both trades identical unless exit_mode == 'time'."""
    bars = get_stock_bars(ticker, entry_dt, entry_dt + timedelta(days=int(MAX_HOLD * 1.5) + 9))
    tb = [b for b in bars if entry_dt < b["t"]]
    if not tb or (tb[0]["t"] - entry_dt).days > 7:
        return None
    p0 = get_price_at(ticker, entry_dt)
    if not p0 or p0 < _cfg.MIN_STOCK_PRICE or p0 > 10000:
        return None
    prev = p0
    for b in tb:
        if prev > 0 and abs(b["c"] / prev - 1) > 0.5:
            return None
        prev = b["c"]

    entry_fill = p0 * (1 + SLIP / 2)
    shares = position_usd / entry_fill

    # Walk for a trailing-stop breach within the hold window.
    peak = p0
    for i, b in enumerate(tb[:MAX_HOLD]):
        peak = max(peak, b["c"])
        if b["c"] <= peak * (1 - TRAIL):
            ex = b["c"] * (1 - SLIP / 2)
            t = {"ticker": ticker, "entry_dt": entry_dt.isoformat(),
                 "pnl_usd": (ex - entry_fill) * shares,
                 "pnl_pct": (ex / entry_fill - 1) * 100}
            return t, t, "trail", 0.0          # identical in both arms

    # Reached the time-stop without a trailing-stop breach.
    if len(tb) <= MAX_HOLD:
        return None                            # no next-day bar to compare → drop
    eod_close = tb[MAX_HOLD - 1]["c"]          # close of the max-hold day
    next_open = tb[MAX_HOLD]["o"]              # open of the following day
    gap_pct = (next_open / eod_close - 1) * 100
    eod_ex  = eod_close * (1 - SLIP / 2)
    nxt_ex  = next_open * (1 - SLIP / 2)
    eod_t = {"ticker": ticker, "entry_dt": entry_dt.isoformat(),
             "pnl_usd": (eod_ex - entry_fill) * shares,
             "pnl_pct": (eod_ex / entry_fill - 1) * 100}
    nxt_t = {"ticker": ticker, "entry_dt": entry_dt.isoformat(),
             "pnl_usd": (nxt_ex - entry_fill) * shares,
             "pnl_pct": (nxt_ex / entry_fill - 1) * 100}
    return eod_t, nxt_t, "time", gap_pct


def run(end_dt, days, cache, regime):
    scored = tune_v2.load_scored_from_dual_cache(Path(cache), "bull", end_dt, days)
    scale = lambda m, c: max(_cfg.MAX_POSITION_USD * 0.10, _cfg.MAX_POSITION_USD * m * c)
    eod, nxt, gaps, n_trail, n_time, seen = [], [], [], 0, 0, {}
    for row in scored:
        m, c = row["magnitude"], row["confidence"]
        if m < _cfg.MIN_MAGNITUDE or c < _cfg.BASE_CONFIDENCE + (1 - m) * _cfg.CONFIDENCE_SLOPE:
            continue
        d = row["created_at"].date()
        if regime is not None and not regime(d):
            continue
        ds = seen.setdefault(d.isoformat(), set())
        for tk in row["tickers"][:2]:
            if tk in ds or tk in ("BTC", "ETH") or not is_valid_stock_ticker(tk):
                continue
            ds.add(tk)
            r = simulate_both(tk, row["created_at"], scale(m, c))
            if not r:
                continue
            eod_t, nxt_t, mode, gap = r
            eod.append(eod_t); nxt.append(nxt_t)
            if mode == "trail":
                n_trail += 1
            else:
                n_time += 1; gaps.append(gap)
    return eod, nxt, gaps, n_trail, n_time


def main():
    global MAX_HOLD
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--max-hold", type=int, default=MAX_HOLD,
                    help=f"max hold days for the time-stop (default {MAX_HOLD} = live NEWS_STOCK_MAX_HOLD_DAYS)")
    args = ap.parse_args()
    MAX_HOLD = args.max_hold
    print("OVERNIGHT-GAP TEST — exit at max-hold CLOSE vs next-day OPEN  (stock strategy)")
    print(f"(cost {SLIP*100:.2f}% RT, {TRAIL*100:.0f}% trail, {MAX_HOLD}d hold, regime-gated like live)\n")
    for label, end_dt, days, cache in sb.WINDOWS:
        reg = build_regime(end_dt, days, 200)
        eod, nxt, gaps, n_trail, n_time = run(end_dt, days, cache, reg)
        se, sn = compute_stats(eod), compute_stats(nxt)
        print(f"═══ {label} ═══")
        print(f"  trades={len(eod)}   trailing-stop exits={n_trail}   time-stop exits={n_time}  "
              f"(only the {n_time} time-stop exits differ between arms)")
        print(f"  EOD-CLOSE  : P&L=${se['total_pnl']:>9,.0f}  Sharpe={se['sharpe']:>5.2f}  win={se['win_rate']:>4.1f}%")
        print(f"  NEXT-OPEN  : P&L=${sn['total_pnl']:>9,.0f}  Sharpe={sn['sharpe']:>5.2f}  win={sn['win_rate']:>4.1f}%")
        diff = sn['total_pnl'] - se['total_pnl']
        print(f"  Δ (next-open − EOD): ${diff:>+9,.0f}   → {'EOD-close BETTER' if diff < 0 else 'NEXT-open better' if diff > 0 else 'tie'}")
        if gaps:
            negs = sum(1 for g in gaps if g < 0)
            print(f"  overnight gap on time-stop days (close→next-open):  mean={statistics.mean(gaps):+.2f}%  "
                  f"median={statistics.median(gaps):+.2f}%  stdev={statistics.pstdev(gaps):.2f}%  "
                  f"negative={negs}/{len(gaps)} ({negs/len(gaps)*100:.0f}%)")
        print()
    print("Read: if EOD-close gives ≥ P&L/Sharpe with a roughly zero-mean gap, the overnight")
    print("hold adds risk without expected return → close at EOD on the max-hold day.")


if __name__ == "__main__":
    main()
