"""
bear_short_backtest.py — does SHORTING STOCK on the model's bearish picks make money?

Different from the old bear_backtest.py in three ways:
  1. INSTRUMENT: simulates the ACTUAL live strategy — SHORT STOCK (bear_short / pairs_short,
     trough-trailing stop at BEAR_SHORT_TRAIL_PCT, no time cap) — not long puts.
  2. SIGNAL: uses the dual cache's MODEL-scored bearish picks (1,152 in 2022 w/ tickers) vs the
     old backtest's news-metadata fallback (model tickered only 5/1298).
  3. QUESTION: can the short side MAKE money / fill the dead bear-regime gap (offense), not just
     "do longs survive a downturn" (defense). Run in BOTH regimes.

Control: RANDOM shorts on the same dates/universe → isolates whether the MODEL's selection adds
edge beyond "shorting anything in this regime."

Usage:  USE_YAHOO_BARS=1 .venv/bin/python bear_short_backtest.py [--seeds 3]
"""
import os, argparse, random, statistics
os.environ.setdefault("USE_YAHOO_BARS", "1")
from datetime import datetime, timezone, timedelta
from pathlib import Path

import tune_v2, config as cfg
from backtest import get_stock_bars, get_price_at, is_valid_stock_ticker
from benchmark import compute_stats
from regime_filter import build_regime

SLIP = 0.001
TRAIL = cfg.BEAR_SHORT_TRAIL_PCT   # 0.15 trough-trail
MAX_HOLD = 30                      # cap the sim window (live has no time cap)
BASE = 1000.0

WINDOWS = [
    ("bull-meltup (shorts INTO a bull)", datetime.now(timezone.utc) - timedelta(days=5), 180, "dual_score_cache.json"),
    ("2022-bear  (shorts INTO a bear)",  datetime(2022, 6, 30, tzinfo=timezone.utc), 90, "bear_dual_cache.json"),
]


def gate(mag, conf):
    return mag >= cfg.MIN_MAGNITUDE and conf >= cfg.BASE_CONFIDENCE + (1 - mag) * cfg.CONFIDENCE_SLOPE


def simulate_short(ticker, entry_dt, position_usd):
    """Short the stock; cover on a rally to trough×(1+TRAIL) or at MAX_HOLD. Profit when price FALLS."""
    bars = get_stock_bars(ticker, entry_dt, entry_dt + timedelta(days=int(MAX_HOLD * 1.5) + 7))
    tb = [b for b in bars if entry_dt < b["t"]]
    if not tb or (tb[0]["t"] - entry_dt).days > 7:
        return None
    p0 = get_price_at(ticker, entry_dt)
    if not p0 or p0 < cfg.MIN_STOCK_PRICE or p0 > 10000:
        return None
    prev = p0
    for b in tb:                                    # bad-data jump guard
        if prev > 0 and abs(b["c"] / prev - 1) > 0.5:
            return None
        prev = b["c"]
    entry_fill = p0 * (1 - SLIP / 2)                # short sale: sell slightly below mid
    shares = position_usd / entry_fill
    trough = p0
    exit_price = tb[min(MAX_HOLD, len(tb)) - 1]["c"]   # default: cover at window end
    for b in tb[:MAX_HOLD]:
        trough = min(trough, b["c"])
        if b["c"] >= trough * (1 + TRAIL):          # rally off the lows → stop (buy to cover)
            exit_price = b["c"]
            break
    exit_fill = exit_price * (1 + SLIP / 2)          # cover: buy slightly above mid
    pnl = (entry_fill - exit_fill) * shares          # SHORT: profit when exit < entry
    return {"ticker": ticker, "entry_dt": entry_dt.isoformat(),
            "pnl_usd": pnl, "pnl_pct": (entry_fill / exit_fill - 1) * 100}


def run(rows, regime, universe=None, seed=None):
    """universe set → RANDOM-short control (ignore the model's ticker, pick a random valid name)."""
    rng = random.Random(seed)
    trades, seen = [], {}
    for r in rows:
        m, c = r["magnitude"], r["confidence"]
        if not gate(m, c):
            continue
        d = r["created_at"].date()
        if regime is not None and not regime(d):
            continue
        ds = seen.setdefault(d.isoformat(), set())
        picks = r["tickers"][:2]
        if universe is not None:
            picks = [rng.choice(universe)]
        for tk in picks:
            if tk in ds or tk in ("BTC", "ETH") or not is_valid_stock_ticker(tk):
                continue
            ds.add(tk)
            try:
                t = simulate_short(tk, r["created_at"], BASE)
            except Exception:
                continue
            if t:
                trades.append(t)
    return trades


def line(tag, trades):
    if not trades:
        return f"  {tag:30} n=   0"
    s = compute_stats(trades)
    win = sum(1 for t in trades if t["pnl_usd"] > 0) / len(trades) * 100
    return (f"  {tag:30} n={len(trades):>4}  P&L=${s['total_pnl']:>8,.0f}  "
            f"Sharpe={s['sharpe']:>5.2f}  win={win:>4.1f}%  avg={s['total_pnl']/len(trades):>+6.0f}$")


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--seeds", type=int, default=3); a = ap.parse_args()
    print(f"═══ BEAR-SHORT (short stock) BACKTEST — {TRAIL*100:.0f}% trough-trail, ≤{MAX_HOLD}d, flat ${BASE:.0f} ═══")
    print("Profit when the shorted stock FALLS. Control = random shorts (same dates/universe).\n")
    for label, end_dt, days, cache in WINDOWS:
        if not Path(cache).exists():
            print(f"[{label}] cache missing — skip"); continue
        bear = tune_v2.load_scored_from_dual_cache(Path(cache), "bear", end_dt, days)
        # bear_short is REGIME-AGNOSTIC live (runs in all regimes) — do NOT apply the 200d-SMA
        # gate. The WINDOW itself supplies the regime context (bull-meltup vs 2022-bear).
        reg = None
        universe = sorted({tk for r in bear for tk in r["tickers"][:2] if is_valid_stock_ticker(tk)})
        print(f"═══ {label} ═══  (bearish signals={len(bear)}, universe={len(universe)} names)")
        model = run(bear, reg)
        print(line("MODEL bearish shorts", model))
        # random-short control, averaged over seeds
        rand_pnls, rand_sh = [], []
        for s in range(a.seeds):
            rt = run(bear, reg, universe=universe, seed=s)
            if rt:
                st = compute_stats(rt); rand_pnls.append(st["total_pnl"]); rand_sh.append(st["sharpe"])
        if rand_pnls:
            print(f"  {'RANDOM shorts (control, avg)':30} "
                  f"P&L=${statistics.mean(rand_pnls):>8,.0f}  Sharpe={statistics.mean(rand_sh):>5.2f}  "
                  f"(over {a.seeds} seeds)")
        print()
    print("Read: MODEL shorts should beat RANDOM (selection edge) AND be net-positive in the bear")
    print("window to be worth activating. Positive in the BULL window = a real all-weather short.")


if __name__ == "__main__":
    main()
