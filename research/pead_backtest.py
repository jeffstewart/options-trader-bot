"""
pead_backtest.py — Post-Earnings Announcement Drift (PEAD) backtest + tune.

PEAD thesis: stocks continue to drift in the direction of a positive earnings
surprise for WEEKS after the announcement.  The LLM already identifies bullish
earnings catalysts with high magnitude; we filter to earnings-related articles
and hold the underlying STOCK for a longer window (30-90 days) rather than the
14-21 DTE options used by the main strategy.

Stocks only (no options): strips IV-crush / spread friction that killed the
options edge, isolating whether the multi-week drift signal exists at all.

Compares against a non-PEAD baseline (same signals, 30d hold, no earnings filter)
to measure whether restricting to earnings articles actually adds value.

Usage:
    USE_YAHOO_BARS=1 .venv/bin/python pead_backtest.py            # baseline
    USE_YAHOO_BARS=1 .venv/bin/python pead_backtest.py --tune     # grid search
"""
import argparse
import itertools
import sys
from datetime import datetime, timezone, timedelta
from pathlib import Path

import backtest as _bt
import tune_v2
from backtest import get_price_at, get_stock_bars, is_valid_stock_ticker
from benchmark import compute_stats
from regime_filter import build_regime
import config as _cfg

STOCK_SLIPPAGE = 0.001   # 0.10% RT (same as stock_backtest.py)

WINDOWS = [
    ("bull-meltup", datetime.now(timezone.utc) - timedelta(days=5), 180, "dual_score_cache.json"),
    ("2022-bear",   datetime(2022, 6, 30, tzinfo=timezone.utc),     90,  "bear_dual_cache.json"),
]

# ── Earnings keyword detection ────────────────────────────────────────────────
# We need the headline to reference an actual earnings event (beats, results,
# guidance changes) — not just mention "revenue" in passing.

_EARNINGS_PATTERNS = [
    "q1 earnings", "q2 earnings", "q3 earnings", "q4 earnings",
    "quarterly earnings", "quarterly results", "quarterly profit",
    "earnings beat", "earnings beats", "beat estimates", "beats estimates",
    "beat expectations", "beats expectations", "topped estimates",
    "exceeded estimates", "smashed estimates", "surpassed estimates",
    "eps beat", "eps of $", "eps topped",
    "raises guidance", "raised guidance", "raises outlook", "boosts outlook",
    "guidance raised", "guidance increase", "guidance raise",
    "reports earnings", "reported earnings", "reports q", "reported q",
]

def is_earnings_article(headline: str) -> bool:
    h = headline.lower()
    return any(p in h for p in _EARNINGS_PATTERNS)


# ── Simulation ────────────────────────────────────────────────────────────────

def simulate_pead(ticker, entry_dt, position_usd, trail=0.15, max_hold=45):
    """Buy stock at entry_dt and hold up to max_hold trading days with trail stop."""
    bars = get_stock_bars(ticker, entry_dt,
                          entry_dt + timedelta(days=int(max_hold * 1.6) + 14))
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


def run_scored(scored_rows, regime=None,
               trail=0.15, max_hold=45, min_mag=0.35, min_conf=0.55,
               earnings_filter=True):
    """Core runner — accepts pre-loaded scored rows for reuse in the tune loop."""
    def conf_floor(mag):
        return min_conf + (1 - mag) * _cfg.CONFIDENCE_SLOPE

    def scale(mag, conf):
        return max(_cfg.MAX_POSITION_USD * 0.10, _cfg.MAX_POSITION_USD * mag * conf)

    trades, seen = [], {}
    for row in scored_rows:
        mag, conf = row["magnitude"], row["confidence"]
        if mag < min_mag or conf < conf_floor(mag):
            continue
        if earnings_filter and not is_earnings_article(row.get("headline", "")):
            continue
        d = row["created_at"].date()
        if regime is not None and not regime(d):
            continue
        ds = seen.setdefault(d.isoformat(), set())
        for tk in row["tickers"][:2]:
            if tk in ds or tk in ("BTC", "ETH") or not is_valid_stock_ticker(tk):
                continue
            ds.add(tk)
            t = simulate_pead(tk, row["created_at"], scale(mag, conf), trail, max_hold)
            if t:
                trades.append(t)
    return trades


def run(end_dt, days, cache, **kw):
    scored = tune_v2.load_scored_from_dual_cache(Path(cache), "bull", end_dt, days)
    return run_scored(scored, **kw)


# ── Reporting ─────────────────────────────────────────────────────────────────

def line(tag, s):
    return (f"  {tag:30}  trades={s['trades']:>4}  P&L=${s['total_pnl']:>9,.0f}  "
            f"Sharpe={s['sharpe']:>5.2f}  win={s['win_rate']:>4.1f}%  "
            f"maxDD=${s['max_dd']:>8,.0f}")


# ── Tune ─────────────────────────────────────────────────────────────────────

TUNE_GRID = {
    "max_hold":  [21, 30, 45, 60, 90],
    "trail":     [0.08, 0.10, 0.15, 0.20],
    "min_mag":   [0.35, 0.45, 0.55],
    "min_conf":  [0.55, 0.65, 0.70],
}

def tune(end_dt, days, cache, holdout_pct=0.20):
    all_scored = tune_v2.load_scored_from_dual_cache(Path(cache), "bull", end_dt, days)
    reg_all    = build_regime(end_dt, days, 200)

    n = len(all_scored)
    split = int(n * (1 - holdout_pct))
    train_scored, hold_scored = all_scored[:split], all_scored[split:]
    train_end = train_scored[-1]["created_at"]
    hold_end  = all_scored[-1]["created_at"]
    train_reg = build_regime(train_end, days, 200)
    hold_reg  = build_regime(hold_end,  days, 200)

    print(f"  Train: {len(train_scored)} articles  Holdout: {len(hold_scored)} articles")

    keys   = list(TUNE_GRID.keys())
    combos = list(itertools.product(*[TUNE_GRID[k] for k in keys]))
    print(f"  {len(combos)} combos ...", flush=True)

    results = []
    for i, vals in enumerate(combos):
        params = dict(zip(keys, vals))
        if (i + 1) % 50 == 0:
            print(f"  {i+1}/{len(combos)}", flush=True)
        train_trades = run_scored(train_scored, regime=train_reg, **params)
        ts = compute_stats(train_trades)
        results.append({**params, "train_sharpe": ts["sharpe"],
                        "train_pnl": ts["total_pnl"], "train_trades": ts["trades"]})

    results.sort(key=lambda r: r["train_sharpe"], reverse=True)
    top5 = results[:5]

    hdr = f"  {'hold':>4} {'trail':>5} {'mag':>4} {'conf':>4}  {'Sharpe':>8}  {'P&L':>10}  {'n':>5}"
    print("\n  ── Top 5 by train Sharpe ──")
    print(hdr)
    for r in top5:
        print(f"  {r['max_hold']:>4} {r['trail']:>5.2f} {r['min_mag']:>4.2f} {r['min_conf']:>4.2f}  "
              f"{r['train_sharpe']:>8.2f}  ${r['train_pnl']:>9,.0f}  {r['train_trades']:>5}")

    print("\n  ── Holdout validation (top 5) ──")
    print(hdr)
    for r in top5:
        params = {k: r[k] for k in keys}
        hold_trades = run_scored(hold_scored, regime=hold_reg, **params)
        hs = compute_stats(hold_trades)
        print(f"  {r['max_hold']:>4} {r['trail']:>5.2f} {r['min_mag']:>4.2f} {r['min_conf']:>4.2f}  "
              f"{hs['sharpe']:>8.2f}  ${hs['total_pnl']:>9,.0f}  {hs['trades']:>5}")

    return top5


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tune", action="store_true", help="Run grid search + holdout")
    args = ap.parse_args()

    if args.tune:
        print("═══ PEAD TUNE ═══\n")
        for label, end_dt, days, cache in WINDOWS:
            if not Path(cache).exists():
                print(f"  [{label}] cache not found: {cache} — skipping")
                continue
            print(f"══ {label} ══")
            tune(end_dt, days, cache)
            print()
        return

    # ── Baseline comparison ───────────────────────────────────────────────────
    print("PEAD backtest  (stock, 0.10% slippage, 200d regime gate)\n")
    for label, end_dt, days, cache in WINDOWS:
        if not Path(cache).exists():
            print(f"  [{label}] cache not found — skipping")
            continue
        print(f"═══ {label} ═══")
        reg = build_regime(end_dt, days, 200)
        # Baseline: no earnings filter, 30d hold
        base = run(end_dt, days, cache, regime=reg,
                   trail=0.10, max_hold=30, earnings_filter=False)
        print(line("no-filter stock 30d", compute_stats(base)))
        # PEAD: earnings filter, various hold lengths
        for hold in [30, 45, 60, 90]:
            t = run(end_dt, days, cache, regime=reg,
                    trail=0.10, max_hold=hold, earnings_filter=True)
            pct = len(t) / max(len(base), 1) * 100
            print(line(f"PEAD earnings {hold}d hold",
                       compute_stats(t)) + f"  ({pct:.0f}% of base signals)")
        print()


if __name__ == "__main__":
    main()
