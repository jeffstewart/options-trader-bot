"""
ma_arb_backtest.py — M&A arbitrage backtest.

After a merger/acquisition announcement, the target company's stock jumps toward
the deal price and then trades at a small discount (the arb spread) while the deal
closes. Buying the target after the announcement and holding until spread compresses
typically yields 2–8% over 30–90 days, and is largely regime-independent (deals
happen in both bull and bear markets).

Method:
  1. Filter dual_score_cache headlines for M&A keywords (announce, acquire, merger…)
  2. LLM must also score it bullish with high magnitude (confirms it's a positive deal)
  3. Buy the target stock, hold with:
     - Tight trailing stop (8%) — protects against deal breaks
     - Max hold 90 days — most deals close within 90 days
  4. Compare vs same-signals without M&A filter (all bullish) as baseline

Regime-independent test: run on both bull and bear windows.

Usage:  USE_YAHOO_BARS=1 .venv/bin/python ma_arb_backtest.py
"""
import os
from datetime import datetime, timezone, timedelta
from pathlib import Path

os.environ.setdefault("USE_YAHOO_BARS", "1")

import tune_v2
from pead_backtest import simulate_pead, STOCK_SLIPPAGE
from backtest import is_valid_stock_ticker
from benchmark import compute_stats
from regime_filter import build_regime
import config as _cfg

WINDOWS = [
    ("bull-meltup", datetime.now(timezone.utc) - timedelta(days=5), 180,
     "dual_score_cache.json"),
    ("2022-bear",   datetime(2022, 6, 30, tzinfo=timezone.utc), 90,
     "bear_dual_cache.json"),
]

MA_TRAIL   = 0.08   # tight — protect against deal breaks
MA_HOLD    = 90     # most deals close within 90 trading days

# Keywords indicating an M&A announcement targeting a specific company
MA_KEYWORDS = [
    "to acquire", "to be acquired", "acquisition of", "merger agreement",
    "definitive agreement", "takeover", "buyout", "go private",
    "purchase agreement", "tender offer", "strategic alternatives",
    "merger with", "acquires ", "acquired by", "agreed to buy",
    "deal to buy", "plans to acquire", "in a deal",
]


def is_ma_article(headline: str) -> bool:
    h = headline.lower()
    return any(kw in h for kw in MA_KEYWORDS)


def _scale(mag, conf):
    return max(_cfg.MAX_POSITION_USD * 0.10, _cfg.MAX_POSITION_USD * mag * conf)


def run(scored_rows, regime=None, ma_filter=True, trail=MA_TRAIL, max_hold=MA_HOLD,
        min_mag=0.35, min_conf=0.45):
    trades, seen = [], {}
    ma_hits = 0
    for row in scored_rows:
        mag, conf = row["magnitude"], row["confidence"]
        req = min_conf + (1 - mag) * _cfg.CONFIDENCE_SLOPE
        if mag < min_mag or conf < req:
            continue
        if ma_filter and not is_ma_article(row.get("headline", "")):
            continue
        if ma_filter:
            ma_hits += 1
        d = row["created_at"].date()
        if regime and not regime(d):
            continue
        ds = seen.setdefault(d.isoformat(), set())
        for tk in row["tickers"][:2]:
            if tk in ds or tk in ("BTC", "ETH") or not is_valid_stock_ticker(tk):
                continue
            ds.add(tk)
            t = simulate_pead(tk, row["created_at"], _scale(mag, conf),
                              trail=trail, max_hold=max_hold)
            if t:
                trades.append(t)
    return trades, ma_hits


def line(tag, s):
    return (f"  {tag:40}  n={s['trades']:>4}  Sharpe={s['sharpe']:>6.2f}"
            f"  P&L=${s['total_pnl']:>9,.0f}  win={s['win_rate']:>4.1f}%"
            f"  maxDD=${s['max_dd']:>7,.0f}")


def main():
    print("M&A ARBITRAGE BACKTEST")
    print(f"Trail: {MA_TRAIL*100:.0f}% (tight — deal-break protection)  "
          f"Max hold: {MA_HOLD}d\n")

    for label, end_dt, days, cache in WINDOWS:
        if not Path(cache).exists():
            print(f"  [{label}] cache not found — skipping"); continue

        print(f"═══ {label} ═══")
        scored = tune_v2.load_scored_from_dual_cache(Path(cache), "bull", end_dt, days)
        reg = build_regime(end_dt, days, 200)

        # Count M&A articles
        ma_articles = [r for r in scored if is_ma_article(r.get("headline", ""))]
        print(f"  M&A headlines detected: {len(ma_articles)} / {len(scored)} "
              f"({len(ma_articles)/max(len(scored),1)*100:.1f}%)")

        # M&A arb — no regime gate (regime-independent is the thesis)
        ma_ng, hits = run(scored, regime=None, ma_filter=True)
        print(f"  M&A signals that passed gate: {hits}")
        print(line("M&A arb — NO regime gate", compute_stats(ma_ng)))

        # M&A arb — with regime gate
        ma_rg, _ = run(scored, regime=reg, ma_filter=True)
        print(line("M&A arb — regime gated",   compute_stats(ma_rg)))

        # Baseline: all bullish signals, same trail/hold (compare like-for-like)
        base_ng, _ = run(scored, regime=None, ma_filter=False, trail=0.10, max_hold=30)
        print(line("All bullish (no M&A filter, 10%/30d)", compute_stats(base_ng),))

        # Tight stop on all bullish (same stop as M&A, fair comparison)
        base_tight, _ = run(scored, regime=None, ma_filter=False)
        print(line("All bullish (tight 8%/90d stop)", compute_stats(base_tight)))

        # Sample M&A headlines for inspection
        print("\n  Sample M&A headlines:")
        shown = set()
        for r in scored:
            h = r.get("headline", "")
            if is_ma_article(h) and h not in shown:
                tks = r.get("tickers", [])
                print(f"    [{r['magnitude']:.2f}/{r['confidence']:.2f}] {tks} — {h[:80]}")
                shown.add(h)
                if len(shown) >= 6:
                    break
        print()


if __name__ == "__main__":
    main()
