"""
lotto_backtest.py — "lottery ticket" cheap out-of-the-money calls.

Thesis: on the STRONGEST bullish signals (high magnitude AND high confidence),
buy cheap OTM short-dated calls for a convex payoff — small fixed cost, large
upside if the expected big swing happens. This deliberately leans INTO the
lottery-ticket profile the cost study flagged: the alpha lives in cheap, wide-
spread small-cap options, so we test it net of realistic spreads and report the
payoff DISTRIBUTION (not just mean), because convex bets live or die on the tail.

Compares, on the same high-conviction signal set, cross-regime:
  • lotto OTM calls  (low delta = cheap, convex)
  • standard ATM-ish calls (0.50 delta = the bot's news_call)
both net of base spreads, with a "let it run" exit.

Usage:  USE_YAHOO_BARS=1 .venv/bin/python lotto_backtest.py
"""
import os, statistics
from datetime import datetime, timezone, timedelta
from pathlib import Path

os.environ.setdefault("USE_YAHOO_BARS", "1")

import backtest as _bt
import tune_v2
from benchmark import compute_stats
from regime_filter import build_regime
import config as _cfg

WINDOWS = [
    ("bull-meltup", datetime.now(timezone.utc) - timedelta(days=5), 180, "dual_score_cache.json"),
    ("2022-bear",   datetime(2022, 6, 30, tzinfo=timezone.utc), 90,  "bear_dual_cache.json"),
]

# High-conviction gate — "very good bullish signal, high chance of a big swing"
LOTTO_MIN_MAG  = 0.70
LOTTO_MIN_CONF = 0.85
LOTTO_POS_USD  = 250     # small — it's a lottery ticket
DTE            = int(os.environ.get("LOTTO_DTE", "14"))   # live target (8-21 window); env-overridable for sweeps
MAX_HOLD       = int(os.environ.get("LOTTO_HOLD", "7"))   # matches live LOTTO_MAX_HOLD_DAYS; env-overridable


def _scale(mag, conf, base):
    return max(base * 0.5, base * mag * conf)


def run_variant(rows, regime, delta, exit_rule, exit_params, spread_mult=1.0):
    """Simulate calls at a given delta on the gated rows. Returns trade dicts."""
    # Patch backtest globals (same mechanism tune_v2 uses)
    save = {k: getattr(_bt, k) for k in
            ("TARGET_DELTA", "DTE_TARGET", "MAX_HOLD_DAYS", "TRAILING_STOP_PCT", "EXIT_PARAMS")}
    _bt.TARGET_DELTA      = delta
    _bt.DTE_TARGET        = DTE
    _bt.MAX_HOLD_DAYS     = MAX_HOLD
    _bt.TRAILING_STOP_PCT = 0.30          # wide — let the convex bet run
    _bt.EXIT_PARAMS       = exit_params
    trades, seen = [], set()
    try:
        for r in rows:
            d = r["created_at"].date()
            if regime and not regime(d):
                continue
            for tk in r["tickers"][:1]:
                if tk in ("BTC", "ETH") or not _bt.is_valid_stock_ticker(tk):
                    continue
                key = f"{d}_{tk}_{delta}"
                if key in seen:
                    continue
                seen.add(key)
                sp = _bt.get_price_at(tk, r["created_at"])
                if not sp:
                    continue
                t = _bt.simulate_option_pnl(
                    tk, r["created_at"], sp, _scale(r["magnitude"], r["confidence"], LOTTO_POS_USD),
                    {"magnitude": r["magnitude"], "confidence": r["confidence"]},
                    option_type="call", exit_rule=exit_rule, spread_mult=spread_mult)
                if t:
                    trades.append(t)
    finally:
        for k, v in save.items():
            setattr(_bt, k, v)
    return trades


def payoff_line(tag, trades):
    if not trades:
        return f"  {tag:34}  n=   0  (no data)"
    s = compute_stats(trades)
    rets = sorted((t["pnl_pct"] for t in trades), reverse=True)
    multibag = sum(1 for r in rets if r >= 100)          # ≥ +100% (2×+)
    big      = sum(1 for r in rets if r >= 300)          # ≥ +300% (4×+)
    top      = max((t["pnl_usd"] for t in trades), default=0)
    top10    = sum(sorted((t["pnl_usd"] for t in trades), reverse=True)[:10])
    tot      = s["total_pnl"]
    top10_sh = (top10 / tot * 100) if tot else 0
    return (f"  {tag:34}  n={s['trades']:>4}  Sharpe={s['sharpe']:>6.2f}"
            f"  P&L=${tot:>8,.0f}  win={s['win_rate']:>4.1f}%"
            f"  2x+={multibag:>3} 4x+={big:>2}  top=${top:>6,.0f}"
            f"  top10={top10_sh:>4.0f}%")


def main():
    print("═══ LOTTO (cheap OTM calls) BACKTEST ═══")
    print(f"Gate: mag≥{LOTTO_MIN_MAG} AND conf≥{LOTTO_MIN_CONF}  |  size ${LOTTO_POS_USD}  "
          f"|  DTE {DTE}  |  NET of base spreads\n")

    tiered = {"tiers": [(1.0, 0.40), (3.0, 0.30), (float("inf"), 0.20)]}

    for label, end_dt, days, cache in WINDOWS:
        if not Path(cache).exists():
            print(f"  [{label}] cache not found — skipping"); continue
        print(f"═══ {label} ═══")
        scored = tune_v2.load_scored_from_dual_cache(Path(cache), "bull", end_dt, days)
        gated  = [r for r in scored
                  if r["magnitude"] >= LOTTO_MIN_MAG and r["confidence"] >= LOTTO_MIN_CONF]
        reg    = build_regime(end_dt, days, 200)
        print(f"  High-conviction signals: {len(gated)} / {len(scored)} "
              f"({len(gated)/max(len(scored),1)*100:.0f}%)")

        # Lotto OTM at a few deltas + standard ATM, regime-gated, tiered "let-it-run" exit
        for tag, delta in [("lotto Δ0.15 (deep OTM)", 0.15),
                           ("lotto Δ0.20 (OTM)",      0.20),
                           ("lotto Δ0.25 (OTM)",      0.25),
                           ("standard Δ0.50 (ATM)",   0.50)]:
            t = run_variant(gated, reg, delta, "tiered_profit", tiered, spread_mult=1.0)
            print(payoff_line(tag, t))
        # Frictionless reference for the OTM bet (shows how much spread eats)
        t_ff = run_variant(gated, reg, 0.20, "tiered_profit", tiered, spread_mult=0.0)
        print(payoff_line("lotto Δ0.20 — FRICTIONLESS", t_ff))
        print()

    print("═══ READ THIS ═══")
    print("  Lotto is a CONVEX bet: judge by tail (2x+/4x+ count, top trade) not just mean.")
    print("  spread_mult=1.0 = realistic costs. OTM options have the WIDEST spreads, so")
    print("  the frictionless-vs-net gap shows how brutal fills are. Low win% is expected;")
    print("  it only works if a few big winners pay for many small losers.")


if __name__ == "__main__":
    main()
