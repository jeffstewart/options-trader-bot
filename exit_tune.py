"""
exit_tune.py — tune the profit-taking / tiered-trail configuration cross-regime.

A single hard profit target caps the runners (the tail trades that carry the P&L).
A profit-TIERED trailing stop tightens as gains grow but never hard-exits a still-
climbing position — letting winners run while protecting gains. This sweeps several
tier configurations (lock-fast → let-it-run) plus a flat time-stop, on the robust
parameter combo, across both regimes.

Reports, per config: P&L / Sharpe / win% / maxDD AND a RUNNER-CAPTURE view (biggest
single trade, sum of top-10) so you can see whether winners are being cut early.

Usage:  USE_YAHOO_BARS=1 .venv/bin/python exit_tune.py
"""
from datetime import datetime, timezone, timedelta
from pathlib import Path

import backtest as _bt
import tune_v2
from backtest import get_price_at, is_valid_stock_ticker
from benchmark import compute_stats
import config as _cfg

# Robust param combo from the cross-regime tuning (the params we'd actually run)
COMBO = {"MIN_DTE": 14, "MAX_DTE": 21, "TARGET_DELTA": 0.50,
         "MIN_MAGNITUDE": 0.25, "BASE_CONFIDENCE": 0.70, "TRAILING_STOP_PCT": 0.10}

INF = float("inf")
# (name, exit_rule, EXIT_PARAMS overrides) — spectrum from lock-fast to let-it-run.
CONFIGS = [
    ("baseline_trail10",   "trail_premium", {}),
    ("hard_target_+75%",   "profit_target", {"target_pct": 0.75}),
    ("tier_lockfast",      "tiered_profit", {"tiers": [(0.5, 0.15), (1.0, 0.10), (INF, 0.07)]}),
    ("tier_balanced",      "tiered_profit", {"tiers": [(0.5, 0.20), (1.0, 0.15), (INF, 0.10)]}),
    ("tier_runner",        "tiered_profit", {"tiers": [(0.75, 0.25), (2.0, 0.18), (INF, 0.12)]}),
    ("tier_runner_wide",   "tiered_profit", {"tiers": [(1.0, 0.30), (3.0, 0.20), (INF, 0.13)]}),
    ("tier_runner+flat7",  "tiered_profit", {"tiers": [(0.75, 0.25), (2.0, 0.18), (INF, 0.12)],
                                             "flat_stop_days": 7, "flat_stop_progress": 0.20}),
    ("tier_balanced+flat7","tiered_profit", {"tiers": [(0.5, 0.20), (1.0, 0.15), (INF, 0.10)],
                                             "flat_stop_days": 7, "flat_stop_progress": 0.20}),
]
WINDOWS = [
    ("bull-meltup", datetime.now(timezone.utc) - timedelta(days=5), 180, "dual_score_cache.json"),
    ("2022-bear",   datetime(2022, 6, 30, tzinfo=timezone.utc),     90,  "bear_dual_cache.json"),
]
_DEFAULT_EP = dict(_bt.EXIT_PARAMS)


def apply_combo():
    _bt.MIN_DAYS_TO_EXPIRY = COMBO["MIN_DTE"]
    _bt.MAX_DAYS_TO_EXPIRY = COMBO["MAX_DTE"]
    _bt.DTE_TARGET = round((COMBO["MIN_DTE"] + COMBO["MAX_DTE"]) / 2)
    _bt.TARGET_DELTA = COMBO["TARGET_DELTA"]
    _bt.TRAILING_STOP_PCT = COMBO["TRAILING_STOP_PCT"]


def run(end_dt, days, cache, exit_rule, ep_overrides):
    apply_combo()
    _bt.EXIT_PARAMS = {**_DEFAULT_EP, **ep_overrides}
    scored = tune_v2.load_scored_from_dual_cache(Path(cache), "bull", end_dt, days)

    def scale(mag, conf):
        return max(_cfg.MAX_POSITION_USD * 0.10, _cfg.MAX_POSITION_USD * mag * conf)

    trades, seen = [], {}
    for row in scored:
        mag, conf = row["magnitude"], row["confidence"]
        if mag < COMBO["MIN_MAGNITUDE"]:
            continue
        if conf < COMBO["BASE_CONFIDENCE"] + (1 - mag) * _cfg.CONFIDENCE_SLOPE:
            continue
        ds = seen.setdefault(row["created_at"].date().isoformat(), set())
        for tk in row["tickers"][:2]:
            if tk in ds or tk in ("BTC", "ETH") or not is_valid_stock_ticker(tk):
                continue
            ds.add(tk)
            px = get_price_at(tk, row["created_at"])
            if not px or px < _cfg.MIN_STOCK_PRICE:
                continue
            t = _bt.simulate_option_pnl(tk, row["created_at"], px, scale(mag, conf),
                                        {"magnitude": mag, "confidence": conf, "reasoning": ""},
                                        option_type="call", exit_rule=exit_rule)
            if t:
                trades.append(t)
    return trades


def main():
    print(f"Exit/profit-taking tuning on robust combo {COMBO}\n")
    rows = {}
    for label, end_dt, days, cache in WINDOWS:
        print(f"═══ {label} ═══")
        print(f"  {'config':22} {'P&L':>10} {'Sharpe':>7} {'win%':>6} {'maxDD':>9} {'bigwin':>9} {'top10':>10}")
        for name, rule, ov in CONFIGS:
            trades = run(end_dt, days, cache, rule, ov)
            s = compute_stats(trades)
            pnls = sorted((t["pnl_usd"] for t in trades), reverse=True)
            bigwin = pnls[0] if pnls else 0
            top10 = sum(pnls[:10])
            rows[(label, name)] = (s, bigwin, top10)
            print(f"  {name:22} ${s['total_pnl']:>9,.0f} {s['sharpe']:>7.2f} {s['win_rate']:>5.1f}% "
                  f"${s['max_dd']:>8,.0f} ${bigwin:>8,.0f} ${top10:>9,.0f}")
        print()

    print("═══ cross-regime net P&L (bull + 2022) — 'runner' = bull bigwin ═══")
    for name, _, _ in CONFIGS:
        b = rows[("bull-meltup", name)][0]["total_pnl"]
        r = rows[("2022-bear", name)][0]["total_pnl"]
        bw = rows[("bull-meltup", name)][1]
        print(f"  {name:22} net ${b + r:>10,.0f}   (bull bigwin ${bw:,.0f})")


if __name__ == "__main__":
    main()
