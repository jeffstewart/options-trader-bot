"""
exit_rules_test.py — compare position-closing rules cross-regime (real Yahoo data).

Holds the parameter combo fixed and varies ONLY the exit rule, so we isolate the
effect of close logic. Runs every rule on both the bull-melt-up and 2022-bear
windows and reports P&L / Sharpe / win% / max-DD / avg-win:avg-loss.

Rules (see backtest.simulate_option_pnl / EXIT_PARAMS):
  trail_premium (baseline) · profit_target · dte_stop · underlying_trail · tiered_trail

Usage:  USE_YAHOO_BARS=1 .venv/bin/python exit_rules_test.py
"""
from datetime import datetime, timezone, timedelta
from pathlib import Path

import backtest as _bt
import tune_v2
from backtest import get_price_at, is_valid_stock_ticker
from benchmark import compute_stats
import config as _cfg

# Fixed combo = the winning bull params (isolate exit-rule effect)
COMBO = {"MIN_DTE": 14, "MAX_DTE": 21, "TARGET_DELTA": 0.50,
         "MIN_MAGNITUDE": 0.25, "BASE_CONFIDENCE": 0.65, "TRAILING_STOP_PCT": 0.15}
RULES = ["trail_premium", "profit_target", "dte_stop", "underlying_trail", "tiered_trail"]
WINDOWS = [
    ("bull-meltup", datetime.now(timezone.utc) - timedelta(days=5), 180, "dual_score_cache.json"),
    ("2022-bear",   datetime(2022, 6, 30, tzinfo=timezone.utc),     90,  "bear_dual_cache.json"),
]


def apply_combo():
    _bt.MIN_DAYS_TO_EXPIRY = COMBO["MIN_DTE"]
    _bt.MAX_DAYS_TO_EXPIRY = COMBO["MAX_DTE"]
    _bt.DTE_TARGET = round((COMBO["MIN_DTE"] + COMBO["MAX_DTE"]) / 2)
    _bt.TARGET_DELTA = COMBO["TARGET_DELTA"]
    _bt.TRAILING_STOP_PCT = COMBO["TRAILING_STOP_PCT"]


def run(end_dt, days, cache, exit_rule):
    apply_combo()
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
    return compute_stats(trades)


def fmt(rule, s):
    awal = (s["avg_win"] / abs(s["avg_loss"])) if s["avg_loss"] else 0
    return (f"  {rule:16} trades={s['trades']:>4}  P&L=${s['total_pnl']:>10,.0f}  "
            f"Sharpe={s['sharpe']:>5.2f}  win={s['win_rate']:>4.1f}%  "
            f"maxDD=${s['max_dd']:>9,.0f}  W/L={awal:>4.1f}")


def main():
    print(f"Exit-rule comparison (combo fixed: {COMBO})\n")
    summary = {}
    for label, end_dt, days, cache in WINDOWS:
        print(f"═══ {label} ═══")
        for rule in RULES:
            s = run(end_dt, days, cache, rule)
            summary[(label, rule)] = s
            print(fmt(rule, s))
        print()
    # Net view: bull P&L + 2022 P&L per rule (which exit is best across regimes)
    print("═══ cross-regime net (bull P&L + 2022 P&L) ═══")
    for rule in RULES:
        b = summary[("bull-meltup", rule)]["total_pnl"]
        r = summary[("2022-bear", rule)]["total_pnl"]
        print(f"  {rule:16} bull ${b:>10,.0f}  +  2022 ${r:>9,.0f}  =  net ${b + r:>10,.0f}")


if __name__ == "__main__":
    main()
