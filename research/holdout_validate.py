"""
holdout_validate.py — out-of-sample check of the full proposed package before going live.

Package = robust params (DTE14-21 / Δ0.50 / Conf0.70 / Trail0.10) + tier_runner_wide
profit-trail exit. We tuned tiers on the full window, so confirm they hold on the
HELD-OUT 20% (and aren't worse than the baseline trailing stop). Also checks the
2022 holdout as a tail-risk read.

GO if: candidate holdout Sharpe doesn't collapse vs its training split (generalizes)
AND candidate ≥ baseline on the bull holdout.

Usage:  USE_YAHOO_BARS=1 .venv/bin/python holdout_validate.py
"""
from datetime import datetime, timezone, timedelta
from pathlib import Path

import backtest as _bt
import tune_v2
from backtest import get_price_at, is_valid_stock_ticker
from benchmark import compute_stats
import config as _cfg

COMBO = {"MIN_DTE": 14, "MAX_DTE": 21, "TARGET_DELTA": 0.50,
         "MIN_MAGNITUDE": 0.25, "BASE_CONFIDENCE": 0.70, "TRAILING_STOP_PCT": 0.10}
INF = float("inf")
RUNNER_WIDE = {"tiers": [(1.0, 0.30), (3.0, 0.20), (INF, 0.13)]}
_DEFAULT_EP = dict(_bt.EXIT_PARAMS)
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


def run(split, exit_rule, ep_overrides):
    apply_combo()
    _bt.EXIT_PARAMS = {**_DEFAULT_EP, **ep_overrides}

    def scale(mag, conf):
        return max(_cfg.MAX_POSITION_USD * 0.10, _cfg.MAX_POSITION_USD * mag * conf)

    trades, seen = [], {}
    for row in split:
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


def line(tag, s):
    return (f"  {tag:34} trades={s['trades']:>4}  P&L=${s['total_pnl']:>9,.0f}  "
            f"Sharpe={s['sharpe']:>5.2f}  win={s['win_rate']:>4.1f}%  maxDD=${s['max_dd']:>8,.0f}")


def main():
    print(f"Holdout validation — package: {COMBO} + tier_runner_wide\n")
    results = {}
    for label, end_dt, days, cache in WINDOWS:
        scored = tune_v2.load_scored_from_dual_cache(Path(cache), "bull", end_dt, days)
        train, hold = tune_v2.split_holdout(scored, 0.20)
        print(f"═══ {label}  (train={len(train)} / holdout={len(hold)} articles) ═══")
        ct = run(train, "tiered_profit", RUNNER_WIDE); print(line("candidate · TRAIN", ct))
        ch = run(hold,  "tiered_profit", RUNNER_WIDE); print(line("candidate · HOLDOUT", ch))
        bh = run(hold,  "trail_premium", {});          print(line("baseline  · HOLDOUT", bh))
        results[label] = (ct, ch, bh)
        print()

    # Verdict (bull window is primary — where the regime-gated strategy trades)
    ct, ch, bh = results["bull-meltup"]
    generalizes = ch["sharpe"] >= 0.6 * ct["sharpe"] and ch["sharpe"] > 0
    beats_base  = ch["total_pnl"] >= bh["total_pnl"] or ch["sharpe"] >= bh["sharpe"]
    print("═══ VERDICT (bull holdout) ═══")
    print(f"  candidate generalizes (holdout Sharpe {ch['sharpe']:.2f} vs train {ct['sharpe']:.2f}): {generalizes}")
    print(f"  candidate ≥ baseline on holdout (P&L ${ch['total_pnl']:,.0f} vs ${bh['total_pnl']:,.0f}, "
          f"Sharpe {ch['sharpe']:.2f} vs {bh['sharpe']:.2f}): {beats_base}")
    print(f"  → {'GO — promote to config.py + bot' if (generalizes and beats_base) else 'NO-GO — revisit'}")


if __name__ == "__main__":
    main()
