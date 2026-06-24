"""
materiality_pnl_test.py — Stage-3 net-of-cost P&L gate for the materiality filter.

The gate test (materiality_gate_test.py) showed materiality≥τ lifts WINNER-RATE on top
of the live mag×conf gate. Winner-rate ≠ dollars. This runs the actual STOCK strategy
backtest (real cost model from stock_backtest.simulate_stock: 0.10% RT slippage, 10%
trailing stop, ≤30d hold, data-sanity guards) with vs without the materiality gate and
compares net P&L / Sharpe / win% / maxDD.

FAIR COMPARISON: both arms are restricted to the SAME universe — articles that have a
llama3.2 materiality score (what live would have, since the bot scores every article).
The only difference between arms is the materiality≥τ gate. This isolates the filter's
effect from scoring coverage. (Bull universe = the held-out TEST split, ~900 articles;
2022 = the 1,500-event sample.)

Usage:  USE_YAHOO_BARS=1 .venv/bin/python materiality_pnl_test.py
"""
import json
from pathlib import Path
from datetime import datetime, timezone, timedelta

import backtest as _bt
import tune_v2
import config as _cfg
import stock_backtest as sb
from backtest import is_valid_stock_ticker
from benchmark import compute_stats
from regime_filter import build_regime

SCORES = json.load(open("prompt_exp_scores.json"))
MODEL, PROMPT = "llama3.2", "materiality_fewshot"
TAUS = [0.10, 0.15, 0.20]          # candidate gate thresholds
WINDOWS = sb.WINDOWS               # reuse identical (label, end_dt, days, cache)


def materiality_map(cache_file):
    """headline -> llama3.2 materiality score (None if parse-failed / unscored)."""
    raw = json.load(open(cache_file))
    m = {}
    for v in raw.values():
        if not isinstance(v, dict):
            continue
        a = v.get("_article", {}) or {}
        h = a.get("headline")
        if not h:
            continue
        ck = _bt.cache_key(h, a.get("summary", ""))
        m[h] = SCORES.get(f"{MODEL}:{PROMPT}:{ck}")   # may be None/missing
    return m


def run(end_dt, days, cache, matmap, regime=None, tau=None):
    """Stock backtest restricted to materiality-scored articles; optional materiality≥tau gate."""
    scored = tune_v2.load_scored_from_dual_cache(Path(cache), "bull", end_dt, days)

    def scale(mag, conf):
        return max(_cfg.MAX_POSITION_USD * 0.10, _cfg.MAX_POSITION_USD * mag * conf)

    trades, seen = [], {}
    for row in scored:
        mag, conf = row["magnitude"], row["confidence"]
        if mag < _cfg.MIN_MAGNITUDE or conf < _cfg.BASE_CONFIDENCE + (1 - mag) * _cfg.CONFIDENCE_SLOPE:
            continue
        mscore = matmap.get(row["headline"])
        if mscore is None:           # not in the materiality-scored universe → exclude from BOTH arms
            continue
        if tau is not None and mscore < tau:
            continue
        d = row["created_at"].date()
        if regime is not None and not regime(d):
            continue
        ds = seen.setdefault(d.isoformat(), set())
        for tk in row["tickers"][:2]:
            if tk in ds or tk in ("BTC", "ETH") or not is_valid_stock_ticker(tk):
                continue
            ds.add(tk)
            t = sb.simulate_stock(tk, row["created_at"], scale(mag, conf))
            if t:
                trades.append(t)
    return trades


def line(tag, s, base_pnl=None):
    extra = ""
    if s["trades"]:
        avg = s["total_pnl"] / s["trades"]
        extra = f"  avg=${avg:>6,.0f}"
    return (f"  {tag:24} trades={s['trades']:>4}  P&L=${s['total_pnl']:>9,.0f}  "
            f"Sharpe={s['sharpe']:>5.2f}  win={s['win_rate']:>4.1f}%  maxDD=${s['max_dd']:>8,.0f}{extra}")


def main():
    print("STAGE-3 net-of-cost P&L — materiality filter on the STOCK strategy")
    print(f"(cost model: {sb.STOCK_SLIPPAGE*100:.2f}% RT slippage, {sb.STOCK_TRAIL*100:.0f}% trail, ≤{sb.MAX_HOLD}d)")
    print("Same universe both arms = articles with a llama3.2 materiality score.\n")

    for label, end_dt, days, cache in WINDOWS:
        matmap = materiality_map(cache)
        reg = build_regime(end_dt, days, 200)
        for gate_name, regime in [("SPY>200d regime gate (LIVE)", reg), ("no regime gate", None)]:
            print(f"═══ {label}  |  {gate_name} ═══")
            base = compute_stats(run(end_dt, days, cache, matmap, regime=regime, tau=None))
            print(line("baseline (gate only)", base))
            for tau in TAUS:
                s = compute_stats(run(end_dt, days, cache, matmap, regime=regime, tau=tau))
                print(line(f"+ materiality ≥ {tau:.2f}", s))
            print()


if __name__ == "__main__":
    main()
