"""
materiality_options_pnl.py — Stage-3 net-of-cost P&L gate for the materiality filter
on the CONVEX legs: news_call (Δ0.50) and lotto (Δ0.25 OTM).

Companion to materiality_pnl_test.py (which did the linear STOCK leg). The hypothesis:
a higher +5%-mover hit-rate should matter MORE for convex option payoffs than for a
linear stock position, because each extra winner is a multi-bagger, not a linear gain.

Same fair A/B as the stock test: both arms restricted to the llama3.2-materiality-scored
universe (what live would have); only the materiality≥τ gate differs. Each strategy uses
its EXACT live config (config.py). Net of realistic option spreads. LIVE regime gate.

Usage:  USE_YAHOO_BARS=1 .venv/bin/python materiality_options_pnl.py
"""
import os, json
os.environ.setdefault("USE_YAHOO_BARS", "1")
from datetime import datetime, timezone, timedelta
from pathlib import Path

import backtest as _bt
import tune_v2
import config as _cfg
from benchmark import compute_stats
from regime_filter import build_regime

SCORES = json.load(open("prompt_exp_scores.json"))
MODEL, PROMPT = "llama3.2", "materiality_fewshot"
TAUS = [0.10, 0.15, 0.20]
WINDOWS = [
    ("bull-meltup", datetime.now(timezone.utc) - timedelta(days=5), 180, "dual_score_cache.json"),
    ("2022-bear",   datetime(2022, 6, 30, tzinfo=timezone.utc), 90,  "bear_dual_cache.json"),
]

# exact live config per strategy
STRATS = {
    "news_call (Δ0.50)": dict(
        delta=_cfg.TARGET_DELTA, dte=(_cfg.MIN_DAYS_TO_EXPIRY + _cfg.MAX_DAYS_TO_EXPIRY) // 2,
        trail=_cfg.TRAILING_STOP_PCT, pos=_cfg.MAX_POSITION_USD, exit={"tiers": _cfg.EXIT_TIERS},
        gate=lambda m, c: not (m < _cfg.MIN_MAGNITUDE or c < _cfg.BASE_CONFIDENCE + (1 - m) * _cfg.CONFIDENCE_SLOPE)),
    "lotto (Δ0.25 OTM)": dict(
        delta=_cfg.LOTTO_TARGET_DELTA, dte=_cfg.LOTTO_MIN_DTE,
        trail=_cfg.LOTTO_EXIT_TIERS[0][1], pos=_cfg.LOTTO_POSITION_USD, exit={"tiers": _cfg.LOTTO_EXIT_TIERS},
        gate=lambda m, c: m >= _cfg.LOTTO_MIN_MAGNITUDE and c >= _cfg.LOTTO_MIN_CONFIDENCE),
}


def materiality_map(cache_file):
    raw = json.load(open(cache_file)); m = {}
    for v in raw.values():
        if not isinstance(v, dict):
            continue
        a = v.get("_article", {}) or {}; h = a.get("headline")
        if not h:
            continue
        m[h] = SCORES.get(f"{MODEL}:{PROMPT}:{_bt.cache_key(h, a.get('summary', ''))}")
    return m


def gate_rows(scored, gate, matmap, tau):
    out = []
    for r in scored:
        if not gate(r["magnitude"], r["confidence"]):
            continue
        ms = matmap.get(r["headline"])
        if ms is None:                      # restrict BOTH arms to the materiality-scored universe
            continue
        if tau is not None and ms < tau:
            continue
        out.append(r)
    return out


def sim_calls(rows, regime, delta, dte, trail, exit_params, pos_usd, spread_mult=1.0):
    save = {k: getattr(_bt, k) for k in
            ("TARGET_DELTA", "DTE_TARGET", "MAX_HOLD_DAYS", "TRAILING_STOP_PCT", "EXIT_PARAMS")}
    _bt.TARGET_DELTA = delta; _bt.DTE_TARGET = dte; _bt.MAX_HOLD_DAYS = dte
    _bt.TRAILING_STOP_PCT = trail; _bt.EXIT_PARAMS = exit_params
    trades, seen = [], set()
    try:
        for r in rows:
            d = r["created_at"].date()
            if regime and not regime(d):
                continue
            for tk in r["tickers"][:1]:
                if tk in ("BTC", "ETH") or not _bt.is_valid_stock_ticker(tk):
                    continue
                key = f"{d}_{tk}"
                if key in seen:
                    continue
                seen.add(key)
                sp = _bt.get_price_at(tk, r["created_at"])
                if not sp:
                    continue
                sz = max(pos_usd * 0.5, pos_usd * r["magnitude"] * r["confidence"])
                t = _bt.simulate_option_pnl(
                    tk, r["created_at"], sp, sz,
                    {"magnitude": r["magnitude"], "confidence": r["confidence"]},
                    option_type="call", exit_rule="tiered_profit", spread_mult=spread_mult)
                if t:
                    trades.append(t)
    finally:
        for k, v in save.items():
            setattr(_bt, k, v)
    return trades


def line(tag, s):
    avg = f"  avg=${s['total_pnl']/s['trades']:>6,.0f}" if s['trades'] else ""
    mb = ""
    return (f"  {tag:22} trades={s['trades']:>4}  P&L=${s['total_pnl']:>9,.0f}  "
            f"Sharpe={s['sharpe']:>6.2f}  win={s['win_rate']:>4.1f}%  maxDD=${s['max_dd']:>8,.0f}{avg}")


def run_one(label, end_dt, days, cache, name):
    """Run ONE strategy×window in this (fresh) process and print its block."""
    c = STRATS[name]
    matmap = materiality_map(cache)
    scored = tune_v2.load_scored_from_dual_cache(Path(cache), "bull", end_dt, days)
    reg = build_regime(end_dt, days, 200)
    print(f"═══ {label}  |  {name} ═══")
    base = compute_stats(sim_calls(gate_rows(scored, c["gate"], matmap, None),
                                   reg, c["delta"], c["dte"], c["trail"], c["exit"], c["pos"]))
    print(line("baseline (gate only)", base))
    for tau in TAUS:
        s = compute_stats(sim_calls(gate_rows(scored, c["gate"], matmap, tau),
                                    reg, c["delta"], c["dte"], c["trail"], c["exit"], c["pos"]))
        print(line(f"+ materiality ≥ {tau:.2f}", s))
    print(flush=True)


def main():
    import argparse, subprocess, sys
    ap = argparse.ArgumentParser()
    ap.add_argument("--window", help="run only this window label")
    ap.add_argument("--only", help="run only this strategy key")
    args = ap.parse_args()

    # Child mode: one strategy×window in this process (no cross-strategy state).
    if args.window and args.only:
        for label, end_dt, days, cache in WINDOWS:
            if label == args.window:
                run_one(label, end_dt, days, cache, args.only)
        return

    # Parent mode: spawn a FRESH process per (window, strategy). Necessary because
    # running many strategies in one process accumulates Yahoo data-fetch load and
    # silently drops trades from later strategies (observed lotto 116→73). Each
    # child stays light → reproducible, matches isolated runs.
    print("STAGE-3 net-of-cost P&L — materiality filter on the CONVEX option legs")
    print("Same universe both arms = articles with a llama3.2 materiality score. LIVE regime gate, net spreads.")
    print("(each strategy×window runs in its own process to avoid cross-strategy data-fetch contamination)\n")
    for label, *_ in WINDOWS:
        for name in STRATS:
            subprocess.run([sys.executable, __file__, "--window", label, "--only", name],
                           env={**os.environ, "USE_YAHOO_BARS": "1"})


if __name__ == "__main__":
    main()
