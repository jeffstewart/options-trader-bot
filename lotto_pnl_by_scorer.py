"""
lotto_pnl_by_scorer.py — which MODEL×PROMPT picks the most profitable LOTTO trades?

rank-IC is a proxy; this uses the metric that matters — realized LOTTO P&L. Same candidate
articles+tickers (from the dual cache's bullish side) for every scorer; the scorer only
RANKS them. Each scorer's top-N picks are simulated as lotto trades at the CORRECTED live
config (Δ0.25, DTE 14, hold 3d, tiered trail + 3× cap). Uses cached scores in
prompt_exp_scores.json → NO new scoring / NO tokens.

Usage:  USE_YAHOO_BARS=1 .venv/bin/python lotto_pnl_by_scorer.py
"""
import os, json
os.environ.setdefault("USE_YAHOO_BARS", "1")
from datetime import datetime, timezone, timedelta
from pathlib import Path

import backtest as _bt
import config as cfg
from backtest import cache_key
from benchmark import compute_stats
from regime_filter import build_regime

INF = float("inf")
DELTA, DTE, HOLD, POS = 0.25, 14, 3, cfg.LOTTO_POSITION_USD
EXIT_PARAMS = {"tiers": [(1.0, 0.40), (3.0, 0.30), (INF, 0.20)], "hard_target": 3.0}
TOPN = 130
END = datetime.now(timezone.utc) - timedelta(days=5)
COMBOS = [
    ("llama3.2",            "materiality_fewshot"),
    ("llama3.1:8b",         "materiality_fewshot"),
    ("llama3.2",            "materiality"),
    ("llama3.2",            "lotto_swing"),
    ("llama3.2",            "surprise_score"),
    ("llama3.2",            "binary_gate"),
    ("openai/gpt-oss-20b",  "materiality_fewshot"),
    ("qwen/qwen3-32b",      "materiality_fewshot"),
]


def build_candidates():
    sc = json.load(open("prompt_exp_scores.json"))
    raw = json.load(open("dual_score_cache.json"))
    reg = build_regime(END, 180, 200)
    cands = []
    for v in raw.values():
        if not isinstance(v, dict):
            continue
        a = v.get("_article", {}) or {}
        b = v.get("bullish", {}) or {}
        tks = [t for t in (b.get("tickers") or []) if t not in ("BTC", "ETH")]
        h, ca = a.get("headline"), a.get("created_at")
        if not tks or not h or not ca:
            continue
        try:
            dt = datetime.fromisoformat(str(ca).replace("Z", "+00:00")).astimezone(timezone.utc)
        except Exception:
            continue
        if not (END - timedelta(days=180) <= dt <= END):
            continue
        if reg and not reg(dt.date()):
            continue
        cands.append((tks[0], dt, cache_key(h, a.get("summary", ""))))
    return sc, cands


def sim_topn(sc, cands, model, prompt):
    pre = f"{model}:{prompt}:"
    scored = [(tk, dt, sc[pre + ck]) for (tk, dt, ck) in cands if (pre + ck) in sc and sc[pre + ck] is not None]
    scored.sort(key=lambda x: x[2], reverse=True)
    save = {k: getattr(_bt, k) for k in ("TARGET_DELTA", "DTE_TARGET", "MAX_HOLD_DAYS", "TRAILING_STOP_PCT", "EXIT_PARAMS")}
    _bt.TARGET_DELTA, _bt.DTE_TARGET, _bt.MAX_HOLD_DAYS = DELTA, DTE, HOLD
    _bt.TRAILING_STOP_PCT, _bt.EXIT_PARAMS = 0.30, EXIT_PARAMS
    trades, seen, taken = [], set(), 0
    try:
        for tk, dt, s in scored:
            if taken >= TOPN:
                break
            k = f"{dt.date()}_{tk}"
            if k in seen or not _bt.is_valid_stock_ticker(tk):
                continue
            seen.add(k); taken += 1
            sp = _bt.get_price_at(tk, dt)
            if not sp:
                continue
            t = _bt.simulate_option_pnl(tk, dt, sp, POS, {"magnitude": 0.8, "confidence": 0.9},
                                        option_type="call", exit_rule="tiered_profit", spread_mult=1.0)
            if t:
                trades.append(t)
    finally:
        for kk, vv in save.items():
            setattr(_bt, kk, vv)
    return len(scored), trades


def main():
    print(f"═══ LOTTO P&L BY SCORER — top-{TOPN} picks, Δ{DELTA}/DTE{DTE}/hold{HOLD}d/3×cap ═══")
    print("(same candidate articles+tickers; scorer only RANKS; metric = realized lotto P&L)\n")
    sc, cands = build_candidates()
    print(f"candidate pool (bull-meltup, regime-gated): {len(cands)} articles\n")
    print(f"  {'model':22} {'prompt':20} {'scored':>7} {'trades':>7} {'P&L':>10} {'Sharpe':>7} {'2x+':>4} {'4x+':>4}")
    for model, prompt in COMBOS:
        n_scored, trades = sim_topn(sc, cands, model, prompt)
        if not trades:
            print(f"  {model:22} {prompt:20} {n_scored:>7} {0:>7}  (no trades / not cached)")
            continue
        s = compute_stats(trades)
        x2 = sum(1 for t in trades if t["pnl_pct"] >= 100)
        x4 = sum(1 for t in trades if t["pnl_pct"] >= 300)
        pnl = "${:+,.0f}".format(s["total_pnl"])
        print(f"  {model:22} {prompt:20} {n_scored:>7} {len(trades):>7} "
              f"{pnl:>10} {s['sharpe']:>7.2f} {x2:>4} {x4:>4}")
    print("\nRead: higher P&L/Sharpe + 4x+ tail = the scorer whose TOP picks make the best lotto")
    print("bets. Prompt rows (llama3.2) isolate the prompt; materiality_fewshot rows isolate the model.")


if __name__ == "__main__":
    main()
