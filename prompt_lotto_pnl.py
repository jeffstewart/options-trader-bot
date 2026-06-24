"""
prompt_lotto_pnl.py — does a new PROMPT beat materiality_fewshot on LOTTO P&L (the real metric)?

Scores a broad candidate pool with LOCAL llama3.2 (the live model, free, no rate limits) for each
prompt in PROMPTS_TO_TEST (cache-aware), ranks, and simulates each prompt's top-N picks as lotto
trades at the live config (Δ0.25/DTE14/hold3/3× cap). Same candidates for every prompt → the
prompt only changes the RANKING. Beat materiality_fewshot's broad-pool P&L (~+$8,063).

Usage:  USE_YAHOO_BARS=1 .venv/bin/python prompt_lotto_pnl.py
"""
import os, json
os.environ.setdefault("USE_YAHOO_BARS", "1")
from datetime import datetime, timezone, timedelta
from pathlib import Path
from openai import OpenAI
from dotenv import load_dotenv
load_dotenv("/Users/jeff/Claude/Trader/.env")

import backtest as _bt
import config as cfg
from backtest import cache_key
from benchmark import compute_stats
from regime_filter import build_regime
from prompt_lab import PROMPTS, score_article

INF = float("inf")
DELTA, DTE, HOLD, POS = 0.25, 14, 3, cfg.LOTTO_POSITION_USD
EXIT = {"tiers": [(1.0, 0.40), (3.0, 0.30), (INF, 0.20)], "hard_target": 3.0}
SAMPLE = int(os.environ.get("PL_SAMPLE", "600"))
TOPN = int(os.environ.get("PL_TOPN", "90"))
END = datetime.now(timezone.utc) - timedelta(days=5)
SCORES_PATH = "prompt_exp_scores.json"
MODEL = "llama3.2"
PROMPTS_TO_TEST = os.environ.get("PL_PROMPTS", "materiality_fewshot,materiality_fewshot_v2").split(",")
OLLAMA = OpenAI(base_url=cfg.OLLAMA_BASE_URL, api_key="ollama")


def candidates():
    raw = json.load(open("dual_score_cache.json"))
    reg = build_regime(END, 180, 200)
    out = []
    for v in raw.values():
        if not isinstance(v, dict):
            continue
        a, b = v.get("_article", {}) or {}, v.get("bullish", {}) or {}
        tks = [t for t in (b.get("tickers") or []) if t not in ("BTC", "ETH")]
        h, ca, body = a.get("headline"), a.get("created_at"), a.get("summary", "")
        if not tks or not h or not ca:
            continue
        try:
            dt = datetime.fromisoformat(str(ca).replace("Z", "+00:00")).astimezone(timezone.utc)
        except Exception:
            continue
        if not (END - timedelta(days=180) <= dt <= END) or (reg and not reg(dt.date())):
            continue
        out.append({"tk": tks[0], "dt": dt, "h": h, "body": body, "ck": cache_key(h, body)})
    out.sort(key=lambda c: c["dt"])
    if len(out) > SAMPLE:
        step = len(out) / SAMPLE
        out = [out[int(i * step)] for i in range(SAMPLE)]
    return out


def score_pool(prompt, cands, sc):
    sys = PROMPTS[prompt]
    n_fresh = 0
    for c in cands:
        key = f"{MODEL}:{prompt}:{c['ck']}"
        if key in sc and sc[key] is not None:
            continue
        try:
            sc[key] = score_article(OLLAMA, MODEL, sys, c["h"], c["body"])
        except Exception:
            sc[key] = None
        n_fresh += 1
        if n_fresh % 25 == 0:
            Path(SCORES_PATH).write_text(json.dumps(sc))
    Path(SCORES_PATH).write_text(json.dumps(sc))
    return n_fresh


def sim_topn(cands, sc, prompt):
    pre = f"{MODEL}:{prompt}:"
    scored = sorted([(c["tk"], c["dt"], sc[pre + c["ck"]]) for c in cands
                     if (pre + c["ck"]) in sc and sc[pre + c["ck"]] is not None],
                    key=lambda x: x[2], reverse=True)
    save = {k: getattr(_bt, k) for k in ("TARGET_DELTA", "DTE_TARGET", "MAX_HOLD_DAYS", "TRAILING_STOP_PCT", "EXIT_PARAMS")}
    _bt.TARGET_DELTA, _bt.DTE_TARGET, _bt.MAX_HOLD_DAYS = DELTA, DTE, HOLD
    _bt.TRAILING_STOP_PCT, _bt.EXIT_PARAMS = 0.30, EXIT
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
    sc = json.loads(Path(SCORES_PATH).read_text()) if Path(SCORES_PATH).exists() else {}
    cands = candidates()
    print(f"═══ PROMPT × LOTTO P&L (llama3.2, broad pool n={len(cands)}, top-{TOPN}, Δ{DELTA}/DTE{DTE}/hold{HOLD}/3×cap) ═══\n")
    print(f"  {'prompt':28} {'scored':>7} {'trades':>7} {'P&L':>10} {'Sharpe':>7} {'2x+':>4} {'4x+':>4}")
    for prompt in PROMPTS_TO_TEST:
        nf = score_pool(prompt, cands, sc)
        n_scored, trades = sim_topn(cands, sc, prompt)
        if trades:
            st = compute_stats(trades)
            x2 = sum(1 for t in trades if t["pnl_pct"] >= 100); x4 = sum(1 for t in trades if t["pnl_pct"] >= 300)
            tag = f"(+{nf} fresh)" if nf else ""
            print(f"  {prompt:28} {n_scored:>7} {len(trades):>7} "
                  f"{'${:+,.0f}'.format(st['total_pnl']):>10} {st['sharpe']:>7.2f} {x2:>4} {x4:>4}  {tag}")
        else:
            print(f"  {prompt:28} {n_scored:>7} {0:>7}  (no trades)")
    print("\nRead: v2 (concrete few-shot, down-rate analyst notes + catch catalysts) beats baseline")
    print("materiality_fewshot only if its top-N picks make MORE lotto P&L on the same candidates.")


if __name__ == "__main__":
    main()
