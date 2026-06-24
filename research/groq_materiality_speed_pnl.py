"""
groq_materiality_speed_pnl.py (v2, corrected) — materiality_fewshot on every chat-capable
Groq model: clean SPEED (latency) + lotto P&L, fixing v1's two flaws:
  • BROAD candidate pool (NOT pre-filtered to high-conviction) so the materiality ranking
    actually selects — sampled evenly across the window, sized to fit the tightest bucket.
  • Dedicated latency loop (12 fresh paced calls/model on the real prompt) so we measure
    inference speed, not 429 backoff. All calls paced (sleep) to avoid self-throttling.

Each model has its own Groq bucket; scores cached to prompt_exp_scores.json (resumable).
Uses the TEST key.  Usage:  USE_YAHOO_BARS=1 .venv/bin/python groq_materiality_speed_pnl.py
"""
import os, json, time, statistics
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
SAMPLE = int(os.environ.get("GS_SAMPLE", "160"))   # broad pool size (env-overridable)
TOPN   = int(os.environ.get("GS_TOPN", "60"))      # select top-N
PACE = 0.2                    # sleep between calls — avoid RPM/TPM bursts
END = datetime.now(timezone.utc) - timedelta(days=5)
SYS = PROMPTS["materiality_fewshot"]
SCORES_PATH = "prompt_exp_scores.json"
_DEFAULT_MODELS = ["llama-3.1-8b-instant", "llama-3.3-70b-versatile",
                   "meta-llama/llama-4-scout-17b-16e-instruct", "openai/gpt-oss-20b",
                   "openai/gpt-oss-120b", "qwen/qwen3-32b"]
MODELS = [m.strip() for m in os.environ.get("GS_MODELS", ",".join(_DEFAULT_MODELS)).split(",") if m.strip()]
INCLUDE_OLLAMA = os.environ.get("GS_OLLAMA", "1") == "1"
OLLAMA = OpenAI(base_url=cfg.OLLAMA_BASE_URL, api_key="ollama")


def candidates():
    raw = json.load(open("dual_score_cache.json"))
    reg = build_regime(END, 180, 200)
    out = []
    for v in raw.values():
        if not isinstance(v, dict):
            continue
        a = v.get("_article", {}) or {}
        b = v.get("bullish", {}) or {}
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
        # NO mag/conf pre-filter — BROAD bullish pool so materiality can rank/select.
        out.append({"tk": tks[0], "dt": dt, "h": h, "body": body, "ck": cache_key(h, body)})
    out.sort(key=lambda c: c["dt"])
    if len(out) > SAMPLE:                      # even broad sample across the window
        step = len(out) / SAMPLE
        out = [out[int(i * step)] for i in range(SAMPLE)]
    return out


def measure_latency(client, model, cands, n=12):
    lats = []
    for c in cands[:n]:
        t0 = time.time()
        try:
            score_article(client, model, SYS, c["h"], c["body"])
        except Exception:
            pass
        lats.append((time.time() - t0) * 1000)
        time.sleep(PACE)
    return statistics.median(lats) if lats else 0


def score_pool(client, model, cands, sc):
    scored, n429 = [], 0
    for c in cands:
        key = f"{model}:materiality_fewshot:{c['ck']}"
        if key in sc and sc[key] is not None:
            scored.append((c["tk"], c["dt"], sc[key])); continue
        try:
            s = score_article(client, model, SYS, c["h"], c["body"])
        except Exception as e:
            s = None
            if "429" in str(e) or "rate" in str(e).lower():
                n429 += 1
        sc[key] = s
        if s is not None:
            scored.append((c["tk"], c["dt"], s))
        time.sleep(PACE)
    Path(SCORES_PATH).write_text(json.dumps(sc))
    return scored, n429


def sim_topn(scored):
    scored = sorted([x for x in scored if x[2] is not None], key=lambda x: x[2], reverse=True)
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
    return trades


def main():
    g = OpenAI(base_url=os.environ["LAB_HOSTED_BASE_URL"],
               api_key=os.environ.get("LAB_HOSTED_TEST_KEY") or os.environ["LAB_HOSTED_KEY"])
    sc = json.loads(Path(SCORES_PATH).read_text()) if Path(SCORES_PATH).exists() else {}
    cands = candidates()
    print(f"═══ materiality_fewshot SPEED + LOTTO P&L (v2) — broad pool n={len(cands)}, top-{TOPN}, Δ{DELTA}/DTE{DTE}/hold{HOLD}/3×cap ═══\n")
    print(f"  {'model':42} {'med_lat_ms':>10} {'429s':>5} {'trades':>7} {'P&L':>10} {'Sharpe':>7} {'2x+':>4} {'4x+':>4}")
    rows = ([("ollama:llama3.2 (LIVE/local)", OLLAMA, "llama3.2")] if INCLUDE_OLLAMA else []) + [(m, g, m) for m in MODELS]
    for label, client, model in rows:
        lat = measure_latency(client, model, cands)
        scored, n429 = score_pool(client, model, cands, sc)
        trades = sim_topn(scored)
        if trades:
            st = compute_stats(trades)
            x2 = sum(1 for t in trades if t["pnl_pct"] >= 100)
            x4 = sum(1 for t in trades if t["pnl_pct"] >= 300)
            pnl = "${:+,.0f}".format(st["total_pnl"])
            print(f"  {label:42} {lat:>10.0f} {n429:>5} {len(trades):>7} {pnl:>10} {st['sharpe']:>7.2f} {x2:>4} {x4:>4}")
        else:
            print(f"  {label:42} {lat:>10.0f} {n429:>5} {0:>7}  (no trades)")
    print("\nRead: med_lat_ms = entry speed (lower=faster). P&L/Sharpe/4x+ on a 160-pool (smaller")
    print("selection room than lotto_pnl_by_scorer's 4942 → treat P&L as directional). Want FAST + good.")


if __name__ == "__main__":
    main()
