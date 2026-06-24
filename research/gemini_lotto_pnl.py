"""
gemini_lotto_pnl.py — does a FRONTIER-class model (Gemini 2.5 Flash) beat local llama3.2 on
LOTTO selection? Scores a broad candidate pool with gemini-2.5-flash (thinking OFF) and with
llama3.2 (cached), ranks each, simulates top-N as lotto (Δ0.25/DTE14/hold3/3×cap), compares.

Gemini free tier is tight (~10 RPM / ~250 req/day) so this is HEAVILY paced and the pool is
small → treat as DIRECTIONAL; the cache (prompt_exp_scores.json) accumulates across days toward
a reliable read. The head-to-head on IDENTICAL candidates is the meaningful part.

Usage:  USE_YAHOO_BARS=1 .venv/bin/python gemini_lotto_pnl.py
"""
import os, re, json, time, statistics
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
from prompt_lab import PROMPTS, extract_score

INF = float("inf")
DELTA, DTE, HOLD, POS = 0.25, 14, 3, cfg.LOTTO_POSITION_USD
EXIT = {"tiers": [(1.0, 0.40), (3.0, 0.30), (INF, 0.20)], "hard_target": 3.0}
SAMPLE, TOPN = 300, 35       # attempt up to 300 to max out the daily RPD cap
PACE = 6.5                    # ~10 RPM free-tier limit
MAX_CONSEC_ERR = 15          # early-stop: daily cap hit → stop wasting paced 429 retries
END = datetime.now(timezone.utc) - timedelta(days=5)
SYS = PROMPTS["materiality_fewshot"]
SCORES_PATH = "prompt_exp_scores.json"
GMODEL = os.environ.get("GEMINI_MODEL", "gemini-2.5-flash")


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


def gemini_score(client, h, body):
    r = client.chat.completions.create(model=GMODEL, temperature=0.1, max_tokens=200,
        reasoning_effort="none",
        messages=[{"role": "system", "content": SYS},
                  {"role": "user", "content": f"Headline: {h}\n\nBody: {body[:1200]}\n\nJSON only."}])
    raw = r.choices[0].message.content.strip()
    obj = json.loads(re.search(r'\{.*\}', raw, re.S).group(0))
    return extract_score(obj)


def collect_gemini(client, cands, sc):
    lats, n_err, consec = [], 0, 0
    for c in cands:
        key = f"{GMODEL}:materiality_fewshot:{c['ck']}"
        if key in sc and sc[key] is not None:
            continue
        t0 = time.time()
        try:
            sc[key] = gemini_score(client, c["h"], c["body"])
            lats.append((time.time() - t0) * 1000); consec = 0
            Path(SCORES_PATH).write_text(json.dumps(sc))   # persist each success (daemon-safe)
        except Exception:
            sc[key] = None; n_err += 1; consec += 1
        time.sleep(PACE)
        if consec >= MAX_CONSEC_ERR:
            print(f"  [early-stop] {consec} consecutive errors — daily cap likely reached "
                  f"({len(lats)} scored this run)", flush=True)
            break
    Path(SCORES_PATH).write_text(json.dumps(sc))
    return (statistics.median(lats) if lats else 0), n_err


def sim_topn(cands, sc, model):
    pre = f"{model}:materiality_fewshot:"
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
    g = OpenAI(base_url=os.environ["GEMINI_BASE_URL"], api_key=os.environ["GEMINI_API_KEY"])
    sc = json.loads(Path(SCORES_PATH).read_text()) if Path(SCORES_PATH).exists() else {}
    cands = candidates()
    have = sum(1 for c in cands if f"{GMODEL}:materiality_fewshot:{c['ck']}" in sc)
    print(f"═══ GEMINI vs llama3.2 — LOTTO P&L, broad pool n={len(cands)}, top-{TOPN} (Δ{DELTA}/DTE{DTE}/hold{HOLD}/3×cap) ═══")
    print(f"  ({have}/{len(cands)} gemini scores already cached; scoring the rest paced ~{PACE}s/call)\n")
    lat, n_err = collect_gemini(g, cands, sc)
    print(f"  {'scorer':28} {'med_lat_ms':>10} {'scored':>7} {'trades':>7} {'P&L':>10} {'Sharpe':>7} {'2x+':>4} {'4x+':>4}")
    for model, latv in [(GMODEL, lat), ("llama3.2", 0)]:
        n_scored, trades = sim_topn(cands, sc, model)
        if trades:
            st = compute_stats(trades)
            x2 = sum(1 for t in trades if t["pnl_pct"] >= 100); x4 = sum(1 for t in trades if t["pnl_pct"] >= 300)
            print(f"  {model:28} {latv:>10.0f} {n_scored:>7} {len(trades):>7} "
                  f"{'${:+,.0f}'.format(st['total_pnl']):>10} {st['sharpe']:>7.2f} {x2:>4} {x4:>4}")
        else:
            print(f"  {model:28} {latv:>10.0f} {n_scored:>7} {0:>7}  (no trades)")
    if n_err:
        print(f"\n  ({n_err} gemini scoring errors/rate-limits)")
    print("\nRead: same candidates, scorer only RANKS. Gemini beats llama3.2 here only if its top-N")
    print("picks make more lotto P&L. Small pool → DIRECTIONAL; cache builds toward reliable over days.")


if __name__ == "__main__":
    main()
