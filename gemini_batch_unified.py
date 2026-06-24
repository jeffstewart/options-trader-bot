"""
gemini_batch_unified.py — same batched-Gemini-vs-llama3.2 lotto test, but on the LIVE unified_v1
prompt. llama's unified_v1 scores are already cached for the full pool (unified_scores.json);
Gemini is batch-scored with unified_v1's magnitude calibration. Compares top-N lotto P&L.

Usage:  USE_YAHOO_BARS=1 .venv/bin/python -u gemini_batch_unified.py
"""
import os, json, time
os.environ.setdefault("USE_YAHOO_BARS", "1")
from pathlib import Path
from openai import OpenAI
from dotenv import load_dotenv
load_dotenv("/Users/jeff/Claude/Trader/.env")

import gemini_lotto_pnl as gl
import gemini_batch_score as gbs
import backtest as _bt
from prompt_lab import extract_score
from benchmark import compute_stats

GMODEL, SCORES_PATH = gl.GMODEL, gl.SCORES_PATH
BATCH, PACE = 50, 13.0

# unified_v1's magnitude calibration, adapted for compact batched output.
UNIFIED_BATCH_SYS = (
    "You are a quantitative equity signal scorer. For EACH numbered news item, score it for a "
    "BULLISH long-call trade on its primary ticker.\n"
    "magnitude 0-1 = how likely the news MEANINGFULLY (and not-yet-priced) moves the stock:\n"
    "  0.0-0.2 noise/routine filings/minor analyst reiterations; 0.2-0.4 routine (in-line earnings, "
    "small contract wins, minor upgrades); 0.4-0.6 meaningful (solid beat, notable partnership, "
    "guidance raise); 0.6-0.8 strong (significant beat, major acquisition, landmark FDA approval); "
    "0.8-1.0 transformative (company-defining deal, paradigm approval).\n"
    "confidence 0-1 = certainty about the bullish direction.\n"
    "Most news is routine — be conservative; reserve magnitude 0.7+ for genuinely exceptional, "
    "RESOLVED events. An analyst rating/price-target is NOT a surprise → magnitude < 0.15. Macro / "
    "no specific company → magnitude 0.\n"
    'Respond with a JSON ARRAY ONLY, one object per item in order: '
    '[{"i":1,"magnitude":0.6,"confidence":0.8},{"i":2,"magnitude":0.1,"confidence":0.5}]'
)


def sim_topn(cands, sc, model, prompt):
    pre = f"{model}:{prompt}:"
    scored = sorted([(c["tk"], c["dt"], sc[pre + c["ck"]]) for c in cands
                     if sc.get(pre + c["ck"]) is not None], key=lambda x: x[2], reverse=True)
    save = {k: getattr(_bt, k) for k in ("TARGET_DELTA", "DTE_TARGET", "MAX_HOLD_DAYS", "TRAILING_STOP_PCT", "EXIT_PARAMS")}
    _bt.TARGET_DELTA, _bt.DTE_TARGET, _bt.MAX_HOLD_DAYS = gl.DELTA, gl.DTE, gl.HOLD
    _bt.TRAILING_STOP_PCT, _bt.EXIT_PARAMS = 0.30, gl.EXIT
    trades, seen, taken = [], set(), 0
    try:
        for tk, dt, s in scored:
            if taken >= gl.TOPN:
                break
            k = f"{dt.date()}_{tk}"
            if k in seen or not _bt.is_valid_stock_ticker(tk):
                continue
            seen.add(k); taken += 1
            sp = _bt.get_price_at(tk, dt)
            if not sp:
                continue
            t = _bt.simulate_option_pnl(tk, dt, sp, gl.POS, {"magnitude": 0.8, "confidence": 0.9},
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
    gl.SAMPLE = int(os.environ.get("UNI_SAMPLE", gl.SAMPLE))   # scale the pool (cache-aware, re-runnable)
    cands = gl.candidates()
    # seed llama3.2 unified_v1 floats from the already-cached unified_scores.json
    uni = json.load(open("unified_scores.json"))
    seeded = 0
    for c in cands:
        u = uni.get(f"unified_v1:{c['ck']}")
        if isinstance(u, dict):
            sc[f"llama3.2:unified_v1:{c['ck']}"] = extract_score(u); seeded += 1
    print(f"seeded {seeded}/{len(cands)} llama3.2 unified_v1 scores from unified_scores.json")

    pre = f"{GMODEL}:unified_v1:"
    todo = [c for c in cands if sc.get(pre + c["ck"]) is None]
    print(f"gemini unified_v1 to batch-score: {len(todo)} (batch {BATCH}, ~{PACE:.0f}s/req)")
    reqs = scored = consec_fail = 0
    for s in range(0, len(todo), BATCH):
        batch = todo[s:s + BATCH]
        try:
            res = gbs.score_batch(g, batch, UNIFIED_BATCH_SYS); reqs += 1; consec_fail = 0
            for i, c in enumerate(batch):
                if i in res:
                    sc[pre + c["ck"]] = extract_score(res[i]); scored += 1
            Path(SCORES_PATH).write_text(json.dumps(sc))
            print(f"  batch {s//BATCH+1}: {len(res)}/{len(batch)} scored (cum {scored})", flush=True)
        except Exception as e:
            reqs += 1; consec_fail += 1
            print(f"  batch {s//BATCH+1} FAILED: {repr(e)[:110]}", flush=True)
            if consec_fail >= 3:
                print(f"  [early-stop] {consec_fail} consecutive failures — daily cap likely hit. "
                      f"Re-run later to finish (cache persists; {scored} scored this run).", flush=True)
                break
        if s + BATCH < len(todo):
            time.sleep(PACE)
    print(f"\ngemini unified_v1: {scored} scored in {reqs} requests\n")

    print("══ GEMINI vs llama3.2 on the LIVE unified_v1 prompt — top-{} lotto ══".format(gl.TOPN))
    print(f"  {'scorer':22} {'scored':>7} {'trades':>7} {'P&L':>10} {'Sharpe':>7} {'2x+':>4} {'4x+':>4}")
    for model in (GMODEL, "llama3.2"):
        n, trades = sim_topn(cands, sc, model, "unified_v1")
        if trades:
            st = compute_stats(trades)
            x2 = sum(1 for t in trades if t["pnl_pct"] >= 100); x4 = sum(1 for t in trades if t["pnl_pct"] >= 300)
            print(f"  {model:22} {n:>7} {len(trades):>7} {('${:+,.0f}'.format(st['total_pnl'])):>10} "
                  f"{st['sharpe']:>7.2f} {x2:>4} {x4:>4}")
        else:
            print(f"  {model:22} {n:>7} {0:>7}  (no trades)")


if __name__ == "__main__":
    main()
