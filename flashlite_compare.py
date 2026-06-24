"""
flashlite_compare.py — does gemini-2.5-flash-LITE pick as well as gemini-2.5-flash on the live
unified_v1 lotto selector? Scores flash-lite ONLY on the candidates flash already scored (so the
3-way is fair), then compares flash vs flash-lite vs llama3.2 on the SHARED-COVERAGE set (each
picks its top-N from the same candidates). flash-lite has higher free-tier RPD, so if it's ≈ flash
it's a strictly better live confirmer (more requests, free).

Usage:  USE_YAHOO_BARS=1 .venv/bin/python -u flashlite_compare.py
"""
import os, json, time
os.environ.setdefault("USE_YAHOO_BARS", "1")
from pathlib import Path
from openai import OpenAI
from dotenv import load_dotenv
load_dotenv("/Users/jeff/Claude/Trader/.env")

import gemini_batch_unified as gbu          # UNIFIED_BATCH_SYS, sim_topn
import gemini_batch_score as gbs
import gemini_lotto_pnl as gl
from prompt_lab import extract_score
from benchmark import compute_stats

FLASH = "gemini-2.5-flash"
LITE  = "gemini-2.5-flash-lite"
BATCH, PACE = 50, 8.0


def main():
    g = OpenAI(base_url=os.environ["GEMINI_BASE_URL"], api_key=os.environ["GEMINI_API_KEY"])
    sc = json.loads(Path(gl.SCORES_PATH).read_text())
    gl.SAMPLE = 3000
    cands = gl.candidates()
    # seed llama floats from the cached unified_scores.json
    uni = json.load(open("unified_scores.json"))
    for c in cands:
        u = uni.get(f"unified_v1:{c['ck']}")
        if isinstance(u, dict):
            sc[f"llama3.2:unified_v1:{c['ck']}"] = extract_score(u)

    flash_pre, lite_pre = f"{FLASH}:unified_v1:", f"{LITE}:unified_v1:"
    # only score flash-lite where FLASH already has a score (fair shared set), and lite doesn't yet
    todo = [c for c in cands if sc.get(flash_pre + c["ck"]) is not None
            and sc.get(lite_pre + c["ck"]) is None]
    print(f"flash-covered candidates: {sum(1 for c in cands if sc.get(flash_pre+c['ck']) is not None)}  "
          f"| flash-lite to score: {len(todo)}", flush=True)

    scored = consec_fail = 0
    for s in range(0, len(todo), BATCH):
        batch = todo[s:s + BATCH]
        try:
            res = gbs.score_batch(g, batch, gbu.UNIFIED_BATCH_SYS, model=LITE)
            for i, c in enumerate(batch):
                if i in res:
                    sc[lite_pre + c["ck"]] = extract_score(res[i]); scored += 1
            Path(gl.SCORES_PATH).write_text(json.dumps(sc))
            consec_fail = 0
            print(f"  batch {s//BATCH+1}: {len(res)}/{len(batch)} (cum {scored})", flush=True)
        except Exception as e:
            consec_fail += 1
            print(f"  batch {s//BATCH+1} FAILED: {repr(e)[:100]}", flush=True)
            if consec_fail >= 3:
                print("  [early-stop] 3 consecutive failures — flash-lite cap likely hit; re-run later.", flush=True)
                break
        if s + BATCH < len(todo):
            time.sleep(PACE)

    # 3-way on SHARED coverage (scored by all three)
    shared = [c for c in cands if all(sc.get(p + c["ck"]) is not None
              for p in (flash_pre, lite_pre, f"llama3.2:unified_v1:"))]
    print(f"\n══ 3-way on SHARED coverage ({len(shared)} candidates, each picks top-{gl.TOPN}) ══")
    print(f"  {'scorer':22} {'trades':>7} {'P&L':>11} {'Sharpe':>7} {'2x+':>4} {'4x+':>4} {'win%':>5}")
    for model in (FLASH, LITE, "llama3.2"):
        n, trades = gbu.sim_topn(shared, sc, model, "unified_v1")
        if trades:
            st = compute_stats(trades)
            x2 = sum(1 for t in trades if t["pnl_pct"] >= 100); x4 = sum(1 for t in trades if t["pnl_pct"] >= 300)
            win = sum(1 for t in trades if t["pnl_usd"] > 0) / len(trades) * 100
            print(f"  {model:22} {len(trades):>7} {('${:+,.0f}'.format(st['total_pnl'])):>11} "
                  f"{st['sharpe']:>7.2f} {x2:>4} {x4:>4} {win:>5.0f}")
        else:
            print(f"  {model:22}   (no trades)")
    print("\nflash-lite is a win if it ≈ flash here — same picks, higher free-tier request ceiling.")


if __name__ == "__main__":
    main()
