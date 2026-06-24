"""
gemini_batch_score.py — BATCHED Gemini scorer + frontier-vs-llama3.2 lotto comparison.

The free-tier wall was ~25 requests/day. But news items are ~41 tokens each, so the 250k input
budget is a non-issue — we pack many articles into ONE request and get back one score per item.
Binding limits are now OUTPUT tokens + the model staying aligned across the batch, so we use a
modest batch size, index every item, and validate the response count.

Scores the lotto candidate pool with gemini-2.5-flash on the SAME materiality scoring as llama3.2
(score = magnitude × confidence), writes to the shared cache, then runs the head-to-head: rank
each model's picks, simulate the top-N as lotto, compare P&L. Re-runnable (cache-aware) — a 2nd
run mops up any items a batch dropped.

Usage:  USE_YAHOO_BARS=1 .venv/bin/python -u gemini_batch_score.py [BATCH]
"""
import os, re, sys, json, time
os.environ.setdefault("USE_YAHOO_BARS", "1")
from pathlib import Path
from openai import OpenAI
from dotenv import load_dotenv
load_dotenv("/Users/jeff/Claude/Trader/.env")

import gemini_lotto_pnl as gl            # reuse candidates(), sim_topn(), SCORES_PATH, GMODEL
from benchmark import compute_stats
from prompt_lab import extract_score     # uniform float score (magnitude × confidence) — matches llama path

BATCH = int(sys.argv[1]) if len(sys.argv) > 1 else 50
PACE = 13.0                              # ~5 requests/min free-tier ceiling
GMODEL, SCORES_PATH = gl.GMODEL, gl.SCORES_PATH
KEYPRE = f"{GMODEL}:materiality_fewshot:"

BATCH_SYS = (
    "You are a financial-news materiality scorer. Score EACH numbered item for a BULLISH long-call "
    "trade on its primary ticker.\n"
    "  magnitude (0-1): proportion of the news NOT already priced — most news is priced, so "
    "magnitude < 0.2; reserve 0.7+ for genuine unpriced surprises (earnings beat/miss, M&A, FDA "
    "decision, trading halt, guidance change). Analyst reiteration/price-target tweak, in-line "
    "results, routine buyback, conference talk, macro/no-company news → magnitude < 0.15.\n"
    "  confidence (0-1): certainty about the bullish direction.\n"
    "Respond with a JSON ARRAY ONLY — exactly one object per item, in order, no prose:\n"
    '[{"i":1,"magnitude":0.6,"confidence":0.7},{"i":2,"magnitude":0.1,"confidence":0.5}]'
)


def score_batch(client, batch, sys_prompt=BATCH_SYS, model=None):
    """Return {batch_index: {magnitude, confidence}} for as many items as the model returned.
    sys_prompt swaps the scoring rubric (materiality_fewshot vs unified_v1) — output format same.
    model overrides GMODEL (e.g. score a different Gemini variant like flash-lite)."""
    user = "\n".join(f'[{i+1}] {c["h"]}  ::  {(c["body"] or "")[:600]}' for i, c in enumerate(batch))
    r = client.chat.completions.create(
        model=model or GMODEL, temperature=0.1, max_tokens=4000, reasoning_effort="none",
        messages=[{"role": "system", "content": sys_prompt}, {"role": "user", "content": user}])
    raw = r.choices[0].message.content.strip()
    arr = json.loads(re.search(r"\[.*\]", raw, re.S).group(0))
    out = {}
    for obj in arr:
        try:
            i = int(obj.get("i", 0)) - 1
            if 0 <= i < len(batch):
                out[i] = {"magnitude": float(obj.get("magnitude", 0) or 0),
                          "confidence": float(obj.get("confidence", 0.5) or 0.5)}
        except Exception:
            continue
    return out


def main():
    g = OpenAI(base_url=os.environ["GEMINI_BASE_URL"], api_key=os.environ["GEMINI_API_KEY"])
    sc = json.loads(Path(SCORES_PATH).read_text()) if Path(SCORES_PATH).exists() else {}
    cands = gl.candidates()
    todo = [c for c in cands if sc.get(KEYPRE + c["ck"]) is None]
    print(f"BATCHED Gemini scorer — pool {len(cands)}, to score {len(todo)} (batch {BATCH}, ~{PACE:.0f}s/req)")
    t0, reqs, scored, missing = time.time(), 0, 0, 0
    for s in range(0, len(todo), BATCH):
        batch = todo[s:s + BATCH]
        try:
            res = score_batch(g, batch)
            reqs += 1
            for i, c in enumerate(batch):
                if i in res:
                    sc[KEYPRE + c["ck"]] = extract_score(res[i])  # store the float, not the raw dict
                    scored += 1
                else:
                    missing += 1
            Path(SCORES_PATH).write_text(json.dumps(sc))
            print(f"  batch {s//BATCH+1}: {len(res)}/{len(batch)} scored  (req#{reqs}, cum {scored})", flush=True)
        except Exception as e:
            reqs += 1; missing += len(batch)
            print(f"  batch {s//BATCH+1} FAILED ({len(batch)} items): {repr(e)[:110]}", flush=True)
        if s + BATCH < len(todo):
            time.sleep(PACE)
    dt = time.time() - t0
    print(f"\nDONE: {scored} scored across {reqs} requests in {dt:.0f}s "
          f"({missing} dropped — re-run to mop up).")
    if reqs:
        print(f"  ⚡ throughput: {scored/reqs:.0f} articles/request — the old 1/request scorer would need "
              f"~{max(scored,1)/25:.0f} days at the 25/day cap; batching did it in {reqs} requests.\n")

    print(f"  {'scorer':22} {'scored':>7} {'trades':>7} {'P&L':>10} {'Sharpe':>7} {'2x+':>4} {'4x+':>4}")
    for model in (GMODEL, "llama3.2"):
        n_scored, trades = gl.sim_topn(cands, sc, model)
        if trades:
            st = compute_stats(trades)
            x2 = sum(1 for t in trades if t["pnl_pct"] >= 100); x4 = sum(1 for t in trades if t["pnl_pct"] >= 300)
            print(f"  {model:22} {n_scored:>7} {len(trades):>7} {('${:+,.0f}'.format(st['total_pnl'])):>10} "
                  f"{st['sharpe']:>7.2f} {x2:>4} {x4:>4}")
        else:
            print(f"  {model:22} {n_scored:>7} {0:>7}  (no trades)")
    print("\nSame candidates, scorer only RANKS → Gemini wins only if its top-N picks make more lotto P&L.")


if __name__ == "__main__":
    main()
