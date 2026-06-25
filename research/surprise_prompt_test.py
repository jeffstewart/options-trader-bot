"""
surprise_prompt_test.py — does asking a THINKING model "how surprising is this news?" produce a score
that predicts forward returns? (The price-based already-moved proxy failed — momentum dominates — so
measure novelty/surprise DIRECTLY.) Scores a sample of bullish-tradeable articles with a reasoning
model on a surprise rubric, then correlates surprise with forward r3 (rank-IC + low/high buckets) and
reports the token/latency cost (the trade-off the user flagged).

Usage:  USE_YAHOO_BARS=1 .venv/bin/python -u research/surprise_prompt_test.py
"""
import os, json, re, time, random
os.environ.setdefault("USE_YAHOO_BARS", "1")
import config
os.chdir(config.DATA_DIR)
import numpy as np
from openai import OpenAI
import gemini_lotto_pnl as gl
import confirm_veto_sweep as cvs

MODEL = "qwen/qwen3.6-27b"          # a reasoning ("thinking") model; separate Groq quota from gpt-oss
CACHE = "surprise_prompt_cache.json"
N = 45

SURPRISE_PROMPT = """You assess financial news for tradeable SURPRISE — how much NEW, unexpected
information it carries versus what the market already knew or had anticipated.

Rate surprise from 0.0 to 1.0:
- 0.0-0.2  fully expected / already priced: scheduled events, reiterated guidance, index rebalances,
           analyst opinions/price targets, recaps of a move that already happened
- 0.3-0.5  incremental: in-line results, minor updates, confirmations of a known direction
- 0.6-0.8  a genuine new development: unexpected deal, real guidance change, approval, material catalyst
- 0.9-1.0  a major shock the market could not have priced: transformative, out-of-nowhere

Think about what the market most likely ALREADY expected before this headline, then rate.
Respond with JSON only: {"surprise": <float 0-1>, "reason": "<one short sentence>"}"""


def score_surprise(cli, h, b):
    t0 = time.time()
    r = cli.chat.completions.create(model=MODEL, temperature=0.2, max_tokens=4000,
        messages=[{"role": "system", "content": SURPRISE_PROMPT},
                  {"role": "user", "content": f"Headline: {h}\n\nBody: {(b or '')[:1500]}"}])
    ms = (time.time() - t0) * 1000
    tok = r.usage.total_tokens
    raw = r.choices[0].message.content or ""
    raw = raw.split("</think>")[-1] if "</think>" in raw else raw
    o = json.loads(re.search(r"\{.*\}", raw, re.S).group(0))
    return float(o.get("surprise", 0) or 0), ms, tok


def sample():
    uni = json.load(open("unified_scores.json")); cp = json.load(open("yahoo_candpool_cache.json"))
    gl.SAMPLE = 100000
    out = []
    for c in gl.candidates():
        u = uni.get("unified_v1:" + c["ck"]); fr = cp.get(f"{c['tk']}_{c['ck']}")
        if (isinstance(u, dict) and u.get("sentiment") == "bullish" and float(u.get("magnitude", 0) or 0) >= 0.75
                and fr and fr.get("px", 0) >= 5 and c.get("h")):
            out.append((c, fr))
    out.sort(key=lambda x: x[0]["ck"])
    return out[:N]


def rank_ic(xs, ys):
    rx = np.argsort(np.argsort(xs)); ry = np.argsort(np.argsort(ys))
    return float(np.corrcoef(rx, ry)[0, 1])


def main():
    smp = sample()
    cli = OpenAI(base_url=cvs.PROV["groq"][0], api_key=cvs.PROV["groq"][1])
    sc = json.load(open(CACHE)) if os.path.exists(CACHE) else {}
    lat, tks = [], []
    print(f"scoring {len(smp)} articles for SURPRISE with {MODEL} (reasoning)…", flush=True)
    for i, (c, fr) in enumerate(smp):
        if c["ck"] not in sc:
            for att in range(4):
                try:
                    s, ms, tk = score_surprise(cli, c["h"], c.get("body", ""))
                    sc[c["ck"]] = s; lat.append(ms); tks.append(tk); break
                except Exception as e:
                    if ("rate" in repr(e).lower() or "429" in repr(e)) and att < 3:
                        time.sleep(20); continue
                    sc[c["ck"]] = None; break
            time.sleep(2)
            if i % 10 == 9:
                json.dump(sc, open(CACHE, "w")); print(f"  {i+1}/{len(smp)}", flush=True)
    json.dump(sc, open(CACHE, "w"))

    rows = [(sc[c["ck"]], fr["r3"]) for c, fr in smp if sc.get(c["ck"]) is not None]
    print(f"\n  scored: {len(rows)}/{len(smp)}")
    if lat:
        print(f"  cost: avg {sum(lat)/len(lat):.0f}ms/call · {sum(tks)/len(tks):.0f} tokens/call "
              f"(vs ~565ms / ~970 tok for non-reasoning gpt-oss) — the thinking-model premium")
    if len(rows) < 15:
        print("  too few scored to correlate — re-run (rate limits)."); return
    sur = [r[0] for r in rows]; ret = [r[1] for r in rows]
    print(f"\n  surprise distribution: min {min(sur):.2f} median {np.median(sur):.2f} max {max(sur):.2f}")
    print(f"  ══ does SURPRISE predict forward r3? ══")
    print(f"  rank-IC(surprise, r3): {rank_ic(sur, ret):+.2f}   (>+0.1 = useful signal)")
    med = np.median(sur)
    lo = [r for s, r in rows if s < med]; hi = [r for s, r in rows if s >= med]
    print(f"  LOW surprise  (<{med:.2f}): n={len(lo):<3} avg r3 {sum(lo)/len(lo):+.2f}% · {sum(1 for x in lo if x>0)/len(lo)*100:.0f}%pos")
    print(f"  HIGH surprise (≥{med:.2f}): n={len(hi):<3} avg r3 {sum(hi)/len(hi):+.2f}% · {sum(1 for x in hi if x>0)/len(hi)*100:.0f}%pos")
    print(f"  edge (high − low): {sum(hi)/len(hi) - sum(lo)/len(lo):+.2f}%")
    print("\n  Positive rank-IC + high>low = surprise carries tradeable signal the magnitude score misses.")


if __name__ == "__main__":
    main()
