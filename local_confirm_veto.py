"""
local_confirm_veto.py — rule out (or in) SMALL LOCAL Ollama models for the confirm/veto gate, on
the 8GB Mac mini. Scores the SAME 40-pick Gemini-covered sample as confirm_veto_sweep.py (single-
article, faithful unified_v1 prompt) via the local Ollama OpenAI endpoint, so results slot straight
into the sweep table. ONE model per invocation (argv) → load exactly one model at a time, keeping
well under the ~2.5GB resident ceiling that's safe on this machine.

SAFETY: only pass models ≤ ~2.5GB resident (3B-class). Never the 4.7-4.9GB ones — those panic the box.

Usage:  USE_YAHOO_BARS=1 .venv/bin/python -u local_confirm_veto.py qwen2.5:3b
"""
import os, sys, json, time
os.environ.setdefault("USE_YAHOO_BARS", "1")
from openai import OpenAI
import gemini_lotto_pnl as gl
import bot
from config import GEMINI_CONFIRM_MIN_MAGNITUDE as CM

LCACHE = "local_cv_cache.json"
OLLAMA = "http://localhost:11434/v1"


import re


def _parse(raw):
    """Robust to small-model junk: scan ALL JSON objects, take the one with sentiment; if none parse,
    fall back to regex-extracting sentiment+magnitude from loose text."""
    if "<think>" in raw:
        raw = raw.split("</think>")[-1]
    raw = raw.strip()
    if raw.startswith("```"):
        raw = raw.split("```")[1]
        raw = raw[4:] if raw.lstrip().startswith("json") else raw
    dec = json.JSONDecoder(); idx = 0; objs = []
    while idx < len(raw):
        b0 = raw.find("{", idx)
        if b0 < 0:
            break
        try:
            o, end = dec.raw_decode(raw[b0:]); objs.append(o); idx = b0 + end
        except json.JSONDecodeError:
            idx = b0 + 1
    for o in objs:
        if isinstance(o, dict) and "sentiment" in o:
            return {"sent": o.get("sentiment"), "mag": float(o.get("magnitude", 0) or 0)}
    sm = re.search(r'sentiment["\s:]+\"?(bullish|bearish|neutral)', raw, re.I)
    mm = re.search(r'magnitude["\s:]+\"?([0-9.]+)', raw, re.I)
    if sm:
        return {"sent": sm.group(1).lower(), "mag": float(mm.group(1)) if mm else 0.0}
    raise ValueError("no sentiment in output")


def score_one(cli, model, h, b):
    """Single-article, faithful unified_v1 prompt. One retry on parse failure (small models are flaky)."""
    msgs = [{"role": "system", "content": bot.SYSTEM_PROMPT},
            {"role": "user", "content": f"Headline: {h}\n\nBody: {(b or '')[:1200]}\n\nRespond with JSON only."}]
    last = None
    for _ in range(2):
        raw = cli.chat.completions.create(model=model, temperature=0.1, max_tokens=1200,
                                          messages=msgs).choices[0].message.content
        try:
            return _parse(raw)
        except ValueError as ex:
            last = ex
    raise last


def sample40():
    uni = json.load(open("unified_scores.json")); cpool = json.load(open("yahoo_candpool_cache.json"))
    gfaith = json.load(open("gemini_value_faithful_cache.json"))
    gl.SAMPLE = 100000
    elig = []
    for c in gl.candidates():
        u = uni.get("unified_v1:" + c["ck"]); fr = cpool.get(f"{c['tk']}_{c['ck']}")
        if (isinstance(u, dict) and u.get("sentiment") == "bullish" and float(u.get("magnitude", 0) or 0) >= 0.75
                and fr and fr.get("px", 0) >= 5 and c.get("h")):
            elig.append((c, fr))
    elig.sort(key=lambda x: x[0]["ck"])
    return [(c, fr) for c, fr in elig if c["ck"] in gfaith][:40]


def main():
    model = sys.argv[1]                       # e.g. qwen2.5:3b
    disp = "local/" + model
    sample = sample40()
    cli = OpenAI(base_url=OLLAMA, api_key="ollama")
    sc = json.load(open(LCACHE)) if os.path.exists(LCACHE) else {}
    todo = [(c, fr) for c, fr in sample if f"{disp}:{c['ck']}" not in sc]
    print(f"▶ {disp}: scoring {len(todo)}/{len(sample)} single-article (local Ollama)", flush=True)
    t0 = time.time()
    for i, (c, fr) in enumerate(todo):
        try:
            sc[f"{disp}:{c['ck']}"] = score_one(cli, model, c["h"], c.get("body", ""))
        except Exception as ex:
            print(f"    [{i}] ERR {repr(ex)[:80]}", flush=True); sc[f"{disp}:{c['ck']}"] = None
        if i % 10 == 9:
            json.dump(sc, open(LCACHE, "w")); print(f"    {i+1}/{len(todo)}  ({(time.time()-t0)/(i+1):.1f}s/call)", flush=True)
    json.dump(sc, open(LCACHE, "w"))

    rows = [(((gx := sc.get(f"{disp}:{c['ck']}"))["sent"] == "bullish" and gx["mag"] >= CM), fr["rday"])
            for c, fr in sample if sc.get(f"{disp}:{c['ck']}")]
    cf = [r[1] for r in rows if r[0]]; vt = [r[1] for r in rows if not r[0]]
    print(f"\n══ {disp} on n={len(rows)} ══")
    if len(cf) < 3 or len(vt) < 3:
        print(f"  insufficient split: {len(cf)} confirm / {len(vt)} veto"); return
    ca, va = sum(cf) / len(cf), sum(vt) / len(vt)
    ch = sum(1 for x in cf if x >= 5) / len(cf) * 100; vh = sum(1 for x in vt if x >= 5) / len(vt) * 100
    print(f"  veto {len(vt)/len(rows)*100:.0f}% · CONFIRM {ca:+.1f}% / VETO {va:+.1f}% · edge {ca-va:+.1f}% · "
          f"C-hit {ch:.0f}% vs V-hit {vh:.0f}%")
    print(f"  (sweep refs: gemini +1.9%, gpt-oss-120b +2.2%, qwen3-32b +2.1%, llama-3.1-8b −1.8%)")


if __name__ == "__main__":
    main()
