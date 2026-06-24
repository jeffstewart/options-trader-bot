"""
groq_value.py — could a GROQ-hosted model do the hybrid CONFIRM/VETO as well as Gemini, but with
much higher quota? Scores the SAME llama tradeable-bullish picks SINGLE-ARTICLE (faithful to the
live path — Groq is fast + high-quota so no batching needed) using the live unified_v1 prompt,
applies the same confirm/veto rule, and compares CONFIRM vs VETO forward returns + veto rate to
Gemini's result. Cached per model.

Gemini reference (faithful): veto 38-62% · CONFIRM beats VETO by ~+1.3-2.7% event-day · 13% vs 4% hit.

Usage:  USE_YAHOO_BARS=1 .venv/bin/python -u groq_value.py
"""
import os, json, re, time, random, statistics
os.environ.setdefault("USE_YAHOO_BARS", "1")
import numpy as np
from openai import OpenAI
from dotenv import load_dotenv
load_dotenv("/Users/jeff/Claude/Trader/.env")
import gemini_lotto_pnl as gl
import bot
from config import GEMINI_CONFIRM_MIN_MAGNITUDE as CONF_MIN

MODELS = ["openai/gpt-oss-120b"]   # DIFFERENT-family from primary llama3.2 (the real test)
GCACHE = "groq_value_cache.json"
PACE = 15.0   # heavy pacing — gpt-oss is a reasoning model (token-heavy) vs Groq free-tier TPM


def score_one(client, model, h, b):
    r = client.chat.completions.create(
        model=model, temperature=0.1, max_tokens=800,
        messages=[{"role": "system", "content": bot.SYSTEM_PROMPT},
                  {"role": "user", "content": f"Headline: {h}\n\nBody: {(b or '')[:1200]}\n\nRespond with JSON only."}])
    raw = r.choices[0].message.content.strip()
    if "<think>" in raw:
        raw = raw.split("</think>")[-1].strip()
    if raw.startswith("```"):
        raw = raw.split("```")[1]
        if raw.startswith("json"):
            raw = raw[4:]
    m = re.search(r"\{.*\}", raw, re.S)
    o = json.loads(m.group(0))
    return {"sent": o.get("sentiment"), "mag": float(o.get("magnitude", 0) or 0)}


def eligible():
    uni = json.load(open("unified_scores.json"))
    cpool = json.load(open("yahoo_candpool_cache.json"))
    gl.SAMPLE = 100000
    out = []
    for c in gl.candidates():
        u = uni.get("unified_v1:" + c["ck"]); fr = cpool.get(f"{c['tk']}_{c['ck']}")
        if (isinstance(u, dict) and u.get("sentiment") == "bullish" and float(u.get("magnitude", 0) or 0) >= 0.75
                and fr and fr.get("px", 0) >= 5 and c.get("h")):
            out.append((c, fr))
    out.sort(key=lambda x: x[0]["ck"])
    return out


def stat(pnl_key, rs):
    v = sorted(r[pnl_key] for r in rs); n = len(v)
    return (n, sum(v) / n, v[n // 2], sum(1 for x in v if x >= 5) / n * 100) if n else (0, 0, 0, 0)


def main():
    elig = eligible()
    print(f"eligible llama tradeable-bullish picks: {len(elig)} (confirm = bullish & mag≥{CONF_MIN})\n", flush=True)
    g = OpenAI(base_url=os.environ["LAB_HOSTED_BASE_URL"],
               api_key=os.environ.get("LAB_HOSTED_TEST_KEY") or os.environ["LAB_HOSTED_KEY"])
    gc = json.load(open(GCACHE)) if os.path.exists(GCACHE) else {}

    for model in MODELS:
        todo = [(c, fr) for c, fr in elig if f"{model}:{c['ck']}" not in gc][:130]   # sample cap
        print(f"▶ {model}: scoring {len(todo)} single-article (pace {PACE}s) …", flush=True)
        for i, (c, fr) in enumerate(todo):
            for attempt in range(4):                          # retry the SAME item on 429
                try:
                    gc[f"{model}:{c['ck']}"] = score_one(g, model, c["h"], c.get("body", "")); break
                except Exception as e:
                    if ("rate" in repr(e).lower() or "429" in repr(e)) and attempt < 3:
                        time.sleep(12); continue
                    gc[f"{model}:{c['ck']}"] = None; break
            time.sleep(PACE)
            if i % 40 == 39:
                json.dump(gc, open(GCACHE, "w")); print(f"    {i+1}/{len(todo)}", flush=True)
        json.dump(gc, open(GCACHE, "w"))

    # compare each Groq model's confirm/veto separation (incl. already-cached llama-70b)
    print(f"\n{'model':26} {'veto%':>6} {'CONFIRM ev-day':>14} {'VETO ev-day':>12} {'edge':>6} {'C-hit':>6} {'V-hit':>6}")
    for model in ["llama-3.3-70b-versatile"] + MODELS:
        rows = []
        for c, fr in elig:
            gx = gc.get(f"{model}:{c['ck']}")
            if not gx:
                continue
            conf = (gx["sent"] == "bullish" and gx["mag"] >= CONF_MIN)
            rows.append({"v": "c" if conf else "v", "rday": fr["rday"]})
        cf = [r for r in rows if r["v"] == "c"]; vt = [r for r in rows if r["v"] == "v"]
        if not cf or not vt:
            print(f"  {model:26} (insufficient: {len(cf)}c/{len(vt)}v)"); continue
        cn, ca, _, ch = stat("rday", cf); vn, va, _, vh = stat("rday", vt)
        print(f"  {model:24} {vn/(cn+vn)*100:>5.0f}% {ca:>+12.1f}% ({cn}) {va:>+8.1f}% ({vn}) {ca-va:>+5.1f}% {ch:>5.0f}% {vh:>5.0f}%")
    print("\n  Gemini ref (faithful n=80): veto 62% · CONFIRM +1.8% / VETO +0.5% · edge +1.3% · 13% vs 4% hit")
    print("  A Groq model that separates confirm>veto similarly = a high-quota drop-in for the hybrid gate.")


if __name__ == "__main__":
    main()
