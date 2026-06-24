"""
confirm_veto_sweep.py — small-sample CONFIRM/VETO bake-off across every reachable model on Groq,
Mistral, OpenAI (+ Gemini reference). Scores the SAME ~40 llama tradeable-bullish picks single-
article with the live unified_v1 prompt, applies the confirm/veto rule, and compares each model's
CONFIRM-vs-VETO forward returns + hit-rate split. Screening pass: which models separate winners
from losers like Gemini? Reuses Gemini + the 2 done Groq scores; fresh-scores the rest (paced).

Usage:  USE_YAHOO_BARS=1 .venv/bin/python -u confirm_veto_sweep.py
"""
import os, json, re, time
os.environ.setdefault("USE_YAHOO_BARS", "1")
from openai import OpenAI
from dotenv import load_dotenv
load_dotenv("/Users/jeff/Claude/Trader/.env")
import gemini_lotto_pnl as gl
import bot
from config import GEMINI_CONFIRM_MIN_MAGNITUDE as CM

E = os.environ
SAMPLE = 40
SCACHE = "confirm_veto_sweep_cache.json"
PROV = {
    "groq":    (E["LAB_HOSTED_BASE_URL"], E.get("LAB_HOSTED_TEST_KEY") or E["LAB_HOSTED_KEY"]),
    "mistral": (E["MISTRAL_BASE_URL"],    E["MISTRAL_API_KEY"]),
    "openai":  (E["OPENAI_BASE_URL"],     E["OPENAI_API_KEY"]),
}
# (display, model_id, provider, pace, reuse_cache_key|None)
MODELS = [
    ("gemini-2.5-flash",   "gemini-2.5-flash",        "gemini",  0, "GEMINI"),
    ("groq/llama-3.3-70b",  "llama-3.3-70b-versatile", "groq",  0, "llama-3.3-70b-versatile"),
    ("groq/gpt-oss-120b",   "openai/gpt-oss-120b",     "groq",  0, "openai/gpt-oss-120b"),
    ("groq/gpt-oss-20b",    "openai/gpt-oss-20b",      "groq",  8, None),
    ("groq/qwen3.6-27b",    "qwen/qwen3.6-27b",        "groq",  6, None),
    ("groq/qwen3-32b",      "qwen/qwen3-32b",          "groq", 10, None),
    ("groq/llama-3.1-8b",   "llama-3.1-8b-instant",    "groq",  4, None),
    ("groq/llama-4-scout",  "meta-llama/llama-4-scout-17b-16e-instruct", "groq", 4, None),
    ("mistral-small",       "mistral-small-latest",    "mistral", 3, None),
    ("mistral-medium",      "mistral-medium-latest",   "mistral", 3, None),
    ("mistral-large",       "mistral-large-latest",    "mistral", 3, None),
    ("mistral-ministral-8b","ministral-8b-latest",     "mistral", 3, None),
    ("mistral-magistral-m", "magistral-medium-latest", "mistral", 5, None),
    ("openai/gpt-5.4-mini", "gpt-5.4-mini",            "openai",  3, None),
]


def score_one(client, model_id, provider, h, b, extra=None):
    kw = {"model": model_id, "messages": [
        {"role": "system", "content": bot.SYSTEM_PROMPT},
        {"role": "user", "content": f"Headline: {h}\n\nBody: {(b or '')[:1200]}\n\nRespond with JSON only."}]}
    if provider == "openai":
        kw["max_completion_tokens"] = 2000           # gpt-5.x: no temperature, room for reasoning
    else:
        kw["max_tokens"] = 1200; kw["temperature"] = 0.1
    if extra:                                        # per-model overrides, e.g. reasoning_effort=low for gpt-oss
        kw.update(extra)
    raw = client.chat.completions.create(**kw).choices[0].message.content.strip()
    if "<think>" in raw:
        raw = raw.split("</think>")[-1].strip()
    if raw.startswith("```"):
        raw = raw.split("```")[1]
        if raw.startswith("json"):
            raw = raw[4:]
    o = json.loads(re.search(r"\{.*\}", raw, re.S).group(0))
    return {"sent": o.get("sentiment"), "mag": float(o.get("magnitude", 0) or 0)}


def main():
    uni = json.load(open("unified_scores.json"))
    cpool = json.load(open("yahoo_candpool_cache.json"))
    gfaith = json.load(open("gemini_value_faithful_cache.json")) if os.path.exists("gemini_value_faithful_cache.json") else {}
    gvcache = json.load(open("groq_value_cache.json")) if os.path.exists("groq_value_cache.json") else {}
    gl.SAMPLE = 100000
    elig = []
    for c in gl.candidates():
        u = uni.get("unified_v1:" + c["ck"]); fr = cpool.get(f"{c['tk']}_{c['ck']}")
        if (isinstance(u, dict) and u.get("sentiment") == "bullish" and float(u.get("magnitude", 0) or 0) >= 0.75
                and fr and fr.get("px", 0) >= 5 and c.get("h")):
            elig.append((c, fr))
    elig.sort(key=lambda x: x[0]["ck"])
    sample = [(c, fr) for c, fr in elig if c["ck"] in gfaith][:SAMPLE]   # in Gemini coverage → fair ref
    print(f"sample: {len(sample)} picks (Gemini-covered) · confirm = bullish & mag≥{CM}\n", flush=True)

    sc = json.load(open(SCACHE)) if os.path.exists(SCACHE) else {}
    # seed reused scores
    for disp, mid, prov, pace, reuse in MODELS:
        if reuse == "GEMINI":
            for c, _ in sample:
                sc.setdefault(f"{disp}:{c['ck']}", gfaith.get(c["ck"]))
        elif reuse:
            for c, _ in sample:
                sc.setdefault(f"{disp}:{c['ck']}", gvcache.get(f"{reuse}:{c['ck']}"))

    for disp, mid, prov, pace, reuse in MODELS:
        if reuse:
            continue
        todo = [(c, fr) for c, fr in sample if f"{disp}:{c['ck']}" not in sc]
        if not todo:
            continue
        cli = OpenAI(base_url=PROV[prov][0], api_key=PROV[prov][1])
        print(f"▶ {disp}: {len(todo)} (pace {pace}s)", flush=True)
        for i, (c, fr) in enumerate(todo):
            for att in range(4):
                try:
                    sc[f"{disp}:{c['ck']}"] = score_one(cli, mid, prov, c["h"], c.get("body", "")); break
                except Exception as ex:
                    if ("rate" in repr(ex).lower() or "429" in repr(ex)) and att < 3:
                        time.sleep(12); continue
                    sc[f"{disp}:{c['ck']}"] = None; break
            time.sleep(pace)
        json.dump(sc, open(SCACHE, "w"))

    # report
    print(f"\n{'model':22} {'n':>3} {'veto%':>5} {'CONFIRM':>9} {'VETO':>9} {'edge':>6} {'C-hit':>5} {'V-hit':>5}")
    res = []
    for disp, mid, prov, pace, reuse in MODELS:
        rows = []
        for c, fr in sample:
            gx = sc.get(f"{disp}:{c['ck']}")
            if not gx:
                continue
            conf = (gx["sent"] == "bullish" and gx["mag"] >= CM)
            rows.append((conf, fr["rday"]))
        cf = [r[1] for r in rows if r[0]]; vt = [r[1] for r in rows if not r[0]]
        if len(cf) < 3 or len(vt) < 3:
            print(f"  {disp:22} n={len(rows):>2}  (insufficient {len(cf)}c/{len(vt)}v)"); continue
        ca, va = sum(cf)/len(cf), sum(vt)/len(vt)
        ch = sum(1 for x in cf if x >= 5)/len(cf)*100; vh = sum(1 for x in vt if x >= 5)/len(vt)*100
        res.append((disp, len(rows), len(vt)/len(rows)*100, ca, va, ca-va, ch, vh))
    for disp, n, vp, ca, va, ed, ch, vh in sorted(res, key=lambda r: -r[5]):
        print(f"  {disp:22} {n:>3} {vp:>4.0f}% {ca:>+8.1f}% {va:>+8.1f}% {ed:>+5.1f}% {ch:>4.0f}% {vh:>4.0f}%")
    print("\n  edge = CONFIRM − VETO event-day. Positive + C-hit > V-hit = the veto catches losers like Gemini.")


if __name__ == "__main__":
    main()
