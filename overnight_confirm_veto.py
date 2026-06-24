"""
overnight_confirm_veto.py — large-sample (full 346-pick eligible set, SAME sample for all models)
CONFIRM/VETO ranking of the top candidates from the n=40 sweep, to settle which is statistically
best as a high-quota stand-in for the Gemini hybrid gate. Single-article scoring (faithful), paced
per provider, incremental cache (survives overnight). Prints bootstrapped edge CIs at the end.

Usage:  USE_YAHOO_BARS=1 .venv/bin/python -u overnight_confirm_veto.py
"""
import os, json, time, random
os.environ.setdefault("USE_YAHOO_BARS", "1")
import numpy as np
from openai import OpenAI
import gemini_lotto_pnl as gl
import confirm_veto_sweep as cvs
from config import GEMINI_CONFIRM_MIN_MAGNITUDE as CM

CACHE = "overnight_cv_cache.json"
# (display, model_id, provider, pace) — the top cluster from the n=40 sweep
CANDIDATES = [
    ("groq/llama-4-scout",  "meta-llama/llama-4-scout-17b-16e-instruct", "groq", 4),
    ("mistral-large",       "mistral-large-latest",                      "mistral", 4),
    ("groq/gpt-oss-120b",   "openai/gpt-oss-120b",                       "groq", 11),
    ("groq/qwen3.6-27b",    "qwen/qwen3.6-27b",                          "groq", 10),
]


def eligible():
    uni = json.load(open("unified_scores.json")); cpool = json.load(open("yahoo_candpool_cache.json"))
    gl.SAMPLE = 100000
    out = []
    for c in gl.candidates():
        u = uni.get("unified_v1:" + c["ck"]); fr = cpool.get(f"{c['tk']}_{c['ck']}")
        if (isinstance(u, dict) and u.get("sentiment") == "bullish" and float(u.get("magnitude", 0) or 0) >= 0.75
                and fr and fr.get("px", 0) >= 5 and c.get("h")):
            out.append((c, fr))
    out.sort(key=lambda x: x[0]["ck"])
    return out


def boot_edge(cf, vt, n=5000):
    rng = random.Random(0)
    d = [sum(rng.choices(cf, k=len(cf))) / len(cf) - sum(rng.choices(vt, k=len(vt))) / len(vt) for _ in range(n)]
    return np.percentile(d, 5), np.percentile(d, 95)


def main():
    sample = eligible()
    print(f"sample: {len(sample)} picks (same for all) · confirm = bullish & mag≥{CM}\n", flush=True)
    sc = json.load(open(CACHE)) if os.path.exists(CACHE) else {}
    for disp, mid, prov, pace in CANDIDATES:
        todo = [(c, fr) for c, fr in sample if f"{disp}:{c['ck']}" not in sc]
        print(f"▶ {disp}: {len(todo)} to score (pace {pace}s)", flush=True)
        cli = OpenAI(base_url=cvs.PROV[prov][0], api_key=cvs.PROV[prov][1])
        for i, (c, fr) in enumerate(todo):
            for att in range(5):
                try:
                    sc[f"{disp}:{c['ck']}"] = cvs.score_one(cli, mid, prov, c["h"], c.get("body", "")); break
                except Exception as ex:
                    if ("rate" in repr(ex).lower() or "429" in repr(ex)) and att < 4:
                        time.sleep(20); continue
                    sc[f"{disp}:{c['ck']}"] = None; break
            time.sleep(pace)
            if i % 50 == 49:
                json.dump(sc, open(CACHE, "w")); print(f"    {disp} {i+1}/{len(todo)}", flush=True)
        json.dump(sc, open(CACHE, "w"))

    print(f"\n══ LARGE-SAMPLE CONFIRM/VETO RANKING ══")
    print(f"  {'model':22} {'n':>4} {'veto%':>5} {'CONFIRM':>8} {'VETO':>7} {'edge':>6} {'boot 5-95% edge CI':>20} {'Chit':>5} {'Vhit':>5}")
    res = []
    for disp, mid, prov, pace in CANDIDATES:
        rows = [(((gx := sc.get(f"{disp}:{c['ck']}"))["sent"] == "bullish" and gx["mag"] >= CM), fr["rday"])
                for c, fr in sample if sc.get(f"{disp}:{c['ck']}")]
        cf = [r[1] for r in rows if r[0]]; vt = [r[1] for r in rows if not r[0]]
        if len(cf) < 5 or len(vt) < 5:
            print(f"  {disp:22} n={len(rows)} (insufficient {len(cf)}c/{len(vt)}v)"); continue
        ca, va = sum(cf) / len(cf), sum(vt) / len(vt)
        lo, hi = boot_edge(cf, vt)
        ch = sum(1 for x in cf if x >= 5) / len(cf) * 100; vh = sum(1 for x in vt if x >= 5) / len(vt) * 100
        res.append((disp, len(rows), len(vt) / len(rows) * 100, ca, va, ca - va, lo, hi, ch, vh))
    for disp, n, vp, ca, va, ed, lo, hi, ch, vh in sorted(res, key=lambda r: -r[5]):
        sig = "✓sig" if lo > 0 else "~ns"
        print(f"  {disp:22} {n:>4} {vp:>4.0f}% {ca:>+7.1f}% {va:>+6.1f}% {ed:>+5.1f}% [{lo:>+4.1f},{hi:>+4.1f}]{sig:>5} {ch:>4.0f}% {vh:>4.0f}%")
    print("\n  edge CI entirely > 0 ⇒ that model's veto significantly catches losers at large sample.")


if __name__ == "__main__":
    main()
