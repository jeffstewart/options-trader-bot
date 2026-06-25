"""
combined_gate_surprise_test.py — can ONE mistral call return the gate's magnitude AND a surprise score
without the surprise signal degrading? Surprise alone scored rank-IC +0.36 (surprise_cheap_models.py);
magnitude alone is ~0. If a single combined call keeps surprise's IC, the gate gets a free second
orthogonal signal — no extra hot-path call. This builds the combined prompt by EXTENDING the live gate
prompt (bot.SYSTEM_PROMPT) in place — adding a `surprise` field + rubric — so the gate's existing
sentiment/magnitude semantics are unchanged, then scores the same 45-article sample with mistral-large.

Usage:  USE_YAHOO_BARS=1 .venv/bin/python -u research/combined_gate_surprise_test.py
"""
import os, json, re, time
os.environ.setdefault("USE_YAHOO_BARS", "1")
import surprise_prompt_test as spt          # reuses sample(), rank_ic() (and chdir to data/)
import numpy as np
from openai import OpenAI
import bot
import confirm_veto_sweep as cvs

MODEL_DISP, MODEL_ID, PROVIDER = "mistral-large", "mistral-large-latest", "mistral"
CACHE = "combined_gate_surprise_cache.json"

# Build the combined prompt by injecting `surprise` into the LIVE gate prompt — production fidelity for
# sentiment/magnitude; only the surprise field is new. Two edits: add it to the schema, add its rubric.
SCHEMA_FROM = '  "magnitude":  0.75,'
SCHEMA_TO   = '  "magnitude":  0.75,\n  "surprise":   0.30,'
RUBRIC_ANCHOR = "Catalyst scale"
SURPRISE_BLOCK = """surprise:   float 0.0–1.0 — how much NEW, unexpected information this carries vs what the market
            already knew or had anticipated (novelty), INDEPENDENT of how bullish or large the move

Surprise scale:
  0.0–0.2  Fully expected / already priced: scheduled events, reiterated guidance, index rebalances,
           analyst opinions/price targets, recaps of a move that already happened
  0.3–0.5  Incremental: in-line results, minor updates, confirmations of a known direction
  0.6–0.8  A genuine new development: unexpected deal, real guidance change, approval, material catalyst
  0.9–1.0  A major shock the market could not have priced: transformative, out-of-nowhere
  Think about what the market most likely ALREADY expected before this headline, then rate.

"""

def build_prompt():
    p = bot.SYSTEM_PROMPT
    assert SCHEMA_FROM in p and RUBRIC_ANCHOR in p, "gate prompt changed — re-anchor the injection"
    p = p.replace(SCHEMA_FROM, SCHEMA_TO, 1)
    p = p.replace(RUBRIC_ANCHOR, SURPRISE_BLOCK + RUBRIC_ANCHOR, 1)
    return p

COMBINED_PROMPT = build_prompt()


def score(cli, h, b):
    t0 = time.time()
    r = cli.chat.completions.create(model=MODEL_ID, temperature=0.1, max_tokens=1200,
        messages=[{"role": "system", "content": COMBINED_PROMPT},
                  {"role": "user", "content": f"Headline: {h}\n\nBody: {(b or '')[:1500]}\n\nRespond with JSON only."}])
    ms = (time.time() - t0) * 1000
    tok = r.usage.total_tokens if r.usage else 0
    raw = r.choices[0].message.content or ""
    raw = raw.split("</think>")[-1] if "</think>" in raw else raw
    o = json.loads(re.search(r"\{.*\}", raw, re.S).group(0))
    return {"sent": o.get("sentiment"), "mag": float(o.get("magnitude", 0) or 0),
            "sur": float(o.get("surprise", 0) or 0)}, ms, tok


def ic_block(label, scores, rets):
    pairs = [(scores[k], rets[k]) for k in scores if scores.get(k) is not None and k in rets]
    if len(pairs) < 15:
        print(f"  {label:26} n={len(pairs)} — too few"); return
    s = [p[0] for p in pairs]; r = [p[1] for p in pairs]; med = np.median(s)
    lo = [r[i] for i in range(len(s)) if s[i] < med]; hi = [r[i] for i in range(len(s)) if s[i] >= med]
    edge = (sum(hi)/len(hi) - sum(lo)/len(lo)) if lo and hi else float("nan")
    print(f"  {label:26} n={len(pairs):<3} rank-IC {spt.rank_ic(s, r):+.2f}   hi-lo edge {edge:+.2f}%   (med {med:.2f})")


def main():
    smp = spt.sample()
    rets = {c["ck"]: fr["r3"] for c, fr in smp}
    sc = json.load(open(CACHE)) if os.path.exists(CACHE) else {}
    cli = OpenAI(base_url=cvs.PROV[PROVIDER][0], api_key=cvs.PROV[PROVIDER][1])
    solo = json.load(open("surprise_cheap_cache.json")) if os.path.exists("surprise_cheap_cache.json") else {}

    lat, tks = [], []
    print(f"▶ scoring {len(smp)} with {MODEL_DISP} on the COMBINED gate+surprise prompt…", flush=True)
    for i, (c, fr) in enumerate(smp):
        k = c["ck"]
        if k not in sc:
            for att in range(5):
                try:
                    v, ms, tk = score(cli, c["h"], c.get("body", "")); sc[k] = v
                    lat.append(ms); tks.append(tk); break
                except Exception as e:
                    if ("rate" in repr(e).lower() or "429" in repr(e)) and att < 4:
                        time.sleep(15); continue
                    sc[k] = None; break
            time.sleep(1.2)
            if i % 10 == 9:
                json.dump(sc, open(CACHE, "w")); print(f"  {i+1}/{len(smp)}", flush=True)
    json.dump(sc, open(CACHE, "w"))

    scored = {k: v for k, v in sc.items() if isinstance(v, dict)}
    sur = {k: v["sur"] for k, v in scored.items()}
    mag = {k: v["mag"] for k, v in scored.items()}
    bull = sum(1 for v in scored.values() if v["sent"] == "bullish")
    print(f"\n  scored {len(scored)}/{len(smp)} · bullish {bull} · "
          f"avg {sum(lat)/len(lat) if lat else 0:.0f}ms / {sum(tks)/len(tks) if tks else 0:.0f} tok per call")
    print(f"\n  ══ does the COMBINED call keep each signal? (same 45-article sample, forward r3) ══")
    ic_block("surprise (combined)", sur, rets)
    ic_block("magnitude (combined)", mag, rets)
    # surprise from the SOLO surprise-only call, for the degradation check
    solo_sur = {c["ck"]: solo.get(f"{MODEL_DISP}:" + c["ck"]) for c, fr in smp}
    ic_block("surprise (solo call ref)", solo_sur, rets)

    # agreement between solo and combined surprise (did bundling shift the scores?)
    both = [(solo_sur[k], sur[k]) for k in sur if solo_sur.get(k) is not None]
    if len(both) >= 15:
        a = [x[0] for x in both]; b = [x[1] for x in both]
        print(f"\n  solo-vs-combined surprise corr: {np.corrcoef(a, b)[0,1]:+.2f}  "
              f"(mean |Δ| {np.mean([abs(x-y) for x,y in both]):.2f})")
    print("\n  surprise IC ≈ +0.36 in the combined call ⇒ one gate call returns mag + a free orthogonal")
    print("  surprise signal. IC collapses ⇒ bundling distracts the model ⇒ keep surprise a separate call.")


if __name__ == "__main__":
    main()
