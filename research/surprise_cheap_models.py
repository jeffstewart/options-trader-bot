"""
surprise_cheap_models.py — does the SURPRISE signal need a reasoning model (qwen3.6, ~9.4s/call,
rank-IC +0.23), or can a fast/cheap model capture it? Scores the SAME 45-article sample on the SAME
surprise rubric with cheaper models and compares rank-IC(surprise, r3) + cost side-by-side. If a fast
model retains the signal, surprise could live near the hot path; if not, it's gate-only.

Usage:  USE_YAHOO_BARS=1 .venv/bin/python -u research/surprise_cheap_models.py
"""
import os, json, re, time
os.environ.setdefault("USE_YAHOO_BARS", "1")
import surprise_prompt_test as spt          # reuses sample(), SURPRISE_PROMPT, rank_ic (and chdir to data/)
import numpy as np
from openai import OpenAI
import confirm_veto_sweep as cvs

CACHE = "surprise_cheap_cache.json"
# (display, model_id, provider, extra_params)
MODELS = [
    ("mistral-large",  "mistral-large-latest",     "mistral", {}),                       # the gate model (non-reasoning)
    ("llama-3.3-70b",  "llama-3.3-70b-versatile",  "groq",    {}),                       # fast non-reasoning
    ("gpt-oss-low",    "openai/gpt-oss-120b",      "groq",    {"reasoning_effort": "low"}),  # light reasoning
]


def score(cli, model, h, b, extra):
    kw = {"model": model, "temperature": 0.2, "max_tokens": 1200,
          "messages": [{"role": "system", "content": spt.SURPRISE_PROMPT},
                       {"role": "user", "content": f"Headline: {h}\n\nBody: {(b or '')[:1500]}"}]}
    kw.update(extra)
    t0 = time.time()
    r = cli.chat.completions.create(**kw)
    ms = (time.time() - t0) * 1000
    tok = r.usage.total_tokens if r.usage else 0
    raw = r.choices[0].message.content or ""
    raw = raw.split("</think>")[-1] if "</think>" in raw else raw
    o = json.loads(re.search(r"\{.*\}", raw, re.S).group(0))
    return float(o.get("surprise", 0) or 0), ms, tok


def main():
    smp = spt.sample()
    rets = {c["ck"]: fr["r3"] for c, fr in smp}
    sc = json.load(open(CACHE)) if os.path.exists(CACHE) else {}
    qwen = json.load(open("surprise_prompt_cache.json")) if os.path.exists("surprise_prompt_cache.json") else {}

    rows = []  # (display, n, rank_ic, edge_high_low, avg_ms, avg_tok)
    # qwen3.6 reasoning baseline (already scored)
    qr = [(qwen[c["ck"]], rets[c["ck"]]) for c, fr in smp if qwen.get(c["ck"]) is not None]
    if len(qr) >= 15:
        s = [x[0] for x in qr]; r = [x[1] for x in qr]; med = np.median(s)
        lo = [r[i] for i in range(len(s)) if s[i] < med]; hi = [r[i] for i in range(len(s)) if s[i] >= med]
        rows.append(("qwen3.6 (reasoning)", len(qr), spt.rank_ic(s, r),
                     sum(hi)/len(hi)-sum(lo)/len(lo), 9444, 1273))

    for disp, mid, prov, extra in MODELS:
        cli = OpenAI(base_url=cvs.PROV[prov][0], api_key=cvs.PROV[prov][1])
        lat, tks = [], []
        print(f"▶ {disp}: scoring {len(smp)} for surprise…", flush=True)
        for i, (c, fr) in enumerate(smp):
            k = f"{disp}:{c['ck']}"
            if k not in sc:
                for att in range(4):
                    try:
                        v, ms, tk = score(cli, mid, c["h"], c.get("body", ""), extra)
                        sc[k] = v; lat.append(ms); tks.append(tk); break
                    except Exception as e:
                        if ("rate" in repr(e).lower() or "429" in repr(e)) and att < 3:
                            time.sleep(15); continue
                        sc[k] = None; break
                time.sleep(1.5)
            if i % 15 == 14:
                json.dump(sc, open(CACHE, "w"))
        json.dump(sc, open(CACHE, "w"))
        pairs = [(sc[f"{disp}:{c['ck']}"], rets[c["ck"]]) for c, fr in smp if sc.get(f"{disp}:{c['ck']}") is not None]
        if len(pairs) < 15:
            rows.append((disp, len(pairs), float("nan"), float("nan"),
                         sum(lat)/len(lat) if lat else 0, sum(tks)/len(tks) if tks else 0)); continue
        s = [x[0] for x in pairs]; r = [x[1] for x in pairs]; med = np.median(s)
        loo = [r[i] for i in range(len(s)) if s[i] < med]; hii = [r[i] for i in range(len(s)) if s[i] >= med]
        edge = (sum(hii)/len(hii) - sum(loo)/len(loo)) if loo and hii else float("nan")
        rows.append((disp, len(pairs), spt.rank_ic(s, r), edge,
                     sum(lat)/len(lat) if lat else 0, sum(tks)/len(tks) if tks else 0))

    print(f"\n  ══ SURPRISE signal vs cost, by model (same {len(smp)}-article sample) ══")
    print(f"  {'model':22} {'n':>3} {'rank-IC':>8} {'hi−lo edge':>11} {'latency':>9} {'tokens':>7}")
    for disp, n, ic, edge, ms, tok in rows:
        print(f"  {disp:22} {n:>3} {ic:>+8.2f} {edge:>+10.2f}% {ms:>7.0f}ms {tok:>7.0f}")
    print("\n  rank-IC ≈ qwen's +0.23 at much lower latency ⇒ surprise can run cheap (near hot path).")
    print("  rank-IC collapses ⇒ the reasoning is doing the work ⇒ keep it as a gate only.")


if __name__ == "__main__":
    main()
