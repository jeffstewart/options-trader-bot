"""
platform_compare.py — N-way scorer bake-off on the LIVE unified_v1 lotto selector, across
PLATFORMS (Gemini / Groq / OpenAI / Mistral / local Ollama). For each model with an available
API key it (1) batch-scores the SAME shared-coverage pool flash already scored, and (2) times
single-article calls for a LATENCY datapoint (single-call = matches live bot use), then prints
top-N lotto P&L + latency side by side.

Providers without a key in .env are skipped automatically. Cached model scores are reused — re-runs
only score what's missing. Per-provider call params are handled (OpenAI o-series rejects temperature
+ uses max_completion_tokens; reasoning_effort='none' only applies to Gemini/Groq non-reasoning).

Usage:  USE_YAHOO_BARS=1 .venv/bin/python -u platform_compare.py
"""
import os, json, re, time, statistics
os.environ.setdefault("USE_YAHOO_BARS", "1")
from pathlib import Path
from openai import OpenAI
from dotenv import load_dotenv
load_dotenv("/Users/jeff/Claude/Trader/.env")

import gemini_batch_unified as gbu          # UNIFIED_BATCH_SYS, sim_topn
import gemini_lotto_pnl as gl
from prompt_lab import extract_score
from benchmark import compute_stats

E = os.environ
FLASH_PRE = "gemini-2.5-flash:unified_v1:"   # the shared-coverage anchor (already scored)
PACE, LAT_SAMPLE = 5.0, int(E.get("LAT_SAMPLE", "12"))
ONLY = set(filter(None, E.get("MODELS_ONLY", "").split(",")))   # restrict to these display names

PROVIDERS = {
    "gemini":  (E.get("GEMINI_BASE_URL"),     E.get("GEMINI_API_KEY")),
    "groq":    (E.get("LAB_HOSTED_BASE_URL"),  E.get("LAB_HOSTED_TEST_KEY") or E.get("LAB_HOSTED_KEY")),
    "openai":  (E.get("OPENAI_BASE_URL"),      E.get("OPENAI_API_KEY")),
    "mistral": (E.get("MISTRAL_BASE_URL"),     E.get("MISTRAL_API_KEY")),
    "ollama":  (E.get("OLLAMA_BASE_URL"),      "ollama"),
}
# (display/cache-key, model_id, provider, reasoning?, batch). Current models (2026-06), verified
# reachable in the smoke test. Cached ones (flash, mistral small/medium, llama3.2) skip scoring.
MODELS = [
    ("gemini-2.5-flash",       "gemini-2.5-flash",        "gemini",  False, 50),  # anchor (cached)
    ("gemini-3.5-flash",       "gemini-3.5-flash",        "gemini",  False, 50),  # NEWER gemini
    ("gemini-3.1-flash-lite",  "gemini-3.1-flash-lite",   "gemini",  False, 50),
    ("gpt-5.4-mini",           "gpt-5.4-mini",            "openai",  False, 20),  # only free OpenAI model
    ("mistral-large-latest",   "mistral-large-latest",    "mistral", False, 30),  # frontier
    ("mistral-medium-latest",  "mistral-medium-latest",   "mistral", False, 30),  # cached
    ("mistral-small-latest",   "mistral-small-latest",    "mistral", False, 30),  # cached
    ("magistral-medium-latest","magistral-medium-latest", "mistral", True,  15),  # reasoning
    ("qwen/qwen3.6-27b",       "qwen/qwen3.6-27b",        "groq",    False, 20),  # newer qwen
    ("llama-3.3-70b-versatile","llama-3.3-70b-versatile", "groq",    False, 30),
    ("openai/gpt-oss-120b",    "openai/gpt-oss-120b",     "groq",    False, 30),
    ("llama3.2",               "llama3.2",                "ollama",  False, 50),  # local baseline (cached)
]


def client_for(provider):
    base, key = PROVIDERS.get(provider, (None, None))
    return OpenAI(base_url=base, api_key=key) if (base and key) else None


def _chat(client, model_id, provider, reasoning, messages, max_out):
    """One chat call with provider-correct params. Modern OpenAI (gpt-5*/o*) reject `temperature`
    and use `max_completion_tokens` (and burn reasoning tokens, so give generous headroom)."""
    kw = {"model": model_id, "messages": messages}
    modern_oa = provider == "openai" and (model_id.startswith("gpt-5") or model_id[0] == "o")
    if modern_oa:
        kw["max_completion_tokens"] = max(max_out, 8000)   # room for hidden reasoning + JSON
        if model_id[0] == "o":
            kw["reasoning_effort"] = "low"
    else:
        kw["max_tokens"] = max_out
        kw["temperature"] = 0.1
        if provider == "gemini" and not reasoning:         # gemini accepts this; suppresses thinking
            kw["reasoning_effort"] = "none"
    return client.chat.completions.create(**kw)


def _content(resp):
    """Extract text from a chat response. Reasoning models (magistral, some gpt-oss) return
    message.content as a LIST of blocks instead of a string — join the text parts."""
    c = resp.choices[0].message.content
    if isinstance(c, list):
        parts = []
        for b in c:
            if isinstance(b, dict):
                parts.append(b.get("text") or b.get("content") or "")
            else:
                parts.append(getattr(b, "text", "") or "")
        c = " ".join(parts)
    return (c or "").strip()


def _parse_array(raw, n):
    arr = json.loads(re.search(r"\[.*\]", raw, re.S).group(0))
    out = {}
    for obj in arr:
        try:
            i = int(obj.get("i", 0)) - 1
            if 0 <= i < n:
                out[i] = {"magnitude": float(obj.get("magnitude", 0) or 0),
                          "confidence": float(obj.get("confidence", 0.5) or 0.5)}
        except Exception:
            continue
    return out


def time_calls(client, model_id, provider, reasoning, samples):
    """Median / p90 single-article latency (ms) — matches the live bot's per-article call."""
    lat = []
    for c in samples:
        msgs = [{"role": "system", "content": gbu.UNIFIED_BATCH_SYS},
                {"role": "user", "content": f'[1] {c["h"]}  ::  {(c["body"] or "")[:600]}'}]
        t0 = time.time()
        try:
            _chat(client, model_id, provider, reasoning, msgs, 2000)
            lat.append((time.time() - t0) * 1000)
        except Exception as e:
            print(f"    latency call failed ({model_id}): {repr(e)[:80]}", flush=True)
    if not lat:
        return None, None
    lat.sort()
    return statistics.median(lat), lat[min(len(lat) - 1, int(len(lat) * 0.9))]


def main():
    sc = json.loads(Path(gl.SCORES_PATH).read_text())
    gl.SAMPLE = 3000
    cands = gl.candidates()
    uni = json.load(open("unified_scores.json"))
    for c in cands:
        u = uni.get(f"unified_v1:{c['ck']}")
        if isinstance(u, dict):
            sc[f"llama3.2:unified_v1:{c['ck']}"] = extract_score(u)
    pool = [c for c in cands if sc.get(FLASH_PRE + c["ck"]) is not None]
    print(f"shared-coverage pool (flash-scored): {len(pool)} candidates · top-{gl.TOPN} lotto\n")

    latency = {}
    for disp, model_id, provider, reasoning, mbatch in MODELS:
        if ONLY and disp not in ONLY:
            continue
        cli = client_for(provider)
        if cli is None:
            print(f"⏭  {disp:24} [{provider}] — no API key, skipping", flush=True)
            continue
        pre = f"{disp}:unified_v1:"
        todo = [c for c in pool if sc.get(pre + c["ck"]) is None]
        if todo and provider != "ollama":
            print(f"▶  {disp:24} [{provider}] scoring {len(todo)} (batch {mbatch}) …", flush=True)
            fails = 0
            for s in range(0, len(todo), mbatch):
                b = todo[s:s + mbatch]
                user = "\n".join(f'[{i+1}] {c["h"]}  ::  {(c["body"] or "")[:600]}' for i, c in enumerate(b))
                msgs = [{"role": "system", "content": gbu.UNIFIED_BATCH_SYS}, {"role": "user", "content": user}]
                try:
                    resp = _chat(cli, model_id, provider, reasoning, msgs, 6000 if reasoning else 4000)
                    res = _parse_array(_content(resp), len(b))
                    for i, c in enumerate(b):
                        if i in res:
                            sc[pre + c["ck"]] = extract_score(res[i])
                    Path(gl.SCORES_PATH).write_text(json.dumps(sc)); fails = 0
                    print(f"     {s//mbatch+1}: +{len(res)}/{len(b)}", flush=True)
                except Exception as ex:
                    fails += 1
                    print(f"     batch fail: {repr(ex)[:100]}", flush=True)
                    if fails >= 3:
                        print(f"     [stop] {disp} cap/err — partial; re-run later", flush=True); break
                if s + mbatch < len(todo):
                    time.sleep(PACE)
        latency[disp] = time_calls(cli, model_id, provider, reasoning, pool[:LAT_SAMPLE])
        med = latency[disp][0]
        print(f"   {disp:24} latency: median={med:.0f}ms p90={latency[disp][1]:.0f}ms" if med
              else f"   {disp:24} latency: n/a", flush=True)

    print(f"\n══ unified_v1 top-{gl.TOPN} lotto — P&L + single-call latency ══")
    print(f"  {'scorer':24} {'trades':>6} {'P&L':>11} {'Sharpe':>7} {'2x+':>4} {'4x+':>4} {'win%':>5} {'med-ms':>7} {'p90-ms':>7}")
    have = [m[0] for m in MODELS if any(sc.get(f"{m[0]}:unified_v1:" + c["ck"]) is not None for c in pool)]
    for disp in have:
        n, trades = gbu.sim_topn(pool, sc, disp, "unified_v1")
        med, p90 = latency.get(disp, (None, None))
        ms = f"{med:.0f}" if med else "—"; p9 = f"{p90:.0f}" if p90 else "—"
        if trades:
            st = compute_stats(trades)
            x2 = sum(1 for t in trades if t["pnl_pct"] >= 100); x4 = sum(1 for t in trades if t["pnl_pct"] >= 300)
            win = sum(1 for t in trades if t["pnl_usd"] > 0) / len(trades) * 100
            print(f"  {disp:24} {len(trades):>6} {('${:+,.0f}'.format(st['total_pnl'])):>11} "
                  f"{st['sharpe']:>7.2f} {x2:>4} {x4:>4} {win:>5.0f} {ms:>7} {p9:>7}")
        else:
            print(f"  {disp:24} {'—':>6} {'(no trades)':>11} {'':>7} {'':>4} {'':>4} {'':>5} {ms:>7} {p9:>7}")
    print("\nGoal: best P&L/Sharpe. Latency = single-call (live-equivalent); convex legs tolerate it.")


if __name__ == "__main__":
    main()
