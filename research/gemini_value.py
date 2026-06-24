"""
gemini_value.py — does the hybrid Gemini CONFIRM/VETO add value over Ollama alone?

Takes the picks the bot WOULD real-trade (llama bullish, mag≥0.75), re-scores each with Gemini,
applies the live confirm/veto rule (confirm = Gemini bullish AND mag ≥ GEMINI_CONFIRM_MIN_MAGNITUDE),
and compares the FORWARD RETURNS of CONFIRMED vs VETOED picks. Vetoed underperform confirmed →
the veto catches losers (value). Indistinguishable → the hybrid adds nothing → drop it.

BATCHED: packs ~BATCH items per Gemini request (sentiment+magnitude per item) so the whole sample
is a handful of requests instead of ~20/day under the free-tier RPM cap. Results cached.
NOTE: batch scoring is a close proxy for the live single-article path (compact prompt) — fine for
the value question, not a byte-exact replay.

Usage:  USE_YAHOO_BARS=1 .venv/bin/python -u gemini_value.py
"""
import os, json, re, time
os.environ.setdefault("USE_YAHOO_BARS", "1")
from openai import OpenAI
from dotenv import load_dotenv
load_dotenv("/Users/jeff/Claude/Trader/.env")
import gemini_lotto_pnl as gl
from config import GEMINI_CONFIRM_MIN_MAGNITUDE, GEMINI_MODEL
import bot   # for the LIVE unified_v1 prompt (faithful to the actual confirm/veto path)

BATCH = 40
GCACHE = "gemini_value_faithful_cache.json"
# FAITHFUL batch: the EXACT live unified_v1 rubric (bot.SYSTEM_PROMPT) + a minimal batch wrapper,
# so Gemini scores each item the same way the live single-article path does (no extra harshness).
VALUE_BATCH_SYS = (
    bot.SYSTEM_PROMPT
    + "\n\n--- BATCH MODE ---\nYou will receive MULTIPLE numbered news items. Score EACH one using "
      "the rubric above, and respond with a JSON ARRAY ONLY — one object per item in order, each with "
      'an integer "i" plus "sentiment", "magnitude", "confidence": '
      '[{"i":1,"sentiment":"bullish","magnitude":0.6,"confidence":0.8}, ...]'
)


def score_batch(client, batch):
    user = "\n".join(f'[{i+1}] {c["h"]}  ::  {(c.get("body") or "")[:500]}' for i, c in enumerate(batch))
    r = client.chat.completions.create(
        model=GEMINI_MODEL, temperature=0.1, max_tokens=4000, reasoning_effort="none",
        messages=[{"role": "system", "content": VALUE_BATCH_SYS}, {"role": "user", "content": user}])
    arr = json.loads(re.search(r"\[.*\]", r.choices[0].message.content.strip(), re.S).group(0))
    out = {}
    for o in arr:
        try:
            i = int(o.get("i", 0)) - 1
            if 0 <= i < len(batch):
                out[i] = {"sent": o.get("sentiment"), "mag": float(o.get("magnitude", 0) or 0)}
        except Exception:
            continue
    return out


def stat(rs, key):
    v = sorted(r[key] for r in rs); n = len(v)
    return (n, sum(v) / n, v[n // 2], sum(1 for x in v if x >= 5) / n * 100) if n else (0, 0, 0, 0)


def main():
    uni = json.load(open("unified_scores.json"))
    cpool = json.load(open("yahoo_candpool_cache.json"))
    gl.SAMPLE = 100000
    cands = {c["ck"]: c for c in gl.candidates()}
    elig = []
    for ck, c in cands.items():
        u = uni.get("unified_v1:" + ck)
        fr = cpool.get(f"{c['tk']}_{ck}")
        if (isinstance(u, dict) and u.get("sentiment") == "bullish"
                and float(u.get("magnitude", 0) or 0) >= 0.75
                and fr and fr.get("px", 0) >= 5 and c.get("h")):
            elig.append((c, fr))
    elig.sort(key=lambda x: x[0]["ck"])
    print(f"eligible llama tradeable-bullish picks: {len(elig)} (confirm thresh mag≥{GEMINI_CONFIRM_MIN_MAGNITUDE})", flush=True)

    g = OpenAI(base_url=os.environ["GEMINI_BASE_URL"], api_key=os.environ["GEMINI_API_KEY"])
    gc = json.load(open(GCACHE)) if os.path.exists(GCACHE) else {}
    todo = [c for c, _ in elig if c["ck"] not in gc]
    print(f"  batch-scoring {len(todo)} with Gemini ({(len(todo)+BATCH-1)//BATCH} requests @ batch {BATCH}) …", flush=True)
    fails = 0
    for s in range(0, len(todo), BATCH):
        b = todo[s:s + BATCH]
        try:
            res = score_batch(g, b)
            for i, c in enumerate(b):
                if i in res:
                    gc[c["ck"]] = res[i]
            json.dump(gc, open(GCACHE, "w")); fails = 0
            print(f"    req {s//BATCH+1}: {len(res)}/{len(b)} scored", flush=True)
        except Exception as e:
            fails += 1
            print(f"    req {s//BATCH+1} FAILED: {repr(e)[:90]}", flush=True)
            if fails >= 3:
                print("    [stop] 3 consecutive failures — re-run later (cache persists)", flush=True); break
        if s + BATCH < len(todo):
            time.sleep(5)

    rows = []
    for c, fr in elig:
        gx = gc.get(c["ck"])
        if not gx:
            continue
        confirm = (gx["sent"] == "bullish" and gx["mag"] >= GEMINI_CONFIRM_MIN_MAGNITUDE)
        rows.append({"v": "confirm" if confirm else "veto", "rday": fr["rday"], "r1": fr["r1"], "r3": fr["r3"]})
    conf = [r for r in rows if r["v"] == "confirm"]; veto = [r for r in rows if r["v"] == "veto"]
    print(f"\n══ Gemini verdict on {len(rows)} llama tradeable-bullish picks ══")
    print(f"  confirmed: {len(conf)} · vetoed: {len(veto)}  (veto rate {len(veto)/max(len(rows),1)*100:.0f}%)\n")
    print(f"  {'group':9} {'n':>4} {'avg event-day':>14} {'median':>8} {'hit≥5%':>7} | {'avg 1d':>7} {'avg 3d':>7}")
    for name, rs in (("CONFIRM", conf), ("VETO", veto)):
        if rs:
            n, a, m, h = stat(rs, "rday")
            print(f"  {name:9} {n:>4} {a:>+13.1f}% {m:>+7.1f}% {h:>6.0f}% | "
                  f"{sum(r['r1'] for r in rs)/n:>+6.1f}% {sum(r['r3'] for r in rs)/n:>+6.1f}%")
    if conf and veto:
        d = sum(r["rday"] for r in conf) / len(conf) - sum(r["rday"] for r in veto) / len(veto)
        print(f"\n  CONFIRM − VETO event-day edge: {d:+.1f}%")
        print("  Positive (confirms beat vetoes) = veto catches losers → hybrid adds value. ~0/neg = drop it.")


if __name__ == "__main__":
    main()
