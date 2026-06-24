"""
score_unified.py — batch-score the broad candidate pool with a UNIFIED prompt (default unified_v1),
caching the FULL output dict {sentiment, confidence, magnitude, catalyst, tickers} per article so the
evaluator can (a) derive the prompt's OWN high-conviction lotto set + catalyst gate and (b) test its
confidence for stock selectivity — all from ONE pass. Resumable (cache-aware), saves incrementally.

Cache: unified_scores.json, keyed "<prompt>:<cache_key>". Run as a daemon on the weekend (bot idle).

Usage:  USE_YAHOO_BARS=1 .venv/bin/python -u score_unified.py            # full pool, unified_v1
        UNI_PROMPT=unified_v2 UNI_SAMPLE=1500 .venv/bin/python -u score_unified.py
"""
import os, re, json
os.environ.setdefault("USE_YAHOO_BARS", "1")
from datetime import datetime, timezone, timedelta
from pathlib import Path
from openai import OpenAI
from dotenv import load_dotenv
load_dotenv("/Users/jeff/Claude/Trader/.env")

import config as cfg
from backtest import cache_key
from regime_filter import build_regime
from prompt_lab import PROMPTS

PROMPT = os.environ.get("UNI_PROMPT", "unified_v1")
SAMPLE = int(os.environ.get("UNI_SAMPLE", "0"))          # 0 = full pool
CACHE = os.environ.get("UNI_CACHE", "dual_score_cache.json")
DAYS = int(os.environ.get("UNI_DAYS", "180"))
_END_ENV = os.environ.get("UNI_END", "")                 # ISO date; default = now-5d (bull)
END = (datetime.fromisoformat(_END_ENV).replace(tzinfo=timezone.utc) if _END_ENV
       else datetime.now(timezone.utc) - timedelta(days=5))
USE_REGIME = os.environ.get("UNI_REGIME", "1") == "1"    # 0 for bear: score all in-window articles
SCORES_PATH = "unified_scores.json"
MODEL = "llama3.2"
OLLAMA = OpenAI(base_url=cfg.OLLAMA_BASE_URL, api_key="ollama")


def candidates():
    raw = json.load(open(CACHE))
    reg = build_regime(END, DAYS, 200) if USE_REGIME else None
    out, seen = [], set()
    for v in raw.values():
        if not isinstance(v, dict):
            continue
        a = v.get("_article", {}) or {}
        h, ca, body = a.get("headline"), a.get("created_at"), a.get("summary", "")
        if not h or not ca:
            continue
        try:
            dt = datetime.fromisoformat(str(ca).replace("Z", "+00:00")).astimezone(timezone.utc)
        except Exception:
            continue
        if not (END - timedelta(days=DAYS) <= dt <= END) or (reg and not reg(dt.date())):
            continue
        ck = cache_key(h, body)
        if ck in seen:
            continue
        seen.add(ck)
        out.append({"h": h, "body": body, "dt": dt, "ck": ck})
    out.sort(key=lambda c: c["dt"])
    if SAMPLE and len(out) > SAMPLE:
        step = len(out) / SAMPLE
        out = [out[int(i * step)] for i in range(SAMPLE)]
    return out


def score_one(sys, h, body):
    r = OLLAMA.chat.completions.create(model=MODEL, temperature=0.1, max_tokens=220,
        messages=[{"role": "system", "content": sys},
                  {"role": "user", "content": f"Headline: {h}\n\nBody: {body[:1200]}\n\nRespond with JSON only."}])
    o = json.loads(re.search(r"\{.*\}", r.choices[0].message.content.strip(), re.S).group(0))
    return {"sentiment": str(o.get("sentiment", "")).lower(),
            "confidence": float(o.get("confidence", 0) or 0),
            "magnitude": float(o.get("magnitude", 0) or 0),
            "catalyst": float(o.get("catalyst", 0) or 0),
            "tickers": [t for t in (o.get("tickers") or []) if isinstance(t, str)]}


def main():
    sys = PROMPTS[PROMPT]
    sc = json.loads(Path(SCORES_PATH).read_text()) if Path(SCORES_PATH).exists() else {}
    cands = candidates()
    todo = [c for c in cands if f"{PROMPT}:{c['ck']}" not in sc]
    print(f"score_unified [{PROMPT}] — pool {len(cands)}, already cached {len(cands)-len(todo)}, to score {len(todo)}", flush=True)
    done = 0
    for c in todo:
        key = f"{PROMPT}:{c['ck']}"
        try:
            sc[key] = score_one(sys, c["h"], c["body"])
        except Exception as e:
            sc[key] = None
        done += 1
        if done % 25 == 0:
            Path(SCORES_PATH).write_text(json.dumps(sc))
            print(f"  scored {done}/{len(todo)}  (cache {len(sc)})", flush=True)
    Path(SCORES_PATH).write_text(json.dumps(sc))
    print(f"DONE [{PROMPT}]: scored {done}, cache now {len(sc)}", flush=True)


if __name__ == "__main__":
    main()
