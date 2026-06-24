"""
score_lotto_signals.py — targeted materiality scoring of the HIGH-CONVICTION (lotto)
signals so the lotto Stage-3 A/B isn't starved by sampling coverage.

The generic dev/test/all samples only happened to cover ~30% of the high-conviction
(mag≥0.70 & conf≥0.85) signals. This scores EVERY high-conviction bullish-with-ticker
signal in both caches (bull + 2022) with local llama3.2 / materiality_fewshot, skipping
any already cached. Resumable; writes into prompt_exp_scores.json under the same key
scheme ("llama3.2:materiality_fewshot:{cache_key}").

Usage:  .venv/bin/python score_lotto_signals.py
"""
import json, os, time
from pathlib import Path

from dotenv import load_dotenv; load_dotenv()
from openai import OpenAI

import backtest as _bt
from prompt_lab import PROMPTS, score_article
import config as _cfg

SCORES = Path("prompt_exp_scores.json")
MODEL, PROMPT = "llama3.2", "materiality_fewshot"
CACHES = ["dual_score_cache.json", "bear_dual_cache.json"]


def main():
    client = OpenAI(base_url=os.environ.get("OLLAMA_BASE_URL", "http://localhost:11434/v1"),
                    api_key="ollama")
    system = PROMPTS[PROMPT]
    sc = json.loads(SCORES.read_text()) if SCORES.exists() else {}

    # collect every high-conviction signal needing a score
    todo = []
    for cache in CACHES:
        raw = json.load(open(cache))
        for v in raw.values():
            if not isinstance(v, dict):
                continue
            b = v.get("bullish", {}) or {}
            a = v.get("_article", {}) or {}
            if b.get("reasoning") == "SCORE_FAILED" or not (b.get("tickers") or []):
                continue
            h = a.get("headline")
            if not h:
                continue
            if float(b.get("magnitude", 0)) < _cfg.LOTTO_MIN_MAGNITUDE or \
               float(b.get("confidence", 0)) < _cfg.LOTTO_MIN_CONFIDENCE:
                continue
            ck = _bt.cache_key(h, a.get("summary", ""))
            key = f"{MODEL}:{PROMPT}:{ck}"
            if key in sc:
                continue
            todo.append((key, h, a.get("summary", "")))

    # de-dup by key (same article can appear via both bull windows)
    seen = set(); todo = [(k, h, b) for (k, h, b) in todo if not (k in seen or seen.add(k))]
    print(f"scoring {len(todo)} high-conviction signals on {MODEL} / {PROMPT} …", flush=True)

    done = none = 0
    for i, (key, h, body) in enumerate(todo, 1):
        s = score_article(client, MODEL, system, h, body)
        sc[key] = s
        done += 1; none += (s is None)
        if done % 25 == 0:
            SCORES.write_text(json.dumps(sc))
            print(f"   …scored {done}/{len(todo)}  (None so far: {none})", flush=True)
    SCORES.write_text(json.dumps(sc))
    print(f"DONE — scored {done} new ({none} parse-fail / {100*none/max(1,done):.0f}%)", flush=True)


if __name__ == "__main__":
    main()
