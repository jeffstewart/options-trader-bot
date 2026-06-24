"""
router_rescore.py — re-score cached articles with an LLM STRATEGY ROUTER prompt.

Unlike the rule-based router (router_backtest.py), here the LLM itself picks the
trading strategy for each article. Output per article:

    {
      "ticker":     "AAPL",      # primary ticker ("" if none / macro)
      "sentiment":  "bullish",   # bullish | bearish | neutral
      "strategy":   "long_calls",# see STRATEGY SET below ("none" = skip)
      "confidence": 0.0-1.0,     # certainty of direction
      "magnitude":  0.0-1.0,     # how much it should move
      "swing":      0.0-1.0,     # probability of a LARGE move (drives lotto)
      "reasoning":  "one sentence"
    }

Reads articles from an existing cache's `_article` fields (NO re-fetch from
Alpaca — the dual cache already holds headline+summary+date for every article).
Writes router_llm_cache.json (resumable, checkpoints every article).

Model: defaults to local llama3.2 (the SAME model the live bot routes with — so
the backtest matches what we'd deploy). --hosted uses Groq (LAB_HOSTED_* in .env)
for a stronger-model comparison on a sample.

Usage:
    .venv/bin/python router_rescore.py --limit 10              # smoke test (local)
    .venv/bin/python router_rescore.py --hosted --limit 50     # smoke test (Groq 70B)
    nohup .venv/bin/python router_rescore.py >> router_rescore.log 2>&1 &   # full overnight
"""
import argparse, hashlib, json, logging, os, time
from pathlib import Path
from typing import Optional

from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()
logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s  %(levelname)-8s  %(message)s", datefmt="%H:%M:%S")
log = logging.getLogger(__name__)

OLLAMA_BASE_URL = os.environ.get("OLLAMA_BASE_URL", "http://localhost:11434/v1")
OLLAMA_MODEL    = os.environ.get("OLLAMA_MODEL", "llama3.2")

ROUTER_SYSTEM_PROMPT = """You are a trading strategist. For each news article, decide how to trade it.
Respond with a JSON object ONLY — no markdown, no code fences, no explanation.

Schema:
{
  "ticker":     "AAPL",
  "sentiment":  "bullish",
  "strategy":   "long_calls",
  "confidence": 0.75,
  "magnitude":  0.60,
  "swing":      0.30,
  "reasoning":  "one sentence"
}

STEP 1 — Is there a specific PUBLIC COMPANY (stock ticker) that this news clearly
helps or hurts? If NO (pure macro/economy/politics, market commentary, crypto-only,
listicles with no single clear name) → set strategy "none", ticker "".
If YES → pick the best strategy below for that company.

STRATEGY SET (choose exactly one):
  "pead"        A positive EARNINGS surprise (beat and/or raised guidance) → hold the
                stock for weeks to ride post-earnings drift.
  "lotto_calls" Strongly bullish AND a LARGE, fast move looks likely (binary catalyst,
                takeover, surprise approval, blowout result; swing ≥ 0.6) → cheap OTM
                calls for a convex high-risk/high-reward payoff.
  "long_calls"  Bullish with a clear catalyst (not earnings, not a giant swing) → ATM calls.
  "long_stock"  Bullish but lower conviction or a slower/steadier mover → buy shares.
  "bear_short"  Bearish — expect the stock to fall → short the shares.
  "none"        ONLY when there is no specific tradeable company (see STEP 1).

Fields:
  ticker:     the one stock symbol to trade ("" only if strategy is "none")
  sentiment:  "bullish" | "bearish" | "neutral"
  confidence: 0.0–1.0 certainty about the direction
  magnitude:  0.0–1.0 how much the stock is likely to move
  swing:      0.0–1.0 probability of a LARGE/violent move (≥0.6 favors lotto_calls)

Rules:
  - If the article names a company with any clear directional lean, DO pick a trading
    strategy (default to long_stock when bullish but unsure which) — do NOT use "none".
  - "none" is for genuinely no-single-company news only.
  - "lotto_calls" requires bullish AND swing ≥ 0.6;  "pead" requires an earnings beat.
  - Return ONLY the JSON object."""

_VALID = {"long_calls", "lotto_calls", "long_stock", "pead", "bear_short", "none"}


def cache_key(headline: str, body: str) -> str:
    return hashlib.md5(f"{headline}||{body[:500]}".encode()).hexdigest()


def score_article(client, model, headline, body) -> Optional[dict]:
    try:
        resp = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": ROUTER_SYSTEM_PROMPT},
                {"role": "user", "content": (
                    f"Headline: {headline}\n\nBody: {body[:1200]}\n\n"
                    "Pick the single best strategy. Respond with JSON only.")},
            ],
            temperature=0.1, max_tokens=220, timeout=90,
        )
        raw = resp.choices[0].message.content.strip()
        if raw.startswith("```"):
            raw = raw.split("```")[1]
            if raw.startswith("json"):
                raw = raw[4:]
        r = json.loads(raw.strip())
        # Normalize
        strat = str(r.get("strategy", "none")).strip().lower()
        r["strategy"]   = strat if strat in _VALID else "none"
        tk = r.get("ticker", "")
        r["ticker"]     = (tk[0] if isinstance(tk, list) and tk else tk) or ""
        r["sentiment"]  = str(r.get("sentiment", "neutral")).strip().lower()
        for f in ("confidence", "magnitude", "swing"):
            try:    r[f] = float(r.get(f, 0.0))
            except Exception: r[f] = 0.0
        r.setdefault("reasoning", "")
        return r
    except json.JSONDecodeError:
        return None
    except Exception as e:
        log.warning("scoring error: %s", e)
        return None


def main():
    ap = argparse.ArgumentParser(description="LLM strategy-router rescore")
    ap.add_argument("--articles-from", default="dual_score_cache.json",
                    help="Cache file whose _article entries supply the headlines (default dual_score_cache.json)")
    ap.add_argument("--output-file", default="router_llm_cache.json")
    ap.add_argument("--hosted", action="store_true", help="Use Groq (LAB_HOSTED_*) instead of local Ollama")
    ap.add_argument("--limit", type=int, default=0, help="Stop after N new scores (0=all)")
    ap.add_argument("--delay", type=float, default=0.0, help="Seconds between calls")
    args = ap.parse_args()

    src = Path(args.articles_from)
    if not src.exists():
        log.error("Article source cache not found: %s", src); return
    out = Path(args.output_file)

    if args.hosted:
        key = os.environ.get("LAB_HOSTED_KEY", "")
        if not key:
            log.error("--hosted set but LAB_HOSTED_KEY missing in .env"); return
        client = OpenAI(base_url=os.environ.get("LAB_HOSTED_BASE_URL",
                        "https://api.groq.com/openai/v1"), api_key=key)
        model  = os.environ.get("LAB_HOSTED_MODEL", "llama-3.3-70b-versatile")
    else:
        client = OpenAI(base_url=OLLAMA_BASE_URL, api_key="ollama")
        model  = OLLAMA_MODEL

    # Unique articles from the source cache
    raw = json.loads(src.read_text())
    seen, articles = set(), []
    for entry in raw.values():
        a = entry.get("_article", {})
        h = a.get("headline", "")
        if not h:
            continue
        k = cache_key(h, a.get("summary", ""))
        if k in seen:
            continue
        seen.add(k)
        articles.append((k, a))

    cache = json.loads(out.read_text()) if out.exists() else {}
    log.info("═" * 60)
    log.info("LLM STRATEGY ROUTER RESCORE   model=%s  hosted=%s", model, args.hosted)
    log.info("Source articles: %d unique  |  already scored: %d  |  output: %s",
             len(articles), len(cache), out)
    log.info("═" * 60)

    scored = failed = 0
    for k, a in articles:
        if k in cache:
            continue
        r = score_article(client, model, a.get("headline", ""), a.get("summary", ""))
        if args.delay:
            time.sleep(args.delay)
        if r is None:
            failed += 1
            cache[k] = {"strategy": "none", "ticker": "", "sentiment": "neutral",
                        "confidence": 0.0, "magnitude": 0.0, "swing": 0.0,
                        "reasoning": "SCORE_FAILED", "_article": a}
        else:
            cache[k] = {**r, "_article": a}
            if r["strategy"] != "none":
                log.info("[%d] %-46s → %-11s %s (sw=%.2f)", scored + 1,
                         a.get("headline", "")[:46], r["strategy"], r["ticker"], r.get("swing", 0))
        out.write_text(json.dumps(cache, indent=2, default=str))
        scored += 1
        if args.limit and scored >= args.limit:
            log.info("--limit %d reached.", args.limit)
            break

    # Summary
    mix = {}
    for v in cache.values():
        mix[v.get("strategy", "none")] = mix.get(v.get("strategy", "none"), 0) + 1
    log.info("─" * 60)
    log.info("Done. new=%d failed=%d total=%d", scored, failed, len(cache))
    log.info("Strategy mix: %s", "  ".join(f"{k}={v}" for k, v in sorted(mix.items())))


if __name__ == "__main__":
    main()
