"""
dual_prompt_test.py — Test a neutral dual-sentiment prompt vs the current bullish-biased one.

Tests on 100 articles from dual_score_cache.json using:
  1. llama3.2 (local, current model)
  2. Groq 70B (hosted, larger)

Metrics:
  - Sentiment distribution (bullish / bearish / neutral)
  - Ticker identification rate
  - Agreement with existing dual_score_cache scores
  - Spearman correlation with 5d realized returns (signal quality)

Pass criteria: bearish signal quality (corr, selection vs random) comparable to bullish.

Usage:  USE_YAHOO_BARS=1 python dual_prompt_test.py
"""
import json
import os
import random
import statistics
import time
from datetime import datetime, timezone, timedelta
from pathlib import Path

from openai import OpenAI
from dotenv import load_dotenv
load_dotenv()

os.environ.setdefault("USE_YAHOO_BARS", "1")

from backtest import get_price_at, is_valid_stock_ticker
from benchmark import compute_stats

CACHE_FILE = Path("dual_score_cache.json")
N_SAMPLE   = 100
SEED       = 42

# ── Neutral dual-sentiment prompt ─────────────────────────────────────────────
DUAL_PROMPT = """You are a quantitative equity trading signal generator.
Respond with a JSON object ONLY — no markdown, no code fences, no explanation.

Schema:
{
  "tickers":    ["AAPL"],
  "sentiment":  "bullish",
  "confidence": 0.82,
  "magnitude":  0.75,
  "reasoning":  "one sentence"
}

sentiment: "bullish" | "bearish" | "neutral"
confidence: float 0.0–1.0 — how certain you are about the sentiment direction
magnitude:  float 0.0–1.0 — how likely this news is to meaningfully move the stock price

Magnitude scale:
  0.0–0.2  Noise or irrelevant
  0.2–0.4  Routine news
  0.4–0.6  Meaningful catalyst
  0.6–0.8  Strong catalyst
  0.8–1.0  Transformative event

Rules:
- Include tickers only when highly confident about a specific company.
- General macro news with no specific company → empty tickers list.
- Be conservative with magnitude; reserve 0.7+ for genuinely exceptional events.
- Flag BOTH bullish AND bearish sentiment accurately — do not bias toward either.
- Return ONLY the JSON object."""


def score_article(client, model, headline, body):
    try:
        resp = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": DUAL_PROMPT},
                {"role": "user",   "content": f"Headline: {headline}\n\nBody: {body[:800]}\n\nRespond with JSON only."},
            ],
            temperature=0.1,
            max_tokens=200,
        )
        raw = resp.choices[0].message.content.strip()
        if raw.startswith("```"):
            raw = raw.split("```")[1]
            if raw.startswith("json"):
                raw = raw[4:]
        return json.loads(raw.strip())
    except Exception:
        return None


def realized_return(ticker, created_at, days=5):
    """5-day forward return from article date."""
    entry = get_price_at(ticker, created_at)
    if not entry:
        return None
    exit_dt = created_at + timedelta(days=days + 3)  # +3 for weekends
    exit_p  = get_price_at(ticker, exit_dt)
    if not exit_p:
        return None
    return (exit_p / entry - 1) * 100


def spearman(xs, ys):
    n = len(xs)
    if n < 3:
        return float("nan")
    rank_x = {v: i for i, v in enumerate(sorted(xs))}
    rank_y = {v: i for i, v in enumerate(sorted(ys))}
    rx = [rank_x[x] for x in xs]
    ry = [rank_y[y] for y in ys]
    mean_rx = sum(rx) / n
    mean_ry = sum(ry) / n
    cov = sum((rx[i] - mean_rx) * (ry[i] - mean_ry) for i in range(n)) / n
    std_rx = (sum((r - mean_rx) ** 2 for r in rx) / n) ** 0.5 or 1
    std_ry = (sum((r - mean_ry) ** 2 for r in ry) / n) ** 0.5 or 1
    return cov / (std_rx * std_ry)


def evaluate(scored_articles):
    """Measure signal quality: Spearman corr and quintile returns."""
    pairs = []
    for a in scored_articles:
        if not a.get("tickers") or a.get("sentiment") not in ("bullish", "bearish"):
            continue
        ticker = a["tickers"][0]
        if not is_valid_stock_ticker(ticker):
            continue
        ret = realized_return(ticker, a["created_at"])
        if ret is None:
            continue
        score = a["magnitude"] * a["confidence"]
        if a["sentiment"] == "bearish":
            score = -score   # negative score for bearish
        pairs.append((score, ret))

    if len(pairs) < 10:
        return {"n": len(pairs), "spearman": float("nan"), "note": "too few"}

    scores, rets = zip(*sorted(pairs, key=lambda x: x[0]))
    corr = spearman(list(scores), list(rets))
    n = len(pairs)
    q = n // 5
    q1_ret = sum(rets[:q]) / q if q else 0
    q5_ret = sum(rets[-q:]) / q if q else 0
    return {
        "n":       n,
        "spearman": round(corr, 3),
        "Q1_ret%": round(q1_ret, 2),
        "Q5_ret%": round(q5_ret, 2),
    }


def run_model(name, client, model, sample, cache_scores):
    print(f"\n── {name} ({model}) ──")
    results = []
    bull = bear = neutral = no_ticker = failed = 0

    for i, art in enumerate(sample):
        if (i + 1) % 20 == 0:
            print(f"  {i+1}/{len(sample)} ...", flush=True)
        s = score_article(client, model, art["headline"], art["body"])
        if not s:
            failed += 1
            continue
        sent = s.get("sentiment", "neutral")
        if sent   == "bullish":  bull    += 1
        elif sent == "bearish":  bear    += 1
        else:                    neutral += 1
        if not s.get("tickers"): no_ticker += 1
        results.append({**s, "created_at": art["created_at"],
                        "headline": art["headline"]})

    total = len(results)
    print(f"  Scored: {total}  (failed: {failed})")
    print(f"  Sentiment: bullish={bull} ({bull/max(total,1)*100:.0f}%)  "
          f"bearish={bear} ({bear/max(total,1)*100:.0f}%)  "
          f"neutral={neutral} ({neutral/max(total,1)*100:.0f}%)")
    print(f"  No ticker: {no_ticker/max(total,1)*100:.0f}%")

    # Agreement with cache
    agreed = 0
    for r in results:
        key = r["headline"]
        cached = cache_scores.get(key)
        if cached:
            cache_sent = cached.get("bullish", {}).get("sentiment") or "bullish"
            if r["sentiment"] == cache_sent:
                agreed += 1
    if cache_scores:
        print(f"  Agrees with cached dual-score: {agreed/max(total,1)*100:.0f}%")

    # Signal quality
    q = evaluate(results)
    print(f"  Signal quality (dual): n={q['n']}  Spearman={q['spearman']:.3f}  "
          f"Q5={q.get('Q5_ret%','?')}%  Q1={q.get('Q1_ret%','?')}%")

    return results


def main():
    if not CACHE_FILE.exists():
        print(f"Cache not found: {CACHE_FILE}"); return

    # Load sample
    raw = json.loads(CACHE_FILE.read_text())
    entries = list(raw.values())
    # Filter to articles with headlines and in the bull window
    valid = [
        {
            "headline":   e["_article"].get("headline", ""),
            "body":       e["_article"].get("summary", ""),
            "created_at": datetime.fromisoformat(
                str(e["_article"].get("created_at", "")).replace("Z", "+00:00")
            ).replace(tzinfo=timezone.utc),
            "cache_bull": e.get("bullish", {}),
            "cache_bear": e.get("bearish", {}),
        }
        for e in entries
        if e.get("_article", {}).get("headline")
        and e.get("_article", {}).get("created_at")
        and e.get("bullish", {}).get("reasoning") != "SCORE_FAILED"
    ]
    random.seed(SEED)
    sample = random.sample(valid, min(N_SAMPLE, len(valid)))
    cache_scores = {a["headline"]: {"bullish": a["cache_bull"], "bearish": a["cache_bear"]}
                    for a in sample}

    print(f"Dual-sentiment prompt test  (n={len(sample)} articles)\n")

    # ── Local llama3.2 ────────────────────────────────────────────────────────
    ollama_client = OpenAI(
        base_url=os.environ.get("OLLAMA_BASE_URL", "http://localhost:11434/v1"),
        api_key="ollama"
    )
    run_model("Local llama3.2", ollama_client, "llama3.2", sample, cache_scores)

    # ── Groq 70B ──────────────────────────────────────────────────────────────
    groq_key  = os.environ.get("LAB_HOSTED_KEY", "")
    groq_base = os.environ.get("LAB_HOSTED_BASE_URL", "https://api.groq.com/openai/v1")
    groq_model= os.environ.get("LAB_HOSTED_MODEL", "llama-3.3-70b-versatile")

    if groq_key:
        groq_client = OpenAI(base_url=groq_base, api_key=groq_key)
        run_model("Groq 70B", groq_client, groq_model, sample, cache_scores)
    else:
        print("\n  [Groq] LAB_HOSTED_KEY not set — skipping Groq test")

    # ── Baseline: existing dual cache quality ─────────────────────────────────
    print("\n── Baseline: existing dual_score_cache quality ──")
    cache_scored = []
    for a in sample:
        bull = a["cache_bull"]
        if bull.get("tickers") and bull.get("magnitude") and bull.get("confidence"):
            cache_scored.append({
                "tickers":    bull["tickers"],
                "sentiment":  "bullish",
                "magnitude":  float(bull.get("magnitude", 0)),
                "confidence": float(bull.get("confidence", 0)),
                "created_at": a["created_at"],
            })
        bear = a["cache_bear"]
        if bear.get("tickers") and bear.get("magnitude") and bear.get("confidence"):
            cache_scored.append({
                "tickers":    bear["tickers"],
                "sentiment":  "bearish",
                "magnitude":  float(bear.get("magnitude", 0)),
                "confidence": float(bear.get("confidence", 0)),
                "created_at": a["created_at"],
            })
    q = evaluate(cache_scored)
    print(f"  Cache scores: n={q['n']}  Spearman={q['spearman']:.3f}  "
          f"Q5={q.get('Q5_ret%','?')}%  Q1={q.get('Q1_ret%','?')}%")

    print("\n── Interpretation ──")
    print("  Good dual prompt: bearish Spearman ≤ bullish (both low is OK)")
    print("  Sentiment distribution: expect ~50% bullish, ~30% neutral, ~20% bearish")
    print("  If bearish rate << 20%: model still biased despite neutral framing")


if __name__ == "__main__":
    main()
