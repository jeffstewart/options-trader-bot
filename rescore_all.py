"""
rescore_all.py — Re-score all 180-day articles with a dual-signal prompt that
extracts BOTH bullish (long-call) and bearish (long-put) tickers from each
article in a single model pass.

Why: in production during mixed or bear regimes the bot needs to identify both
winners and losers from the same news stream. The original scorer was told
"only flag bullish" so the bearish side was never extracted. This script
replaces that with a two-sided pass over every article.

Flow:
  1. Stream news from Alpaca day-by-day for the same 180-day window.
  2. For each article, check whether it's already in dual_score_cache.json
     (resume support). Skip if present.
  3. Score with the dual prompt → save to dual_score_cache.json immediately
     (checkpoint after every article).

Reads:  backtest_score_cache.json  (existing cache — for reference only, NEVER modified)
Writes: dual_score_cache.json      (new two-sided scores)

Output schema (one entry per cache_key):
    {
      "<md5_key>": {
        "bullish": {
          "tickers":    ["AAPL"],
          "confidence": 0.75,
          "magnitude":  0.60,
          "reasoning":  "strong earnings beat drives multiple expansion"
        },
        "bearish": {
          "tickers":    ["MSFT"],
          "confidence": 0.80,
          "magnitude":  0.65,
          "reasoning":  "guidance cut signals slowing Azure growth"
        },
        "_article": {
          "headline":   "...",
          "summary":    "...",
          "created_at": "...",
          "symbols":    [...]
        }
      }
    }

Either "bullish" or "bearish" (or both) can have an empty tickers list — that
is the correct output for a one-sided or irrelevant article.

Usage:
    .venv/bin/python rescore_all.py               # full 180-day run (all articles)
    .venv/bin/python rescore_all.py --days 30     # shorter window (quick test)
    .venv/bin/python rescore_all.py --limit 10    # stop after 10 scores (smoke test)
    .venv/bin/python rescore_all.py --delay 0     # no inter-call delay (fast, local Ollama)

Estimated runtime: ~8,600 articles × ~3 s/article ≈ 7–8 hours unattended.
Start it before bed; it will checkpoint every article and resume cleanly on restart.

Kick-off command:
    nohup .venv/bin/python rescore_all.py --delay 0 >> rescore_all.log 2>&1 &
    echo "PID: $!"

Monitor:
    tail -f rescore_all.log
    python3 -c "import json; c=json.load(open('dual_score_cache.json')); bull=sum(1 for v in c.values() if v.get('bullish',{}).get('tickers')); bear=sum(1 for v in c.values() if v.get('bearish',{}).get('tickers')); print(f'Scored: {len(c)}  bull signals: {bull}  bear signals: {bear}')"
"""

import argparse
import hashlib
import json
import logging
import os
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Optional

from dotenv import load_dotenv
from openai import OpenAI
from alpaca.data.historical.news import NewsClient
from alpaca.data.requests import NewsRequest

load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

# ─── Config ──────────────────────────────────────────────────────────────────

DUAL_CACHE_FILE   = Path("dual_score_cache.json")
OLLAMA_BASE_URL   = os.environ.get("OLLAMA_BASE_URL", "http://localhost:11434/v1")
OLLAMA_MODEL      = os.environ.get("OLLAMA_MODEL", "llama3.2")
ALPACA_KEY        = os.environ.get("ALPACA_KEY") or os.environ["ALPACA_API_KEY"]
ALPACA_SECRET     = os.environ.get("ALPACA_SECRET") or os.environ["ALPACA_SECRET_KEY"]

# ─── Prompt ──────────────────────────────────────────────────────────────────

DUAL_SYSTEM_PROMPT = """You are a quantitative equity trading signal generator.
For each news article, identify stocks that will be HELPED (long calls) and stocks that will be HURT (long puts).
Respond with a JSON object ONLY — no markdown, no code fences, no explanation.

Schema:
{
  "bullish": {
    "tickers":    ["AAPL"],
    "confidence": 0.75,
    "magnitude":  0.60,
    "reasoning":  "one sentence explaining why these stocks benefit"
  },
  "bearish": {
    "tickers":    ["MSFT"],
    "confidence": 0.80,
    "magnitude":  0.65,
    "reasoning":  "one sentence explaining why these stocks are hurt"
  }
}

Fields:
  tickers:    list of 1–3 ticker symbols you are highly confident about; [] if none
  confidence: float 0.0–1.0 — certainty about direction for these specific tickers
  magnitude:  float 0.0–1.0 — how much these tickers are likely to move

Magnitude scale (same for both sides):
  0.0–0.2  Noise / irrelevant (routine filings, minor reiterations, fluff)
  0.2–0.4  Routine (in-line earnings, small contract, minor rating change)
  0.4–0.6  Meaningful catalyst (solid beat/miss, notable guidance change, partnership)
  0.6–0.8  Strong catalyst (large beat/miss, major acquisition/loss, FDA action)
  0.8–1.0  Transformative (company-defining event, paradigm shift, major scandal)

Rules:
  - Both lists can be non-empty: e.g. a merger benefits the acquiree, hurts the acquirer.
  - Either list can be empty [] if this news has no clear directional implication for that side.
  - General macro news with no specific company → both lists empty.
  - Only include tickers you are highly confident about (1–3 per side max).
  - Be conservative: most news is routine. Reserve magnitude ≥ 0.65 for genuinely exceptional events.
  - A ticker may appear on BOTH sides only if the news has genuinely mixed implications for it.
  - Return ONLY the JSON object."""

# ─── Helpers ─────────────────────────────────────────────────────────────────

def cache_key(headline: str, body: str) -> str:
    return hashlib.md5(f"{headline}||{body[:500]}".encode()).hexdigest()

ollama      = OpenAI(base_url=OLLAMA_BASE_URL, api_key="ollama")
news_client = NewsClient(ALPACA_KEY, ALPACA_SECRET)

_EMPTY_SIDE = {"tickers": [], "confidence": 0.0, "magnitude": 0.0, "reasoning": ""}

def score_article(headline: str, body: str) -> Optional[dict]:
    """Call Ollama with the dual prompt. Returns {"bullish": {...}, "bearish": {...}} or None."""
    try:
        resp = ollama.chat.completions.create(
            model=OLLAMA_MODEL,
            messages=[
                {"role": "system", "content": DUAL_SYSTEM_PROMPT},
                {"role": "user",   "content": (
                    f"Headline: {headline}\n\nBody: {body[:1200]}\n\n"
                    "Which stocks BENEFIT (bullish) and which are HURT (bearish)? Respond with JSON only."
                )},
            ],
            temperature=0.1,
            timeout=90,
        )
        raw = resp.choices[0].message.content.strip()
        # Strip markdown code fences if model wraps output
        if raw.startswith("```"):
            raw = raw.split("```")[1]
            if raw.startswith("json"):
                raw = raw[4:]
        result = json.loads(raw.strip())

        # Normalise: ensure both keys exist with required sub-fields
        for side in ("bullish", "bearish"):
            if side not in result or not isinstance(result[side], dict):
                result[side] = dict(_EMPTY_SIDE)
            s = result[side]
            s.setdefault("tickers",    [])
            s.setdefault("confidence", 0.0)
            s.setdefault("magnitude",  0.0)
            s.setdefault("reasoning",  "")
            # Clamp to list in case model returns a string
            if isinstance(s["tickers"], str):
                s["tickers"] = [s["tickers"]] if s["tickers"] else []

        return result

    except json.JSONDecodeError as e:
        log.warning("JSON parse failed: %s", e)
        return None
    except Exception as e:
        log.warning("Ollama error: %s", e)
        return None

def fetch_day(day: datetime) -> list[dict]:
    start = day.replace(hour=0, minute=0, second=0, microsecond=0)
    end   = start + timedelta(hours=23, minutes=59, seconds=59)
    try:
        req   = NewsRequest(start=start, end=end, limit=50, sort="desc",
                            include_content=False, exclude_contentless=True)
        resp  = news_client.get_news(req)
        items = (resp.data or {}).get("news", [])
        return [{
            "headline":   getattr(i, "headline", "") or "",
            "summary":    getattr(i, "summary",  "") or "",
            "created_at": str(getattr(i, "created_at", "") or ""),
            "symbols":    list(getattr(i, "symbols", []) or []),
        } for i in items]
    except Exception as e:
        log.warning("News fetch failed for %s: %s", day.date(), e)
        return []

def load_dual_cache() -> dict:
    if DUAL_CACHE_FILE.exists():
        try:
            data = json.loads(DUAL_CACHE_FILE.read_text())
            log.info("📦 Loaded %d existing dual scores from %s", len(data), DUAL_CACHE_FILE)
            return data
        except Exception:
            log.warning("Could not load %s — starting fresh", DUAL_CACHE_FILE)
    return {}

def save_dual_cache(cache: dict):
    DUAL_CACHE_FILE.write_text(json.dumps(cache, indent=2, default=str))

# ─── Main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Dual-signal (bull+bear) article scorer")
    parser.add_argument("--days",  type=int,   default=180, help="Window size in days (default 180)")
    parser.add_argument("--limit", type=int,   default=0,   help="Stop after N scored articles (0=unlimited)")
    parser.add_argument("--delay", type=float, default=0.1, help="Seconds between Ollama calls (default 0.1)")
    parser.add_argument("--end-date", type=str, default=None,
                        help="Window end date YYYY-MM-DD (default: today-5d). Use to target historical regimes.")
    parser.add_argument("--output-file", type=str, default=None,
                        help="Override output cache file (default: dual_score_cache.json). "
                             "Use to run bull + bear rescores in parallel without write conflicts.")
    args = parser.parse_args()

    # Allow --output-file to redirect the output without clobbering the default
    global DUAL_CACHE_FILE
    if args.output_file:
        DUAL_CACHE_FILE = Path(args.output_file)

    dual_cache = load_dual_cache()

    # Date window
    if args.end_date:
        end_dt = datetime.strptime(args.end_date, "%Y-%m-%d").replace(tzinfo=timezone.utc)
    else:
        end_dt = datetime.now(timezone.utc) - timedelta(days=5)
    start_dt = end_dt - timedelta(days=args.days)

    log.info("═" * 60)
    log.info("DUAL RESCORE  %s → %s  (%d days)",
             start_dt.date(), end_dt.date(), args.days)
    log.info("Model: %s  |  Output: %s", OLLAMA_MODEL, DUAL_CACHE_FILE)
    log.info("Already scored: %d | --limit: %s",
             len(dual_cache), args.limit if args.limit else "none")
    log.info("═" * 60)

    day          = start_dt
    days_total   = args.days
    days_scanned = 0
    scored       = 0
    skipped      = 0
    failed       = 0

    while day <= end_dt:
        articles = fetch_day(day)
        for art in articles:
            headline = art["headline"]
            body     = art["summary"]
            if not headline:
                continue

            key = cache_key(headline, body)

            # Resume: skip already scored
            if key in dual_cache:
                skipped += 1
                continue

            result = score_article(headline, body)
            if args.delay > 0:
                time.sleep(args.delay)

            if result is None:
                failed += 1
                # Store failure marker — won't be retried this run, will be on next
                dual_cache[key] = {
                    "bullish":  dict(_EMPTY_SIDE, reasoning="SCORE_FAILED"),
                    "bearish":  dict(_EMPTY_SIDE, reasoning="SCORE_FAILED"),
                    "_article": art,
                }
            else:
                dual_cache[key] = {**result, "_article": art}

                bull_t = result["bullish"].get("tickers", [])
                bear_t = result["bearish"].get("tickers", [])
                if bull_t or bear_t:
                    log.info("[%d] %-52s  bull=%s  bear=%s",
                             scored + 1, headline[:52], bull_t or "[]", bear_t or "[]")

            # Checkpoint immediately
            save_dual_cache(dual_cache)
            scored += 1

            if args.limit > 0 and scored >= args.limit:
                log.info("--limit %d reached, stopping.", args.limit)
                _print_summary(dual_cache, scored, failed)
                return

        days_scanned += 1
        if days_scanned % 30 == 0:
            log.info("  …%d/%d days | scored=%d skipped=%d failed=%d",
                     days_scanned, days_total, scored, skipped, failed)
        day += timedelta(days=1)

    log.info("─" * 60)
    log.info("Scan complete. Days=%d  New scores=%d  Skipped=%d  Failed=%d",
             days_scanned, scored, skipped, failed)
    _print_summary(dual_cache, scored, failed)


def _print_summary(cache: dict, scored: int = 0, failed: int = 0):
    total     = len(cache)
    bull_hits = sum(1 for v in cache.values()
                    if v.get("bullish", {}).get("tickers")
                    and v["bullish"].get("reasoning") != "SCORE_FAILED")
    bear_hits = sum(1 for v in cache.values()
                    if v.get("bearish", {}).get("tickers")
                    and v["bearish"].get("reasoning") != "SCORE_FAILED")
    both_hits = sum(1 for v in cache.values()
                    if v.get("bullish", {}).get("tickers")
                    and v.get("bearish", {}).get("tickers")
                    and v["bullish"].get("reasoning") != "SCORE_FAILED")
    log.info("─" * 60)
    log.info("dual_score_cache.json  total=%d  bull_signals=%d  bear_signals=%d  both=%d",
             total, bull_hits, bear_hits, both_hits)
    log.info("Output: %s", DUAL_CACHE_FILE.resolve())


if __name__ == "__main__":
    main()
