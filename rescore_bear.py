"""
rescore_bear.py — Re-score bearish articles with a bearish-aware prompt so
the MODEL picks put tickers.

Problem: backtest_score_cache.json has 1,311 bearish articles but only 30 of
them have article text saved in _article. The other 1,281 were cached before
the _article field was added. We must re-fetch the news from Alpaca for the
same window and re-score any article whose existing cache score is "bearish".

Flow:
  1. Stream news from Alpaca for the same 180-day window day-by-day
  2. For each article:
       a. Look it up in backtest_score_cache.json by cache_key(headline, body)
       b. If the existing score is "bearish" → re-score with the bear prompt
       c. Save the new score to bear_score_cache.json (checkpoint immediately)
  3. Skip articles already in bear_score_cache.json (resumable)

Reads:  backtest_score_cache.json  (existing scores — NEVER modified)
Writes: bear_score_cache.json      (new bear scores, one entry per article)

Usage:
    .venv/bin/python rescore_bear.py                   # full 180-day window
    .venv/bin/python rescore_bear.py --days 30         # shorter window (test)
    .venv/bin/python rescore_bear.py --min-mag 0.0     # score ALL bearish (default)
    .venv/bin/python rescore_bear.py --min-mag 0.35    # only meaningful signals
    .venv/bin/python rescore_bear.py --limit 20        # stop after 20 rescores (smoke test)

Output bear_score_cache.json schema:
    {
      "<orig_cache_key>": {
        "tickers":    ["CRM"],
        "sentiment":  "bearish",
        "confidence": 0.80,
        "magnitude":  0.55,
        "reasoning":  "...",
        "_orig_key":  "<same key>",
        "_article":   { headline, summary, created_at, symbols }
      }
    }
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

SOURCE_CACHE_FILE = Path("backtest_score_cache.json")
BEAR_CACHE_FILE   = Path("bear_score_cache.json")

OLLAMA_BASE_URL   = os.environ.get("OLLAMA_BASE_URL", "http://localhost:11434/v1")
OLLAMA_MODEL      = os.environ.get("OLLAMA_MODEL", "llama3.2")

ALPACA_KEY    = os.environ["ALPACA_KEY"]
ALPACA_SECRET = os.environ["ALPACA_SECRET"]

BEAR_SYSTEM_PROMPT = """You are a quantitative equity trading signal generator identifying SHORT opportunities.
Respond with a JSON object ONLY — no markdown, no code fences, no explanation.

Schema:
{
  "tickers":    ["CRM"],
  "sentiment":  "bearish",
  "confidence": 0.82,
  "magnitude":  0.75,
  "reasoning":  "one sentence"
}

sentiment: always "bearish" — you are identifying stocks to short via put options
confidence: float 0.0–1.0 — how certain that named stocks will be hurt by this news
magnitude:  float 0.0–1.0 — how likely this news is to move the stock DOWNWARD

Magnitude scale:
  0.0–0.2  Noise (routine filings, minor analyst reiterations, fluff)
  0.2–0.4  Routine negative (in-line miss, small contract loss, minor downgrade)
  0.4–0.6  Meaningful negative catalyst (earnings miss, guidance cut, notable loss)
  0.6–0.8  Strong negative catalyst (large miss, regulatory action, product failure)
  0.8–1.0  Transformative negative (existential threat, major scandal, catastrophic earnings)

Rules:
- Identify the specific ticker(s) MOST DIRECTLY HURT by this news (1–3 at most).
- Only include tickers you are highly confident will be negatively impacted.
- General macro/political news with no specific company victim → empty tickers list.
- Reserve 0.7+ magnitude for genuinely severe events.
- Do NOT flag tickers that BENEFIT from this news; only the losers.
- Return ONLY the JSON object."""

# ─── Helpers ─────────────────────────────────────────────────────────────────

def cache_key(headline: str, body: str) -> str:
    return hashlib.md5(f"{headline}||{body[:500]}".encode()).hexdigest()

news_client = NewsClient(ALPACA_KEY, ALPACA_SECRET)

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
        log.warning("Alpaca news fetch failed for %s: %s", day.date(), e)
        return []

ollama = OpenAI(base_url=OLLAMA_BASE_URL, api_key="ollama")

def score_article(headline: str, body: str) -> Optional[dict]:
    try:
        resp = ollama.chat.completions.create(
            model=OLLAMA_MODEL,
            messages=[
                {"role": "system", "content": BEAR_SYSTEM_PROMPT},
                {"role": "user",   "content": (
                    f"Headline: {headline}\n\nBody: {body[:1200]}\n\n"
                    "Which specific stock(s) are HURT by this news? Respond with JSON only."
                )},
            ],
            temperature=0.1,
            timeout=90,
        )
        raw = resp.choices[0].message.content.strip()
        if raw.startswith("```"):
            raw = raw.split("```")[1]
            if raw.startswith("json"):
                raw = raw[4:]
        result = json.loads(raw.strip())
        result["sentiment"] = "bearish"
        return result
    except json.JSONDecodeError as e:
        log.warning("JSON parse failed: %s", e)
        return None
    except Exception as e:
        log.warning("Ollama error: %s", e)
        return None

def load_bear_cache() -> dict:
    if BEAR_CACHE_FILE.exists():
        try:
            data = json.loads(BEAR_CACHE_FILE.read_text())
            log.info("📦 Loaded %d existing bear scores from %s", len(data), BEAR_CACHE_FILE)
            return data
        except Exception:
            log.warning("Could not load %s — starting fresh", BEAR_CACHE_FILE)
    return {}

def save_bear_cache(cache: dict):
    BEAR_CACHE_FILE.write_text(json.dumps(cache, indent=2, default=str))

# ─── Main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--days",    type=int,   default=180,  help="Window size in days (default 180)")
    parser.add_argument("--min-mag", type=float, default=0.0,  help="Min original magnitude to re-score (default 0)")
    parser.add_argument("--limit",   type=int,   default=0,    help="Stop after N re-scores (0=unlimited, for testing)")
    parser.add_argument("--delay",   type=float, default=0.1,  help="Seconds between Ollama calls (default 0.1)")
    args = parser.parse_args()

    if not SOURCE_CACHE_FILE.exists():
        log.error("Source cache %s not found", SOURCE_CACHE_FILE)
        return

    source_cache = json.loads(SOURCE_CACHE_FILE.read_text())
    log.info("📰 Source cache: %d entries", len(source_cache))

    bear_cache = load_bear_cache()

    # Date window — same logic as backtest.py
    end_dt   = datetime.now(timezone.utc) - timedelta(days=5)
    start_dt = end_dt - timedelta(days=args.days)
    log.info("🗓  Window: %s → %s  (%d days)",
             start_dt.date(), end_dt.date(), args.days)

    # Identify how many bearish entries we expect
    total_bearish = sum(
        1 for v in source_cache.values()
        if v.get("sentiment") == "bearish"
        and v.get("magnitude", 0) >= args.min_mag
    )
    log.info("🐻 Source cache has %d bearish entries (min_mag=%.2f) to find and re-score",
             total_bearish, args.min_mag)

    # Stream through days
    day          = start_dt
    days_scanned = 0
    rescored     = 0
    skipped_done = 0
    not_bearish  = 0
    failed       = 0

    while day <= end_dt:
        articles = fetch_day(day)
        for art in articles:
            headline = art["headline"]
            body     = art["summary"]
            key      = cache_key(headline, body)

            # Must exist in source cache
            if key not in source_cache:
                continue

            orig = source_cache[key]

            # Must be bearish with sufficient magnitude
            if orig.get("sentiment") != "bearish":
                not_bearish += 1
                continue
            if orig.get("magnitude", 0) < args.min_mag:
                continue

            # Skip if already rescored (resume support)
            if key in bear_cache:
                skipped_done += 1
                continue

            # Re-score with bear prompt
            result = score_article(headline, body)
            if args.delay > 0:
                time.sleep(args.delay)

            if result is None:
                failed += 1
                log.warning("FAILED [%d rescored] %s", rescored, headline[:60])
                # Store failure marker so we don't retry this run
                bear_cache[key] = {
                    "tickers":    [],
                    "sentiment":  "bearish",
                    "confidence": float(orig.get("confidence", 0)),
                    "magnitude":  float(orig.get("magnitude", 0)),
                    "reasoning":  "SCORE_FAILED",
                    "_orig_key":  key,
                    "_article":   art,
                }
            else:
                tickers = result.get("tickers", [])
                bear_cache[key] = {
                    **result,
                    "_orig_key": key,
                    "_article":  art,
                }
                if tickers:
                    log.info("[%d] ✓ %-55s → %s (mag=%.2f)",
                             rescored + 1, headline[:55], tickers,
                             result.get("magnitude", 0))

            # Checkpoint after every article
            save_bear_cache(bear_cache)
            rescored += 1

            if args.limit > 0 and rescored >= args.limit:
                log.info("--limit %d reached, stopping.", args.limit)
                _print_summary(bear_cache, rescored, failed)
                return

        days_scanned += 1
        if days_scanned % 30 == 0:
            log.info("  …%d/%d days scanned | rescored=%d skipped=%d",
                     days_scanned, args.days, rescored, skipped_done)

        day += timedelta(days=1)

    log.info("─" * 60)
    log.info("Scan complete. Days=%d  Articles rescored=%d  Failed=%d  Already-done=%d",
             days_scanned, rescored, failed, skipped_done)
    _print_summary(bear_cache, rescored, failed)


def _print_summary(cache: dict, rescored: int = 0, failed: int = 0):
    total        = len(cache)
    with_tickers = sum(1 for v in cache.values()
                       if v.get("tickers") and v.get("reasoning") != "SCORE_FAILED")
    log.info("─" * 60)
    log.info("bear_score_cache.json: %d total | %d with ≥1 ticker (%.1f%%) | %d failed",
             total, with_tickers,
             100 * with_tickers / total if total else 0,
             failed)
    log.info("Output: %s", BEAR_CACHE_FILE.resolve())


if __name__ == "__main__":
    main()
