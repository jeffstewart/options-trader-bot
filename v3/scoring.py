"""
scoring.py (v3) — pluggable, context-aware LLM scorer.
Features:
  - Supports multiple providers: Ollama (default), Gemini, Anthropic, Moonshot/Kimi.
  - Injects point-in-time ticker news history (trailing lookback window) into the prompt.
  - Injects trailing 24h market news sentiment flow and macro headlines.
  - Explicit directives to detect stale echoes/recaps and ongoing corporate trends.
  - Robust JSON parsing with graceful fallbacks.
  - Local Ollama ticker-selection corrector.
"""

import csv
import json
import logging
import os
import re
import time
from datetime import datetime, timezone
from typing import Optional, Union

import requests
from openai import OpenAI

try:
    from v3 import config as cfg
    from v3 import news_db
except ImportError:
    import config as cfg
    import news_db

log = logging.getLogger(__name__)

SYSTEM_PROMPT = """You are an expert quantitative equity options analyst evaluating market news for directional catalysts.
Respond with a JSON object ONLY — no markdown code fences, no commentary outside the JSON.

JSON Schema:
{
  "tickers": ["AAPL"],
  "sentiment": "bullish",
  "confidence": 0.85,
  "magnitude": 0.75,
  "reasoning": "Direct comparison to prior news and assessment of catalyst freshness",
  "is_stale_echo": false
}

Definitions:
- sentiment: "bullish" | "bearish" | "neutral" (we trade long call options, so identify genuine upside catalysts)
- confidence: float 0.0–1.0 — certainty in the directional sentiment
- magnitude: float 0.0–1.0 — expected price impact / significance:
    0.0–0.2  Noise, routine update, recap, speculation, or already fully priced in
    0.2–0.4  Minor/incremental news (modest contract win, routine analyst chatter, sympathy move)
    0.4–0.6  Meaningful catalyst (unexpected earnings beat with raised guidance, strategic partnership, FDA clearance)
    0.6–0.8  Strong catalyst (transformative buyout offer, blockbuster drug approval)
    0.8–1.0  Rare paradigm shift (reserved for verified, historic corporate events)
- is_stale_echo: boolean — true if this article is merely recapping, explaining ("why stock surged"), or echoing news/themes already covered in the prior 1-7 days

Strict Decision Rules:
1. ECHO / RECAP DETECTION:
   - Carefully review the PRIOR NEWS HISTORY for the ticker.
   - If the current article refers to an event, theme, earnings report, clinical data, or product launch that already appeared in the prior 1 to 7 days, IT IS ALREADY DIGESTED BY THE MARKET.
   - YOU MUST set "is_stale_echo": true, "sentiment": "neutral", and "magnitude": <= 0.25.
   - Example of an Echo:
     Prior News: "[2026-09-15] Company X reports Q2 earnings beat, stock up 10%"
     Target Article: "[2026-09-17] Why Company X is rallying today as analysts praise quarter"
     -> is_stale_echo = true, magnitude = 0.20, sentiment = "neutral".
2. RETAIL HYPE & SPECULATION TRAPS:
   - Rumors ("Traders circulate unconfirmed rumor..."), sympathy moves ("X rises ahead of Tesla event..."), technical chart talk ("Stock breaks major resistance..."), or influencer chatter ("Billionaire criticizes company...") are high-IV retail traps.
   - For all such articles: set "magnitude": <= 0.25 and "sentiment": "neutral".
3. GENUINE FRESH CATALYSTS:
   - A genuine catalyst must be a NEW, verified, unannounced fundamental development (e.g. surprise buyout offer, unexpected quarterly earnings blowout with raised guidance, surprise FDA approval) with NO prior coverage in the trailing history.
   - In that case: "is_stale_echo": false, "sentiment": "bullish", "magnitude": 0.50–0.80.
4. REASONING:
   - State clearly in one sentence: (1) whether the event is novel or an echo of prior news, and (2) why it does or does not represent a tradeable catalyst."""


def build_context_prompt(headline: str, body: str, primary_ticker: str,
                         article_ts: Optional[datetime] = None,
                         lookback_days: Optional[int] = None,
                         db_path: Optional[str] = None) -> str:
    """
    Constructs the rich multi-part prompt including target article,
    trailing ticker history, and trailing market news sentiment.
    """
    days = lookback_days if lookback_days is not None else cfg.NEWS_LOOKBACK_DAYS
    ts = article_ts or datetime.now(timezone.utc)
    ts_str = ts.strftime("%Y-%m-%d %H:%M UTC")

    # 1. Ticker prior history
    ticker_history = []
    if primary_ticker:
        try:
            ticker_history = news_db.get_ticker_news_history(
                primary_ticker, as_of_ts=ts, days=days,
                limit=cfg.NEWS_MAX_TICKER_ARTICLES, db_path=db_path
            )
        except Exception as e:
            log.debug("Could not fetch ticker history for %s: %s", primary_ticker, e)

    # 2. Market sentiment context
    market_ctx = {}
    try:
        market_ctx = news_db.get_market_sentiment_context(
            as_of_ts=ts, hours=cfg.NEWS_MARKET_SENTIMENT_HOURS, db_path=db_path
        )
    except Exception as e:
        log.debug("Could not fetch market context: %s", e)

    # Format Ticker History
    if ticker_history:
        history_lines = []
        for a in ticker_history:
            ats = a.get("created_at", "")[:16].replace("T", " ")
            h_text = f"• [{ats}] {a['headline']}"
            if a.get("summary"):
                h_text += f" — Summary: {a['summary'][:150]}"
            if a.get("sentiment"):
                h_text += f" (Prior Scored: {a['sentiment']}, mag={a.get('magnitude')})"
            history_lines.append(h_text)
        history_block = "\n".join(history_lines)
    else:
        history_block = (f"• No other articles recorded for {primary_ticker or 'this ticker'} in the past {days} days. "
                         "(Treat as a fresh catalyst with no recent news overhang).")

    # Format Market Sentiment
    total_arts = market_ctx.get("total_articles", 0)
    scored_tot = market_ctx.get("scored_total", 0)
    macro_headlines = market_ctx.get("macro_headlines", [])
    if scored_tot > 0:
        pct_bull = round(market_ctx.get("bullish_count", 0) / scored_tot * 100)
        pct_bear = round(market_ctx.get("bearish_count", 0) / scored_tot * 100)
        sentiment_summary = f"{pct_bull}% Bullish, {pct_bear}% Bearish, {100 - pct_bull - pct_bear}% Neutral (based on {scored_tot} scored articles)"
    else:
        sentiment_summary = f"{total_arts} total articles in trailing 24h"

    macro_lines = [f"  - {h}" for h in macro_headlines] if macro_headlines else ["  - No major index headlines"]
    macro_block = "\n".join(macro_lines)

    prompt = f"""[TARGET ARTICLE FOR EVALUATION]
Ticker Symbol: {primary_ticker or "Unknown / Multiple"}
Published: {ts_str}
Headline: {headline}
Body: {body[:1500] if body else "(No additional body text provided)"}

[PRIOR NEWS HISTORY FOR {primary_ticker or 'TICKER'} — Trailing {days} Days, Chronological]
{history_block}

[OVERALL MARKET NEWS FLOW — Trailing 24 Hours]
• Volume & Sentiment: {sentiment_summary}
• Key Macro / Market Headlines:
{macro_block}

Evaluate whether the TARGET ARTICLE represents a fresh, actionable catalyst for long calls. Respond with JSON ONLY."""

    return prompt


def _clean_json_response(raw: str) -> dict:
    """Extract and parse JSON object from LLM response text."""
    text = (raw or "").strip()
    if text.startswith("```"):
        # Strip markdown fences
        parts = text.split("```")
        if len(parts) >= 2:
            inner = parts[1]
            if inner.startswith("json"):
                inner = inner[4:]
            text = inner.strip()

    # Find first { and last }
    start = text.find("{")
    end = text.rfind("}")
    if start != -1 and end != -1 and end > start:
        text = text[start:end+1]

    data = json.loads(text)
    # Validate required fields
    if not isinstance(data.get("tickers"), list):
        data["tickers"] = [str(data["tickers"])] if data.get("tickers") else []
    data["sentiment"] = str(data.get("sentiment", "neutral")).lower()
    data["confidence"] = float(data.get("confidence", 0.0))
    data["magnitude"] = float(data.get("magnitude", 0.0))
    data["reasoning"] = str(data.get("reasoning", ""))
    data["is_stale_echo"] = bool(data.get("is_stale_echo", False))
    return data


def _score_with_ollama(prompt: str, model: str, base_url: str) -> Optional[dict]:
    client = OpenAI(base_url=base_url, api_key="ollama")
    resp = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
        ],
        temperature=0.1,
        max_tokens=600,
        timeout=cfg.SCORER_TIMEOUT_S,
    )
    raw = resp.choices[0].message.content or ""
    return _clean_json_response(raw)


_last_gemini_call_time = 0.0
GEMINI_USAGE_FILE = cfg.DATA_DIR / "gemini_usage.json"


def _check_gemini_daily_limit() -> bool:
    """Ensure Gemini daily call count stays within safety cap."""
    today_str = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    data = {"date": today_str, "count": 0}
    try:
        if os.path.exists(GEMINI_USAGE_FILE):
            with open(GEMINI_USAGE_FILE) as f:
                data = json.load(f)
                if data.get("date") != today_str:
                    data = {"date": today_str, "count": 0}
    except Exception:
        data = {"date": today_str, "count": 0}

    if data["count"] >= cfg.GEMINI_MAX_DAILY_CALLS:
        log.warning("🛑 Gemini daily safety cap reached (%d/%d calls today). Halting Gemini calls to protect quota.",
                    data["count"], cfg.GEMINI_MAX_DAILY_CALLS)
        return False
    return True


def _record_gemini_call(exhausted: bool = False) -> None:
    """Increment Gemini daily count or mark exhausted."""
    today_str = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    data = {"date": today_str, "count": 0}
    try:
        if os.path.exists(GEMINI_USAGE_FILE):
            with open(GEMINI_USAGE_FILE) as f:
                data = json.load(f)
                if data.get("date") != today_str:
                    data = {"date": today_str, "count": 0}
    except Exception:
        pass

    if exhausted:
        data["count"] = max(data["count"], cfg.GEMINI_MAX_DAILY_CALLS)
    else:
        data["count"] += 1

    try:
        with open(GEMINI_USAGE_FILE, "w") as f:
            json.dump(data, f)
    except Exception as e:
        log.debug("Failed to write gemini usage: %s", e)


def _score_with_gemini(prompt: str, model: str, api_key: str) -> Optional[dict]:
    """Call Google Gemini REST API with strict pacing and daily quota protection."""
    global _last_gemini_call_time

    if not api_key:
        log.warning("GEMINI_API_KEY is not set.")
        return None

    if not _check_gemini_daily_limit():
        return None

    # Strict pacing: enforce delay between consecutive calls to stay well below 15 RPM
    now = time.monotonic()
    elapsed = now - _last_gemini_call_time
    if elapsed < cfg.GEMINI_PACING_SECS:
        sleep_needed = cfg.GEMINI_PACING_SECS - elapsed
        time.sleep(sleep_needed)
    _last_gemini_call_time = time.monotonic()

    # Clean model identifier (strip 'models/' prefix if present)
    clean_model = model[7:] if model.startswith("models/") else model
    url = f"https://generativelanguage.googleapis.com/v1beta/models/{clean_model}:generateContent?key={api_key}"
    payload = {
        "contents": [
            {"role": "user", "parts": [{"text": f"{SYSTEM_PROMPT}\n\n{prompt}"}]}
        ],
        "generationConfig": {
            "temperature": 0.1,
            "maxOutputTokens": 2048,
            "responseMimeType": "application/json",
            "thinkingConfig": {"thinkingBudget": 0},
        }
    }

    for attempt in range(1, 4):
        try:
            resp = requests.post(url, json=payload, timeout=cfg.SCORER_TIMEOUT_S)
            if resp.status_code == 200:
                data = resp.json()
                candidates = data.get("candidates", [])
                if not candidates:
                    return None
                raw = candidates[0].get("content", {}).get("parts", [{}])[0].get("text", "")
                _record_gemini_call(exhausted=False)
                return _clean_json_response(raw)

            if resp.status_code == 429:
                err_text = resp.text
                if "free_tier_requests" in err_text or "GenerateRequestsPerDay" in err_text or "daily" in err_text.lower():
                    log.warning("🛑 Gemini daily free-tier quota exhausted by API. Halting calls for today.")
                    _record_gemini_call(exhausted=True)
                    return None

                log.warning("Gemini 429 rate limit hit on attempt %d. Backing off 15s...", attempt)
                time.sleep(15.0)
                continue

            log.warning("Gemini API error %d: %s", resp.status_code, resp.text[:200])
            return None
        except Exception as e:
            log.warning("Gemini request attempt %d failed: %s", attempt, e)
            time.sleep(2.0 * attempt)

    return None


def _score_with_anthropic(prompt: str, model: str, api_key: str) -> Optional[dict]:
    import anthropic
    client = anthropic.Anthropic(api_key=api_key or None, timeout=cfg.SCORER_TIMEOUT_S)
    resp = client.messages.create(
        model=model,
        max_tokens=600,
        system=SYSTEM_PROMPT,
        messages=[{"role": "user", "content": prompt}],
    )
    raw = resp.content[0].text if resp.content else ""
    return _clean_json_response(raw)


def _score_with_moonshot(prompt: str, model: str, api_key: str, base_url: str) -> Optional[dict]:
    client = OpenAI(base_url=base_url, api_key=api_key, timeout=cfg.SCORER_TIMEOUT_S)
    resp = client.chat.completions.create(
        model=model,
        max_tokens=600,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
        ],
        temperature=0.1,
    )
    raw = resp.choices[0].message.content or ""
    return _clean_json_response(raw)


def score_article(headline: str, body: str, primary_ticker: str = "",
                  article_ts: Optional[datetime] = None,
                  lookback_days: Optional[int] = None,
                  provider: Optional[str] = None,
                  model: Optional[str] = None,
                  db_path: Optional[str] = None) -> Optional[dict]:
    """
    Score a news article with context enrichment.
    Dispatches to the configured provider (ollama, gemini, anthropic, moonshot).
    """
    prov = (provider or cfg.SCORER_PROVIDER).lower()
    mod = model or cfg.SCORER_MODEL

    prompt = build_context_prompt(
        headline=headline,
        body=body,
        primary_ticker=primary_ticker,
        article_ts=article_ts,
        lookback_days=lookback_days,
        db_path=db_path,
    )

    t0 = time.monotonic()
    result = None

    for attempt in range(1, cfg.SCORER_MAX_RETRIES + 1):
        try:
            if prov == "ollama":
                result = _score_with_ollama(prompt, mod, cfg.OLLAMA_BASE_URL)
            elif prov == "gemini":
                result = _score_with_gemini(prompt, mod, cfg.GEMINI_API_KEY)
                if result is None:
                    return None
            elif prov == "anthropic":
                result = _score_with_anthropic(prompt, mod, cfg.ANTHROPIC_API_KEY)
            elif prov in ("moonshot", "kimi"):
                result = _score_with_moonshot(prompt, mod, cfg.MOONSHOT_API_KEY, cfg.MOONSHOT_BASE_URL)
            else:
                log.error("Unknown scorer provider: %s", prov)
                return None

            if result is not None:
                elapsed_ms = (time.monotonic() - t0) * 1000
                log.info("Scored [%s/%s] in %.0fms: sentiment=%s mag=%.2f conf=%.2f echo=%s",
                         prov, mod, elapsed_ms, result.get("sentiment"),
                         result.get("magnitude", 0), result.get("confidence", 0),
                         result.get("is_stale_echo", False))
                return result
        except Exception as e:
            log.warning("Scorer call attempt %d failed (%s/%s): %s", attempt, prov, mod, e)
            if attempt < cfg.SCORER_MAX_RETRIES:
                time.sleep(1.0 * attempt)

    return None


def correct_ticker(headline: str, body: str, symbols: list[str],
                   model_ticker: str = "") -> Optional[str]:
    """
    Local Ollama ticker-selection corrector.
    Selects the single primary beneficiary from news feed symbols metadata.
    """
    if not cfg.TICKER_CORRECTOR_ENABLED or not symbols:
        return model_ticker or (symbols[0] if symbols else None)

    if len(symbols) == 1:
        return symbols[0]

    prompt = f"""Given this news article and list of candidate ticker symbols, return ONLY the ticker of the primary positive beneficiary.
Article Headline: {headline}
Summary: {body[:500]}
Candidate Symbols: {', '.join(symbols)}

Return ONLY the single ticker symbol, or NONE if no single company is the clear beneficiary."""

    try:
        client = OpenAI(base_url=cfg.OLLAMA_BASE_URL, api_key="ollama")
        resp = client.chat.completions.create(
            model=cfg.TICKER_CORRECTOR_MODEL,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=20,
            temperature=0.0,
            timeout=10.0,
        )
        text = (resp.choices[0].message.content or "").strip().upper()
        # Clean up response
        for sym in symbols:
            if sym.upper() in text:
                return sym.upper()
    except Exception as e:
        log.debug("Ticker corrector fallback: %s", e)

    return model_ticker or symbols[0]

