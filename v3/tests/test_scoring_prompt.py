"""
Unit tests for v3/scoring.py.
Verifies prompt construction with empty and populated history,
echo/recap rules in system prompt, and JSON response parsing.
"""

import tempfile
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest
from v3 import news_db, scoring


@pytest.fixture
def temp_db():
    with tempfile.NamedTemporaryFile(suffix=".db") as f:
        db_path = Path(f.name)
        news_db.init_db(db_path)
        yield str(db_path)


def test_build_context_prompt_empty_history(temp_db):
    headline = "Apple Unveils New Product"
    body = "Apple today announced its next generation lineup."
    ticker = "AAPL"
    ts = datetime(2026, 9, 20, 14, 0, 0, tzinfo=timezone.utc)

    prompt = scoring.build_context_prompt(
        headline=headline, body=body, primary_ticker=ticker,
        article_ts=ts, lookback_days=7, db_path=temp_db
    )

    assert "[TARGET ARTICLE FOR EVALUATION]" in prompt
    assert "Headline: Apple Unveils New Product" in prompt
    assert "Ticker Symbol: AAPL" in prompt
    assert "[PRIOR NEWS HISTORY FOR AAPL — Trailing 7 Days, Chronological]" in prompt
    assert "No other articles recorded for AAPL" in prompt
    assert "[OVERALL MARKET NEWS FLOW — Trailing 24 Hours]" in prompt


def test_build_context_prompt_with_history(temp_db):
    ts = datetime(2026, 9, 20, 14, 0, 0, tzinfo=timezone.utc)

    # Insert prior articles for TSLA
    news_db.store_article({
        "id": "tsla_prior_1",
        "headline": "Tesla Q3 Deliveries Beat by 10%",
        "summary": "Record quarterly vehicle production and deliveries.",
        "created_at": (ts - timedelta(days=2)).isoformat(),
        "symbols": ["TSLA"]
    }, db_path=temp_db)
    news_db.record_score("tsla_prior_1", model="llama3.2", sentiment="bullish", confidence=0.85, magnitude=0.75, reasoning="Major beat", db_path=temp_db)

    # Target article is a follow-up 2 days later
    current_headline = "Why Tesla Shares Are Moving Today"
    current_body = "Tesla shares continue their post-earnings rally after analyst upgrades."

    prompt = scoring.build_context_prompt(
        headline=current_headline, body=current_body, primary_ticker="TSLA",
        article_ts=ts, lookback_days=5, db_path=temp_db
    )

    assert "Headline: Why Tesla Shares Are Moving Today" in prompt
    assert "Tesla Q3 Deliveries Beat by 10%" in prompt
    assert "Prior Scored: bullish, mag=0.75" in prompt
    assert "Trailing 5 Days" in prompt


def test_clean_json_response_direct():
    raw = '{"tickers": ["AAPL"], "sentiment": "bullish", "confidence": 0.88, "magnitude": 0.72, "reasoning": "Solid beat", "is_stale_echo": false}'
    res = scoring._clean_json_response(raw)
    assert res["tickers"] == ["AAPL"]
    assert res["sentiment"] == "bullish"
    assert res["confidence"] == 0.88
    assert res["magnitude"] == 0.72
    assert res["is_stale_echo"] is False


def test_clean_json_response_markdown_fence():
    raw = """Here is the score:
```json
{
  "tickers": ["NVDA"],
  "sentiment": "neutral",
  "confidence": 0.65,
  "magnitude": 0.20,
  "reasoning": "Recap of last week's keynote speech",
  "is_stale_echo": true
}
```
Hope this helps!"""
    res = scoring._clean_json_response(raw)
    assert res["tickers"] == ["NVDA"]
    assert res["sentiment"] == "neutral"
    assert res["is_stale_echo"] is True
    assert res["magnitude"] == 0.20

