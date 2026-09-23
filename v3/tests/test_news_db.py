"""
Tests for v3/news_db.py.
Verifies table creation, article & symbol insertion, deduplication,
point-in-time ticker history retrieval, and market sentiment aggregation.
"""

import tempfile
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest
from v3 import news_db


@pytest.fixture
def temp_db():
    with tempfile.NamedTemporaryFile(suffix=".db") as f:
        db_path = Path(f.name)
        news_db.init_db(db_path)
        yield db_path


def test_init_db(temp_db):
    # Verify tables created
    with news_db._get_connection(temp_db) as conn:
        tables = conn.execute("SELECT name FROM sqlite_master WHERE type='table';").fetchall()
        table_names = {r["name"] for r in tables}
        assert "articles" in table_names
        assert "article_symbols" in table_names
        assert "article_scores" in table_names


def test_store_and_dedup_article(temp_db):
    item = {
        "id": "12345",
        "headline": "Apple Reports Record iPhone Sales in Q4",
        "summary": "Revenue up 12% driven by services and strong hardware demand.",
        "created_at": "2026-09-20T14:30:00Z",
        "symbols": ["AAPL"],
        "source": "Benzinga",
    }

    # First insert -> True
    assert news_db.store_article(item, db_path=temp_db) is True

    # Duplicate insert -> False
    assert news_db.store_article(item, db_path=temp_db) is False

    # Check database content
    with news_db._get_connection(temp_db) as conn:
        row = conn.execute("SELECT * FROM articles WHERE article_id = '12345'").fetchone()
        assert row is not None
        assert row["headline"] == "Apple Reports Record iPhone Sales in Q4"

        sym_rows = conn.execute("SELECT * FROM article_symbols WHERE article_id = '12345'").fetchall()
        assert len(sym_rows) == 1
        assert sym_rows[0]["symbol"] == "AAPL"


def test_point_in_time_ticker_history(temp_db):
    base_time = datetime(2026, 9, 20, 12, 0, 0, tzinfo=timezone.utc)

    # Insert 3 articles for NVDA across past days
    articles = [
        {"id": "nvda_1", "headline": "NVDA 4 days ago", "created_at": (base_time - timedelta(days=4)).isoformat(), "symbols": ["NVDA"]},
        {"id": "nvda_2", "headline": "NVDA 2 days ago", "created_at": (base_time - timedelta(days=2)).isoformat(), "symbols": ["NVDA"]},
        {"id": "nvda_3", "headline": "NVDA 1 hour ago", "created_at": (base_time - timedelta(hours=1)).isoformat(), "symbols": ["NVDA"]},
        # Future article relative to base_time (must NOT be returned)
        {"id": "nvda_future", "headline": "NVDA future article", "created_at": (base_time + timedelta(hours=1)).isoformat(), "symbols": ["NVDA"]},
        # Different ticker
        {"id": "msft_1", "headline": "MSFT article", "created_at": (base_time - timedelta(days=1)).isoformat(), "symbols": ["MSFT"]},
    ]

    for a in articles:
        news_db.store_article(a, db_path=temp_db)

    # Query history as of base_time with 7-day lookback
    history = news_db.get_ticker_news_history("NVDA", as_of_ts=base_time, days=7, db_path=temp_db)

    assert len(history) == 3
    # Check chronological ordering: oldest to newest
    assert history[0]["article_id"] == "nvda_1"
    assert history[1]["article_id"] == "nvda_2"
    assert history[2]["article_id"] == "nvda_3"

    # Verify lookahead bias prevention: future article must NOT be present
    article_ids = [h["article_id"] for h in history]
    assert "nvda_future" not in article_ids
    assert "msft_1" not in article_ids

    # Query with 3-day lookback -> should only return nvda_2 and nvda_3
    history_3d = news_db.get_ticker_news_history("NVDA", as_of_ts=base_time, days=3, db_path=temp_db)
    assert len(history_3d) == 2
    assert [h["article_id"] for h in history_3d] == ["nvda_2", "nvda_3"]


def test_market_sentiment_context(temp_db):
    base_time = datetime(2026, 9, 20, 12, 0, 0, tzinfo=timezone.utc)

    # Insert macro and individual articles
    news_db.store_article({"id": "spy_1", "headline": "S&P 500 Dips on Rate Worries", "created_at": (base_time - timedelta(hours=2)).isoformat(), "symbols": ["SPY"]}, db_path=temp_db)
    news_db.store_article({"id": "tsla_1", "headline": "Tesla Delivers Strong Results", "created_at": (base_time - timedelta(hours=5)).isoformat(), "symbols": ["TSLA"]}, db_path=temp_db)

    # Record a score
    news_db.record_score("tsla_1", model="test_model", sentiment="bullish", confidence=0.85, magnitude=0.70, reasoning="Strong deliveries", db_path=temp_db)

    context = news_db.get_market_sentiment_context(as_of_ts=base_time, hours=24, db_path=temp_db)

    assert context["total_articles"] == 2
    assert context["scored_total"] == 1
    assert context["bullish_count"] == 1
    assert len(context["macro_headlines"]) == 1
    assert "S&P 500 Dips on Rate Worries" in context["macro_headlines"][0]

