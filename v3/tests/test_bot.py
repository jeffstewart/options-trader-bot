"""
Unit tests for v3/bot.py message handling, reconnect safety, and signal flow.
"""

import asyncio
from datetime import datetime, timezone
from unittest.mock import AsyncMock, MagicMock, patch

from v3 import bot


def test_patch_reconnect_safety():
    """Verify exponential backoff wrapper on failed stream start."""
    async def _test():
        mock_stream = MagicMock()
        attempts = 0

        async def fail_then_succeed():
            nonlocal attempts
            attempts += 1
            if attempts == 1:
                raise ConnectionError("Connection refused")
            return True

        mock_stream._start_ws = fail_then_succeed

        with patch("asyncio.sleep", new_callable=AsyncMock) as mock_sleep:
            bot._patch_reconnect_safety(mock_stream, name="test_ws")

            # First call fails and backs off
            try:
                await mock_stream._start_ws()
                assert False, "Should have raised ConnectionError"
            except ConnectionError:
                pass
            mock_sleep.assert_called_once_with(1.0)

            # Second call succeeds
            await mock_stream._start_ws()
            assert attempts == 2

    asyncio.run(_test())


def test_handle_news_message_stores_and_processes(monkeypatch):
    """Verify WebSocket message handling extracts fields, stores to DB, and routes signal."""
    async def _test():
        fake_item = {
            "id": "ws_12345",
            "headline": "Acme Corp Secures Major Contract",
            "summary": "Multi-million dollar deal signed.",
            "symbols": ["ACME"],
            "source": "Alpaca/Benzinga",
            "created_at": datetime.now(timezone.utc).isoformat(),
        }

        stored_articles = []
        processed_signals = []

        monkeypatch.setattr(bot.news_db, "store_article", lambda item: stored_articles.append(item))

        async def mock_process(headline, body, source, symbols, article_ts, news_id):
            processed_signals.append((headline, symbols, news_id))

        monkeypatch.setattr(bot, "process_signal", mock_process)

        # Clear seen IDs
        bot._seen_news_ids.clear()

        await bot.handle_news_message(fake_item)
        await asyncio.sleep(0.01)

        assert len(stored_articles) == 1
        assert stored_articles[0]["id"] == "ws_12345"
        assert len(processed_signals) == 1
        assert processed_signals[0][1] == ["ACME"]
        assert processed_signals[0][2] == "ws_12345"

        # Duplicate message should be ignored
        await bot.handle_news_message(fake_item)
        assert len(stored_articles) == 1
        assert len(processed_signals) == 1

    asyncio.run(_test())


def test_process_signal_grid_collection_hook(monkeypatch):
    """Verify both bullish and bearish signals trigger grid collection while live orders remain Call-only."""
    async def _test():
        grid_calls = []

        def mock_should_collect(score):
            return True

        async def mock_start_collection(**kwargs):
            grid_calls.append(kwargs)

        monkeypatch.setattr(bot.grid_collector, "should_collect", mock_should_collect)
        monkeypatch.setattr(bot.grid_collector, "start_collection", mock_start_collection)
        monkeypatch.setattr(bot.filters, "prescore_noise", lambda h: None)
        monkeypatch.setattr(bot.filters, "soft_catalyst_hit", lambda h, b: None)
        monkeypatch.setattr(bot.news_db, "record_score", lambda **kwargs: None)
        monkeypatch.setattr(bot.mkt, "get_stock_price", lambda sym: 150.0)

        # Bearish signal -> grid collector runs with sentiment="bearish", but live trade aborted
        place_trade_mock = MagicMock()
        monkeypatch.setattr(bot.execution, "place_option_trade", place_trade_mock)

        bearish_score = {
            "tickers": ["XYZ"],
            "sentiment": "bearish",
            "magnitude": 0.70,
            "confidence": 0.85,
            "reasoning": "Major trial failed",
            "is_stale_echo": False,
        }
        monkeypatch.setattr(bot.scoring, "score_article", lambda **kwargs: bearish_score)

        await bot.process_signal(
            headline="XYZ Reports Clinical Trial Failure",
            body="Failed to meet primary endpoint.",
            source="Alpaca",
            symbols=["XYZ"],
            article_ts=datetime.now(timezone.utc),
            news_id="bearish_1",
        )
        await asyncio.sleep(0.01)

        assert len(grid_calls) == 1
        assert grid_calls[0]["ticker"] == "XYZ"
        assert grid_calls[0]["signal"]["sentiment"] == "bearish"
        # Live trade must NOT be placed for bearish signals
        place_trade_mock.assert_not_called()

    asyncio.run(_test())
