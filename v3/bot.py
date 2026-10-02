"""
bot.py (v3) — event-driven options trading bot with context-aware news archive.
Orchestrates:
  1. Boot: SQLite news DB init + 7-day startup backfill from Alpaca REST.
  2. News ingest: every article stored into SQLite news archive.
  3. Pre-score deterministic filters: eliminate noise, listicles, analyst notes, and soft catalysts.
  4. Regime check: SPY 200d SMA, 3d momentum, and chop brake.
  5. Context-aware LLM scoring: prompt enriched with trailing ticker news history and market sentiment.
  6. Empirical execution: Delta ~0.50 ATM calls, hard <=6.5% spread cap, >=10 OI floor.
  7. Position monitor: timed 30-minute horizon exits.
"""

import asyncio
import logging
import signal
import sys
import time
from datetime import datetime, timedelta, timezone
from typing import Optional

import aiohttp
from alpaca.data.live import NewsDataStream

try:
    from v3 import config as cfg
    from v3 import execution
    from v3 import filters
    from v3 import grid_collector
    from v3 import market as mkt
    from v3 import news_db
    from v3 import scoring
except ImportError:
    import config as cfg
    import execution
    import filters
    import grid_collector
    import market as mkt
    import news_db
    import scoring

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("v3")

_seen_news_ids: set[str] = set()
_recent_trades: dict[str, float] = {}  # ticker -> epoch
_running = True


async def process_signal(headline: str, body: str, source: str,
                         symbols: list[str], article_ts: Optional[datetime],
                         news_id: str) -> None:
    """Main signal processing pipeline."""
    now = datetime.now(timezone.utc)
    ts = article_ts or now

    # 1. Stale news check (feed lag)
    feed_lag = (now - ts).total_seconds()
    if feed_lag > cfg.MAX_FEED_LAG_SECS:
        log.debug("Dropping stale news (lag %.0fs > %ds): %s", feed_lag, cfg.MAX_FEED_LAG_SECS, headline[:80])
        return

    # 2. Deterministic pre-score noise filter
    if cfg.PRESCORE_FILTER_ENABLED:
        noise = filters.prescore_noise(headline)
        if noise:
            log.info("🚫 Noise filter matched [%s] — skipping: %s", noise, headline[:80])
            return

    # 3. Deterministic soft catalyst filter (analyst ratings, chart patterns, acquirer deal talk)
    if cfg.SOFT_CATALYST_GATE_ENABLED:
        soft = filters.soft_catalyst_hit(headline, body)
        if soft:
            log.info("🚫 Soft catalyst matched [%s] — skipping: %s", soft, headline[:80])
            return

    # 4. Candidate ticker resolution
    candidate_ticker = symbols[0] if symbols else ""

    # 5. Context-Aware LLM Scoring
    score = scoring.score_article(
        headline=headline,
        body=body,
        primary_ticker=candidate_ticker,
        article_ts=ts,
        lookback_days=cfg.NEWS_LOOKBACK_DAYS,
    )

    if not score:
        return

    # Record score into SQLite news archive
    if news_id:
        news_db.record_score(
            article_id=news_id,
            model=cfg.SCORER_MODEL,
            sentiment=score.get("sentiment", "neutral"),
            confidence=score.get("confidence", 0.0),
            magnitude=score.get("magnitude", 0.0),
            reasoning=score.get("reasoning", ""),
        )

    # 6. Ticker resolution and correction
    model_tickers = score.get("tickers", [])
    primary_ticker = candidate_ticker or (model_tickers[0] if model_tickers else "")
    if len(symbols) > 1:
        corrected = scoring.correct_ticker(headline, body, symbols, primary_ticker)
        if corrected:
            primary_ticker = corrected

    if not primary_ticker:
        log.info("No definitive ticker identified — skipping signal")
        return

    # 7. Contract Grid Research Telemetry Hook
    # Collects quotes & Greeks for both Bullish (Calls) and Bearish (Puts) without placing orders
    if grid_collector.should_collect(score):
        signal_id = grid_collector.make_signal_id(primary_ticker, ts)
        asyncio.create_task(
            grid_collector.start_collection(
                signal_id=signal_id,
                signal_ts=ts,
                ticker=primary_ticker,
                signal=score,
                headline=headline,
                source=source,
                get_stock_price=mkt.get_stock_price,
            )
        )

    # 8. Live Trading Execution Gates (Call-only for now; Put trading gated pending research telemetry)
    sentiment = (score.get("sentiment") or "neutral").lower()
    confidence = float(score.get("confidence", 0.0))
    magnitude = float(score.get("magnitude", 0.0))
    is_echo = bool(score.get("is_stale_echo", False))

    if sentiment != "bullish":
        log.debug("Non-bullish sentiment (%s) — skipping live trade", sentiment)
        return

    if is_echo:
        log.info("🔁 Stale echo/recap detected by LLM context — skipping %s: %s",
                 primary_ticker, headline[:80])
        return

    if confidence < cfg.MIN_CONFIDENCE or magnitude < cfg.MIN_MAGNITUDE:
        log.info("Signal below threshold (mag=%.2f/%.2f, conf=%.2f/%.2f) — skipping %s",
                 magnitude, cfg.MIN_MAGNITUDE, confidence, cfg.MIN_CONFIDENCE, primary_ticker)
        return

    # Cooldown check
    last_traded = _recent_trades.get(primary_ticker, 0.0)
    if time.time() - last_traded < cfg.COOLDOWN_SECS:
        log.debug("Cooldown active for %s (%.0fs remaining) — skipping live trade",
                  primary_ticker, cfg.COOLDOWN_SECS - (time.time() - last_traded))
        return

    # Market regime check (SPY 200d SMA, 3-day momentum, chop brake)
    regime = mkt.evaluate_regime()
    if not regime.get("passes_regime", True):
        log.debug("Regime pause active (uptrend=%s mom=%s chop=%s) — skipping live trade",
                  regime.get("uptrend_ok"), regime.get("mom_ok"), regime.get("chop_ok"))
        return

    # Circuit breaker check
    if mkt.check_daily_loss_breaker():
        log.warning("Daily loss breaker active — skipping live trade for %s", primary_ticker)
        return

    # Stock price check
    stock_price = mkt.get_stock_price(primary_ticker)
    if not stock_price or stock_price < cfg.MIN_STOCK_PRICE:
        log.info("Stock price missing or below minimum ($%.2f < $%.2f) for %s — skipping",
                 stock_price or 0.0, cfg.MIN_STOCK_PRICE, primary_ticker)
        return

    # 9. Position Sizing
    equity = mkt.get_account_equity()
    position_usd = round(equity * cfg.POSITION_FRAC_OF_EQUITY, 2)

    log.info("⚡ QUALIFIED SIGNAL: %s (mag=%.2f conf=%.2f) price=$%.2f budget=$%.2f",
             primary_ticker, magnitude, confidence, stock_price, position_usd)

    # 10. Contract Selection & Order Execution
    signal_meta = {
        "id": news_id,
        "ticker": primary_ticker,
        "magnitude": magnitude,
        "confidence": confidence,
        "headline": headline,
    }

    fill = execution.place_option_trade(
        ticker=primary_ticker,
        stock_price=stock_price,
        signal=signal_meta,
        position_usd=position_usd,
    )

    if fill:
        _recent_trades[primary_ticker] = time.time()
        log.info("🎉 Trade opened for %s on %s", primary_ticker, fill["symbol"])


async def news_poller():
    """Poll Alpaca News REST API and feed incoming items to pipeline and DB."""
    log.info("Starting Alpaca News poller (every %ds) ...", cfg.NEWS_POLL_SECS)
    url = "https://data.alpaca.markets/v1beta1/news"
    headers = {"APCA-API-KEY-ID": cfg.ALPACA_KEY, "APCA-API-SECRET-KEY": cfg.ALPACA_SECRET}
    since = datetime.now(timezone.utc) - timedelta(seconds=cfg.NEWS_POLL_SECS * 2)

    async with aiohttp.ClientSession() as session:
        while _running:
            try:
                params = {
                    "start": since.strftime("%Y-%m-%dT%H:%M:%SZ"),
                    "sort": "asc",
                    "limit": "50",
                    "include_content": "false",
                }
                async with session.get(url, headers=headers, params=params,
                                       timeout=aiohttp.ClientTimeout(total=10)) as resp:
                    if resp.status == 200:
                        data = await resp.json()
                        items = data.get("news", [])
                        for item in items:
                            news_id = str(item.get("id"))
                            if news_id in _seen_news_ids:
                                continue
                            _seen_news_ids.add(news_id)

                            # Store article immediately in SQLite archive
                            news_db.store_article(item)

                            headline = item.get("headline", "") or ""
                            body = item.get("summary", "") or ""
                            symbols = item.get("symbols") or []
                            raw_ts = item.get("created_at") or item.get("updated_at")
                            article_ts = None
                            if raw_ts:
                                try:
                                    article_ts = datetime.fromisoformat(raw_ts.replace("Z", "+00:00"))
                                except Exception:
                                    pass

                            log.info("📰 [Alpaca] %s (%s)", headline[:90], ", ".join(symbols[:3]) if symbols else "Macro")
                            asyncio.create_task(
                                process_signal(headline, body, "Alpaca/Benzinga", symbols, article_ts, news_id)
                            )

                        if items:
                            last_ts = items[-1].get("created_at") or items[-1].get("updated_at")
                            if last_ts:
                                try:
                                    since = datetime.fromisoformat(last_ts.replace("Z", "+00:00"))
                                except Exception:
                                    pass
                    else:
                        log.warning("News poll error %d: %s", resp.status, await resp.text())
            except Exception as e:
                log.warning("News poll exception: %s", e)

            await asyncio.sleep(cfg.NEWS_POLL_SECS)


def _patch_reconnect_safety(stream, name: str = "news"):
    """
    Wrap stream._start_ws with exponential backoff on reconnection
    to prevent tight reconnect loops during Alpaca websocket outages.
    """
    orig_start = stream._start_ws
    backoff = 1.0

    async def _safe_start_ws():
        nonlocal backoff
        try:
            await orig_start()
            backoff = 1.0
        except Exception as e:
            sleep_time = min(backoff, 30.0)
            log.warning("WebSocket [%s] connection failed: %s. Backing off %.1fs...", name, e, sleep_time)
            await asyncio.sleep(sleep_time)
            backoff = min(backoff * 2, 30.0)
            raise

    stream._start_ws = _safe_start_ws


async def handle_news_message(news_item):
    """Callback for Alpaca NewsDataStream WebSocket messages."""
    try:
        if hasattr(news_item, "model_dump"):
            item = news_item.model_dump()
        elif hasattr(news_item, "dict"):
            item = news_item.dict()
        elif isinstance(news_item, dict):
            item = news_item
        else:
            item = {
                "id": getattr(news_item, "id", None),
                "headline": getattr(news_item, "headline", ""),
                "summary": getattr(news_item, "summary", ""),
                "symbols": getattr(news_item, "symbols", []),
                "source": getattr(news_item, "source", "Alpaca/Benzinga"),
                "url": getattr(news_item, "url", ""),
                "created_at": getattr(news_item, "created_at", None),
                "updated_at": getattr(news_item, "updated_at", None),
            }

        news_id = str(item.get("id") or "")
        if news_id and news_id in _seen_news_ids:
            return
        if news_id:
            _seen_news_ids.add(news_id)

        # Store article immediately in SQLite archive
        news_db.store_article(item)

        headline = (item.get("headline") or "").strip()
        body = (item.get("summary") or "").strip()
        symbols = item.get("symbols") or []
        raw_ts = item.get("created_at") or item.get("updated_at")
        article_ts = None
        if isinstance(raw_ts, datetime):
            article_ts = raw_ts.astimezone(timezone.utc)
        elif isinstance(raw_ts, str):
            try:
                article_ts = datetime.fromisoformat(raw_ts.replace("Z", "+00:00"))
            except Exception:
                pass

        log.info("⚡ [WebSocket News] %s (%s)", headline[:90], ", ".join(symbols[:3]) if symbols else "Macro")
        asyncio.create_task(
            process_signal(headline, body, item.get("source") or "Alpaca/Benzinga", symbols, article_ts, news_id)
        )
    except Exception as e:
        log.warning("Error processing websocket news item: %s", e)


async def news_stream_listener():
    """Stream real-time news via Alpaca NewsDataStream."""
    log.info("Starting Alpaca News WebSocket stream (NewsDataStream)...")
    while _running:
        try:
            stream = NewsDataStream(cfg.ALPACA_KEY, cfg.ALPACA_SECRET)
            _patch_reconnect_safety(stream, "news")
            stream.subscribe_news(handle_news_message, "*")
            await stream._run_forever()
        except asyncio.CancelledError:
            break
        except Exception as e:
            log.warning("WebSocket stream exception: %s — restarting in 5s...", e)
            await asyncio.sleep(5)


async def sec_rss_poller():
    """Poll SEC EDGAR Atom feed for material 8-K filings and route to pipeline."""
    if not getattr(cfg, "SEC_FEED_ENABLED", False):
        return

    import feedparser
    from v3 import sec_edgar

    log.info("📡 SEC EDGAR 8-K RSS poller started (every %ds)", cfg.SEC_POLL_SECS)
    async with aiohttp.ClientSession() as session:
        while _running:
            try:
                async with session.get(cfg.SEC_RSS_URL, headers={"User-Agent": cfg.SEC_USER_AGENT},
                                       timeout=aiohttp.ClientTimeout(total=10)) as r:
                    if r.status == 200:
                        raw = await r.text()
                        feed = feedparser.parse(raw)
                        cik_map = await sec_edgar.load_cik_ticker_map(session, cfg.SEC_USER_AGENT)

                        for entry in feed.entries:
                            eid = entry.get("id", entry.get("link", ""))
                            if not eid or eid in _seen_news_ids:
                                continue
                            _seen_news_ids.add(eid)

                            title = entry.get("title", "")
                            summary = entry.get("summary", "")
                            link = entry.get("link", "")

                            cik, accession = sec_edgar.extract_cik_accession(link)
                            ticker = cik_map.get(cik) if cik else None
                            if not ticker:
                                continue

                            headline = f"{title} [{ticker}]"
                            body = summary
                            if cik and accession and (sec_edgar.item_codes_in_summary(summary) & sec_edgar.MATERIAL_ITEMS):
                                try:
                                    full_text = await sec_edgar.fetch_8k_content(session, cik, accession, cfg.SEC_USER_AGENT)
                                    if full_text:
                                        body = full_text
                                except Exception as e:
                                    log.debug("SEC 8-K content fetch failed for %s: %s", eid, e)

                            # Store article in SQLite archive
                            news_item = {
                                "id": eid,
                                "headline": headline,
                                "summary": body,
                                "symbols": [ticker],
                                "source": "SEC",
                                "url": link,
                                "created_at": datetime.now(timezone.utc).isoformat(),
                            }
                            news_db.store_article(news_item)

                            log.info("📰 [SEC EDGAR 8-K] %s (%s)", headline[:80], ticker)
                            asyncio.create_task(
                                process_signal(headline, body, "SEC", [ticker], datetime.now(timezone.utc), eid)
                            )
            except Exception as e:
                log.warning("SEC RSS poll exception: %s", e)

            await asyncio.sleep(cfg.SEC_POLL_SECS)


async def position_monitor_loop():
    """Periodically evaluate open positions for timed horizon exits."""
    log.info("Starting position monitor (horizon exit = %d min) ...", cfg.EXIT_HORIZON_MINUTES)
    while _running:
        try:
            if mkt.is_market_open():
                execution.monitor_open_positions()
        except Exception as e:
            log.warning("Position monitor exception: %s", e)
        await asyncio.sleep(15)


def handle_shutdown(signum, frame):
    global _running
    log.info("Received termination signal (%d) — initiating graceful shutdown...", signum)
    _running = False
    execution.save_state()
    sys.exit(0)


async def main():
    log.info("=" * 60)
    log.info("TRADER BOT v3 STARTING")
    log.info("News Ingestion Mode: %s", cfg.NEWS_FEED_MODE)
    log.info("Scorer Provider: %s | Model: %s", cfg.SCORER_PROVIDER, cfg.SCORER_MODEL)
    log.info("News Lookback Window: %d days | Max Ticker Articles: %d",
             cfg.NEWS_LOOKBACK_DAYS, cfg.NEWS_MAX_TICKER_ARTICLES)
    log.info("Execution Geometry: Delta=%.2f | Max Spread=%.1f%% | Min OI=%d | Horizon=%dm",
             cfg.TARGET_DELTA, cfg.MAX_SPREAD_PCT * 100, cfg.MIN_OPEN_INTEREST, cfg.EXIT_HORIZON_MINUTES)
    log.info("Grid Research Collector: Enabled=%s | Threshold=%.2f | SameDayOnly=%s",
             cfg.GRID_COLLECTOR_ENABLED, cfg.GRID_LOG_THRESHOLD, cfg.GRID_TRACK_SAME_DAY_ONLY)
    log.info("=" * 60)

    # 1. Initialize SQLite news database
    news_db.init_db()

    # 2. Recover open positions
    execution.load_state()

    # 3. Startup news backfill
    if cfg.BACKFILL_ON_STARTUP:
        log.info("Running cold-start news backfill (%d days) ...", cfg.BACKFILL_DAYS)
        try:
            # Run blocking REST backfill in threadpool executor
            loop = asyncio.get_running_loop()
            count = await loop.run_in_executor(None, news_db.backfill_news_from_alpaca,
                                              cfg.ALPACA_KEY, cfg.ALPACA_SECRET, None, None, cfg.BACKFILL_DAYS)
            log.info("Backfilled %d articles into news archive.", count)
        except Exception as e:
            log.warning("Startup backfill failed: %s", e)

    # 4. Start concurrent background tasks
    signal.signal(signal.SIGINT, handle_shutdown)
    signal.signal(signal.SIGTERM, handle_shutdown)

    tasks = [
        position_monitor_loop(),
        grid_collector.run_poll_loop(mkt.is_market_open),
    ]

    feed_mode = getattr(cfg, "NEWS_FEED_MODE", "websocket").lower()
    if feed_mode == "websocket":
        tasks.append(news_stream_listener())
    else:
        tasks.append(news_poller())

    if getattr(cfg, "SEC_FEED_ENABLED", True):
        tasks.append(sec_rss_poller())

    await asyncio.gather(*tasks)


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except (KeyboardInterrupt, SystemExit):
        log.info("Trader Bot v3 stopped.")

