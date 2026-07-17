"""
bot.py (v2) — entrypoint + the reordered signal pipeline.

Pipeline order (the key structural change from v1):
  feed-lag / market-open checks
    -> REGIME GATE (before any scoring, local or paid -- see config.py: with pairs/bear_short/
       qqq_macro all dropped, nothing in v2 needs scoring during a downtrend, so this can be a
       hard gate rather than the two-stage compromise a still-live all-regime strategy would need)
    -> pre-score noise / soft-catalyst filters (free, headline-only)
    -> Ollama scoring
    -> sentiment must be bullish (v2 has no short/bearish strategy to feed)
    -> magnitude/confidence threshold gate
    -> ticker-selection corrector (picks ONE best beneficiary from the feed's own symbols list)
    -> leg dispatch: news_call, lotto, pead -- no confirm/veto gate, no shadow infrastructure.

Usage:
    pip install -r requirements.txt
    cp .env.example .env   # fill in a FRESH Alpaca paper account's keys
    ollama serve &
    ollama pull llama3.2
    python bot.py
"""

import asyncio
import logging
import signal as os_signal
from datetime import datetime, timezone

import feedparser
from alpaca.data.live import NewsDataStream

import config as cfg
import market as mkt
import execution as ex
import scoring

logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(levelname)-8s  %(message)s", datefmt="%H:%M:%S")
log = logging.getLogger(__name__)

_seen_news_ids: set[str] = set()


async def process_signal(headline: str, body: str, source: str, symbols: list[str] = None,
                         article_ts: "datetime | None" = None):
    received_at = datetime.now(timezone.utc)
    if article_ts and article_ts.tzinfo is not None:
        feed_lag_s = (received_at - article_ts).total_seconds()
        if feed_lag_s > cfg.MAX_FEED_LAG_SECS:
            log.info("  → stale news (%.0fs) — skipping", feed_lag_s)
            return

    if not mkt.market_is_open():
        return

    # ── Regime gate FIRST — see module docstring for why this is safe in v2 ────────────────
    if not mkt.market_in_uptrend():
        return

    # ── Pre-score filters (free, headline-only) ─────────────────────────────────────────────
    if cfg.PRESCORE_FILTER_ENABLED:
        from filters import prescore_noise
        noise = prescore_noise(headline)
        if noise:
            log.info("  → pre-score noise filter (%r) — skipping (no Ollama call)", noise)
            return
    if cfg.SOFT_CATALYST_GATE_ENABLED:
        from filters import soft_catalyst_hit
        soft = soft_catalyst_hit(headline, body)
        if soft:
            log.info("  🚧 soft-catalyst (%r) — pre-score drop", soft)
            return

    signal = scoring.score_article(headline, body, source)
    if not signal:
        return
    signal["_headline"] = headline

    sentiment = signal.get("sentiment")
    if sentiment != "bullish":
        return   # v2 has no bearish/short strategy to feed -- nothing else to do with this signal

    magnitude, confidence = float(signal.get("magnitude", 0)), float(signal.get("confidence", 0))
    if magnitude < cfg.MIN_MAGNITUDE:
        return
    required_conf = cfg.BASE_CONFIDENCE + (1 - magnitude) * cfg.CONFIDENCE_SLOPE
    if confidence < required_conf:
        return

    tickers = signal.get("tickers") or []
    if not tickers:
        return   # macro/no-ticker signal -- qqq_macro not implemented in v2, see config.py

    ticker = tickers[0]
    if cfg.TICKER_CORRECTOR_ENABLED and symbols:
        corrected = scoring.correct_ticker(headline, body, symbols)
        if corrected and corrected != ticker:
            log.info("  🔧 ticker corrected: %s → %s", ticker, corrected)
            signal["_ticker_corrected"] = corrected
            ticker = corrected

    ex.log_regime_decision(signal, in_uptrend=True)

    if mkt.on_cooldown(ticker):
        log.info("  → %s on cooldown", ticker)
        return
    if ticker in ("BTC", "ETH"):
        return

    log.info("  🤖 %s: bullish conf=%.2f mag=%.2f ticker=%s", cfg.OLLAMA_MODEL, confidence, magnitude, ticker)
    log.info("     %s", signal.get("reasoning", ""))

    loop = asyncio.get_event_loop()

    if cfg.NEWS_CALL_ENABLED and magnitude >= cfg.NEWS_CALL_MIN_MAGNITUDE:
        try:
            await asyncio.wait_for(loop.run_in_executor(None, ex.execute_news_call, ticker, signal), timeout=30)
        except asyncio.TimeoutError:
            log.error("execute_news_call timed out for %s", ticker)

    if ex._is_earnings_headline(headline):
        try:
            await asyncio.wait_for(loop.run_in_executor(None, ex.execute_pead, ticker, signal), timeout=30)
        except asyncio.TimeoutError:
            log.error("execute_pead timed out for %s", ticker)

    if cfg.LOTTO_ENABLED and magnitude >= cfg.LOTTO_MIN_MAGNITUDE and confidence >= cfg.LOTTO_MIN_CONFIDENCE:
        try:
            await asyncio.wait_for(loop.run_in_executor(None, ex.execute_lotto, ticker, signal), timeout=30)
        except asyncio.TimeoutError:
            log.error("execute_lotto timed out for %s", ticker)


# ═══════════════════════════════════════════════════════════════════════════════
# NEWS SOURCES
# ═══════════════════════════════════════════════════════════════════════════════


async def handle_alpaca_news(news):
    for item in (news if isinstance(news, list) else [news]):
        news_id = getattr(item, "id", None) or str(item)
        if news_id in _seen_news_ids:
            continue
        _seen_news_ids.add(news_id)
        headline = getattr(item, "headline", "") or ""
        body = getattr(item, "summary", "") or ""
        symbols = list(getattr(item, "symbols", None) or [])
        raw_ts = getattr(item, "created_at", None) or getattr(item, "updated_at", None)
        article_ts = None
        if raw_ts:
            try:
                article_ts = raw_ts if raw_ts.tzinfo else raw_ts.replace(tzinfo=timezone.utc)
            except Exception:
                pass
        log.info("📰 [Alpaca] %s", headline[:100])
        await process_signal(headline, body, "Alpaca/Benzinga", symbols=symbols, article_ts=article_ts)


async def sec_rss_poller():
    log.info("📡 SEC EDGAR 8-K RSS poller started (every %ds)", cfg.SEC_POLL_SECS)
    while True:
        try:
            feed = feedparser.parse(cfg.SEC_RSS_URL, agent=cfg.SEC_USER_AGENT)
            for entry in feed.entries:
                eid = entry.get("id", entry.get("link", ""))
                if eid in _seen_news_ids:
                    continue
                _seen_news_ids.add(eid)
                headline = entry.get("title", "")
                log.info("📰 [SEC 8-K] %s", headline[:100])
                await process_signal(headline, "", "SEC 8-K")
        except Exception as e:
            log.warning("SEC RSS poll failed: %s", e)
        await asyncio.sleep(cfg.SEC_POLL_SECS)


async def _supervised(make_coro, name):
    """Restart a coroutine on crash instead of killing the whole bot (module-global state
    means a restart resumes cleanly). The external watchdog is the backstop if the whole
    process dies -- see docs/UNATTENDED.md for the v1 pattern this mirrors."""
    while True:
        try:
            await make_coro()
            log.warning("⚠️ %s exited cleanly — restarting in 5s", name)
        except asyncio.CancelledError:
            raise
        except Exception as e:
            log.error("⚠️ %s crashed (%s: %s) — restarting in 5s", name, type(e).__name__, e)
        await asyncio.sleep(5)


async def main():
    log.info("🚀 Trader bot v2 starting (paper mode · model: %s)", cfg.OLLAMA_MODEL)
    log.info("   Position budget      : %.0f%% of equity (flat, not magnitude-scaled)",
              cfg.MAX_POSITION_FRAC_OF_EQUITY * 100)
    log.info("   Max open positions   : %d", cfg.MAX_OPEN_POSITIONS)
    log.info("   Daily loss limit     : %.0f%% of equity, resets 00:00 UTC", cfg.DAILY_LOSS_LIMIT_PCT * 100)
    log.info("   Ticker corrector     : %s", "ON" if cfg.TICKER_CORRECTOR_ENABLED else "OFF")
    log.info("   Confirm/veto gate    : REMOVED (see config.py module docstring)")
    log.info("   Pairs / bear_short   : REMOVED (see config.py module docstring)")

    log.info("📂 Loading open positions from Alpaca…")
    ex.load_open_positions_from_alpaca()

    PRICE_WATCHLIST = sorted(cfg.TRADEABLE_ETFS | {
        "AAPL", "MSFT", "NVDA", "AMZN", "GOOGL", "META", "TSLA",
        "JPM", "GS", "MS", "BAC", "AMD", "INTC", "QCOM", "MU",
    })
    mkt.subscribe_price_stream(PRICE_WATCHLIST)

    stream = NewsDataStream(cfg.ALPACA_KEY, cfg.ALPACA_SECRET)
    mkt.patch_reconnect_safety(stream, "news")
    stream.subscribe_news(handle_alpaca_news, "*")

    log.info("📡 All news sources starting…")
    tasks = [
        asyncio.create_task(_supervised(stream._run_forever, "news stream")),
        asyncio.create_task(_supervised(mkt.stock_stream._run_forever, "stock stream")),
        asyncio.create_task(_supervised(ex.trailing_stop_monitor, "trailing-stop monitor")),
        asyncio.create_task(_supervised(sec_rss_poller, "SEC RSS poller")),
    ]

    # ── Graceful shutdown (2026-07-17) ───────────────────────────────────────────────────────
    # `pkill`/manage.sh send SIGTERM, and with NO handler Python's default disposition kills the
    # process immediately -- no finally blocks run, no chance to close the websockets. Alpaca's
    # server then sees a dead TCP connection instead of a clean close handshake, and (observed
    # directly this session, on both v1 and this bot) won't accept a reconnect with the same
    # credentials until whatever timeout it uses to notice the orphaned connection expires --
    # this is very likely the actual mechanism behind the "connection limit exceeded" cooldown
    # seen after every restart. Catching SIGTERM/SIGINT to cancel the tasks and explicitly close
    # both streams should let Alpaca release the slot immediately instead of waiting it out.
    stop_event = asyncio.Event()
    loop = asyncio.get_running_loop()

    def _on_signal(sig_name):
        log.info("🛑 received %s — shutting down cleanly", sig_name)
        stop_event.set()

    for sig, name in ((os_signal.SIGTERM, "SIGTERM"), (os_signal.SIGINT, "SIGINT")):
        loop.add_signal_handler(sig, _on_signal, name)

    await stop_event.wait()
    for t in tasks:
        t.cancel()
    await asyncio.gather(*tasks, return_exceptions=True)

    log.info("🔌 Closing data-stream connections cleanly…")
    for s in (stream, mkt.stock_stream):
        try:
            await s.close()
        except Exception as e:
            log.debug("stream close error: %s", e)
    log.info("👋 Shutdown complete")


if __name__ == "__main__":
    asyncio.run(main())
