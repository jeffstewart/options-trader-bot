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
import csv
import logging
import os
import signal as os_signal
from datetime import datetime, timedelta, timezone

import aiohttp
import feedparser
from alpaca.data.live import NewsDataStream

import config as cfg
import market as mkt
import execution as ex
import scoring
import sec_edgar

logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(levelname)-8s  %(message)s", datefmt="%m-%d %H:%M:%S")
log = logging.getLogger(__name__)

_seen_news_ids: set[str] = set()

# Throttled logging for the two high-volume pipeline gates (market-closed / regime). Every
# incoming article hits these, so per-item logging would bury the log -- instead emit the first
# drop of each reason immediately, then one summary line every GATE_LOG_EVERY drops.
# Counters self-reset on UTC date change (rather than being driven by market.py's daily-loss
# reset) to avoid inverting the bot -> market import direction for a logging concern.
GATE_LOG_EVERY = 50
_gate_drop_counts: dict[str, int] = {}
_gate_drop_day: "str | None" = None


def _log_gate_drop(reason: str) -> None:
    global _gate_drop_day
    today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    if today != _gate_drop_day:
        _gate_drop_day = today
        _gate_drop_counts.clear()
    n = _gate_drop_counts.get(reason, 0) + 1
    _gate_drop_counts[reason] = n
    if n == 1 or n % GATE_LOG_EVERY == 0:
        log.info("  ⏸️  %s — dropping signals before scoring (%d so far today)", reason, n)


async def process_signal(headline: str, body: str, source: str, symbols: list[str] = None,
                         article_ts: "datetime | None" = None):
    received_at = datetime.now(timezone.utc)
    if article_ts and article_ts.tzinfo is not None:
        feed_lag_s = (received_at - article_ts).total_seconds()
        if feed_lag_s > cfg.MAX_FEED_LAG_SECS:
            log.info("  → stale news (%.0fs) — skipping", feed_lag_s)
            return

    # Both of the next two gates used to `return` with NO log output at all. That made a fully
    # gated bot indistinguishable from a broken one: diagnosed 2026-07-29, v2 had ingested 10k+
    # news items and scored NOTHING since 07-23 (the last uptrend day) with zero trace in the log
    # of why. These are the highest-volume drops in the pipeline, so they log at a throttled
    # 1-line-per-N-drops cadence rather than per item.
    if not mkt.market_is_open():
        _log_gate_drop("market closed")
        return

    # ── Regime gate FIRST — see module docstring for why this is safe in v2 ────────────────
    if not mkt.market_in_uptrend():
        _log_gate_drop("regime gate: not in uptrend (long-only legs paused)")
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

    # Lotto entry cutoff, checked BEFORE scoring (2026-07-18): a signal arriving this late can
    # only route to lotto right now (news_call/PEAD disabled), and lotto force-closes same-day --
    # scoring it is pure wasted spend once the scorer swap to a paid model (Sonnet) happens. Only
    # safe to skip scoring entirely because lotto is the ONLY enabled strategy today; if
    # news_call/PEAD are ever re-enabled this condition stops applying on its own (they don't
    # share lotto's same-day-exit constraint). execute_lotto's own post-score check stays in place
    # as a backstop regardless -- scoring + dispatch takes real time, so the window can still be
    # crossed between this check and order placement.
    if (cfg.LOTTO_ENABLED and not cfg.NEWS_CALL_ENABLED and not cfg.PEAD_ENABLED
            and mkt.near_market_close(within_min=cfg.LOTTO_ENTRY_CUTOFF_MIN_BEFORE_CLOSE)):
        log.info("  → too close to the same-day forced exit (<%dmin left) — skipping scoring (lotto is the only leg live)",
                 cfg.LOTTO_ENTRY_CUTOFF_MIN_BEFORE_CLOSE)
        return

    signal = scoring.score_article(headline, body, source)
    if not signal:
        return
    signal["_headline"] = headline
    signal["_source"] = source

    sentiment = signal.get("sentiment")
    if sentiment != "bullish":
        return   # v2 has no bearish/short strategy to feed -- nothing else to do with this signal

    # The ONE threshold check (2026-07-29): a flat magnitude+confidence floor, replacing v1's
    # sliding `BASE_CONFIDENCE + (1-mag)*CONFIDENCE_SLOPE` gate -- see config.py for why that went
    # away. It stays HERE rather than at leg dispatch because everything below it costs money once
    # the scorer swap lands: the ticker corrector is a second LLM call, and it would be spent on
    # signals that cannot open a position anyway.
    magnitude, confidence = float(signal.get("magnitude", 0)), float(signal.get("confidence", 0))
    if magnitude < cfg.MIN_MAGNITUDE or confidence < cfg.MIN_CONFIDENCE:
        log.info("  → below threshold (mag=%.2f/%.2f conf=%.2f/%.2f) — skipping",
                 magnitude, cfg.MIN_MAGNITUDE, confidence, cfg.MIN_CONFIDENCE)
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

    # cfg.SCORER_MODEL, not OLLAMA_MODEL -- this line still said "llama3.2" after the 2026-07-29
    # Sonnet swap, so the 07-31 AMZN trade was logged as scored by a model that never saw it.
    # OLLAMA_MODEL is now only the ticker corrector.
    log.info("  🤖 %s: bullish conf=%.2f mag=%.2f ticker=%s", cfg.SCORER_MODEL, confidence, magnitude, ticker)
    log.info("     %s", signal.get("reasoning", ""))

    loop = asyncio.get_event_loop()

    if cfg.NEWS_CALL_ENABLED and magnitude >= cfg.NEWS_CALL_MIN_MAGNITUDE:
        try:
            await asyncio.wait_for(loop.run_in_executor(None, ex.execute_news_call, ticker, signal), timeout=30)
        except asyncio.TimeoutError:
            log.error("execute_news_call timed out for %s", ticker)

    if cfg.PEAD_ENABLED and ex._is_earnings_headline(headline):
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


async def alpaca_news_rest_poller():
    """REST-polling substitute for NewsDataStream (2026-07-19) -- see config.py's
    USE_ALPACA_STREAMS docstring for why: the websocket counts against Alpaca's 1-connection
    -per-login cap and collides with v1's own stream, REST calls don't. Cursors on the last
    article's timestamp (falls back to `since` unchanged if a poll returns nothing), with
    _seen_news_ids as a belt-and-suspenders dedup against boundary repeats."""
    log.info("📡 Alpaca news REST poller started (every %ds)", cfg.NEWS_POLL_SECS)
    url = "https://data.alpaca.markets/v1beta1/news"
    headers = {"APCA-API-KEY-ID": cfg.ALPACA_KEY, "APCA-API-SECRET-KEY": cfg.ALPACA_SECRET}
    since = datetime.now(timezone.utc) - timedelta(seconds=cfg.NEWS_POLL_SECS)
    async with aiohttp.ClientSession() as session:
        while True:
            try:
                params = {"start": since.strftime("%Y-%m-%dT%H:%M:%SZ"), "sort": "asc", "limit": "50"}
                async with session.get(url, headers=headers, params=params,
                                        timeout=aiohttp.ClientTimeout(total=10)) as r:
                    data = await r.json()
                items = data.get("news", [])
                for item in items:
                    news_id = str(item.get("id"))
                    if news_id in _seen_news_ids:
                        continue
                    _seen_news_ids.add(news_id)
                    headline = item.get("headline", "") or ""
                    body = item.get("summary", "") or ""
                    symbols = item.get("symbols") or []
                    raw_ts = item.get("updated_at") or item.get("created_at")
                    article_ts = None
                    if raw_ts:
                        try:
                            article_ts = datetime.fromisoformat(raw_ts.replace("Z", "+00:00"))
                        except Exception:
                            pass
                    log.info("📰 [Alpaca] %s", headline[:100])
                    await process_signal(headline, body, "Alpaca/Benzinga", symbols=symbols, article_ts=article_ts)
                if items:
                    last_ts = items[-1].get("updated_at") or items[-1].get("created_at")
                    if last_ts:
                        try:
                            since = datetime.fromisoformat(last_ts.replace("Z", "+00:00"))
                        except Exception:
                            pass
            except Exception as e:
                log.warning("Alpaca news REST poll failed: %s", e)
            await asyncio.sleep(cfg.NEWS_POLL_SECS)


def _log_sec_pointer(ticker: str, cik: int, accession: str, headline: str):
    """Logs (cik, accession) for every SEC filing that got the full-content fetch (2026-07-23).
    Since sec_rss_poller now sends the model a self-contained body derived entirely from this
    pointer (see its comment), this is all that's needed to regenerate the EXACT historical
    prompt input later for backtesting/prompt-tuning -- no need to store the document text
    itself, which can be re-fetched from SEC's permanent archive at any time via
    sec_edgar.fetch_8k_content(session, cik, accession, ...)."""
    try:
        path = "sec_scored_filings.csv"
        write_header = not os.path.exists(path)
        with open(path, "a", newline="") as f:
            w = csv.writer(f)
            if write_header:
                w.writerow(["timestamp", "ticker", "cik", "accession", "headline"])
            w.writerow([datetime.now(timezone.utc).isoformat(), ticker or "", cik, accession, headline[:150]])
    except Exception as e:
        log.debug("SEC pointer log failed: %s", e)


async def sec_rss_poller():
    """Was scoring the bare RSS filer-index title with an EMPTY body -- no ticker, no content at
    all (see sec_edgar.py docstring / project_news_source_quality memory: 99.7% of SEC signals
    never resolved to a ticker). Now resolves CIK -> ticker deterministically and, for a curated
    set of material Item codes, fetches the actual primary 8-K document + press-release exhibit."""
    log.info("📡 SEC EDGAR 8-K RSS poller started (every %ds)", cfg.SEC_POLL_SECS)
    async with aiohttp.ClientSession() as session:
        while True:
            try:
                async with session.get(cfg.SEC_RSS_URL, headers={"User-Agent": cfg.SEC_USER_AGENT},
                                        timeout=aiohttp.ClientTimeout(total=10)) as r:
                    raw = await r.text()
                feed = feedparser.parse(raw)
                cik_map = await sec_edgar.load_cik_ticker_map(session, cfg.SEC_USER_AGENT)

                for entry in feed.entries:
                    eid = entry.get("id", entry.get("link", ""))
                    if eid in _seen_news_ids:
                        continue
                    _seen_news_ids.add(eid)
                    title   = entry.get("title", "")
                    summary = entry.get("summary", "")
                    link    = entry.get("link", "")

                    cik, accession = sec_edgar.extract_cik_accession(link)
                    ticker = cik_map.get(cik) if cik else None

                    headline = f"{title} [{ticker}]" if ticker else title
                    body = summary
                    if cik and accession and sec_edgar.item_codes_in_summary(summary) & sec_edgar.MATERIAL_ITEMS:
                        try:
                            full_text = await sec_edgar.fetch_8k_content(session, cik, accession, cfg.SEC_USER_AGENT)
                        except Exception as e:
                            log.debug("SEC 8-K content fetch failed for %s: %s", eid, e)
                            full_text = ""
                        if full_text:
                            # Self-contained body, NOT summary+full_text (2026-07-23, ported from
                            # core/bot.py): fetch_8k_content's skip_to_item logic already lands the
                            # extracted text right at the "Item X.XX" heading, so the ephemeral RSS
                            # `summary` adds nothing -- dropping it means everything the model sees
                            # is 100% derivable, forever, from just (cik, accession). Log the
                            # pointer so future prompt/backtest work can regenerate this exact
                            # input without storing the document itself.
                            body = full_text
                            _log_sec_pointer(ticker, cik, accession, headline)

                    log.info("📰 [SEC 8-K] %s", headline[:100])
                    await process_signal(headline, body, "SEC EDGAR 8-K")
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
    log.info("🚀 Trader bot v2 starting (paper mode · scorer: %s)", cfg.SCORER_MODEL)
    log.info("   Position budget      : %.0f%% of equity (flat, not magnitude-scaled)",
              cfg.MAX_POSITION_FRAC_OF_EQUITY * 100)
    log.info("   Max open positions   : %d", cfg.MAX_OPEN_POSITIONS)
    log.info("   Daily loss limit     : %.0f%% of equity, resets 00:00 UTC", cfg.DAILY_LOSS_LIMIT_PCT * 100)
    log.info("   Ticker corrector     : %s", "ON" if cfg.TICKER_CORRECTOR_ENABLED else "OFF")
    log.info("   Confirm/veto gate    : REMOVED (see config.py module docstring)")
    log.info("   SEC EDGAR 8-K feed   : %s", "ON" if cfg.SEC_FEED_ENABLED else "OFF (unproven — see config.py)")
    log.info("   Pairs / bear_short   : REMOVED (see config.py module docstring)")
    log.info("   Scorer thresholds    : mag>=%.2f AND conf>=%.2f (single gate, %s scale)",
             cfg.MIN_MAGNITUDE, cfg.MIN_CONFIDENCE, cfg.SCORER_MODEL)
    log.info("   Scorer effort/think  : %s / %s", cfg.SCORER_EFFORT, cfg.SCORER_THINKING)

    # Fail fast and LOUD: without a working scorer this bot cannot trade, and a silent scorer
    # failure is indistinguishable from a quiet market. Exit rather than run blind.
    if not scoring.preflight():
        log.error("🛑 scorer preflight failed — exiting instead of running with a dead scorer.")
        return

    log.info("📂 Loading open positions from Alpaca…")
    ex.load_open_positions_from_alpaca()

    stream = None
    news_task = None
    stock_task = None
    if cfg.USE_ALPACA_STREAMS:
        PRICE_WATCHLIST = sorted(cfg.TRADEABLE_ETFS | {
            "AAPL", "MSFT", "NVDA", "AMZN", "GOOGL", "META", "TSLA",
            "JPM", "GS", "MS", "BAC", "AMD", "INTC", "QCOM", "MU",
        })
        mkt.subscribe_price_stream(PRICE_WATCHLIST)

        stream = NewsDataStream(cfg.ALPACA_KEY, cfg.ALPACA_SECRET)
        mkt.patch_reconnect_safety(stream, "news")
        stream.subscribe_news(handle_alpaca_news, "*")

        news_task = asyncio.create_task(_supervised(stream._run_forever, "news stream"))
        stock_task = asyncio.create_task(_supervised(mkt.stock_stream._run_forever, "stock stream"))
    else:
        log.info("📡 USE_ALPACA_STREAMS=false — news via REST polling, prices via on-demand REST (no persistent connections)")
        news_task = asyncio.create_task(_supervised(alpaca_news_rest_poller, "news REST poller"))

    log.info("📡 All news sources starting…")
    tasks = [t for t in (news_task, stock_task) if t is not None] + [
        asyncio.create_task(_supervised(ex.trailing_stop_monitor, "trailing-stop monitor")),
    ]
    if cfg.SEC_FEED_ENABLED:
        tasks.append(asyncio.create_task(_supervised(sec_rss_poller, "SEC RSS poller")))
    else:
        log.info("📡 SEC EDGAR 8-K feed: DISABLED (config.SEC_FEED_ENABLED) — Alpaca/Benzinga only")

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

    if cfg.USE_ALPACA_STREAMS:
        log.info("🔌 Closing data-stream connections cleanly…")
        for s in (stream, mkt.stock_stream):
            try:
                await s.close()
            except Exception as e:
                log.debug("stream close error: %s", e)
    log.info("👋 Shutdown complete")


if __name__ == "__main__":
    asyncio.run(main())
