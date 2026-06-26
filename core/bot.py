"""
News-driven trading bot — Alpaca paper trading + local Ollama sentiment scoring.

Sources: Alpaca/Benzinga news stream · SEC EDGAR 8-K RSS · NewsAPI headlines
Assets:  Equity options (long calls) · ETF options

Usage:
    pip install alpaca-py openai python-dotenv aiohttp feedparser
    cp .env.example .env   # fill in keys
    ollama serve &
    ollama pull llama3.2
    python bot.py
"""

import asyncio
import csv
import json
import logging
import os
import re
import time
from datetime import date, datetime, timedelta, timezone
from typing import Optional

import aiohttp
import feedparser
from openai import OpenAI, RateLimitError
from alpaca.data.live import NewsDataStream, StockDataStream
from alpaca.data.historical.option import OptionHistoricalDataClient
from alpaca.data.historical import StockHistoricalDataClient
from alpaca.data.requests import (
    OptionLatestQuoteRequest, OptionSnapshotRequest,
    StockLatestTradeRequest, StockBarsRequest,
)
from alpaca.data.timeframe import TimeFrame
from alpaca.trading.client import TradingClient
from alpaca.trading.enums import (
    AssetClass, OrderSide, TimeInForce, ContractType, PositionIntent, OrderType
)
from alpaca.trading.requests import (
    GetOptionContractsRequest,
    LimitOrderRequest,
    MarketOrderRequest,
    StopLimitOrderRequest,
    GetOrdersRequest,
)
from alpaca.trading.enums import QueryOrderStatus
from dotenv import load_dotenv
from config import (
    ALPACA_KEY, ALPACA_SECRET, NEWSAPI_KEY,
    OLLAMA_BASE_URL, OLLAMA_MODEL,
    GROQ_SCORER_SHADOW_ENABLED, GROQ_BASE_URL, GROQ_KEY, GROQ_MODEL, GROQ_AB_MODELS,
    HYBRID_SCORER_ENABLED, GEMINI_BASE_URL, GEMINI_KEY, GEMINI_MODEL,
    GEMINI_DAILY_CAP, GEMINI_CONFIRM_MIN_MAGNITUDE, GEMINI_VETO_SHADOW_ENABLED,
    GATE_LABEL, GATE_LATCH_ON_RATELIMIT,
    SOFT_CATALYST_GATE_ENABLED, SOFT_CATALYST_PATTERNS,
    MAGNITUDE_SIZING_ENABLED, NONMAG_SIZE_FRAC, MAX_FEED_LAG_SECS,
    PRESCORE_FILTER_ENABLED, PRESCORE_SKIP_PATTERNS,
    COOLDOWN_SECS, MIN_MAGNITUDE, BASE_CONFIDENCE, CONFIDENCE_SLOPE,
    STOCK_MIN_CONFIDENCE, STOCK_MIN_MAGNITUDE,
    MAX_POSITION_USD, DAILY_LOSS_LIMIT, MAX_CONTRACT_BUDGET_MULT,
    PRICE_CROSSCHECK_ENABLED, PRICE_CROSSCHECK_TOL,
    MIN_DAYS_TO_EXPIRY, MAX_DAYS_TO_EXPIRY, TARGET_DELTA,
    QQQ_MACRO_DTE_MIN, QQQ_MACRO_DTE_MAX, QQQ_MACRO_TARGET_DELTA, QQQ_MACRO_ENABLED,
    MIN_MARKET_CAP_B, MAX_SPREAD_PCT, MIN_OPEN_INTEREST, MIN_STOCK_PRICE,
    TRAILING_STOP_PCT, EXIT_TIERS, MONITOR_INTERVAL,
    PEAD_EXIT_TIERS, PEAD_BREAKEVEN_LOCK, BEAR_SHORT_TRAIL_PCT, BEAR_SHORT_ENABLED,
    PEAD_MAX_POSITIONS, PEAD_SHADOW_ENABLED,
    PAIRS_ENABLED, PAIRS_LONG_TRAIL, PAIRS_SHORT_TRAIL,
    ROUTER_SHADOW_ENABLED,
    LOTTO_ENABLED, LOTTO_MIN_MAGNITUDE, LOTTO_MIN_CONFIDENCE, LOTTO_TARGET_DELTA,
    LOTTO_POSITION_USD, LOTTO_MIN_DTE, LOTTO_MAX_DTE, LOTTO_STRIKE_HI_MULT,
    LOTTO_OOW_SHADOW_ENABLED, LOTTO_OOW_MAX_DTE,
    LOTTO_HARD_CAP_ENABLED, LOTTO_HARD_CAP_MULT,
    LOTTO_EXIT_TIERS,
    MATERIALITY_SHADOW_ENABLED, MATERIALITY_GATE, MATERIALITY_GATE_ENABLED,
    STOCK_ENABLED, STOCK_SHADOW_ENABLED,
    MAX_OPEN_POSITIONS, NEWS_CALL_ENABLED, NEWS_CALL_MIN_MAGNITUDE, NEWS_CALL_SHADOW_ENABLED,
    NEWS_CALL_DTE_MIN, NEWS_CALL_DTE_MAX, NEWS_CALL_TARGET_DELTA,
    NEWS_CALL_MAX_HOLD_DAYS, NEWS_CALL_TRAIL_PCT, NEWS_CALL_EXIT_TIERS,
    NEWS_CALL_BUDGET_MULT, NEWS_CALL_MAX_PER_TICKER_DAY,
    NEWS_STOCK_MAX_HOLD_DAYS, PEAD_MAX_HOLD_DAYS, LOTTO_MAX_HOLD_DAYS,
    TIME_STOP_EOD_WINDOW_MIN,
    TRADEABLE_ETFS,
    REGIME_FILTER_ENABLED, REGIME_INDEX, REGIME_MA_DAYS, REGIME_MOMENTUM_DAYS,
    REGIME_BYPASS_MIN_MAGNITUDE,
    SEC_RSS_URL, SEC_POLL_SECS, SEC_USER_AGENT,
    NEWSAPI_URL, NEWSAPI_POLL_SECS, NEWSAPI_PARAMS,
)

load_dotenv()
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

# ═══════════════════════════════════════════════════════════════════════════════
# STATE
# ═══════════════════════════════════════════════════════════════════════════════

_recent_trades:      dict[str, float] = {}   # underlying → last trade epoch
_seen_news_ids:      set[str]         = set()
_daily_loss_usd:     float            = 0.0   # realized losses accrued this UTC day (resets daily)
_trading_halted:     bool             = False  # daily-loss breaker latch (clears at new UTC day)
_loss_day:           "datetime.date | None" = None   # UTC date the breaker counters belong to
# Cached account equity for the dynamic (2%-of-equity) loss limit — refreshed at most every
# _EQUITY_TTL secs so the per-signal guardrail check doesn't hammer the Alpaca API.
_equity_cache:       dict = {"val": None, "ts": 0.0}
_EQUITY_TTL:         float = 300.0
DAILY_LOSS_LIMIT_PCT: float = 0.02   # halt new trades once a day's realized loss ≥ 2% of equity

# symbol → {qty, entry_price, peak_price, underlying, asset_type}
_monitored_positions: dict[str, dict] = {}

# SHADOW positions — paper-only (NO orders, NO capital). Used to track what a DISABLED
# strategy (currently news_call) WOULD do, so we accumulate forward data risk-free.
# Completely isolated from _monitored_positions; all access is try/except-wrapped so a
# shadow bug can never affect real trading.
_shadow_positions: dict[str, dict] = {}
SHADOW_STATE_FILE  = "shadow_state.json"
SHADOW_TRADES_FILE = "shadow_trades.csv"
MAX_SHADOW_POSITIONS = 60   # soft cap so shadow quote-fetches don't balloon

# L/S pairs: accumulate the day's best bull + best bear, enter one pair per day
# when both sides are present. Reset on a new UTC calendar day (matches the
# backtest's by-date grouping). entered=True after a pair is opened that day.
_pairs_day: dict = {"date": None, "bull": None, "bear": None, "entered": False}

# Shadow router: ensure at most one "pairs" routing decision is logged per day.
_router_day: dict = {"date": None, "logged_pairs": False}

# Hybrid scorer (Gemini confirm/veto gate) — per-UTC-day counters so we can track how
# often Gemini is consulted, confirms, vetoes, and (key for the free-tier) how often we
# fall back to Ollama because the daily cap / rate-limit was hit. _capped latches the
# fallback for the rest of the day once Gemini signals it's out of free quota.
_gemini_day: dict = {"date": None, "calls": 0, "confirms": 0, "vetoes": 0,
                     "fallbacks": 0, "capped": False}
GEMINI_DECISIONS_FILE = "gemini_decisions.csv"

BOT_STATE_FILE = "bot_state.json"

STOCK_TRAIL_PCT = 0.10  # flat 10% trailing stop on stock price (matches backtest)

def tiered_trail_pct(gain: float) -> float:
    """
    Profit-tiered trailing-stop width for a given gain (peak/entry − 1). Wide
    early (let a move run), tightening once it's a big winner. From EXIT_TIERS;
    falls back to the flat TRAILING_STOP_PCT if tiers are unset.
    """
    if not EXIT_TIERS:
        return TRAILING_STOP_PCT
    for thresh, trail in EXIT_TIERS:
        if gain < thresh:
            return trail
    return EXIT_TIERS[-1][1]


def news_call_trail_pct(gain: float) -> float:
    """RATCHET trail for news_call (deployed 2026-06-26): tightens as the PEAK gain grows —
    NEWS_CALL_EXIT_TIERS = 40% until +15%, 25% until +35%, 15% above. Since long_trail_for is fed the
    peak gain, this only ever tightens (true ratchet). Falls back to flat NEWS_CALL_TRAIL_PCT if the
    tiers are emptied (the revert switch). See exit_intraday_sweep.py for the evidence."""
    if not NEWS_CALL_EXIT_TIERS:
        return NEWS_CALL_TRAIL_PCT
    for thresh, trail in NEWS_CALL_EXIT_TIERS:
        if gain < thresh:
            return trail
    return NEWS_CALL_EXIT_TIERS[-1][1]

def pead_trail_pct(gain: float) -> float:
    """Tiered trailing stop for PEAD stock positions. Tightens as gain grows."""
    for thresh, trail in PEAD_EXIT_TIERS:
        if gain < thresh:
            return trail
    return PEAD_EXIT_TIERS[-1][1]

def lotto_trail_pct(gain: float) -> float:
    """Wide 'let it run' tiered trail for lotto OTM calls (LOTTO_EXIT_TIERS)."""
    for thresh, trail in LOTTO_EXIT_TIERS:
        if gain < thresh:
            return trail
    return LOTTO_EXIT_TIERS[-1][1]

def long_trail_for(strategy: str, asset_type: str, gain: float) -> float:
    """Trail width (fraction below peak) for a LONG position, by strategy.
    pead → tiered; lotto → wide tiered; pairs_long → flat PAIRS_LONG_TRAIL;
    plain stock → flat STOCK_TRAIL_PCT; option → profit-tiered."""
    if strategy == "pead":
        return pead_trail_pct(gain)
    if strategy == "lotto":
        return lotto_trail_pct(gain)
    if strategy == "news_call":
        return news_call_trail_pct(gain)    # RATCHET (2026-06-26): 40/25/15 by peak-gain @ +15/+35%
    if strategy == "pairs_long":
        return PAIRS_LONG_TRAIL
    if asset_type == "stock":
        return STOCK_TRAIL_PCT
    return tiered_trail_pct(gain)            # qqq_macro + any other option leg: tiered EXIT_TIERS

def short_trail_for(strategy: str) -> float:
    """Trail width (fraction above trough) for a SHORT position, by strategy."""
    return PAIRS_SHORT_TRAIL if strategy == "pairs_short" else BEAR_SHORT_TRAIL_PCT

def max_hold_days_for(strategy: str, asset_type: str) -> int:
    """Return the max calendar days to hold a position before time-exit.
    0 = no time cap. Trailing stop still fires regardless."""
    if strategy == "stock":
        return NEWS_STOCK_MAX_HOLD_DAYS
    if strategy == "pead":
        return PEAD_MAX_HOLD_DAYS
    if strategy == "lotto":
        return LOTTO_MAX_HOLD_DAYS
    if strategy == "news_call":
        return NEWS_CALL_MAX_HOLD_DAYS      # 3d cap (exit sweep) — short-dated Δ0.40/DTE10 calls
    return 0   # pairs, bear_short, qqq_macro: no time cap


def _time_stop_hit(pos: dict) -> bool:
    """True if the position has been held longer than its strategy's max hold."""
    entry_dt = pos.get("entry_dt")
    if not entry_dt:
        return False
    max_days = max_hold_days_for(pos.get("strategy", ""), pos.get("asset_type", ""))
    if max_days <= 0:
        return False
    days_held = (datetime.now(timezone.utc) - entry_dt).days
    return days_held >= max_days


def _at_position_cap(strategy: str) -> bool:
    """Return True (blocking the trade) if we've hit MAX_OPEN_POSITIONS.
    Logs a clear reason so the dashboard errors panel stays clean.
    Lotto uses a separate smaller sub-cap (MAX_OPEN_POSITIONS // 4) so a
    handful of cheap $250 tickets don't consume slots needed by stock/pairs legs.
    PEAD gets its OWN reserved sub-cap (PEAD_MAX_POSITIONS) and is excluded from the
    main denominator, so a full main book can never starve it (the root cause of pead
    firing once ever). Its overflow is shadowed by the caller, not dropped."""
    if strategy == "lotto":
        lotto_n = sum(1 for p in _monitored_positions.values() if p.get("strategy") == "lotto")
        lotto_cap = max(5, MAX_OPEN_POSITIONS // 4)
        if lotto_n >= lotto_cap:
            log.info("  → lotto sub-cap (%d/%d) reached — skipping lotto on this signal",
                     lotto_n, lotto_cap)
            return True
        return False   # lotto doesn't count against the main cap
    if strategy == "pead":
        pead_n = sum(1 for p in _monitored_positions.values() if p.get("strategy") == "pead")
        if pead_n >= PEAD_MAX_POSITIONS:
            log.info("  → pead sub-cap (%d/%d) reached — %s", pead_n, PEAD_MAX_POSITIONS,
                     "shadowing the overflow" if PEAD_SHADOW_ENABLED else "skipping")
            return True
        return False   # pead is reserved — never blocked by, nor counted against, the main cap
    # Main cap excludes pead (its reserved slots are additive, not part of the general budget).
    n = sum(1 for p in _monitored_positions.values() if p.get("strategy") != "pead")
    if n >= MAX_OPEN_POSITIONS:
        log.info("  → position cap (%d/%d) reached — skipping new %s trade "
                 "(existing positions will free slots as stops trigger)",
                 n, MAX_OPEN_POSITIONS, strategy)
        return True
    return False

def _write_bot_state():
    """Write current stop prices + position metadata to bot_state.json.
    Dashboard reads stop_price/peak_price/entry_price; restart recovery reads
    asset_type/strategy/qty/underlying to reload stock positions."""
    state = {}
    for symbol, pos in _monitored_positions.items():
        asset_type = pos.get("asset_type", "option")
        entry = pos["entry_price"]
        entry_dt_iso = pos["entry_dt"].isoformat() if pos.get("entry_dt") else None
        if asset_type == "stock_short":
            trough = pos.get("trough_price", entry)
            stop   = round(trough * (1 + short_trail_for(pos.get("strategy", "bear_short"))), 4)
            state[symbol] = {
                "stop_price":   stop,
                "trough_price": trough,
                "entry_price":  entry,
                "asset_type":   asset_type,
                "strategy":     pos.get("strategy", "bear_short"),
                "qty":          pos.get("qty", 1),
                "underlying":   pos.get("underlying", symbol),
                "entry_dt":     entry_dt_iso,
            }
        else:
            gain  = (pos["peak_price"] / entry - 1.0) if entry else 0.0
            trail = long_trail_for(pos.get("strategy", "news_call"), asset_type, gain)
            state[symbol] = {
                "stop_price":  round(pos["peak_price"] * (1 - trail), 4),
                "peak_price":  pos["peak_price"],
                "entry_price": entry,
                "asset_type":  asset_type,
                "strategy":    pos.get("strategy", "news_call"),
                "qty":         pos.get("qty", 1),
                "underlying":  pos.get("underlying", symbol),
                "entry_dt":    entry_dt_iso,
            }
    try:
        with open(BOT_STATE_FILE, "w") as f:
            json.dump(state, f)
    except Exception:
        pass

# Real-time price cache populated by StockDataStream WebSocket.
# Falls back to REST if ticker not yet cached.
_price_cache: dict[str, float] = {}

# Symbols whose stop is breached but can't close yet (market closed). Tracked so we
# log the deferral ONCE per episode instead of every monitor cycle (30s). Cleared
# when the market reopens (the deferred closes then execute).
_deferred_closes: set[str] = set()

def _log_deferred_once(symbol: str, msg: str):
    if symbol not in _deferred_closes:
        log.warning(msg, symbol)
        _deferred_closes.add(symbol)
    else:
        log.debug(msg, symbol)

# ═══════════════════════════════════════════════════════════════════════════════
# CLIENTS
# ═══════════════════════════════════════════════════════════════════════════════

trading_client     = TradingClient(ALPACA_KEY, ALPACA_SECRET, paper=True)
option_data_client = OptionHistoricalDataClient(ALPACA_KEY, ALPACA_SECRET)
# Reused across all stock REST lookups (latest-trade on price-cache miss + daily regime
# bars). Module-level singleton so we don't build a fresh client (new HTTP/TLS session)
# on every cache-miss in get_stock_price — that handshake sat on the entry path.
stock_data_client  = StockHistoricalDataClient(ALPACA_KEY, ALPACA_SECRET)
ollama_client      = OpenAI(base_url=OLLAMA_BASE_URL, api_key="ollama")
# Groq scorer A/B (shadow only — never drives a trade). None unless keys are present.
groq_client = (OpenAI(base_url=GROQ_BASE_URL, api_key=GROQ_KEY)
               if (GROQ_SCORER_SHADOW_ENABLED and GROQ_BASE_URL and GROQ_KEY) else None)
# Hybrid scorer: Gemini frontier model as an ACTIVE confirm/veto gate (not shadow). None
# unless keys present — when None the bot runs pure-Ollama (graceful no-op).
gemini_client = (OpenAI(base_url=GEMINI_BASE_URL, api_key=GEMINI_KEY)
                 if (HYBRID_SCORER_ENABLED and GEMINI_BASE_URL and GEMINI_KEY) else None)
stock_stream       = StockDataStream(ALPACA_KEY, ALPACA_SECRET)

# ── Data-WebSocket reconnect hardening (workaround for an alpaca-py reconnect bug) ──────
# alpaca-py's _run_forever() reconnect loop, on a connect-time ValueError (e.g. Alpaca's
# "connection limit exceeded", surfaced when the connected-message check fails), neither
# closes the socket nor backs off (its `finally` is `sleep(0)`). Worse, _connect() overwrites
# self._ws WITHOUT closing the previous one — so every failed reconnect LEAKS a socket, they
# pile up against Alpaca's 1-connection-per-account limit, and it tight-loops (observed 186
# "connection limit exceeded" in a single session). We wrap the stock stream's _connect to
# (1) close any stale socket FIRST (stops the leak) and (2) apply exponential backoff so the
# old connection slot is released before we retry. No effect on the healthy path (one clean
# connect at startup → _ws is None, backoff 0). News stream is untouched (it never churns).
_stock_orig_connect = stock_stream._connect
_stock_reconnect_backoff = {"s": 0.0}
async def _stock_connect_safe():
    if getattr(stock_stream, "_ws", None) is not None:
        try:
            await stock_stream._ws.close()
        except Exception:
            pass
        stock_stream._ws = None
    b = _stock_reconnect_backoff["s"]
    if b:
        log.info("⏳ data-ws reconnect backoff %.0fs (avoiding connection-limit churn)", b)
        await asyncio.sleep(b)
    try:
        await _stock_orig_connect()
        _stock_reconnect_backoff["s"] = 0.0          # connected → reset backoff
    except Exception:
        _stock_reconnect_backoff["s"] = min(max(b * 2, 1.0), 30.0)
        raise
stock_stream._connect = _stock_connect_safe

# ═══════════════════════════════════════════════════════════════════════════════
# MARKET DATA HELPERS
# ═══════════════════════════════════════════════════════════════════════════════

def market_is_open() -> bool:
    """Fails to False on a clock/network error rather than propagating — a transient
    Alpaca ConnectionReset here once killed the whole trailing_stop_monitor (and the bot).
    False is the safe default: stop-close branches defer ("market closed, will retry") and
    re-check next cycle; the position stays protected. See near_market_close (same pattern)."""
    try:
        return trading_client.get_clock().is_open
    except Exception as e:
        log.warning("market_is_open clock error (%s) — assuming closed this cycle", e)
        return False

def near_market_close(within_min: int = TIME_STOP_EOD_WINDOW_MIN) -> bool:
    """True if the regular session is open AND within `within_min` of the close.
    Used to run time-stop exits at end-of-day (avoid the overnight gap) rather than
    intraday or carried to next morning. Fails to False on a clock error — the
    trailing stop still protects the position, and it'll exit at the next EOD window."""
    try:
        clk = trading_client.get_clock()
        if not clk.is_open:
            return False
        secs_to_close = (clk.next_close - clk.timestamp).total_seconds()
        return 0 <= secs_to_close <= within_min * 60
    except Exception:
        return False

# Regime cache: the 200-day SMA barely moves intraday, so compute once per day.
_regime_cache: dict = {"date": None, "uptrend": True}

def market_in_uptrend() -> bool:
    """
    True if the index (SPY) is above its REGIME_MA_DAYS-day SMA. Long calls are
    leveraged long-beta; the backtest shows gating on this keeps ~91% of bull
    profit while cutting bear-market losses ~53%. Cached once per calendar day.
    On any data error, default to True (fail-open — don't silently halt trading).
    """
    if not REGIME_FILTER_ENABLED:
        return True
    today = date.today()
    if _regime_cache["date"] == today:
        return _regime_cache["uptrend"]
    try:
        start = datetime.now(timezone.utc) - timedelta(days=int(REGIME_MA_DAYS * 1.6) + 30)
        resp = stock_data_client.get_stock_bars(StockBarsRequest(
            symbol_or_symbols=REGIME_INDEX, timeframe=TimeFrame.Day, start=start, feed="iex"))
        bars = (resp.data or {}).get(REGIME_INDEX, [])
        closes = [float(b.close) for b in bars]
        if len(closes) < REGIME_MA_DAYS:
            log.warning("Regime: only %d bars for %s (<%d) — defaulting to uptrend",
                        len(closes), REGIME_INDEX, REGIME_MA_DAYS)
            up = True
        else:
            sma = sum(closes[-REGIME_MA_DAYS:]) / REGIME_MA_DAYS
            above_sma = closes[-1] >= sma
            # Short-term momentum brake: SPY today must also be ≥ SPY REGIME_MOMENTUM_DAYS trading
            # days ago — pauses the long legs during multi-day pullbacks the slow 200d misses.
            mom_ok = (REGIME_MOMENTUM_DAYS <= 0 or len(closes) <= REGIME_MOMENTUM_DAYS
                      or closes[-1] >= closes[-1 - REGIME_MOMENTUM_DAYS])
            up = above_sma and mom_ok
            ret_nd = (closes[-1] / closes[-1 - REGIME_MOMENTUM_DAYS] - 1) * 100 if (
                REGIME_MOMENTUM_DAYS > 0 and len(closes) > REGIME_MOMENTUM_DAYS) else 0.0
            log.info("📐 Regime: %s last=$%.2f vs %dd SMA=$%.2f (%s) · %dd mom %+.1f%% (%s) → %s",
                     REGIME_INDEX, closes[-1], REGIME_MA_DAYS, sma,
                     "above" if above_sma else "below", REGIME_MOMENTUM_DAYS, ret_nd,
                     "ok" if mom_ok else "down", "UPTREND (trading)" if up else "PAUSED (calls off)")
    except Exception as e:
        log.warning("Regime check failed (%s) — defaulting to uptrend", e)
        up = True
    _regime_cache.update(date=today, uptrend=up)
    return up

def on_cooldown(ticker: str) -> bool:
    return (time.time() - _recent_trades.get(ticker, 0)) < COOLDOWN_SECS

async def handle_stock_trade(trade):
    # Update real-time price cache from StockDataStream trade events
    ticker = getattr(trade, 'symbol', None)
    price  = getattr(trade, 'price', None)
    if ticker and price:
        _price_cache[ticker] = float(price)

def subscribe_price_stream(tickers: list):
    # Subscribe the stock stream to tickers for real-time prices
    if not tickers:
        return
    stock_stream.subscribe_trades(handle_stock_trade, *tickers)
    log.info("📡 Price stream subscribed: %s", ", ".join(tickers[:5]) + ("…" if len(tickers) > 5 else ""))

def get_stock_price(ticker: str, subscribe: bool = False) -> Optional[float]:
    # subscribe defaults to False to avoid hitting the IEX 25-symbol stream limit.
    # The startup watchlist (subscribe_price_stream called once at boot) handles
    # real-time prices for the liquid names; everything else falls back to REST.
    # Check real-time WebSocket cache first (zero latency)
    cached = _price_cache.get(ticker)
    if cached:
        log.debug("💹 %s price from cache: $%.2f", ticker, cached)
        return cached
    # Cache miss — fetch via REST; only subscribe if requested (IEX limit is 25 symbols)
    log.debug("💹 %s not in cache, fetching via REST", ticker)
    try:
        resp   = stock_data_client.get_stock_latest_trade(StockLatestTradeRequest(symbol_or_symbols=ticker))
        price  = float(resp[ticker].price)
        _price_cache[ticker] = price
        if subscribe:
            subscribe_price_stream([ticker])
        return price
    except Exception as e:
        log.warning("Stock price fetch failed for %s: %s", ticker, e)
        return None

def get_option_quote(symbol: str) -> Optional[dict]:
    """Return {bid, ask, mid, spread_pct} or None if illiquid/unavailable."""
    try:
        resp = option_data_client.get_option_latest_quote(
            OptionLatestQuoteRequest(symbol_or_symbols=symbol)
        )
        q = resp[symbol]
        bid, ask = float(q.bid_price), float(q.ask_price)
        if ask <= 0 or bid < 0:
            return None
        mid = round((bid + ask) / 2, 4)
        spread_pct = (ask - bid) / mid if mid > 0 else 1.0
        return {"bid": bid, "ask": ask, "mid": mid, "spread_pct": spread_pct}
    except Exception as e:
        log.warning("Option quote failed for %s: %s", symbol, e)
        return None

def get_option_greeks(symbol: str) -> Optional[dict]:
    """Real delta/IV/greeks from Alpaca's option snapshot. None on failure."""
    try:
        snap = option_data_client.get_option_snapshot(OptionSnapshotRequest(symbol_or_symbols=symbol))
        s = snap[symbol]
        g = getattr(s, "greeks", None)
        return {
            "delta": getattr(g, "delta", None) if g else None,
            "gamma": getattr(g, "gamma", None) if g else None,
            "theta": getattr(g, "theta", None) if g else None,
            "vega":  getattr(g, "vega", None) if g else None,
            "rho":   getattr(g, "rho", None) if g else None,
            "iv":    getattr(s, "implied_volatility", None),
        }
    except Exception as e:
        log.debug("Option greeks fetch failed for %s: %s", symbol, e)
        return None

def log_contract_pick(ticker: str, stock_price: float, strategy: str, target_delta: float,
                      dte_target: int, contract: dict, qty: int):
    """Phase-0 (read-only) contract-selection probe — records, for every option trade,
    the FIDELITY of the pick (target vs actual delta/moneyness/DTE/spread/IV/liquidity)
    to contract_selection.csv. Diagnoses whether the bot hits its intended moneyness and
    how much spread its picks carry, BEFORE we change any selection logic. Trading-neutral."""
    try:
        g = get_option_greeks(contract["symbol"]) or {}
        expiry = datetime.strptime(contract["expiry"], "%Y-%m-%d").date()
        dte = (expiry - date.today()).days
        moneyness = (contract["strike"] / stock_price) if stock_price else None
        path = "contract_selection.csv"
        write_header = not os.path.exists(path)
        with open(path, "a", newline="") as f:
            w = csv.writer(f)
            if write_header:
                w.writerow(["timestamp", "ticker", "strategy", "stock_price",
                            "target_delta", "actual_delta", "target_dte", "actual_dte",
                            "symbol", "strike", "moneyness", "expiry",
                            "bid", "ask", "mid", "spread_pct", "open_interest",
                            "premium_per_contract", "qty", "cost", "implied_vol",
                            "gamma", "theta", "vega", "rho"])
            w.writerow([
                datetime.now(timezone.utc).isoformat(), ticker, strategy, f"{stock_price:.4f}",
                target_delta, g.get("delta", ""), dte_target, dte,
                contract["symbol"], contract["strike"],
                f"{moneyness:.4f}" if moneyness else "", contract["expiry"],
                contract["bid"], contract["ask"], contract["mid"], f"{contract['spread_pct']:.4f}",
                contract.get("open_interest", ""), f"{contract['ask']*100:.2f}", qty,
                f"{contract['ask']*100*qty:.2f}", g.get("iv", ""), g.get("gamma", ""),
                g.get("theta", ""), g.get("vega", ""), g.get("rho", ""),
            ])
    except Exception as e:
        log.debug("contract-pick probe failed: %s", e)

# ═══════════════════════════════════════════════════════════════════════════════
# GUARDRAILS
# ═══════════════════════════════════════════════════════════════════════════════

def get_market_cap_billions(ticker: str) -> Optional[float]:
    """Fetch market cap via Alpaca asset info. Returns billions or None."""
    # ETFs are always allowed through — they have no traditional market cap
    if ticker in TRADEABLE_ETFS:
        return 999.0
    try:
        asset = trading_client.get_asset(ticker)
        # alpaca-py doesn't expose market cap directly; use a simple yfinance-free
        # fallback: trust the asset is tradeable on a major exchange as a proxy.
        # For a real filter, integrate a free data source like FMP or yfinance.
        # Here we return None to trigger a skip only if the asset is not found.
        if asset and asset.exchange in ("NYSE", "NASDAQ", "AMEX"):
            return None   # signal: check not possible, allow through with warning
        return 0.0
    except Exception:
        return 0.0

def _loss_reset_if_new_day():
    """Reset the daily-loss breaker at each new UTC calendar day. Without this the counter
    accumulated forever and the halt latched permanently until a process restart — i.e. the
    'daily' limit was neither daily nor self-clearing (fixed 2026-06-18)."""
    global _daily_loss_usd, _trading_halted, _loss_day
    today = datetime.now(timezone.utc).date()
    if _loss_day != today:
        if _loss_day is not None and (_daily_loss_usd or _trading_halted):
            log.info("🔄 New UTC day — resetting daily-loss breaker (was $%.0f, halted=%s)",
                     _daily_loss_usd, _trading_halted)
        _loss_day = today
        _daily_loss_usd = 0.0
        _trading_halted = False


def _account_equity() -> float | None:
    """Account equity, cached for _EQUITY_TTL secs (the per-signal guardrail must not hammer
    the API). Returns None on failure so callers can fall back to a safe absolute limit."""
    now = time.time()
    if _equity_cache["val"] is not None and now - _equity_cache["ts"] < _EQUITY_TTL:
        return _equity_cache["val"]
    try:
        eq = float(trading_client.get_account().equity)
        _equity_cache.update(val=eq, ts=now)
        return eq
    except Exception as e:
        log.debug("equity fetch for loss-limit failed: %s", e)
        return _equity_cache["val"]   # last-known (may be None) — caller handles None


def _daily_loss_limit() -> float:
    """Dynamic daily-loss limit = DAILY_LOSS_LIMIT_PCT × equity, with the static
    DAILY_LOSS_LIMIT as a fail-safe floor when equity is unavailable."""
    eq = _account_equity()
    return eq * DAILY_LOSS_LIMIT_PCT if eq else float(DAILY_LOSS_LIMIT)


def passes_guardrails(ticker: str, quote: dict, skip_halt: bool = False) -> bool:
    """Return True if the ticker+contract passes all pre-trade safety checks.
    skip_halt=True bypasses the daily-loss circuit breaker — used by the SHADOW path so
    it still records what a strategy WOULD pick even when real trading is halted/capped."""
    global _trading_halted

    # Daily loss circuit breaker (real trades only; shadow observes regardless)
    if not skip_halt:
        _loss_reset_if_new_day()
        limit = _daily_loss_limit()
        if _trading_halted:
            log.warning("🛑 Trading halted — daily loss limit reached ($%.0f)", limit)
            return False
        if _daily_loss_usd >= limit:
            _trading_halted = True
            log.warning("🛑 Daily loss limit hit ($%.0f ≥ $%.0f, %.1f%% of equity). "
                        "Halting new trades until next UTC day.",
                        _daily_loss_usd, limit, DAILY_LOSS_LIMIT_PCT * 100)
            return False

    # Bid/ask spread filter
    if quote["spread_pct"] > MAX_SPREAD_PCT:
        log.info(
            "  → spread too wide: %.1f%% > %.0f%% max — skipping (illiquid contract)",
            quote["spread_pct"] * 100, MAX_SPREAD_PCT * 100,
        )
        return False

    return True


# Short-TTL cache of Yahoo prices so the stock+pead+lotto legs on the same ticker
# (and repeated signals) don't each hit Yahoo → avoids rate limits.
_yahoo_px_cache: dict[str, tuple] = {}   # ticker -> (price, ts)

def _yahoo_latest_price(ticker: str, ttl: float = 90.0) -> Optional[float]:
    """Yahoo's latest price for a cross-check, cached for `ttl` seconds. Uses
    _fetch_yahoo directly (in-memory, NO disk cache → no contention with backtests).
    Returns None if unavailable (caller fails open)."""
    now = time.time()
    hit = _yahoo_px_cache.get(ticker)
    if hit and now - hit[1] < ttl:
        return hit[0]
    try:
        from yahoo_data import _fetch_yahoo, _parse
        # 2-day lookback (was 7d): this guardrail only runs during RTH, where Yahoo's
        # 1d endpoint already returns TODAY's live/partial bar — so 2 days captures
        # today's price plus yesterday as a fallback, with no weekend/holiday gap. We
        # don't go to 1d: it leaves no cushion at the open if today's partial bar is
        # momentarily absent. (Latency here is round-trip-bound, not payload-bound —
        # this is a safe cleanup; the real hot-path win would be running it concurrently.)
        series = _parse(_fetch_yahoo(ticker, int(now) - 86400 * 2, int(now) + 86400, tries=2))
        yp = series[max(series)][3] if series else None   # latest day's close
    except Exception as e:
        log.debug("Yahoo cross-check fetch failed for %s: %s", ticker, e)
        yp = None
    if yp and yp > 0:
        _yahoo_px_cache[ticker] = (yp, now)
    return yp

def passes_stock_price_guardrail(ticker: str, stock_price: float) -> bool:
    if stock_price < MIN_STOCK_PRICE:
        log.info("  → %s @ $%.2f below MIN_STOCK_PRICE $%.2f — skipping", ticker, stock_price, MIN_STOCK_PRICE)
        return False
    # Cross-check Alpaca's price against Yahoo (second source). Skip on gross
    # disagreement (suspected bad data); fail-OPEN if Yahoo is unavailable.
    if PRICE_CROSSCHECK_ENABLED:
        yp = _yahoo_latest_price(ticker)
        if yp and abs(stock_price / yp - 1) > PRICE_CROSSCHECK_TOL:
            log.warning("  → %s PRICE MISMATCH: Alpaca $%.2f vs Yahoo $%.2f (%.0f%% off) — "
                        "skipping (suspected bad data)", ticker, stock_price, yp,
                        abs(stock_price / yp - 1) * 100)
            return False
    return True

# ═══════════════════════════════════════════════════════════════════════════════
# OPTIONS — CONTRACT PICKER & EXECUTION
# ═══════════════════════════════════════════════════════════════════════════════

def pick_call_contract(ticker: str, stock_price: float,
                       dte_min: int = MIN_DAYS_TO_EXPIRY,
                       dte_max: int = MAX_DAYS_TO_EXPIRY,
                       target_delta: float = TARGET_DELTA,
                       strike_hi_mult: float = 1.10,
                       for_shadow: bool = False) -> Optional[dict]:
    today   = date.today()
    min_exp = today + timedelta(days=dte_min)
    max_exp = today + timedelta(days=dte_max)

    try:
        resp = trading_client.get_option_contracts(
            GetOptionContractsRequest(
                underlying_symbols=[ticker],
                type=ContractType.CALL,
                expiration_date_gte=min_exp,
                expiration_date_lte=max_exp,
                strike_price_gte=str(round(stock_price * 0.95, 2)),
                strike_price_lte=str(round(stock_price * strike_hi_mult, 2)),
            )
        )
    except Exception as e:
        log.warning("Contract lookup failed for %s: %s", ticker, e)
        return None

    contracts = getattr(resp, "option_contracts", [])
    if not contracts:
        log.info("  → no contracts found for %s in range", ticker)
        return None

    # Sort: prefer the configured DTE target (mid of the expiry window), then
    # strike closest to target. Lower target_delta → strike further OTM (cheaper).
    dte_target = round((dte_min + dte_max) / 2)
    target_strike = stock_price * (1 + (1 - target_delta) * 0.15)
    contracts.sort(key=lambda c: (
        abs((datetime.strptime(str(c.expiration_date), "%Y-%m-%d").date() - today).days - dte_target),
        abs(float(c.strike_price) - target_strike),
    ))

    # Walk down sorted list until we find one that passes liquidity checks
    for c in contracts[:5]:
        # Skip untradable or non-standard/adjusted contracts (e.g. "FDX1…" from a
        # corporate action) — their root differs from the ticker and orders on
        # them are rejected ("contract is not tradable").
        if getattr(c, "tradable", True) is False:
            continue
        root = getattr(c, "root_symbol", None)
        if root and root != ticker:
            log.info("  → %s is a non-standard/adjusted contract (root %s) — skipping", c.symbol, root)
            continue
        quote = get_option_quote(c.symbol)
        if quote is None:
            continue
        # Data-sanity (Alpaca option quotes are unreliable): a call must be worth at
        # least its intrinsic value (stock − strike) and no more than the stock price.
        # Catches garbage quotes disconnected from the underlying — e.g. the SNDK $90
        # mid on a contract carrying ~$1,385 of intrinsic value.
        intrinsic = max(0.0, stock_price - float(c.strike_price))
        if quote["mid"] <= 0 or quote["mid"] < intrinsic * 0.9 or quote["mid"] > stock_price:
            log.info("  → %s quote $%.2f inconsistent w/ stock $%.2f strike $%.2f (intrinsic $%.2f) — skipping (bad data)",
                     c.symbol, quote["mid"], stock_price, float(c.strike_price), intrinsic)
            continue
        oi = getattr(c, "open_interest", None)
        if oi is not None and int(oi) < MIN_OPEN_INTEREST:
            log.info("  → %s OI=%s below min %d, skipping", c.symbol, oi, MIN_OPEN_INTEREST)
            continue
        if not passes_guardrails(ticker, quote, skip_halt=for_shadow):
            continue
        return {
            "symbol": c.symbol,
            "strike": float(c.strike_price),
            "expiry": str(c.expiration_date),
            "bid":    quote["bid"],
            "ask":    quote["ask"],
            "mid":    quote["mid"],
            "spread_pct": quote["spread_pct"],
            "open_interest": int(oi) if oi is not None else None,
        }

    log.info("  → no liquid contracts found for %s after guardrail checks", ticker)
    return None

def pick_closest_oow_contract(ticker: str, stock_price: float, dte_min: int, dte_max: int,
                              target_delta: float, strike_hi_mult: float):
    """OUT-OF-WINDOW fallback (paper shadow only). When [dte_min,dte_max] has NO contracts,
    query a wider expiry range and pick the liquid contract on the expiry CLOSEST to the
    window. Returns (contract_dict, actual_dte) or None. Returns None if the ticker is truly
    optionless OR if contracts DO exist inside the window (not an empty-window case → the
    real path handles it). Mirrors pick_call_contract's liquidity walk; for_shadow semantics
    (skips the daily-loss halt). Never used by the real-money path."""
    today = date.today()
    try:
        resp = trading_client.get_option_contracts(GetOptionContractsRequest(
            underlying_symbols=[ticker], type=ContractType.CALL,
            expiration_date_gte=today + timedelta(days=1),
            expiration_date_lte=today + timedelta(days=LOTTO_OOW_MAX_DTE),
            strike_price_gte=str(round(stock_price * 0.95, 2)),
            strike_price_lte=str(round(stock_price * strike_hi_mult, 2)),
        ))
    except Exception as e:
        log.debug("OOW contract lookup failed for %s: %s", ticker, e)
        return None
    contracts = getattr(resp, "option_contracts", []) or []
    if not contracts:
        return None                                  # truly optionless (e.g. ELLI, STAK)

    def dte_of(c):
        return (datetime.strptime(str(c.expiration_date), "%Y-%m-%d").date() - today).days

    # If anything is INSIDE the window this isn't an empty-window case — defer to the real path.
    if any(dte_min <= dte_of(c) <= dte_max for c in contracts):
        return None

    # All candidates are outside the window: prefer the closest expiry, then strike nearest
    # the lotto target (same heuristic as pick_call_contract).
    target_strike = stock_price * (1 + (1 - target_delta) * 0.15)
    def dist_to_window(d):
        return max(dte_min - d, d - dte_max, 0)
    contracts.sort(key=lambda c: (dist_to_window(dte_of(c)),
                                  abs(float(c.strike_price) - target_strike)))
    for c in contracts[:5]:
        if getattr(c, "tradable", True) is False:
            continue
        root = getattr(c, "root_symbol", None)
        if root and root != ticker:
            continue
        quote = get_option_quote(c.symbol)
        if quote is None:
            continue
        intrinsic = max(0.0, stock_price - float(c.strike_price))
        if quote["mid"] <= 0 or quote["mid"] < intrinsic * 0.9 or quote["mid"] > stock_price:
            continue
        oi = getattr(c, "open_interest", None)
        if oi is not None and int(oi) < MIN_OPEN_INTEREST:
            continue
        if not passes_guardrails(ticker, quote, skip_halt=True):   # shadow → bypass halt
            continue
        return ({
            "symbol": c.symbol, "strike": float(c.strike_price),
            "expiry": str(c.expiration_date), "bid": quote["bid"], "ask": quote["ask"],
            "mid": quote["mid"], "spread_pct": quote["spread_pct"],
            "open_interest": int(oi) if oi is not None else None,
        }, dte_of(c))
    return None

def place_option_trade(ticker: str, stock_price: float, signal: dict, position_usd: float = MAX_POSITION_USD,
                       strategy: str = "news_call", dte_min: int = MIN_DAYS_TO_EXPIRY,
                       dte_max: int = MAX_DAYS_TO_EXPIRY, target_delta: float = TARGET_DELTA,
                       strike_hi_mult: float = 1.10,
                       budget_mult: float = MAX_CONTRACT_BUDGET_MULT):
    contract = pick_call_contract(ticker, stock_price, dte_min, dte_max, target_delta, strike_hi_mult)
    if not contract:
        return None

    # Dedupe: never open a second position in a contract we already hold/monitor.
    # A repeat signal on the same ticker (past the cooldown) re-picks the same
    # contract; re-buying it triggers Alpaca "potential wash trade" rejections.
    if contract["symbol"] in _monitored_positions:
        log.info("  → already holding %s — skipping duplicate buy [%s]", contract["symbol"], strategy)
        return None

    # Budget guard: if even ONE contract costs more than budget_mult × the budget, SKIP.
    # Without it, the max(1,...) floor below forces a single contract no matter the price
    # (bought a $12,201 lotto contract on a $250 budget). lotto passes budget_mult=1.0 →
    # a HARD $250 cap; other option strategies use the backtest's 3× (MAX_CONTRACT_BUDGET_MULT).
    contract_cost = contract["ask"] * 100
    if contract_cost > position_usd * budget_mult:
        log.info("  → skipping %s [%s]: 1 contract $%.0f > %.1f× budget $%.0f (would oversize)",
                 contract["symbol"], strategy, contract_cost, budget_mult, position_usd)
        return None

    # Size by scaled position budget, paying ask price (realistic fill)
    qty = max(1, int(position_usd / contract_cost))

    log.info(
        "  📋 %s  strike=$%.2f  exp=%s  bid=$%.4f  ask=$%.4f  mid=$%.4f  spread=%.1f%%  qty=%d",
        contract["symbol"], contract["strike"], contract["expiry"],
        contract["bid"], contract["ask"], contract["mid"],
        contract["spread_pct"] * 100, qty,
    )

    # Use a limit order at ask — avoids "no quote" rejection on market orders
    req = LimitOrderRequest(
        symbol=contract["symbol"],
        qty=qty,
        side=OrderSide.BUY,
        time_in_force=TimeInForce.DAY,
        limit_price=round(contract["ask"], 2),
        position_intent=PositionIntent.BUY_TO_OPEN,
    )
    try:
        order = trading_client.submit_order(req)
        max_loss = contract["ask"] * 100 * qty
        log.info(
            "✅ OPTION BUY  %s x%d  limit=$%.4f  cost_basis=$%.4f/contract  max_loss=$%.0f",
            contract["symbol"], qty, contract["ask"], contract["ask"], max_loss,
        )
    except Exception as e:
        log.error("Option buy failed: %s", e)
        return None

    # Protection is the IN-PROCESS monitor, not an exchange-held stop. This
    # account cannot hold uncovered option sell-stops (every place_stop_order is
    # rejected with code 40310000), and placing one right after the buy — before
    # it fills — also triggered "sell_to_open intent mismatch" + "wash trade"
    # rejections (29 + 7 in one session). So we DON'T place a resting stop; the
    # monitor watches price each cycle and exits via close_option_position (the
    # working stop-limit-above-market method) when the trail is breached.
    entry          = contract["ask"]
    init_trail     = long_trail_for(strategy, "option", 0.0)
    _monitored_positions[contract["symbol"]] = {
        "qty":           qty,
        "entry_price":   entry,
        "peak_price":    entry,
        "underlying":    ticker,
        "asset_type":    "option",
        "stop_order_id": None,            # no exchange stop — monitor protects
        "strategy":      strategy,
        "entry_dt":      datetime.now(timezone.utc),
    }
    log.info("🔒 In-process trailing stop registered for %s @ cost $%.4f  (trail %.0f%%)  [%s]",
             contract["symbol"], entry, init_trail * 100, strategy)

    _recent_trades[ticker] = time.time()
    log_trade(ticker, contract, qty, signal, contract["ask"], strategy=strategy)

    # Phase-0 contract-selection probe (read-only; logs pick fidelity, no behavior
    # change). Moved AFTER submit_order so its get_option_greeks round-trip never sits
    # on the critical entry path — it's research telemetry, not part of the order.
    log_contract_pick(ticker, stock_price, strategy, target_delta,
                      (dte_min + dte_max) // 2, contract, qty)

    # Shadow materiality A/B record (lotto only) — keyed by option symbol so it joins
    # to closed_trades.csv for realized-P&L attribution. Own file → no schema risk.
    # Scored by execute_lotto_trade_fn AFTER this order is placed (off the entry path),
    # then stashed on signal["_materiality"] and logged via _log_lotto_materiality there.
    return contract

def _news_call_count_today(ticker: str) -> int:
    """How many news_call entries we've already logged for `ticker` today (UTC). Durable across
    restarts (reads trades.csv) → backs the per-ticker daily cap that stops same-story echo-buys."""
    try:
        today = datetime.now(timezone.utc).date().isoformat()
        n = 0
        with open("trades.csv", newline="") as f:
            for r in csv.DictReader(f):
                if (r.get("strategy") == "news_call" and r.get("source_ticker") == ticker
                        and (r.get("timestamp") or "")[:10] == today):
                    n += 1
        return n
    except Exception:
        return 0


def execute_equity_trade(ticker: str, signal: dict, option_usd: float, strategy: str = "news_call"):
    """
    Full synchronous equity-options path: price lookup → guardrail → place trade.
    Runs in a worker thread (off the event loop) so blocking Alpaca/websocket
    calls — including dynamic price-stream subscription on a cache miss — can
    never freeze the bot. Wrapped with a timeout by the caller.
    """
    if _at_position_cap(strategy):
        return
    # Per-ticker daily cap (news_call): don't re-buy the same name on echoed/republished news.
    if strategy == "news_call":
        n_today = _news_call_count_today(ticker)
        if n_today >= NEWS_CALL_MAX_PER_TICKER_DAY:
            log.info("  → news_call SKIP %s: already %d trade(s) today (per-ticker cap %d) — likely echoed news",
                     ticker, n_today, NEWS_CALL_MAX_PER_TICKER_DAY)
            return
    stock_price = get_stock_price(ticker)
    if not stock_price:
        return
    if not passes_stock_price_guardrail(ticker, stock_price):
        return
    log.info("  → %s @ $%.2f — searching for call contract  size=$%.0f  [%s]",
             ticker, stock_price, option_usd, strategy)
    t0 = datetime.now(timezone.utc)
    # Strategy-specific option geometry (each its own sweep optimum; other equity-option
    # callers fall through to place_option_trade's generic defaults DTE 14-21 / Δ0.50):
    #   qqq_macro → ~DTE10 / Δ0.60 (higher-delta liquid-index macro call)
    #   news_call → ~DTE10 / Δ0.40 (cheap short-dated slightly-OTM convex call; geometry sweep)
    opt_kw = {}
    if strategy == "qqq_macro":
        opt_kw = dict(dte_min=QQQ_MACRO_DTE_MIN, dte_max=QQQ_MACRO_DTE_MAX,
                      target_delta=QQQ_MACRO_TARGET_DELTA)
    elif strategy == "news_call":
        opt_kw = dict(dte_min=NEWS_CALL_DTE_MIN, dte_max=NEWS_CALL_DTE_MAX,
                      target_delta=NEWS_CALL_TARGET_DELTA, budget_mult=NEWS_CALL_BUDGET_MULT)
    place_option_trade(ticker, stock_price, signal, option_usd, strategy=strategy, **opt_kw)
    log.info("⏱  Trade path completed in %.0fms  [%s %s]",
             (datetime.now(timezone.utc) - t0).total_seconds() * 1000, ticker, strategy)

# ── STOCK EXECUTION ──────────────────────────────────────────────────────────

def place_stock_trade(ticker: str, stock_price: float, signal: dict,
                      position_usd: float = MAX_POSITION_USD, strategy: str = "stock"):
    """Buy stock and register with monitor. Key = ticker__strategy (composite)."""
    shares = max(1, int(position_usd / stock_price))
    try:
        trading_client.submit_order(MarketOrderRequest(
            symbol=ticker, qty=shares, side=OrderSide.BUY, time_in_force=TimeInForce.DAY))
        log.info("✅ STOCK BUY  %s x%d @ ~$%.2f  cost~$%.0f  [%s]",
                 ticker, shares, stock_price, shares * stock_price, strategy)
    except Exception as e:
        # During a trading halt/auction Alpaca rejects MARKET orders and asks for
        # a limit. Retry once with a marketable limit (1% through the price).
        if "halt" in str(e).lower() or "limit order" in str(e).lower():
            try:
                trading_client.submit_order(LimitOrderRequest(
                    symbol=ticker, qty=shares, side=OrderSide.BUY, time_in_force=TimeInForce.DAY,
                    limit_price=round(stock_price * 1.01, 2)))
                log.info("✅ STOCK BUY (limit, halt fallback)  %s x%d  limit=$%.2f  [%s]",
                         ticker, shares, round(stock_price * 1.01, 2), strategy)
            except Exception as e2:
                log.error("Stock buy failed for %s (market+limit): %s", ticker, e2)
                return None
        else:
            log.error("Stock buy failed for %s: %s", ticker, e)
            return None

    key = f"{ticker}__{strategy}"
    _monitored_positions[key] = {
        "qty":           shares,
        "entry_price":   stock_price,
        "peak_price":    stock_price,
        "underlying":    ticker,
        "asset_type":    "stock",
        "stop_order_id": None,
        "strategy":      strategy,
        "entry_dt":      datetime.now(timezone.utc),
    }
    trail0 = long_trail_for(strategy, "stock", 0.0)
    log.info("🔒 Stock trailing stop for %s @ $%.2f  stop=$%.2f (%.0f%% initial)  [%s]",
             ticker, stock_price, stock_price * (1 - trail0), trail0 * 100, strategy)
    _recent_trades[ticker] = time.time()
    log_trade_stock(ticker, shares, stock_price, signal, strategy=strategy)
    return {"ticker": ticker, "shares": shares, "price": stock_price}


def _await_fill_price(order, timeout_s: float = 3.0) -> Optional[float]:
    """Poll a just-submitted order for its ACTUAL average fill price.
    Returns the filled_avg_price (float) once filled, or None if it hasn't filled
    within timeout_s (caller falls back to the mid-price estimate). Market orders fill
    almost instantly; option stop-limits that trigger immediately fill fast too. The
    rare not-filled case (illiquid near-$0 option resting order) falls back to mid."""
    try:
        oid = order.id
    except Exception:
        return None
    deadline = time.time() + timeout_s
    while time.time() < deadline:
        try:
            o = trading_client.get_order_by_id(oid)
            fap = getattr(o, "filled_avg_price", None)
            if fap is not None and float(fap) > 0:
                return float(fap)
            if any(x in str(getattr(o, "status", "")).lower()
                   for x in ("canceled", "rejected", "expired")):
                return None
        except Exception:
            pass
        time.sleep(0.3)
    return None

def _realized_pnl(asset_type: str, entry: float, exit_price: float, qty: float) -> float:
    """Realized $ P&L for a closed position, by asset type."""
    if asset_type == "stock":
        return (exit_price - entry) * qty
    if asset_type == "stock_short":
        return (entry - exit_price) * qty          # profit when covered lower
    return (exit_price - entry) * 100 * qty         # option (100 shares/contract)

def close_stock_position(key: str, qty: int, reason: str = "trailing stop"):
    """Close a stock position. key = ticker__strategy; sends sell order for just ticker.
    Returns (submitted_ok, actual_fill_price | None)."""
    ticker = key.split("__")[0]
    req = MarketOrderRequest(
        symbol=ticker, qty=qty, side=OrderSide.SELL,
        time_in_force=TimeInForce.DAY,
    )
    try:
        order = trading_client.submit_order(req)
        fill  = _await_fill_price(order)
        log.info("✅ STOCK SELL  %s x%d  (%s)%s", ticker, qty, reason,
                 f"  fill=${fill:.4f}" if fill else "  (fill pending → mid fallback)")
        return True, fill
    except Exception as e:
        log.error("Stock close failed for %s: %s — will retry next cycle", ticker, e)
        return False, None


def execute_stock_trade_fn(ticker: str, signal: dict, position_usd: float, strategy: str = "stock"):
    """Sync wrapper for stock buy path. Runs in executor off the event loop."""
    if _at_position_cap(strategy):
        # pead overflow (reserved sub-cap hit) → paper-track instead of dropping the signal.
        if strategy == "pead" and PEAD_SHADOW_ENABLED:
            shadow_pead_fn(ticker, signal, position_usd)
        return
    # ── Stock selectivity gate (strategy=="stock" ONLY) ──────────────────────────
    # The always-on stock leg floods with mediocre names if it fires on every passing
    # bullish signal ($4/tr). CONFIDENCE is the discriminator (selectivity sweep +
    # unified_v1 eval): conf≥0.85 ~triples per-trade quality. Gated here (not pead —
    # pead reuses this fn with strategy="pead" but has its own earnings gate; and NOT
    # the shared MIN_MAGNITUDE). Demoted signals just don't open a stock position.
    if strategy == "stock":
        conf = float(signal.get("confidence", 0))
        mag  = float(signal.get("magnitude", 0))
        if conf < STOCK_MIN_CONFIDENCE or mag < STOCK_MIN_MAGNITUDE:
            log.info("  → stock SKIP %s: conf %.2f<%.2f or mag %.2f<%.2f [stock selectivity gate]",
                     ticker, conf, STOCK_MIN_CONFIDENCE, mag, STOCK_MIN_MAGNITUDE)
            return
    stock_price = get_stock_price(ticker)
    if not stock_price:
        return
    if not passes_stock_price_guardrail(ticker, stock_price):
        return
    key = f"{ticker}__{strategy}"
    if key in _monitored_positions:
        log.info("  → already holding %s position in %s, skipping duplicate", strategy, ticker)
        return
    log.info("  → %s @ $%.2f — buying stock  size=$%.0f  [%s]",
             ticker, stock_price, position_usd, strategy)
    place_stock_trade(ticker, stock_price, signal, position_usd, strategy=strategy)


def execute_pead_trade_fn(ticker: str, signal: dict, position_usd: float):
    """PEAD: same stock buy but tiered-trail, only fires on earnings-signal context."""
    execute_stock_trade_fn(ticker, signal, position_usd, strategy="pead")


def execute_lotto_trade_fn(ticker: str, signal: dict, headline: str = "", body: str = ""):
    """Lotto: cheap OTM short-dated call on a very-high-conviction bullish signal.
    Small fixed size, OTM strike, wide let-it-run trail. Runs off the event loop."""
    stock_price = get_stock_price(ticker)
    if not stock_price:
        return
    if not passes_stock_price_guardrail(ticker, stock_price):
        return
    if f"{ticker}__lotto" in _monitored_positions or any(
            p.get("strategy") == "lotto" and p.get("underlying") == ticker
            for p in _monitored_positions.values()):
        log.info("  → already holding lotto on %s, skipping duplicate", ticker)
        return
    # Position cap hit → SHADOW the lotto pick (paper, no capital) so the high-conviction
    # signal is still captured for forward data instead of being silently dropped.
    if _at_position_cap("lotto"):
        shadow_lotto_fn(ticker, stock_price)
        return

    mag = float(signal.get("magnitude", 0))
    conf = float(signal.get("confidence", 0))
    # ── MATERIALITY GATE (promoted 2026-06-10 from shadow A/B → the live lotto SELECTOR) ──
    # lotto_pnl_by_scorer showed materiality_fewshot is the dominant lotto-selection lever
    # (+$8,934 vs money-losing alternatives on the same candidates); materiality A/B backtest
    # +11% P&L / Sharpe 2.39→3.03. Lotto now fires ONLY if materiality ≥ gate. Signals that
    # pass the old mag/conf gate but FAIL materiality are DEMOTED to shadow logging (paper) for
    # the ongoing forward A/B. Scored BEFORE the order so it can gate (+~2.5s entry latency, the
    # deliberate cost of better selection). FAIL-OPEN: mat None (scoring glitch) → trade anyway.
    # Materiality 2nd pass DROPPED 2026-06-13 (MATERIALITY_GATE_ENABLED=False): unified_v1's
    # own magnitude self-rejects analyst noise, so the separate gate became vestigial in the
    # eval. When disabled we SKIP the score_materiality call entirely — no redundant ~2.5s 2nd
    # LLM call. mat stays None → log_lotto_materiality records n/a (telemetry only).
    mat = score_materiality(headline, body) if MATERIALITY_GATE_ENABLED else None
    if MATERIALITY_GATE_ENABLED and mat is not None and mat < MATERIALITY_GATE:
        log.info("  → lotto SKIP %s: materiality %.3f < gate %.2f → demoted to shadow [materiality gate]",
                 ticker, mat, MATERIALITY_GATE)
        shadow_lotto_fn(ticker, stock_price)                 # shadow the "old logic" would-be-trade
        log_lotto_materiality(ticker, "(gated_out)", mat, mag, conf)
        return

    log.info("  → %s @ $%.2f — buying LOTTO OTM call  size=$%.0f  Δ~%.2f  materiality=%s  [lotto]",
             ticker, stock_price, LOTTO_POSITION_USD, LOTTO_TARGET_DELTA,
             f"{mat:.3f}≥{MATERIALITY_GATE:.2f}" if mat is not None else "n/a(fail-open)")
    contract = place_option_trade(ticker, stock_price, signal, LOTTO_POSITION_USD, strategy="lotto",
                       dte_min=LOTTO_MIN_DTE, dte_max=LOTTO_MAX_DTE,
                       target_delta=LOTTO_TARGET_DELTA, strike_hi_mult=LOTTO_STRIKE_HI_MULT,
                       budget_mult=1.0)   # HARD $250 cap — lotto never exceeds its budget
    if not contract:
        # No tradeable contract inside the 8-21 DTE window. If the window is genuinely EMPTY
        # (truly optionless, or the nearest expiry is outside it), shadow the closest-to-window
        # contract (paper) so we capture the name instead of silently forfeiting it.
        shadow_lotto_oow_fn(ticker, stock_price)
        return

    # Record the (gate-passing) materiality against the chosen contract for realized-P&L attribution.
    sym = contract["symbol"]
    if sym in _monitored_positions:
        _monitored_positions[sym]["materiality"] = mat
    log_lotto_materiality(ticker, sym, mat, mag, conf)


# ── BEAR-SHORT EXECUTION ──────────────────────────────────────────────────────

def place_short_trade(ticker: str, stock_price: float, signal: dict,
                      position_usd: float = MAX_POSITION_USD, strategy: str = "bear_short"):
    """Sell short the stock; monitor tracks trough and stops on a rally."""
    shares = max(1, int(position_usd / stock_price))
    req = MarketOrderRequest(
        symbol=ticker, qty=shares, side=OrderSide.SELL,
        time_in_force=TimeInForce.DAY,
    )
    try:
        trading_client.submit_order(req)
        log.info("✅ SHORT SELL  %s x%d @ ~$%.2f  exposure~$%.0f  [%s]",
                 ticker, shares, stock_price, shares * stock_price, strategy)
    except Exception as e:
        log.error("Short sell failed for %s: %s", ticker, e)
        return None

    key = f"{ticker}__{strategy}"
    strail = short_trail_for(strategy)
    initial_stop = stock_price * (1 + strail)
    _monitored_positions[key] = {
        "qty":           shares,
        "entry_price":   stock_price,
        "trough_price":  stock_price,
        "underlying":    ticker,
        "asset_type":    "stock_short",
        "stop_order_id": None,
        "strategy":      strategy,
        "entry_dt":      datetime.now(timezone.utc),
    }
    log.info("🔒 Short stop for %s @ $%.2f  stop=$%.2f (%.0f%% trail)  [%s]",
             ticker, stock_price, initial_stop, strail * 100, strategy)
    _recent_trades[ticker] = time.time()
    log_trade_stock(ticker, shares, stock_price, signal, strategy=strategy)
    return {"ticker": ticker, "shares": shares, "price": stock_price}


def close_short_position(key: str, qty: int, reason: str = "trailing stop"):
    """Buy to cover. key = ticker__strategy. Returns (submitted_ok, actual_fill_price | None)."""
    ticker = key.split("__")[0]
    req = MarketOrderRequest(
        symbol=ticker, qty=qty, side=OrderSide.BUY,
        time_in_force=TimeInForce.DAY,
    )
    try:
        order = trading_client.submit_order(req)
        fill  = _await_fill_price(order)
        log.info("✅ SHORT COVER  %s x%d  (%s)%s", ticker, qty, reason,
                 f"  fill=${fill:.4f}" if fill else "  (fill pending → mid fallback)")
        return True, fill
    except Exception as e:
        log.error("Short cover failed for %s: %s — will retry next cycle", ticker, e)
        return False, None


def execute_bear_short_fn(ticker: str, signal: dict, position_usd: float):
    """Sync wrapper for short-sell path. Runs in executor off the event loop."""
    if _at_position_cap("bear_short"):
        return
    stock_price = get_stock_price(ticker)
    if not stock_price:
        return
    if not passes_stock_price_guardrail(ticker, stock_price):
        return
    key = f"{ticker}__bear_short"
    if key in _monitored_positions:
        log.info("  → already holding bear_short on %s, skipping duplicate", ticker)
        return
    log.info("  → %s @ $%.2f — shorting stock  size=$%.0f  [bear_short]",
             ticker, stock_price, position_usd)
    place_short_trade(ticker, stock_price, signal, position_usd, strategy="bear_short")


# ── L/S PAIRS EXECUTION ────────────────────────────────────────────────────────

def _first_tradeable_ticker(signal: dict) -> Optional[str]:
    """Top ticker from a signal that isn't crypto. None if no usable ticker."""
    for tk in signal.get("tickers", [])[:2]:
        if tk and tk not in ("BTC", "ETH"):
            return tk
    return None

def _pairs_reset_if_new_day():
    today = datetime.now(timezone.utc).date()
    if _pairs_day["date"] != today:
        _pairs_day.update(date=today, bull=None, bear=None, entered=False)

def record_pairs_signal(side: str, signal: dict, mag: float, conf: float):
    """Track the day's highest-scoring (mag×conf) bull and bear signal.
    side ∈ {"bull","bear"}. Only the top scorer per side is retained."""
    if not PAIRS_ENABLED:
        return
    ticker = _first_tradeable_ticker(signal)
    if not ticker:
        return
    _pairs_reset_if_new_day()
    score = mag * conf
    cur = _pairs_day[side]
    if cur is None or score > cur["score"]:
        _pairs_day[side] = {"ticker": ticker, "signal": signal,
                            "mag": mag, "conf": conf, "score": score}

def enter_pairs_trade(bull: dict, bear: dict, size: float):
    """Open both legs: long the bull name, short the bear name, equal $ size.
    Each leg independently checked against the position cap so a near-full book
    doesn't silently open only one leg. Either leg skipped if already held or
    price/guardrail fails; a single surviving leg is acceptable."""
    long_ticker, short_ticker = bull["ticker"], bear["ticker"]

    long_key = f"{long_ticker}__pairs_long"
    if long_key in _monitored_positions:
        log.info("  → already holding pairs_long %s — skip long leg", long_ticker)
    elif _at_position_cap("pairs_long"):
        pass
    else:
        lp = get_stock_price(long_ticker)
        if lp and passes_stock_price_guardrail(long_ticker, lp):
            place_stock_trade(long_ticker, lp, bull["signal"], size, strategy="pairs_long")

    short_key = f"{short_ticker}__pairs_short"
    if short_key in _monitored_positions:
        log.info("  → already holding pairs_short %s — skip short leg", short_ticker)
    elif _at_position_cap("pairs_short"):
        pass
    else:
        sp = get_stock_price(short_ticker)
        if sp and passes_stock_price_guardrail(short_ticker, sp):
            place_short_trade(short_ticker, sp, bear["signal"], size, strategy="pairs_short")

async def maybe_enter_pairs(loop):
    """If the day has both a bull and bear signal and we haven't entered yet,
    open the pair. Runs in ALL regimes (pairs are ~market-neutral — no gate)."""
    if not PAIRS_ENABLED:
        return
    _pairs_reset_if_new_day()
    if _pairs_day["entered"]:
        return
    bull, bear = _pairs_day["bull"], _pairs_day["bear"]
    if not bull or not bear or bull["ticker"] == bear["ticker"]:
        return
    size = scale_position_usd(MAX_POSITION_USD, bull["mag"], bull["conf"])
    _pairs_day["entered"] = True   # set before await so a concurrent call can't double-enter
    log.info("🔗 PAIRS entry: LONG %s / SHORT %s  size=$%.0f each  [pairs_long/pairs_short]",
             bull["ticker"], bear["ticker"], size)
    try:
        await asyncio.wait_for(
            loop.run_in_executor(None, enter_pairs_trade, bull, bear, size),
            timeout=45,
        )
    except asyncio.TimeoutError:
        log.error("enter_pairs_trade timed out — partial/no pair may have been opened")


# ── SHADOW STRATEGY ROUTER (logging-only — opens NO orders) ────────────────────
# Mirrors the rule-based router from router_backtest.py. For every qualifying
# signal it records which ONE strategy the router would pick, so we can later
# attribute the already-placed trades' realized P&L to the router's choices and
# test whether routing beats any single fixed strategy (router_eval.py).

ROUTER_DECISIONS_FILE = "router_decisions.csv"

# Earnings-headline patterns — mirrors pead_backtest.is_earnings_article so the
# router's pead-vs-stock split matches the backtest. Keep in sync if that changes.
_ROUTER_EARNINGS_PATTERNS = (
    "q1 earnings", "q2 earnings", "q3 earnings", "q4 earnings",
    "quarterly earnings", "quarterly results", "quarterly profit",
    "earnings beat", "earnings beats", "beat estimates", "beats estimates",
    "beat expectations", "beats expectations", "topped estimates",
    "exceeded estimates", "smashed estimates", "surpassed estimates",
    "eps beat", "eps of $", "eps topped",
    "raises guidance", "raised guidance", "raises outlook", "boosts outlook",
    "guidance raised", "guidance increase", "guidance raise",
    "reports earnings", "reported earnings", "reports q", "reported q",
)

def _is_earnings_headline(headline: str) -> bool:
    h = (headline or "").lower()
    return any(p in h for p in _ROUTER_EARNINGS_PATTERNS)

def _soft_catalyst_hit(headline: str, body: str):
    """Return the matched SOFT_CATALYST_PATTERN (analyst/PT, technical, acquirer-M&A) or None.
    These catalyst types are the bot's systematic losers (loser_analysis 2026-06-21) — the model
    over-scores them despite the prompt, so we gate them deterministically."""
    text = f"{headline or ''} {body or ''}".lower()
    return next((p for p in SOFT_CATALYST_PATTERNS if p in text), None)

def _prescore_noise(headline: str) -> "str | None":
    """Return the matched PRESCORE_SKIP_PATTERN (macro wrap / listicle / reactive recap) or None.
    Checked on the HEADLINE only, BEFORE scoring — these never warrant an Ollama/Groq call."""
    h = (headline or "").lower()
    return next((p for p in PRESCORE_SKIP_PATTERNS if p in h), None)

def _router_reset_if_new_day():
    today = datetime.now(timezone.utc).date()
    if _router_day["date"] != today:
        _router_day.update(date=today, logged_pairs=False)

def _write_router_row(sentiment, ticker, partner, mag, conf, choice,
                      is_earn, uptrend, headline, reason):
    path = ROUTER_DECISIONS_FILE
    write_header = not os.path.exists(path)
    try:
        with open(path, "a", newline="") as f:
            w = csv.writer(f)
            if write_header:
                w.writerow(["timestamp", "trading_day", "sentiment", "ticker",
                            "pair_partner", "magnitude", "confidence", "score",
                            "is_earnings", "regime_uptrend", "router_choice",
                            "reason", "headline"])
            now = datetime.now(timezone.utc)
            w.writerow([now.isoformat(), now.date().isoformat(), sentiment, ticker,
                        partner, f"{mag:.2f}", f"{conf:.2f}", f"{mag*conf:.3f}",
                        int(is_earn), int(uptrend), choice, reason,
                        (headline or "")[:160]])
        log.info("🧭 Router pick: %-10s  %s%s  [shadow — no order placed]",
                 choice, ticker, f" / {partner}" if partner else "")
    except Exception as e:
        log.warning("Router decision log failed: %s", e)

def log_router_decision(side: str, signal: dict, mag: float, conf: float, headline: str):
    """Record the router's single-strategy pick for this signal. No orders placed.
    side ∈ {"bull","bear"}. Same routing rules as router_backtest.py."""
    if not ROUTER_SHADOW_ENABLED:
        return
    _pairs_reset_if_new_day()
    _router_reset_if_new_day()
    uptrend = market_in_uptrend()
    ticker  = _first_tradeable_ticker(signal)

    # Macro-bullish (no specific ticker) → router routes to a QQQ call
    if side == "bull" and not ticker:
        _write_router_row("bullish", "QQQ", "", mag, conf, "qqq_macro",
                          False, uptrend, headline, "macro-bullish (no ticker)")
        return
    if not ticker:
        return   # bearish with no ticker — nothing to short

    # Opposing-side signal already present today → the day routes to PAIRS
    # (logged once/day, using the actual top bull/bear the bot would pair).
    opp = _pairs_day["bear"] if side == "bull" else _pairs_day["bull"]
    if opp and opp["ticker"] != ticker:
        b, s = _pairs_day["bull"], _pairs_day["bear"]
        if (not _router_day["logged_pairs"] and b and s
                and b["ticker"] != s["ticker"]):
            _router_day["logged_pairs"] = True
            _write_router_row("both", b["ticker"], s["ticker"], b["mag"], b["conf"],
                              "pairs", False, uptrend, headline,
                              "bull+bear same day → pairs")
        return

    if side == "bull":
        is_earn = _is_earnings_headline(headline)
        choice  = "pead" if is_earn else "stock"
        reason  = "bullish earnings → pead" if is_earn else "bullish → stock"
        _write_router_row("bullish", ticker, "", mag, conf, choice,
                          is_earn, uptrend, headline, reason)
    else:
        _write_router_row("bearish", ticker, "", mag, conf, "bear_short",
                          False, uptrend, headline, "bearish → short")


def close_option_position(symbol: str, qty: int, reason: str = "trailing stop"):
    """
    Close with a MARKETABLE limit sell-to-close (limit set just below the bid so it
    fills immediately at the bid). Confirmed working on this paper account — the old
    "uncovered option contracts" rejection that previously forced a stop-limit-above-
    market workaround no longer applies; the stop-limit also silently FAILED to fill on
    illiquid contracts (the stop never triggered without a trade print). A marketable
    limit sell fills reliably. Returns (submitted_ok, actual_fill_price | None).
    """
    if not market_is_open():
        log.warning("  → market closed/halted — deferring close of %s", symbol)
        return False, None

    quote = get_option_quote(symbol)
    if quote and quote["bid"] > 0:
        # Limit 10% below bid → marketable (fills at the bid), with a small floor so a
        # stale-wide quote can't dump us far below fair value.
        limit_price = max(round(quote["bid"] * 0.90, 2), 0.01)
    else:
        # No bid — option is illiquid/near-zero. Try to dump at the floor; may not
        # fill, but a no-bid option has negligible remaining value/risk.
        log.warning("  → no bid quote for %s; placing floor limit sell (may not fill)", symbol)
        limit_price = 0.01

    req = LimitOrderRequest(
        symbol=symbol, qty=qty, side=OrderSide.SELL,
        time_in_force=TimeInForce.DAY,
        limit_price=limit_price,
        position_intent=PositionIntent.SELL_TO_CLOSE,
    )
    try:
        order = trading_client.submit_order(req)
        fill  = _await_fill_price(order)
        log.info("✅ SELL TO CLOSE  %s x%d  (%s)  limit=%.2f%s",
                 symbol, qty, reason, limit_price,
                 f"  fill=${fill:.4f}" if fill else "  (fill pending → mid fallback)")
        return True, fill
    except Exception as e:
        log.error("Close order failed for %s: %s — will retry next cycle", symbol, e)
        return False, None

# ── Stop order management ─────────────────────────────────────────────────────

def place_stop_order(symbol: str, qty: int, stop_price: float) -> Optional[str]:
    """
    Place a GTC stop_limit sell-to-close order at stop_price.
    Limit price is set 2% below stop to ensure fill on illiquid options
    while still protecting against catastrophic slippage.
    Returns the Alpaca order ID, or None on failure.
    """
    limit_price = round(stop_price * 0.98, 2)
    stop_price  = round(stop_price, 2)
    req = StopLimitOrderRequest(
        symbol=symbol,
        qty=qty,
        side=OrderSide.SELL,
        time_in_force=TimeInForce.GTC,
        stop_price=stop_price,
        limit_price=limit_price,
        position_intent=PositionIntent.SELL_TO_CLOSE,
    )
    try:
        order = trading_client.submit_order(req)
        log.info(
            "🛡  Stop order placed for %s  stop=$%.4f  limit=$%.4f  id=%s",
            symbol, stop_price, limit_price, order.id,
        )
        return str(order.id)
    except Exception as e:
        log.warning("Stop order failed for %s: %s — monitor will handle manually", symbol, e)
        return None

def cancel_stop_order(order_id: str, symbol: str):
    """Cancel an existing stop order by ID."""
    try:
        trading_client.cancel_order_by_id(order_id)
        log.info("❌ Stop order cancelled for %s (id=%s)", symbol, order_id)
    except Exception as e:
        log.warning("Could not cancel stop order %s: %s", order_id, e)

def replace_stop_order(symbol: str, qty: int, old_order_id: Optional[str], new_stop_price: float) -> Optional[str]:
    """Cancel old stop order (if any) and place a new one at new_stop_price."""
    if old_order_id:
        cancel_stop_order(old_order_id, symbol)
    return place_stop_order(symbol, qty, new_stop_price)

# ═══════════════════════════════════════════════════════════════════════════════
# TRAILING STOP MONITOR  (options)
# ═══════════════════════════════════════════════════════════════════════════════

def _parse_option_ticker(symbol: str) -> str:
    """Extract the underlying ticker from an OCC option symbol (e.g. 'SNOW260626C00270000' → 'SNOW')."""
    import re
    m = re.match(r'^([A-Z]+)\d{6}[CP]\d+$', symbol)
    return m.group(1) if m else symbol[:4]

def load_open_positions_from_alpaca():
    """
    On startup, query Alpaca for all open positions and register them
    with the trailing stop monitor so restarts don't leave orphaned positions.
    Cancels any existing open stop orders first so we can place fresh ones at
    the correct current stop prices (stale GTC orders from a prior session would
    block new placement with an 'uncovered option' error).
    """
    # Cancel all open option sell orders left over from a previous session.
    try:
        open_orders = trading_client.get_orders(
            filter=GetOrdersRequest(status=QueryOrderStatus.OPEN)
        )
        for o in open_orders:
            if o.side == OrderSide.SELL:
                try:
                    trading_client.cancel_order_by_id(o.id)
                    log.info("  🗑  Cancelled stale stop order for %s", o.symbol)
                except Exception as ce:
                    log.warning("  Could not cancel stale order %s: %s", o.id, ce)
    except Exception as e:
        log.warning("Could not list open orders on startup: %s", e)

    try:
        positions = trading_client.get_all_positions()
    except Exception as e:
        log.warning("Could not fetch open positions from Alpaca: %s", e)
        return

    # bot_state.json holds live state (peak/entry_dt/materiality) but is REWRITTEN every
    # cycle, so once an option's strategy was wrongly persisted it stays wrong. trades.csv
    # is APPEND-ONLY — the strategy recorded at entry never changes, so it's the
    # authoritative source for an option's strategy. Use trades.csv for the strategy tag
    # (recovers positions already mis-tagged news_call) and bot_state for live state.
    try:
        with open(BOT_STATE_FILE) as f:
            saved = json.load(f)
    except Exception:
        saved = {}
    entry_log: dict[str, dict] = {}
    try:
        with open("trades.csv") as f:
            for row in csv.DictReader(f):
                sym = row.get("option_symbol")
                if sym:                       # later rows overwrite → keep the latest entry
                    entry_log[sym] = row
    except Exception:
        pass

    loaded = 0
    for p in positions:
        symbol = p.symbol
        if symbol in _monitored_positions:
            continue  # already tracked

        try:
            entry = float(p.avg_entry_price)
            qty   = int(float(p.qty))
            asset_class = str(p.asset_class).lower().replace("assetclass.", "")

            if asset_class == "us_option":
                quote = get_option_quote(symbol)
                current = quote["mid"] if quote else entry
                info = saved.get(symbol, {})
                entry_row = entry_log.get(symbol, {})
                # Strategy: trades.csv (authoritative entry record) → bot_state → default.
                strategy = entry_row.get("strategy") or info.get("strategy") or "news_call"
                # Preserve the ratcheted peak so the trailing stop doesn't reset.
                peak = max(entry, current, float(info.get("peak_price", 0) or 0))
                # entry_dt for the time-stop: bot_state, else the trades.csv entry timestamp.
                edt = info.get("entry_dt") or entry_row.get("timestamp")
                try:
                    entry_dt = datetime.fromisoformat(edt) if edt else None
                except Exception:
                    entry_dt = None
                # No exchange-held stop (account can't hold uncovered option
                # sell-stops); the in-process monitor protects it.
                _monitored_positions[symbol] = {
                    "qty":           qty,
                    "entry_price":   entry,
                    "peak_price":    peak,
                    "underlying":    _parse_option_ticker(symbol),
                    "asset_type":    "option",
                    "stop_order_id": None,
                    "strategy":      strategy,
                    "entry_dt":      entry_dt,
                }
                if info.get("materiality") is not None:   # preserve lotto shadow A/B tag
                    _monitored_positions[symbol]["materiality"] = info.get("materiality")
            else:
                continue   # equity / crypto positions — not managed by this bot

            loaded += 1
            log.info("  📂 Loaded position: %s  entry=$%.4f  qty=%d", symbol, entry, qty)
        except Exception as e:
            log.warning("Could not load position %s: %s", symbol, e)

    if loaded:
        log.info("📂 Loaded %d open position(s) from Alpaca into trailing stop monitor", loaded)
    else:
        log.info("📂 No existing positions found on Alpaca")

    # Reload bot-managed stock positions from bot_state.json (loaded above as `saved`).
    # Alpaca equity positions can't be distinguished from user trades without this
    # side-channel; bot_state.json is the authoritative record of what the bot owns.
    try:
        # Multiple strategies (e.g. stock + pead) buy shares of the SAME ticker;
        # Alpaca aggregates them into ONE position. Split the aggregate qty across
        # the strategies that share a ticker so each monitor entry owns its real
        # slice — otherwise every leg claims the full position (double-count →
        # double-sell on stop + 2× dashboard P&L). stock+pead buy equal $, so an
        # even split (remainder to the first legs) reconstructs the true qtys.
        share_groups: dict[tuple, list] = {}
        for sym, inf in saved.items():
            if inf.get("asset_type") in ("stock", "stock_short"):
                tk = inf.get("underlying", sym)
                share_groups.setdefault((tk, inf["asset_type"]), []).append(inf.get("strategy", "stock"))
        for k in share_groups:
            share_groups[k] = sorted(set(share_groups[k]))

        for symbol, info in saved.items():
            if info.get("asset_type") not in ("stock", "stock_short") or symbol in _monitored_positions:
                continue
            try:
                asset_type = info.get("asset_type", "stock")
                strategy   = info.get("strategy", "stock")
                ticker     = info.get("underlying", symbol)
                p    = trading_client.get_open_position(ticker)
                total_qty = abs(int(float(p.qty)))  # Alpaca aggregate (neg for shorts)
                # Split the aggregate across the strategies sharing this ticker.
                peers = share_groups.get((ticker, asset_type), [strategy])
                n     = max(1, len(peers))
                idx   = peers.index(strategy) if strategy in peers else 0
                qty   = total_qty // n + (1 if idx < (total_qty % n) else 0)
                if qty <= 0:
                    log.info("  📂 %s [%s] splits to 0 shares of %d (shared by %d) — skipping leg",
                             ticker, strategy, total_qty, n)
                    continue
                entry = float(p.avg_entry_price)
                key   = f"{ticker}__{strategy}"
                # Restore entry_dt from saved state (persisted as ISO string)
                raw_edt = info.get("entry_dt")
                entry_dt_restored = None
                if raw_edt:
                    try:
                        entry_dt_restored = datetime.fromisoformat(raw_edt)
                        if entry_dt_restored.tzinfo is None:
                            entry_dt_restored = entry_dt_restored.replace(tzinfo=timezone.utc)
                    except Exception:
                        pass

                if asset_type == "stock_short":
                    trough = min(entry, info.get("trough_price", entry))
                    _monitored_positions[key] = {
                        "qty":          qty,
                        "entry_price":  entry,
                        "trough_price": trough,
                        "underlying":   ticker,
                        "asset_type":   "stock_short",
                        "stop_order_id": None,
                        "strategy":     strategy,
                        "entry_dt":     entry_dt_restored,
                    }
                    log.info("  📂 Reloaded SHORT: %s  entry=$%.2f  qty=%d  trough=$%.2f  [%s]",
                             ticker, entry, qty, trough, strategy)
                else:
                    peak = max(entry, info.get("peak_price", entry))
                    _monitored_positions[key] = {
                        "qty":          qty,
                        "entry_price":  entry,
                        "peak_price":   peak,
                        "underlying":   ticker,
                        "asset_type":   "stock",
                        "stop_order_id": None,
                        "strategy":     strategy,
                        "entry_dt":     entry_dt_restored,
                    }
                    log.info("  📂 Reloaded stock position: %s  entry=$%.2f  qty=%d  peak=$%.2f  [%s]",
                             ticker, entry, qty, peak, strategy)
            except Exception:
                pass  # position closed or not found — skip silently
    except (FileNotFoundError, Exception):
        pass  # no saved state yet

# ═══════════════════════════════════════════════════════════════════════════════
# SHADOW STRATEGY TRACKING (paper-only — NO orders, NO capital)
# ═══════════════════════════════════════════════════════════════════════════════

def _log_shadow_trade(action: str, symbol: str, pos: dict, price: float,
                      pnl_usd: float = 0.0, reason: str = ""):
    """Append an open/close row for a shadow position to shadow_trades.csv."""
    try:
        entry = pos.get("entry_price", 0)
        pnl_pct = ((price / entry - 1) * 100) if entry else 0
        write_header = not os.path.exists(SHADOW_TRADES_FILE)
        with open(SHADOW_TRADES_FILE, "a", newline="") as f:
            w = csv.writer(f)
            if write_header:
                w.writerow(["timestamp", "action", "symbol", "underlying", "strategy", "qty",
                            "entry_price", "price", "pnl_usd", "pnl_pct", "peak_price", "reason"])
            w.writerow([datetime.now(timezone.utc).isoformat(), action, symbol,
                        pos.get("underlying", ""), pos.get("strategy", "news_call_shadow"),
                        pos.get("qty", 1), f"{entry:.4f}", f"{price:.4f}",
                        f"{pnl_usd:.2f}", f"{pnl_pct:.2f}",
                        f"{pos.get('peak_price', entry):.4f}", reason])
    except Exception as e:
        log.debug("shadow trade log failed: %s", e)

def _write_shadow_state():
    try:
        state = {}
        for sym, pos in _shadow_positions.items():
            p = dict(pos)
            if isinstance(p.get("entry_dt"), datetime):
                p["entry_dt"] = p["entry_dt"].isoformat()
            state[sym] = p
        with open(SHADOW_STATE_FILE, "w") as f:
            json.dump(state, f)
    except Exception:
        pass

def _load_shadow_state():
    try:
        with open(SHADOW_STATE_FILE) as f:
            saved = json.load(f)
        for sym, p in saved.items():
            edt = p.get("entry_dt")
            try:
                p["entry_dt"] = datetime.fromisoformat(edt) if edt else None
            except Exception:
                p["entry_dt"] = None
            _shadow_positions[sym] = p
        if _shadow_positions:
            log.info("📂 Reloaded %d shadow position(s) [paper]", len(_shadow_positions))
    except FileNotFoundError:
        pass
    except Exception as e:
        log.debug("shadow state reload failed: %s", e)

def _shadow_has_ticker(base_strategy: str, ticker: str) -> bool:
    return any(p.get("base_strategy") == base_strategy and p.get("underlying") == ticker
               for p in _shadow_positions.values())

def _open_shadow(base_strategy: str, ticker: str, stock_price: float, contract: dict,
                 qty: int, target_delta: float, dte_target: int):
    """Register a paper shadow position (NO order) + log the pick. Generic across the
    shadowed strategies (news_call, lotto)."""
    sym = contract["symbol"]
    if sym in _shadow_positions or len(_shadow_positions) >= MAX_SHADOW_POSITIONS:
        return
    entry = contract["ask"]
    pos = {"qty": qty, "entry_price": entry, "peak_price": entry, "underlying": ticker,
           "asset_type": "option", "strategy": f"{base_strategy}_shadow",
           "base_strategy": base_strategy, "entry_dt": datetime.now(timezone.utc)}
    _shadow_positions[sym] = pos
    log_contract_pick(ticker, stock_price, f"{base_strategy}_shadow", target_delta, dte_target, contract, qty)
    _log_shadow_trade("open", sym, pos, entry, 0.0, "entry")
    _write_shadow_state()
    log.info("  👻 shadow %s: %s x%d @ $%.4f (paper, no order) [%s_shadow]",
             base_strategy, sym, qty, entry, base_strategy)

def shadow_news_call_fn(ticker: str, signal: dict, option_usd: float):
    """Paper-only: shadow the contract news_call WOULD buy on signals it does NOT trade
    (mag<NEWS_CALL_MIN_MAGNITUDE now that news_call is live). Uses the SAME news_call geometry
    (Δ0.40/DTE~10) so the forward shadow data is faithful to the real leg."""
    try:
        if _shadow_has_ticker("news_call", ticker):
            return
        stock_price = get_stock_price(ticker)
        if not stock_price or not passes_stock_price_guardrail(ticker, stock_price):
            return
        contract = pick_call_contract(ticker, stock_price, NEWS_CALL_DTE_MIN,
                                      NEWS_CALL_DTE_MAX, NEWS_CALL_TARGET_DELTA, 1.10, for_shadow=True)
        if not contract:
            return
        # Apply the LIVE budget guard (place_option_trade): if ONE contract costs more than
        # MAX_CONTRACT_BUDGET_MULT × the budget, the real leg would SKIP it — so the shadow must
        # too, else a single high-premium contract (e.g. the SNDK $12k call) silently oversizes
        # and distorts the shadow P&L. Keeps the shadow a faithful proxy of what news_call trades.
        contract_cost = contract["ask"] * 100
        if contract_cost > option_usd * MAX_CONTRACT_BUDGET_MULT:
            log.debug("shadow news_call SKIP %s: 1 contract $%.0f > %.1f× budget $%.0f",
                      ticker, contract_cost, MAX_CONTRACT_BUDGET_MULT, option_usd)
            return
        qty = max(1, int(option_usd / contract_cost))
        _open_shadow("news_call", ticker, stock_price, contract, qty, NEWS_CALL_TARGET_DELTA,
                     (NEWS_CALL_DTE_MIN + NEWS_CALL_DTE_MAX) // 2)
    except Exception as e:
        log.debug("shadow news_call failed: %s", e)

def shadow_stock_fn(ticker: str, signal: dict, position_usd: float):
    """Paper-only: shadow the stock trade the (now real-disabled) stock leg WOULD make, so we keep
    gathering stock-strategy data without deploying capital or consuming a cap slot. Sized like the
    real leg (scale_position_usd). Marked-to-market by _run_shadow_monitor's stock branch using the
    real stock exit rules (STOCK_TRAIL_PCT flat trail + NEWS_STOCK_MAX_HOLD_DAYS time-stop)."""
    try:
        if _shadow_has_ticker("stock", ticker):
            return
        if len(_shadow_positions) >= MAX_SHADOW_POSITIONS:
            return
        price = get_stock_price(ticker)
        if not price or not passes_stock_price_guardrail(ticker, price):
            return
        sym = f"{ticker}__stock_shadow"
        if sym in _shadow_positions:
            return
        shares = max(1, int(position_usd / price))
        pos = {"qty": shares, "entry_price": price, "peak_price": price, "underlying": ticker,
               "asset_type": "stock", "strategy": "stock_shadow", "base_strategy": "stock",
               "entry_dt": datetime.now(timezone.utc)}
        _shadow_positions[sym] = pos
        _log_shadow_trade("open", sym, pos, price, 0.0, "entry")
        _write_shadow_state()
        log.info("  👻 shadow stock: %s x%d @ $%.2f (paper, no order) [stock_shadow]", ticker, shares, price)
    except Exception as e:
        log.debug("shadow stock failed: %s", e)

def shadow_pead_fn(ticker: str, signal: dict, position_usd: float):
    """Paper-only: shadow a PEAD entry that overflowed the reserved PEAD sub-cap
    (PEAD_MAX_POSITIONS). base_strategy='pead' so _run_shadow_monitor's stock branch marks it
    to market with PEAD's OWN exit rules (tiered trail + PEAD_MAX_HOLD_DAYS) — a faithful proxy
    of the trade we'd have taken, and a way to see whether 20 reserved slots ever binds."""
    try:
        if _shadow_has_ticker("pead", ticker):
            return
        if len(_shadow_positions) >= MAX_SHADOW_POSITIONS:
            return
        price = get_stock_price(ticker)
        if not price or not passes_stock_price_guardrail(ticker, price):
            return
        sym = f"{ticker}__pead_shadow"
        if sym in _shadow_positions:
            return
        shares = max(1, int(position_usd / price))
        pos = {"qty": shares, "entry_price": price, "peak_price": price, "underlying": ticker,
               "asset_type": "stock", "strategy": "pead_shadow", "base_strategy": "pead",
               "entry_dt": datetime.now(timezone.utc)}
        _shadow_positions[sym] = pos
        _log_shadow_trade("open", sym, pos, price, 0.0, "entry")
        _write_shadow_state()
        log.info("  👻 shadow pead: %s x%d @ $%.2f (paper, no order, sub-cap overflow) [pead_shadow]",
                 ticker, shares, price)
    except Exception as e:
        log.debug("shadow pead failed: %s", e)

def shadow_lotto_fn(ticker: str, stock_price: float):
    """Paper-only: shadow the lotto pick when the REAL lotto is blocked by the position
    cap — so a high-conviction lotto signal isn't lost. Respects the same hard $250 cap."""
    try:
        if _shadow_has_ticker("lotto", ticker):
            return
        contract = pick_call_contract(ticker, stock_price, LOTTO_MIN_DTE, LOTTO_MAX_DTE,
                                      LOTTO_TARGET_DELTA, LOTTO_STRIKE_HI_MULT, for_shadow=True)
        if not contract:
            # Nothing in-window → try the closest-to-window contract (out-of-window shadow).
            shadow_lotto_oow_fn(ticker, stock_price)
            return
        if contract["ask"] * 100 > LOTTO_POSITION_USD:   # hard $250 cap
            return
        qty = max(1, int(LOTTO_POSITION_USD / (contract["ask"] * 100)))
        _open_shadow("lotto", ticker, stock_price, contract, qty, LOTTO_TARGET_DELTA,
                     (LOTTO_MIN_DTE + LOTTO_MAX_DTE) // 2)
    except Exception as e:
        log.debug("shadow lotto failed: %s", e)

def shadow_lotto_oow_fn(ticker: str, stock_price: float):
    """Paper-only OUT-OF-WINDOW shadow: fired when a high-conviction lotto signal has NO
    contract inside the 8-21 DTE window. Picks the closest-to-window contract and shadows it
    (tagged 'lotto' so the shadow monitor uses lotto exit rules; the out-of-window DTE in
    contract_selection.csv distinguishes these). No-op if truly optionless or already shadowed."""
    if not LOTTO_OOW_SHADOW_ENABLED:
        return
    try:
        if _shadow_has_ticker("lotto", ticker):
            return
        picked = pick_closest_oow_contract(ticker, stock_price, LOTTO_MIN_DTE, LOTTO_MAX_DTE,
                                           LOTTO_TARGET_DELTA, LOTTO_STRIKE_HI_MULT)
        if not picked:
            return                                   # truly optionless, or window wasn't empty
        contract, actual_dte = picked
        if contract["ask"] * 100 > LOTTO_POSITION_USD:   # respect the hard $250 cap
            return
        qty = max(1, int(LOTTO_POSITION_USD / (contract["ask"] * 100)))
        _open_shadow("lotto", ticker, stock_price, contract, qty, LOTTO_TARGET_DELTA,
                     (LOTTO_MIN_DTE + LOTTO_MAX_DTE) // 2)
        log.info("  👻 lotto OUT-OF-WINDOW shadow: %s @ %dDTE (window %d-%d) — closest-to-window",
                 contract["symbol"], actual_dte, LOTTO_MIN_DTE, LOTTO_MAX_DTE)
    except Exception as e:
        log.debug("shadow lotto OOW failed: %s", e)

def _run_shadow_monitor():
    """Forward mark-to-market of shadow positions using each one's REAL strategy exit
    rules (trail width + time-stop, by base_strategy). Logs a shadow close + P&L when it
    trips. Paper-only; per-position try/except so it can never affect real trading."""
    if not _shadow_positions:
        return
    now = datetime.now(timezone.utc)
    for sym, pos in list(_shadow_positions.items()):
        try:
            base = pos.get("base_strategy", "news_call")
            # ── STOCK shadow: mark at the stock price, real stock exit rules (flat trail + time-stop) ──
            if pos.get("asset_type") == "stock":
                price = get_stock_price(pos["underlying"])
                if not price:
                    continue
                entry = pos["entry_price"]; qty = pos["qty"]
                edt = pos.get("entry_dt"); max_days = max_hold_days_for(base, "stock")
                if edt and max_days > 0 and (now - edt).days >= max_days:
                    pnl = (price - entry) * qty
                    _log_shadow_trade("close", sym, pos, price, pnl, "time-stop")
                    del _shadow_positions[sym]
                    log.info("  👻 shadow %s CLOSED %s  P&L $%+.0f  (time-stop)", base, sym, pnl)
                    continue
                if price > pos.get("peak_price", entry):
                    pos["peak_price"] = price
                peak = pos["peak_price"]; gain = (peak / entry - 1.0) if entry else 0.0
                stop = peak * (1 - long_trail_for(base, "stock", gain))
                if price <= stop:
                    pnl = (price - entry) * qty
                    _log_shadow_trade("close", sym, pos, price, pnl, "trail-stop")
                    del _shadow_positions[sym]
                    log.info("  👻 shadow %s CLOSED %s  P&L $%+.0f  (px $%.2f ≤ stop $%.2f)",
                             base, sym, pnl, price, stop)
                continue
            # ── OPTION shadow (news_call / lotto): mark at the option mid ──
            q = get_option_quote(sym)
            if not q:
                continue
            mid = q["mid"]; entry = pos["entry_price"]; qty = pos["qty"]
            # time-stop (e.g. lotto = 7d; news_call = none)
            edt = pos.get("entry_dt")
            max_days = max_hold_days_for(base, "option")
            if edt and max_days > 0 and (now - edt).days >= max_days:
                pnl = (mid - entry) * 100 * qty
                _log_shadow_trade("close", sym, pos, mid, pnl, "time-stop")
                del _shadow_positions[sym]
                log.info("  👻 shadow %s CLOSED %s  P&L $%+.0f  (time-stop)", base, sym, pnl)
                continue
            # trailing stop (per-strategy width)
            if mid > pos.get("peak_price", entry):
                pos["peak_price"] = mid
            peak = pos["peak_price"]
            gain = (peak / entry - 1.0) if entry else 0.0
            trail = long_trail_for(base, "option", gain)
            stop = peak * (1 - trail)
            if mid <= stop:
                pnl = (mid - entry) * 100 * qty
                _log_shadow_trade("close", sym, pos, mid, pnl, "trail-stop")
                del _shadow_positions[sym]
                log.info("  👻 shadow %s CLOSED %s  P&L $%+.0f  (mid $%.4f ≤ stop $%.4f)",
                         base, sym, pnl, mid, stop)
        except Exception:
            continue
    _write_shadow_state()


def _log_position_path(symbol: str, pos: dict, mid: float, pnl_pct: float, stop_price, phase: str = "open") -> None:
    """Append one monitor-cycle sample of a position's premium/price path to position_paths.csv.
    The monitor already fetches the REAL option mid (or stock price) every ~30s — this persists it.
    phase='open' = while we hold it; phase='postclose' = AFTER we exited, kept until the position's
    natural horizon so we can replay alternate exit rules on the FULL real path (would a different exit
    have done better, incl. holding past our exit?). Real quotes beat the BS-synthesized daily bars where
    the trail/ratchet distinctions washed out (trail_winrate_sweep / exit_intraday_sweep). Market-hours only."""
    try:
        new = not os.path.exists("position_paths.csv")
        with open("position_paths.csv", "a", newline="") as f:
            w = csv.writer(f)
            if new:
                w.writerow(["ts", "symbol", "strategy", "asset_type", "underlying",
                            "entry", "mid", "peak", "pnl_pct", "stop", "phase"])
            w.writerow([datetime.now(timezone.utc).isoformat(), symbol, pos.get("strategy", ""),
                        pos.get("asset_type", ""), pos.get("underlying", ""),
                        pos.get("entry_price", ""), round(mid, 4),
                        pos.get("peak_price") or pos.get("trough_price") or pos.get("entry_price", ""),
                        round(pnl_pct, 2), stop_price if stop_price is not None else "", phase])
    except Exception as e:
        log.debug("position-path log failed for %s: %s", symbol, e)


# ── Post-close path capture: keep sampling a CLOSED position's real ~30s quote until its natural
#    horizon, so we can replay alternate exit rules on the full path (request 2026-06-26). ──
_postclose: dict[str, dict] = {}
_POSTCLOSE_FILE = "postclose_tracking.json"


def _write_postclose() -> None:
    try:
        json.dump(_postclose, open(_POSTCLOSE_FILE, "w"))
    except Exception as e:
        log.debug("postclose write failed: %s", e)


def _load_postclose() -> None:
    global _postclose
    try:
        if os.path.exists(_POSTCLOSE_FILE):
            _postclose = json.load(open(_POSTCLOSE_FILE))
    except Exception:
        _postclose = {}


def _register_postclose(symbol: str, pos: dict, exit_price) -> None:
    """On close, start tracking the position's would-be path until ~its natural max-hold horizon."""
    try:
        edt = pos.get("entry_dt")
        base = edt if isinstance(edt, datetime) else datetime.now(timezone.utc)
        strat = pos.get("strategy", ""); at = pos.get("asset_type", "option")
        mh = max_hold_days_for(strat, at) or 3
        _postclose[symbol] = {
            "entry_price": pos.get("entry_price"), "entry_dt": base.isoformat(),
            "exit_dt": datetime.now(timezone.utc).isoformat(), "exit_price": exit_price,
            "strategy": strat, "asset_type": at,
            "underlying": pos.get("underlying", symbol.split("__")[0]), "qty": pos.get("qty"),
            "peak_price": pos.get("peak_price") or pos.get("entry_price"),
            "horizon_dt": (base + timedelta(days=max(2, int(mh * 1.5) + 2))).isoformat(),
        }
        _write_postclose()
    except Exception as e:
        log.debug("postclose register failed for %s: %s", symbol, e)


def _sample_postclose() -> None:
    """One market-hours sample of each tracked closed position; drop those past their horizon."""
    if not _postclose:
        return
    now = datetime.now(timezone.utc)
    changed = False
    for sym in list(_postclose):
        pc = _postclose[sym]
        try:
            if now >= datetime.fromisoformat(pc["horizon_dt"]):
                del _postclose[sym]; changed = True; continue
        except Exception:
            del _postclose[sym]; changed = True; continue
        if sym in _monitored_positions:          # re-opened → live monitor logs it; skip post-close
            continue
        if pc.get("asset_type") in ("stock", "stock_short"):
            mid = _price_cache.get(pc.get("underlying")) or get_stock_price(pc.get("underlying"), subscribe=False)
        else:
            q = get_option_quote(sym); mid = q["mid"] if q else None
        if mid is None:
            continue
        if mid > (pc.get("peak_price") or 0):    # track post-exit peak (did it run higher than we got?)
            pc["peak_price"] = mid; changed = True
        entry = pc.get("entry_price") or 0
        pnl_pct = ((mid - entry) / entry * 100) if entry else 0
        _log_position_path(sym, pc, mid, pnl_pct, "", phase="postclose")
    if changed:
        _write_postclose()


async def trailing_stop_monitor():
    global _daily_loss_usd   # declared once at function top to satisfy Python scoping
    log.info("📊 Trailing stop monitor started (interval: %ds, trail: %.0f%%)",
             MONITOR_INTERVAL, TRAILING_STOP_PCT * 100)
    _load_postclose()
    while True:
        await asyncio.sleep(MONITOR_INTERVAL)

        # Roll the daily-loss breaker to the current UTC day BEFORE any closes accrue, so
        # losses on a new day can't be added to (then wiped from) the prior day's counter.
        _loss_reset_if_new_day()

        # Shadow (paper) positions — track forward P&L regardless of real positions.
        try:
            _run_shadow_monitor()
        except Exception as e:
            log.debug("shadow monitor error: %s", e)

        # Post-close path capture runs even when no positions are open (a closed position keeps
        # getting sampled until its horizon). One clock call/cycle, reused below.
        mkt_open = market_is_open()
        if mkt_open:
            try:
                _sample_postclose()
            except Exception as e:
                log.debug("postclose sample error: %s", e)

        if not _monitored_positions:
            continue

        # Market reopened → forget prior deferrals so the next real breach logs once
        # (and the deferred closes below will now actually execute).
        if _deferred_closes and market_is_open():
            _deferred_closes.clear()

        # Compute the EOD window ONCE per cycle (one clock call, not one per position).
        near_close = near_market_close()

        for symbol, pos in list(_monitored_positions.items()):
            asset_type = pos.get("asset_type", "option")
            strategy   = pos.get("strategy", asset_type)

            # ── Get current price ─────────────────────────────────
            if asset_type in ("stock", "stock_short"):
                underlying = pos.get("underlying", symbol.split("__")[0])
                mid = _price_cache.get(underlying) or get_stock_price(underlying, subscribe=False)
            else:
                q   = get_option_quote(symbol)
                mid = q["mid"] if q else None

            if mid is None:
                log.debug("  👁 %s — no price available", symbol)
                continue

            entry = pos["entry_price"]
            qty   = pos["qty"]

            # ── Time-stop: close NEAR THE CLOSE on the final hold day ─────────
            # The position has reached its max hold, but we exit it in the EOD
            # window (near_close) rather than intraday or carried to next morning —
            # the overnight gap is uncompensated risk (overnight_gap_test.py). Until
            # the EOD window, we DON'T close here and fall through to the trailing-stop
            # logic, so the position is still protected intraday on its final day.
            if _time_stop_hit(pos) and near_close:
                days_held = (datetime.now(timezone.utc) - pos["entry_dt"]).days
                if asset_type == "stock":
                    pnl_usd = (mid - entry) * qty
                elif asset_type == "stock_short":
                    pnl_usd = (entry - mid) * qty
                else:
                    pnl_usd = (mid - entry) * 100 * qty
                pnl_pct = (mid / entry - 1) * 100 if entry else 0
                log.info("⏱ TIME-STOP (EOD): %s held %dd ≥ %dd max  P&L: %+.1f%% (%+.0f$)  [%s]",
                         symbol, days_held,
                         max_hold_days_for(pos.get("strategy",""), asset_type),
                         pnl_pct, pnl_usd, strategy)
                # near_close implies the market is open, so we can close now.
                if asset_type == "stock":
                    ok, fill = close_stock_position(symbol, qty, reason="time-stop")
                elif asset_type == "stock_short":
                    ok, fill = close_short_position(symbol, qty, reason="time-stop")
                else:
                    ok, fill = close_option_position(symbol, qty, reason="time-stop")
                if ok:
                    exit_px = fill if fill is not None else mid   # ACTUAL fill, mid fallback
                    pnl_usd = _realized_pnl(asset_type, entry, exit_px, qty)
                    log_closed_trade(symbol, pos, exit_px, pnl_usd, reason="time-stop")
                    if pnl_usd < 0:
                        _daily_loss_usd += abs(pnl_usd)
                    _register_postclose(symbol, pos, exit_px)
                    del _monitored_positions[symbol]
                else:
                    log.warning("⏳ Time-stop close not confirmed for %s — will retry", symbol)
                continue

            # ── SHORT positions: track trough, stop fires on RALLY ────────────
            if asset_type == "stock_short":
                strail = short_trail_for(strategy)
                trough = pos.get("trough_price", entry)
                if mid < trough:
                    pos["trough_price"] = mid
                    trough = mid
                    new_stop = trough * (1 + strail)
                    log.info("📉 %s  new trough: $%.4f  → raising stop to $%.4f (trail %.0f%%)  [%s]",
                             symbol, trough, new_stop, strail * 100, strategy)
                stop_price = trough * (1 + strail)
                pnl_usd    = (entry - mid) * qty   # positive when price falls
                pnl_pct    = (entry - mid) / entry * 100 if entry else 0
                log.info(
                    "👁  %-30s  entry=$%.4f  mid=$%.4f  trough=$%.4f  stop=$%.4f  P&L: %+.1f%% (%+.2f$)  [%s]",
                    symbol, entry, mid, trough, stop_price, pnl_pct, pnl_usd, strategy,
                )
                _write_bot_state()
                if mkt_open:
                    _log_position_path(symbol, pos, mid, pnl_pct, stop_price)
                if mid >= stop_price:
                    log.info("🔴 SHORT STOP triggered: %s  mid=$%.4f ≥ stop=$%.4f  P&L: %+.1f%%",
                             symbol, mid, stop_price, pnl_pct)
                    if not market_is_open():
                        _log_deferred_once(symbol, "⏳ %s short stop deferred — market closed, will retry")
                        continue
                    ok, fill = close_short_position(symbol, qty)
                    if not ok:
                        log.warning("⏳ Cover not confirmed for %s — will retry", symbol)
                        continue
                    exit_px = fill if fill is not None else mid   # ACTUAL fill, mid fallback
                    pnl_usd = _realized_pnl("stock_short", entry, exit_px, qty)
                    log_closed_trade(symbol, pos, exit_px, pnl_usd, reason="stop")
                    if pnl_usd < 0:
                        _daily_loss_usd += abs(pnl_usd)
                    _register_postclose(symbol, pos, exit_px)
                    del _monitored_positions[symbol]
                continue   # short handled — skip the long logic below

            # ── LONG positions (stock, pead, option) ─────────────────────────
            peak = pos["peak_price"]

            # Trail width: PEAD/option tiered, flat for stock & pairs_long.
            gain = (peak / entry - 1.0) if entry else 0.0
            trail = long_trail_for(strategy, asset_type, gain)

            # Update peak and ratchet stop upward if price rose
            if mid > peak:
                pos["peak_price"] = mid
                peak = mid
                gain = (peak / entry - 1.0) if entry else 0.0
                trail = long_trail_for(strategy, asset_type, gain)
                new_stop = peak * (1 - trail)
                log.info("📈 %s  new peak: $%.4f  → trailing stop now $%.4f (trail %.0f%%)  [%s]",
                         symbol, peak, new_stop, trail * 100, strategy)
                # No exchange-held option stop (account can't hold them) — the
                # in-process monitor enforces the trail and closes via
                # close_option_position when breached.

            stop_price = peak * (1 - trail)
            if strategy == "pead" and PEAD_BREAKEVEN_LOCK > 0 and gain >= PEAD_BREAKEVEN_LOCK:
                stop_price = max(stop_price, entry * 1.005)

            if asset_type == "stock":
                pnl_usd = (mid - entry) * qty
            else:
                pnl_usd = (mid - entry) * 100 * qty
            pnl_pct = ((mid - entry) / entry * 100) if entry > 0 else 0

            log.info(
                "👁  %-30s  entry=$%.4f  mid=$%.4f  peak=$%.4f  stop=$%.4f  P&L: %+.1f%% (%+.2f$)  [%s]",
                symbol, entry, mid, peak, stop_price, pnl_pct, pnl_usd, strategy,
            )
            _write_bot_state()
            if mkt_open:
                _log_position_path(symbol, pos, mid, pnl_pct, stop_price)

            # ── LOTTO hard profit cap (Tier-1-validated; asymmetric-safe — only exits a WINNER
            #    early at +200%, never adds a loss). Take full profit the first time premium ≥
            #    cap×entry, before the convex short-DTE winner round-trips through the wide trail. ──
            if (LOTTO_HARD_CAP_ENABLED and strategy == "lotto" and asset_type == "option"
                    and entry > 0 and mid >= LOTTO_HARD_CAP_MULT * entry):
                log.info("🎯 LOTTO %.0f× CAP: %s  mid=$%.4f ≥ %.0f×$%.4f  P&L: %+.1f%%",
                         LOTTO_HARD_CAP_MULT, symbol, mid, LOTTO_HARD_CAP_MULT, entry, pnl_pct)
                if not market_is_open():
                    _log_deferred_once(symbol, "⏳ %s lotto cap hit but market closed — deferring, will retry")
                    continue
                ok, fill = close_option_position(symbol, qty, reason=f"lotto {LOTTO_HARD_CAP_MULT:.0f}x cap")
                if not ok:
                    log.warning("⏳ Lotto cap close not confirmed for %s — will retry", symbol)
                    continue
                exit_px = fill if fill is not None else mid
                pnl_usd = _realized_pnl(asset_type, entry, exit_px, qty)
                _log_lotto_cap(symbol, pos, entry, mid, exit_px, qty, pnl_usd)
                log_closed_trade(symbol, pos, exit_px, pnl_usd, reason="lotto_cap")
                if pnl_usd < 0:                       # defensive — a +200% cap is ~never a loss
                    _daily_loss_usd += abs(pnl_usd)
                _register_postclose(symbol, pos, exit_px)
                del _monitored_positions[symbol]
                continue

            if mid <= stop_price:
                log.info("🔴 TRAILING STOP triggered: %s  mid=$%.4f ≤ stop=$%.4f  P&L: %+.1f%%",
                         symbol, mid, stop_price, pnl_pct)

                if not market_is_open():
                    _log_deferred_once(symbol, "⏳ %s stop breached but market closed/halted — deferring close, will retry")
                    continue

                if asset_type == "stock":
                    ok, fill = close_stock_position(symbol, qty)
                else:
                    if pos.get("stop_order_id"):
                        cancel_stop_order(pos["stop_order_id"], symbol)
                        pos["stop_order_id"] = None
                    ok, fill = close_option_position(symbol, qty)

                if not ok:
                    log.warning("⏳ Close not confirmed for %s — keeping under monitoring, will retry", symbol)
                    continue

                exit_px = fill if fill is not None else mid   # ACTUAL fill, mid fallback
                pnl_usd = _realized_pnl(asset_type, entry, exit_px, qty)
                log_closed_trade(symbol, pos, exit_px, pnl_usd, reason="stop")
                if pnl_usd < 0:
                    _daily_loss_usd += abs(pnl_usd)
                    log.info("   Daily loss running total: $%.0f / $%.0f limit",
                             _daily_loss_usd, DAILY_LOSS_LIMIT)

                _register_postclose(symbol, pos, exit_px)
                del _monitored_positions[symbol]

# ═══════════════════════════════════════════════════════════════════════════════
# OLLAMA SCORER
# ═══════════════════════════════════════════════════════════════════════════════

# ── LIVE SCORER PROMPT = unified_v1 (adopted 2026-06-13) ───────────────────────
# Unified single-pass prompt: emits tickers/sentiment/confidence/magnitude/catalyst/
# reasoning so the routing RULES below select strategies from ONE LLM call — replacing
# the old separate materiality 2nd pass (now dropped; see execute_lotto_trade_fn +
# MATERIALITY_GATE_ENABLED=False). Chosen over v3/v4 in the rigorous overnight eval
# (unified_v1: +$25,485 lotto / Sharpe 5.12 / $19-22/tr stock; v3/v4 floods rejected).
# The `catalyst` field is kept for telemetry but is VESTIGIAL for routing (collinear
# with magnitude). KEEP THIS TEXT CHAR-EXACT IN SYNC with prompt_lab.PROMPTS["unified_v1"]
# — the eval that validated it scored against that exact string.
SYSTEM_PROMPT = """You are a quantitative equity trading signal generator.
Respond with a JSON object ONLY — no markdown, no code fences, no explanation.

Schema:
{
  "tickers":    ["AAPL"],
  "sentiment":  "bullish",
  "confidence": 0.82,
  "magnitude":  0.75,
  "catalyst":   0.60,
  "reasoning":  "one sentence"
}

sentiment:  "bullish" | "bearish" | "neutral"
confidence: float 0.0–1.0 — how certain you are about the sentiment direction
magnitude:  float 0.0–1.0 — how likely this news is to meaningfully move the stock price
catalyst:   float 0.0–1.0 — probability this is a RESOLVED, UNPRICED, binary event likely to trigger a
            LARGE, FAST repricing (a gap up or a trading halt) — NOT a slow drift and NOT an opinion

Magnitude scale:
  0.0–0.2  Noise or irrelevant (routine filings, minor analyst reiterations, fluff)
  0.2–0.4  Routine news (in-line earnings, small contract wins, minor upgrades)
  0.4–0.6  Meaningful catalyst (solid earnings beat, notable partnership, meaningful guidance raise)
  0.6–0.8  Strong catalyst (significant beat, major acquisition, landmark FDA approval for key drug)
  0.8–1.0  Transformative event (company-defining deal, paradigm-shifting approval, massive revision)

Catalyst scale — ONLY a RESOLVED binary event that just REMOVED uncertainty earns high catalyst:
  earnings PRINTED (big beat/miss), FDA decision MADE, deal SIGNED/announced, a trading HALT, a
  guidance change, a major contract WON → catalyst 0.7–0.95.
  Anticipation, speculation, opinion, ANALYST ratings/price targets, "could/may/plans to",
  partnerships, expansions, routine product launches, incremental updates → catalyst < 0.2 EVEN WHEN
  BULLISH. Macro / no specific company → catalyst 0.0.
  Examples: "Reports Q3 EPS $2.10 vs $1.60 est, Raises Guidance"→0.85 · "Shares Halted, Circuit
  Breaker To The Upside"→0.90 · "Wins Surprise FDA Approval"→0.85 · "To Acquire X at 40% Premium"→0.92
  · "Wells Fargo Maintains Overweight, Raises PT"→0.10 · "Morgan Stanley Upgrades to Buy"→0.18 ·
  "Announces Partnership With Beta Corp"→0.15 · "CEO Presents at Tech Conference"→0.05

Rules:
- Only include tickers you are highly confident about.
- General macro news with no specific company → empty tickers list.
- Most news is routine — be conservative; reserve magnitude 0.7+ AND catalyst 0.7+ for genuinely
  exceptional, RESOLVED events.
- Only flag bullish sentiment — we trade long calls only.
- Return ONLY the JSON object."""

def score_with_ollama(headline: str, body: str, source: str = "") -> Optional[dict]:
    prefix = f"[Source: {source}]\n" if source else ""
    try:
        resp = ollama_client.chat.completions.create(
            model=OLLAMA_MODEL,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user",   "content": f"{prefix}Headline: {headline}\n\nBody: {body[:1200]}\n\nRespond with JSON only."},
            ],
            temperature=0.1,
        )
        raw = resp.choices[0].message.content.strip()
        if raw.startswith("```"):
            raw = raw.split("```")[1]
            if raw.startswith("json"):
                raw = raw[4:]
        return json.loads(raw.strip())
    except json.JSONDecodeError as e:
        log.warning("JSON parse failed: %s", e)
        return None
    except Exception as e:
        log.warning("Ollama scoring failed: %s", e)
        return None

# ── Groq scorer A/B (forward latency + free-tier rate-limit test) ───────────────
# SHADOW only: on every live signal we ALSO score the SAME headline+prompt on Groq's
# hosted model, OFF the trade path, and log latency / agreement / rate-limit status
# to scorer_ab.csv. The live trade decision NEVER uses the Groq result — this is a
# pure forward measurement to see (a) is Groq consistently faster than local Ollama,
# (b) does the free tier rate-limit at our actual daily signal volume.
def _score_with_groq_timed(headline: str, body: str, source: str, model: str = None):
    """Returns (signal_dict|None, latency_ms, status). status ∈
    {ok, parse_fail, rate_limited, error:<Type>}."""
    prefix = f"[Source: {source}]\n" if source else ""
    t0 = time.time()
    try:
        resp = groq_client.chat.completions.create(
            model=model or GROQ_MODEL,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user",   "content": f"{prefix}Headline: {headline}\n\nBody: {body[:1200]}\n\nRespond with JSON only."},
            ],
            temperature=0.1,
        )
        ms = (time.time() - t0) * 1000
        raw = resp.choices[0].message.content.strip()
        if "<think>" in raw:                       # strip reasoning-model blocks
            raw = raw.split("</think>")[-1].strip()
        if raw.startswith("```"):
            raw = raw.split("```")[1]
            if raw.startswith("json"):
                raw = raw[4:]
        try:
            return json.loads(raw.strip()), ms, "ok"
        except json.JSONDecodeError:
            return None, ms, "parse_fail"
    except RateLimitError:
        return None, (time.time() - t0) * 1000, "rate_limited"
    except Exception as e:
        return None, (time.time() - t0) * 1000, f"error:{type(e).__name__}"

def _log_scorer_ab(source, o_sig, o_ms, g_sig, g_ms, status, headline, groq_model=""):
    """Append one Groq-vs-Ollama comparison row to scorer_ab.csv (own file → no schema risk).
    `groq_model` tags WHICH Groq model produced this row (one row per GROQ_AB_MODELS candidate).
    Logs each scorer's TICKERS so forward-return P&L (Ollama vs Groq as the PRIMARY scorer) can be
    attributed later by groq_vs_ollama_pnl.py — the data to validate moving live scoring to paid
    Groq in the cloud. Without the ticker the article can't be joined to a forward return."""
    def fld(d, k):
        return "" if not d else d.get(k, "")
    def tks(d):
        return ",".join(t for t in (d.get("tickers") or []) if t) if d else ""
    o_sent, g_sent = fld(o_sig, "sentiment"), fld(g_sig, "sentiment")
    path = "scorer_ab.csv"
    write_header = not os.path.exists(path)
    try:
        with open(path, "a", newline="") as f:
            w = csv.writer(f)
            if write_header:
                w.writerow(["timestamp", "source", "groq_model", "ollama_ms", "groq_ms", "groq_status",
                            "ollama_sentiment", "groq_sentiment", "sentiment_agree",
                            "ollama_mag", "groq_mag", "ollama_conf", "groq_conf",
                            "ollama_tickers", "groq_tickers", "headline"])
            w.writerow([
                datetime.now(timezone.utc).isoformat(), source, groq_model,
                f"{o_ms:.0f}", f"{g_ms:.0f}", status, o_sent, g_sent,
                "" if (not o_sig or not g_sig) else int(o_sent == g_sent),
                fld(o_sig, "magnitude"), fld(g_sig, "magnitude"),
                fld(o_sig, "confidence"), fld(g_sig, "confidence"),
                tks(o_sig), tks(g_sig),
                (headline or "")[:120],
            ])
    except Exception as e:
        log.debug("scorer A/B log failed: %s", e)

async def _groq_shadow_compare(headline, body, source, ollama_signal, ollama_ms):
    """create_task'd from process_signal → runs Groq in an executor so it NEVER delays the live
    (Ollama) trade path. Scores the article with EACH GROQ_AB_MODELS candidate (primary-scorer
    bake-off) → one scorer_ab.csv row per model: latency + agreement + rate-limit + tickers (P&L)."""
    if groq_client is None:
        return
    loop = asyncio.get_event_loop()
    for model in GROQ_AB_MODELS:
        try:
            g_sig, g_ms, status = await loop.run_in_executor(
                None, _score_with_groq_timed, headline, body, source, model)
        except Exception as e:
            g_sig, g_ms, status = None, 0.0, f"error:{type(e).__name__}"
        _log_scorer_ab(source, ollama_signal, ollama_ms, g_sig, g_ms, status, headline, model)
        short = model.split("/")[-1]
        if status == "rate_limited":
            log.info("⚖️  Groq A/B [%s]: RATE-LIMITED at free-tier (ollama was %.0fms)", short, ollama_ms)
        elif status == "ok":
            agree = (g_sig and ollama_signal
                     and g_sig.get("sentiment") == ollama_signal.get("sentiment"))
            log.info("⚖️  Groq A/B [%s]: groq=%.0fms vs ollama=%.0fms  (%s)",
                     short, g_ms, ollama_ms, "sentiment-agree" if agree else "sentiment-differ")
        else:
            log.info("⚖️  Groq A/B [%s]: %s  (groq=%.0fms vs ollama=%.0fms)", short, status, g_ms, ollama_ms)

# ── Hybrid scorer: Gemini frontier CONFIRM/VETO gate ───────────────────────────────────
# Active (not shadow): when Ollama's bullish signal would open a REAL leg, Gemini re-scores
# the SAME article. Confirm → trade; veto → shadow-only; cap/error → Ollama fallback. All
# free-tier — the daily cap latches a graceful fallback so the bot never pays.

def _gemini_roll_day():
    """Reset the Gemini daily counters + cap latch at each new UTC day (lazy, like the
    loss breaker). Logs the prior day's tally so cap-hit frequency is visible in the log."""
    today = datetime.now(timezone.utc).date()
    if _gemini_day["date"] != today:
        if _gemini_day["date"] is not None and _gemini_day["calls"]:
            log.info("📊 Gemini scorer (prev day %s): %d calls · %d confirm · %d veto · %d fallback%s",
                     _gemini_day["date"], _gemini_day["calls"], _gemini_day["confirms"],
                     _gemini_day["vetoes"], _gemini_day["fallbacks"],
                     "  ⚠️ CAP HIT" if _gemini_day["capped"] else "")
        _gemini_day.update(date=today, calls=0, confirms=0, vetoes=0, fallbacks=0, capped=False)


def _score_with_gemini(headline: str, body: str, source: str):
    """Blocking single-article Gemini score on the SAME unified_v1 prompt as Ollama (fair
    second opinion). Returns (signal_dict|None, status, latency_ms) where status ∈
    {ok, parse_fail, rate_limited, error:<Type>}. Run via run_in_executor off the loop."""
    prefix = f"[Source: {source}]\n" if source else ""
    t0 = time.time()
    try:
        resp = gemini_client.chat.completions.create(
            model=GEMINI_MODEL,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user",   "content": f"{prefix}Headline: {headline}\n\nBody: {body[:1200]}\n\nRespond with JSON only."},
            ],
            temperature=0.1,
        )
        ms = (time.time() - t0) * 1000
        raw = resp.choices[0].message.content.strip()
        if "<think>" in raw:
            raw = raw.split("</think>")[-1].strip()
        if raw.startswith("```"):
            raw = raw.split("```")[1]
            if raw.startswith("json"):
                raw = raw[4:]
        # Robust extract (matches the validated n=345 backtest parser): grab the JSON object even
        # when the model wraps it in prose or emits a stray leading "{}" — else graceful parse_fail.
        m = re.search(r"\{.*\}", raw, re.S)
        if not m:
            return None, "parse_fail", ms
        try:
            return json.loads(m.group(0)), "ok", ms
        except json.JSONDecodeError:
            return None, "parse_fail", ms
    except RateLimitError:
        return None, "rate_limited", (time.time() - t0) * 1000
    except Exception as e:
        return None, f"error:{type(e).__name__}", (time.time() - t0) * 1000


def _log_gemini_decision(source, ticker, o_sig, g_sig, decision, status, headline,
                         ollama_ms=0.0, gemini_ms=0.0):
    """One row per Gemini consultation → gemini_decisions.csv. The analytics surface for
    'is Gemini blocking genuinely bad picks?' (join veto rows to shadow_trades.csv P&L),
    'how often do we hit the free-tier cap?' (decision='fallback' rows), and the Ollama-vs-
    Gemini scoring LATENCY (ollama_ms / gemini_ms — gemini_ms=0 when no call was made, e.g.
    already capped)."""
    def fld(d, k):
        return "" if not d else d.get(k, "")
    try:
        write_header = not os.path.exists(GEMINI_DECISIONS_FILE)
        with open(GEMINI_DECISIONS_FILE, "a", newline="") as f:
            w = csv.writer(f)
            if write_header:
                w.writerow(["timestamp", "source", "ticker", "decision", "gemini_status",
                            "ollama_ms", "gemini_ms",
                            "ollama_sentiment", "ollama_mag", "ollama_conf",
                            "gemini_sentiment", "gemini_mag", "gemini_conf", "headline"])
            w.writerow([
                datetime.now(timezone.utc).isoformat(), source, ticker, decision, status,
                f"{ollama_ms:.0f}", f"{gemini_ms:.0f}",
                fld(o_sig, "sentiment"), fld(o_sig, "magnitude"), fld(o_sig, "confidence"),
                fld(g_sig, "sentiment"), fld(g_sig, "magnitude"), fld(g_sig, "confidence"),
                (headline or "")[:120],
            ])
    except Exception as e:
        log.debug("gemini decision log failed: %s", e)


def _gemini_gate(signal: dict, headline: str, body: str, source: str, ollama_ms: float = 0.0):
    """Decide whether Gemini CONFIRMS the Ollama bullish pick. Returns (decision, gemini_sig,
    gemini_ms) with decision ∈ {confirm, veto, fallback}. Blocking (HTTP) → call via
    run_in_executor. Updates the daily counters + writes gemini_decisions.csv (incl. Ollama-vs-
    Gemini latency). On the daily cap / rate-limit / any error it returns 'fallback' (trade on
    Ollama) so the free tier never blocks trading."""
    _gemini_roll_day()
    ticker = (signal.get("tickers") or [""])[0]

    # Free-tier guard: already capped (latched) or over the daily budget → Ollama fallback.
    if _gemini_day["capped"] or _gemini_day["calls"] >= GEMINI_DAILY_CAP:
        _gemini_day["fallbacks"] += 1
        _log_gemini_decision(source, ticker, signal, None, "fallback", "capped", headline, ollama_ms, 0.0)
        return "fallback", None, 0.0

    g_sig, status, g_ms = _score_with_gemini(headline, body, source)

    if status == "rate_limited":
        # Groq 429 = transient per-minute throttle → one-off fallback, don't disable the gate.
        # (Gemini's free-tier 429 was a daily wall → set GATE_LATCH_ON_RATELIMIT=1 to latch off.)
        if GATE_LATCH_ON_RATELIMIT:
            _gemini_day["capped"] = True
            log.warning("⚠️ %s RATE-LIMITED (%.0fms) — latching Ollama fallback for the rest of the UTC day", GATE_LABEL, g_ms)
        else:
            log.warning("⚠️ %s rate-limited (%.0fms) → Ollama fallback for this pick (transient)", GATE_LABEL, g_ms)
        _gemini_day["fallbacks"] += 1
        _log_gemini_decision(source, ticker, signal, None, "fallback", status, headline, ollama_ms, g_ms)
        return "fallback", None, g_ms
    if status != "ok" or not g_sig:
        _gemini_day["fallbacks"] += 1           # transient error → fallback (don't latch)
        _log_gemini_decision(source, ticker, signal, g_sig, "fallback", status, headline, ollama_ms, g_ms)
        return "fallback", None, g_ms

    _gemini_day["calls"] += 1
    g_sent = g_sig.get("sentiment")
    g_mag  = float(g_sig.get("magnitude", 0) or 0)
    confirm = (g_sent == "bullish" and g_mag >= GEMINI_CONFIRM_MIN_MAGNITUDE)
    decision = "confirm" if confirm else "veto"
    _gemini_day["confirms" if confirm else "vetoes"] += 1
    _log_gemini_decision(source, ticker, signal, g_sig, decision, status, headline, ollama_ms, g_ms)
    return decision, g_sig, g_ms


def _would_open_real_leg(signal: dict, headline: str, magnitude: float, confidence: float) -> bool:
    """True if this bullish signal would open a REAL (capital-committing) leg — the only case
    worth spending a Gemini call on. Mirrors the live leg gates: news_call (mag≥thresh),
    lotto (mag+conf), pead (earnings headline). Stock is shadow-only and qqq_macro is off, so
    those never make it 'real'. Sub-threshold signals only shadow → no Gemini needed."""
    if NEWS_CALL_ENABLED and magnitude >= NEWS_CALL_MIN_MAGNITUDE:
        return True
    if LOTTO_ENABLED and magnitude >= LOTTO_MIN_MAGNITUDE and confidence >= LOTTO_MIN_CONFIDENCE:
        return True
    if _is_earnings_headline(headline):          # pead (bullish earnings, uptrend already checked)
        return True
    return False


def _shadow_vetoed_pick(loop, ticker: str, signal: dict, option_usd: float,
                        headline: str, magnitude: float, confidence: float):
    """Gemini vetoed a real-trade-eligible Ollama pick → paper-track whichever real legs it
    WOULD have opened, so we can later measure (via gemini_decisions.csv ⋈ shadow_trades.csv)
    whether Gemini blocked genuine losers. No capital, no cap slot."""
    async def _run():
        if NEWS_CALL_SHADOW_ENABLED:
            await asyncio.wait_for(loop.run_in_executor(
                None, shadow_news_call_fn, ticker, signal, option_usd), timeout=30)
        if LOTTO_ENABLED and magnitude >= LOTTO_MIN_MAGNITUDE and confidence >= LOTTO_MIN_CONFIDENCE:
            sp = get_stock_price(ticker)
            if sp:
                await asyncio.wait_for(loop.run_in_executor(
                    None, shadow_lotto_fn, ticker, sp), timeout=30)
        if _is_earnings_headline(headline):
            await asyncio.wait_for(loop.run_in_executor(
                None, shadow_pead_fn, ticker, signal, option_usd), timeout=30)
    return _run()

# ── Materiality scorer — the LIVE lotto SELECTOR (v2, promoted 2026-06-12 after the THRESHOLD-GATE
# test on the bot's REAL logic: at the live materiality≥0.15 gate on the high-conviction set, v2
# trades the most P&L (+$15,686/Sh3.80, 10× 4x+) vs v3 +$13,957 and baseline +$13,911. v3 won the
# top-130 RANKING bake-off but is too restrictive at 0.15 (53% pass vs v2's 69%) → fires too rarely.
# KEEP IN SYNC with prompt_lab.PROMPTS["materiality_fewshot_v2"] — inline copy so the live bot
# doesn't import the research module. Score = magnitude × confidence (mirrors extract_score).
MATERIALITY_PROMPT = """You are a trading signal generator. Score news by how much UNPRICED
SURPRISE it contains — information the market has NOT already priced in. Most news is routine and
already priced (magnitude <0.2). Only a genuine, unexpected EVENT earns a high magnitude.

CRITICAL RULE: an analyst rating, price-target change, or "maintains/reiterates" note is NOT
surprise — it is routine opinion the market already discounts. Score magnitude LOW even when
bullish. The surprise that MOVES a stock is a concrete EVENT (earnings beat, M&A, FDA, a trading
halt, a guidance change), not someone's opinion about the stock.

Few-shot examples (headline → the magnitude/confidence you should output):
- "Wells Fargo Maintains Overweight on JM Smucker, Raises Price Target to $130" → magnitude 0.12, confidence 0.85 (routine analyst note — already priced)
- "Guggenheim Reiterates Buy on Zscaler, Maintains $214 Target" → magnitude 0.10, confidence 0.85 (no new event)
- "Acme Corp Reports Q3 EPS $2.10 vs $1.60 est, Raises FY Guidance" → magnitude 0.82, confidence 0.90 (large unexpected beat + raise)
- "XYZ Shares Halted, Circuit Breaker To The Upside" → magnitude 0.88, confidence 0.85 (a violent unpriced move underway)
- "BioCo Receives Surprise FDA Approval for Lead Drug" → magnitude 0.85, confidence 0.90 (binary catalyst resolved favorably)
- "MegaCorp to Acquire SmallCo for $50/share, 40% Premium" → magnitude 0.90, confidence 0.95 (M&A at premium)
- "Company Announces Routine $1B Buyback" → magnitude 0.12, confidence 0.80 (expected capital return)
- "5 Stocks Investors Should Watch" / macro / political → magnitude 0.0, confidence 0.0, sentiment neutral (no specific unpriced event)

Respond JSON ONLY:
{"tickers":["AAPL"], "sentiment":"bullish", "confidence":0.7, "magnitude":0.6, "reasoning":"one sentence"}
magnitude (0–1): proportion of this news NOT already priced. confidence (0–1): direction certainty.
Return ONLY the JSON object."""

def score_materiality(headline: str, body: str) -> Optional[float]:
    """Second llama3.2 pass with the materiality prompt → magnitude×confidence in [0,1].
    Best-effort: returns None on any failure (must never break lotto trading)."""
    try:
        resp = ollama_client.chat.completions.create(
            model=OLLAMA_MODEL,
            messages=[
                {"role": "system", "content": MATERIALITY_PROMPT},
                {"role": "user",   "content": f"Headline: {headline}\n\nBody: {body[:1200]}\n\nJSON only."},
            ],
            temperature=0.1,
        )
        raw = resp.choices[0].message.content.strip()
        if "<think>" in raw:
            raw = raw.split("</think>")[-1].strip()
        if raw.startswith("```"):
            raw = raw.split("```")[1].lstrip("json").strip()
        obj = json.loads(raw)
        return float(obj.get("magnitude", 0)) * float(obj.get("confidence", 0.5) or 0.5)
    except Exception as e:
        log.debug("materiality scoring failed: %s", e)
        return None

def log_lotto_materiality(ticker: str, option_symbol: str, materiality: Optional[float],
                          mag: float, conf: float):
    """Append the materiality A/B record for a lotto entry (own file → no schema risk)."""
    path = "lotto_materiality.csv"
    write_header = not os.path.exists(path)
    try:
        with open(path, "a", newline="") as f:
            w = csv.writer(f)
            if write_header:
                w.writerow(["timestamp", "ticker", "option_symbol", "materiality",
                            "magnitude", "confidence", "gate", "passed_gate"])
            passed = "" if materiality is None else int(materiality >= MATERIALITY_GATE)
            w.writerow([
                datetime.now(timezone.utc).isoformat(), ticker, option_symbol,
                "" if materiality is None else f"{materiality:.4f}",
                f"{mag:.4f}", f"{conf:.4f}", f"{MATERIALITY_GATE:.2f}", passed,
            ])
    except Exception as e:
        log.debug("log_lotto_materiality failed: %s", e)

def _log_lotto_cap(symbol, pos, entry, trigger_mid, fill, qty, pnl_usd):
    """Forward log of live lotto 3× cap exits — records the trigger premium (the 30s-cycle mid
    that crossed the cap) vs the ACTUAL fill, so we can confirm the cap fills near 3× live (the
    daily-bar backtest couldn't show intraday-fill quality). Own file → no schema risk."""
    path = "lotto_cap_log.csv"
    write_header = not os.path.exists(path)
    try:
        with open(path, "a", newline="") as f:
            w = csv.writer(f)
            if write_header:
                w.writerow(["timestamp", "symbol", "underlying", "qty", "entry_premium",
                            "trigger_mid", "fill_premium", "cap_mult", "trigger_mult",
                            "realized_mult", "slippage_pct", "pnl_usd"])
            trig_mult = (trigger_mid / entry) if entry else 0
            real_mult = (fill / entry) if entry else 0
            slip = ((fill - trigger_mid) / trigger_mid * 100) if trigger_mid else 0
            w.writerow([
                datetime.now(timezone.utc).isoformat(), symbol, pos.get("underlying", ""), qty,
                f"{entry:.4f}", f"{trigger_mid:.4f}", f"{fill:.4f}", f"{LOTTO_HARD_CAP_MULT:.1f}",
                f"{trig_mult:.3f}", f"{real_mult:.3f}", f"{slip:.2f}", f"{pnl_usd:.2f}",
            ])
    except Exception as e:
        log.debug("lotto cap log failed: %s", e)

# ═══════════════════════════════════════════════════════════════════════════════
# SIGNAL DISPATCHER
# ═══════════════════════════════════════════════════════════════════════════════

def signal_checks(signal: dict) -> tuple[bool, float, float]:
    """
    Validate signal against magnitude gate and dynamic confidence floor.
    Returns (passes, magnitude, required_confidence).
    """
    magnitude  = float(signal.get("magnitude", 0))
    confidence = float(signal.get("confidence", 0))

    if magnitude < MIN_MAGNITUDE:
        log.info(
            "  → magnitude %.2f below gate %.2f — skipping (not a meaningful catalyst)",
            magnitude, MIN_MAGNITUDE,
        )
        return False, magnitude, 0

    required_confidence = BASE_CONFIDENCE + (1 - magnitude) * CONFIDENCE_SLOPE
    if confidence < required_confidence:
        log.info(
            "  → confidence %.2f below dynamic floor %.2f (magnitude=%.2f requires ≥%.2f)",
            confidence, required_confidence, magnitude, required_confidence,
        )
        return False, magnitude, required_confidence

    return True, magnitude, required_confidence

def scale_position_usd(base_usd: float, magnitude: float, confidence: float) -> float:
    """Position size. Magnitude is UNCALIBRATED (missed_movers rank-IC≈0; loser avg mag 0.77 ≈
    winner 0.78), so magnitude-scaling is false precision — when MAGNITUDE_SIZING_ENABLED is off
    we size FLAT at NONMAG_SIZE_FRAC×base (≈ the old average mag×conf, so avg exposure is ~flat)."""
    if not MAGNITUDE_SIZING_ENABLED:
        return base_usd * NONMAG_SIZE_FRAC
    scaled = base_usd * magnitude * confidence
    # Floor at 10% of max so we never place a trivially small order
    return max(base_usd * 0.10, scaled)

async def process_signal(headline: str, body: str, source: str,
                         article_ts: Optional[datetime] = None):
    """Score a headline and dispatch trades if signal is strong.
    article_ts: original publish timestamp from the news source (for lag measurement)."""
    received_at = datetime.now(timezone.utc)
    if article_ts and article_ts.tzinfo is not None:
        feed_lag_s = (received_at - article_ts).total_seconds()
        log.info("⏱  Feed lag: %.1fs  (article %s → received %s)  [%s]",
                 feed_lag_s, article_ts.strftime("%H:%M:%S"),
                 received_at.strftime("%H:%M:%S"), source)
        # Stale-news drop: skip BEFORE the ~3-8s scoring so a backlog can't compound (the move is
        # already gone on hour-old news anyway). Keeps the bot on live news; bounds the lag.
        if feed_lag_s > MAX_FEED_LAG_SECS:
            log.info("  → stale news (%.0fs > %ds max) — skipping (stay live / clear backlog)",
                     feed_lag_s, MAX_FEED_LAG_SECS)
            return

    if not market_is_open():
        log.debug("  → market closed, skipping")
        return

    # Pre-score noise filter: drop high-confidence-noise (macro wraps / listicles / reactive
    # recaps) BEFORE scoring — saves the Ollama + Groq-A/B calls and reduces backlog pressure.
    if PRESCORE_FILTER_ENABLED:
        noise = _prescore_noise(headline)
        if noise:
            log.info("  → pre-score noise filter (%r) — skipping (no Ollama/Groq call)", noise)
            return

    # Soft-catalyst PRE-SCORE drop (promoted 2026-06-22 from the old post-score gate after
    # prefilter_pnl validated it: the removed set loses -$4,038, 95% boot CI < 0). Analyst/PT,
    # technical, acquirer-M&A headlines never warrant a model call — drop before scoring
    # (~11% of all articles → big Ollama+Groq savings). No shadow now; the backtest is the proof.
    if SOFT_CATALYST_GATE_ENABLED:
        soft = _soft_catalyst_hit(headline, body)
        if soft:
            log.info("  🚧 soft-catalyst (%r) — pre-score drop (no Ollama/Groq call)", soft)
            return

    # NOTE: the regime filter is applied PER-STRATEGY below, not here. Long-beta
    # strategies (news_call / qqq_macro / stock / pead) only fire in an uptrend
    # (index > 200d SMA). The bear_short and market-neutral pairs strategies run
    # in ALL regimes, so we must still score the article during a downtrend.
    score_start = datetime.now(timezone.utc)
    signal = score_with_ollama(headline, body, source)
    score_ms = (datetime.now(timezone.utc) - score_start).total_seconds() * 1000
    log.info("⏱  Score time: %.0fms  (total from receipt: %.1fs)",
             score_ms, (datetime.now(timezone.utc) - received_at).total_seconds())

    # Groq scorer A/B (shadow): score the same headline on Groq off the trade path and
    # log latency / agreement / rate-limit status. Fire-and-forget — never awaited here,
    # so it cannot delay the live (Ollama-driven) trade decision below.
    if groq_client is not None:
        asyncio.create_task(_groq_shadow_compare(headline, body, source, signal, score_ms))

    if not signal:
        return

    magnitude  = float(signal.get("magnitude", 0))
    confidence = float(signal.get("confidence", 0))

    log.info(
        "  🤖 %s: %s  conf=%.2f  mag=%.2f  tickers=%s",
        OLLAMA_MODEL, signal.get("sentiment"), confidence,
        magnitude, signal.get("tickers"),
    )
    log.info("     %s", signal.get("reasoning", ""))

    sentiment = signal.get("sentiment")

    # ── Bearish path: short the stock ────────────────────────────────────────
    if sentiment == "bearish":
        passes, magnitude, req_conf = signal_checks(signal)
        if not passes:
            return
        loop = asyncio.get_event_loop()
        # Feed the L/S pairs tracker, then try to open a pair (runs in any regime)
        record_pairs_signal("bear", signal, magnitude, confidence)
        await maybe_enter_pairs(loop)
        try:
            log_router_decision("bear", signal, magnitude, confidence, headline)
        except Exception as e:
            log.warning("Router decision (bear) failed: %s", e)

        # bear_short RETIRED (BEAR_SHORT_ENABLED=False): edge over random shorts is negligible
        # (~85% of its bear-window P&L is beta, not selection — see legs_unified_backtest). The
        # pairs tracker above still records the bear signal so the dormant pairs leg stays intact.
        if BEAR_SHORT_ENABLED:
            short_usd = scale_position_usd(MAX_POSITION_USD, magnitude, confidence)
            log.info(
                "  💰 Bear-short sizing: mag=%.2f × conf=%.2f → short=$%.0f",
                magnitude, confidence, short_usd,
            )
            for ticker in signal.get("tickers", [])[:2]:
                if on_cooldown(ticker) or ticker in ("BTC", "ETH"):
                    continue
                try:
                    await asyncio.wait_for(
                        loop.run_in_executor(None, execute_bear_short_fn, ticker, signal, short_usd),
                        timeout=30,
                    )
                except asyncio.TimeoutError:
                    log.error("execute_bear_short_fn timed out for %s — skipping", ticker)
        return

    if sentiment != "bullish":
        log.info("  → neutral/unknown sentiment, skipping")
        return

    passes, magnitude, req_conf = signal_checks(signal)
    if not passes:
        return

    loop = asyncio.get_event_loop()
    # Feed the L/S pairs tracker, then try to open a pair (runs in any regime)
    record_pairs_signal("bull", signal, magnitude, confidence)
    await maybe_enter_pairs(loop)
    try:
        log_router_decision("bull", signal, magnitude, confidence, headline)
    except Exception as e:
        log.warning("Router decision (bull) failed: %s", e)

    # Regime gate: the long-only beta strategies (news_call / qqq_macro / stock /
    # pead) are leveraged long-beta — only open them in an uptrend (index > 200d
    # SMA). Pairs/bear_short above already ran regardless of regime.
    regime_bypass = False
    if not market_in_uptrend():
        # High-conviction bypass: strong idiosyncratic catalysts (mag ≥ threshold) clear the regime
        # filter even in a downtrend — backtested to stay profitable on options in down-regimes incl.
        # the 2022 bear. BUT a downtrend bypass now REQUIRES an explicit CONFIRM from the secondary
        # (mistral-large) gate: on a gate timeout/cap/error we do NOT fall back to trading on Ollama
        # alone (enforced just before the trade loop below). 2026-06-25: a gate TimeoutError let a
        # marginal "Boeing wins $2B contract" call through this path and lost -34% same day.
        if magnitude >= REGIME_BYPASS_MIN_MAGNITUDE:
            regime_bypass = True
            log.info("  ⚡ downtrend BUT high-conviction (mag=%.2f ≥ %.2f) → bypassing regime filter "
                     "(gate CONFIRM required — no Ollama-fallback trade)", magnitude, REGIME_BYPASS_MIN_MAGNITUDE)
        else:
            log.info("  → downtrend (regime filter) — skipping long-beta strategies "
                     "(pairs/bear_short still active)")
            return

    # Compute scaled position size once for this signal
    option_usd = scale_position_usd(MAX_POSITION_USD, magnitude, confidence)
    log.info(
        "  💰 Position sizing: mag=%.2f × conf=%.2f → options=$%.0f",
        magnitude, confidence, option_usd,
    )

    tickers = signal.get("tickers", [])
    if not tickers:
        # Macro / market-wide bullish signal with no specific name. Was routed to a QQQ call
        # (qqq_macro). DISABLED 2026-06-14: qqq_macro_eval + qqq_macro_geom_sweep on unified_v1
        # showed the macro-bullish→QQQ signal has NO tradeable edge as an option — net −$2,336
        # (frictionless ≈ $0, i.e. just paying spread), and NO Δ×DTE geometry (25-cell sweep) is
        # robustly positive (the few positives are melt-up beta-lottery, bootstrap CI spans 0).
        # The QQQ-stock version was only +$184 (trivial). So we drop the leg and just log+skip
        # macro signals (as before qqq_macro existed), freeing the 30-slot cap for legs with edge.
        if not QQQ_MACRO_ENABLED:
            log.info("  → macro-bullish (no ticker) → qqq_macro disabled, skipping")
            return
        # Cooldown on QQQ prevents stacking.
        if on_cooldown("QQQ"):
            log.info("  → macro-bullish, QQQ on cooldown — skipping")
            return
        log.info("  → macro-bullish (no ticker) → routing to QQQ call [qqq_macro]")
        loop = asyncio.get_event_loop()
        try:
            await asyncio.wait_for(
                loop.run_in_executor(None, execute_equity_trade, "QQQ", signal, option_usd, "qqq_macro"),
                timeout=30,
            )
        except asyncio.TimeoutError:
            log.error("qqq_macro trade timed out — skipping")
        return

    # (Soft-catalyst is now dropped PRE-score above — no post-score gate needed.)

    # ── Hybrid scorer: Gemini frontier CONFIRM/VETO on REAL-trade-eligible bullish signals ──
    # Ollama already picked; if this would open a real leg, get Gemini's second opinion.
    # confirm → trade (tag gemini) · veto → shadow-only + return · cap/err → Ollama fallback.
    signal["_scorer"] = "ollama"
    if (gemini_client is not None
            and _would_open_real_leg(signal, headline, magnitude, confidence)):
        try:
            decision, _g, g_ms = await asyncio.wait_for(
                loop.run_in_executor(None, _gemini_gate, signal, headline, body, source, score_ms),
                timeout=12)
        except Exception as e:
            decision, g_ms = "fallback", 0.0
            _gemini_day["fallbacks"] += 1
            log.warning("%s gate error/timeout (%s) → Ollama fallback", GATE_LABEL, type(e).__name__)
        if decision == "veto":
            log.info("  🚫 %s VETO on %s (Ollama mag=%.2f conf=%.2f) → shadow-only, no real trade "
                     "(gate %.0fms vs ollama %.0fms)", GATE_LABEL, tickers[:2], magnitude, confidence, g_ms, score_ms)
            signal["_scorer"] = "gemini_veto"
            if GEMINI_VETO_SHADOW_ENABLED:
                for tk in tickers[:2]:
                    if tk in ("BTC", "ETH") or on_cooldown(tk):
                        continue
                    try:
                        await _shadow_vetoed_pick(loop, tk, signal, option_usd,
                                                  headline, magnitude, confidence)
                    except Exception as e:
                        log.debug("veto-shadow %s failed: %s", tk, e)
            return
        signal["_scorer"] = "gemini" if decision == "confirm" else "ollama_fallback"
        if decision == "confirm":
            log.info("  ✅ %s CONFIRMS %s → trading [scorer=gemini]  (gate %.0fms vs ollama %.0fms)",
                     GATE_LABEL, tickers[:2], g_ms, score_ms)
        else:
            log.info("  ↩️  %s fallback on %s → trading on Ollama [scorer=ollama_fallback]  (ollama %.0fms)",
                     GATE_LABEL, tickers[:2], score_ms)

    # Downtrend bypass safety: only trade into a downtrend when the secondary model EXPLICITLY confirmed
    # (_scorer == "gemini"). A gate timeout/cap/error (→ ollama_fallback) or a disabled gate (→ ollama)
    # must NOT trade here — fail CLOSED, unlike the normal uptrend path which keeps trading on Ollama when
    # the gate is unavailable. (A veto already returned above, so this only catches the no-confirm cases.)
    if regime_bypass and signal.get("_scorer") != "gemini":
        log.warning("  ⛔ downtrend bypass NOT confirmed by %s (scorer=%s) → skipping: no trade into a "
                    "downtrend without the secondary model", GATE_LABEL, signal.get("_scorer"))
        return

    for ticker in tickers[:2]:
        if on_cooldown(ticker):
            log.info("  → %s on cooldown", ticker)
            continue

        # Crypto is no longer traded — skip any BTC/ETH signals.
        if ticker in ("BTC", "ETH"):
            log.info("  → %s is crypto, skipping (crypto trading disabled)", ticker)
            continue

        loop = asyncio.get_event_loop()

        # ── Stock trade — SHADOW-ONLY since 2026-06-15 (STOCK_ENABLED=False). The real stock leg
        # lost to SPY/QQQ over its live window (stock_vs_index_live.py), so we stopped committing
        # real capital/cap-slots to it but keep PAPER-tracking every stock signal for forward data.
        # Re-enable by flipping STOCK_ENABLED. (Real path placed first historically so research
        # shadows below never delayed the live order; with real off, the shadow runs here.)
        if STOCK_ENABLED:
            try:
                await asyncio.wait_for(
                    loop.run_in_executor(None, execute_stock_trade_fn, ticker, signal, option_usd, "stock"),
                    timeout=30,
                )
            except asyncio.TimeoutError:
                log.error("execute_stock_trade_fn timed out for %s after 30s — skipping", ticker)
        elif STOCK_SHADOW_ENABLED:
            try:
                await asyncio.wait_for(
                    loop.run_in_executor(None, shadow_stock_fn, ticker, signal, option_usd),
                    timeout=30,
                )
            except asyncio.TimeoutError:
                log.error("shadow_stock_fn timed out for %s after 30s — skipping", ticker)

        # ── news_call: standard ATM (Δ0.50) options ───────────────────────
        # RE-ENABLED 2026-06-14 at the unified_v1 SWEET-SPOT gate (mag≥NEWS_CALL_MIN_MAGNITUDE
        # 0.75; conf already ≥ the dynamic floor from signal_checks above). Was disabled
        # 2026-06-05 (-$6,601/13% win live) when realistic option costs ate the edge on the
        # OLD scorer; news_call_sweep_unified.py showed unified_v1's name-selection makes it
        # net-positive after costs (~333 tr, $105/tr, Sharpe 1.76, bootstrap CI>0). The mag≥0.75
        # floor takes only strong-catalyst signals → cost-survivable AND lighter on the shared
        # 30-slot cap. Weaker (mag<0.75) signals fall through to the shadow leg below. Geometry
        # is the place_option_trade default (DTE 14-21, Δ0.50) — matches the swept config.
        if NEWS_CALL_ENABLED and magnitude >= NEWS_CALL_MIN_MAGNITUDE:
            try:
                await asyncio.wait_for(
                    loop.run_in_executor(None, execute_equity_trade, ticker, signal, option_usd, "news_call"),
                    timeout=30,
                )
            except asyncio.TimeoutError:
                log.error("execute_equity_trade timed out for %s after 30s — skipping", ticker)
        elif NEWS_CALL_SHADOW_ENABLED:
            # news_call not taken (disabled OR mag<0.75) → SHADOW it (paper, no order) for
            # forward data + delta-0.50 contract-selection samples, without spending capital.
            # Runs AFTER the real stock entry above (research telemetry must not delay the order).
            try:
                await asyncio.wait_for(
                    loop.run_in_executor(None, shadow_news_call_fn, ticker, signal, option_usd),
                    timeout=30,
                )
            except asyncio.TimeoutError:
                log.error("shadow_news_call_fn timed out for %s after 30s — skipping", ticker)

        # ── PEAD trade — stock with tiered trail + breakeven lock ─────────────
        # Post-Earnings-Announcement-Drift: ONLY fires on earnings-related headlines
        # (beat/results/guidance), matching pead_backtest.py (earnings_filter=True) —
        # that is the strategy whose tiered-trail params were validated. _is_earnings_headline
        # mirrors the backtest's is_earnings_article exactly (verified in-sync). Without
        # this gate PEAD was firing on EVERY bullish signal → just a duplicate of `stock`,
        # not the backtested earnings-drift strategy.
        if _is_earnings_headline(headline):
            try:
                await asyncio.wait_for(
                    loop.run_in_executor(None, execute_pead_trade_fn, ticker, signal, option_usd),
                    timeout=30,
                )
            except asyncio.TimeoutError:
                log.error("execute_pead_trade_fn timed out for %s after 30s — skipping", ticker)

        # ── LOTTO trade — cheap OTM call, ONLY on the strongest bullish signals ──
        # (high magnitude AND high confidence → likely big swing). Small convex bet.
        if (LOTTO_ENABLED and magnitude >= LOTTO_MIN_MAGNITUDE
                and confidence >= LOTTO_MIN_CONFIDENCE):
            log.info("  🎟  high-conviction (mag=%.2f conf=%.2f) → adding LOTTO leg on %s",
                     magnitude, confidence, ticker)
            try:
                await asyncio.wait_for(
                    loop.run_in_executor(None, execute_lotto_trade_fn, ticker, signal, headline, body),
                    timeout=30,
                )
            except asyncio.TimeoutError:
                log.error("execute_lotto_trade_fn timed out for %s after 30s — skipping", ticker)

# ═══════════════════════════════════════════════════════════════════════════════
# NEWS SOURCES
# ═══════════════════════════════════════════════════════════════════════════════

# ── 1. Alpaca / Benzinga WebSocket ────────────────────────────────────────────

async def handle_alpaca_news(news):
    for item in (news if isinstance(news, list) else [news]):
        news_id = getattr(item, "id", None) or str(item)
        if news_id in _seen_news_ids:
            continue
        _seen_news_ids.add(news_id)
        headline = getattr(item, "headline", "") or ""
        body     = getattr(item, "summary", "") or ""
        # Parse article publish time for feed-lag measurement
        raw_ts = getattr(item, "created_at", None) or getattr(item, "updated_at", None)
        article_ts = None
        if raw_ts:
            try:
                if isinstance(raw_ts, datetime):
                    article_ts = raw_ts if raw_ts.tzinfo else raw_ts.replace(tzinfo=timezone.utc)
                else:
                    article_ts = datetime.fromisoformat(str(raw_ts).replace("Z", "+00:00"))
            except Exception:
                pass
        log.info("📰 [Alpaca] %s", headline[:100])
        await process_signal(headline, body, "Alpaca/Benzinga", article_ts=article_ts)

# ── 2. SEC EDGAR 8-K RSS (poll) ───────────────────────────────────────────────

async def sec_rss_poller():
    log.info("📡 SEC EDGAR 8-K RSS poller started (every %ds)", SEC_POLL_SECS)
    while True:
        try:
            feed = feedparser.parse(SEC_RSS_URL, agent=SEC_USER_AGENT)
            for entry in feed.entries:
                eid = entry.get("id", entry.get("link", ""))
                if eid in _seen_news_ids:
                    continue
                _seen_news_ids.add(eid)
                title   = entry.get("title", "")
                summary = entry.get("summary", "")
                log.info("📰 [SEC 8-K] %s", title[:100])
                await process_signal(title, summary, "SEC EDGAR 8-K")
        except Exception as e:
            log.warning("SEC RSS poll error: %s", e)
        await asyncio.sleep(SEC_POLL_SECS)

# ── 3. NewsAPI (poll) ─────────────────────────────────────────────────────────

async def newsapi_poller():
    if not NEWSAPI_KEY:
        log.info("ℹ️  NEWSAPI_KEY not set — NewsAPI poller disabled")
        return
    log.info("📡 NewsAPI poller started (every %ds)", NEWSAPI_POLL_SECS)
    async with aiohttp.ClientSession() as session:
        while True:
            try:
                params = {**NEWSAPI_PARAMS, "apiKey": NEWSAPI_KEY}
                async with session.get(NEWSAPI_URL, params=params, timeout=aiohttp.ClientTimeout(total=10)) as r:
                    data = await r.json()
                    for article in data.get("articles", []):
                        uid = article.get("url", "")
                        if uid in _seen_news_ids:
                            continue
                        _seen_news_ids.add(uid)
                        title = article.get("title", "") or ""
                        body  = article.get("description", "") or ""
                        log.info("📰 [NewsAPI] %s", title[:100])
                        await process_signal(title, body, "NewsAPI")
            except Exception as e:
                log.warning("NewsAPI poll error: %s", e)
            await asyncio.sleep(NEWSAPI_POLL_SECS)

# ═══════════════════════════════════════════════════════════════════════════════
# TRADE LOG
# ═══════════════════════════════════════════════════════════════════════════════

def log_trade(ticker, contract, qty, signal, entry_price, strategy="news_call"):
    path = "trades.csv"
    write_header = not os.path.exists(path)
    with open(path, "a", newline="") as f:
        w = csv.writer(f)
        if write_header:
            w.writerow([
                "timestamp", "source_ticker", "option_symbol", "strike", "expiry",
                "qty", "bid", "ask", "cost_basis", "max_loss",
                "spread_pct", "trailing_stop_pct",
                "confidence", "magnitude", "reasoning", "strategy", "scorer",
            ])
        w.writerow([
            datetime.now(timezone.utc).isoformat(),
            ticker, contract["symbol"], contract["strike"], contract["expiry"],
            qty, f"{contract['bid']:.4f}", f"{contract['ask']:.4f}",
            f"{entry_price:.4f}",
            f"{entry_price * 100 * qty:.2f}",
            f"{contract['spread_pct'] * 100:.1f}%",
            f"{TRAILING_STOP_PCT * 100:.0f}%",
            signal.get("confidence"), signal.get("magnitude"),
            signal.get("reasoning", ""), strategy, signal.get("_scorer", "ollama"),
        ])

CLOSED_TRADES_FILE = "closed_trades.csv"

def log_closed_trade(symbol: str, pos: dict, exit_price: float, pnl_usd: float, reason: str = "stop"):
    """Append one row to closed_trades.csv when a monitored position is confirmed closed."""
    path = CLOSED_TRADES_FILE
    write_header = not os.path.exists(path)
    asset_type = pos.get("asset_type", "option")
    entry = pos["entry_price"]
    pnl_pct = ((exit_price - entry) / entry * 100) if entry > 0 and asset_type == "stock" else (
               (exit_price / entry - 1) * 100 if entry > 0 else 0)
    with open(path, "a", newline="") as f:
        w = csv.writer(f)
        if write_header:
            w.writerow(["timestamp", "symbol", "underlying", "strategy", "asset_type",
                        "qty", "entry_price", "exit_price", "pnl_usd", "pnl_pct",
                        "peak_price", "reason"])
        w.writerow([
            datetime.now(timezone.utc).isoformat(),
            symbol,
            pos.get("underlying", symbol),
            pos.get("strategy", "unknown"),
            asset_type,
            pos.get("qty", 0),
            f"{entry:.4f}",
            f"{exit_price:.4f}",
            f"{pnl_usd:.2f}",
            f"{pnl_pct:.2f}",
            f"{pos.get('peak_price', entry):.4f}",
            reason,
        ])


def log_trade_stock(ticker: str, shares: int, entry_price: float, signal: dict, strategy: str = "stock"):
    path = "trades.csv"
    write_header = not os.path.exists(path)
    with open(path, "a", newline="") as f:
        w = csv.writer(f)
        if write_header:
            w.writerow([
                "timestamp", "source_ticker", "option_symbol", "strike", "expiry",
                "qty", "bid", "ask", "cost_basis", "max_loss",
                "spread_pct", "trailing_stop_pct",
                "confidence", "magnitude", "reasoning", "strategy", "scorer",
            ])
        w.writerow([
            datetime.now(timezone.utc).isoformat(),
            ticker, "-", "-", "-",
            shares, f"{entry_price:.4f}", f"{entry_price:.4f}",
            f"{entry_price:.4f}",
            f"{entry_price * shares:.2f}",
            "0.0%",
            f"{STOCK_TRAIL_PCT * 100:.0f}%",
            signal.get("confidence"), signal.get("magnitude"),
            signal.get("reasoning", ""), strategy, signal.get("_scorer", "ollama"),
        ])

# ═══════════════════════════════════════════════════════════════════════════════
# ENTRY POINT
# ═══════════════════════════════════════════════════════════════════════════════

async def main():
    log.info("🚀 News trading bot starting (paper mode · model: %s)", OLLAMA_MODEL)
    log.info("   Magnitude gate       : %.2f",      MIN_MAGNITUDE)
    log.info("   Base confidence      : %.2f",      BASE_CONFIDENCE)
    log.info("   Confidence slope     : %.2f",      CONFIDENCE_SLOPE)
    log.info("   Position sizing      : %s", "FLAT %.0f%% of $%d (magnitude uncalibrated)" %
             (NONMAG_SIZE_FRAC * 100, MAX_POSITION_USD) if not MAGNITUDE_SIZING_ENABLED
             else "magnitude×confidence of $%d" % MAX_POSITION_USD)
    log.info("   Soft-catalyst filter : %s", "ON — analyst-PT/technical/acquirer-M&A dropped PRE-score"
             if SOFT_CATALYST_GATE_ENABLED else "OFF")
    log.info("   Pre-score noise filter: %s", "ON" if PRESCORE_FILTER_ENABLED else "OFF")
    log.info("   Daily loss limit     : %.0f%% of equity ≈ $%.0f/day (floor $%d), resets 00:00 UTC",
             DAILY_LOSS_LIMIT_PCT * 100, _daily_loss_limit(), DAILY_LOSS_LIMIT)
    if gemini_client is not None:
        log.info("   Confirm/veto gate    : Ollama → %s(%s) · cap %d/day (high-quota, Ollama fallback)",
                 GATE_LABEL, GEMINI_MODEL, GEMINI_DAILY_CAP)
    else:
        log.info("   Confirm/veto gate    : OFF (pure Ollama — gate keys absent or disabled)")
    log.info("   Spread filter        : ≤%.0f%%",   MAX_SPREAD_PCT * 100)
    log.info("   Min open interest    : %d",         MIN_OPEN_INTEREST)
    log.info("   Trailing stop        : %.0f%% below peak", TRAILING_STOP_PCT * 100)
    log.info("   Monitor interval     : %ds",        MONITOR_INTERVAL)
    log.info("   Tradeable ETFs       : %s",         ", ".join(sorted(TRADEABLE_ETFS)))

    # Load existing open positions from Alpaca before starting monitor
    log.info("📂 Loading open positions from Alpaca…")
    load_open_positions_from_alpaca()
    _load_shadow_state()   # reload paper shadow positions (news_call_shadow) across restarts

    # Pre-subscribe real-time prices for ETFs + liquid watchlist
    # so the cache is warm before the first signal arrives
    PRICE_WATCHLIST = sorted(TRADEABLE_ETFS | {
        "AAPL", "MSFT", "NVDA", "AMZN", "GOOGL", "META", "TSLA",
        "JPM", "GS", "MS", "BAC", "AMD", "INTC", "QCOM", "MU",
        "ORCL", "CRM", "ADBE", "NOW", "SHOP",
    })
    subscribe_price_stream(PRICE_WATCHLIST)
    log.info("📡 Pre-subscribed real-time prices for %d symbols", len(PRICE_WATCHLIST))

    # Start Alpaca news WebSocket
    stream = NewsDataStream(ALPACA_KEY, ALPACA_SECRET)
    stream.subscribe_news(handle_alpaca_news, "*")

    log.info("📡 All news sources starting…")
    await asyncio.gather(
        stream._run_forever(),
        stock_stream._run_forever(),
        trailing_stop_monitor(),
        sec_rss_poller(),
        newsapi_poller(),
    )

if __name__ == "__main__":
    asyncio.run(main())