"""
market.py (v2) — Alpaca clients, quotes, contract selection, order execution, position
persistence, and the trailing-stop monitor. Ported from core/bot.py with the hard-won fixes
kept intact (marketable-limit-at-ask option entry/exit, close-only-deregisters-on-confirmed-
success, market-halt handling, data-quality guards on option quotes) but with:
  - NO short-side code at all (pairs/bear_short both dropped — v2 never shorts)
  - NO shadow-position tracking (see config.py module docstring)
  - Equity-relative position sizing instead of a fixed dollar constant (see position_budget_usd)
"""

import csv
import json
import logging
import time
from datetime import date, datetime, timedelta, timezone
from typing import Optional

from alpaca.data.live import StockDataStream
from alpaca.data.historical.option import OptionHistoricalDataClient
from alpaca.data.historical import StockHistoricalDataClient
from alpaca.data.requests import (
    OptionLatestQuoteRequest, StockLatestTradeRequest, StockBarsRequest,
)
from alpaca.data.timeframe import TimeFrame
from alpaca.trading.client import TradingClient
from alpaca.trading.enums import AssetClass, OrderSide, TimeInForce, ContractType, PositionIntent
from alpaca.trading.requests import (
    GetOptionContractsRequest, LimitOrderRequest, MarketOrderRequest, GetOrdersRequest,
)
from alpaca.trading.enums import QueryOrderStatus

import config as cfg

log = logging.getLogger(__name__)

# ═══════════════════════════════════════════════════════════════════════════════
# CLIENTS
# ═══════════════════════════════════════════════════════════════════════════════

trading_client     = TradingClient(cfg.ALPACA_KEY, cfg.ALPACA_SECRET, paper=True)
option_data_client = OptionHistoricalDataClient(cfg.ALPACA_KEY, cfg.ALPACA_SECRET)
stock_data_client  = StockHistoricalDataClient(cfg.ALPACA_KEY, cfg.ALPACA_SECRET)
stock_stream       = StockDataStream(cfg.ALPACA_KEY, cfg.ALPACA_SECRET)


def patch_reconnect_safety(stream, label: str):
    """Data-WS reconnect hardening (workaround for an alpaca-py bug: on a connect-time
    ValueError -- e.g. Alpaca's "connection limit exceeded" -- _run_forever()'s reconnect loop
    neither closes the socket nor backs off, so a failed connect LEAKS the old socket and
    tight-loops (observed ~17/s, 350 in 20s on v2's news+stock streams during a smoke test
    2026-07-17 -- the stock stream had this fix from day one, the news stream didn't, on the
    mistaken assumption from v1 that news 'never churns'; it clearly can). Apply to ANY stream:
    closes the stale socket first, then backs off exponentially before retrying, so a real
    connection-limit condition degrades gracefully instead of hammering the API."""
    orig_connect = stream._connect
    backoff = {"s": 0.0}

    async def _connect_safe():
        if getattr(stream, "_ws", None) is not None:
            try:
                await stream._ws.close()
            except Exception:
                pass
            stream._ws = None
        b = backoff["s"]
        if b:
            log.info("⏳ %s data-ws reconnect backoff %.0fs", label, b)
            import asyncio
            await asyncio.sleep(b)
        try:
            await orig_connect()
            backoff["s"] = 0.0
        except Exception:
            backoff["s"] = min(max(b * 2, 1.0), 30.0)
            raise

    stream._connect = _connect_safe


patch_reconnect_safety(stock_stream, "stock")

# ═══════════════════════════════════════════════════════════════════════════════
# STATE
# ═══════════════════════════════════════════════════════════════════════════════

_recent_trades: dict[str, float] = {}          # underlying -> last trade epoch
_price_cache:   dict[str, float] = {}          # real-time price cache (StockDataStream)
_monitored_positions: dict[str, dict] = {}     # symbol -> {qty, entry_price, peak_price, ...}
_deferred_closes: set[str] = set()             # symbols whose close is deferred (market closed)

_daily_loss_usd  = 0.0
_trading_halted  = False
_loss_day: "date | None" = None
_equity_cache = {"val": None, "ts": 0.0}
_EQUITY_TTL   = 300.0
_regime_cache = {"date": None, "uptrend": True}

BOT_STATE_FILE     = "bot_state.json"
TRADES_FILE        = "trades.csv"
CLOSED_TRADES_FILE = "closed_trades.csv"
REGIME_DECISIONS_FILE = "regime_decisions.csv"

STOCK_TRAIL_PCT = 0.10

# ═══════════════════════════════════════════════════════════════════════════════
# TRAIL / HOLD-TIME RULES
# ═══════════════════════════════════════════════════════════════════════════════


def news_call_trail_pct(gain: float) -> float:
    for thresh, trail in cfg.NEWS_CALL_EXIT_TIERS:
        if gain < thresh:
            return trail
    return cfg.NEWS_CALL_EXIT_TIERS[-1][1]


def pead_trail_pct(gain: float) -> float:
    for thresh, trail in cfg.PEAD_EXIT_TIERS:
        if gain < thresh:
            return trail
    return cfg.PEAD_EXIT_TIERS[-1][1]


def lotto_trail_pct(gain: float) -> float:
    for thresh, trail in cfg.LOTTO_EXIT_TIERS:
        if gain < thresh:
            return trail
    return cfg.LOTTO_EXIT_TIERS[-1][1]


def long_trail_for(strategy: str, asset_type: str, gain: float) -> float:
    if strategy == "pead":
        return pead_trail_pct(gain)
    if strategy == "lotto":
        return lotto_trail_pct(gain)
    if strategy == "news_call":
        return news_call_trail_pct(gain)
    if strategy == "stock":
        return STOCK_TRAIL_PCT
    for thresh, trail in cfg.EXIT_TIERS:
        if gain < thresh:
            return trail
    return cfg.EXIT_TIERS[-1][1]


def max_hold_days_for(strategy: str, asset_type: str) -> int:
    if strategy == "stock":
        return 0   # only news_call/pead/lotto trade stock/options in v2; bare "stock" unused for now
    if strategy == "pead":
        return cfg.PEAD_MAX_HOLD_DAYS
    if strategy == "lotto":
        return cfg.LOTTO_MAX_HOLD_DAYS
    if strategy == "news_call":
        return cfg.NEWS_CALL_MAX_HOLD_DAYS
    return 0


def _time_stop_hit(pos: dict) -> bool:
    entry_dt = pos.get("entry_dt")
    if not entry_dt:
        return False
    max_days = max_hold_days_for(pos.get("strategy", ""), pos.get("asset_type", ""))
    if max_days <= 0:
        return False
    return (datetime.now(timezone.utc) - entry_dt).days >= max_days


def _at_position_cap(strategy: str) -> bool:
    """MAX_OPEN_POSITIONS with a reserved PEAD sub-cap (same pattern as v1, smaller numbers)."""
    if strategy == "pead":
        pead_n = sum(1 for p in _monitored_positions.values() if p.get("strategy") == "pead")
        if pead_n >= cfg.PEAD_MAX_POSITIONS:
            log.info("  → pead sub-cap (%d/%d) reached — skipping", pead_n, cfg.PEAD_MAX_POSITIONS)
            return True
        return False
    n = sum(1 for p in _monitored_positions.values() if p.get("strategy") != "pead")
    if n >= cfg.MAX_OPEN_POSITIONS:
        log.info("  → position cap (%d/%d) reached — skipping new %s trade",
                  n, cfg.MAX_OPEN_POSITIONS, strategy)
        return True
    return False


def _write_bot_state():
    """Persist stop/peak/entry metadata for restart recovery."""
    state = {}
    for symbol, pos in _monitored_positions.items():
        asset_type = pos.get("asset_type", "option")
        entry = pos["entry_price"]
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
            "entry_dt":    pos["entry_dt"].isoformat() if pos.get("entry_dt") else None,
        }
    try:
        with open(BOT_STATE_FILE, "w") as f:
            json.dump(state, f)
    except Exception:
        pass


def _log_deferred_once(symbol: str, msg: str):
    if symbol not in _deferred_closes:
        log.warning(msg, symbol)
        _deferred_closes.add(symbol)
    else:
        log.debug(msg, symbol)


# ═══════════════════════════════════════════════════════════════════════════════
# MARKET DATA / REGIME
# ═══════════════════════════════════════════════════════════════════════════════


def market_is_open() -> bool:
    """Fails to False on a clock error (safe default — stop-close branches defer & retry)."""
    try:
        return trading_client.get_clock().is_open
    except Exception as e:
        log.warning("market_is_open clock error (%s) — assuming closed this cycle", e)
        return False


def near_market_close(within_min: int = cfg.TIME_STOP_EOD_WINDOW_MIN) -> bool:
    try:
        clk = trading_client.get_clock()
        if not clk.is_open:
            return False
        secs_to_close = (clk.next_close - clk.timestamp).total_seconds()
        return 0 <= secs_to_close <= within_min * 60
    except Exception:
        return False


def market_in_uptrend() -> bool:
    """200d SMA + momentum brake + chop brake (2026-07-16). No bypass in v2 — the mechanism
    was tightly coupled to the confirm/veto gate's fail-closed logic, which no longer exists,
    and the live evidence that killed the bypass (8% win, -$2.0k) doesn't depend on gate
    architecture anyway. Cached once per calendar day; fails open (True) on data errors."""
    if not cfg.REGIME_FILTER_ENABLED:
        return True
    today = date.today()
    if _regime_cache["date"] == today:
        return _regime_cache["uptrend"]
    try:
        start = datetime.now(timezone.utc) - timedelta(days=int(cfg.REGIME_MA_DAYS * 1.6) + 30)
        resp = stock_data_client.get_stock_bars(StockBarsRequest(
            symbol_or_symbols=cfg.REGIME_INDEX, timeframe=TimeFrame.Day, start=start, feed="iex"))
        closes = [float(b.close) for b in (resp.data or {}).get(cfg.REGIME_INDEX, [])]
        if len(closes) < cfg.REGIME_MA_DAYS:
            log.warning("Regime: only %d bars (<%d) — defaulting to uptrend", len(closes), cfg.REGIME_MA_DAYS)
            up = True
        else:
            sma = sum(closes[-cfg.REGIME_MA_DAYS:]) / cfg.REGIME_MA_DAYS
            above_sma = closes[-1] >= sma
            mom_ok = (cfg.REGIME_MOMENTUM_DAYS <= 0 or len(closes) <= cfg.REGIME_MOMENTUM_DAYS
                      or closes[-1] >= closes[-1 - cfg.REGIME_MOMENTUM_DAYS])
            dd_win = cfg.REGIME_DOWNDAY_WINDOW
            if dd_win > 0 and len(closes) > dd_win:
                dd_density = sum(1 for a, b in zip(closes[-dd_win - 1:-1], closes[-dd_win:])
                                 if b < a) / dd_win
            else:
                dd_density = 0.0
            chop_ok = dd_win <= 0 or dd_density < cfg.REGIME_DOWNDAY_MAX_DENSITY
            up = above_sma and mom_ok and chop_ok
            log.info("📐 Regime: %s $%.2f vs %dd SMA $%.2f (%s) · mom %s · %dd down-days %.0f%% (%s) → %s",
                      cfg.REGIME_INDEX, closes[-1], cfg.REGIME_MA_DAYS, sma,
                      "above" if above_sma else "below", "ok" if mom_ok else "down",
                      dd_win, dd_density * 100, "ok" if chop_ok else "choppy",
                      "UPTREND" if up else "PAUSED")
    except Exception as e:
        log.warning("Regime check failed (%s) — defaulting to uptrend", e)
        up = True
    _regime_cache.update(date=today, uptrend=up)
    return up


def on_cooldown(ticker: str) -> bool:
    return (time.time() - _recent_trades.get(ticker, 0)) < cfg.COOLDOWN_SECS


async def handle_stock_trade(trade):
    ticker = getattr(trade, "symbol", None)
    price  = getattr(trade, "price", None)
    if ticker and price:
        _price_cache[ticker] = float(price)


def subscribe_price_stream(tickers: list):
    if not tickers:
        return
    stock_stream.subscribe_trades(handle_stock_trade, *tickers)
    log.info("📡 Price stream subscribed: %s", ", ".join(tickers[:5]) + ("…" if len(tickers) > 5 else ""))


def get_stock_price(ticker: str, subscribe: bool = False) -> Optional[float]:
    cached = _price_cache.get(ticker)
    if cached:
        return cached
    try:
        resp = stock_data_client.get_stock_latest_trade(StockLatestTradeRequest(symbol_or_symbols=ticker))
        price = float(resp[ticker].price)
        _price_cache[ticker] = price
        if subscribe:
            subscribe_price_stream([ticker])
        return price
    except Exception as e:
        log.warning("Stock price fetch failed for %s: %s", ticker, e)
        return None


def get_option_quote(symbol: str) -> Optional[dict]:
    try:
        resp = option_data_client.get_option_latest_quote(OptionLatestQuoteRequest(symbol_or_symbols=symbol))
        q = resp[symbol]
        bid, ask = float(q.bid_price), float(q.ask_price)
        if ask <= 0 or bid < 0:
            return None
        mid = round((bid + ask) / 2, 4)
        spread_pct = (ask - bid) / mid if mid > 0 else 1.0
        return {"bid": bid, "ask": ask, "mid": mid, "spread_pct": spread_pct}
    except Exception as e:
        log.debug("Option quote fetch failed for %s: %s", symbol, e)
        return None


# ═══════════════════════════════════════════════════════════════════════════════
# GUARDRAILS
# ═══════════════════════════════════════════════════════════════════════════════


def _loss_reset_if_new_day():
    global _daily_loss_usd, _trading_halted, _loss_day
    today = datetime.now(timezone.utc).date()
    if _loss_day != today:
        if _loss_day is not None and (_daily_loss_usd or _trading_halted):
            log.info("🔄 New UTC day — resetting daily-loss breaker (was $%.0f, halted=%s)",
                      _daily_loss_usd, _trading_halted)
        _loss_day = today
        _daily_loss_usd = 0.0
        _trading_halted = False


def account_equity() -> Optional[float]:
    """Cached for _EQUITY_TTL secs. Returns None on failure (callers must handle)."""
    now = time.time()
    if _equity_cache["val"] is not None and now - _equity_cache["ts"] < _EQUITY_TTL:
        return _equity_cache["val"]
    try:
        eq = float(trading_client.get_account().equity)
        _equity_cache.update(val=eq, ts=now)
        return eq
    except Exception as e:
        log.debug("equity fetch failed: %s", e)
        return _equity_cache["val"]


def _daily_loss_limit() -> float:
    """DAILY_LOSS_LIMIT_PCT x equity, with a small (not $2,000!) fail-safe floor for when the
    equity fetch itself fails -- v1's $2,000 floor exceeded an entire $500-1000 account, so in
    that narrow failure window the breaker provided no protection at all."""
    eq = account_equity()
    return eq * cfg.DAILY_LOSS_LIMIT_PCT if eq else cfg.DAILY_LOSS_MIN_FLOOR_USD


def position_budget_usd() -> float:
    """Flat position size = MAX_POSITION_FRAC_OF_EQUITY x current equity. Flat, not
    magnitude-scaled: v1's own analysis found magnitude uncalibrated (rank-IC~=0, loser avg
    mag ~= winner avg mag), so scaling size by it would be false precision."""
    eq = account_equity()
    if not eq:
        log.warning("equity unavailable — falling back to a conservative $200 position budget")
        return 200.0
    return eq * cfg.MAX_POSITION_FRAC_OF_EQUITY


def lotto_budget_usd() -> float:
    eq = account_equity()
    if not eq:
        return 50.0
    return eq * cfg.LOTTO_POSITION_FRAC_OF_EQUITY


def passes_guardrails(quote: dict, skip_halt: bool = False) -> bool:
    global _trading_halted
    if not skip_halt:
        _loss_reset_if_new_day()
        limit = _daily_loss_limit()
        if _trading_halted:
            log.warning("🛑 Trading halted — daily loss limit reached ($%.0f)", limit)
            return False
        if _daily_loss_usd >= limit:
            _trading_halted = True
            log.warning("🛑 Daily loss limit hit ($%.0f >= $%.0f). Halting until next UTC day.",
                        _daily_loss_usd, limit)
            return False
    if quote["spread_pct"] > cfg.MAX_SPREAD_PCT:
        log.info("  → spread too wide: %.1f%% > %.0f%% max — skipping", quote["spread_pct"] * 100, cfg.MAX_SPREAD_PCT * 100)
        return False
    return True


_yahoo_px_cache: dict[str, tuple] = {}


def _yahoo_latest_price(ticker: str, ttl: float = 90.0) -> Optional[float]:
    now = time.time()
    hit = _yahoo_px_cache.get(ticker)
    if hit and now - hit[1] < ttl:
        return hit[0]
    try:
        from yahoo_data import _fetch_yahoo, _parse
        series = _parse(_fetch_yahoo(ticker, int(now) - 86400 * 2, int(now) + 86400, tries=2))
        yp = series[max(series)][3] if series else None
    except Exception as e:
        log.debug("Yahoo cross-check fetch failed for %s: %s", ticker, e)
        yp = None
    if yp and yp > 0:
        _yahoo_px_cache[ticker] = (yp, now)
    return yp


def passes_stock_price_guardrail(ticker: str, stock_price: float) -> bool:
    if stock_price < cfg.MIN_STOCK_PRICE:
        log.info("  → %s @ $%.2f below MIN_STOCK_PRICE — skipping", ticker, stock_price)
        return False
    if cfg.PRICE_CROSSCHECK_ENABLED:
        yp = _yahoo_latest_price(ticker)
        if yp and abs(stock_price / yp - 1) > cfg.PRICE_CROSSCHECK_TOL:
            log.warning("  → %s PRICE MISMATCH: Alpaca $%.2f vs Yahoo $%.2f — skipping (bad data)",
                        ticker, stock_price, yp)
            return False
    return True
