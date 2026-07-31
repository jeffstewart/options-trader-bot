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


def patch_request_timeout(client, label: str, timeout_s: float = 10.0):
    """alpaca-py's synchronous REST clients (TradingClient, *HistoricalDataClient) pass NO
    timeout to the underlying requests.Session -- confirmed by reading RESTClient._one_request,
    which calls self._session.request(method, url, **opts) with no timeout kwarg anywhere. These
    clients are called directly from async code all over v2 (market_is_open, market_in_uptrend,
    account_equity, get_option_quote, ...), NOT via run_in_executor. If a connection ever stalls
    -- observed 2026-07-20 after v2 ran silently at 0% CPU for ~17h overnight, right after a
    cluster of SEC-poll network hiccups -- a synchronous call with no timeout blocks forever, and
    since asyncio is single-threaded that freezes the ENTIRE event loop, not just the one task
    that made the call: every poller and monitor goes silent at once with no exception for
    _supervised to catch and no way to notice short of watching the logs stop. Patching the
    session here is one choke point that bounds every call through this client, current and
    future, instead of auditing/wrapping each call site individually."""
    orig_request = client._session.request

    def _request_with_timeout(method, url, **kwargs):
        kwargs.setdefault("timeout", timeout_s)
        return orig_request(method, url, **kwargs)

    client._session.request = _request_with_timeout
    log.info("⏱️  %s client: default request timeout %.0fs", label, timeout_s)


patch_request_timeout(trading_client, "trading")
patch_request_timeout(option_data_client, "option-data")
patch_request_timeout(stock_data_client, "stock-data")


def patch_reconnect_safety(stream, label: str):
    """Data-WS reconnect hardening. Root-caused 2026-07-17 (a v2 smoke test tight-looped ~17/s,
    350 times in 20s): alpaca-py's "connection limit exceeded" is raised by _auth() (a SERVER-
    SIDE auth-time rejection), not by _connect() (which succeeds -- the raw socket opens fine).
    _run_forever()'s except ValueError branch for this case just logs and loops with
    `await asyncio.sleep(0)` -- no backoff, AND it never closes the socket _connect() already
    opened, so every retry leaks another one. Wrapping _connect() alone (the original version of
    this fix) never saw the failure at all. The real fix has to wrap _start_ws() (connect+auth
    together): on ANY failure, close the stale socket and back off exponentially before
    re-raising, so _run_forever's existing log-and-loop behavior continues but with actual delay
    and no leak. Apply to ANY stream (stock, news)."""
    orig_start_ws = stream._start_ws
    backoff = {"s": 0.0}

    async def _start_ws_safe():
        b = backoff["s"]
        if b:
            log.info("⏳ %s data-ws reconnect backoff %.0fs", label, b)
            import asyncio
            await asyncio.sleep(b)
        try:
            await orig_start_ws()
            backoff["s"] = 0.0
        except Exception:
            if getattr(stream, "_ws", None) is not None:
                try:
                    await stream._ws.close()
                except Exception:
                    pass
                stream._ws = None
            backoff["s"] = min(max(b * 2, 1.0), 30.0)
            raise

    stream._start_ws = _start_ws_safe


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


def long_trail_for(strategy: str, asset_type: str, gain: float) -> float:
    # lotto has no peak-trailing branch -- it uses a dedicated entry-anchored stop-loss +
    # same-day EOD exit instead (see _lotto_stop_loss_hit / _lotto_same_day_exit_hit below).
    if strategy == "pead":
        return pead_trail_pct(gain)
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


def lotto_stop_price(pos: dict) -> float:
    """Current stop level for a lotto position: the HIGHER of an entry-anchored floor and a
    trailing stop off the running peak. Monotonic -- peak_price only rises, so the level can never
    loosen. Returns an absolute premium, not a percentage."""
    entry = pos.get("entry_price")
    if not entry:
        return 0.0
    floor = entry * (1 - cfg.LOTTO_STOP_LOSS_PCT)     # worst case, anchored to entry
    peak = max(pos.get("peak_price") or entry, entry)
    trail = peak * (1 - cfg.LOTTO_TRAIL_PCT)          # rises with every new peak
    return max(floor, trail)


def _lotto_stop_loss_hit(pos: dict, mid: float) -> bool:
    """Hard stop -- checked continuously, any time of day. Anchored to entry until the ratchet
    arms, then to the locked-in level (see lotto_stop_price)."""
    if not pos.get("entry_price"):
        return False
    return mid <= lotto_stop_price(pos)


def _lotto_same_day_exit_hit(pos: dict) -> bool:
    """True once we're on the same UTC calendar date as entry (== the ET trading day, since the
    whole session falls within one UTC date). Paired with near_market_close() by the caller."""
    if not cfg.LOTTO_SAME_DAY_EXIT:
        return False
    entry_dt = pos.get("entry_dt")
    if not entry_dt:
        return False
    return entry_dt.date() == datetime.now(timezone.utc).date()


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
        strategy = pos.get("strategy", "news_call")
        entry = pos["entry_price"]
        if strategy == "lotto":
            stop_price = entry * (1 - cfg.LOTTO_STOP_LOSS_PCT)
        else:
            gain  = (pos["peak_price"] / entry - 1.0) if entry else 0.0
            trail = long_trail_for(strategy, asset_type, gain)
            stop_price = pos["peak_price"] * (1 - trail)
        state[symbol] = {
            "stop_price":  round(stop_price, 4),
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


def _downday_detail(bars, dd_win: int, dd_max: float):
    """Per-session up/down sequence for the chop window, plus a best-case estimate of how many
    more sessions must pass before the down-day density can clear.

    WHY THE SEQUENCE AND NOT JUST THE COUNT (jeff, 2026-07-30): the density is a rolling window, so
    WHEN the down days sit determines when the gate can reopen. 7 down days bunched at the OLD end
    roll off within a couple of sessions; the same 7 at the NEW end mean a long wait. The count
    alone cannot distinguish those two, and it is the difference between "maybe trades tomorrow"
    and "definitely nothing this week".

    `sessions_to_clear` assumes EVERY future session closes up -- it is a floor ("at least this
    many"), never a forecast. Returns 0 when the chop test already passes.
    """
    if dd_win <= 0 or len(bars) <= dd_win:
        return [], 0
    window = bars[-dd_win - 1:]                       # dd_win+1 bars -> dd_win comparisons
    days = []
    for prev, cur in zip(window[:-1], window[1:]):
        ts = getattr(cur, "timestamp", None)
        days.append({"date": ts.strftime("%Y-%m-%d") if ts else "",
                     "down": float(cur.close) < float(prev.close),
                     "pct": round((float(cur.close) / float(prev.close) - 1) * 100, 2)})
    limit = dd_max * dd_win                           # density < dd_max  <=>  count < limit
    downs = [d["down"] for d in days]
    sessions = next((k for k in range(dd_win + 1) if sum(downs[k:]) < limit), dd_win)
    return days, sessions


def regime_status() -> dict:
    """Live regime computation with the full diagnostic breakdown (price, SMA, momentum, chop),
    not just the pass/fail bool -- factored out of market_in_uptrend() (2026-07-20) so the
    dashboard can call the exact same math for display (what's failing and by how much) without
    duplicating it and risking drift. No caching, no logging, no side effects -- safe to call
    from a read-only process on every page load."""
    if not cfg.REGIME_FILTER_ENABLED:
        return {"enabled": False, "uptrend": True}
    try:
        start = datetime.now(timezone.utc) - timedelta(days=int(cfg.REGIME_MA_DAYS * 1.6) + 30)
        resp = stock_data_client.get_stock_bars(StockBarsRequest(
            symbol_or_symbols=cfg.REGIME_INDEX, timeframe=TimeFrame.Day, start=start, feed="iex"))
        _regime_bars = (resp.data or {}).get(cfg.REGIME_INDEX, [])
        closes = [float(b.close) for b in _regime_bars]
        if len(closes) < cfg.REGIME_MA_DAYS:
            return {"enabled": True, "uptrend": True, "insufficient_data": True,
                    "bars": len(closes), "bars_needed": cfg.REGIME_MA_DAYS}

        sma = sum(closes[-cfg.REGIME_MA_DAYS:]) / cfg.REGIME_MA_DAYS
        price = closes[-1]
        above_sma = price >= sma

        mom_days = cfg.REGIME_MOMENTUM_DAYS
        has_mom = mom_days > 0 and len(closes) > mom_days
        mom_price_then = closes[-1 - mom_days] if has_mom else None
        mom_ok = mom_days <= 0 or not has_mom or price >= mom_price_then

        dd_win = cfg.REGIME_DOWNDAY_WINDOW
        if dd_win > 0 and len(closes) > dd_win:
            dd_density = sum(1 for a, b in zip(closes[-dd_win - 1:-1], closes[-dd_win:])
                              if b < a) / dd_win
        else:
            dd_density = 0.0
        chop_ok = dd_win <= 0 or dd_density < cfg.REGIME_DOWNDAY_MAX_DENSITY
        dd_days, dd_to_clear = _downday_detail(_regime_bars, dd_win, cfg.REGIME_DOWNDAY_MAX_DENSITY)

        return {
            "enabled": True, "uptrend": above_sma and mom_ok and chop_ok,
            "index": cfg.REGIME_INDEX, "price": price,
            "sma": round(sma, 2), "sma_days": cfg.REGIME_MA_DAYS, "above_sma": above_sma,
            "mom_ok": mom_ok, "mom_days": mom_days, "mom_price_then": mom_price_then,
            "dd_density": round(dd_density, 4), "dd_window": dd_win,
            "dd_days": dd_days, "dd_sessions_to_clear": dd_to_clear,
            "dd_max": cfg.REGIME_DOWNDAY_MAX_DENSITY, "chop_ok": chop_ok,
        }
    except Exception as e:
        return {"enabled": True, "uptrend": True, "error": str(e)}


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
    st = regime_status()
    up = st.get("uptrend", True)
    if st.get("error"):
        log.warning("Regime check failed (%s) — defaulting to uptrend", st["error"])
    elif st.get("insufficient_data"):
        log.warning("Regime: only %d bars (<%d) — defaulting to uptrend", st["bars"], st["bars_needed"])
    else:
        log.info("📐 Regime: %s $%.2f vs %dd SMA $%.2f (%s) · mom %s · %dd down-days %.0f%% (%s) → %s",
                  st["index"], st["price"], st["sma_days"], st["sma"],
                  "above" if st["above_sma"] else "below", "ok" if st["mom_ok"] else "down",
                  st["dd_window"], st["dd_density"] * 100, "ok" if st["chop_ok"] else "choppy",
                  "UPTREND" if up else "PAUSED")
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


def trading_halted() -> bool:
    """Is the daily-loss breaker tripped? THE single owner of that decision -- passes_guardrails
    delegates here so the rule can't drift between the two call sites.

    Exposed so bot.process_signal can check it BEFORE paying for a scoring call. The halt used to
    be evaluated only inside execute_lotto, i.e. AFTER the scorer had already run, so on
    2026-07-31 v2 made 4 paid Sonnet calls in the 23 minutes after halting -- scoring articles that
    could not possibly trade. Same mistake as the lotto entry cutoff (hoisted 2026-07-18 for the
    same reason): every cheap, absolute "no trade is possible" gate belongs ahead of the paid call.

    Cheap to call on every news item: short-circuits before account_equity() unless a loss has
    actually been booked today, and that fetch is itself TTL-cached.
    """
    global _trading_halted
    _loss_reset_if_new_day()
    if _trading_halted:
        return True
    if _daily_loss_usd <= 0:            # no losses booked -> cannot be halted, skip the equity call
        return False
    limit = _daily_loss_limit()
    if _daily_loss_usd >= limit:
        _trading_halted = True
        log.warning("🛑 Daily loss limit hit ($%.0f >= $%.0f). Halting until next UTC day.",
                    _daily_loss_usd, limit)
        return True
    return False


def passes_guardrails(quote: dict, skip_halt: bool = False) -> bool:
    if not skip_halt:
        if trading_halted():
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
