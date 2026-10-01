"""
market.py (v3) — market data interface, quotes, regime checking, and account telemetry.
Features:
  - Alpaca trading & data clients.
  - SPY regime gate (200d SMA, 3-day momentum, 10-day chop brake).
  - Real-time stock & option quote retrieval with bid/ask spread calculation.
  - Daily loss circuit breaker and account equity caching.
"""

import logging
import time
from datetime import date, datetime, timedelta, timezone
from typing import Optional

from alpaca.data.historical.stock import StockHistoricalDataClient
from alpaca.data.historical.option import OptionHistoricalDataClient
from alpaca.data.requests import StockLatestQuoteRequest, OptionLatestQuoteRequest, StockBarsRequest
from alpaca.data.timeframe import TimeFrame
from alpaca.trading.client import TradingClient

try:
    from v3 import config as cfg
except ImportError:
    import config as cfg

log = logging.getLogger(__name__)

# Alpaca clients
trading_client = TradingClient(cfg.ALPACA_KEY, cfg.ALPACA_SECRET, paper=True)
stock_data_client = StockHistoricalDataClient(cfg.ALPACA_KEY, cfg.ALPACA_SECRET)
option_data_client = OptionHistoricalDataClient(cfg.ALPACA_KEY, cfg.ALPACA_SECRET)

# Monitored positions cache: symbol -> position dict
_monitored_positions: dict[str, dict] = {}

# Equity cache
_equity_cache: dict = {"val": None, "ts": 0.0}
_EQUITY_TTL = 300.0  # 5 minutes

# Daily loss tracking
_daily_loss_usd: float = 0.0
_trading_halted: bool = False
_loss_day: Optional[date] = None

# Regime cache: (date, uptrend_ok, mom_ok, chop_ok)
_regime_cache: dict = {"date": None, "uptrend_ok": False, "mom_ok": False, "chop_ok": False, "details": {}}


def get_account_equity() -> float:
    """Fetch current account equity with TTL cache."""
    now = time.time()
    if _equity_cache["val"] is not None and now - _equity_cache["ts"] < _EQUITY_TTL:
        return _equity_cache["val"]
    try:
        acct = trading_client.get_account()
        val = float(acct.equity)
        _equity_cache["val"] = val
        _equity_cache["ts"] = now
        return val
    except Exception as e:
        log.warning("Equity fetch failed: %s", e)
        return _equity_cache["val"] or 1000.0


def check_daily_loss_breaker() -> bool:
    """Check if the daily loss circuit breaker has tripped."""
    global _trading_halted, _daily_loss_usd, _loss_day
    today_utc = datetime.now(timezone.utc).date()
    if _loss_day != today_utc:
        _loss_day = today_utc
        _daily_loss_usd = 0.0
        _trading_halted = False

    if _trading_halted:
        return True

    equity = get_account_equity()
    limit = equity * cfg.DAILY_LOSS_LIMIT_PCT
    if _daily_loss_usd >= limit:
        _trading_halted = True
        log.warning("🚨 DAILY LOSS BREAKER TRIPPED: loss $%.2f >= limit $%.2f (2%% of $%.2f)",
                    _daily_loss_usd, limit, equity)
        return True
    return False


def record_realized_loss(loss_usd: float) -> None:
    """Record a realized loss toward the daily breaker."""
    global _daily_loss_usd
    if loss_usd > 0:
        _daily_loss_usd += loss_usd
        check_daily_loss_breaker()


def is_market_open() -> bool:
    """Check if the US equity market is currently open."""
    try:
        clock = trading_client.get_clock()
        return bool(clock.is_open)
    except Exception as e:
        log.debug("Clock check failed: %s", e)
        # Fallback to local time heuristic
        now = datetime.now(timezone.utc)
        # Weekdays 13:30 to 20:00 UTC (9:30 AM to 4:00 PM EDT)
        if now.weekday() >= 5:
            return False
        return 13 <= now.hour <= 20


def get_stock_price(ticker: str) -> Optional[float]:
    """Fetch latest stock trade or mid price."""
    try:
        req = StockLatestQuoteRequest(symbol_or_symbols=ticker)
        res = stock_data_client.get_stock_latest_quote(req)
        q = res.get(ticker)
        if q:
            bid = float(q.bid_price) if q.bid_price else 0.0
            ask = float(q.ask_price) if q.ask_price else 0.0
            if bid > 0 and ask > 0:
                return round((bid + ask) / 2.0, 4)
            if ask > 0:
                return float(ask)
            if bid > 0:
                return float(bid)
    except Exception as e:
        log.debug("Stock quote fetch failed for %s: %s", ticker, e)
    return None


def get_option_quote(symbol: str) -> Optional[dict]:
    """Fetch current option quote with bid, ask, mid, and spread_pct."""
    try:
        req = OptionLatestQuoteRequest(symbol_or_symbols=symbol)
        res = option_data_client.get_option_latest_quote(req)
        q = res.get(symbol)
        if not q:
            return None

        bid = float(q.bid_price) if q.bid_price is not None else 0.0
        ask = float(q.ask_price) if q.ask_price is not None else 0.0
        if ask <= 0 or bid < 0:
            return None

        mid = round((bid + ask) / 2.0, 4)
        spread_pct = round((ask - bid) / mid, 4) if mid > 0 else 1.0

        return {
            "bid": bid,
            "ask": ask,
            "mid": mid,
            "spread_pct": spread_pct,
        }
    except Exception as e:
        log.debug("Option quote fetch failed for %s: %s", symbol, e)
        return None


def evaluate_regime() -> dict:
    """
    Evaluate SPY 200d SMA, 3-day momentum, and 10-day chop brake.
    Returns dict: {'passes_regime': bool, 'uptrend_ok': bool, 'mom_ok': bool, 'chop_ok': bool}
    """
    if not cfg.REGIME_GATE_ENABLED:
        return {"passes_regime": True, "uptrend_ok": True, "mom_ok": True, "chop_ok": True}

    today = datetime.now(timezone.utc).date()
    if _regime_cache["date"] == today:
        return _regime_cache["details"]

    try:
        start_dt = datetime.now(timezone.utc) - timedelta(days=320)
        req = StockBarsRequest(
            symbol_or_symbols="SPY",
            timeframe=TimeFrame.Day,
            start=start_dt,
            limit=250,
        )
        resp = stock_data_client.get_stock_bars(req)
        bars = resp.data.get("SPY", []) if hasattr(resp, "data") and isinstance(resp.data, dict) else (resp.get("SPY", []) if hasattr(resp, "get") else resp["SPY"])
        if len(bars) < 200:
            log.warning("Insufficient SPY bars (%d) for 200d SMA regime evaluation.", len(bars))
            return {"passes_regime": True, "uptrend_ok": True, "mom_ok": True, "chop_ok": True}

        closes = [float(b.close) for b in bars]
        latest_close = closes[-1]
        sma_200 = sum(closes[-200:]) / 200.0

        # 1. 200d SMA Uptrend
        uptrend_ok = latest_close >= sma_200

        # 2. 3-day Momentum (close >= close 3 sessions ago)
        mom_ok = latest_close >= closes[-4] if len(closes) >= 4 else True

        # 3. 10-session Chop Brake: down-day density < 60%
        window = closes[-10:]
        down_days = sum(1 for i in range(1, len(window)) if window[i] < window[i-1])
        dd_density = down_days / 9.0
        chop_ok = dd_density < cfg.REGIME_DOWNDAY_MAX_DENSITY

        passes = bool(uptrend_ok and mom_ok and chop_ok)

        details = {
            "passes_regime": passes,
            "uptrend_ok": uptrend_ok,
            "mom_ok": mom_ok,
            "chop_ok": chop_ok,
            "spy_price": latest_close,
            "spy_sma200": round(sma_200, 2),
            "dd_density": round(dd_density, 2),
        }

        _regime_cache["date"] = today
        _regime_cache["details"] = details
        log.info("Regime status: passes=%s (SPY=$%.2f SMA=$%.2f uptrend=%s mom=%s chop_ok=%s)",
                 passes, latest_close, sma_200, uptrend_ok, mom_ok, chop_ok)
        return details
    except Exception as e:
        log.warning("Regime evaluation exception: %s", e)
        return {"passes_regime": True, "uptrend_ok": True, "mom_ok": True, "chop_ok": True}

