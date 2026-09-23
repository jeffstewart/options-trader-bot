"""
execution.py (v3) — contract selection, order execution, and timed horizon exit monitoring.
Features:
  - Enforces ~0.50 Delta ATM contract selection.
  - Strict empirical guardrails: hard 6.5% spread cap and 10 open interest floor.
  - Timed 30-minute horizon exit (empirically superior to un-floored trailing stops).
  - Marketable limit orders with fill verification to prevent ghost positions.
  - Position state persistence in JSON and append-only trade logs.
"""

import csv
import json
import logging
import os
import time
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import Optional

from alpaca.trading.enums import OrderSide, TimeInForce, ContractType, PositionIntent, QueryOrderStatus
from alpaca.trading.requests import GetOptionContractsRequest, LimitOrderRequest, GetOrdersRequest

try:
    from v3 import config as cfg
    from v3 import market as mkt
except ImportError:
    import config as cfg
    import market as mkt

log = logging.getLogger(__name__)

TRADES_CSV = cfg.DATA_DIR / "trades.csv"
CLOSED_TRADES_CSV = cfg.DATA_DIR / "closed_trades.csv"
STATE_FILE = cfg.DATA_DIR / "bot_state.json"


def _passes_contract_guardrails(c, quote: Optional[dict], stock_price: float, ticker: str) -> bool:
    """
    Empirical contract-level checks:
      1. Tradable & matching root ticker
      2. Non-zero mid quote
      3. Quote sanity (mid between 90% of intrinsic and stock price)
      4. Hard Open Interest floor (>= 10)
      5. Hard Spread Cap (<= 6.5%)
    """
    if getattr(c, "tradable", True) is False:
        return False
    root = getattr(c, "root_symbol", None)
    if root and root != ticker:
        return False
    if quote is None or quote["mid"] <= 0:
        return False

    strike = float(c.strike_price)
    intrinsic = max(0.0, stock_price - strike)
    if quote["mid"] < intrinsic * 0.9 or quote["mid"] > stock_price:
        return False

    # Open interest floor (rejections here avoid illiquid traps)
    oi = getattr(c, "open_interest", None)
    if oi is None or int(oi) < cfg.MIN_OPEN_INTEREST:
        return False

    # Hard spread cap (discovered in tuner analysis: >6.5% spreads severely hurt P&L)
    if quote["spread_pct"] > cfg.MAX_SPREAD_PCT:
        return False

    return True


def pick_call_contract(ticker: str, stock_price: float,
                       dte_min: int = None, dte_max: int = None,
                       target_delta: float = None) -> Optional[dict]:
    """
    Select an ATM call contract close to target_delta (~0.50) meeting strict spread and OI rules.
    """
    min_dte = dte_min if dte_min is not None else cfg.DTE_MIN
    max_dte = dte_max if dte_max is not None else cfg.DTE_MAX
    delta_target = target_delta if target_delta is not None else cfg.TARGET_DELTA

    today = date.today()
    try:
        req = GetOptionContractsRequest(
            underlying_symbols=[ticker],
            type=ContractType.CALL,
            expiration_date_gte=today + timedelta(days=min_dte),
            expiration_date_lte=today + timedelta(days=max_dte),
            strike_price_gte=str(round(stock_price * cfg.STRIKE_LO_MULT, 2)),
            strike_price_lte=str(round(stock_price * cfg.STRIKE_HI_MULT, 2)),
            limit=50,
        )
        resp = mkt.trading_client.get_option_contracts(req)
        contracts = getattr(resp, "option_contracts", []) or []
    except Exception as e:
        log.warning("Contract lookup failed for %s: %s", ticker, e)
        return None

    if not contracts:
        log.info("  → no option contracts returned for %s in DTE %d-%d", ticker, min_dte, max_dte)
        return None

    # Target ATM strike
    target_strike = stock_price * (1.0 + (1.0 - delta_target) * 0.15)
    target_dte = (min_dte + max_dte) // 2

    def _sort_key(c):
        exp_dt = datetime.strptime(str(c.expiration_date), "%Y-%m-%d").date()
        dte_diff = abs((exp_dt - today).days - target_dte)
        strike_diff = abs(float(c.strike_price) - target_strike)
        return (dte_diff, strike_diff)

    contracts.sort(key=_sort_key)

    # Inspect top candidates against empirical liquidity filters
    for c in contracts[:10]:
        quote = mkt.get_option_quote(c.symbol)
        if not _passes_contract_guardrails(c, quote, stock_price, ticker):
            continue

        oi = int(getattr(c, "open_interest", 0) or 0)
        return {
            "symbol": c.symbol,
            "strike": float(c.strike_price),
            "expiry": str(c.expiration_date),
            "bid": quote["bid"],
            "ask": quote["ask"],
            "mid": quote["mid"],
            "spread_pct": quote["spread_pct"],
            "open_interest": oi,
        }

    log.info("  → no liquid contracts found for %s meeting <=%.1f%% spread & >=%d OI",
             ticker, cfg.MAX_SPREAD_PCT * 100, cfg.MIN_OPEN_INTEREST)
    return None


def _await_fill(order_id: str, timeout_s: float = 10.0) -> Optional[float]:
    """Poll order until filled or timeout, then cancel if unfilled."""
    t0 = time.monotonic()
    while time.monotonic() - t0 < timeout_s:
        try:
            o = mkt.trading_client.get_order_by_id(order_id)
            if o.status == QueryOrderStatus.FILLED:
                return float(o.filled_avg_price)
            if o.status in (QueryOrderStatus.CANCELED, QueryOrderStatus.EXPIRED, QueryOrderStatus.REJECTED):
                return None
        except Exception as e:
            log.debug("Order status poll error: %s", e)
        time.sleep(0.5)

    # Order timed out without full fill — cancel to avoid phantom execution
    try:
        mkt.trading_client.cancel_order_by_id(order_id)
        log.warning("Order %s unfilled after %.1fs — cancelled.", order_id, timeout_s)
    except Exception as e:
        log.debug("Cancel order error: %s", e)
    return None


def place_option_trade(ticker: str, stock_price: float, signal: dict,
                       position_usd: float) -> Optional[dict]:
    """Execute long call entry on candidate ticker."""
    if len(mkt._monitored_positions) >= cfg.MAX_OPEN_POSITIONS:
        log.info("Max open positions cap reached (%d/%d) — skipping %s",
                 len(mkt._monitored_positions), cfg.MAX_OPEN_POSITIONS, ticker)
        return None

    contract = pick_call_contract(ticker, stock_price)
    if not contract:
        return None

    symbol = contract["symbol"]
    if symbol in mkt._monitored_positions:
        log.info("Already holding %s — skipping duplicate entry", symbol)
        return None

    contract_cost = contract["ask"] * 100.0
    if contract_cost > position_usd * cfg.MAX_CONTRACT_BUDGET_MULT:
        log.info("Contract cost $%.0f exceeds budget mult (budget $%.0f) — skipping %s",
                 contract_cost, position_usd, symbol)
        return None

    qty = max(1, int(position_usd / contract_cost))
    marketable_limit = round(contract["ask"] * 1.05, 2)

    log.info("ENTRY ORDER: %s strike=$%.2f exp=%s qty=%d ask=$%.2f spread=%.1f%% limit=$%.2f",
             symbol, contract["strike"], contract["expiry"], qty,
             contract["ask"], contract["spread_pct"] * 100, marketable_limit)

    if cfg.DRY_RUN:
        simulated_fill = contract["ask"]
        pos = {
            "symbol": symbol,
            "underlying": ticker,
            "qty": qty,
            "entry_price": simulated_fill,
            "entry_time": datetime.now(timezone.utc).isoformat(),
            "peak_price": simulated_fill,
            "strike": contract["strike"],
            "expiry": contract["expiry"],
            "signal_id": signal.get("id", ""),
            "dry_run": True,
        }
        mkt._monitored_positions[symbol] = pos
        _log_trade_entry(pos)
        save_state()
        return pos

    try:
        req = LimitOrderRequest(
            symbol=symbol,
            qty=qty,
            side=OrderSide.BUY,
            time_in_force=TimeInForce.DAY,
            limit_price=marketable_limit,
            position_intent=PositionIntent.BUY_TO_OPEN,
        )
        order = mkt.trading_client.submit_order(req)
        fill_price = _await_fill(str(order.id), timeout_s=10.0)
        if fill_price is None:
            log.warning("Entry order for %s failed to fill in time.", symbol)
            return None

        pos = {
            "symbol": symbol,
            "underlying": ticker,
            "qty": qty,
            "entry_price": fill_price,
            "entry_time": datetime.now(timezone.utc).isoformat(),
            "peak_price": fill_price,
            "strike": contract["strike"],
            "expiry": contract["expiry"],
            "signal_id": signal.get("id", ""),
            "dry_run": False,
        }
        mkt._monitored_positions[symbol] = pos
        _log_trade_entry(pos)
        save_state()
        log.info("FILLED BUY %s x%d @ $%.2f", symbol, qty, fill_price)
        return pos
    except Exception as e:
        log.error("Order submission failed for %s: %s", symbol, e)
        return None


def close_option_position(symbol: str, reason: str) -> bool:
    """Close an open option position using a marketable limit sell order."""
    pos = mkt._monitored_positions.get(symbol)
    if not pos:
        return False

    qty = pos["qty"]
    quote = mkt.get_option_quote(symbol)
    bid = quote["bid"] if quote else pos["entry_price"] * 0.8
    marketable_limit = max(0.01, round(bid * 0.95, 2))

    log.info("EXIT TRIGGERED [%s]: Closing %s x%d (bid=$%.2f limit=$%.2f)",
             reason, symbol, qty, bid, marketable_limit)

    if pos.get("dry_run", False) or cfg.DRY_RUN:
        exit_price = max(0.01, bid)
        _finalize_trade_exit(pos, exit_price, reason)
        return True

    try:
        req = LimitOrderRequest(
            symbol=symbol,
            qty=qty,
            side=OrderSide.SELL,
            time_in_force=TimeInForce.DAY,
            limit_price=marketable_limit,
            position_intent=PositionIntent.SELL_TO_CLOSE,
        )
        order = mkt.trading_client.submit_order(req)
        fill_price = _await_fill(str(order.id), timeout_s=15.0)
        if fill_price is None:
            # Fallback to market order if limit didn't fill
            log.warning("Exit limit unfilled for %s — submitting market close", symbol)
            mkt.trading_client.close_position(symbol)
            fill_price = max(0.01, bid)

        _finalize_trade_exit(pos, fill_price, reason)
        return True
    except Exception as e:
        log.error("Failed to close position %s: %s", symbol, e)
        return False


def _finalize_trade_exit(pos: dict, exit_price: float, reason: str) -> None:
    symbol = pos["symbol"]
    entry_price = pos["entry_price"]
    qty = pos["qty"]
    pnl_usd = round((exit_price - entry_price) * 100.0 * qty, 2)
    pnl_pct = round((exit_price / entry_price - 1.0) * 100.0, 2) if entry_price > 0 else 0.0

    log.info("CLOSED %s: entry=$%.2f exit=$%.2f P&L=$%.2f (%+.1f%%) [%s]",
             symbol, entry_price, exit_price, pnl_usd, pnl_pct, reason)

    # Log to closed_trades.csv
    try:
        new_file = not os.path.exists(CLOSED_TRADES_CSV)
        with open(CLOSED_TRADES_CSV, "a", newline="") as f:
            w = csv.writer(f)
            if new_file:
                w.writerow(["timestamp", "symbol", "underlying", "strategy", "qty",
                            "entry_price", "exit_price", "pnl_usd", "pnl_pct", "reason", "signal_id"])
            w.writerow([
                datetime.now(timezone.utc).isoformat(), symbol, pos["underlying"],
                "news_call_v3", qty, entry_price, exit_price, pnl_usd, pnl_pct, reason, pos.get("signal_id", "")
            ])
    except Exception as e:
        log.warning("Failed to append to closed_trades.csv: %s", e)

    if pnl_usd < 0:
        mkt.record_realized_loss(abs(pnl_usd))

    mkt._monitored_positions.pop(symbol, None)
    save_state()


def _log_trade_entry(pos: dict) -> None:
    try:
        new_file = not os.path.exists(TRADES_CSV)
        with open(TRADES_CSV, "a", newline="") as f:
            w = csv.writer(f)
            if new_file:
                w.writerow(["timestamp", "symbol", "underlying", "strategy", "qty",
                            "entry_price", "strike", "expiry", "signal_id"])
            w.writerow([
                pos["entry_time"], pos["symbol"], pos["underlying"], "news_call_v3",
                pos["qty"], pos["entry_price"], pos["strike"], pos["expiry"], pos.get("signal_id", "")
            ])
    except Exception as e:
        log.warning("Failed to log trade entry: %s", e)


def monitor_open_positions() -> None:
    """Check all open positions against timed horizon exit rule."""
    if not mkt._monitored_positions:
        return

    now = datetime.now(timezone.utc)
    for symbol, pos in list(mkt._monitored_positions.items()):
        try:
            entry_dt = datetime.fromisoformat(pos["entry_time"])
            if entry_dt.tzinfo is None:
                entry_dt = entry_dt.replace(tzinfo=timezone.utc)
            minutes_held = (now - entry_dt).total_seconds() / 60.0

            # Empirical 30-minute timed exit
            if minutes_held >= cfg.EXIT_HORIZON_MINUTES:
                close_option_position(symbol, reason=f"horizon_{cfg.EXIT_HORIZON_MINUTES}m")
                continue

            # Update peak price for tracking
            quote = mkt.get_option_quote(symbol)
            if quote and quote["bid"] > pos.get("peak_price", 0):
                pos["peak_price"] = quote["bid"]

            # Optional trailing stop with floor
            if cfg.TRAILING_STOP_ENABLED and minutes_held >= cfg.TRAILING_STOP_FLOOR_M:
                if quote and pos.get("peak_price", 0) > 0:
                    dd = quote["bid"] / pos["peak_price"] - 1.0
                    if dd <= -cfg.TRAILING_STOP_PCT:
                        close_option_position(symbol, reason="trailing_stop")
                        continue
        except Exception as e:
            log.warning("Position monitoring exception for %s: %s", symbol, e)


def save_state() -> None:
    try:
        with open(STATE_FILE, "w") as f:
            json.dump({"monitored_positions": mkt._monitored_positions}, f, indent=2)
    except Exception as e:
        log.debug("State save failed: %s", e)


def load_state() -> None:
    if os.path.exists(STATE_FILE):
        try:
            with open(STATE_FILE) as f:
                data = json.load(f)
                mkt._monitored_positions = data.get("monitored_positions", {})
                log.info("Loaded %d open positions from %s", len(mkt._monitored_positions), STATE_FILE)
        except Exception as e:
            log.warning("State load failed: %s", e)

