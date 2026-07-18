"""
execution.py (v2) — contract selection, order placement, position close, the trailing-stop
monitor, and restart recovery. Ported from core/bot.py; the hard-won fixes are kept intact:
  - marketable limit at ask/bid for option entry/exit (avoids "no quote" market-order rejection)
  - close only de-registers the position on a CONFIRMED fill (a halt-time failure keeps it monitored)
  - market-halt handling on the stock-buy path (limit-order retry)
  - data-sanity guard on option quotes (mid must sit between intrinsic value and stock price)
No shadow-position code, no short-side code (see config.py module docstring).
"""

import csv
import logging
import os
import time
from datetime import date, datetime, timedelta, timezone
from typing import Optional

from alpaca.trading.enums import OrderSide, TimeInForce, ContractType, PositionIntent
from alpaca.trading.requests import (
    GetOptionContractsRequest, LimitOrderRequest, MarketOrderRequest, GetOrdersRequest,
)
from alpaca.trading.enums import QueryOrderStatus

import config as cfg
import market as mkt
from pricing import implied_vol_call

log = logging.getLogger(__name__)


def _is_earnings_headline(headline: str) -> bool:
    h = (headline or "").lower()
    patterns = (
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
    return any(p in h for p in patterns)


# ═══════════════════════════════════════════════════════════════════════════════
# CONTRACT SELECTION
# ═══════════════════════════════════════════════════════════════════════════════


def _passes_contract_checks(c, quote, stock_price: float, ticker: str) -> bool:
    """Shared liquidity/quote-sanity gate -- used both for the initial target pick and to make
    sure a mispricing-switch candidate is independently tradeable, not just cheap on paper."""
    if getattr(c, "tradable", True) is False:
        return False
    root = getattr(c, "root_symbol", None)
    if root and root != ticker:
        return False
    if quote is None or quote["mid"] <= 0:
        return False
    intrinsic = max(0.0, stock_price - float(c.strike_price))
    if quote["mid"] < intrinsic * 0.9 or quote["mid"] > stock_price:
        return False
    oi = getattr(c, "open_interest", None)
    if oi is not None and int(oi) < cfg.MIN_OPEN_INTEREST:
        return False
    return mkt.passes_guardrails(quote)


def _fit_quadratic(xs: list, ys: list):
    """Least-squares quadratic y = a*x^2 + b*x + c via the normal equations, solved with Cramer's
    rule. No numpy -- v2 stays dependency-lean (see requirements.txt). Returns None if the strikes
    are degenerate (e.g. all identical)."""
    n = len(xs)
    s1 = sum(xs); s2 = sum(x * x for x in xs); s3 = sum(x ** 3 for x in xs); s4 = sum(x ** 4 for x in xs)
    sy = sum(ys); sxy = sum(x * y for x, y in zip(xs, ys)); sxxy = sum(x * x * y for x, y in zip(xs, ys))
    A = [[s4, s3, s2], [s3, s2, s1], [s2, s1, n]]
    B = [sxxy, sxy, sy]

    def det3(m):
        return (m[0][0] * (m[1][1] * m[2][2] - m[1][2] * m[2][1])
              - m[0][1] * (m[1][0] * m[2][2] - m[1][2] * m[2][0])
              + m[0][2] * (m[1][0] * m[2][1] - m[1][1] * m[2][0]))

    det = det3(A)
    if abs(det) < 1e-12:
        return None
    coeffs = []
    for col in range(3):
        Ai = [row[:] for row in A]
        for r in range(3):
            Ai[r][col] = B[r]
        coeffs.append(det3(Ai) / det)
    return tuple(coeffs)   # (a, b, c)


def _maybe_switch_to_cheaper_contract(ticker: str, target: dict, stock_price: float, all_contracts: list) -> dict:
    """jeff's contract-selection idea (2026-07-18): switch off the target contract ONLY if a
    same-expiry neighbor's implied vol sits MISPRICING_MIN_RESIDUAL_GAP below a local quadratic
    smile fit AND passes the exact same liquidity/quote-sanity checks the target had to pass --
    never switches into a contract that wouldn't have been independently tradeable. v1 runs this
    same check as a shadow only (never switches); see config.py for the validation status."""
    if not cfg.MISPRICING_SWITCH_ENABLED:
        return target
    try:
        same_expiry = [c for c in all_contracts if str(c.expiration_date) == target["expiry"]]
        same_expiry.sort(key=lambda c: abs(float(c.strike_price) - target["strike"]))
        same_expiry = same_expiry[:7]
        if len(same_expiry) < 4:
            return target

        dte_days = (datetime.strptime(target["expiry"], "%Y-%m-%d").date() - date.today()).days
        T = max(dte_days, 1) / 365.0
        candidates = []
        for c in same_expiry:
            quote = mkt.get_option_quote(c.symbol)
            iv = implied_vol_call(quote["mid"], stock_price, float(c.strike_price), T) if quote else None
            if iv is None or iv <= 0.02 or iv >= 4.9:   # solver-boundary / degenerate, not real
                continue
            candidates.append({"symbol": c.symbol, "strike": float(c.strike_price), "iv": iv,
                                "quote": quote, "contract_obj": c,
                                "ok": _passes_contract_checks(c, quote, stock_price, ticker)})
        if len(candidates) < 4:
            return target

        fit = _fit_quadratic([c["strike"] for c in candidates], [c["iv"] for c in candidates])
        if fit is None:
            return target
        a, b, cc = fit
        for cand in candidates:
            cand["residual"] = cand["iv"] - (a * cand["strike"] ** 2 + b * cand["strike"] + cc)

        target_cand = next((c for c in candidates if c["symbol"] == target["symbol"]), None)
        eligible = [c for c in candidates if c["ok"]]
        if target_cand is None or not eligible:
            return target

        cheapest = min(eligible, key=lambda c: c["residual"])
        gap = target_cand["residual"] - cheapest["residual"]
        if cheapest["symbol"] == target["symbol"] or gap < cfg.MISPRICING_MIN_RESIDUAL_GAP:
            return target

        log.info("  🔍 mispricing switch: %s (resid %+.4f) -> %s (resid %+.4f, gap %.4f) [%s]",
                 target["symbol"], target_cand["residual"], cheapest["symbol"], cheapest["residual"], gap, ticker)
        q, oi = cheapest["quote"], getattr(cheapest["contract_obj"], "open_interest", None)
        return {"symbol": cheapest["symbol"], "strike": cheapest["strike"], "expiry": target["expiry"],
                "bid": q["bid"], "ask": q["ask"], "mid": q["mid"], "spread_pct": q["spread_pct"],
                "open_interest": int(oi) if oi is not None else None}
    except Exception as e:
        log.debug("mispricing switch check failed for %s: %s", ticker, e)
        return target


def pick_call_contract(ticker: str, stock_price: float, dte_min: int, dte_max: int,
                       target_delta: float, strike_hi_mult: float = 1.10,
                       strike_lo_mult: float = 0.95) -> Optional[dict]:
    today = date.today()
    try:
        resp = mkt.trading_client.get_option_contracts(GetOptionContractsRequest(
            underlying_symbols=[ticker], type=ContractType.CALL,
            expiration_date_gte=today + timedelta(days=dte_min),
            expiration_date_lte=today + timedelta(days=dte_max),
            strike_price_gte=str(round(stock_price * strike_lo_mult, 2)),
            strike_price_lte=str(round(stock_price * strike_hi_mult, 2)),
        ))
    except Exception as e:
        log.warning("Contract lookup failed for %s: %s", ticker, e)
        return None

    contracts = getattr(resp, "option_contracts", []) or []
    if not contracts:
        log.info("  → no contracts found for %s in range", ticker)
        return None

    dte_target = round((dte_min + dte_max) / 2)
    target_strike = stock_price * (1 + (1 - target_delta) * 0.15)
    contracts.sort(key=lambda c: (
        abs((datetime.strptime(str(c.expiration_date), "%Y-%m-%d").date() - today).days - dte_target),
        abs(float(c.strike_price) - target_strike),
    ))

    target = None
    for c in contracts[:5]:
        quote = mkt.get_option_quote(c.symbol)
        if not _passes_contract_checks(c, quote, stock_price, ticker):
            continue
        oi = getattr(c, "open_interest", None)
        target = {"symbol": c.symbol, "strike": float(c.strike_price), "expiry": str(c.expiration_date),
                  "bid": quote["bid"], "ask": quote["ask"], "mid": quote["mid"],
                  "spread_pct": quote["spread_pct"], "open_interest": int(oi) if oi is not None else None}
        break

    if target is None:
        log.info("  → no liquid contracts found for %s after guardrail checks", ticker)
        return None

    return _maybe_switch_to_cheaper_contract(ticker, target, stock_price, contracts)


# ═══════════════════════════════════════════════════════════════════════════════
# OPTION EXECUTION
# ═══════════════════════════════════════════════════════════════════════════════


def place_option_trade(ticker: str, stock_price: float, signal: dict, position_usd: float,
                       strategy: str, dte_min: int, dte_max: int, target_delta: float,
                       strike_hi_mult: float = 1.10, budget_mult: float = None) -> Optional[dict]:
    budget_mult = budget_mult if budget_mult is not None else cfg.MAX_CONTRACT_BUDGET_MULT
    contract = pick_call_contract(ticker, stock_price, dte_min, dte_max, target_delta, strike_hi_mult)
    if not contract:
        return None
    if contract["symbol"] in mkt._monitored_positions:
        log.info("  → already holding %s — skipping duplicate buy [%s]", contract["symbol"], strategy)
        return None

    contract_cost = contract["ask"] * 100
    if contract_cost > position_usd * budget_mult:
        log.info("  → skipping %s [%s]: 1 contract $%.0f > %.1fx budget $%.0f",
                 contract["symbol"], strategy, contract_cost, budget_mult, position_usd)
        return None

    qty = max(1, int(position_usd / contract_cost))
    log.info("  📋 %s strike=$%.2f exp=%s bid=$%.4f ask=$%.4f spread=%.1f%% qty=%d",
             contract["symbol"], contract["strike"], contract["expiry"],
             contract["bid"], contract["ask"], contract["spread_pct"] * 100, qty)

    if cfg.DRY_RUN:
        log.info("🧪 DRY RUN — would BUY %s x%d limit=$%.4f max_loss=$%.0f [%s] (no order submitted)",
                 contract["symbol"], qty, contract["ask"], contract["ask"] * 100 * qty, strategy)
        return None

    # Limit order at ask — avoids "no quote" rejection on market orders.
    req = LimitOrderRequest(symbol=contract["symbol"], qty=qty, side=OrderSide.BUY,
                           time_in_force=TimeInForce.DAY, limit_price=round(contract["ask"], 2),
                           position_intent=PositionIntent.BUY_TO_OPEN)
    try:
        mkt.trading_client.submit_order(req)
        log.info("✅ OPTION BUY  %s x%d  limit=$%.4f  max_loss=$%.0f",
                 contract["symbol"], qty, contract["ask"], contract["ask"] * 100 * qty)
    except Exception as e:
        log.error("Option buy failed: %s", e)
        return None

    # No exchange-held stop (account can't hold uncovered option sell-stops) — the in-process
    # monitor is the sole protection, exactly as on v1.
    entry = contract["ask"]
    mkt._monitored_positions[contract["symbol"]] = {
        "qty": qty, "entry_price": entry, "peak_price": entry, "underlying": ticker,
        "asset_type": "option", "strategy": strategy, "entry_dt": datetime.now(timezone.utc),
    }
    log.info("🔒 Trailing stop registered for %s @ $%.4f [%s]", contract["symbol"], entry, strategy)
    mkt._recent_trades[ticker] = time.time()
    log_trade(ticker, contract, qty, signal, entry, strategy=strategy)
    return contract


def _news_call_count_today(ticker: str) -> int:
    """How many news_call entries already logged for `ticker` today (UTC). Durable across
    restarts (reads trades.csv) — backs the per-ticker daily cap against echoed/republished news."""
    try:
        today = datetime.now(timezone.utc).date().isoformat()
        n = 0
        with open(mkt.TRADES_FILE, newline="") as f:
            for r in csv.DictReader(f):
                if (r.get("strategy") == "news_call" and r.get("source_ticker") == ticker
                        and (r.get("timestamp") or "")[:10] == today):
                    n += 1
        return n
    except Exception:
        return 0


def execute_news_call(ticker: str, signal: dict):
    if mkt._at_position_cap("news_call"):
        return
    n_today = _news_call_count_today(ticker)
    if n_today >= cfg.NEWS_CALL_MAX_PER_TICKER_DAY:
        log.info("  → news_call SKIP %s: already %d trade(s) today (cap %d)",
                 ticker, n_today, cfg.NEWS_CALL_MAX_PER_TICKER_DAY)
        return
    stock_price = mkt.get_stock_price(ticker)
    if not stock_price or not mkt.passes_stock_price_guardrail(ticker, stock_price):
        return
    budget = mkt.position_budget_usd()
    log.info("  → %s @ $%.2f — searching for news_call contract size=$%.0f", ticker, stock_price, budget)
    place_option_trade(ticker, stock_price, signal, budget, strategy="news_call",
                       dte_min=cfg.NEWS_CALL_DTE_MIN, dte_max=cfg.NEWS_CALL_DTE_MAX,
                       target_delta=cfg.NEWS_CALL_TARGET_DELTA)


def execute_lotto(ticker: str, signal: dict):
    if mkt.near_market_close(within_min=cfg.LOTTO_ENTRY_CUTOFF_MIN_BEFORE_CLOSE):
        log.info("  → %s: too close to the same-day forced exit (<%dmin left) — skipping new lotto entry",
                 ticker, cfg.LOTTO_ENTRY_CUTOFF_MIN_BEFORE_CLOSE)
        return
    if mkt._at_position_cap("lotto"):
        return
    if any(p.get("strategy") == "lotto" and p.get("underlying") == ticker
           for p in mkt._monitored_positions.values()):
        log.info("  → already holding lotto on %s, skipping duplicate", ticker)
        return
    stock_price = mkt.get_stock_price(ticker)
    if not stock_price or not mkt.passes_stock_price_guardrail(ticker, stock_price):
        return
    budget = mkt.lotto_budget_usd()
    log.info("  → %s @ $%.2f — buying LOTTO OTM call size=$%.0f Δ~%.2f",
             ticker, stock_price, budget, cfg.LOTTO_TARGET_DELTA)
    place_option_trade(ticker, stock_price, signal, budget, strategy="lotto",
                       dte_min=cfg.LOTTO_MIN_DTE, dte_max=cfg.LOTTO_MAX_DTE,
                       target_delta=cfg.LOTTO_TARGET_DELTA, strike_hi_mult=cfg.LOTTO_STRIKE_HI_MULT,
                       budget_mult=1.0)   # HARD cap -- lotto never exceeds its own budget


def close_option_position(symbol: str, qty: int, reason: str = "trailing stop"):
    """Marketable limit sell-to-close (limit just below bid, fills immediately). Returns
    (submitted_ok, actual_fill_price | None) -- caller only de-registers on submitted_ok=True."""
    if not mkt.market_is_open():
        log.warning("  → market closed/halted — deferring close of %s", symbol)
        return False, None
    quote = mkt.get_option_quote(symbol)
    if quote and quote["bid"] > 0:
        limit_price = max(round(quote["bid"] * 0.90, 2), 0.01)
    else:
        log.warning("  → no bid quote for %s; placing floor limit sell (may not fill)", symbol)
        limit_price = 0.01
    req = LimitOrderRequest(symbol=symbol, qty=qty, side=OrderSide.SELL, time_in_force=TimeInForce.DAY,
                           limit_price=limit_price, position_intent=PositionIntent.SELL_TO_CLOSE)
    try:
        order = mkt.trading_client.submit_order(req)
        fill = _await_fill_price(order)
        log.info("✅ SELL TO CLOSE  %s x%d (%s) limit=%.2f%s", symbol, qty, reason, limit_price,
                 f"  fill=${fill:.4f}" if fill else "  (fill pending → mid fallback)")
        return True, fill
    except Exception as e:
        log.error("Close order failed for %s: %s — will retry next cycle", symbol, e)
        return False, None


# ═══════════════════════════════════════════════════════════════════════════════
# STOCK EXECUTION (PEAD only in v2 -- see config.py module docstring)
# ═══════════════════════════════════════════════════════════════════════════════


def place_stock_trade(ticker: str, stock_price: float, signal: dict, position_usd: float,
                      strategy: str = "pead") -> Optional[dict]:
    shares = max(1, int(position_usd / stock_price))
    if cfg.DRY_RUN:
        log.info("🧪 DRY RUN — would BUY %s x%d @ ~$%.2f [%s] (no order submitted)",
                 ticker, shares, stock_price, strategy)
        return None
    try:
        mkt.trading_client.submit_order(MarketOrderRequest(
            symbol=ticker, qty=shares, side=OrderSide.BUY, time_in_force=TimeInForce.DAY))
        log.info("✅ STOCK BUY  %s x%d @ ~$%.2f  [%s]", ticker, shares, stock_price, strategy)
    except Exception as e:
        # Trading-halt/auction rejects MARKET orders; retry once with a marketable limit.
        if "halt" in str(e).lower() or "limit order" in str(e).lower():
            try:
                mkt.trading_client.submit_order(LimitOrderRequest(
                    symbol=ticker, qty=shares, side=OrderSide.BUY, time_in_force=TimeInForce.DAY,
                    limit_price=round(stock_price * 1.01, 2)))
                log.info("✅ STOCK BUY (limit, halt fallback)  %s x%d  [%s]", ticker, shares, strategy)
            except Exception as e2:
                log.error("Stock buy failed for %s (market+limit): %s", ticker, e2)
                return None
        else:
            log.error("Stock buy failed for %s: %s", ticker, e)
            return None

    key = f"{ticker}__{strategy}"
    mkt._monitored_positions[key] = {
        "qty": shares, "entry_price": stock_price, "peak_price": stock_price, "underlying": ticker,
        "asset_type": "stock", "strategy": strategy, "entry_dt": datetime.now(timezone.utc),
    }
    mkt._recent_trades[ticker] = time.time()
    log_trade_stock(ticker, shares, stock_price, signal, strategy=strategy)
    return {"ticker": ticker, "shares": shares, "price": stock_price}


def close_stock_position(key: str, qty: int, reason: str = "trailing stop"):
    ticker = key.split("__")[0]
    req = MarketOrderRequest(symbol=ticker, qty=qty, side=OrderSide.SELL, time_in_force=TimeInForce.DAY)
    try:
        order = mkt.trading_client.submit_order(req)
        fill = _await_fill_price(order)
        log.info("✅ STOCK SELL  %s x%d (%s)%s", ticker, qty, reason,
                 f"  fill=${fill:.4f}" if fill else "  (fill pending → mid fallback)")
        return True, fill
    except Exception as e:
        log.error("Stock close failed for %s: %s — will retry next cycle", ticker, e)
        return False, None


def execute_pead(ticker: str, signal: dict):
    if not _is_earnings_headline(signal.get("_headline", "")):
        return
    if mkt._at_position_cap("pead"):
        return
    stock_price = mkt.get_stock_price(ticker)
    if not stock_price or not mkt.passes_stock_price_guardrail(ticker, stock_price):
        return
    key = f"{ticker}__pead"
    if key in mkt._monitored_positions:
        log.info("  → already holding pead position in %s, skipping duplicate", ticker)
        return
    budget = mkt.position_budget_usd()
    log.info("  → %s @ $%.2f — buying PEAD stock size=$%.0f", ticker, stock_price, budget)
    place_stock_trade(ticker, stock_price, signal, budget, strategy="pead")


def _await_fill_price(order, timeout_s: float = 3.0) -> Optional[float]:
    try:
        oid = order.id
    except Exception:
        return None
    deadline = time.time() + timeout_s
    while time.time() < deadline:
        try:
            o = mkt.trading_client.get_order_by_id(oid)
            fap = getattr(o, "filled_avg_price", None)
            if fap is not None and float(fap) > 0:
                return float(fap)
            if any(x in str(getattr(o, "status", "")).lower() for x in ("canceled", "rejected", "expired")):
                return None
        except Exception:
            pass
        time.sleep(0.3)
    return None


def _realized_pnl(asset_type: str, entry: float, exit_price: float, qty: float) -> float:
    if asset_type == "stock":
        return (exit_price - entry) * qty
    return (exit_price - entry) * 100 * qty


# ═══════════════════════════════════════════════════════════════════════════════
# TRADE LOGGING
# ═══════════════════════════════════════════════════════════════════════════════


def log_trade(ticker, contract, qty, signal, entry_price, strategy):
    write_header = not os.path.exists(mkt.TRADES_FILE)
    with open(mkt.TRADES_FILE, "a", newline="") as f:
        w = csv.writer(f)
        if write_header:
            w.writerow(["timestamp", "source_ticker", "option_symbol", "strike", "expiry", "qty",
                        "bid", "ask", "cost_basis", "max_loss", "spread_pct", "confidence",
                        "magnitude", "reasoning", "strategy", "corrected_ticker", "news_source"])
        w.writerow([datetime.now(timezone.utc).isoformat(), ticker, contract["symbol"],
                   contract["strike"], contract["expiry"], qty, f"{contract['bid']:.4f}",
                   f"{contract['ask']:.4f}", f"{entry_price:.4f}", f"{entry_price * 100 * qty:.2f}",
                   f"{contract['spread_pct'] * 100:.1f}%", signal.get("confidence"),
                   signal.get("magnitude"), signal.get("reasoning", ""), strategy,
                   signal.get("_ticker_corrected", ""), signal.get("_source", "")])


def log_trade_stock(ticker, shares, entry_price, signal, strategy):
    write_header = not os.path.exists(mkt.TRADES_FILE)
    with open(mkt.TRADES_FILE, "a", newline="") as f:
        w = csv.writer(f)
        if write_header:
            w.writerow(["timestamp", "source_ticker", "option_symbol", "strike", "expiry", "qty",
                        "bid", "ask", "cost_basis", "max_loss", "spread_pct", "confidence",
                        "magnitude", "reasoning", "strategy", "corrected_ticker", "news_source"])
        w.writerow([datetime.now(timezone.utc).isoformat(), ticker, "-", "-", "-", shares,
                   f"{entry_price:.4f}", f"{entry_price:.4f}", f"{entry_price:.4f}",
                   f"{entry_price * shares:.2f}", "0.0%", signal.get("confidence"),
                   signal.get("magnitude"), signal.get("reasoning", ""), strategy,
                   signal.get("_ticker_corrected", ""), signal.get("_source", "")])


def log_closed_trade(symbol: str, pos: dict, exit_price: float, pnl_usd: float, reason: str = "stop"):
    write_header = not os.path.exists(mkt.CLOSED_TRADES_FILE)
    asset_type = pos.get("asset_type", "option")
    entry = pos["entry_price"]
    pnl_pct = ((exit_price - entry) / entry * 100) if entry > 0 and asset_type == "stock" else (
               (exit_price / entry - 1) * 100 if entry > 0 else 0)
    with open(mkt.CLOSED_TRADES_FILE, "a", newline="") as f:
        w = csv.writer(f)
        if write_header:
            w.writerow(["timestamp", "symbol", "underlying", "strategy", "asset_type", "qty",
                        "entry_price", "exit_price", "pnl_usd", "pnl_pct", "peak_price", "reason"])
        w.writerow([datetime.now(timezone.utc).isoformat(), symbol, pos.get("underlying", symbol),
                   pos.get("strategy", "unknown"), asset_type, pos.get("qty", 0), f"{entry:.4f}",
                   f"{exit_price:.4f}", f"{pnl_usd:.2f}", f"{pnl_pct:.2f}",
                   f"{pos.get('peak_price', entry):.4f}", reason])


def log_regime_decision(signal, in_uptrend):
    try:
        new = not os.path.exists(mkt.REGIME_DECISIONS_FILE)
        with open(mkt.REGIME_DECISIONS_FILE, "a", newline="") as f:
            w = csv.writer(f)
            if new:
                w.writerow(["ts", "tickers", "in_uptrend", "magnitude", "confidence", "headline"])
            w.writerow([datetime.now(timezone.utc).isoformat(),
                       "|".join((signal.get("tickers") or [])[:2]), int(bool(in_uptrend)),
                       signal.get("magnitude"), signal.get("confidence"),
                       (signal.get("_headline", "") or "")[:80]])
    except Exception as e:
        log.debug("regime-decision log failed: %s", e)


# ═══════════════════════════════════════════════════════════════════════════════
# TRAILING STOP MONITOR
# ═══════════════════════════════════════════════════════════════════════════════

import asyncio


async def trailing_stop_monitor():
    log.info("📊 Trailing stop monitor started (interval %ds)", cfg.MONITOR_INTERVAL)
    while True:
        await asyncio.sleep(cfg.MONITOR_INTERVAL)
        mkt._loss_reset_if_new_day()

        if not mkt._monitored_positions:
            continue
        if mkt._deferred_closes and mkt.market_is_open():
            mkt._deferred_closes.clear()

        for symbol, pos in list(mkt._monitored_positions.items()):
            asset_type = pos.get("asset_type", "option")
            strategy = pos.get("strategy", asset_type)

            if asset_type == "stock":
                underlying = pos.get("underlying", symbol.split("__")[0])
                mid = mkt._price_cache.get(underlying) or mkt.get_stock_price(underlying, subscribe=False)
            else:
                q = mkt.get_option_quote(symbol)
                mid = q["mid"] if q else None
                if q and q.get("spread_pct", 0) > cfg.MAX_MONITOR_SPREAD_PCT:
                    mid = None   # blown-spread garbage quote -> treat as no-quote (protects peak/stop)

            if mid is None:
                continue

            entry, qty = pos["entry_price"], pos["qty"]

            if strategy == "lotto" and asset_type == "option":
                # Lotto exit rule (research/lotto_intraday_exit_test.py, 2026-07-17): entry-anchored
                # stop-loss + forced same-day EOD exit, replacing peak-trailing entirely for this leg.
                pnl_usd = (mid - entry) * 100 * qty
                pnl_pct = ((mid - entry) / entry * 100) if entry > 0 else 0
                stop_price = entry * (1 - cfg.LOTTO_STOP_LOSS_PCT) if entry else 0.0
                log.info("👁  %-24s entry=$%.4f mid=$%.4f stop=$%.4f P&L %+.1f%% [lotto]",
                         symbol, entry, mid, stop_price, pnl_pct)
                mkt._write_bot_state()

                # Hard profit cap (asymmetric-safe: only ever exits a WINNER early, never adds a loss).
                if (cfg.LOTTO_HARD_CAP_ENABLED and entry > 0 and mid >= cfg.LOTTO_HARD_CAP_MULT * entry):
                    if not mkt.market_is_open():
                        mkt._log_deferred_once(symbol, "⏳ %s lotto cap hit but market closed — deferring")
                        continue
                    ok, fill = close_option_position(symbol, qty, reason=f"lotto {cfg.LOTTO_HARD_CAP_MULT:.0f}x cap")
                    if not ok:
                        continue
                    exit_px = fill if fill is not None else mid
                    pnl_usd = _realized_pnl(asset_type, entry, exit_px, qty)
                    log_closed_trade(symbol, pos, exit_px, pnl_usd, reason="lotto_cap")
                    del mkt._monitored_positions[symbol]
                    continue

                if mkt._lotto_stop_loss_hit(pos, mid):
                    if not mkt.market_is_open():
                        mkt._log_deferred_once(symbol, "⏳ %s lotto stop-loss hit but market closed — deferring")
                        continue
                    ok, fill = close_option_position(symbol, qty, reason="lotto stop-loss")
                    if not ok:
                        log.warning("⏳ Close not confirmed for %s — keeping monitored, will retry", symbol)
                        continue
                    exit_px = fill if fill is not None else mid
                    pnl_usd = _realized_pnl(asset_type, entry, exit_px, qty)
                    log_closed_trade(symbol, pos, exit_px, pnl_usd, reason="lotto_stop_loss")
                    if pnl_usd < 0:
                        mkt._daily_loss_usd += abs(pnl_usd)
                    del mkt._monitored_positions[symbol]
                    continue

                if mkt._lotto_same_day_exit_hit(pos) and mkt.near_market_close():
                    ok, fill = close_option_position(symbol, qty, reason="lotto same-day exit")
                    if ok:
                        exit_px = fill if fill is not None else mid
                        pnl_usd = _realized_pnl(asset_type, entry, exit_px, qty)
                        log_closed_trade(symbol, pos, exit_px, pnl_usd, reason="lotto_eod")
                        if pnl_usd < 0:
                            mkt._daily_loss_usd += abs(pnl_usd)
                        del mkt._monitored_positions[symbol]
                    continue

                # Fallback safety net: shouldn't normally fire since same-day exit fires first.
                if mkt._time_stop_hit(pos) and mkt.near_market_close():
                    ok, fill = close_option_position(symbol, qty, reason="time-stop")
                    if ok:
                        exit_px = fill if fill is not None else mid
                        pnl_usd = _realized_pnl(asset_type, entry, exit_px, qty)
                        log_closed_trade(symbol, pos, exit_px, pnl_usd, reason="time-stop")
                        if pnl_usd < 0:
                            mkt._daily_loss_usd += abs(pnl_usd)
                        del mkt._monitored_positions[symbol]
                    continue

                continue

            # Time-stop: exit at end-of-day on the final hold day (avoid the overnight gap).
            if mkt._time_stop_hit(pos) and mkt.near_market_close():
                if asset_type == "stock":
                    ok, fill = close_stock_position(symbol, qty, reason="time-stop")
                else:
                    ok, fill = close_option_position(symbol, qty, reason="time-stop")
                if ok:
                    exit_px = fill if fill is not None else mid
                    pnl_usd = _realized_pnl(asset_type, entry, exit_px, qty)
                    log_closed_trade(symbol, pos, exit_px, pnl_usd, reason="time-stop")
                    if pnl_usd < 0:
                        mkt._daily_loss_usd += abs(pnl_usd)
                    del mkt._monitored_positions[symbol]
                continue

            peak = pos["peak_price"]
            gain = (peak / entry - 1.0) if entry else 0.0
            trail = long_trail_for(strategy, asset_type, gain)
            if mid > peak:
                pos["peak_price"] = mid
                peak = mid
                gain = (peak / entry - 1.0) if entry else 0.0
                trail = long_trail_for(strategy, asset_type, gain)
            stop_price = peak * (1 - trail)
            if strategy == "pead" and cfg.PEAD_BREAKEVEN_LOCK > 0 and gain >= cfg.PEAD_BREAKEVEN_LOCK:
                stop_price = max(stop_price, entry * 1.005)

            pnl_usd = (mid - entry) * qty if asset_type == "stock" else (mid - entry) * 100 * qty
            pnl_pct = ((mid - entry) / entry * 100) if entry > 0 else 0
            log.info("👁  %-24s entry=$%.4f mid=$%.4f peak=$%.4f stop=$%.4f P&L %+.1f%% [%s]",
                     symbol, entry, mid, peak, stop_price, pnl_pct, strategy)
            mkt._write_bot_state()

            if mid <= stop_price:
                if not mkt.market_is_open():
                    mkt._log_deferred_once(symbol, "⏳ %s stop breached but market closed — deferring")
                    continue
                if asset_type == "stock":
                    ok, fill = close_stock_position(symbol, qty)
                else:
                    ok, fill = close_option_position(symbol, qty)
                if not ok:
                    log.warning("⏳ Close not confirmed for %s — keeping monitored, will retry", symbol)
                    continue
                exit_px = fill if fill is not None else mid
                pnl_usd = _realized_pnl(asset_type, entry, exit_px, qty)
                log_closed_trade(symbol, pos, exit_px, pnl_usd, reason="stop")
                if pnl_usd < 0:
                    mkt._daily_loss_usd += abs(pnl_usd)
                del mkt._monitored_positions[symbol]


# ═══════════════════════════════════════════════════════════════════════════════
# RESTART RECOVERY
# ═══════════════════════════════════════════════════════════════════════════════


def _parse_option_ticker(symbol: str) -> str:
    import re
    m = re.match(r"^([A-Z]+)\d{6}[CP]\d+$", symbol)
    return m.group(1) if m else symbol[:4]


def load_open_positions_from_alpaca():
    """On startup, register every open Alpaca position with the monitor so a restart never
    leaves an orphaned, unprotected position. Cancels stale open sell orders first."""
    try:
        for o in mkt.trading_client.get_orders(filter=GetOrdersRequest(status=QueryOrderStatus.OPEN)):
            if o.side == OrderSide.SELL:
                try:
                    mkt.trading_client.cancel_order_by_id(o.id)
                    log.info("  🗑  Cancelled stale order for %s", o.symbol)
                except Exception as ce:
                    log.warning("  Could not cancel stale order %s: %s", o.id, ce)
    except Exception as e:
        log.warning("Could not list open orders on startup: %s", e)

    try:
        positions = mkt.trading_client.get_all_positions()
    except Exception as e:
        log.warning("Could not fetch open positions from Alpaca: %s", e)
        return

    import json
    try:
        with open(mkt.BOT_STATE_FILE) as f:
            saved = json.load(f)
    except Exception:
        saved = {}
    entry_log: dict[str, dict] = {}
    try:
        with open(mkt.TRADES_FILE) as f:
            for row in csv.DictReader(f):
                sym = row.get("option_symbol")
                if sym and sym != "-":
                    entry_log[sym] = row
    except Exception:
        pass

    loaded = 0
    for p in positions:
        symbol = p.symbol
        if symbol in mkt._monitored_positions:
            continue
        try:
            entry = float(p.avg_entry_price)
            qty = int(float(p.qty))
            asset_class = str(p.asset_class).lower().replace("assetclass.", "")
            if asset_class != "us_option":
                continue   # equity positions handled via bot_state.json below
            quote = mkt.get_option_quote(symbol)
            current = quote["mid"] if quote else entry
            info = saved.get(symbol, {})
            entry_row = entry_log.get(symbol, {})
            strategy = entry_row.get("strategy") or info.get("strategy") or "news_call"
            peak = max(entry, current, float(info.get("peak_price", 0) or 0))
            edt = info.get("entry_dt") or entry_row.get("timestamp")
            try:
                entry_dt = datetime.fromisoformat(edt) if edt else None
            except Exception:
                entry_dt = None
            mkt._monitored_positions[symbol] = {
                "qty": qty, "entry_price": entry, "peak_price": peak,
                "underlying": _parse_option_ticker(symbol), "asset_type": "option",
                "strategy": strategy, "entry_dt": entry_dt,
            }
            loaded += 1
            log.info("  📂 Loaded position: %s entry=$%.4f qty=%d [%s]", symbol, entry, qty, strategy)
        except Exception as e:
            log.warning("Could not load position %s: %s", symbol, e)

    # Reload stock (pead) positions from bot_state.json -- Alpaca can't distinguish
    # bot-managed equity positions from anything else the account might hold.
    for symbol, info in saved.items():
        if info.get("asset_type") != "stock" or symbol in mkt._monitored_positions:
            continue
        try:
            ticker = info.get("underlying", symbol)
            strategy = info.get("strategy", "pead")
            p = mkt.trading_client.get_open_position(ticker)
            qty = abs(int(float(p.qty)))
            if qty <= 0:
                continue
            entry = float(p.avg_entry_price)
            raw_edt = info.get("entry_dt")
            entry_dt_restored = None
            if raw_edt:
                try:
                    entry_dt_restored = datetime.fromisoformat(raw_edt)
                    if entry_dt_restored.tzinfo is None:
                        entry_dt_restored = entry_dt_restored.replace(tzinfo=timezone.utc)
                except Exception:
                    pass
            peak = max(entry, info.get("peak_price", entry))
            mkt._monitored_positions[f"{ticker}__{strategy}"] = {
                "qty": qty, "entry_price": entry, "peak_price": peak, "underlying": ticker,
                "asset_type": "stock", "strategy": strategy, "entry_dt": entry_dt_restored,
            }
            log.info("  📂 Reloaded stock position: %s entry=$%.2f qty=%d [%s]", ticker, entry, qty, strategy)
        except Exception:
            pass   # position closed or not found -- skip silently

    if loaded:
        log.info("📂 Loaded %d open position(s) from Alpaca into the trailing stop monitor", loaded)
    else:
        log.info("📂 No existing option positions found on Alpaca")
