"""
grid_collector.py (v3) — normalized background market-data collection for contract selection and exit tuning.

Captures a wide strike/expiry grid of options for sufficiently impactful signals
(mag * conf >= cfg.GRID_LOG_THRESHOLD), supporting BOTH:
  - Bullish signals (capturing Call option grids)
  - Bearish signals (capturing Put option grids)

Storage Optimization:
  1. Normalized Relational Schema:
     - contract_grid_signals.csv: 1 row per signal containing the headline, scores,
       catalyst, stock price at signal, and market regime context.
     - contract_grid_snapshots.csv: recurring quote + Greeks snapshots referencing only
       signal_id (eliminating 21 redundant text/metadata columns per row).
  2. Same-Day Bounded Cadence:
     - Restricts polling to the remainder of the signal day (GRID_TRACK_SAME_DAY_ONLY,
       up to GRID_MAX_TRACK_MINUTES = 390m / 6.5h) rather than multi-week accumulation.

Pure market-data reads — NO ORDERS ARE EVER PLACED BY THIS MODULE.
Runs concurrently as background asyncio tasks without blocking trading.
"""

import asyncio
import csv
import logging
import os
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import Callable, Optional

from alpaca.data.historical.option import OptionHistoricalDataClient
from alpaca.data.historical.stock import StockHistoricalDataClient
from alpaca.data.requests import OptionSnapshotRequest, StockBarsRequest
from alpaca.data.timeframe import TimeFrame
from alpaca.trading.client import TradingClient
from alpaca.trading.enums import ContractType
from alpaca.trading.requests import GetOptionContractsRequest

try:
    from v3 import config as cfg
except ImportError:
    import config as cfg

log = logging.getLogger(__name__)

# Alpaca clients for market data
trading_client = TradingClient(cfg.ALPACA_KEY, cfg.ALPACA_SECRET, paper=True)
option_data_client = OptionHistoricalDataClient(cfg.ALPACA_KEY, cfg.ALPACA_SECRET)
stock_data_client = StockHistoricalDataClient(cfg.ALPACA_KEY, cfg.ALPACA_SECRET)

# 1 row per signal: static metadata, scores, and macro regime state
_SIGNAL_HEADER = [
    "signal_id", "signal_ts", "ticker", "headline", "source", "scorer_model",
    "sentiment", "magnitude", "confidence", "catalyst", "stock_price_at_signal",
    "spy_price", "spy_sma", "regime_uptrend", "regime_mom_ok", "regime_chop_ok", "regime_dd_density",
    "vol_proxy_symbol", "vol_proxy_price", "vol_proxy_pctile",
    "passes_news_call_gate", "passes_lotto_gate",
]

# Recurring snapshots: compact quote & Greeks time series referencing signal_id
_SNAPSHOT_HEADER = [
    "signal_id", "snapshot_ts", "minutes_since_signal", "stock_price",
    "symbol", "contract_type", "strike", "expiry", "dte_at_signal",
    "oi_at_signal", "bid", "ask", "mid", "spread_pct", "iv",
    "delta", "gamma", "theta", "vega", "rho",
]

# Alias for backward compatibility
_HEADER = _SNAPSHOT_HEADER

# In-memory tracking state:
# signal_id -> {
#   ticker, headline, source, sentiment, contract_type, magnitude, confidence,
#   signal_ts, last_poll, last_expiry (date), contracts_meta: {symbol: (strike, expiry_str, dte, oi)},
#   get_stock_price
# }
_tracked: dict[str, dict] = {}


def _chunks(lst: list, n: int):
    for i in range(0, len(lst), n):
        yield lst[i:i + n]


def _interval_for(elapsed_minutes: float) -> int:
    for max_minutes, interval_s in cfg.GRID_POLL_SCHEDULE:
        if elapsed_minutes <= max_minutes:
            return interval_s
    return cfg.GRID_POLL_SCHEDULE[-1][1]


def _write_signal_row(signal_id: str, state: dict) -> None:
    """Write 1 row of signal metadata to contract_grid_signals.csv."""
    try:
        csv_path = Path(getattr(cfg, "GRID_SIGNALS_CSV", Path(cfg.DATA_DIR) / "contract_grid_signals.csv"))
        csv_path.parent.mkdir(parents=True, exist_ok=True)
        new = not csv_path.exists()
        row = [
            signal_id, state["signal_ts"].isoformat(), state["ticker"], state["headline"],
            state["source"], state["scorer_model"], state.get("sentiment", ""),
            state["magnitude"], state["confidence"], state["catalyst"], state["stock_price_at_signal"],
            state["spy_price"], state["spy_sma"], state["regime_uptrend"], state["regime_mom_ok"],
            state["regime_chop_ok"], state["regime_dd_density"], state["vol_proxy_symbol"],
            state["vol_proxy_price"], state["vol_proxy_pctile"], state["passes_news_call_gate"],
            state["passes_lotto_gate"],
        ]
        with open(csv_path, "a", newline="") as f:
            w = csv.writer(f)
            if new:
                w.writerow(_SIGNAL_HEADER)
            w.writerow(row)
    except Exception as e:
        log.debug("grid collector: write signal metadata failed: %s", e)


def _write_snapshot_rows(rows: list[list]) -> None:
    """Write recurring quote/Greeks snapshot rows to contract_grid_snapshots.csv."""
    if not rows:
        return
    try:
        csv_path = Path(cfg.GRID_SNAPSHOT_CSV)
        csv_path.parent.mkdir(parents=True, exist_ok=True)
        new = not csv_path.exists()
        with open(csv_path, "a", newline="") as f:
            w = csv.writer(f)
            if new:
                w.writerow(_SNAPSHOT_HEADER)
            w.writerows(rows)
    except Exception as e:
        log.debug("grid collector: write snapshot rows failed: %s", e)


# Alias for backward compatibility
_write_rows = _write_snapshot_rows


def should_collect(signal: dict) -> bool:
    """Check if signal meets the research grid logging threshold."""
    if not cfg.GRID_COLLECTOR_ENABLED:
        return False
    try:
        mag = float(signal.get("magnitude", 0))
        conf = float(signal.get("confidence", 0))
    except (TypeError, ValueError):
        return False
    return mag * conf >= cfg.GRID_LOG_THRESHOLD


def _gate_eligibility(sentiment: str, magnitude: float, confidence: float) -> tuple[bool, bool]:
    """Check coarse eligibility for live call trade vs lotto gate."""
    call_ok = (sentiment == "bullish" and
               magnitude >= cfg.MIN_MAGNITUDE and
               confidence >= cfg.MIN_CONFIDENCE)
    lotto_ok = (sentiment == "bullish" and
                magnitude >= 0.70 and
                confidence >= 0.85)
    return call_ok, lotto_ok


_regime_cache: dict = {"date": None, "state": {}}


def _regime_snapshot() -> dict:
    """Capture SPY regime and market volatility proxy once per day."""
    today = date.today()
    if _regime_cache.get("date") == today:
        return _regime_cache["state"]

    state = {}
    if getattr(cfg, "REGIME_GATE_ENABLED", True):
        try:
            start = datetime.now(timezone.utc) - timedelta(days=int(cfg.REGIME_SPY_SMA_DAYS * 1.6) + 30)
            resp = stock_data_client.get_stock_bars(StockBarsRequest(
                symbol_or_symbols="SPY", timeframe=TimeFrame.Day, start=start, feed="iex"))
            closes = [float(b.close) for b in (resp.data or {}).get("SPY", [])]
            if len(closes) >= cfg.REGIME_SPY_SMA_DAYS:
                sma = sum(closes[-cfg.REGIME_SPY_SMA_DAYS:]) / cfg.REGIME_SPY_SMA_DAYS
                price = closes[-1]
                above_sma = price >= sma
                mom_days = cfg.REGIME_MOM_DAYS
                has_mom = mom_days > 0 and len(closes) > mom_days
                mom_ok = mom_days <= 0 or not has_mom or price >= closes[-1 - mom_days]
                dd_win = cfg.REGIME_DOWNDAY_WINDOW
                if dd_win > 0 and len(closes) > dd_win:
                    dd_density = sum(1 for a, b in zip(closes[-dd_win - 1:-1], closes[-dd_win:])
                                      if b < a) / dd_win
                else:
                    dd_density = 0.0
                chop_ok = dd_win <= 0 or dd_density < cfg.REGIME_DOWNDAY_MAX_DENSITY
                state = {
                    "spy_price": price, "spy_sma": round(sma, 2),
                    "regime_uptrend": above_sma and mom_ok and chop_ok,
                    "regime_mom_ok": mom_ok, "regime_chop_ok": chop_ok,
                    "regime_dd_density": round(dd_density, 3),
                }
        except Exception as e:
            log.debug("grid collector: regime snapshot failed: %s", e)

    state.update(_vol_proxy_snapshot())
    _regime_cache.update(date=today, state=state)
    return state


def _vol_proxy_snapshot() -> dict:
    """Capture VIXY trailing lookback percentile rank."""
    sym = getattr(cfg, "GRID_VOL_PROXY_SYMBOL", "VIXY")
    lookback = getattr(cfg, "GRID_VOL_PROXY_LOOKBACK_DAYS", 20)
    try:
        start = datetime.now(timezone.utc) - timedelta(days=int(lookback * 2.5) + 10)
        resp = stock_data_client.get_stock_bars(StockBarsRequest(
            symbol_or_symbols=sym, timeframe=TimeFrame.Day, start=start, feed="iex"))
        closes = [float(b.close) for b in (resp.data or {}).get(sym, [])]
        if len(closes) < 2:
            return {}
        window = closes[-lookback:]
        latest = window[-1]
        pctile = sum(1 for c in window if c <= latest) / len(window)
        return {"vol_proxy_symbol": sym, "vol_proxy_price": latest, "vol_proxy_pctile": round(pctile, 3)}
    except Exception as e:
        log.debug("grid collector: vol proxy snapshot failed: %s", e)
        return {}


def _current_stock_price(state: dict) -> str:
    try:
        get_price = state.get("get_stock_price")
        price = get_price(state["ticker"]) if get_price else None
        return price if price is not None else ""
    except Exception:
        return ""


def _fetch_contracts_meta(ticker: str, stock_price: float, contract_type: ContractType) -> dict:
    """
    Fetch paginated contracts metadata from Alpaca.
    Returns: {symbol: (strike, expiry_str, dte, oi)}
    """
    today = date.today()
    min_exp = today + timedelta(days=cfg.GRID_DTE_MIN)
    max_exp = today + timedelta(days=cfg.GRID_DTE_MAX)
    contracts = []
    page_token = None

    for _ in range(5):
        resp = trading_client.get_option_contracts(GetOptionContractsRequest(
            underlying_symbols=[ticker],
            type=contract_type,
            expiration_date_gte=min_exp,
            expiration_date_lte=max_exp,
            strike_price_gte=str(round(stock_price * cfg.GRID_STRIKE_LO_MULT, 2)),
            strike_price_lte=str(round(stock_price * cfg.GRID_STRIKE_HI_MULT, 2)),
            limit=500,
            page_token=page_token,
        ))
        contracts.extend(getattr(resp, "option_contracts", []) or [])
        page_token = getattr(resp, "next_page_token", None)
        if not page_token:
            break

    meta = {}
    for c in contracts:
        if getattr(c, "tradable", True) is False:
            continue
        root = getattr(c, "root_symbol", None)
        if root and root != ticker:
            continue
        expiry = c.expiration_date if isinstance(c.expiration_date, date) else \
            datetime.strptime(str(c.expiration_date), "%Y-%m-%d").date()
        dte = (expiry - today).days
        oi = getattr(c, "open_interest", None)
        meta[c.symbol] = (float(c.strike_price), expiry.isoformat(), dte,
                          int(oi) if oi is not None else None)
    return meta


def _snapshot_rows(signal_id: str, state: dict) -> list[list]:
    """Snapshot quotes and Greeks for all tracked contracts for this signal without redundant signal metadata."""
    contracts_meta = state["contracts_meta"]
    symbols = list(contracts_meta.keys())
    now = datetime.now(timezone.utc)
    minutes_since = round((now - state["signal_ts"]).total_seconds() / 60, 2)
    stock_price = _current_stock_price(state)
    contract_type_str = "call" if state.get("contract_type") == ContractType.CALL else "put"

    rows = []
    for chunk in _chunks(symbols, 100):
        try:
            snap = option_data_client.get_option_snapshot(
                OptionSnapshotRequest(symbol_or_symbols=chunk))
        except Exception as e:
            log.debug("grid collector: snapshot fetch failed for %s: %s", state["ticker"], e)
            continue

        for sym in chunk:
            strike, expiry, dte, oi = contracts_meta[sym]
            s = snap.get(sym) if snap else None
            q = s.latest_quote if s else None
            if q is None or q.ask_price is None:
                rows.append([
                    signal_id, now.isoformat(), minutes_since, stock_price, sym,
                    contract_type_str, strike, expiry, dte, oi, "", "", "", "", "", "", "", "", "", ""
                ])
                continue

            bid, ask = float(q.bid_price or 0), float(q.ask_price or 0)
            mid = round((bid + ask) / 2, 4) if ask > 0 else 0
            spread_pct = round((ask - bid) / mid, 4) if mid > 0 else ""
            g = s.greeks
            rows.append([
                signal_id, now.isoformat(), minutes_since, stock_price, sym,
                contract_type_str, strike, expiry, dte, oi,
                bid, ask, mid, spread_pct, s.implied_volatility or "", g.delta if g else "",
                g.gamma if g else "", g.theta if g else "", g.vega if g else "", g.rho if g else "",
            ])
    return rows


def make_signal_id(ticker: str, signal_ts: datetime) -> str:
    """Generate deterministic signal ID based on ticker and timestamp."""
    return f"{ticker}_{signal_ts.strftime('%Y%m%dT%H%M%S%f')}"


async def start_collection(signal_id: str, signal_ts: datetime, ticker: str, signal: dict,
                           headline: str, source: str,
                           get_stock_price: Callable[[str], Optional[float]]) -> None:
    """
    Launch contract grid data collection for a signal.
    Captures CALL options for bullish sentiment, PUT options for bearish sentiment.
    Writes 1 signal row to contract_grid_signals.csv, then appends initial quotes to contract_grid_snapshots.csv.
    """
    try:
        loop = asyncio.get_event_loop()
        stock_price = await loop.run_in_executor(None, get_stock_price, ticker)
        if not stock_price:
            return

        sentiment = (signal.get("sentiment") or "bullish").lower()
        contract_type = ContractType.PUT if sentiment == "bearish" else ContractType.CALL

        contracts_meta = await loop.run_in_executor(None, _fetch_contracts_meta, ticker, stock_price, contract_type)
        if not contracts_meta:
            return

        regime = await loop.run_in_executor(None, _regime_snapshot)
        last_expiry = max(date.fromisoformat(exp) for _, exp, _, _ in contracts_meta.values())
        magnitude = float(signal.get("magnitude", 0))
        confidence = float(signal.get("confidence", 0))
        news_call_ok, lotto_ok = _gate_eligibility(sentiment, magnitude, confidence)

        state = {
            "ticker": ticker,
            "headline": (headline or "")[:160],
            "source": source or "",
            "scorer_model": getattr(cfg, "SCORER_MODEL", "llama3.2"),
            "sentiment": sentiment,
            "contract_type": contract_type,
            "magnitude": magnitude,
            "confidence": confidence,
            "catalyst": signal.get("catalyst", ""),
            "stock_price_at_signal": stock_price,
            "spy_price": regime.get("spy_price", ""),
            "spy_sma": regime.get("spy_sma", ""),
            "regime_uptrend": regime.get("regime_uptrend", ""),
            "regime_mom_ok": regime.get("regime_mom_ok", ""),
            "regime_chop_ok": regime.get("regime_chop_ok", ""),
            "regime_dd_density": regime.get("regime_dd_density", ""),
            "vol_proxy_symbol": regime.get("vol_proxy_symbol", ""),
            "vol_proxy_price": regime.get("vol_proxy_price", ""),
            "vol_proxy_pctile": regime.get("vol_proxy_pctile", ""),
            "passes_news_call_gate": news_call_ok,
            "passes_lotto_gate": lotto_ok,
            "signal_ts": signal_ts,
            "last_poll": signal_ts,
            "last_expiry": last_expiry,
            "contracts_meta": contracts_meta,
            "get_stock_price": get_stock_price,
        }

        # 1. Write the single metadata row for this signal
        await loop.run_in_executor(None, _write_signal_row, signal_id, state)

        # 2. Write initial quotes/Greeks snapshot
        rows = await loop.run_in_executor(None, _snapshot_rows, signal_id, state)
        await loop.run_in_executor(None, _write_snapshot_rows, rows)

        _tracked[signal_id] = state
        type_label = "PUT" if contract_type == ContractType.PUT else "CALL"
        log.info("📊 grid collector: tracking %s %s (%d contracts, mag=%.2f conf=%.2f) — %s",
                 ticker, type_label, len(contracts_meta), state["magnitude"], state["confidence"], signal_id)
    except Exception as e:
        log.debug("grid collector: start_collection failed for %s: %s", ticker, e)


async def run_poll_loop(market_is_open: Callable[[], bool], poll_sleep_s: int = 15) -> None:
    """
    Decaying cadence poll loop restricted to the signal day.
    Re-snapshots tracked option contracts and prunes signals once the trading day ends or max session time elapses.
    """
    loop = asyncio.get_event_loop()
    while True:
        await asyncio.sleep(poll_sleep_s)
        if not _tracked:
            continue
        if not market_is_open():
            continue

        now = datetime.now(timezone.utc)
        due, finished = [], []
        same_day_limit = getattr(cfg, "GRID_TRACK_SAME_DAY_ONLY", True)
        max_minutes = getattr(cfg, "GRID_MAX_TRACK_MINUTES", 390)
        max_days = getattr(cfg, "GRID_MAX_TRACK_DAYS", 1)

        for sid, state in _tracked.items():
            elapsed_min = (now - state["signal_ts"]).total_seconds() / 60
            is_past_day = now.date() > state["signal_ts"].date()
            is_expired = now.date() > state["last_expiry"]
            is_past_limit = elapsed_min >= max_minutes if same_day_limit else (now - state["signal_ts"]).days > max_days

            if is_past_day or is_expired or is_past_limit:
                finished.append(sid)
                continue

            if (now - state["last_poll"]).total_seconds() >= _interval_for(elapsed_min):
                due.append(sid)

        for sid in due:
            state = _tracked[sid]
            try:
                rows = await loop.run_in_executor(None, _snapshot_rows, sid, state)
                await loop.run_in_executor(None, _write_snapshot_rows, rows)
            except Exception as e:
                log.debug("grid collector: poll failed for %s: %s", sid, e)
            state["last_poll"] = now

        for sid in finished:
            log.info("📊 grid collector: finished tracking %s (%s)", _tracked[sid]["ticker"], sid)
            del _tracked[sid]
