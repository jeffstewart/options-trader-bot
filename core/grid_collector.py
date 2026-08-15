"""
grid_collector.py — background market-data collection for the contract-picking research project
(jeff, 2026-08-XX).

The question this exists to answer: the model scores news well, but is the BOT PICKING THE RIGHT
CONTRACT out of what's actually available at that moment? No existing bot can answer this on its
own -- each only ever observes the forward price path of the ONE contract it happened to buy.

For every sufficiently-bullish signal (mag*confidence >= cfg.GRID_LOG_THRESHOLD -- deliberately
looser than the live TRADE gate; see config.py for the derivation), this module snapshots quotes +
greeks for a WIDE grid of strikes/expiries at signal time, then keeps re-snapshotting the SAME
contracts on a decaying cadence through each one's own expiry -- well past where any current exit
rule would have closed a real position, so the same dataset can validate exit logic (stop level,
EOD forced exit, hold duration) as well as entry contract selection.

Lives in v1 (jeff's call, 2026-08-XX), not v2: v1 scores every article that reaches it with the
free local Ollama scorer regardless of whether it will trade, while v2 deliberately filters BEFORE
the paid scorer to avoid spending money on signals that can't trade. v1's wider article coverage
makes it the better hook point for a "log more than what currently trades" research collector.

This is pure market-data reads via Alpaca's option snapshot endpoint (quote + IV + delta/gamma/
theta/vega/rho, batched up to 100 symbols/call) -- NO ORDERS ARE EVER PLACED. No dedicated account
or capital needed; it builds its own read-only clients from the same credentials bot.py already
has. A synthetic fill (entry_fill = mid*(1+spread/2), same model used elsewhere in this codebase)
can be applied analytically during analysis without ever having held the contract.

Deliberately has NO import of bot.py (bot.py imports this module, not the other way around) --
get_stock_price and market_is_open are passed in by the caller (bot.py already defines both) rather
than re-imported, so the two modules don't form a cycle. Wired in as a fire-and-forget background
task from bot.py's process_signal (BEFORE the live trade gate, so it sees ~88% of historically-
bullish signals rather than just what currently trades) plus one long-running poll loop task
alongside the bot's other supervised tasks. Must never block or slow the real trading path -- every
Alpaca call here runs in an executor, and every exception is caught and logged rather than raised.
"""
import asyncio
import csv
import logging
import os
from datetime import date, datetime, timedelta, timezone
from typing import Callable, Optional

from alpaca.data.historical.option import OptionHistoricalDataClient
from alpaca.data.requests import OptionSnapshotRequest
from alpaca.trading.client import TradingClient
from alpaca.trading.enums import ContractType
from alpaca.trading.requests import GetOptionContractsRequest

import config as cfg

log = logging.getLogger(__name__)

# Own clients (read-only market data + a trading client used only for get_option_contracts, which
# lives on the trading API despite being pure metadata) -- deliberately NOT bot.py's instances, to
# keep this module import-independent of bot.py. Harmless to have a second client pair; these are
# stateless HTTP clients, not connections that need to be singletons.
trading_client = TradingClient(cfg.ALPACA_KEY, cfg.ALPACA_SECRET, paper=True)
option_data_client = OptionHistoricalDataClient(cfg.ALPACA_KEY, cfg.ALPACA_SECRET)

_HEADER = [
    "signal_id", "signal_ts", "ticker", "headline", "source", "magnitude", "confidence",
    "snapshot_ts", "minutes_since_signal", "symbol", "strike", "expiry", "dte_at_signal",
    "oi_at_signal", "bid", "ask", "mid", "spread_pct", "iv", "delta", "gamma", "theta", "vega", "rho",
]

# signal_id -> {ticker, headline, source, magnitude, confidence, signal_ts, last_poll,
#               last_expiry (date), contracts_meta: {symbol: (strike, expiry_str, dte, oi)}}
_tracked: dict[str, dict] = {}


def _chunks(lst: list, n: int):
    for i in range(0, len(lst), n):
        yield lst[i:i + n]


def _interval_for(elapsed_minutes: float) -> int:
    for max_minutes, interval_s in cfg.GRID_POLL_SCHEDULE:
        if elapsed_minutes <= max_minutes:
            return interval_s
    return cfg.GRID_POLL_SCHEDULE[-1][1]


def _write_rows(rows: list[list]) -> None:
    if not rows:
        return
    try:
        new = not os.path.exists(cfg.GRID_SNAPSHOT_CSV)
        with open(cfg.GRID_SNAPSHOT_CSV, "a", newline="") as f:
            w = csv.writer(f)
            if new:
                w.writerow(_HEADER)
            w.writerows(rows)
    except Exception as e:      # telemetry must never break the caller
        log.debug("grid collector: write failed: %s", e)


def should_collect(signal: dict) -> bool:
    try:
        mag = float(signal.get("magnitude", 0))
        conf = float(signal.get("confidence", 0))
    except (TypeError, ValueError):
        return False
    return mag * conf >= cfg.GRID_LOG_THRESHOLD


def _fetch_contracts_meta(ticker: str, stock_price: float) -> dict:
    """One (paginated) get_option_contracts call -> {symbol: (strike, expiry_str, dte, oi)}. OI
    comes from THIS call only (the snapshot endpoint doesn't carry it) and is not re-fetched on
    later polls -- open interest moves slowly enough that a signal-time value is fine for this.

    PAGINATED, not a single call: get_option_contracts silently caps at ~100 results with no
    `limit` set, which for a name like MSFT truncates the whole grid to just the nearest few days
    of expiry and quietly drops the rest of the GRID_DTE_MIN-MAX window (caught live: a dry run's
    max expiry came back 4 days out instead of the requested 30). `limit=500` plus following
    `next_page_token` closes this; capped at 5 pages (2,500 contracts) as a runaway-loop backstop."""
    today = date.today()
    min_exp = today + timedelta(days=cfg.GRID_DTE_MIN)
    max_exp = today + timedelta(days=cfg.GRID_DTE_MAX)
    contracts = []
    page_token = None
    for _ in range(5):
        resp = trading_client.get_option_contracts(GetOptionContractsRequest(
            underlying_symbols=[ticker], type=ContractType.CALL,
            expiration_date_gte=min_exp, expiration_date_lte=max_exp,
            strike_price_gte=str(round(stock_price * cfg.GRID_STRIKE_LO_MULT, 2)),
            strike_price_lte=str(round(stock_price * cfg.GRID_STRIKE_HI_MULT, 2)),
            limit=500, page_token=page_token,
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
    """Batch-snapshot every tracked contract for one signal; returns rows ready for _write_rows.
    Missing/failed quotes still get a row (blank quote fields) so a contract's disappearance from
    the tape is itself visible in the data, not silently dropped."""
    contracts_meta = state["contracts_meta"]
    symbols = list(contracts_meta.keys())
    now = datetime.now(timezone.utc)
    minutes_since = round((now - state["signal_ts"]).total_seconds() / 60, 2)
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
                rows.append([signal_id, state["signal_ts"].isoformat(), state["ticker"],
                             state["headline"], state["source"], state["magnitude"],
                             state["confidence"], now.isoformat(), minutes_since, sym, strike,
                             expiry, dte, oi, "", "", "", "", "", "", "", "", "", ""])
                continue
            bid, ask = float(q.bid_price or 0), float(q.ask_price or 0)
            mid = round((bid + ask) / 2, 4) if ask > 0 else 0
            spread_pct = round((ask - bid) / mid, 4) if mid > 0 else ""
            g = s.greeks
            rows.append([
                signal_id, state["signal_ts"].isoformat(), state["ticker"], state["headline"],
                state["source"], state["magnitude"], state["confidence"], now.isoformat(),
                minutes_since, sym, strike, expiry, dte, oi, bid, ask, mid, spread_pct,
                s.implied_volatility or "", g.delta if g else "", g.gamma if g else "",
                g.theta if g else "", g.vega if g else "", g.rho if g else "",
            ])
    return rows


async def start_collection(ticker: str, signal: dict, headline: str, source: str,
                            get_stock_price: Callable[[str], Optional[float]]) -> None:
    """Fire-and-forget entry point -- called from process_signal as asyncio.create_task(...).
    Never raises: any failure here must not affect real trading. `get_stock_price` is injected by
    the caller (bot.py's own function) to avoid a circular import between this module and bot.py."""
    try:
        loop = asyncio.get_event_loop()
        stock_price = await loop.run_in_executor(None, get_stock_price, ticker)
        if not stock_price:
            return
        contracts_meta = await loop.run_in_executor(None, _fetch_contracts_meta, ticker, stock_price)
        if not contracts_meta:
            return

        signal_ts = datetime.now(timezone.utc)
        signal_id = f"{ticker}_{signal_ts.strftime('%Y%m%dT%H%M%S%f')}"
        last_expiry = max(date.fromisoformat(exp) for _, exp, _, _ in contracts_meta.values())
        state = {
            "ticker": ticker, "headline": (headline or "")[:160], "source": source or "",
            "magnitude": float(signal.get("magnitude", 0)), "confidence": float(signal.get("confidence", 0)),
            "signal_ts": signal_ts, "last_poll": signal_ts, "last_expiry": last_expiry,
            "contracts_meta": contracts_meta,
        }
        rows = await loop.run_in_executor(None, _snapshot_rows, signal_id, state)
        _write_rows(rows)
        _tracked[signal_id] = state
        log.info("📊 grid collector: tracking %s (%d contracts, mag=%.2f conf=%.2f) — %s",
                  ticker, len(contracts_meta), state["magnitude"], state["confidence"], signal_id)
    except Exception as e:
        log.debug("grid collector: start_collection failed for %s: %s", ticker, e)


async def run_poll_loop(market_is_open: Callable[[], bool]) -> None:
    """Long-running background task -- add alongside the bot's other supervised tasks. Wakes every
    15s, re-snapshots whatever's due per GRID_POLL_SCHEDULE, prunes signals past their last
    contract's expiry or GRID_MAX_TRACK_DAYS. Skips polling entirely while the market is closed --
    quotes don't move overnight/weekends, so those hours would just waste API calls; `next_poll`
    is simply left unadvanced so the schedule resumes exactly where it left off once the market
    reopens, rather than trying to reconstruct "elapsed market minutes" separately from wall clock.
    `market_is_open` is injected by the caller for the same reason get_stock_price is above."""
    loop = asyncio.get_event_loop()
    while True:
        await asyncio.sleep(15)
        if not _tracked:
            continue
        if not market_is_open():
            continue
        now = datetime.now(timezone.utc)
        due, finished = [], []
        for sid, state in _tracked.items():
            elapsed_days = (now - state["signal_ts"]).days
            if now.date() > state["last_expiry"] or elapsed_days > cfg.GRID_MAX_TRACK_DAYS:
                finished.append(sid)
                continue
            elapsed_min = (now - state["signal_ts"]).total_seconds() / 60
            if (now - state["last_poll"]).total_seconds() >= _interval_for(elapsed_min):
                due.append(sid)
        for sid in due:
            state = _tracked[sid]
            try:
                rows = await loop.run_in_executor(None, _snapshot_rows, sid, state)
                _write_rows(rows)
            except Exception as e:
                log.debug("grid collector: poll failed for %s: %s", sid, e)
            state["last_poll"] = now
        for sid in finished:
            log.info("📊 grid collector: finished tracking %s (%s)", _tracked[sid]["ticker"], sid)
            del _tracked[sid]
