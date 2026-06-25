"""
Backtest — replay 6 months of Alpaca historical news through the signal pipeline
and simulate trades against historical price data.

Usage:
    python backtest.py                        # full 6-month run
    python backtest.py --days 30              # shorter window for quick tests
    python backtest.py --no-cache             # ignore cached Ollama scores
    python backtest.py --workers 4            # parallel Ollama scoring

Outputs:
    backtest_trades.csv   — simulated trade log
    backtest_signals.csv  — every scored article (pass + skip)
    backtest_report.html  — equity curve, charts, summary stats
"""

import argparse
import asyncio
import csv
import json
import logging
import math
import os
import hashlib
import time
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import Optional
from concurrent.futures import ThreadPoolExecutor

from alpaca.data.historical import StockHistoricalDataClient
from alpaca.data.historical.news import NewsClient
from alpaca.data.historical.option import OptionHistoricalDataClient
from alpaca.data.requests import StockBarsRequest, NewsRequest
from alpaca.data.timeframe import TimeFrame
from alpaca.data.enums import Adjustment
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

# All parameters imported from config.py — edit that file to tune the backtest.
from config import (
    ALPACA_KEY, ALPACA_SECRET,
    OLLAMA_BASE_URL, OLLAMA_MODEL,
    MIN_MAGNITUDE, BASE_CONFIDENCE, CONFIDENCE_SLOPE,
    MAX_POSITION_USD, DAILY_LOSS_LIMIT,
    TRAILING_STOP_PCT, MAX_SPREAD_PCT, MIN_OPEN_INTEREST,
    MIN_DAYS_TO_EXPIRY, MAX_DAYS_TO_EXPIRY, TARGET_DELTA,
    MIN_STOCK_PRICE,
    MAX_HOLD_DAYS, MOVE_WINDOWS,
    RISK_FREE_RATE, REALIZED_VOL_WINDOW, NEWS_IV_MULTIPLIER,
    IV_CRUSH_HALFLIFE_DAYS, IV_FLOOR, IV_CAP,
    MAX_SANE_STOCK_PRICE, MAX_CONTRACT_BUDGET_MULT,
)
from pricing import (
    bs_call_price, strike_for_delta, realized_vol, iv_crush_path,
    bs_put_price, put_strike_for_delta,
)

CACHE_FILE = Path(os.environ.get("BACKTEST_CACHE_FILE", "backtest_score_cache.json"))

# ═══════════════════════════════════════════════════════════════════════════════
# CLIENTS
# ═══════════════════════════════════════════════════════════════════════════════

stock_client  = StockHistoricalDataClient(ALPACA_KEY, ALPACA_SECRET)
option_client = OptionHistoricalDataClient(ALPACA_KEY, ALPACA_SECRET)
news_client   = NewsClient(ALPACA_KEY, ALPACA_SECRET)   # dedicated news client
ollama        = OpenAI(base_url=OLLAMA_BASE_URL, api_key="ollama")

# ═══════════════════════════════════════════════════════════════════════════════
# SCORE CACHE  — avoid re-calling Ollama on reruns
# ═══════════════════════════════════════════════════════════════════════════════

_score_cache: dict = {}

def load_cache():
    global _score_cache
    if CACHE_FILE.exists():
        try:
            _score_cache = json.loads(CACHE_FILE.read_text())
            log.info("📦 Loaded %d cached scores from %s", len(_score_cache), CACHE_FILE)
        except Exception:
            _score_cache = {}

def save_cache():
    def default_serializer(obj):
        if hasattr(obj, "isoformat"):
            return obj.isoformat()
        raise TypeError(f"Object of type {type(obj).__name__} is not JSON serializable")
    CACHE_FILE.write_text(json.dumps(_score_cache, indent=2, default=default_serializer))

def cache_key(headline: str, body: str) -> str:
    return hashlib.md5(f"{headline}||{body[:500]}".encode()).hexdigest()

# ═══════════════════════════════════════════════════════════════════════════════
# OLLAMA SCORER  (identical prompt to bot.py)
# ═══════════════════════════════════════════════════════════════════════════════

SYSTEM_PROMPT = """You are a quantitative equity trading signal generator.
Respond with a JSON object ONLY — no markdown, no code fences, no explanation.

Schema:
{
  "tickers":    ["AAPL"],
  "sentiment":  "bullish",
  "confidence": 0.82,
  "magnitude":  0.75,
  "reasoning":  "one sentence"
}

sentiment: "bullish" | "bearish" | "neutral"
confidence: float 0.0–1.0 — how certain you are about the sentiment direction
magnitude:  float 0.0–1.0 — how likely this news is to meaningfully move the stock price

Magnitude scale:
  0.0–0.2  Noise or irrelevant (routine filings, minor analyst reiterations, fluff)
  0.2–0.4  Routine news (in-line earnings, small contract wins, minor upgrades)
  0.4–0.6  Meaningful catalyst (solid earnings beat, notable partnership, meaningful guidance raise)
  0.6–0.8  Strong catalyst (significant beat, major acquisition, landmark FDA approval for key drug)
  0.8–1.0  Transformative event (company-defining deal, paradigm-shifting approval, massive guidance revision)

Rules:
- Only include tickers you are highly confident about.
- General macro news with no specific company → empty tickers list.
- Most news is routine — be conservative with magnitude; reserve 0.7+ for genuinely exceptional events.
- Only flag bullish sentiment — we trade long calls and long crypto only.
- Return ONLY the JSON object."""

def score_article(headline: str, body: str, use_cache: bool = True) -> Optional[dict]:
    key = cache_key(headline, body)
    if use_cache and key in _score_cache:
        return _score_cache[key]
    try:
        resp = ollama.chat.completions.create(
            model=OLLAMA_MODEL,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user",   "content": f"Headline: {headline}\n\nBody: {body[:1200]}\n\nRespond with JSON only."},
            ],
            temperature=0.1,
        )
        raw = resp.choices[0].message.content.strip()
        if raw.startswith("```"):
            raw = raw.split("```")[1]
            if raw.startswith("json"):
                raw = raw[4:]
        result = json.loads(raw.strip())
        if use_cache:
            # Store only the model output — exclude _article to keep cache clean
            _score_cache[key] = {k: v for k, v in result.items() if k != "_article"}
        return result
    except Exception as e:
        log.debug("Score failed: %s", e)
        return None

# ═══════════════════════════════════════════════════════════════════════════════
# SIGNAL CHECKS  (identical logic to bot.py)
# ═══════════════════════════════════════════════════════════════════════════════

def signal_checks(signal: dict) -> tuple[bool, float, float]:
    magnitude  = float(signal.get("magnitude", 0))
    confidence = float(signal.get("confidence", 0))
    if magnitude < MIN_MAGNITUDE:
        return False, magnitude, 0
    required = BASE_CONFIDENCE + (1 - magnitude) * CONFIDENCE_SLOPE
    if confidence < required:
        return False, magnitude, required
    return True, magnitude, required

def scale_position_usd(base_usd: float, magnitude: float, confidence: float) -> float:
    return max(base_usd * 0.10, base_usd * magnitude * confidence)

# ═══════════════════════════════════════════════════════════════════════════════
# TICKER VALIDATION
# ═══════════════════════════════════════════════════════════════════════════════

_valid_ticker_cache: dict[str, bool] = {}

def is_valid_stock_ticker(ticker: str) -> bool:
    """
    Return True if the ticker is a valid US stock/ETF on Alpaca.
    Rejects full company names, crypto tokens, and unknown symbols.
    """
    # Quick format check: valid tickers are 1-5 uppercase letters
    if not ticker or not ticker.isalpha() or len(ticker) > 5:
        return False
    # Must be uppercase (model sometimes returns mixed case like "Dell", "Micron")
    if ticker != ticker.upper():
        return False
    if ticker in _valid_ticker_cache:
        return _valid_ticker_cache[ticker]
    try:
        from alpaca.trading.client import TradingClient
        from alpaca.trading.enums import AssetClass
        tc = TradingClient(ALPACA_KEY, ALPACA_SECRET, paper=True)
        asset = tc.get_asset(ticker)
        valid = (asset is not None
                 and asset.tradable
                 and str(asset.asset_class) in ("us_equity", "AssetClass.US_EQUITY"))
        _valid_ticker_cache[ticker] = valid
        return valid
    except Exception:
        _valid_ticker_cache[ticker] = False
        return False

# ═══════════════════════════════════════════════════════════════════════════════
# HISTORICAL DATA HELPERS
# ═══════════════════════════════════════════════════════════════════════════════

# Opt-in real-data source: set USE_YAHOO_BARS=1 to source historical bars from
# Yahoo (real-world-correct) instead of this env's unreliable Alpaca store. Used
# for cross-regime (e.g. 2022) backtests; leave off for the live-data bull window.
_USE_YAHOO = os.environ.get("USE_YAHOO_BARS", "") == "1"

def get_stock_bars(ticker: str, start: datetime, end: datetime) -> list[dict]:
    if _USE_YAHOO:
        try:
            from yahoo_data import get_yahoo_bars
            bars = get_yahoo_bars(ticker, start, end)
            if bars:
                return bars
            # fall through to Alpaca if Yahoo returned nothing
        except Exception as e:
            log.debug("Yahoo bars failed for %s: %s — falling back to Alpaca", ticker, e)
    # resp.data[ticker] is a list of Bar objects — use attribute access, not .get()
    try:
        resp = stock_client.get_stock_bars(StockBarsRequest(
            symbol_or_symbols=ticker,
            timeframe=TimeFrame.Day,
            start=start,
            end=end,
            feed="iex",
            adjustment=Adjustment.ALL,   # split + dividend adjusted → continuous prices
        ))
        bars_raw = (resp.data or {}).get(ticker, [])
        return [
            {"t": b.timestamp, "o": float(b.open), "h": float(b.high),
             "l": float(b.low),  "c": float(b.close), "v": int(b.volume)}
            for b in bars_raw
        ]
    except Exception as e:
        log.debug("Bars failed for %s: %s", ticker, e)
        return []

def get_price_at(ticker: str, dt: datetime) -> Optional[float]:
    """
    Get the entry price for a trade on dt.
    - If dt is a trading day: use that day's close.
    - If dt is a weekend/holiday: use the NEXT trading day's open
      (simulates entering on the next available market open).
    Looks back up to 14 days to find a prior close, and forward
    up to 5 trading days to find the next open.
    """
    # Look back up to 14 calendar days for a valid prior close
    bars = get_stock_bars(ticker, dt - timedelta(days=14), dt + timedelta(days=8))
    if not bars:
        return None

    # Make dt timezone-aware for comparison if needed
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)

    # Check if there's a bar on the same calendar day
    same_day = [b for b in bars if b["t"].date() == dt.date()]

    if same_day:
        return same_day[-1]["c"]

    # No bar on signal day (weekend/holiday) — use next trading day's open
    future_bars = [b for b in bars if b["t"].date() > dt.date()]
    if future_bars:
        return future_bars[0]["o"]  # open of next trading day

    # Fall back to most recent prior close
    prior_bars = [b for b in bars if b["t"].date() < dt.date()]
    return prior_bars[-1]["c"] if prior_bars else None

def get_price_n_days_after(ticker: str, entry_dt: datetime, n: int) -> Optional[float]:
    """Get closing price n trading days after entry_dt."""
    bars = get_stock_bars(ticker, entry_dt, entry_dt + timedelta(days=n * 2 + 10))
    trading_bars = [b for b in bars if b["t"] > entry_dt]
    if len(trading_bars) >= n:
        return trading_bars[n - 1]["c"]
    elif trading_bars:
        return trading_bars[-1]["c"]
    return None

def base_iv_for(ticker: str, entry_dt: datetime) -> float:
    """
    Base implied vol = trailing realized volatility of the underlying, computed
    from the dense daily stock bars in the REALIZED_VOL_WINDOW days before entry.
    Clamped to [IV_FLOOR, IV_CAP]. Falls back to IV_FLOOR if history is missing.
    """
    lookback_start = entry_dt - timedelta(days=REALIZED_VOL_WINDOW * 2 + 10)
    bars = get_stock_bars(ticker, lookback_start, entry_dt)
    closes = [b["c"] for b in bars if b["t"] <= entry_dt][-(REALIZED_VOL_WINDOW + 1):]
    rv = realized_vol(closes)
    if rv is None:
        return IV_FLOOR
    return min(max(rv, IV_FLOOR), IV_CAP)


# Target days-to-expiry centered in the configured window (~37 DTE).
DTE_TARGET = round((MIN_DAYS_TO_EXPIRY + MAX_DAYS_TO_EXPIRY) / 2)

# ── Exit-rule parameters (for backtesting alternative close logic) ────────────
# simulate_option_pnl(..., exit_rule=...) selects how a position is closed:
#   "trail_premium"   baseline — 15% trailing stop on the option premium
#   "profit_target"   full exit at +target_pct, else premium trailing stop
#   "dte_stop"        exit at dte_floor days-to-expiry, else premium trailing stop
#   "underlying_trail" trail on the STOCK price (less IV/theta whipsaw) + wide
#                      premium catastrophe backstop
#   "tiered_trail"    trailing % tightens as profit grows
EXIT_PARAMS = {
    "target_pct":            0.75,   # +75% → take profit (profit_target)
    "dte_floor":             7,      # close when ≤7 days to expiry (dte_stop)
    "underlying_trail_pct":  0.08,   # stock −8% from peak (underlying_trail)
    "catastrophe_trail":     0.40,   # wide premium backstop for underlying_trail
    "tiers": [(0.50, 0.20), (1.00, 0.15), (float("inf"), 0.10)],  # (profit<thr → trail)
    # "tiered_profit" rule: a profit-tiered TRAILING stop (tightens as gains grow
    # but never hard-exits a still-climbing position → lets winners run).
    "hard_target":           None,   # optional hard cap (e.g. 3.0 = exit at +300%); None = no cap
    # Flat time-stop (composes with ANY rule): cut dead money that never worked.
    "flat_stop_days":        0,      # 0 = off; else cut if held this many days…
    "flat_stop_progress":    0.20,   # …and peak gain never reached this (+20%)
}

# ── Bid/ask spread + slippage model (for friction-aware backtests) ────────────
# simulate_option_pnl(..., spread_mult=M): M=0 → frictionless (default). M>0 →
# buy at ask, sell at bid. Spread (% of mid) is tiered by premium (cheaper
# options trade wider) and tightened for liquid underlyings; scaled by M for
# sensitivity (0.5 optimistic / 1.0 base / 1.5 pessimistic).
LIQUID_UNDERLYINGS = {"SPY", "QQQ", "IWM", "GLD", "TLT", "DIA",
                      "AAPL", "MSFT", "NVDA", "AMD", "TSLA", "META", "AMZN",
                      "GOOGL", "GOOG", "AVGO", "NFLX", "INTC", "ORCL", "QCOM", "CRM"}
def _spread_pct(premium: float, ticker: str, mult: float) -> float:
    """Full bid/ask spread as a fraction of mid, before the mult."""
    if premium < 0.25:   base = 0.30
    elif premium < 1.0:  base = 0.18
    elif premium < 3.0:  base = 0.10
    elif premium < 10.0: base = 0.06
    else:                base = 0.04
    if ticker in LIQUID_UNDERLYINGS:
        base *= 0.35      # liquid names: much tighter
    base += 0.01          # ~1% slippage/impact baseline
    return base * mult


def simulate_option_pnl(
    ticker: str,
    entry_dt: datetime,
    entry_stock_price: float,
    position_usd: float,
    signal: dict,
    option_type: str = "call",
    exit_rule: str = "trail_premium",
    spread_mult: float = 0.0,
    min_premium: float = 0.0,        # skip options cheaper than this (wide % spreads)
    max_entry_spread: float = 1.0,   # skip if base bid/ask spread exceeds this fraction
    scale_out: Optional[tuple] = None,  # (profit_mult, fraction): sell `fraction` of the
                                        # position the first time premium ≥ profit_mult×entry,
                                        # let the rest ride the normal exit. None = off (default).
) -> Optional[dict]:
    """
    Simulate a long option trade entered at entry_dt, priced with Black-Scholes
    off the dense daily stock bars. option_type="call" (bullish) or "put"
    (bearish — gains when the stock falls).

    Captures the three forces on a long option:
      • delta/gamma — stock price path (bar closes)
      • theta       — time-to-expiry shrinks each day (T)
      • vega / IV crush — implied vol starts elevated (news) and decays toward
                          the underlying's realized vol (see iv_crush_path)
    Returns a trade result dict.
    """
    magnitude  = float(signal.get("magnitude", 0))
    confidence = float(signal.get("confidence", 0))
    r          = RISK_FREE_RATE
    # Data-sanity guard: reject absurd entry prices (bad IEX bars, e.g. EDBL at
    # $417,000) before they generate a fake multi-million-dollar trade.
    if entry_stock_price > MAX_SANE_STOCK_PRICE:
        return None
    is_put     = option_type == "put"
    price_fn   = bs_put_price if is_put else bs_call_price
    strike_fn  = put_strike_for_delta if is_put else strike_for_delta

    # ── Volatility regime + contract selection at entry ──────────────────────
    base_iv   = base_iv_for(ticker, entry_dt)
    entry_iv  = base_iv * NEWS_IV_MULTIPLIER
    T_entry   = DTE_TARGET / 365.0
    # Strike at the target delta (priced with the elevated entry IV, matching
    # how the live bot would pick a contract right after the news hits).
    strike    = round(strike_fn(entry_stock_price, T_entry, entry_iv,
                                TARGET_DELTA, r), 2)

    entry_premium = price_fn(entry_stock_price, strike, T_entry, entry_iv, r)
    if entry_premium < 0.05:
        return None  # degenerate / near-zero premium — skip
    # Over-budget guard: if a single contract costs far more than the position
    # budget, the qty=max(1,…) below would force a wildly oversized position
    # (this is how EDBL's $444K/contract slipped through). Skip instead.
    if entry_premium * 100 > position_usd * MAX_CONTRACT_BUDGET_MULT:
        return None
    # Tradeability filters: skip cheap/wide-spread options that can't be filled well.
    if entry_premium < min_premium:
        return None
    if _spread_pct(entry_premium, ticker, 1.0) > max_entry_spread:
        return None

    # Buy at the ask: entry fill = mid + half the bid/ask spread.
    entry_fill = entry_premium * (1 + _spread_pct(entry_premium, ticker, spread_mult) / 2) \
        if spread_mult > 0 else entry_premium
    qty        = max(1, int(position_usd / (entry_fill * 100)))
    cost_basis = entry_fill * 100 * qty

    peak_value   = entry_premium
    exit_dt      = None
    exit_reason  = "max_hold"
    exit_premium = entry_premium

    # Fetch only the intended hold window (~MAX_HOLD_DAYS trading days ≈ 1.5×
    # calendar days, plus buffer). Fetching too wide a window can trip a data
    # hole in the free IEX feed that returns a later, discontinuous price
    # segment (e.g. a post-split regime) — which previously produced fake
    # 25-60× option "gains".
    window_cal_days = int(MAX_HOLD_DAYS * 1.5) + 7
    window_end = entry_dt + timedelta(days=window_cal_days)
    bars = get_stock_bars(ticker, entry_dt, window_end)
    trading_bars = [b for b in bars if entry_dt < b["t"] <= window_end]

    # ── Data-quality guards (skip the trade rather than price on bad data) ──
    if not trading_bars:
        return None
    # 1) Large entry→first-bar gap = missing data for the actual hold period.
    if (trading_bars[0]["t"] - entry_dt).days > 7:
        return None
    # 2) Implausible single-day jump = data discontinuity / unadjusted action.
    prev_c = entry_stock_price
    for b in trading_bars:
        if prev_c > 0 and abs(b["c"] / prev_c - 1) > 0.50:
            return None
        prev_c = b["c"]

    peak_stock = entry_stock_price
    ep = EXIT_PARAMS
    scaled_value  = 0.0          # $ booked from scale-out partial sale(s)
    remaining_qty = qty          # contracts still held after any scale-out
    scaled        = False
    for i, bar in enumerate(trading_bars[:MAX_HOLD_DAYS], start=1):
        days_held_cal = max((bar["t"] - entry_dt).days, 0)
        T   = max((DTE_TARGET - days_held_cal) / 365.0, 1.0 / 365.0)
        # i = trading days since entry; IV_CRUSH_HALFLIFE_DAYS is in trading days
        iv  = iv_crush_path(base_iv, i, NEWS_IV_MULTIPLIER, IV_CRUSH_HALFLIFE_DAYS)
        premium = price_fn(bar["c"], strike, T, iv, r)
        premium = max(0.01, premium)

        if premium > peak_value:
            peak_value = premium
        if bar["c"] > peak_stock:
            peak_stock = bar["c"]

        # Directional move on the option: puts gain when the stock falls, so use
        # the underlying drawdown in the trade's favor for the underlying trail.
        ret          = premium / entry_premium - 1.0
        peak_ret     = peak_value / entry_premium - 1.0   # best gain reached so far
        dte_remaining = DTE_TARGET - days_held_cal
        prem_stop    = peak_value * (1 - TRAILING_STOP_PCT)

        # ── Scale-out: the first time premium reaches profit_mult×entry, sell `fraction`
        #    of the position and let the rest ride the normal exit below. ──
        if scale_out and not scaled and premium >= scale_out[0] * entry_premium:
            frac = scale_out[1]
            sell_fill = premium * (1 - _spread_pct(premium, ticker, spread_mult) / 2) \
                if spread_mult > 0 else premium
            scaled_value += max(0.0, sell_fill) * 100 * (qty * frac)
            remaining_qty = qty * (1 - frac)
            scaled = True

        hit, reason = False, "trailing_stop"
        # ── Flat time-stop (composes with every rule): cut dead money that
        #    never got meaningfully into profit after flat_stop_days. ──
        if (ep.get("flat_stop_days", 0) and days_held_cal >= ep["flat_stop_days"]
                and peak_ret < ep.get("flat_stop_progress", 0.20)):
            hit, reason = True, "flat_stop"
        elif exit_rule == "trail_premium":
            if premium <= prem_stop:
                hit, reason = True, "trailing_stop"
        elif exit_rule == "profit_target":
            if ret >= ep["target_pct"]:
                hit, reason = True, "profit_target"
            elif premium <= prem_stop:
                hit, reason = True, "trailing_stop"
        elif exit_rule == "tiered_profit":
            # Profit-tiered trailing stop — tighten the trail as the gain grows
            # so winners keep running; optional hard cap as a ceiling.
            ht = ep.get("hard_target")
            if ht is not None and ret >= ht:
                hit, reason = True, "hard_target"
            else:
                tr = next(t for thresh, t in ep["tiers"] if ret < thresh)
                if premium <= peak_value * (1 - tr):
                    hit, reason = True, "tiered_profit"
        elif exit_rule == "dte_stop":
            if dte_remaining <= ep["dte_floor"]:
                hit, reason = True, "dte_stop"
            elif premium <= prem_stop:
                hit, reason = True, "trailing_stop"
        elif exit_rule == "underlying_trail":
            # Trail on the underlying (call: stock down from peak; put: stock up
            # from trough) — cleaner directional signal, less IV/theta whipsaw.
            if not is_put and bar["c"] <= peak_stock * (1 - ep["underlying_trail_pct"]):
                hit, reason = True, "underlying_trail"
            elif premium <= peak_value * (1 - ep["catastrophe_trail"]):
                hit, reason = True, "prem_catastrophe"
        elif exit_rule == "tiered_trail":
            tr = next(t for thresh, t in ep["tiers"] if ret < thresh)
            if premium <= peak_value * (1 - tr):
                hit, reason = True, "tiered_trail"
        elif exit_rule == "trail_breakeven":
            # Normal premium trail (TRAILING_STOP_PCT), BUT once the peak gain reaches be_arm the stop
            # floor is lifted to breakeven+be_lock, so a winner that fades can't round-trip deep into the
            # red. Trades give-back upside (a faded-then-recovered winner gets cut at ~breakeven) for a
            # higher win rate — the +15%→-25% case the live 40% flat trail allows.
            stop = prem_stop
            if peak_ret >= ep.get("be_arm", 0.15):
                stop = max(stop, entry_premium * (1.0 + ep.get("be_lock", 0.0)))
            if premium <= stop:
                hit, reason = True, "trail_breakeven"
        else:  # unknown rule → baseline
            if premium <= prem_stop:
                hit, reason = True, "trailing_stop"

        if hit:
            exit_premium = premium
            exit_dt      = bar["t"]
            exit_reason  = reason
            break
    else:
        # Held to MAX_HOLD_DAYS — value the position at the last available bar.
        if trading_bars:
            last_bar = trading_bars[min(MAX_HOLD_DAYS, len(trading_bars)) - 1]
            n = min(MAX_HOLD_DAYS, len(trading_bars))
            days_held_cal = max((last_bar["t"] - entry_dt).days, 0)
            T  = max((DTE_TARGET - days_held_cal) / 365.0, 1.0 / 365.0)
            iv = iv_crush_path(base_iv, n, NEWS_IV_MULTIPLIER, IV_CRUSH_HALFLIFE_DAYS)
            exit_premium = max(0.01, price_fn(last_bar["c"], strike, T, iv, r))
            exit_dt = last_bar["t"]

    # Sell at the bid: exit fill = mid − half the bid/ask spread.
    exit_fill  = exit_premium * (1 - _spread_pct(exit_premium, ticker, spread_mult) / 2) \
        if spread_mult > 0 else exit_premium
    # remaining_qty == qty and scaled_value == 0 when scale_out is off → identical to before.
    exit_value = max(0.0, exit_fill) * 100 * remaining_qty + scaled_value
    pnl_usd    = exit_value - cost_basis
    pnl_pct    = (pnl_usd / cost_basis) * 100 if cost_basis > 0 else 0

    return {
        "ticker":            ticker,
        "entry_dt":          entry_dt.isoformat(),
        "exit_dt":           exit_dt.isoformat() if exit_dt else "",
        "exit_reason":       exit_reason,
        "option_type":       option_type,
        "entry_stock_price": round(entry_stock_price, 2),
        "strike":            strike,
        "base_iv":           round(base_iv, 4),
        "entry_iv":          round(entry_iv, 4),
        "entry_premium":     round(entry_premium, 4),
        "exit_premium":      round(exit_premium, 4),
        "qty":               qty,
        "cost_basis":        round(cost_basis, 2),
        "exit_value":        round(exit_value, 2),
        "pnl_usd":           round(pnl_usd, 2),
        "pnl_pct":           round(pnl_pct, 2),
        "peak_ret":          round(peak_value / entry_premium - 1.0, 4),   # best gain reached (for exit-policy analysis)
        "magnitude":         round(magnitude, 3),
        "confidence":        round(confidence, 3),
        "position_usd":      round(position_usd, 2),
        "reasoning":         signal.get("reasoning", ""),
    }

# ═══════════════════════════════════════════════════════════════════════════════
# HISTORICAL NEWS FETCH
# ═══════════════════════════════════════════════════════════════════════════════

def _parse_news_item(item) -> dict:
    """Normalise a News pydantic object into a plain dict."""
    if hasattr(item, "model_dump"):
        d = item.model_dump()
        return {
            "id":         str(d.get("id", "")),
            "headline":   d.get("headline", "") or "",
            "summary":    d.get("summary", "") or "",
            "created_at": d.get("created_at", None),
            "symbols":    d.get("symbols", []) or [],
            "source":     "alpaca",
        }
    return {
        "id":         str(getattr(item, "id", "")),
        "headline":   getattr(item, "headline", "") or "",
        "summary":    getattr(item, "summary", "") or "",
        "created_at": getattr(item, "created_at", None),
        "symbols":    getattr(item, "symbols", []) or [],
        "source":     "alpaca",
    }

def fetch_alpaca_news_day(day: datetime) -> list[dict]:
    """
    Fetch up to 50 articles for a single calendar day from Alpaca.
    Free tier is capped at 50 per request — fetching per day maximises coverage.
    """
    start = day.replace(hour=0, minute=0, second=0, microsecond=0)
    end   = start + timedelta(hours=23, minutes=59, seconds=59)
    try:
        req  = NewsRequest(
            start=start, end=end, limit=50, sort="desc",
            include_content=False, exclude_contentless=True,
        )
        resp  = news_client.get_news(req)
        items = (resp.data or {}).get("news", [])
        return [_parse_news_item(i) for i in items]
    except Exception as e:
        log.warning("Alpaca news fetch failed for %s: %s", day.date(), e)
        return []

def fetch_sec_edgar_8k(start: datetime, end: datetime) -> list[dict]:
    """
    Fetch historical 8-K filings from SEC EDGAR full-text search API.
    Free, no API key required. Returns up to 200 filings per date range.
    Each filing becomes a scoreable article with the company name + filing items as context.
    """
    import urllib.request, urllib.parse
    articles = []
    url = "https://efts.sec.gov/LATEST/search-index?q=%228-K%22&dateRange=custom&startdt={}&enddt={}&forms=8-K&hits.hits._source=period_of_report,file_date,entity_name,file_num,period_of_report,items&hits.hits.total.value=true&hits.hits.highlight=true"
    # Use the public EDGAR full-text search endpoint
    edgar_url = (
        "https://efts.sec.gov/LATEST/search-index?"
        + urllib.parse.urlencode({
            "q": '"material" OR "agreement" OR "acquisition" OR "approval" OR "results"',
            "dateRange": "custom",
            "startdt": start.strftime("%Y-%m-%d"),
            "enddt":   end.strftime("%Y-%m-%d"),
            "forms":   "8-K",
            "hits.hits.total.value": "true",
        })
        + "&hits.hits._source=period_of_report,file_date,entity_name,items,period_of_report"
    )
    try:
        req = urllib.request.Request(edgar_url, headers={"User-Agent": "news-trader-backtest admin@example.com"})
        with urllib.request.urlopen(req, timeout=15) as resp:
            data = json.loads(resp.read())
        hits = data.get("hits", {}).get("hits", [])
        for hit in hits:
            src      = hit.get("_source", {})
            entity   = src.get("entity_name", "Unknown")
            items    = src.get("items", "")
            filed    = src.get("file_date", "")
            if not filed:
                continue
            try:
                created_at = datetime.strptime(filed, "%Y-%m-%d").replace(tzinfo=timezone.utc)
            except Exception:
                continue
            # Build a synthetic headline + summary from the filing metadata
            headline = f"{entity} filed SEC 8-K"
            summary  = f"{entity} submitted an 8-K filing on {filed}."
            if items:
                summary += f" Items: {items}."
            articles.append({
                "id":         hit.get("_id", f"sec-{filed}-{entity}"),
                "headline":   headline,
                "summary":    summary,
                "created_at": created_at,
                "symbols":    [],   # no ticker in basic EDGAR metadata
                "source":     "sec_edgar",
            })
        log.info("📋 SEC EDGAR: fetched %d 8-K filings (%s → %s)",
                 len(articles), start.strftime("%Y-%m-%d"), end.strftime("%Y-%m-%d"))
    except Exception as e:
        log.warning("SEC EDGAR fetch failed: %s", e)
    return articles

# ═══════════════════════════════════════════════════════════════════════════════
# BACKTEST RUNNER
# ═══════════════════════════════════════════════════════════════════════════════

def run_backtest(days: int = 180, use_cache: bool = True, workers: int = 2, end_date: str = None):
    load_cache()

    if end_date:
        end_dt = datetime.strptime(end_date, "%Y-%m-%d").replace(tzinfo=timezone.utc)
    else:
        end_dt = datetime.now(timezone.utc) - timedelta(days=5)   # allow time for price data to settle
    start_dt = end_dt - timedelta(days=days)

    log.info("═" * 60)
    log.info("BACKTEST  %s → %s  (%d days)",
             start_dt.strftime("%Y-%m-%d"), end_dt.strftime("%Y-%m-%d"), days)
    log.info("Model: %s  |  MIN_MAGNITUDE: %.2f  |  BASE_CONF: %.2f",
             OLLAMA_MODEL, MIN_MAGNITUDE, BASE_CONFIDENCE)
    log.info("═" * 60)

    # ── 1. Fetch news — Alpaca (50/day) + SEC EDGAR 8-Ks ────────
    seen_ids: set[str] = set()
    all_articles: list[dict] = []

    def add_articles(new_items: list[dict]):
        for a in new_items:
            if a["id"] not in seen_ids and a["headline"].strip():
                seen_ids.add(a["id"])
                all_articles.append(a)

    # Alpaca: fetch 50 articles per calendar day
    log.info("📰 Fetching Alpaca news (50/day × %d days)…", days)
    day = start_dt
    while day < end_dt:
        add_articles(fetch_alpaca_news_day(day))
        day += timedelta(days=1)
        time.sleep(0.15)   # gentle rate limit — ~7 req/sec

    alpaca_count = len(all_articles)
    log.info("📰 Alpaca: %d articles", alpaca_count)

    # SEC EDGAR: fetch 8-K filings across the full window in 30-day chunks
    log.info("📋 Fetching SEC EDGAR 8-K filings…")
    chunk_start = start_dt
    while chunk_start < end_dt:
        chunk_end = min(chunk_start + timedelta(days=30), end_dt)
        add_articles(fetch_sec_edgar_8k(chunk_start, chunk_end))
        chunk_start = chunk_end + timedelta(days=1)
        time.sleep(0.5)

    log.info("📰 Total unique articles to score: %d  (alpaca=%d  sec=%d)",
             len(all_articles), alpaca_count, len(all_articles) - alpaca_count)

    # ── 2. Score articles (parallel) ─────────────────────────────
    log.info("🤖 Scoring with %s (workers=%d, cache=%s)…", OLLAMA_MODEL, workers, use_cache)
    scored = []

    def score_one(article):
        sig = score_article(article["headline"], article["summary"], use_cache=use_cache)
        if sig:
            sig = dict(sig)  # copy to avoid mutating the shared cache dict
            sig["_article"] = article
        return sig

    with ThreadPoolExecutor(max_workers=workers) as pool:
        results = list(pool.map(score_one, all_articles))

    # Save cache after scoring batch
    save_cache()
    log.info("💾 Cache saved (%d entries)", len(_score_cache))

    # ── 3. Apply signal filters ───────────────────────────────────
    signal_rows  = []   # all scored articles for signal quality CSV
    passing      = []   # signals that passed all filters

    seen_tickers_by_day: dict[str, set] = {}   # date → set of tickers (cooldown proxy)

    for sig in results:
        if sig is None:
            continue
        article = sig.pop("_article")
        magnitude  = float(sig.get("magnitude", 0))
        confidence = float(sig.get("confidence", 0))
        sentiment  = sig.get("sentiment", "")
        tickers    = sig.get("tickers", [])
        created_at = article.get("created_at")
        if not created_at:
            continue
        if isinstance(created_at, str):
            created_at = datetime.fromisoformat(created_at.replace("Z", "+00:00"))

        passes, _, req_conf = signal_checks(sig)
        trade_day = created_at.date().isoformat()

        signal_rows.append({
            "created_at":  created_at.isoformat(),
            "headline":    article["headline"][:120],
            "tickers":     ",".join(tickers),
            "sentiment":   sentiment,
            "magnitude":   round(magnitude, 3),
            "confidence":  round(confidence, 3),
            "passes_gate": passes and sentiment == "bullish",
            "reasoning":   sig.get("reasoning", ""),
        })

        if not passes or sentiment != "bullish" or not tickers:
            continue

        # Cooldown: one trade per ticker per calendar day in backtest
        day_set = seen_tickers_by_day.setdefault(trade_day, set())
        for ticker in tickers[:2]:
            if ticker in day_set:
                continue
            if ticker in ("BTC", "ETH"):
                continue  # skip crypto in backtest (no historical option analog)
            if not is_valid_stock_ticker(ticker):
                log.info("  ⚠️  Skipping invalid/unrecognized ticker: %r", ticker)
                continue
            day_set.add(ticker)
            passing.append({
                "signal":     sig,
                "ticker":     ticker,
                "created_at": created_at,
                "headline":   article["headline"],
            })

    log.info("✅ Passing signals: %d / %d scored", len(passing), len(signal_rows))
    for p in passing:
        log.info("  → %s  ticker=%s  date=%s  headline=%s",
                 p["signal"].get("sentiment"), p["ticker"],
                 p["created_at"].date(), p["headline"][:80])

    # ── 4. Simulate trades ────────────────────────────────────────
    trade_rows      = []
    daily_loss      = {}   # date → cumulative loss
    halted_days     = set()
    total_pnl       = 0.0
    equity_curve    = []   # (date, cumulative_pnl)
    running_pnl     = 0.0

    for p in passing:
        ticker     = p["ticker"]
        created_at = p["created_at"]
        signal     = p["signal"]
        trade_day  = created_at.date().isoformat()

        if trade_day in halted_days:
            log.debug("Skipping %s on %s — halted day", ticker, trade_day)
            continue

        # Get stock price at time of signal
        entry_price = get_price_at(ticker, created_at)
        if not entry_price:
            log.info("  ⚠️  No price data for %s at %s — skipping", ticker, created_at.date())
            continue
        if entry_price < MIN_STOCK_PRICE:
            log.info("  ⚠️  %s @ $%.2f below MIN_STOCK_PRICE $%.2f — skipping", ticker, entry_price, MIN_STOCK_PRICE)
            continue
        log.info("  📈 Simulating trade: %s @ $%.2f on %s", ticker, entry_price, created_at.date())

        magnitude  = float(signal.get("magnitude", 0))
        confidence = float(signal.get("confidence", 0))
        position_usd = scale_position_usd(MAX_POSITION_USD, magnitude, confidence)

        trade = simulate_option_pnl(ticker, created_at, entry_price, position_usd, signal)
        if not trade:
            continue

        # Track daily loss for circuit breaker
        day_loss = daily_loss.get(trade_day, 0.0)
        if trade["pnl_usd"] < 0:
            day_loss += abs(trade["pnl_usd"])
            daily_loss[trade_day] = day_loss
            if day_loss >= DAILY_LOSS_LIMIT:
                halted_days.add(trade_day)
                log.info("🛑 Daily loss limit hit on %s ($%.0f)", trade_day, day_loss)

        running_pnl += trade["pnl_usd"]
        total_pnl   += trade["pnl_usd"]
        trade["cumulative_pnl"] = round(running_pnl, 2)
        trade_rows.append(trade)

        equity_curve.append({"date": trade_day, "cumulative_pnl": round(running_pnl, 2)})

    # ── 5. Signal quality: measure actual stock moves ─────────────
    log.info("📊 Measuring actual stock moves for signal quality…")
    for row in signal_rows:
        tickers = [t for t in row["tickers"].split(",") if t]
        if not tickers:
            continue
        ticker = tickers[0]
        try:
            created_at = datetime.fromisoformat(row["created_at"])
        except Exception:
            continue
        entry_price = get_price_at(ticker, created_at)
        if not entry_price:
            continue
        for n in MOVE_WINDOWS:
            price_n = get_price_n_days_after(ticker, created_at, n)
            if price_n:
                row[f"move_{n}d_pct"] = round((price_n - entry_price) / entry_price * 100, 3)
            else:
                row[f"move_{n}d_pct"] = None

    # ── 6. Compute summary stats ──────────────────────────────────
    wins    = [t for t in trade_rows if t["pnl_usd"] > 0]
    losses  = [t for t in trade_rows if t["pnl_usd"] <= 0]
    n       = len(trade_rows)

    avg_win  = sum(t["pnl_usd"] for t in wins)  / len(wins)  if wins   else 0
    avg_loss = sum(t["pnl_usd"] for t in losses) / len(losses) if losses else 0
    win_rate = len(wins) / n * 100 if n else 0

    # Max drawdown from equity curve
    peak_eq, max_dd = 0.0, 0.0
    for pt in equity_curve:
        eq = pt["cumulative_pnl"]
        if eq > peak_eq:
            peak_eq = eq
        dd = peak_eq - eq
        if dd > max_dd:
            max_dd = dd

    # Sharpe (simplified daily returns)
    if len(trade_rows) > 1:
        returns = [t["pnl_pct"] for t in trade_rows]
        mean_r  = sum(returns) / len(returns)
        std_r   = math.sqrt(sum((r - mean_r) ** 2 for r in returns) / len(returns)) or 1
        sharpe  = (mean_r / std_r) * math.sqrt(252)
    else:
        sharpe = 0.0

    stats = {
        "period_days":      days,
        "articles_scored":  len(signal_rows),
        "signals_passing":  len(passing),
        "trades_simulated": n,
        "total_pnl":        round(total_pnl, 2),
        "win_rate_pct":     round(win_rate, 1),
        "avg_win_usd":      round(avg_win, 2),
        "avg_loss_usd":     round(avg_loss, 2),
        "profit_factor":    round(abs(avg_win / avg_loss), 2) if avg_loss else 0,
        "max_drawdown_usd": round(max_dd, 2),
        "sharpe_ratio":     round(sharpe, 2),
        "halted_days":      len(halted_days),
    }

    log.info("─" * 60)
    log.info("RESULTS")
    log.info("  Articles scored   : %d", stats["articles_scored"])
    log.info("  Passing signals   : %d", stats["signals_passing"])
    log.info("  Trades simulated  : %d", stats["trades_simulated"])
    log.info("  Total P&L         : $%.2f", stats["total_pnl"])
    log.info("  Win rate          : %.1f%%", stats["win_rate_pct"])
    log.info("  Avg win / loss    : $%.2f / $%.2f", stats["avg_win_usd"], stats["avg_loss_usd"])
    log.info("  Profit factor     : %.2f", stats["profit_factor"])
    log.info("  Max drawdown      : $%.2f", stats["max_drawdown_usd"])
    log.info("  Sharpe ratio      : %.2f", stats["sharpe_ratio"])
    log.info("  Halted days       : %d", stats["halted_days"])
    log.info("─" * 60)

    return trade_rows, signal_rows, stats, equity_curve

# ═══════════════════════════════════════════════════════════════════════════════
# CSV WRITERS
# ═══════════════════════════════════════════════════════════════════════════════

def write_trades_csv(trade_rows: list[dict], path: str = "backtest_trades.csv"):
    if not trade_rows:
        return
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=trade_rows[0].keys())
        w.writeheader()
        w.writerows(trade_rows)
    log.info("💾 Trades written to %s", path)

def write_signals_csv(signal_rows: list[dict], path: str = "backtest_signals.csv"):
    if not signal_rows:
        return
    move_cols = [f"move_{n}d_pct" for n in MOVE_WINDOWS]
    base_fields = [k for k in signal_rows[0].keys() if k not in move_cols]
    fieldnames = base_fields + move_cols
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        w.writeheader()
        for row in signal_rows:
            for col in move_cols:
                row.setdefault(col, None)
            w.writerow(row)
    log.info("💾 Signals written to %s", path)

# ═══════════════════════════════════════════════════════════════════════════════
# HTML REPORT
# ═══════════════════════════════════════════════════════════════════════════════

def write_html_report(
    trade_rows: list[dict],
    signal_rows: list[dict],
    stats: dict,
    equity_curve: list[dict],
    path: str = "backtest_report.html",
):
    # Prepare chart data
    eq_dates  = [p["date"] for p in equity_curve]
    eq_values = [p["cumulative_pnl"] for p in equity_curve]

    # Magnitude vs 1-day move scatter (passing bullish signals only)
    scatter_mag  = [r["magnitude"]   for r in signal_rows if r.get("passes_gate") and r.get("move_1d_pct") is not None]
    scatter_move = [r["move_1d_pct"] for r in signal_rows if r.get("passes_gate") and r.get("move_1d_pct") is not None]

    # Confidence calibration: bucket by 0.1 intervals, show % correct direction
    conf_buckets: dict[str, list] = {}
    for r in signal_rows:
        if r.get("move_1d_pct") is None:
            continue
        bucket = f"{int(float(r['confidence']) * 10) / 10:.1f}"
        conf_buckets.setdefault(bucket, []).append(r["move_1d_pct"] > 0)
    conf_labels   = sorted(conf_buckets.keys())
    conf_accuracy = [round(sum(conf_buckets[k]) / len(conf_buckets[k]) * 100, 1) if conf_buckets[k] else 0
                     for k in conf_labels]

    # Trade table rows (last 50)
    trade_table = ""
    for t in reversed(trade_rows[-50:]):
        color = "#2ecc71" if t["pnl_usd"] > 0 else "#e74c3c"
        trade_table += f"""
        <tr>
          <td>{t['entry_dt'][:10]}</td>
          <td><strong>{t['ticker']}</strong></td>
          <td>{t['magnitude']:.2f}</td>
          <td>{t['confidence']:.2f}</td>
          <td>${t['position_usd']:.0f}</td>
          <td>${t['cost_basis']:.2f}</td>
          <td style="color:{color};font-weight:600">${t['pnl_usd']:+.2f}</td>
          <td style="color:{color}">{t['pnl_pct']:+.1f}%</td>
          <td>{t['exit_reason']}</td>
          <td style="font-size:0.8em;color:#888">{t['reasoning'][:80]}</td>
        </tr>"""

    pnl_color = "#2ecc71" if stats["total_pnl"] >= 0 else "#e74c3c"

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Backtest Report — {stats['period_days']} days</title>
<script src="https://cdnjs.cloudflare.com/ajax/libs/Chart.js/4.4.1/chart.umd.min.js"></script>
<style>
  * {{ box-sizing: border-box; margin: 0; padding: 0; }}
  body {{ font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
          background: #0f1117; color: #e0e0e0; padding: 24px; }}
  h1 {{ font-size: 1.6em; margin-bottom: 4px; color: #fff; }}
  .subtitle {{ color: #888; font-size: 0.9em; margin-bottom: 28px; }}
  .grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(160px, 1fr));
           gap: 14px; margin-bottom: 28px; }}
  .card {{ background: #1a1d27; border-radius: 10px; padding: 18px; }}
  .card .label {{ font-size: 0.75em; color: #888; text-transform: uppercase;
                  letter-spacing: .05em; margin-bottom: 6px; }}
  .card .value {{ font-size: 1.5em; font-weight: 700; color: #fff; }}
  .card .value.pos {{ color: #2ecc71; }}
  .card .value.neg {{ color: #e74c3c; }}
  .charts {{ display: grid; grid-template-columns: 1fr 1fr; gap: 20px; margin-bottom: 28px; }}
  .chart-box {{ background: #1a1d27; border-radius: 10px; padding: 20px; }}
  .chart-box h2 {{ font-size: 0.95em; color: #aaa; margin-bottom: 14px; }}
  .chart-box.full {{ grid-column: 1 / -1; }}
  table {{ width: 100%; border-collapse: collapse; background: #1a1d27;
           border-radius: 10px; overflow: hidden; font-size: 0.85em; }}
  th {{ background: #252836; padding: 10px 12px; text-align: left;
        color: #aaa; font-weight: 600; font-size: 0.8em; text-transform: uppercase; }}
  td {{ padding: 9px 12px; border-bottom: 1px solid #252836; }}
  tr:last-child td {{ border-bottom: none; }}
  tr:hover td {{ background: #21232f; }}
  .section-title {{ font-size: 1em; font-weight: 600; color: #ccc;
                    margin: 28px 0 12px; }}
</style>
</head>
<body>
<h1>📊 Backtest Report</h1>
<div class="subtitle">
  {stats['period_days']}-day window &nbsp;·&nbsp;
  Model: {OLLAMA_MODEL} &nbsp;·&nbsp;
  MIN_MAGNITUDE: {MIN_MAGNITUDE} &nbsp;·&nbsp;
  BASE_CONFIDENCE: {BASE_CONFIDENCE}
</div>

<div class="grid">
  <div class="card">
    <div class="label">Total P&L</div>
    <div class="value {'pos' if stats['total_pnl'] >= 0 else 'neg'}">${stats['total_pnl']:+,.2f}</div>
  </div>
  <div class="card">
    <div class="label">Win Rate</div>
    <div class="value">{stats['win_rate_pct']}%</div>
  </div>
  <div class="card">
    <div class="label">Trades</div>
    <div class="value">{stats['trades_simulated']}</div>
  </div>
  <div class="card">
    <div class="label">Avg Win</div>
    <div class="value pos">${stats['avg_win_usd']:+,.2f}</div>
  </div>
  <div class="card">
    <div class="label">Avg Loss</div>
    <div class="value neg">${stats['avg_loss_usd']:+,.2f}</div>
  </div>
  <div class="card">
    <div class="label">Profit Factor</div>
    <div class="value">{stats['profit_factor']}</div>
  </div>
  <div class="card">
    <div class="label">Max Drawdown</div>
    <div class="value neg">-${stats['max_drawdown_usd']:,.2f}</div>
  </div>
  <div class="card">
    <div class="label">Sharpe Ratio</div>
    <div class="value">{stats['sharpe_ratio']}</div>
  </div>
  <div class="card">
    <div class="label">Articles Scored</div>
    <div class="value">{stats['articles_scored']:,}</div>
  </div>
  <div class="card">
    <div class="label">Signals Passing</div>
    <div class="value">{stats['signals_passing']}</div>
  </div>
  <div class="card">
    <div class="label">Halted Days</div>
    <div class="value">{stats['halted_days']}</div>
  </div>
</div>

<div class="charts">
  <div class="chart-box full">
    <h2>Equity Curve (Cumulative P&L)</h2>
    <canvas id="equity"></canvas>
  </div>
  <div class="chart-box">
    <h2>Magnitude vs 1-Day Stock Move</h2>
    <canvas id="scatter"></canvas>
  </div>
  <div class="chart-box">
    <h2>Confidence Calibration (% bullish signals where stock rose in 1 day)</h2>
    <canvas id="calibration"></canvas>
  </div>
</div>

<div class="section-title">Recent Trades (last 50)</div>
<table>
  <thead>
    <tr>
      <th>Date</th><th>Ticker</th><th>Mag</th><th>Conf</th>
      <th>Size</th><th>Cost</th><th>P&L $</th><th>P&L %</th>
      <th>Exit</th><th>Reasoning</th>
    </tr>
  </thead>
  <tbody>{trade_table}</tbody>
</table>

<script>
const eq = document.getElementById('equity').getContext('2d');
new Chart(eq, {{
  type: 'line',
  data: {{
    labels: {json.dumps(eq_dates)},
    datasets: [{{
      label: 'Cumulative P&L ($)',
      data: {json.dumps(eq_values)},
      borderColor: '{pnl_color}',
      backgroundColor: '{pnl_color}22',
      fill: true,
      tension: 0.3,
      pointRadius: 2,
    }}]
  }},
  options: {{
    responsive: true,
    plugins: {{ legend: {{ labels: {{ color: '#aaa' }} }} }},
    scales: {{
      x: {{ ticks: {{ color: '#666', maxTicksLimit: 12 }}, grid: {{ color: '#252836' }} }},
      y: {{ ticks: {{ color: '#666', callback: v => '$' + v.toLocaleString() }},
            grid: {{ color: '#252836' }} }}
    }}
  }}
}});

const sc = document.getElementById('scatter').getContext('2d');
new Chart(sc, {{
  type: 'scatter',
  data: {{
    datasets: [{{
      label: 'Signal',
      data: {json.dumps([{"x": m, "y": v} for m, v in zip(scatter_mag, scatter_move)])},
      backgroundColor: '#3498db88',
      pointRadius: 4,
    }}]
  }},
  options: {{
    responsive: true,
    plugins: {{ legend: {{ display: false }} }},
    scales: {{
      x: {{ title: {{ display: true, text: 'Magnitude', color: '#888' }},
            ticks: {{ color: '#666' }}, grid: {{ color: '#252836' }} }},
      y: {{ title: {{ display: true, text: '1-Day Move %', color: '#888' }},
            ticks: {{ color: '#666' }}, grid: {{ color: '#252836' }} }}
    }}
  }}
}});

const cal = document.getElementById('calibration').getContext('2d');
new Chart(cal, {{
  type: 'bar',
  data: {{
    labels: {json.dumps(conf_labels)},
    datasets: [{{
      label: '% Correct Direction',
      data: {json.dumps(conf_accuracy)},
      backgroundColor: '#9b59b688',
      borderColor: '#9b59b6',
      borderWidth: 1,
    }}]
  }},
  options: {{
    responsive: true,
    plugins: {{ legend: {{ labels: {{ color: '#aaa' }} }} }},
    scales: {{
      x: {{ title: {{ display: true, text: 'Confidence Score', color: '#888' }},
            ticks: {{ color: '#666' }}, grid: {{ color: '#252836' }} }},
      y: {{ min: 0, max: 100,
            title: {{ display: true, text: '% Correct', color: '#888' }},
            ticks: {{ color: '#666', callback: v => v + '%' }},
            grid: {{ color: '#252836' }} }}
    }}
  }}
}});
</script>
</body>
</html>"""

    Path(path).write_text(html)
    log.info("📊 HTML report written to %s", path)

# ═══════════════════════════════════════════════════════════════════════════════
# ENTRY POINT
# ═══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="News trading bot backtest")
    parser.add_argument("--days",     type=int,  default=180, help="Lookback window in days (default 180)")
    parser.add_argument("--no-cache", action="store_true",    help="Ignore cached Ollama scores")
    parser.add_argument("--workers",  type=int,  default=2,   help="Parallel Ollama scoring threads (default 2)")
    parser.add_argument("--end-date", type=str,  default=None,
                        help="Backtest end date YYYY-MM-DD (default: today-5d). Use to target historical regimes.")
    args = parser.parse_args()

    trade_rows, signal_rows, stats, equity_curve = run_backtest(
        days=args.days,
        use_cache=not args.no_cache,
        workers=args.workers,
        end_date=args.end_date,
    )

    write_trades_csv(trade_rows)
    write_signals_csv(signal_rows)
    write_html_report(trade_rows, signal_rows, stats, equity_curve)

    log.info("✅ Done. Open backtest_report.html in your browser to view results.")