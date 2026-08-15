"""
sec_backtest_sameday.py — research-only (2026-07-23): re-runs the SEC 8-K backtest's already-
scored filings (data/sec_backtest.csv, produced by sec_backtest.py) through a SAME-DAY exit
instead of the multi-day tiered_trail hold, matching the lotto strategy's exit mechanics
(LOTTO_HARD_CAP_MULT hard profit-take + LOTTO_EXIT_TIERS trailing stop, core/config.py) applied
intraday and forced flat at the entry day's close if neither triggers.

Does NOT re-score with Ollama -- reuses the sentiment/magnitude/confidence already recorded per
prompt variant in data/sec_backtest.csv, so this isolates the exit-parameter change as the only
variable. Entry mechanics (contract selection: NC_DELTA/NC_DTE, the live news_call gate) are
unchanged from sec_backtest.py -- only how long the position is held and how it's exited differs.

simulate_option_pnl() in core/backtest.py operates on DAILY bars and can't represent an
intraday same-day exit (get_price_at already returns the entry day's CLOSE, so a same-day daily-
bar exit would be a zero-length hold). This script fetches real Alpaca IEX hourly bars for just
the entry day instead -- confirmed available back through 2022 (bear-2022 regime) via a direct
API check, unlike Yahoo's ~2-year intraday retention limit.

Usage:  .venv/bin/python research/sec_backtest_sameday.py
"""
import csv
import os
import sys
import time
from datetime import datetime, timedelta, timezone

from dotenv import load_dotenv

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "core"))
import backtest as _bt
import config as _cfg
from alpaca.data.historical import StockHistoricalDataClient
from alpaca.data.requests import StockBarsRequest
from alpaca.data.timeframe import TimeFrame
from alpaca.data.enums import Adjustment

load_dotenv()

IN_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "sec_backtest.csv")
OUT_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "sec_backtest_sameday.csv")

_stock_client = StockHistoricalDataClient(os.environ["ALPACA_API_KEY"], os.environ["ALPACA_SECRET_KEY"])

# Same live news_call contract-selection geometry as sec_backtest.py -- only the exit changes.
NC_DELTA = _cfg.NEWS_CALL_TARGET_DELTA
NC_DTE   = round((_cfg.NEWS_CALL_DTE_MIN + _cfg.NEWS_CALL_DTE_MAX) / 2)

VARIANTS = ["current", "full_addendum", "routine_only"]

_bars_cache: dict = {}


def _fetch_day_bars(ticker: str, entry_dt: datetime) -> list:
    """All hourly bars for the entry calendar day (market open through close), single fetch
    covering both the entry anchor and the post-entry walk so both come from the same
    adjustment convention -- mixing sources here previously fabricated a fake intraday jump
    (see the Adjustment.ALL comment below)."""
    key = (ticker, entry_dt.date())
    if key in _bars_cache:
        return _bars_cache[key]
    bars = []
    try:
        day_start = entry_dt.replace(hour=0, minute=0, second=0, microsecond=0)
        resp = _stock_client.get_stock_bars(StockBarsRequest(
            symbol_or_symbols=ticker, timeframe=TimeFrame.Hour,
            start=day_start, end=day_start + timedelta(days=1), feed="iex",
            adjustment=Adjustment.ALL))  # split+div adjusted -- a raw/adjusted mismatch between
                                          # the entry-anchor bar and the walk bars reads as a fake
                                          # intraday jump (confirmed on LTC: raw $37.35 vs
                                          # adjusted $28.43 same day -- a fabricated +31% move)
        raw = (resp.data or {}).get(ticker, [])
        bars = [{"t": b.timestamp, "c": float(b.close)} for b in raw
                if b.timestamp.date() == entry_dt.date()]
    except Exception as e:
        print(f"    intraday fetch failed for {ticker}: {e}")
    _bars_cache[key] = bars
    return bars


def _passes_live_gate(sentiment, magnitude, confidence) -> bool:
    if sentiment != "bullish":
        return False
    if magnitude < _cfg.MIN_MAGNITUDE:
        return False
    if confidence < _cfg.BASE_CONFIDENCE + (1 - magnitude) * _cfg.CONFIDENCE_SLOPE:
        return False
    if magnitude < _cfg.NEWS_CALL_MIN_MAGNITUDE:
        return False
    return True


def _scale(mag, conf):
    return max(_cfg.MAX_POSITION_USD * 0.10, _cfg.MAX_POSITION_USD * mag * conf)


def simulate_same_day_exit(ticker: str, date: str, sig: dict) -> "dict | None":
    """Same entry gate/contract selection as sec_backtest.py's simulate_trade(), but exits
    same-day using lotto's actual exit mechanics (hard cap + tiered trail) checked hourly,
    forced flat at the entry day's last available bar if neither fires."""
    magnitude, confidence = float(sig.get("magnitude") or 0), float(sig.get("confidence") or 0)
    if not _passes_live_gate(sig.get("sentiment"), magnitude, confidence):
        return None
    if not _bt.is_valid_stock_ticker(ticker):
        return None
    entry_dt = datetime.fromisoformat(f"{date}T15:00:00+00:00")

    day_bars = _fetch_day_bars(ticker, entry_dt)
    if not day_bars:
        return None
    # Entry-anchor bar: the last bar AT OR BEFORE entry_dt (falls back to the first bar of the
    # day if entry_dt is before the first available bar, e.g. a pre-market gap in the feed).
    # Using get_price_at's same-day CLOSE here (as sec_backtest.py's multi-day version does) was
    # the original bug: it anchors the strike/entry premium to a price from LATER in the day
    # than the intraday bars being walked, which is look-ahead bias -- confirmed by a near-0%
    # same-day win rate (every intraday mark reads as "below entry" because entry was priced at
    # the day's end). The exit walk only uses bars strictly AFTER this anchor.
    before = [b for b in day_bars if b["t"] <= entry_dt]
    anchor = before[-1] if before else day_bars[0]
    entry_stock_price = anchor["c"]
    intraday_bars = [b for b in day_bars if b["t"] > anchor["t"]]
    if not intraday_bars:
        return None  # no post-entry data this day -- skip rather than fake a same-bar exit
    if entry_stock_price > _cfg.MAX_SANE_STOCK_PRICE:
        return None

    position_usd = _scale(magnitude, confidence)
    r = _cfg.RISK_FREE_RATE
    price_fn, strike_fn = _bt.bs_call_price, _bt.strike_for_delta

    base_iv = _bt.base_iv_for(ticker, entry_dt)
    entry_iv = base_iv * _cfg.NEWS_IV_MULTIPLIER
    T_entry = NC_DTE / 365.0
    strike = round(strike_fn(entry_stock_price, T_entry, entry_iv, NC_DELTA, r), 2)
    entry_premium = price_fn(entry_stock_price, strike, T_entry, entry_iv, r)
    if entry_premium < 0.05:
        return None
    if entry_premium * 100 > position_usd * _cfg.MAX_CONTRACT_BUDGET_MULT:
        return None

    entry_fill = entry_premium * (1 + _bt._spread_pct(entry_premium, ticker, 1.0) / 2)
    qty = max(1, int(position_usd / (entry_fill * 100)))
    cost_basis = entry_fill * 100 * qty

    peak_value = entry_premium
    exit_premium, exit_dt, exit_reason = entry_premium, None, "eod"
    hard_cap_mult = _cfg.LOTTO_HARD_CAP_MULT if _cfg.LOTTO_HARD_CAP_ENABLED else None
    tiers = _cfg.LOTTO_EXIT_TIERS

    for bar in intraday_bars:
        days_held = (bar["t"] - anchor["t"]).total_seconds() / 86400.0
        T = max((NC_DTE - days_held) / 365.0, 1.0 / 365.0)
        iv = _bt.iv_crush_path(base_iv, days_held, _cfg.NEWS_IV_MULTIPLIER, _cfg.IV_CRUSH_HALFLIFE_DAYS)
        premium = max(0.01, price_fn(bar["c"], strike, T, iv, r))
        if premium > peak_value:
            peak_value = premium
        ret = premium / entry_premium - 1.0

        if hard_cap_mult and premium >= entry_premium * hard_cap_mult:
            exit_premium, exit_dt, exit_reason = premium, bar["t"], "hard_cap"
            break
        trail = next(t for thresh, t in tiers if ret < thresh)
        if premium <= peak_value * (1 - trail):
            exit_premium, exit_dt, exit_reason = premium, bar["t"], "tiered_trail"
            break
    else:
        last = intraday_bars[-1]
        days_held = (last["t"] - anchor["t"]).total_seconds() / 86400.0
        T = max((NC_DTE - days_held) / 365.0, 1.0 / 365.0)
        iv = _bt.iv_crush_path(base_iv, days_held, _cfg.NEWS_IV_MULTIPLIER, _cfg.IV_CRUSH_HALFLIFE_DAYS)
        exit_premium = max(0.01, price_fn(last["c"], strike, T, iv, r))
        exit_dt, exit_reason = last["t"], "eod"

    exit_fill = max(0.0, exit_premium * (1 - _bt._spread_pct(exit_premium, ticker, 1.0) / 2))
    exit_value = exit_fill * 100 * qty
    pnl_usd = exit_value - cost_basis
    return {
        "ticker": ticker, "entry_dt": entry_dt.isoformat(), "exit_dt": exit_dt.isoformat() if exit_dt else None,
        "exit_reason": exit_reason, "entry_stock_price": entry_stock_price, "strike": strike,
        "entry_premium": round(entry_premium, 4), "exit_premium": round(exit_premium, 4),
        "qty": qty, "cost_basis": round(cost_basis, 2), "exit_value": round(exit_value, 2),
        "pnl_usd": round(pnl_usd, 2), "pnl_pct": round(pnl_usd / cost_basis * 100, 2) if cost_basis else 0,
    }


def main():
    rows_in = list(csv.DictReader(open(IN_PATH)))
    print(f"loaded {len(rows_in)} scored filings from {IN_PATH}")

    out_rows = []
    for i, r in enumerate(rows_in):
        out = {"regime": r["regime"], "ticker": r["ticker"], "date": r["date"]}
        any_traded = False
        for v in VARIANTS:
            sig = {"sentiment": r.get(f"{v}_sent"), "magnitude": r.get(f"{v}_mag"), "confidence": r.get(f"{v}_conf")}
            trade = simulate_same_day_exit(r["ticker"], r["date"], sig)
            out[f"{v}_traded"] = trade is not None
            out[f"{v}_pnl_usd"] = trade["pnl_usd"] if trade else ""
            out[f"{v}_pnl_pct"] = trade["pnl_pct"] if trade else ""
            out[f"{v}_exit_reason"] = trade["exit_reason"] if trade else ""
            any_traded = any_traded or trade is not None
        out_rows.append(out)
        if any_traded:
            print(f"[{i+1}/{len(rows_in)}] {r['ticker']:>6} {r['date']}  " + "  ".join(
                f"{v}: " + (f"${out[f'{v}_pnl_usd']:.0f} ({out[f'{v}_exit_reason']})" if out[f"{v}_traded"] else "no-trade")
                for v in VARIANTS))
        if (i + 1) % 25 == 0:
            with open(OUT_PATH, "w", newline="") as fh:
                w = csv.DictWriter(fh, fieldnames=list(out_rows[0].keys()))
                w.writeheader()
                w.writerows(out_rows)
        time.sleep(0.05)

    with open(OUT_PATH, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(out_rows[0].keys()))
        w.writeheader()
        w.writerows(out_rows)
    print(f"\n{len(out_rows)} rows written to {OUT_PATH}")


if __name__ == "__main__":
    main()
