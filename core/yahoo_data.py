"""
yahoo_data.py — real historical daily bars from Yahoo Finance.

Why: this environment's Alpaca historical store is unreliable for older windows
(e.g. NFLX priced at $19.95 in 2022, EDBL at $423K). Yahoo's chart endpoint
returns real-world-correct data — verified against AAPL/MSFT/JPM/KO/MCD for
May-2022. It's the trustworthy source for cross-regime (2022) backtests.

Connectivity notes (this env):
  • `fc.yahoo.com` (yfinance's crumb host) is null-routed to 0.0.0.0 → we skip it
    and hit the public chart endpoint directly (no crumb needed).
  • Yahoo 429s non-browser user-agents → we use curl_cffi with chrome impersonation.

Returns the same bar dict shape as backtest.get_stock_bars:
    {"t": datetime(UTC), "o","h","l","c": float, "v": int}

On-disk cache (yahoo_bars_cache.json) keyed by ticker so repeated/over­lapping
backtest windows don't re-hit Yahoo (which rate-limits at volume).
"""

from __future__ import annotations

import json
import time
from datetime import datetime, timezone
from pathlib import Path
from threading import Lock

from curl_cffi import requests as _creq

_CHART = "https://query1.finance.yahoo.com/v8/finance/chart/{tk}"
_CACHE_FILE = Path("yahoo_bars_cache.json")
_lock = Lock()

# ticker -> {iso_date: [o,h,l,c,v]}  (one merged series per ticker)
_cache: dict[str, dict] = {}
_loaded = False


def _load_cache():
    global _cache, _loaded
    if _loaded:
        return
    if _CACHE_FILE.exists():
        try:
            _cache = json.loads(_CACHE_FILE.read_text())
        except Exception:
            _cache = {}
    _loaded = True


def _save_cache():
    try:
        _CACHE_FILE.write_text(json.dumps(_cache, separators=(",", ":")))
    except Exception:
        pass


def _fetch_yahoo(ticker: str, p1: int, p2: int, tries: int = 5):
    """Hit the chart endpoint with browser impersonation + 429 backoff."""
    url = _CHART.format(tk=ticker)
    params = {"period1": p1, "period2": p2, "interval": "1d", "events": "split"}
    last = None
    for i in range(tries):
        try:
            r = _creq.get(url, params=params, impersonate="chrome", timeout=15)
            last = r.status_code
            if r.status_code == 200:
                return r.json()
            if r.status_code in (429, 999):       # rate-limited → back off
                time.sleep(1.5 * (i + 1))
                continue
            return None                            # 404 etc. — no data
        except Exception:
            time.sleep(1.0 * (i + 1))
    return None


def _parse(payload) -> dict:
    """Yahoo chart JSON → {iso_date: [o,h,l,c,v]}, split-adjusted closes."""
    try:
        res = payload["chart"]["result"][0]
        ts = res["timestamp"]
        q = res["indicators"]["quote"][0]
        out = {}
        for i, t in enumerate(ts):
            o, h, l, c, v = q["open"][i], q["high"][i], q["low"][i], q["close"][i], q["volume"][i]
            if c is None:
                continue
            day = datetime.fromtimestamp(t, timezone.utc).date().isoformat()
            out[day] = [o, h, l, c, v]
        return out
    except Exception:
        return {}


def get_yahoo_bars(ticker: str, start: datetime, end: datetime) -> list[dict]:
    """
    Real daily bars for [start, end] from Yahoo, cached per ticker.
    Returns [] on failure (caller can fall back to another source).
    """
    _load_cache()
    if start.tzinfo is None:
        start = start.replace(tzinfo=timezone.utc)
    if end.tzinfo is None:
        end = end.replace(tzinfo=timezone.utc)
    s_iso, e_iso = start.date().isoformat(), end.date().isoformat()

    with _lock:
        series = _cache.get(ticker)
        have = series and min(series) <= s_iso and max(series) >= e_iso if series else False

    if not have:
        # Fetch a generous superset window once, then cache & slice.
        p1 = int(start.timestamp()) - 86400 * 7
        p2 = int(end.timestamp()) + 86400 * 7
        payload = _fetch_yahoo(ticker, p1, p2)
        parsed = _parse(payload) if payload else {}
        with _lock:
            merged = _cache.get(ticker, {})
            merged.update(parsed)
            _cache[ticker] = merged
            _save_cache()
            series = merged

    if not series:
        return []
    bars = []
    for day, (o, h, l, c, v) in sorted(series.items()):
        if s_iso <= day <= e_iso:
            bars.append({
                "t": datetime.fromisoformat(day).replace(tzinfo=timezone.utc),
                "o": float(o) if o is not None else float(c),
                "h": float(h) if h is not None else float(c),
                "l": float(l) if l is not None else float(c),
                "c": float(c),
                "v": int(v) if v is not None else 0,
            })
    return bars


if __name__ == "__main__":
    # Smoke test: real May-2022 prices
    for tk in ["AAPL", "MSFT", "NFLX"]:
        b = get_yahoo_bars(tk, datetime(2022, 5, 2, tzinfo=timezone.utc),
                           datetime(2022, 5, 6, tzinfo=timezone.utc))
        print(tk, "→", [f"{x['t'].date()}:${x['c']:.2f}" for x in b[:3]])
