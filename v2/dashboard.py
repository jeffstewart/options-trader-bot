"""
dashboard.py (v2) — slimmed-down local web dashboard. v2 currently runs ONE live strategy
(lotto; news_call/PEAD disabled, no pairs/bear_short/qqq_macro/router at all), so this drops
everything core/dashboard.py needed for a multi-strategy book: the per-strategy P&L breakdown
table, the per-strategy chart overlay lines, stock/short position handling, the shadow-router
attribution view, and the baseline-date chart picker (fixed 30-day window instead). Positions
and trades are lotto-only in practice today, but nothing here HARDCODES that -- if news_call/PEAD
are ever re-enabled, they show up in the same tables/rows automatically (the `strategy` column
already carries whichever leg produced the trade).

Run alongside bot.py:
    .venv/bin/python dashboard.py
Then open http://localhost:5002 (a different default port than v1's 5001, so both can run at once).
"""

import csv
import json
import os
import re
import subprocess
import time as _time
from collections import Counter
from datetime import datetime, timezone, date, timedelta
from pathlib import Path

from flask import Flask, jsonify, render_template_string
from alpaca.trading.client import TradingClient
from alpaca.trading.requests import GetPortfolioHistoryRequest
from alpaca.data.historical import StockHistoricalDataClient
from alpaca.data.requests import StockBarsRequest
from alpaca.data.timeframe import TimeFrame
from dotenv import load_dotenv

import config as cfg

load_dotenv()

app = Flask(__name__)

# Bare filenames -- daemons run with CWD=data/ (config.py), same convention as core/dashboard.py.
TRADES_CSV        = Path("trades.csv")
CLOSED_TRADES_CSV = Path("closed_trades.csv")
BOT_STATE_FILE    = Path("bot_state.json")
BOT_LOG           = cfg.LOG_DIR / "bot.log"
BOT_PROCESS       = "v2/bot.py"
PORT              = int(os.environ.get("V2_DASHBOARD_PORT", 5002))

trading_client = TradingClient(cfg.ALPACA_KEY, cfg.ALPACA_SECRET, paper=True)
_data_client   = StockHistoricalDataClient(cfg.ALPACA_KEY, cfg.ALPACA_SECRET)

_bench_cache = {"ts": 0.0, "data": None}
_BENCH_TTL = 300.0


# ── helpers ──────────────────────────────────────────────────────────────────

def bot_is_running() -> bool:
    try:
        return bool(subprocess.check_output(["pgrep", "-f", BOT_PROCESS], text=True).strip())
    except subprocess.CalledProcessError:
        return False


def get_account() -> dict:
    try:
        a = trading_client.get_account()
        last_eq = float(a.last_equity)
        return {
            "equity":        float(a.equity),
            "cash":          float(a.cash),
            "buying_power":  float(a.buying_power),
            "pnl_today":     float(a.equity) - last_eq,
            "pnl_today_pct": (float(a.equity) - last_eq) / last_eq * 100 if last_eq else 0,
        }
    except Exception as e:
        return {"error": str(e)}


def _load_bot_state() -> dict:
    try:
        return json.loads(BOT_STATE_FILE.read_text()) if BOT_STATE_FILE.exists() else {}
    except Exception:
        return {}


def get_positions() -> list[dict]:
    """v2 only ever holds long options (lotto today; news_call would be the same shape if
    re-enabled) -- no stock/short branching needed, unlike v1's multi-strategy book."""
    bot_state = _load_bot_state()
    try:
        result = []
        for p in trading_client.get_all_positions():
            state = bot_state.get(p.symbol, {})
            entry = float(p.avg_entry_price)
            current = float(p.current_price or 0)
            result.append({
                "symbol":            p.symbol,
                "strategy":          state.get("strategy", "lotto"),
                "qty":               float(p.qty),
                "entry_price":       entry,
                "current_price":     current,
                "unrealized_pl":     float(p.unrealized_pl or 0),
                "unrealized_pl_pct": float(p.unrealized_plpc or 0) * 100,
                "stop_price":        state.get("stop_price"),
                "peak_price":        state.get("peak_price"),
            })
        result.sort(key=lambda x: x["unrealized_pl"], reverse=True)
        return result
    except Exception as e:
        return [{"error": str(e)}]


CLOSE_REASON_LABELS = {
    "lotto_eod":        "Same-day close",
    "lotto_stop_loss":  "Stop-loss",
    "lotto_cap":        "Take-profit cap (3x)",
    "time-stop":        "Max hold",
}


def _close_reason_label(reason: str) -> str:
    r = (reason or "").strip()
    if r in CLOSE_REASON_LABELS:
        return CLOSE_REASON_LABELS[r]
    rl = r.lower()
    if "cap" in rl or "profit" in rl:
        return "Take-profit cap"
    if "stop" in rl:
        return "Stop-loss"
    if "eod" in rl or "time" in rl or "hold" in rl:
        return "Same-day close"
    return r or "—"


def get_recent_trades(n: int = 30, open_positions: list[dict] | None = None) -> list[dict]:
    """Unified, time-sorted feed of position OPENS (trades.csv) and CLOSES (closed_trades.csv).
    OPEN events are flagged still_open against the broker's live holdings so a same-day-closed
    entry isn't shown with a stale live badge."""
    events: list[dict] = []

    def _read(path, kind):
        if not path.exists():
            return
        try:
            with open(path, newline="") as f:
                for r in csv.DictReader(f):
                    row = dict(r)
                    row["kind"] = kind
                    if kind == "close":
                        row["reason_label"] = _close_reason_label(row.get("reason", ""))
                    events.append(row)
        except Exception:
            pass

    _read(TRADES_CSV, "open")
    _read(CLOSED_TRADES_CSV, "close")

    held = {p.get("symbol") for p in (open_positions or []) if isinstance(p, dict) and "error" not in p}
    seen_live: set = set()
    for e in sorted((e for e in events if e["kind"] == "open"),
                    key=lambda e: e.get("timestamp", ""), reverse=True):
        sym = e.get("option_symbol") or ""
        live = bool(sym and sym != "-" and sym in held and sym not in seen_live)
        e["still_open"] = live
        if live:
            seen_live.add(sym)

    events.sort(key=lambda e: e.get("timestamp", ""), reverse=True)
    return events[:n]


def get_stats() -> dict:
    stats = {"total_trades": 0, "today_trades": 0, "total_closed": 0,
             "win_rate": None, "total_realized": 0.0}
    if TRADES_CSV.exists():
        try:
            with open(TRADES_CSV, newline="") as f:
                rows = list(csv.DictReader(f))
            today_s = date.today().isoformat()
            stats["total_trades"] = len(rows)
            stats["today_trades"] = sum(1 for r in rows if r.get("timestamp", "").startswith(today_s))
        except Exception:
            pass
    if CLOSED_TRADES_CSV.exists():
        try:
            with open(CLOSED_TRADES_CSV, newline="") as f:
                rows = list(csv.DictReader(f))
            pnls = [float(r.get("pnl_usd", 0) or 0) for r in rows]
            stats["total_closed"] = len(pnls)
            stats["total_realized"] = round(sum(pnls), 2)
            if pnls:
                stats["win_rate"] = round(sum(1 for p in pnls if p > 0) / len(pnls) * 100, 1)
        except Exception:
            pass
    return stats


def get_benchmark(account: dict) -> dict:
    """Account return vs SPY over 'today' and the trailing month. Cached 5 min."""
    if _time.time() - _bench_cache["ts"] < _BENCH_TTL and _bench_cache["data"]:
        return _bench_cache["data"]
    d = {}
    try:
        ph = trading_client.get_portfolio_history(GetPortfolioHistoryRequest(period="1M", timeframe="1D"))
        pts = [(t, e) for t, e in zip(ph.timestamp or [], ph.equity or []) if e]
        closes = []
        if len(pts) >= 2:
            start_date = datetime.fromtimestamp(int(pts[0][0]), tz=timezone.utc).date()
            bars = _data_client.get_stock_bars(StockBarsRequest(
                symbol_or_symbols="SPY", timeframe=TimeFrame.Day,
                start=datetime.combine(start_date, datetime.min.time(), tzinfo=timezone.utc) - timedelta(days=6)
            )).data.get("SPY", [])
            sbars = [(b.timestamp.date(), float(b.close)) for b in bars]
            on_after = [c for dte, c in sbars if dte >= start_date]
            closes = on_after if len(on_after) >= 2 else [c for _, c in sbars]
        if len(pts) >= 2 and len(closes) >= 2:
            acct_p = (pts[-1][1] / pts[0][1] - 1) * 100
            spy_p = (closes[-1] / closes[0] - 1) * 100
            spy_today = (closes[-1] / closes[-2] - 1) * 100
            acct_today = float(account.get("pnl_today_pct", 0.0) or 0.0)
            d = {
                "today_account": round(acct_today, 2), "today_spy": round(spy_today, 2),
                "today_alpha": round(acct_today - spy_today, 2),
                "period_account": round(acct_p, 2), "period_spy": round(spy_p, 2),
                "period_alpha": round(acct_p - spy_p, 2),
            }
            _bench_cache.update(ts=_time.time(), data=d)
    except Exception as e:
        d = {"error": str(e)}
    return d


def _ffill_grid(days, by_date, pre=None):
    out, last = [], pre
    for d in days:
        if d in by_date:
            last = by_date[d]
        out.append(last)
    return out


def get_chart_data(window_days: int = 30) -> dict:
    """Portfolio equity vs SPY, both rebased to $0 at the window start. Fixed window (no
    baseline picker, unlike v1) -- keeps the chart simple for a single-strategy book."""
    today = date.today()
    baseline = today - timedelta(days=window_days)
    days = [baseline + timedelta(days=i) for i in range((today - baseline).days + 1)]
    labels = [d.isoformat() for d in days]
    series, base_eq = {}, None
    try:
        ph = trading_client.get_portfolio_history(GetPortfolioHistoryRequest(period=f"{window_days}D", timeframe="1D"))
        eq_by_date = {}
        for t, e in zip(ph.timestamp or [], ph.equity or []):
            if e:
                eq_by_date[datetime.fromtimestamp(int(t), tz=timezone.utc).date()] = float(e)
        if eq_by_date:
            sorted_eq = sorted(eq_by_date.items())
            prior = [v for dt, v in sorted_eq if dt <= baseline]
            base_eq = prior[-1] if prior else sorted_eq[0][1]
            eq_grid = _ffill_grid(days, eq_by_date, pre=base_eq)
            series["Portfolio"] = [round(v - base_eq, 2) if v is not None else None for v in eq_grid]

        if base_eq:
            bars = _data_client.get_stock_bars(StockBarsRequest(
                symbol_or_symbols="SPY", timeframe=TimeFrame.Day,
                start=datetime.combine(baseline, datetime.min.time(), tzinfo=timezone.utc) - timedelta(days=6)
            )).data.get("SPY", [])
            px_by_date = {b.timestamp.date(): float(b.close) for b in bars}
            if px_by_date:
                sp = sorted(px_by_date.items())
                prior_px = [v for dt, v in sp if dt <= baseline]
                base_px = prior_px[-1] if prior_px else sp[0][1]
                px_grid = _ffill_grid(days, px_by_date, pre=base_px)
                series["SPY"] = [round(base_eq * (v / base_px - 1), 2) if v is not None else None for v in px_grid]
    except Exception as e:
        return {"labels": labels, "series": series, "error": str(e)}
    return {"labels": labels, "series": series}


# ── bot log parsing (v2's own log format -- singular `ticker=`, not v1's `tickers=[...]`) ──────

_RE_TS       = re.compile(r"^(\d{2}:\d{2}:\d{2})\s+(INFO|WARNING|ERROR|CRITICAL)\s+(.+)$")
_RE_ARTICLE  = re.compile(r"📰 \[(.+?)\] (.+)")
_RE_LLM      = re.compile(r"🤖 \S+: (\w+)\s+conf=([\d.]+)\s+mag=([\d.]+)\s+ticker=(\S+)")
_RE_TRADE    = re.compile(r"✅ (OPTION BUY|STOCK BUY)")
_RE_CLOSED   = re.compile(r"✅ SELL TO CLOSE|✅ STOCK SELL")
_RE_CUTOFF   = re.compile(r"too close to the same-day forced exit")
_RE_SWITCH   = re.compile(r"🔍 mispricing switch")


def parse_bot_log(tail_lines: int = 200) -> dict:
    if not BOT_LOG.exists():
        return {"activity": [], "errors": [], "counters": {}}
    try:
        with open(BOT_LOG, "r", errors="replace") as f:
            lines = f.readlines()
    except Exception:
        return {"activity": [], "errors": [], "counters": {}}

    lines = lines[-tail_lines:]
    activity, errors = [], []
    counters = Counter()

    for line in lines:
        m = _RE_TS.match(line.rstrip("\n"))
        if not m:
            continue
        ts, level, msg = m.groups()

        if level in ("WARNING", "ERROR", "CRITICAL"):
            errors.append({"ts": ts, "level": level, "msg": msg[:300]})
            continue

        am = _RE_ARTICLE.search(msg)
        if am:
            counters["articles"] += 1
            activity.append({"ts": ts, "icon": "📰", "kind": "article",
                             "text": f"[{am.group(1)}] {am.group(2)}"})
            continue

        lm = _RE_LLM.search(msg)
        if lm:
            sentiment, conf, mag, ticker = lm.groups()
            counters["scored"] += 1
            if sentiment == "bullish":
                counters["bullish"] += 1
            activity.append({"ts": ts, "icon": "🤖", "kind": "signal", "direction": sentiment,
                             "text": f"{ticker}: {sentiment} conf={conf} mag={mag}"})
            continue
        if _RE_CUTOFF.search(msg):
            counters["cutoff_skips"] += 1
            text = re.sub(r"^\s*→\s*", "", msg.strip())
            activity.append({"ts": ts, "icon": "⏰", "kind": "cutoff", "text": text[:200]})
            continue
        if _RE_SWITCH.search(msg):
            counters["mispricing_switches"] += 1
            activity.append({"ts": ts, "icon": "🔍", "kind": "switch", "text": msg.strip()[:200]})
            continue
        if _RE_TRADE.search(msg):
            counters["trades"] += 1
            activity.append({"ts": ts, "icon": "🛒", "kind": "trade", "text": msg.strip()[:200]})
            continue
        if _RE_CLOSED.search(msg):
            activity.append({"ts": ts, "icon": "🔴", "kind": "closed", "text": msg.strip()[:200]})
            continue

    activity.reverse()
    errors.reverse()
    return {"activity": activity[:60], "errors": errors[:30], "counters": dict(counters)}


# ── API routes ────────────────────────────────────────────────────────────────

@app.route("/api/status")
def api_status():
    account = get_account()
    positions = get_positions()
    return jsonify({
        "bot_running": bot_is_running(),
        "account":     account,
        "benchmark":   get_benchmark(account),
        "positions":   positions,
        "trades":      get_recent_trades(30, positions),
        "stats":       get_stats(),
        "log":         parse_bot_log(300),
    })


@app.route("/api/chart")
def api_chart():
    return jsonify(get_chart_data(30))


@app.route("/api/log")
def api_log():
    from flask import request
    n = int(request.args.get("lines", 200))
    return jsonify(parse_bot_log(n))


# ── HTML ──────────────────────────────────────────────────────────────────────

HTML = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8"/>
<meta name="viewport" content="width=device-width, initial-scale=1"/>
<meta name="theme-color" content="#0f1117"/>
<title>Trader Bot v2</title>
<style>
  :root {
    --bg: #0f1117; --surface: #1a1d27; --border: #2a2d3a; --text: #e2e8f0;
    --muted: #64748b; --green: #22c55e; --red: #ef4444; --yellow: #f59e0b;
    --blue: #3b82f6; --accent: #6366f1;
  }
  * { box-sizing: border-box; margin: 0; padding: 0; }
  body { background: var(--bg); color: var(--text); font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
         font-size: 14px; line-height: 1.5; min-height: 100vh; }
  header { background: var(--surface); border-bottom: 1px solid var(--border); padding: 14px 20px;
           display: flex; align-items: center; justify-content: space-between; position: sticky; top: 0; z-index: 10; }
  .logo { font-weight: 700; font-size: 16px; letter-spacing: .5px; }
  .status-pill { display: inline-flex; align-items: center; gap: 6px; font-size: 12px; font-weight: 600;
                 padding: 4px 10px; border-radius: 20px; background: var(--border); }
  .dot { width: 7px; height: 7px; border-radius: 50%; background: var(--muted); }
  .dot.green { background: var(--green); box-shadow: 0 0 6px var(--green); }
  .dot.red { background: var(--red); }
  @keyframes pulse { 0%,100%{opacity:1} 50%{opacity:.4} }

  main { padding: 16px; max-width: 900px; margin: 0 auto; }
  .grid-2 { display: grid; grid-template-columns: 1fr 1fr; gap: 12px; }
  .grid-3 { display: grid; grid-template-columns: repeat(3,1fr); gap: 12px; margin-bottom: 12px; }
  @media(max-width:640px) { .grid-2 { grid-template-columns: 1fr; } .grid-3 { grid-template-columns: repeat(2,1fr); } }

  .card { background: var(--surface); border: 1px solid var(--border); border-radius: 12px; padding: 16px; margin-bottom: 12px; }
  .card-title { font-size: 11px; font-weight: 600; text-transform: uppercase; letter-spacing: .8px; color: var(--muted); margin-bottom: 12px; }
  .card-title-row { display: flex; align-items: center; justify-content: space-between; margin-bottom: 12px; }
  .card-title-row .card-title { margin-bottom: 0; }
  .err-count { font-size: 11px; font-weight: 700; background: var(--red); color: #fff; border-radius: 10px; padding: 1px 7px; }

  .mini-card { background: var(--surface); border: 1px solid var(--border); border-radius: 12px; padding: 14px 16px; }
  .mini-label { font-size: 11px; color: var(--muted); margin-bottom: 4px; }
  .mini-val { font-size: 22px; font-weight: 700; line-height: 1; }
  .mini-sub { font-size: 11px; color: var(--muted); margin-top: 3px; }

  .big-num { font-size: 28px; font-weight: 700; line-height: 1; }
  .sub { font-size: 12px; color: var(--muted); margin-top: 4px; }
  .pos { color: var(--green); } .neg { color: var(--red); } .neu { color: var(--text); }

  table { width: 100%; border-collapse: collapse; }
  th { text-align: left; font-size: 11px; font-weight: 600; text-transform: uppercase; letter-spacing: .6px;
       color: var(--muted); padding: 6px 8px; border-bottom: 1px solid var(--border); }
  td { padding: 8px 8px; border-bottom: 1px solid var(--border); vertical-align: top; }
  tr:last-child td { border-bottom: none; }

  .strategy-pill { display: inline-block; font-size: 10px; font-weight: 700; padding: 2px 7px; border-radius: 10px;
                   text-transform: uppercase; background: #4c1d5522; color: #ec4899; }

  .feed { display: flex; flex-direction: column; gap: 0; }
  .feed-item { display: flex; align-items: baseline; gap: 8px; padding: 7px 0; border-bottom: 1px solid var(--border); font-size: 13px; }
  .feed-item:last-child { border-bottom: none; }
  .feed-ts { font-size: 11px; color: var(--muted); white-space: nowrap; flex-shrink: 0; }
  .feed-icon { flex-shrink: 0; }
  .feed-text { color: var(--text); word-break: break-word; }
  .feed-text.bullish { color: var(--green); } .feed-text.bearish { color: var(--red); }
  .feed-text.neutral { color: var(--muted); } .feed-text.trade { color: var(--yellow); font-weight: 600; }
  .feed-text.closed { color: var(--red); } .feed-text.cutoff { color: var(--muted); }
  .feed-text.switch { color: #a78bfa; }

  .err-item { display: flex; gap: 8px; padding: 7px 0; border-bottom: 1px solid var(--border); font-size: 12px; }
  .err-item:last-child { border-bottom: none; }
  .err-ts { color: var(--muted); white-space: nowrap; flex-shrink: 0; font-size: 11px; }
  .err-level { flex-shrink: 0; font-weight: 700; }
  .err-level.WARNING { color: var(--yellow); } .err-level.ERROR, .err-level.CRITICAL { color: var(--red); }
  .err-msg { color: var(--text); word-break: break-word; }

  .refresh-bar { text-align: center; font-size: 11px; color: var(--muted); padding: 12px 0 20px; }
  .empty { color: var(--muted); font-size: 13px; padding: 12px 0; }
  .two-col { display: grid; grid-template-columns: 1fr 1fr; gap: 12px; }
  @media(max-width:640px) { .two-col { grid-template-columns: 1fr; } }

  .log-wrap { background: #0a0c10; border-radius: 8px; padding: 12px; height: 340px; overflow-y: auto;
              font-family: "SF Mono", "Fira Code", "Menlo", monospace; font-size: 11.5px; line-height: 1.55; scroll-behavior: smooth; }
  .log-wrap::-webkit-scrollbar { width: 6px; }
  .log-wrap::-webkit-scrollbar-track { background: transparent; }
  .log-wrap::-webkit-scrollbar-thumb { background: var(--border); border-radius: 3px; }
  .log-line { white-space: pre-wrap; word-break: break-all; color: #94a3b8; }
  .log-line.INFO { color: #94a3b8; } .log-line.WARNING { color: var(--yellow); }
  .log-line.ERROR, .log-line.CRITICAL { color: var(--red); }
  .log-controls { display: flex; align-items: center; gap: 10px; margin-bottom: 8px; }
  .log-controls label { font-size: 12px; color: var(--muted); }
  .log-controls input[type=checkbox] { accent-color: var(--accent); }
  .log-controls select { background: var(--border); color: var(--text); border: none; border-radius: 6px; padding: 3px 8px; font-size: 12px; }
</style>
<script src="https://cdn.jsdelivr.net/npm/chart.js@4.4.1/dist/chart.umd.min.js"></script>
</head>
<body>
<header>
  <div class="logo">📈 Trader Bot v2 · lotto</div>
  <div class="status-pill">
    <div class="dot" id="statusDot"></div>
    <span id="statusText">Loading…</span>
  </div>
</header>

<main>
  <div class="grid-2" style="margin-bottom:12px">
    <div class="card" style="margin-bottom:0">
      <div class="card-title">Today's P&amp;L</div>
      <div class="big-num" id="pnlToday">—</div>
      <div class="sub" id="pnlTodayPct"></div>
    </div>
    <div class="card" style="margin-bottom:0">
      <div class="card-title">Portfolio Value</div>
      <div class="big-num" id="equity">—</div>
      <div class="sub" id="cash"></div>
    </div>
  </div>

  <div class="card">
    <div class="card-title">vs SPY</div>
    <table>
      <thead><tr><th></th><th style="text-align:right">Account</th><th style="text-align:right">SPY</th><th style="text-align:right">vs&nbsp;SPY</th></tr></thead>
      <tbody>
        <tr><td>Latest session</td><td style="text-align:right" id="bTodayAcct">—</td><td style="text-align:right" id="bTodaySpy">—</td><td style="text-align:right;font-weight:600" id="bTodayAlpha">—</td></tr>
        <tr><td>Past month</td><td style="text-align:right" id="bPerAcct">—</td><td style="text-align:right" id="bPerSpy">—</td><td style="text-align:right;font-weight:600" id="bPerAlpha">—</td></tr>
      </tbody>
    </table>
  </div>

  <div class="card">
    <div class="card-title">Portfolio vs SPY — last 30 days</div>
    <div style="position:relative;height:280px;margin-top:6px"><canvas id="perfChart"></canvas></div>
  </div>

  <div class="grid-3">
    <div class="mini-card">
      <div class="mini-label">Signals scored</div>
      <div class="mini-val" id="cntScored">—</div>
      <div class="mini-sub" id="cntBullish"></div>
    </div>
    <div class="mini-card">
      <div class="mini-label">Trades today</div>
      <div class="mini-val" id="statToday">—</div>
      <div class="mini-sub" id="statTotal"></div>
    </div>
    <div class="mini-card">
      <div class="mini-label">Win rate (all-time)</div>
      <div class="mini-val" id="statWinRate">—</div>
      <div class="mini-sub" id="statRealized"></div>
    </div>
  </div>

  <div class="two-col">
    <div class="card" style="margin-bottom:0">
      <div class="card-title">Activity Feed</div>
      <div class="feed" id="feedWrap"><div class="empty">Loading…</div></div>
    </div>
    <div class="card" style="margin-bottom:0">
      <div class="card-title-row">
        <div class="card-title">Errors &amp; Warnings</div>
        <span class="err-count" id="errCount" style="display:none"></span>
      </div>
      <div id="errWrap"><div class="empty">No errors</div></div>
    </div>
  </div>

  <div class="card">
    <div class="card-title">Open Positions</div>
    <div id="positionsWrap"><div class="empty">Loading…</div></div>
  </div>

  <div class="card">
    <div class="card-title">Recent Trades</div>
    <div id="tradesWrap"><div class="empty">Loading…</div></div>
  </div>

  <div class="card">
    <div class="card-title-row">
      <div class="card-title">Bot Log</div>
      <div class="log-controls">
        <label><input type="checkbox" id="logAutoScroll" checked> Auto-scroll</label>
        <select id="logLines" onchange="fetchLog()">
          <option value="100">100 lines</option>
          <option value="200" selected>200 lines</option>
          <option value="500">500 lines</option>
        </select>
      </div>
    </div>
    <div class="log-wrap" id="logWrap"><span class="log-line">Loading…</span></div>
  </div>

  <div class="refresh-bar">Auto-refreshes every 15 s &nbsp;·&nbsp; <span id="lastUpdated"></span></div>
</main>

<script>
const fmt  = n => new Intl.NumberFormat("en-US",{style:"currency",currency:"USD",maximumFractionDigits:2}).format(n);
const fmtP = n => (n>=0?"+":"")+n.toFixed(2)+"%";
const esc  = s => (s||"").replace(/&/g,"&amp;").replace(/</g,"&lt;").replace(/>/g,"&gt;");
function colorClass(n) { return n > 0 ? "pos" : n < 0 ? "neg" : "neu"; }
function stratPill(s) { return `<span class="strategy-pill">${esc(s||"lotto")}</span>`; }

function renderPositions(positions) {
  if (!positions.length) return '<div class="empty">No open positions</div>';
  let html = '<div style="overflow-x:auto"><table><thead><tr>'
    + '<th>Symbol</th><th>Strategy</th><th>Qty</th><th>Entry</th><th>Current</th><th>Stop</th><th>Unreal. P&L</th>'
    + '</tr></thead><tbody>';
  for (const p of positions) {
    if (p.error) { html += `<tr><td colspan="7" class="neg">${esc(p.error)}</td></tr>`; continue; }
    const cls = colorClass(p.unrealized_pl);
    const stop = p.stop_price != null ? `<span style="color:var(--yellow)">${fmt(p.stop_price)}</span>` : '<span style="color:var(--muted)">—</span>';
    html += `<tr>
      <td style="font-weight:600;font-size:12px">${esc(p.symbol)}</td>
      <td>${stratPill(p.strategy)}</td>
      <td>${p.qty}</td>
      <td>${fmt(p.entry_price)}</td>
      <td>${fmt(p.current_price)}</td>
      <td>${stop}</td>
      <td class="${cls}">${fmt(p.unrealized_pl)}<br><span style="font-size:11px">${fmtP(p.unrealized_pl_pct)}</span></td>
    </tr>`;
  }
  return html + '</tbody></table></div>';
}

function renderTrades(trades) {
  if (!trades.length) return '<div class="empty">No trades yet</div>';
  let html = '<div style="overflow-x:auto"><table><thead><tr>'
    + '<th>Time</th><th>Ticker</th><th>Event</th><th>Result</th><th>Detail</th>'
    + '</tr></thead><tbody>';
  for (const t of trades) {
    const ts = t.timestamp ? new Date(t.timestamp).toLocaleString([],{month:"numeric",day:"numeric",hour:"2-digit",minute:"2-digit"}) : "";
    const ticker = t.source_ticker || t.underlying || t.symbol || "—";
    let event, result, detail;
    if (t.kind === "close") {
      const pnl = parseFloat(t.pnl_usd), pct = parseFloat(t.pnl_pct);
      const cls = colorClass(isNaN(pnl) ? 0 : pnl);
      event = '<span style="font-weight:700;font-size:10px;letter-spacing:.5px;color:#cbd5e1;background:#374151;padding:2px 6px;border-radius:4px">CLOSED</span>';
      result = `<span class="${cls}">${isNaN(pnl)?"—":fmt(pnl)}${isNaN(pct)?"":` <span style="font-size:11px">(${pct>=0?"+":""}${pct.toFixed(1)}%)</span>`}</span>`;
      detail = `<span style="font-size:12px;color:var(--muted)">⮐ ${esc(t.reason_label || t.reason || "—")}</span>`;
    } else {
      const cost = t.cost_basis ? fmt(parseFloat(t.cost_basis)) : "—";
      const conf = t.confidence ? (parseFloat(t.confidence)*100).toFixed(0)+"%" : "";
      event = (t.still_open === false)
        ? '<span style="font-weight:700;font-size:10px;letter-spacing:.5px;color:var(--muted);background:#374151;padding:2px 6px;border-radius:4px">OPENED</span>'
        : '<span style="font-weight:700;font-size:10px;letter-spacing:.5px;color:#7db3ff;background:#1e3a5f;padding:2px 6px;border-radius:4px">OPEN</span>';
      result = `${cost}${conf?` <span style="font-size:11px;color:var(--muted)">${conf}</span>`:""}`;
      detail = `<span style="font-size:11px;color:var(--muted)">${esc((t.reasoning||"").slice(0,60))}</span>`;
    }
    html += `<tr><td style="white-space:nowrap;color:var(--muted)">${ts}</td><td style="font-weight:600">${esc(ticker)}</td><td>${event}</td><td>${result}</td><td>${detail}</td></tr>`;
  }
  return html + '</tbody></table></div>';
}

function renderFeed(items) {
  if (!items.length) return '<div class="empty">No activity yet</div>';
  return items.map(item => {
    let cls = "";
    if (item.kind === "signal") cls = item.direction || "";
    if (item.kind === "trade") cls = "trade";
    if (item.kind === "closed") cls = "closed";
    if (item.kind === "cutoff") cls = "cutoff";
    if (item.kind === "switch") cls = "switch";
    return `<div class="feed-item"><span class="feed-ts">${item.ts}</span><span class="feed-icon">${item.icon}</span><span class="feed-text ${cls}">${esc(item.text)}</span></div>`;
  }).join("");
}

function renderErrors(items) {
  if (!items.length) return '<div class="empty">No errors</div>';
  return items.map(e => `<div class="err-item"><span class="err-ts">${e.ts}</span><span class="err-level ${e.level}">${e.level}</span><span class="err-msg">${esc(e.msg)}</span></div>`).join("");
}

let perfChart = null;
async function fetchChart() {
  try {
    const res = await fetch("/api/chart");
    const data = await res.json();
    const ctx = document.getElementById("perfChart").getContext("2d");
    const datasets = [];
    const colors = { Portfolio: "#6366f1", SPY: "#64748b" };
    for (const [name, values] of Object.entries(data.series || {})) {
      datasets.push({ label: name, data: values, borderColor: colors[name] || "#94a3b8",
                      backgroundColor: "transparent", borderWidth: name === "Portfolio" ? 2.5 : 1.5,
                      pointRadius: 0, tension: 0.15, spanGaps: true });
    }
    if (perfChart) perfChart.destroy();
    perfChart = new Chart(ctx, {
      type: "line",
      data: { labels: data.labels || [], datasets },
      options: {
        responsive: true, maintainAspectRatio: false,
        interaction: { mode: "index", intersect: false },
        scales: {
          x: { ticks: { color: "#64748b", maxTicksLimit: 8 }, grid: { color: "#2a2d3a" } },
          y: { ticks: { color: "#64748b", callback: v => fmt(v) }, grid: { color: "#2a2d3a" } },
        },
        plugins: { legend: { labels: { color: "#e2e8f0" } },
                   tooltip: { callbacks: { label: c => `${c.dataset.label}: ${fmt(c.parsed.y)}` } } },
      },
    });
  } catch (e) { /* chart is best-effort */ }
}

async function fetchLog() {
  const n = document.getElementById("logLines").value;
  const res = await fetch(`/api/log?lines=${n}`);
  const data = await res.json();
  const wrap = document.getElementById("logWrap");
  const auto = document.getElementById("logAutoScroll").checked;
  wrap.innerHTML = (data.activity || []).slice().reverse()
    .map(a => `<div class="log-line INFO">${a.ts}  ${esc(a.icon)} ${esc(a.text)}</div>`).join("")
    + (data.errors || []).slice().reverse()
      .map(e => `<div class="log-line ${e.level}">${e.ts}  ${e.level}  ${esc(e.msg)}</div>`).join("");
  if (auto) wrap.scrollTop = wrap.scrollHeight;
}

async function refresh() {
  try {
    const res = await fetch("/api/status");
    const data = await res.json();

    document.getElementById("statusDot").className = "dot " + (data.bot_running ? "green" : "red");
    document.getElementById("statusText").textContent = data.bot_running ? "Bot running" : "Bot stopped";

    const a = data.account || {};
    if (a.equity !== undefined) {
      const pnl = a.pnl_today || 0;
      const pnlEl = document.getElementById("pnlToday");
      pnlEl.textContent = fmt(pnl); pnlEl.className = "big-num " + colorClass(pnl);
      document.getElementById("pnlTodayPct").textContent = fmtP(a.pnl_today_pct || 0);
      document.getElementById("equity").textContent = fmt(a.equity);
      document.getElementById("cash").textContent = "Cash: " + fmt(a.cash);
    }

    const b = data.benchmark || {};
    const setBench = (id, v) => {
      const el = document.getElementById(id);
      if (v === undefined || v === null) { el.textContent = "—"; return; }
      el.textContent = fmtP(v); el.className = colorClass(v);
    };
    setBench("bTodayAcct", b.today_account); setBench("bTodaySpy", b.today_spy); setBench("bTodayAlpha", b.today_alpha);
    setBench("bPerAcct", b.period_account); setBench("bPerSpy", b.period_spy); setBench("bPerAlpha", b.period_alpha);

    const c = (data.log || {}).counters || {};
    const s = data.stats || {};
    document.getElementById("cntScored").textContent = c.scored ?? 0;
    document.getElementById("cntBullish").textContent = `${c.bullish ?? 0} bullish · ${c.cutoff_skips ?? 0} cutoff-skipped`;
    document.getElementById("statToday").textContent = s.today_trades ?? 0;
    document.getElementById("statTotal").textContent = `${s.total_trades ?? 0} all-time`;
    document.getElementById("statWinRate").textContent = s.win_rate != null ? s.win_rate + "%" : "—";
    document.getElementById("statRealized").textContent = `${fmt(s.total_realized ?? 0)} realized (${s.total_closed ?? 0} closed)`;

    document.getElementById("feedWrap").innerHTML = renderFeed((data.log || {}).activity || []);
    const errs = (data.log || {}).errors || [];
    document.getElementById("errWrap").innerHTML = renderErrors(errs);
    const errCount = document.getElementById("errCount");
    if (errs.length) { errCount.style.display = "inline-block"; errCount.textContent = errs.length; }
    else { errCount.style.display = "none"; }

    document.getElementById("positionsWrap").innerHTML = renderPositions(data.positions || []);
    document.getElementById("tradesWrap").innerHTML = renderTrades(data.trades || []);

    document.getElementById("lastUpdated").textContent = "Updated " + new Date().toLocaleTimeString();
  } catch (e) {
    document.getElementById("statusText").textContent = "Dashboard error";
  }
}

refresh(); fetchChart(); fetchLog();
setInterval(refresh, 15000);
setInterval(fetchChart, 60000);
setInterval(fetchLog, 15000);
</script>
</body>
</html>"""


@app.route("/")
def index():
    return render_template_string(HTML)


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=PORT, debug=False)
