"""
dashboard.py — local web dashboard for the trading bot.

Run alongside bot.py:
    .venv/bin/python dashboard.py

Then open http://localhost:5001 (or your LAN IP on port 5001 from phone).

To expose on the internet later:
    ngrok http 5001
  or
    cloudflared tunnel --url http://localhost:5001
"""

import csv
import json
import os
import re
import subprocess
from collections import defaultdict
from datetime import datetime, timezone, date, timedelta
import time as _time
from pathlib import Path

from flask import Flask, jsonify, render_template_string, request
from alpaca.trading.client import TradingClient
from alpaca.trading.requests import GetPortfolioHistoryRequest
from alpaca.data.historical import StockHistoricalDataClient
from alpaca.data.requests import StockBarsRequest
from alpaca.data.timeframe import TimeFrame
from dotenv import load_dotenv
import router_eval   # shared shadow-router attribution logic
import config        # for LOG_DIR (logs live outside the data/ CWD)

load_dotenv()

app = Flask(__name__)

# Data files (CSVs, state) resolve via the daemon's CWD=data/; the bot LOG lives in logs/.
TRADES_CSV         = Path("trades.csv")
CLOSED_TRADES_CSV  = Path("closed_trades.csv")
ROUTER_DECISIONS_CSV = Path("router_decisions.csv")
BOT_LOG            = config.LOG_DIR / "bot.log"
BOT_STATE_FILE     = Path("bot_state.json")
BOT_PROCESS        = "bot.py"
PORT               = int(os.environ.get("DASHBOARD_PORT", 5001))

STRATEGY_COLORS = {
    "news_call":  "#22c55e",
    "qqq_macro":  "#3b82f6",
    "stock":      "#93c5fd",
    "pead":       "#f59e0b",
    "bear_short": "#ef4444",
    "lotto":      "#ec4899",
    "pairs_long": "#a855f7",
    "pairs_short":"#c084fc",
    "router":     "#14b8a6",
    "unknown":    "#64748b",
}

trading_client = TradingClient(
    os.environ["ALPACA_API_KEY"],
    os.environ["ALPACA_SECRET_KEY"],
    paper=True,
)

# Market-data client for the SPY benchmark (same keys; read-only daily bars).
_data_client = StockHistoricalDataClient(
    os.environ["ALPACA_API_KEY"], os.environ["ALPACA_SECRET_KEY"]
)
# Benchmark is cached — daily bars + portfolio history move slowly, and this saves
# 2 Alpaca calls on every status poll.
_bench_cache = {"ts": 0.0, "data": {}}
_BENCH_TTL = 300   # seconds

# ── helpers ──────────────────────────────────────────────────────────────────


def _period_for(baseline_date) -> str:
    dd = (date.today() - baseline_date).days
    if dd <= 27:  return "1M"
    if dd <= 88:  return "3M"
    if dd <= 360: return "1A"
    return "all"


def _cost_basis(row) -> float:
    """Capital deployed by a trade = qty × entry_price × (100 for options, 1 for stock)."""
    try:
        q = float(row.get("qty", 0) or 0)
        ep = float(row.get("entry_price", 0) or 0)
        mult = 100 if "option" in (row.get("asset_type", "") or "").lower() else 1
        return abs(q * ep * mult)
    except Exception:
        return 0.0


def _ffill_grid(days, by_date, pre=None):
    """Align a {date: value} map onto the daily `days` grid, carrying the last known value
    forward. Cells before the first known value get `pre`."""
    out, last = [], pre
    for d in days:
        if d in by_date:
            last = by_date[d]
        out.append(last)
    return out


def _router_cum_by_date(baseline_date):
    """Cumulative router-attributed realized P&L by date (router's per-signal pick → the matching
    actual closed trade, greedy/time-ordered). Returns {date: cumulative_pnl} for trades closed
    on/after baseline. Reuses router_eval's loaders + claim semantics."""
    try:
        decisions = sorted(router_eval.load_decisions(), key=lambda d: (d.get("_ts") or datetime.min.replace(tzinfo=timezone.utc)))
        rows = router_eval.load_closed()
    except Exception:
        return {}, 0.0
    used, cum, cap, by_date = set(), 0.0, 0.0, {}
    def claim_row(strategy, underlying, after):
        best_i, best = None, None
        for i, c in enumerate(rows):
            if i in used or c.get("strategy") != strategy or c.get("underlying") != underlying:
                continue
            if after and c["_ts"] and c["_ts"] < after:
                continue
            if best is None or (c["_ts"] and best["_ts"] and c["_ts"] < best["_ts"]):
                best, best_i = c, i
        if best_i is not None:
            used.add(best_i)
            return best
        return None
    claimed = []
    for d in decisions:
        choice, tk, after = d.get("router_choice"), d.get("ticker"), d.get("_ts")
        got = []
        if choice == "pairs":
            for s, u in (("pairs_long", tk), ("pairs_short", d.get("pair_partner"))):
                r = claim_row(s, u, after)
                if r: got.append(r)
        else:
            r = claim_row(choice, tk, after)
            if r: got.append(r)
        for r in got:
            if r["_ts"]:
                claimed.append((r["_ts"].date(), r["_pnl"], _cost_basis(r)))
    for dt, pnl, cb in sorted(claimed, key=lambda x: x[0]):
        if dt <= baseline_date:   # measure from baseline-close → shared zero origin with other lines
            continue
        cum += pnl
        cap += cb
        by_date[dt] = cum
    return by_date, cap


def get_chart_data(baseline_date) -> dict:
    """Cumulative $ P&L since `baseline_date` for: total Portfolio, each strategy, the shadow
    Router, and SPY/QQQ benchmarks (sized to the portfolio's baseline equity → comparable $).
    All series are rebased to $0 at the baseline and sampled on a common daily grid, so a deploy
    that changes performance shows up as a slope change. Fully guarded."""
    today = date.today()
    if baseline_date > today:
        baseline_date = today
    days = [baseline_date + timedelta(days=i) for i in range((today - baseline_date).days + 1)]
    labels = [d.isoformat() for d in days]
    series, base_eq, capital = {}, None, {}
    try:
        # ── total Portfolio (Alpaca equity curve), rebased to baseline ──
        ph = trading_client.get_portfolio_history(
            GetPortfolioHistoryRequest(period=_period_for(baseline_date), timeframe="1D"))
        eq_by_date = {}
        for t, e in zip(ph.timestamp or [], ph.equity or []):
            if e:
                eq_by_date[datetime.fromtimestamp(int(t), tz=timezone.utc).date()] = float(e)
        if eq_by_date:
            sorted_eq = sorted(eq_by_date.items())
            prior = [v for dt, v in sorted_eq if dt <= baseline_date]
            base_eq = prior[-1] if prior else sorted_eq[0][1]
            eq_grid = _ffill_grid(days, eq_by_date, pre=base_eq)
            series["Portfolio"] = [round(v - base_eq, 2) if v is not None else None for v in eq_grid]

        # ── SPY / QQQ benchmarks: $ P&L if the baseline portfolio value were in the index ──
        if base_eq:
            for sym in ("SPY", "QQQ"):
                bars = _data_client.get_stock_bars(StockBarsRequest(
                    symbol_or_symbols=sym, timeframe=TimeFrame.Day,
                    start=datetime.combine(baseline_date, datetime.min.time(), tzinfo=timezone.utc) - timedelta(days=6)
                )).data.get(sym, [])
                px_by_date = {b.timestamp.date(): float(b.close) for b in bars}
                if px_by_date:
                    sp = sorted(px_by_date.items())
                    prior_px = [v for dt, v in sp if dt <= baseline_date]
                    base_px = prior_px[-1] if prior_px else sp[0][1]
                    px_grid = _ffill_grid(days, px_by_date, pre=base_px)
                    series[sym] = [round(base_eq * (v / base_px - 1), 2) if v is not None else None for v in px_grid]

        # ── per-strategy cumulative realized P&L + deployed capital (cost basis) ──
        strat_events = defaultdict(list)
        strat_capital = defaultdict(float)
        if CLOSED_TRADES_CSV.exists():
            with open(CLOSED_TRADES_CSV, newline="") as f:
                for row in csv.DictReader(f):
                    ts = router_eval._ts(row.get("timestamp"))
                    # Count trades AFTER the baseline day's close, so every line shares the same
                    # zero origin at the baseline (Portfolio/SPY/QQQ rebase to baseline-close too).
                    if not ts or ts.date() <= baseline_date:
                        continue
                    strat = row.get("strategy", "unknown")
                    strat_events[strat].append((ts.date(), float(row.get("pnl_usd", 0) or 0)))
                    strat_capital[strat] += _cost_basis(row)
        for strat, evs in strat_events.items():
            cum, cum_by_date = 0.0, {}
            for dt, pnl in sorted(evs):
                cum += pnl
                cum_by_date[dt] = cum
            series[strat] = [round(v, 2) for v in _ffill_grid(days, cum_by_date, pre=0.0)]
            if strat_capital[strat] > 0:
                capital[strat] = round(strat_capital[strat], 2)

        # ── shadow Router cumulative attributed P&L + deployed capital ──
        rcum, rcap = _router_cum_by_date(baseline_date)
        if rcum:
            series["router"] = [round(v, 2) for v in _ffill_grid(days, rcum, pre=0.0)]
            if rcap > 0:
                capital["router"] = round(rcap, 2)
    except Exception as e:
        return {"labels": labels, "series": series, "baseline": baseline_date.isoformat(),
                "base_equity": base_eq, "capital": capital, "error": str(e)}
    return {"labels": labels, "series": series, "baseline": baseline_date.isoformat(),
            "base_equity": base_eq, "capital": capital}


def get_benchmark(account: dict) -> dict:
    """Strategy performance vs just being long the market (SPY), so the user can see whether the
    news-driven strategies beat or trail a passive index. Two date-aligned windows: the most-recent
    session ('today') and the trailing ~1 month. Account return from Alpaca portfolio history;
    SPY return from daily bars over the SAME span. Cached 5 min; fully guarded → returns
    {"error": ...} (or {}) on any failure so the dashboard never breaks."""
    if _time.time() - _bench_cache["ts"] < _BENCH_TTL and _bench_cache["data"]:
        return _bench_cache["data"]
    d = {}
    try:
        ph = trading_client.get_portfolio_history(
            GetPortfolioHistoryRequest(period="1M", timeframe="1D"))
        pts = [(t, e) for t, e in zip(ph.timestamp or [], ph.equity or []) if e]
        closes = []
        if len(pts) >= 2:
            start_date = datetime.fromtimestamp(int(pts[0][0]), tz=timezone.utc).date()
            bars = _data_client.get_stock_bars(StockBarsRequest(
                symbol_or_symbols="SPY", timeframe=TimeFrame.Day,
                start=datetime.combine(start_date, datetime.min.time(), tzinfo=timezone.utc) - timedelta(days=6)
            )).data.get("SPY", [])
            sbars = [(b.timestamp.date(), float(b.close)) for b in bars]
            # Align SPY's start to the account window's first date (first SPY bar on/after it)
            on_after = [c for dte, c in sbars if dte >= start_date]
            closes = on_after if len(on_after) >= 2 else [c for _, c in sbars]
        if len(pts) >= 2 and len(closes) >= 2:
            acct_p     = (pts[-1][1] / pts[0][1] - 1) * 100
            spy_p      = (closes[-1] / closes[0] - 1) * 100
            spy_today  = (closes[-1] / closes[-2] - 1) * 100
            acct_today = float(account.get("pnl_today_pct", 0.0) or 0.0)
            d = {
                "period_label":   "1M",
                "period_account": round(acct_p, 2),
                "period_spy":     round(spy_p, 2),
                "period_alpha":   round(acct_p - spy_p, 2),
                "today_account":  round(acct_today, 2),
                "today_spy":      round(spy_today, 2),
                "today_alpha":    round(acct_today - spy_today, 2),
            }
            _bench_cache.update(ts=_time.time(), data=d)   # cache only on success
    except Exception as e:
        d = {"error": str(e)}
    return d



def bot_is_running() -> bool:
    try:
        out = subprocess.check_output(["pgrep", "-f", BOT_PROCESS], text=True).strip()
        return bool(out)
    except subprocess.CalledProcessError:
        return False


def get_account() -> dict:
    try:
        a = trading_client.get_account()
        return {
            "equity":        float(a.equity),
            "cash":          float(a.cash),
            "buying_power":  float(a.buying_power),
            "pnl_today":     float(a.equity) - float(a.last_equity),
            "pnl_today_pct": (float(a.equity) - float(a.last_equity))
                              / float(a.last_equity) * 100
                              if float(a.last_equity) else 0,
        }
    except Exception as e:
        return {"error": str(e)}


def _load_bot_state() -> dict:
    try:
        return json.loads(BOT_STATE_FILE.read_text()) if BOT_STATE_FILE.exists() else {}
    except Exception:
        return {}


def get_positions() -> list[dict]:
    bot_state = _load_bot_state()
    # Build a lookup: underlying ticker → list of (key, state) for stock strategies
    stock_states: dict[str, list[dict]] = defaultdict(list)
    for key, info in bot_state.items():
        if info.get("asset_type") in ("stock", "stock_short"):
            ticker = info.get("underlying", key.split("__")[0])
            stock_states[ticker].append({"key": key, **info})

    try:
        positions = trading_client.get_all_positions()
        result = []
        seen_tickers: set[str] = set()

        for p in positions:
            asset_class = str(p.asset_class).lower().replace("assetclass.", "")
            if asset_class == "us_equity":
                # Expand stock positions into per-strategy rows using bot_state.
                # Alpaca reports short positions as negative qty.
                ticker  = p.symbol
                infos   = stock_states.get(ticker, [])
                current = float(p.current_price or 0)
                raw_qty = float(p.qty)
                is_short = raw_qty < 0
                if infos:
                    for info in infos:
                        entry   = info.get("entry_price", float(p.avg_entry_price))
                        qty     = abs(info.get("qty", raw_qty))
                        strat   = info.get("strategy", "stock")
                        is_sh   = info.get("asset_type") == "stock_short"
                        unreal  = (entry - current) * qty if is_sh else (current - entry) * qty
                        unreal_pct = (unreal / (entry * qty) * 100) if entry else 0
                        result.append({
                            "symbol":            ticker,
                            "side":              "short" if is_sh else "long",
                            "asset_type":        "stock_short" if is_sh else "stock",
                            "strategy":          strat,
                            "qty":               qty,
                            "entry_price":       entry,
                            "current_price":     current,
                            "market_value":      current * qty,
                            "unrealized_pl":     unreal,
                            "unrealized_pl_pct": unreal_pct,
                            "stop_price":        info.get("stop_price"),
                        })
                    seen_tickers.add(ticker)
                else:
                    # Stock not in bot_state (manual trade or pre-PEAD position)
                    state  = bot_state.get(ticker, {})
                    result.append({
                        "symbol":            ticker,
                        "side":              "short" if is_short else "long",
                        "asset_type":        "stock",
                        "strategy":          state.get("strategy", "—"),
                        "qty":               abs(raw_qty),
                        "entry_price":       float(p.avg_entry_price),
                        "current_price":     current,
                        "market_value":      float(p.market_value or 0),
                        "unrealized_pl":     float(p.unrealized_pl or 0),
                        "unrealized_pl_pct": float(p.unrealized_plpc or 0) * 100,
                        "stop_price":        state.get("stop_price"),
                    })
            else:
                # Options: look up by OCC symbol
                state = bot_state.get(p.symbol, {})
                result.append({
                    "symbol":            p.symbol,
                    "side":              "short" if float(p.qty) < 0 else "long",
                    "asset_type":        asset_class,
                    "strategy":          state.get("strategy", "news_call"),
                    "qty":               float(p.qty),
                    "entry_price":       float(p.avg_entry_price),
                    "current_price":     float(p.current_price or 0),
                    "market_value":      float(p.market_value or 0),
                    "unrealized_pl":     float(p.unrealized_pl or 0),
                    "unrealized_pl_pct": float(p.unrealized_plpc or 0) * 100,
                    "stop_price":        state.get("stop_price"),
                })
        result.sort(key=lambda x: x["unrealized_pl"], reverse=True)
        return result
    except Exception as e:
        return [{"error": str(e)}]


# closed_trades.csv `reason` → human-friendly close mechanism shown in the dashboard.
CLOSE_REASON_LABELS = {
    "time-stop":        "Max hold",
    "stop":             "Trailing stop",
    "lotto_cap":        "Take-profit cap",
    "oversize-cleanup": "Risk trim",
}


def _close_reason_label(reason: str) -> str:
    r = (reason or "").strip()
    if r in CLOSE_REASON_LABELS:
        return CLOSE_REASON_LABELS[r]
    rl = r.lower()
    if "lotto" in rl or "cap" in rl or "profit" in rl:
        return "Take-profit cap"
    if "time" in rl or "hold" in rl or "eod" in rl:
        return "Max hold"
    if "stop" in rl or "trail" in rl:
        return "Trailing stop"
    return r or "—"


def get_recent_trades(n: int = 30) -> list[dict]:
    """Unified, time-sorted feed of position OPENS (trades.csv) and CLOSES (closed_trades.csv).
    Each row carries kind ∈ {open, close}; closes include realized P&L + the close mechanism."""
    events: list[dict] = []

    def _read(path, kind):
        if not path.exists():
            return
        try:
            with open(path, newline="") as f:
                for r in csv.DictReader(f):
                    row = {k: (", ".join(v) if isinstance(v, list) else (v or ""))
                           for k, v in r.items() if k is not None}
                    row["kind"] = kind
                    if kind == "close":
                        row["reason_label"] = _close_reason_label(row.get("reason", ""))
                    events.append(row)
        except Exception:
            pass

    _read(TRADES_CSV, "open")
    _read(CLOSED_TRADES_CSV, "close")
    events.sort(key=lambda e: e.get("timestamp", ""), reverse=True)
    return events[:n]


def get_stats() -> dict:
    if not TRADES_CSV.exists():
        return {}
    try:
        with open(TRADES_CSV, newline="") as f:
            rows = list(csv.DictReader(f))
        today_s = date.today().isoformat()
        today   = [r for r in rows if r.get("timestamp", "").startswith(today_s)]
        return {
            "total_trades": len(rows),
            "today_trades": len(today),
        }
    except Exception:
        return {}


# Log line patterns
_RE_TS       = re.compile(r"^(\d{2}:\d{2}:\d{2})\s+(INFO|WARNING|ERROR|CRITICAL)\s+(.+)$")
_RE_ARTICLE  = re.compile(r"📰 \[(.+?)\] (.+)")
_RE_LLM      = re.compile(r"🤖 \S+: (\w+)\s+conf=([\d.]+)\s+mag=([\d.]+)\s+tickers=(\[.*?\])")
_RE_SKIP     = re.compile(r"→ (.+)")
_RE_TRADE    = re.compile(r"🛒|Placing (option|crypto) trade|📝 Placed")
_RE_CLOSED   = re.compile(r"🔴 Closed|trailing stop triggered|stop hit")
_RE_POSITION = re.compile(r"👁\s+(\S+)\s+entry=\$[\d.]+\s+mid=\$[\d.]+\s+peak=\$[\d.]+\s+stop=\$[\d.]+\s+P&L: ([+\-][\d.]+%)")
_RE_SIZING   = re.compile(r"💰 Position sizing")


def parse_bot_log(tail_lines: int = 500) -> dict:
    """
    Read the last `tail_lines` lines of bot.log and return structured data:
      - activity:  list of recent log events for the feed (newest first)
      - errors:    recent WARNING/ERROR lines (newest first)
      - counters:  articles seen, signals fired, trades placed today
    """
    if not BOT_LOG.exists():
        return {"activity": [], "errors": [], "counters": {}}

    try:
        with open(BOT_LOG, "r", errors="replace") as f:
            lines = f.readlines()
    except Exception:
        return {"activity": [], "errors": [], "counters": {}}

    lines = lines[-tail_lines:]
    today_prefix = date.today().strftime("%Y-%m-%d")

    activity = []
    errors   = []
    counters = defaultdict(int)

    for line in lines:
        line = line.rstrip()
        m = _RE_TS.match(line)
        if not m:
            continue
        ts, level, msg = m.group(1), m.group(2), m.group(3).strip()

        # Errors / warnings
        if level in ("WARNING", "ERROR", "CRITICAL"):
            # Skip the noisy stop-order-not-eligible warning — shown in positions
            if "not eligible to trade uncovered option contracts" in msg:
                continue
            errors.append({"ts": ts, "level": level, "msg": msg})
            counters["errors"] += 1

        # Article seen
        am = _RE_ARTICLE.search(msg)
        if am:
            counters["articles"] += 1
            activity.append({
                "ts":   ts,
                "kind": "article",
                "icon": "📰",
                "text": f"[{am.group(1)}] {am.group(2)}",
            })
            continue

        # LLM decision
        lm = _RE_LLM.search(msg)
        if lm:
            direction, conf, mag = lm.group(1), lm.group(2), lm.group(3)
            counters["scored"] += 1
            if direction == "bullish":
                counters["bullish"] += 1
            activity.append({
                "ts":   ts,
                "kind": "signal",
                "icon": "🤖",
                "text": f"{direction}  conf={float(conf):.0%}  mag={mag}",
                "direction": direction,
            })
            continue

        # Position sizing → trade firing
        if _RE_SIZING.search(msg):
            counters["trades_placed"] += 1
            activity.append({
                "ts":   ts,
                "kind": "trade",
                "icon": "💰",
                "text": msg,
            })
            continue

        # Closed position
        if _RE_CLOSED.search(msg):
            counters["closed"] += 1
            activity.append({
                "ts":   ts,
                "kind": "closed",
                "icon": "🔴",
                "text": msg,
            })
            continue

    activity.reverse()
    errors.reverse()

    return {
        "activity": activity[:60],
        "errors":   errors[:30],
        "counters": dict(counters),
    }


# ── API routes ────────────────────────────────────────────────────────────────

@app.route("/api/log")
def api_log():
    n = int(request.args.get("lines", 200))
    if not BOT_LOG.exists():
        return jsonify({"lines": []})
    try:
        with open(BOT_LOG, "r", errors="replace") as f:
            lines = f.readlines()
        return jsonify({"lines": [l.rstrip() for l in lines[-n:]]})
    except Exception as e:
        return jsonify({"lines": [], "error": str(e)})


def get_strategy_pnl() -> dict:
    """
    Per-strategy P&L summary.
    Realized: from closed_trades.csv (logged by bot on each confirmed close).
    Unrealized: entry_price from bot_state.json × current Alpaca price.
    """
    summary: dict[str, dict] = {}
    for s in ("news_call", "qqq_macro", "stock", "pead", "bear_short", "lotto",
              "pairs_long", "pairs_short"):
        summary[s] = {"realized": 0.0, "unrealized": 0.0, "closed": 0,
                      "open": 0, "wins": 0, "losses": 0}

    # Realized from closed_trades.csv (also collect normalized rows for router attribution)
    closed_norm: list[dict] = []
    if CLOSED_TRADES_CSV.exists():
        try:
            with open(CLOSED_TRADES_CSV, newline="") as f:
                for row in csv.DictReader(f):
                    strat = row.get("strategy", "unknown")
                    pnl   = float(row.get("pnl_usd", 0) or 0)
                    if strat not in summary:
                        summary[strat] = {"realized": 0, "unrealized": 0,
                                          "closed": 0, "open": 0, "wins": 0, "losses": 0}
                    summary[strat]["realized"] += pnl
                    summary[strat]["closed"]   += 1
                    if pnl > 0: summary[strat]["wins"]   += 1
                    else:       summary[strat]["losses"] += 1
                    closed_norm.append({"strategy": strat,
                                        "underlying": row.get("underlying", ""),
                                        "_ts": router_eval._ts(row.get("timestamp")),
                                        "_pnl": pnl})
        except Exception:
            pass

    # Unrealized: use Alpaca's ACTUAL unrealized P&L per position (the account truth),
    # split across the strategies that share a position (e.g. stock+pead on one ticker)
    # in proportion to each leg's qty. This makes the panel's unrealized total tie to
    # the account's real mark-to-market instead of a recomputed entry×price estimate
    # (which used the bot's recorded entry, not the actual fill → didn't match the top).
    bot_state = _load_bot_state()
    try:
        positions = trading_client.get_all_positions()
    except Exception:
        positions = []
    alp_unreal = {p.symbol: float(p.unrealized_pl or 0) for p in positions}
    alp_px     = {p.symbol: float(p.current_price or 0) for p in positions}

    # Group bot_state legs by their Alpaca-facing symbol (ticker for stock/short, OCC for option).
    groups: dict[str, list[dict]] = {}
    for key, info in bot_state.items():
        if not float(info.get("entry_price", 0) or 0):
            continue
        at  = info.get("asset_type", "option")
        und = info.get("underlying", key.split("__")[0])
        sym = und if at in ("stock", "stock_short") else key
        groups.setdefault(sym, []).append(info)

    open_norm: list[dict] = []
    for sym, legs in groups.items():
        total_qty = sum(float(l.get("qty", 1) or 0) for l in legs) or 1.0
        actual    = alp_unreal.get(sym)        # Alpaca's true unrealized for this position
        for info in legs:
            strat = info.get("strategy", "unknown")
            qty   = float(info.get("qty", 1) or 0)
            if actual is not None:
                unreal = actual * (qty / total_qty)        # split the real P&L by qty share
            else:
                # Alpaca has no row for it (timing skew) → fall back to a recomputed mark.
                at = info.get("asset_type", "option"); entry = float(info.get("entry_price", 0))
                cur = alp_px.get(sym, 0)
                if not cur:               unreal = 0.0
                elif at == "stock_short": unreal = (entry - cur) * qty
                elif at == "stock":       unreal = (cur - entry) * qty
                else:                     unreal = (cur - entry) * 100 * qty
            if strat not in summary:
                summary[strat] = {"realized": 0, "unrealized": 0,
                                  "closed": 0, "open": 0, "wins": 0, "losses": 0}
            summary[strat]["unrealized"] += unreal
            summary[strat]["open"]       += 1
            open_norm.append({"strategy": strat,
                              "underlying": info.get("underlying", sym),
                              "_ts": None, "_pnl": unreal})

    # ── Shadow router: attribute the bot's real trades to the router's picks ──
    # Realized first (claims closed trades), then the still-open picks against
    # current open positions. No double-counting — each trade claimed once.
    try:
        if ROUTER_DECISIONS_CSV.exists():
            decisions = router_eval.load_decisions()  # reads ROUTER_DECISIONS_CSV
            r_real, flags_r, wins, losses = router_eval.attribute(decisions, closed_norm)
            unmatched = [d for d, f in zip(decisions, flags_r) if not f]
            r_unreal, flags_o, _, _ = router_eval.attribute(unmatched, open_norm)
            summary["router"] = {
                "realized": r_real, "unrealized": r_unreal,
                "closed": sum(flags_r), "open": sum(flags_o),
                "wins": wins, "losses": losses,
            }
    except Exception:
        pass

    return summary


@app.route("/api/strategy_pnl")
def api_strategy_pnl():
    return jsonify(get_strategy_pnl())


@app.route("/api/chart")
def api_chart():
    """Cumulative-P&L-since-baseline series for the performance chart. ?baseline=YYYY-MM-DD
    (default: 30 days ago)."""
    raw = request.args.get("baseline", "")
    try:
        baseline = datetime.fromisoformat(raw).date() if raw else (date.today() - timedelta(days=30))
    except Exception:
        baseline = date.today() - timedelta(days=30)
    return jsonify(get_chart_data(baseline))


@app.route("/api/status")
def api_status():
    account   = get_account()
    positions = get_positions()
    trades    = get_recent_trades(30)
    stats     = get_stats()
    log_data  = parse_bot_log()
    benchmark = get_benchmark(account)
    return jsonify({
        "bot_running":   bot_is_running(),
        "timestamp":     datetime.now(timezone.utc).isoformat(),
        "account":       account,
        "positions":     positions,
        "recent_trades": trades,
        "stats":         stats,
        "log":           log_data,
        "benchmark":     benchmark,
    })


# ── HTML ──────────────────────────────────────────────────────────────────────

HTML = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8"/>
<meta name="viewport" content="width=device-width, initial-scale=1"/>
<meta name="theme-color" content="#0f1117"/>
<title>Trader Bot</title>
<style>
  :root {
    --bg:      #0f1117;
    --surface: #1a1d27;
    --border:  #2a2d3a;
    --text:    #e2e8f0;
    --muted:   #64748b;
    --green:   #22c55e;
    --red:     #ef4444;
    --yellow:  #f59e0b;
    --blue:    #3b82f6;
    --accent:  #6366f1;
  }
  * { box-sizing: border-box; margin: 0; padding: 0; }
  body {
    background: var(--bg);
    color: var(--text);
    font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
    font-size: 14px;
    line-height: 1.5;
    min-height: 100vh;
  }
  header {
    background: var(--surface);
    border-bottom: 1px solid var(--border);
    padding: 14px 20px;
    display: flex;
    align-items: center;
    justify-content: space-between;
    position: sticky;
    top: 0;
    z-index: 10;
  }
  .logo { font-weight: 700; font-size: 16px; letter-spacing: .5px; }
  .status-pill {
    display: inline-flex;
    align-items: center;
    gap: 6px;
    font-size: 12px;
    font-weight: 600;
    padding: 4px 10px;
    border-radius: 20px;
    background: var(--border);
  }
  .dot { width: 7px; height: 7px; border-radius: 50%; background: var(--muted); }
  .dot.green  { background: var(--green); box-shadow: 0 0 6px var(--green); }
  .dot.red    { background: var(--red); }
  .dot.yellow { background: var(--yellow); animation: pulse 1.4s infinite; }
  @keyframes pulse { 0%,100%{opacity:1} 50%{opacity:.4} }

  main { padding: 16px; max-width: 960px; margin: 0 auto; }
  .grid-2 { display: grid; grid-template-columns: 1fr 1fr; gap: 12px; }
  .grid-4 { display: grid; grid-template-columns: repeat(4,1fr); gap: 12px; margin-bottom: 12px; }
  @media(max-width:640px) {
    .grid-2 { grid-template-columns: 1fr; }
    .grid-4 { grid-template-columns: repeat(2,1fr); }
  }

  .card {
    background: var(--surface);
    border: 1px solid var(--border);
    border-radius: 12px;
    padding: 16px;
    margin-bottom: 12px;
  }
  .card-title {
    font-size: 11px;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: .8px;
    color: var(--muted);
    margin-bottom: 12px;
  }
  .card-title-row {
    display: flex;
    align-items: center;
    justify-content: space-between;
    margin-bottom: 12px;
  }
  .card-title-row .card-title { margin-bottom: 0; }
  .err-count {
    font-size: 11px;
    font-weight: 700;
    background: var(--red);
    color: #fff;
    border-radius: 10px;
    padding: 1px 7px;
  }

  .mini-card {
    background: var(--surface);
    border: 1px solid var(--border);
    border-radius: 12px;
    padding: 14px 16px;
  }
  .mini-label { font-size: 11px; color: var(--muted); margin-bottom: 4px; }
  .mini-val   { font-size: 22px; font-weight: 700; line-height: 1; }
  .mini-sub   { font-size: 11px; color: var(--muted); margin-top: 3px; }

  .big-num { font-size: 28px; font-weight: 700; line-height: 1; }
  .sub { font-size: 12px; color: var(--muted); margin-top: 4px; }
  .pos { color: var(--green); }
  .neg { color: var(--red); }
  .neu { color: var(--text); }

  table { width: 100%; border-collapse: collapse; }
  th {
    text-align: left;
    font-size: 11px;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: .6px;
    color: var(--muted);
    padding: 6px 8px;
    border-bottom: 1px solid var(--border);
  }
  td { padding: 8px 8px; border-bottom: 1px solid var(--border); vertical-align: top; }
  tr:last-child td { border-bottom: none; }

  .badge {
    display: inline-block;
    font-size: 10px;
    font-weight: 700;
    padding: 2px 6px;
    border-radius: 4px;
    text-transform: uppercase;
  }
  .badge-opt       { background: #312e81; color: #a5b4fc; }
  .badge-crypto    { background: #451a03; color: #fbbf24; }
  .badge-us_equity { background: #1e3a5f; color: #93c5fd; }
  .badge-stock     { background: #1e3a5f; color: #93c5fd; }
  .strategy-pill {
    display: inline-block; font-size: 10px; font-weight: 700;
    padding: 2px 7px; border-radius: 10px; text-transform: uppercase;
  }

  /* Activity feed */
  .feed { display: flex; flex-direction: column; gap: 0; }
  .feed-item {
    display: flex;
    align-items: baseline;
    gap: 8px;
    padding: 7px 0;
    border-bottom: 1px solid var(--border);
    font-size: 13px;
  }
  .feed-item:last-child { border-bottom: none; }
  .feed-ts   { font-size: 11px; color: var(--muted); white-space: nowrap; flex-shrink: 0; }
  .feed-icon { flex-shrink: 0; }
  .feed-text { color: var(--text); word-break: break-word; }
  .feed-text.bullish { color: var(--green); }
  .feed-text.bearish { color: var(--red); }
  .feed-text.neutral { color: var(--muted); }
  .feed-text.trade   { color: var(--yellow); font-weight: 600; }
  .feed-text.closed  { color: var(--red); }

  /* Errors */
  .err-item {
    display: flex;
    gap: 8px;
    padding: 7px 0;
    border-bottom: 1px solid var(--border);
    font-size: 12px;
  }
  .err-item:last-child { border-bottom: none; }
  .err-ts    { color: var(--muted); white-space: nowrap; flex-shrink: 0; font-size: 11px; }
  .err-level { flex-shrink: 0; font-weight: 700; }
  .err-level.WARNING  { color: var(--yellow); }
  .err-level.ERROR    { color: var(--red); }
  .err-level.CRITICAL { color: var(--red); }
  .err-msg   { color: var(--text); word-break: break-word; }

  .reasoning {
    font-size: 11px; color: var(--muted); margin-top: 2px;
    white-space: nowrap; overflow: hidden; text-overflow: ellipsis; max-width: 240px;
  }
  .refresh-bar { text-align: center; font-size: 11px; color: var(--muted); padding: 12px 0 20px; }
  .empty { color: var(--muted); font-size: 13px; padding: 12px 0; }
  .stat-row { display: flex; justify-content: space-between; padding: 5px 0; border-bottom: 1px solid var(--border); }
  .stat-row:last-child { border-bottom: none; }
  .stat-label { color: var(--muted); }

  .two-col { display: grid; grid-template-columns: 1fr 1fr; gap: 12px; }
  @media(max-width:640px) { .two-col { grid-template-columns: 1fr; } }

  /* Bot log terminal */
  .log-wrap {
    background: #0a0c10;
    border-radius: 8px;
    padding: 12px;
    height: 380px;
    overflow-y: auto;
    font-family: "SF Mono", "Fira Code", "Menlo", monospace;
    font-size: 11.5px;
    line-height: 1.55;
    scroll-behavior: smooth;
  }
  .log-wrap::-webkit-scrollbar { width: 6px; }
  .log-wrap::-webkit-scrollbar-track { background: transparent; }
  .log-wrap::-webkit-scrollbar-thumb { background: var(--border); border-radius: 3px; }
  .log-line { white-space: pre-wrap; word-break: break-all; color: #94a3b8; }
  .log-line.INFO     { color: #94a3b8; }
  .log-line.WARNING  { color: var(--yellow); }
  .log-line.ERROR    { color: var(--red); }
  .log-line.CRITICAL { color: var(--red); font-weight: bold; }
  .log-line.trade    { color: var(--yellow); }
  .log-line.bullish  { color: var(--green); }
  .log-line.closed   { color: #f87171; }
  .log-controls {
    display: flex; align-items: center; gap: 10px;
    margin-bottom: 8px;
  }
  .log-controls label { font-size: 12px; color: var(--muted); }
  .log-controls input[type=checkbox] { accent-color: var(--accent); }
  .log-controls select {
    background: var(--border); color: var(--text);
    border: none; border-radius: 6px; padding: 3px 8px; font-size: 12px;
  }
  .range-btn {
    background: var(--border); color: var(--text); border: none; border-radius: 6px;
    padding: 3px 9px; font-size: 12px; cursor: pointer;
  }
  .range-btn:hover { background: #334155; }
  .range-btn.active { background: #2563eb; color: #fff; }
  #chartBaseline {
    background: var(--border); color: var(--text); border: 1px solid var(--border);
    border-radius: 6px; padding: 3px 6px; font-size: 12px; color-scheme: dark;
  }
</style>
<script src="https://cdn.jsdelivr.net/npm/chart.js@4.4.1/dist/chart.umd.min.js"></script>
</head>
<body>
<header>
  <div class="logo">📈 Trader Bot</div>
  <div class="status-pill">
    <div class="dot yellow" id="statusDot"></div>
    <span id="statusText">Loading…</span>
  </div>
</header>

<main>
  <!-- P&L + Account -->
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

  <!-- Strategy vs Market (SPY) -->
  <div class="card" style="margin-bottom:12px">
    <div class="card-title">Strategy vs Market (SPY)</div>
    <table>
      <thead><tr>
        <th></th>
        <th style="text-align:right">Strategy</th>
        <th style="text-align:right">SPY</th>
        <th style="text-align:right">vs&nbsp;SPY</th>
      </tr></thead>
      <tbody>
        <tr>
          <td>Latest session</td>
          <td style="text-align:right" id="bTodayAcct">—</td>
          <td style="text-align:right" id="bTodaySpy">—</td>
          <td style="text-align:right;font-weight:600" id="bTodayAlpha">—</td>
        </tr>
        <tr>
          <td>Past month</td>
          <td style="text-align:right" id="bPerAcct">—</td>
          <td style="text-align:right" id="bPerSpy">—</td>
          <td style="text-align:right;font-weight:600" id="bPerAlpha">—</td>
        </tr>
      </tbody>
    </table>
    <div class="sub" id="benchNote"></div>
  </div>

  <!-- Performance chart -->
  <div class="card" style="margin-bottom:12px">
    <div class="card-title-row">
      <div class="card-title">Performance vs Benchmarks — cumulative P&amp;L since baseline</div>
      <div style="display:flex;gap:6px;align-items:center;flex-wrap:wrap">
        <label class="sub" for="chartBaseline" style="margin:0">Baseline:</label>
        <input type="date" id="chartBaseline"/>
        <button class="range-btn" data-days="7">1W</button>
        <button class="range-btn active" data-days="30">1M</button>
        <button class="range-btn" data-days="90">3M</button>
        <button class="range-btn" data-days="365">Max</button>
        <span style="display:inline-block;width:10px"></span>
        <button class="range-btn active" data-unit="usd">$</button>
        <button class="range-btn" data-unit="pct">%</button>
      </div>
    </div>
    <div style="position:relative;height:360px;margin-top:6px"><canvas id="perfChart"></canvas></div>
    <div class="sub" id="chartNote" style="margin-top:6px">Click a legend item to show/hide a line. Strategy lines = cumulative <b>realized</b> P&amp;L; Portfolio = total equity change (incl. unrealized); SPY/QQQ = $ P&amp;L if the baseline portfolio value were in the index.</div>
  </div>

  <!-- Counters -->
  <div class="grid-4">
    <div class="mini-card">
      <div class="mini-label">Articles seen</div>
      <div class="mini-val" id="cntArticles">—</div>
      <div class="mini-sub">from log</div>
    </div>
    <div class="mini-card">
      <div class="mini-label">Signals scored</div>
      <div class="mini-val" id="cntScored">—</div>
      <div class="mini-sub" id="cntBullish"></div>
    </div>
    <div class="mini-card">
      <div class="mini-label">Trades placed</div>
      <div class="mini-val" id="cntTrades">—</div>
      <div class="mini-sub" id="statToday"></div>
    </div>
    <div class="mini-card">
      <div class="mini-label">Open positions</div>
      <div class="mini-val" id="statPositions">—</div>
      <div class="mini-sub" id="statTotal"></div>
    </div>
  </div>

  <!-- Activity feed + Errors side by side on wide screens -->
  <div class="two-col">
    <div class="card" style="margin-bottom:0">
      <div class="card-title">Activity Feed</div>
      <div class="feed" id="feedWrap">
        <div class="empty">Loading…</div>
      </div>
    </div>

    <div class="card" style="margin-bottom:0">
      <div class="card-title-row">
        <div class="card-title">Errors &amp; Warnings</div>
        <span class="err-count" id="errCount" style="display:none"></span>
      </div>
      <div id="errWrap">
        <div class="empty">No errors</div>
      </div>
    </div>
  </div>

  <!-- Strategy P&L -->
  <div class="card" style="margin-top:12px">
    <div class="card-title">Strategy P&amp;L</div>
    <div id="strategyWrap"><div class="empty">Loading…</div></div>
  </div>

  <!-- Open Positions -->
  <div class="card">
    <div class="card-title">Open Positions</div>
    <div id="positionsWrap"><div class="empty">Loading…</div></div>
  </div>

  <!-- Recent Trades -->
  <div class="card">
    <div class="card-title">Recent Trades</div>
    <div id="tradesWrap"><div class="empty">Loading…</div></div>
  </div>

  <!-- Bot Log -->
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
const esc  = s => s.replace(/&/g,"&amp;").replace(/</g,"&lt;").replace(/>/g,"&gt;");

function colorClass(n) { return n > 0 ? "pos" : n < 0 ? "neg" : "neu"; }

const STRAT_COLORS = {
  news_call:"#22c55e", qqq_macro:"#3b82f6", stock:"#93c5fd",
  pead:"#f59e0b", bear_short:"#ef4444", lotto:"#ec4899",
  pairs_long:"#a855f7", pairs_short:"#c084fc", router:"#14b8a6"
};
function stratPill(s) {
  const c = STRAT_COLORS[s] || "#64748b";
  return `<span class="strategy-pill" style="background:${c}22;color:${c}">${esc(s||"—")}</span>`;
}

function renderStrategy(data) {
  if (!data || !Object.keys(data).length)
    return '<div class="empty">No strategy data yet — trades will appear here after the first close</div>';
  const order = ["news_call","qqq_macro","stock","pead","bear_short","lotto","pairs_long","pairs_short","router"];
  const keys  = [...new Set([...order, ...Object.keys(data)])];
  let html = '<div style="overflow-x:auto"><table><thead><tr>'
    + '<th>Strategy</th><th>Open</th><th>Unrealized</th>'
    + '<th>Closed</th><th>Realized</th><th>Win%</th><th>Total P&amp;L</th>'
    + '</tr></thead><tbody>';
  let totUnreal=0, totReal=0, totClosed=0, totWins=0;
  for (const s of keys) {
    const d   = data[s] || {};
    const unr = d.unrealized || 0;
    const rel = d.realized   || 0;
    const tot = unr + rel;
    const wr  = d.closed > 0 ? ((d.wins||0)/d.closed*100).toFixed(0)+"%" : "—";
    // "router" is an attribution overlay of trades already counted under the
    // other strategies — show it, but DON'T add it to the TOTAL (avoid double-count).
    const isOverlay = (s === "router");
    if (!isOverlay) { totUnreal += unr; totReal += rel; totClosed += d.closed||0; totWins += d.wins||0; }
    const rowStyle = isOverlay ? ' style="border-top:1px dashed var(--border);opacity:0.95"' : '';
    const label = isOverlay ? stratPill(s) + ' <span style="color:var(--muted);font-size:11px">(overlay)</span>' : stratPill(s);
    html += `<tr${rowStyle}>
      <td>${label}</td>
      <td style="color:var(--muted)">${d.open||0}</td>
      <td class="${colorClass(unr)}">${fmt(unr)}</td>
      <td style="color:var(--muted)">${d.closed||0}</td>
      <td class="${colorClass(rel)}">${fmt(rel)}</td>
      <td>${wr}</td>
      <td class="${colorClass(tot)}" style="font-weight:700">${fmt(tot)}</td>
    </tr>`;
  }
  const totTot  = totUnreal + totReal;
  const totWr   = totClosed > 0 ? (totWins/totClosed*100).toFixed(0)+"%" : "—";
  html += `<tr style="border-top:2px solid var(--border)">
    <td style="font-weight:700;color:var(--muted)">TOTAL</td>
    <td></td>
    <td class="${colorClass(totUnreal)}" style="font-weight:700">${fmt(totUnreal)}</td>
    <td style="color:var(--muted)">${totClosed}</td>
    <td class="${colorClass(totReal)}" style="font-weight:700">${fmt(totReal)}</td>
    <td>${totWr}</td>
    <td class="${colorClass(totTot)}" style="font-weight:700;font-size:15px">${fmt(totTot)}</td>
  </tr>`;
  return html + '</tbody></table></div>';
}

function renderPositions(positions) {
  if (!positions.length) return '<div class="empty">No open positions</div>';
  let html = '<div style="overflow-x:auto"><table><thead><tr>'
    + '<th>Symbol</th><th>Side</th><th>Strategy</th><th>Qty</th><th>Entry</th><th>Current</th><th>Stop</th><th>Unreal. P&L</th>'
    + '</tr></thead><tbody>';
  for (const p of positions) {
    if (p.error) { html += `<tr><td colspan="8" class="neg">${esc(p.error)}</td></tr>`; continue; }
    const cls  = colorClass(p.unrealized_pl);
    const stop = p.stop_price != null
      ? `<span style="color:var(--yellow)">${fmt(p.stop_price)}</span>`
      : '<span style="color:var(--muted)">—</span>';
    const side = p.side === "short"
      ? '<span style="font-weight:700;font-size:10px;letter-spacing:.5px;color:#fca5a5;background:#7f1d1d;padding:2px 6px;border-radius:4px">SHORT</span>'
      : '<span style="font-weight:700;font-size:10px;letter-spacing:.5px;color:#86efac;background:#14532d;padding:2px 6px;border-radius:4px">LONG</span>';
    html += `<tr>
      <td style="font-weight:600;font-size:12px">${esc(p.symbol)}</td>
      <td>${side}</td>
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
    + '<th>Time</th><th>Ticker</th><th>Strategy</th><th>Event</th><th>Result</th><th>Detail</th>'
    + '</tr></thead><tbody>';
  for (const t of trades) {
    const ts = t.timestamp
      ? new Date(t.timestamp).toLocaleString([],{month:"numeric",day:"numeric",hour:"2-digit",minute:"2-digit"})
      : "";
    const ticker = t.source_ticker || t.ticker || t.underlying || t.symbol || "—";
    const strat  = t.strategy || "—";
    let event, result, detail;
    if (t.kind === "close") {
      const pnl = parseFloat(t.pnl_usd), pct = parseFloat(t.pnl_pct);
      const cls = colorClass(isNaN(pnl) ? 0 : pnl);
      event  = '<span style="font-weight:700;font-size:10px;letter-spacing:.5px;color:#cbd5e1;background:#374151;padding:2px 6px;border-radius:4px">CLOSED</span>';
      result = `<span class="${cls}">${isNaN(pnl)?"—":fmt(pnl)}`
             + `${isNaN(pct)?"":` <span style="font-size:11px">(${pct>=0?"+":""}${pct.toFixed(1)}%)</span>`}</span>`;
      detail = `<span style="font-size:12px;color:var(--muted)">⮐ ${esc(t.reason_label || t.reason || "—")}</span>`;
    } else {
      const cost = t.cost_basis ? fmt(parseFloat(t.cost_basis)) : (t.price ? fmt(parseFloat(t.price)) : "—");
      const conf = t.confidence ? (parseFloat(t.confidence)*100).toFixed(0)+"%" : "";
      event  = '<span style="font-weight:700;font-size:10px;letter-spacing:.5px;color:#7db3ff;background:#1e3a5f;padding:2px 6px;border-radius:4px">OPEN</span>';
      result = `${cost}${conf?` <span style="font-size:11px;color:var(--muted)">${conf}</span>`:""}`;
      detail = `<div class="reasoning">${esc(t.reasoning||"")}</div>`;
    }
    html += `<tr>
      <td style="white-space:nowrap;color:var(--muted)">${ts}</td>
      <td style="font-weight:600">${esc(ticker)}</td>
      <td>${stratPill(strat)}</td>
      <td>${event}</td>
      <td>${result}</td>
      <td>${detail}</td>
    </tr>`;
  }
  return html + '</tbody></table></div>';
}

function renderFeed(items) {
  if (!items.length) return '<div class="empty">No activity yet</div>';
  return items.map(item => {
    let cls = "";
    if (item.kind === "signal") cls = item.direction || "";
    if (item.kind === "trade")  cls = "trade";
    if (item.kind === "closed") cls = "closed";
    return `<div class="feed-item">
      <span class="feed-ts">${item.ts}</span>
      <span class="feed-icon">${item.icon}</span>
      <span class="feed-text ${cls}">${esc(item.text)}</span>
    </div>`;
  }).join("");
}

function renderErrors(items) {
  if (!items.length) return '<div class="empty">No errors</div>';
  return items.map(e => `<div class="err-item">
    <span class="err-ts">${e.ts}</span>
    <span class="err-level ${e.level}">${e.level}</span>
    <span class="err-msg">${esc(e.msg)}</span>
  </div>`).join("");
}

async function refresh() {
  try {
    const res  = await fetch("/api/status");
    const data = await res.json();

    // Status pill
    document.getElementById("statusDot").className = "dot " + (data.bot_running ? "green" : "red");
    document.getElementById("statusText").textContent = data.bot_running ? "Bot running" : "Bot stopped";

    // Account
    const a = data.account || {};
    if (a.equity !== undefined) {
      const pnl   = a.pnl_today || 0;
      const pnlEl = document.getElementById("pnlToday");
      pnlEl.textContent = fmt(pnl);
      pnlEl.className   = "big-num " + colorClass(pnl);
      document.getElementById("pnlTodayPct").textContent = fmtP(a.pnl_today_pct || 0);
      document.getElementById("equity").textContent = fmt(a.equity);
      document.getElementById("cash").textContent   = "Cash: " + fmt(a.cash);
    }

    // Strategy vs Market (SPY)
    const b = data.benchmark || {};
    const setBench = (id, v) => {
      const el = document.getElementById(id);
      if (!el) return;
      if (v === undefined || v === null) { el.textContent = "—"; el.className = el.className.replace(/\b(pos|neg|neu)\b/g,"").trim(); }
      else { el.textContent = fmtP(v); el.className = (el.className.replace(/\b(pos|neg|neu)\b/g,"").trim() + " " + colorClass(v)).trim(); }
    };
    setBench("bTodayAcct", b.today_account); setBench("bTodaySpy", b.today_spy); setBench("bTodayAlpha", b.today_alpha);
    setBench("bPerAcct",  b.period_account); setBench("bPerSpy",  b.period_spy);  setBench("bPerAlpha",  b.period_alpha);
    const benchNote = document.getElementById("benchNote");
    if (benchNote) {
      benchNote.textContent = b.error ? "benchmark unavailable"
        : (b.period_alpha !== undefined
            ? (b.period_alpha >= 0 ? "Beating the market over the past month" : "Trailing the market over the past month")
            : "");
    }

    // Counters
    const c = (data.log || {}).counters || {};
    const s = data.stats || {};
    document.getElementById("cntArticles").textContent  = c.articles  ?? "—";
    document.getElementById("cntScored").textContent    = c.scored    ?? "—";
    document.getElementById("cntBullish").textContent   = c.bullish ? `${c.bullish} bullish` : "";
    // Trades placed: drive from trades.csv (authoritative for ALL entries — stock+option),
    // NOT the log-tail counter c.trades_placed (its regex is stale + only scans 500 lines).
    document.getElementById("cntTrades").textContent    = s.today_trades ?? "—";
    document.getElementById("statToday").textContent    = s.total_trades != null ? `${s.total_trades} all-time` : "";
    document.getElementById("statPositions").textContent = (data.positions||[]).length;
    document.getElementById("statTotal").textContent    = "";

    // Feed
    document.getElementById("feedWrap").innerHTML = renderFeed((data.log||{}).activity || []);

    // Errors
    const errs = (data.log||{}).errors || [];
    const errCountEl = document.getElementById("errCount");
    if (errs.length) {
      errCountEl.textContent = errs.length;
      errCountEl.style.display = "";
    } else {
      errCountEl.style.display = "none";
    }
    document.getElementById("errWrap").innerHTML = renderErrors(errs);

    // Positions
    document.getElementById("positionsWrap").innerHTML = renderPositions(data.positions || []);

    // Trades
    document.getElementById("tradesWrap").innerHTML = renderTrades(data.recent_trades || []);

    // Strategy P&L (separate endpoint — slightly stale is fine)
    fetch("/api/strategy_pnl").then(r=>r.json()).then(d => {
      document.getElementById("strategyWrap").innerHTML = renderStrategy(d);
    }).catch(()=>{});

    // Timestamp
    document.getElementById("lastUpdated").textContent =
      "Updated " + new Date().toLocaleTimeString([],{hour:"2-digit",minute:"2-digit",second:"2-digit"});

  } catch(e) {
    document.getElementById("statusText").textContent = "Error";
    document.getElementById("statusDot").className = "dot red";
  }
}

// ── Bot log ──────────────────────────────────────────────────────────────────

function classifyLine(line) {
  if (/WARNING/.test(line))  return "WARNING";
  if (/ERROR|CRITICAL/.test(line)) return "ERROR";
  if (/💰 Position sizing|🛒|Placing (option|crypto)/.test(line)) return "trade";
  if (/bullish/.test(line))  return "bullish";
  if (/🔴 Closed|trailing stop/.test(line)) return "closed";
  if (/INFO/.test(line))     return "INFO";
  return "";
}

async function fetchLog() {
  const n    = document.getElementById("logLines").value;
  const wrap = document.getElementById("logWrap");
  const wasAtBottom = wrap.scrollHeight - wrap.scrollTop - wrap.clientHeight < 40;
  try {
    const res  = await fetch(`/api/log?lines=${n}`);
    const data = await res.json();
    wrap.innerHTML = data.lines.map(l =>
      `<div class="log-line ${classifyLine(l)}">${esc(l)}</div>`
    ).join("");
    const autoScroll = document.getElementById("logAutoScroll").checked;
    if (autoScroll && wasAtBottom) {
      wrap.scrollTop = wrap.scrollHeight;
    }
  } catch(e) {
    wrap.innerHTML = `<span class="log-line ERROR">Failed to fetch log</span>`;
  }
}

// ── Performance vs Benchmarks chart ────────────────────────────────
const CHART_COLORS = Object.assign({ Portfolio:"#f8fafc", SPY:"#94a3b8", QQQ:"#60a5fa" }, STRAT_COLORS);
let perfChart = null, lastChartData = null, chartMode = "usd";
const isoDaysAgo = n => { const d = new Date(); d.setDate(d.getDate() - n); return d.toISOString().slice(0,10); };

function renderChart() {
  const data = lastChartData;
  if (!data || !data.labels) return;
  const pct = (chartMode === "pct");
  const base = data.base_equity || 0, cap = data.capital || {};
  const order = ["Portfolio","SPY","QQQ","router","news_call","stock","lotto","pead","qqq_macro","bear_short","pairs_long","pairs_short"];
  const names = [...new Set([...order, ...Object.keys(data.series||{})])].filter(n => data.series && data.series[n]);
  const datasets = names.map(n => {
    const isBench = (n === "SPY" || n === "QQQ"), isPort = (n === "Portfolio");
    let arr = data.series[n];
    if (pct) {
      const denom = (n in cap) ? cap[n] : base;   // strategies/router → deployed capital; else baseline equity
      arr = denom ? arr.map(v => v===null ? null : +(v/denom*100).toFixed(3)) : arr.map(() => null);
    }
    return {
      label: n, data: arr,
      borderColor: CHART_COLORS[n] || "#64748b", backgroundColor: "transparent",
      borderWidth: isPort ? 2.6 : (isBench ? 2 : 1.6),
      borderDash: isBench ? [6,4] : [],
      pointRadius: 0, pointHoverRadius: 3, tension: 0.15, spanGaps: true,
    };
  });
  const fmtAxis = pct ? (v => (v>=0?"+":"")+v.toFixed(1)+"%") : (v => (v>=0?"+":"")+"$"+Math.round(v).toLocaleString());
  const fmtTip  = pct ? (y => `${y>=0?"+":""}${(y||0).toFixed(2)}%`) : (y => `${y>=0?"+":""}$${(y||0).toLocaleString(undefined,{maximumFractionDigits:0})}`);
  if (perfChart) perfChart.destroy();
  perfChart = new Chart(document.getElementById("perfChart").getContext("2d"), {
    type: "line",
    data: { labels: data.labels, datasets },
    options: {
      responsive: true, maintainAspectRatio: false, animation: false,
      interaction: { mode: "index", intersect: false },
      plugins: {
        legend: { position: "bottom", labels: { color:"#cbd5e1", boxWidth:12, padding:10, font:{size:11} } },
        tooltip: { callbacks: { label: c => `${c.dataset.label}: ${fmtTip(c.parsed.y)}` } }
      },
      scales: {
        x: { ticks: { color:"#64748b", maxTicksLimit:8, autoSkip:true, font:{size:10} }, grid: { color:"rgba(148,163,184,0.08)" } },
        y: { ticks: { color:"#64748b", font:{size:10}, callback: fmtAxis },
             grid: { color: c => (c.tick && c.tick.value===0) ? "rgba(148,163,184,0.45)" : "rgba(148,163,184,0.08)" } }
      }
    }
  });
}

async function loadChart(baseline) {
  try { lastChartData = await (await fetch(`/api/chart?baseline=${baseline}`)).json(); }
  catch (e) { return; }
  renderChart();
}

document.querySelectorAll(".range-btn[data-days]").forEach(btn => btn.addEventListener("click", () => {
  const iso = isoDaysAgo(+btn.dataset.days);
  document.getElementById("chartBaseline").value = iso;
  document.querySelectorAll(".range-btn[data-days]").forEach(b => b.classList.toggle("active", b === btn));
  loadChart(iso);
}));
document.querySelectorAll(".range-btn[data-unit]").forEach(btn => btn.addEventListener("click", () => {
  chartMode = btn.dataset.unit;
  document.querySelectorAll(".range-btn[data-unit]").forEach(b => b.classList.toggle("active", b === btn));
  document.getElementById("chartNote").innerHTML = chartMode === "pct"
    ? 'Percentage view. Strategy/Router lines = return on <b>capital deployed</b> (ROIC); Portfolio = total account return; SPY/QQQ = index return. Click a legend item to show/hide.'
    : 'Dollar view (cumulative P&amp;L). Strategy lines = realized P&amp;L; Portfolio = total equity change (incl. unrealized); SPY/QQQ = $ P&amp;L if the baseline portfolio value were in the index. Click a legend item to show/hide.';
  renderChart();
}));
document.getElementById("chartBaseline").addEventListener("change", e => {
  document.querySelectorAll(".range-btn[data-days]").forEach(b => b.classList.remove("active"));
  loadChart(e.target.value);
});
document.getElementById("chartBaseline").value = isoDaysAgo(30);
loadChart(isoDaysAgo(30));
setInterval(() => loadChart(document.getElementById("chartBaseline").value), 60000);

refresh();
fetchLog();
setInterval(refresh, 15000);
setInterval(fetchLog, 5000);
</script>
</body>
</html>
"""

@app.route("/")
def index():
    return render_template_string(HTML)


if __name__ == "__main__":
    host = "0.0.0.0"
    print(f"\n  Dashboard running at http://localhost:{PORT}")
    print(f"  On your phone (same WiFi): http://<your-mac-ip>:{PORT}\n")
    app.run(host=host, port=PORT, debug=False)
