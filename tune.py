"""
tune.py — Parameter sweep for the news trading backtest.

Fetches + scores articles once (using the existing cache), then grid-searches
over signal and execution parameters. Results are ranked by Sharpe ratio and
written to tune_results.csv + tune_report.html.

Usage:
    python tune.py                        # full 180-day window
    python tune.py --days 30              # shorter window for speed
    python tune.py --workers 4            # parallel Ollama scoring
    python tune.py --no-cache             # ignore cached scores
"""

import argparse
import csv
import itertools
import json
import logging
import math
import time
from copy import deepcopy
from datetime import datetime, timedelta, timezone
from pathlib import Path

import backtest as _backtest_mod
from backtest import (
    load_cache, save_cache, fetch_alpaca_news_day, fetch_sec_edgar_8k,
    score_article, get_price_at, get_stock_bars,
    is_valid_stock_ticker,
)
from concurrent.futures import ThreadPoolExecutor

# ── In-memory bar cache — keyed by ticker only, stores the full wide window.
# _cached_get_stock_bars slices by date on the way out so any sub-range
# requested by get_price_at or _simulate_trade is served from memory.
_bar_cache: dict[str, list] = {}

def _cached_get_stock_bars(ticker, start, end):
    if ticker not in _bar_cache:
        _bar_cache[ticker] = get_stock_bars(ticker, start, end)
    bars = _bar_cache[ticker]
    # Slice to the requested window
    if start.tzinfo is None:
        start = start.replace(tzinfo=__import__('datetime').timezone.utc)
    if end.tzinfo is None:
        end = end.replace(tzinfo=__import__('datetime').timezone.utc)
    return [b for b in bars if start <= b["t"] <= end]

# Patch module-level get_stock_bars so get_price_at + _simulate_trade both
# read from the in-memory cache instead of making HTTP calls.
_backtest_mod.get_stock_bars = _cached_get_stock_bars

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

# ═══════════════════════════════════════════════════════════════════════════════
# PARAMETER GRID  — edit these lists to control the sweep
# ═══════════════════════════════════════════════════════════════════════════════

GRID = {
    "MIN_MAGNITUDE":     [0.35, 0.40, 0.45, 0.50],
    "BASE_CONFIDENCE":   [0.55, 0.60, 0.65],
    "CONFIDENCE_SLOPE":  [0.15, 0.25],
    "TRAILING_STOP_PCT": [0.15],
    # DTE midpoint used for premium + theta decay estimation in simulation.
    # Short-dated options (1-5 DTE) have higher gamma — news-driven moves
    # amplify more per dollar spent, but theta is brutal if the stock stalls.
    "DTE_TARGET":        [2, 5, 10, 21, 37],
}

# ═══════════════════════════════════════════════════════════════════════════════
# PHASE 1 — fetch + score (runs once)
# ═══════════════════════════════════════════════════════════════════════════════

def fetch_and_score(days: int, use_cache: bool, workers: int) -> list[dict]:
    """
    Return a list of scored-article dicts, each with keys:
      headline, created_at, magnitude, confidence, sentiment, tickers, reasoning
    Articles that failed scoring are excluded.
    """
    load_cache()

    end_dt   = datetime.now(timezone.utc) - timedelta(days=5)
    start_dt = end_dt - timedelta(days=days)

    log.info("═" * 60)
    log.info("FETCH + SCORE  %s → %s  (%d days)",
             start_dt.strftime("%Y-%m-%d"), end_dt.strftime("%Y-%m-%d"), days)
    log.info("═" * 60)

    seen_ids: set[str] = set()
    all_articles: list[dict] = []

    def add_articles(items):
        for a in items:
            if a["id"] not in seen_ids and a["headline"].strip():
                seen_ids.add(a["id"])
                all_articles.append(a)

    log.info("📰 Fetching Alpaca news (50/day × %d days)…", days)
    day = start_dt
    while day < end_dt:
        add_articles(fetch_alpaca_news_day(day))
        day += timedelta(days=1)
        time.sleep(0.15)

    alpaca_count = len(all_articles)
    log.info("📰 Alpaca: %d articles", alpaca_count)

    log.info("📋 Fetching SEC EDGAR 8-K filings…")
    chunk = start_dt
    while chunk < end_dt:
        chunk_end = min(chunk + timedelta(days=30), end_dt)
        add_articles(fetch_sec_edgar_8k(chunk, chunk_end))
        chunk += timedelta(days=31)
        time.sleep(0.5)

    log.info("📰 Total articles: %d  (alpaca=%d  sec=%d)",
             len(all_articles), alpaca_count, len(all_articles) - alpaca_count)

    log.info("🤖 Scoring with workers=%d  cache=%s…", workers, use_cache)

    def score_one(article):
        sig = score_article(article["headline"], article["summary"], use_cache=use_cache)
        if sig:
            sig = dict(sig)
            sig["_article"] = article
        return sig

    with ThreadPoolExecutor(max_workers=workers) as pool:
        results = list(pool.map(score_one, all_articles))

    save_cache()
    log.info("💾 Cache saved (%d entries)", len(__import__('backtest')._score_cache))

    scored = []
    for sig in results:
        if sig is None:
            continue
        article    = sig.pop("_article")
        created_at = article.get("created_at")
        if not created_at:
            continue
        if isinstance(created_at, str):
            created_at = datetime.fromisoformat(created_at.replace("Z", "+00:00"))
        scored.append({
            "headline":   article["headline"],
            "created_at": created_at,
            "magnitude":  float(sig.get("magnitude", 0)),
            "confidence": float(sig.get("confidence", 0)),
            "sentiment":  sig.get("sentiment", ""),
            "tickers":    sig.get("tickers", []),
            "reasoning":  sig.get("reasoning", ""),
        })

    log.info("✅ Scored %d articles", len(scored))
    return scored


# ═══════════════════════════════════════════════════════════════════════════════
# TRADE SIMULATOR  (self-contained so DTE + trail are per-run params)
# ═══════════════════════════════════════════════════════════════════════════════

def _simulate_trade(ticker, entry_dt, entry_stock_price, position_usd,
                    magnitude, confidence, reasoning,
                    dte_target, trailing_stop_pct):
    from config import TARGET_DELTA, MAX_HOLD_DAYS
    from datetime import timedelta

    premium_estimate = entry_stock_price * 0.04 * _math.sqrt(dte_target / 30)
    qty              = max(1, int(position_usd / (premium_estimate * 100)))
    cost_basis       = premium_estimate * 100 * qty

    peak_value   = premium_estimate
    exit_dt      = None
    exit_reason  = "max_hold"
    exit_premium = premium_estimate

    bars         = _cached_get_stock_bars(ticker, entry_dt - timedelta(days=14), entry_dt + timedelta(days=MAX_HOLD_DAYS * 2))
    trading_bars = [b for b in bars if b["t"] > entry_dt]

    for bar in trading_bars[:MAX_HOLD_DAYS]:
        stock_move_pct  = (bar["c"] - entry_stock_price) / entry_stock_price
        option_move_pct = stock_move_pct * TARGET_DELTA * (1 + max(0, stock_move_pct) * 2)
        days_held       = (bar["t"] - entry_dt).days
        theta_decay     = 0.02 * (days_held / dte_target)
        est_premium     = max(0.01, premium_estimate * (1 + option_move_pct - theta_decay))

        if est_premium > peak_value:
            peak_value = est_premium

        if est_premium <= peak_value * (1 - trailing_stop_pct):
            exit_premium = est_premium
            exit_dt      = bar["t"]
            exit_reason  = "trailing_stop"
            break
    else:
        if trading_bars:
            last           = trading_bars[min(MAX_HOLD_DAYS - 1, len(trading_bars) - 1)]
            smv            = (last["c"] - entry_stock_price) / entry_stock_price
            omv            = smv * TARGET_DELTA * (1 + max(0, smv) * 2)
            exit_premium   = max(0.01, premium_estimate * (1 + omv - 0.02))
            exit_dt        = last["t"]

    exit_value = exit_premium * 100 * qty
    pnl_usd    = exit_value - cost_basis
    pnl_pct    = (pnl_usd / cost_basis) * 100 if cost_basis > 0 else 0

    return {
        "ticker":            ticker,
        "entry_dt":          entry_dt.isoformat(),
        "exit_dt":           exit_dt.isoformat() if exit_dt else "",
        "exit_reason":       exit_reason,
        "entry_stock_price": round(entry_stock_price, 2),
        "entry_premium":     round(premium_estimate, 4),
        "exit_premium":      round(exit_premium, 4),
        "qty":               qty,
        "cost_basis":        round(cost_basis, 2),
        "exit_value":        round(exit_value, 2),
        "pnl_usd":           round(pnl_usd, 2),
        "pnl_pct":           round(pnl_pct, 2),
        "magnitude":         round(magnitude, 3),
        "confidence":        round(confidence, 3),
        "position_usd":      round(position_usd, 2),
        "reasoning":         reasoning,
    }


# ═══════════════════════════════════════════════════════════════════════════════
# PHASE 2 — simulate trades for one parameter combination
# ═══════════════════════════════════════════════════════════════════════════════

def simulate(scored: list[dict], params: dict) -> dict:
    MIN_MAGNITUDE     = params["MIN_MAGNITUDE"]
    BASE_CONFIDENCE   = params["BASE_CONFIDENCE"]
    CONFIDENCE_SLOPE  = params["CONFIDENCE_SLOPE"]
    TRAILING_STOP_PCT = params["TRAILING_STOP_PCT"]
    DTE_TARGET        = params["DTE_TARGET"]

    from config import MAX_POSITION_USD, DAILY_LOSS_LIMIT, MIN_STOCK_PRICE

    def signal_checks(mag, conf):
        if mag < MIN_MAGNITUDE:
            return False
        required = BASE_CONFIDENCE + (1 - mag) * CONFIDENCE_SLOPE
        return conf >= required

    def scale_usd(mag, conf):
        return max(MAX_POSITION_USD * 0.10, MAX_POSITION_USD * mag * conf)

    seen_tickers_by_day: dict[str, set] = {}
    passing = []

    for row in scored:
        if row["sentiment"] != "bullish":
            continue
        mag  = row["magnitude"]
        conf = row["confidence"]
        if not signal_checks(mag, conf):
            continue
        tickers    = row["tickers"]
        created_at = row["created_at"]
        trade_day  = created_at.date().isoformat()
        day_set    = seen_tickers_by_day.setdefault(trade_day, set())

        for ticker in tickers[:2]:
            if ticker in day_set:
                continue
            if ticker in ("BTC", "ETH"):
                continue
            if not is_valid_stock_ticker(ticker):
                continue
            day_set.add(ticker)
            passing.append({
                "ticker":     ticker,
                "created_at": created_at,
                "magnitude":  mag,
                "confidence": conf,
                "reasoning":  row["reasoning"],
            })

    trade_rows   = []
    daily_loss   = {}
    halted_days  = set()
    running_pnl  = 0.0
    equity_curve = []

    for p in passing:
        ticker     = p["ticker"]
        created_at = p["created_at"]
        trade_day  = created_at.date().isoformat()

        if trade_day in halted_days:
            continue

        entry_price = get_price_at(ticker, created_at)
        if not entry_price or entry_price < MIN_STOCK_PRICE:
            continue

        mag  = p["magnitude"]
        conf = p["confidence"]
        position_usd = scale_usd(mag, conf)

        trade = _simulate_trade(
            ticker, created_at, entry_price, position_usd,
            mag, conf, p["reasoning"],
            DTE_TARGET, TRAILING_STOP_PCT,
        )
        if not trade:
            continue

        day_loss = daily_loss.get(trade_day, 0.0)
        if trade["pnl_usd"] < 0:
            day_loss += abs(trade["pnl_usd"])
            daily_loss[trade_day] = day_loss
            if day_loss >= DAILY_LOSS_LIMIT:
                halted_days.add(trade_day)

        running_pnl += trade["pnl_usd"]
        trade["cumulative_pnl"] = round(running_pnl, 2)
        trade_rows.append(trade)
        equity_curve.append({"date": trade_day, "cumulative_pnl": round(running_pnl, 2)})

    n      = len(trade_rows)
    wins   = [t for t in trade_rows if t["pnl_usd"] > 0]
    losses = [t for t in trade_rows if t["pnl_usd"] <= 0]

    total_pnl  = round(running_pnl, 2)
    win_rate   = len(wins) / n * 100 if n else 0
    avg_win    = sum(t["pnl_usd"] for t in wins)  / len(wins)  if wins   else 0
    avg_loss   = sum(t["pnl_usd"] for t in losses) / len(losses) if losses else 0
    pf         = round(abs(avg_win / avg_loss), 2) if avg_loss else 0

    peak_eq, max_dd = 0.0, 0.0
    for pt in equity_curve:
        eq = pt["cumulative_pnl"]
        peak_eq = max(peak_eq, eq)
        max_dd  = max(max_dd, peak_eq - eq)

    if n > 1:
        returns = [t["pnl_pct"] for t in trade_rows]
        mean_r  = sum(returns) / len(returns)
        std_r   = math.sqrt(sum((r - mean_r) ** 2 for r in returns) / len(returns)) or 1
        sharpe  = round((mean_r / std_r) * math.sqrt(252), 2)
    else:
        sharpe = 0.0

    return {
        **params,
        "trades":       n,
        "passing":      len(passing),
        "total_pnl":    total_pnl,
        "win_rate":     round(win_rate, 1),
        "avg_win":      round(avg_win, 2),
        "avg_loss":     round(avg_loss, 2),
        "profit_factor": pf,
        "max_drawdown": round(max_dd, 2),
        "sharpe":       sharpe,
        "halted_days":  len(halted_days),
    }


# ═══════════════════════════════════════════════════════════════════════════════
# PATCH simulate_option_pnl to accept dte_override + trailing_stop_pct
# ═══════════════════════════════════════════════════════════════════════════════

import math as _math


# ═══════════════════════════════════════════════════════════════════════════════
# HTML REPORT
# ═══════════════════════════════════════════════════════════════════════════════

def write_tune_report(results: list[dict], path: str = "tune_report.html"):
    cols = [
        ("MIN_MAGNITUDE", "Min Mag"),
        ("BASE_CONFIDENCE", "Base Conf"),
        ("CONFIDENCE_SLOPE", "Slope"),
        ("TRAILING_STOP_PCT", "Trail Stop"),
        ("DTE_TARGET", "DTE Target"),
        ("trades", "Trades"),
        ("passing", "Signals"),
        ("total_pnl", "Total P&L"),
        ("win_rate", "Win Rate"),
        ("avg_win", "Avg Win"),
        ("avg_loss", "Avg Loss"),
        ("profit_factor", "Prof. Factor"),
        ("max_drawdown", "Max DD"),
        ("sharpe", "Sharpe"),
        ("halted_days", "Halted"),
    ]

    rows_html = ""
    for i, r in enumerate(results):
        rank_style = ""
        if i == 0:
            rank_style = "background:#1a3a2a;"
        elif i == 1:
            rank_style = "background:#162a3a;"
        elif i == 2:
            rank_style = "background:#2a2a1a;"

        pnl_color = "#2ecc71" if r["total_pnl"] >= 0 else "#e74c3c"
        rows_html += f"<tr style='{rank_style}'>"
        rows_html += f"<td style='color:#aaa'>{i+1}</td>"
        for key, _ in cols:
            val = r[key]
            style = ""
            if key == "total_pnl":
                style = f"color:{pnl_color};font-weight:600"
                val = f"${val:+,.2f}"
            elif key == "win_rate":
                val = f"{val}%"
            elif key in ("avg_win", "avg_loss"):
                val = f"${val:+,.2f}"
            elif key == "max_drawdown":
                val = f"${val:,.2f}"
                style = "color:#e74c3c"
            elif key == "sharpe":
                style = "color:#3498db;font-weight:600"
            elif key == "TRAILING_STOP_PCT":
                val = f"{int(val*100)}%"
            rows_html += f"<td style='{style}'>{val}</td>"
        rows_html += "</tr>"

    header_html = "<th>Rank</th>" + "".join(f"<th>{label}</th>" for _, label in cols)

    # Sharpe distribution chart
    sharpes = [r["sharpe"] for r in results]
    pnls    = [r["total_pnl"] for r in results]

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<title>Parameter Sweep Results</title>
<script src="https://cdnjs.cloudflare.com/ajax/libs/Chart.js/4.4.1/chart.umd.min.js"></script>
<style>
  * {{ box-sizing: border-box; margin: 0; padding: 0; }}
  body {{ font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
          background: #0f1117; color: #e0e0e0; padding: 24px; }}
  h1 {{ font-size: 1.5em; margin-bottom: 4px; color: #fff; }}
  .subtitle {{ color: #888; font-size: 0.85em; margin-bottom: 24px; }}
  .charts {{ display: grid; grid-template-columns: 1fr 1fr; gap: 20px; margin-bottom: 28px; }}
  .chart-box {{ background: #1a1d27; border-radius: 10px; padding: 20px; }}
  .chart-box h2 {{ font-size: 0.9em; color: #aaa; margin-bottom: 14px; }}
  table {{ width: 100%; border-collapse: collapse; background: #1a1d27;
           border-radius: 10px; overflow: hidden; font-size: 0.8em; }}
  th {{ background: #252836; padding: 8px 10px; text-align: left;
        color: #aaa; font-weight: 600; font-size: 0.75em; text-transform: uppercase; white-space: nowrap; }}
  td {{ padding: 7px 10px; border-bottom: 1px solid #1e2130; white-space: nowrap; }}
  tr:last-child td {{ border-bottom: none; }}
  tr:hover td {{ background: #21232f; }}
</style>
</head>
<body>
<h1>📊 Parameter Sweep Results</h1>
<div class="subtitle">{len(results)} combinations evaluated · ranked by Sharpe ratio</div>

<div class="charts">
  <div class="chart-box">
    <h2>Sharpe Ratio Distribution</h2>
    <canvas id="sharpeChart"></canvas>
  </div>
  <div class="chart-box">
    <h2>Total P&L Distribution</h2>
    <canvas id="pnlChart"></canvas>
  </div>
</div>

<table>
  <thead><tr>{header_html}</tr></thead>
  <tbody>{rows_html}</tbody>
</table>

<script>
const sharpes = {json.dumps(sharpes)};
const pnls    = {json.dumps(pnls)};
const labels  = Array.from({{length: sharpes.length}}, (_, i) => i + 1);

new Chart(document.getElementById('sharpeChart'), {{
  type: 'bar',
  data: {{
    labels,
    datasets: [{{
      data: sharpes,
      backgroundColor: sharpes.map(v => v >= 0 ? '#3498db88' : '#e74c3c88'),
      borderColor:     sharpes.map(v => v >= 0 ? '#3498db' : '#e74c3c'),
      borderWidth: 1,
    }}]
  }},
  options: {{
    responsive: true,
    plugins: {{ legend: {{ display: false }} }},
    scales: {{
      x: {{ ticks: {{ display: false }}, grid: {{ color: '#252836' }} }},
      y: {{ ticks: {{ color: '#666' }}, grid: {{ color: '#252836' }} }}
    }}
  }}
}});

new Chart(document.getElementById('pnlChart'), {{
  type: 'bar',
  data: {{
    labels,
    datasets: [{{
      data: pnls,
      backgroundColor: pnls.map(v => v >= 0 ? '#2ecc7188' : '#e74c3c88'),
      borderColor:     pnls.map(v => v >= 0 ? '#2ecc71' : '#e74c3c'),
      borderWidth: 1,
    }}]
  }},
  options: {{
    responsive: true,
    plugins: {{ legend: {{ display: false }} }},
    scales: {{
      x: {{ ticks: {{ display: false }}, grid: {{ color: '#252836' }} }},
      y: {{
        ticks: {{ color: '#666', callback: v => '$' + v.toLocaleString() }},
        grid: {{ color: '#252836' }}
      }}
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
    parser = argparse.ArgumentParser(description="Backtest parameter sweep")
    parser.add_argument("--days",     type=int, default=180)
    parser.add_argument("--workers",  type=int, default=2)
    parser.add_argument("--no-cache", action="store_true")
    args = parser.parse_args()

    # Phase 1: fetch + score once
    scored = fetch_and_score(args.days, use_cache=not args.no_cache, workers=args.workers)

    # Phase 1.5: pre-fetch all price bars for every candidate ticker so the
    # sweep makes zero HTTP calls (reads only from _bar_cache).
    from config import MIN_STOCK_PRICE, MAX_HOLD_DAYS
    candidate_tickers = set()
    for row in scored:
        if row["sentiment"] == "bullish" and row["magnitude"] >= min(GRID["MIN_MAGNITUDE"]):
            for t in row["tickers"][:2]:
                if t not in ("BTC", "ETH") and is_valid_stock_ticker(t):
                    candidate_tickers.add(t)

    log.info("📡 Pre-fetching price bars for %d candidate tickers…", len(candidate_tickers))
    dates = sorted({r["created_at"].date() for r in scored})
    if dates:
        fetch_start = datetime(dates[0].year, dates[0].month, dates[0].day, tzinfo=timezone.utc) - timedelta(days=14)
        fetch_end   = datetime(dates[-1].year, dates[-1].month, dates[-1].day, tzinfo=timezone.utc) + timedelta(days=MAX_HOLD_DAYS * 2 + 10)
        for i, ticker in enumerate(sorted(candidate_tickers)):
            if ticker not in _bar_cache:
                _bar_cache[ticker] = get_stock_bars(ticker, fetch_start, fetch_end)
            if (i + 1) % 10 == 0:
                log.info("  prefetched %d / %d tickers", i + 1, len(candidate_tickers))
    log.info("✅ Price bar cache warm (%d entries)", len(_bar_cache))

    # Phase 2: grid search
    keys   = list(GRID.keys())
    combos = list(itertools.product(*[GRID[k] for k in keys]))
    log.info("🔬 Sweeping %d parameter combinations…", len(combos))

    results = []
    for i, values in enumerate(combos):
        params = dict(zip(keys, values))
        result = simulate(scored, params)
        results.append(result)
        if (i + 1) % 50 == 0 or (i + 1) == len(combos):
            log.info("  %d / %d done", i + 1, len(combos))

    # Rank by Sharpe, break ties by total P&L
    results.sort(key=lambda r: (r["sharpe"], r["total_pnl"]), reverse=True)

    # Write CSV
    csv_path = "tune_results.csv"
    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=results[0].keys())
        w.writeheader()
        w.writerows(results)
    log.info("💾 Results written to %s", csv_path)

    write_tune_report(results)

    log.info("─" * 60)
    log.info("TOP 5 COMBINATIONS (by Sharpe)")
    log.info("─" * 60)
    for i, r in enumerate(results[:5]):
        log.info(
            "#%d  Sharpe=%.2f  P&L=$%.0f  WinRate=%.1f%%  Trades=%d  "
            "MinMag=%.2f  BaseConf=%.2f  Slope=%.2f  Trail=%.0f%%  DTE=%d",
            i + 1, r["sharpe"], r["total_pnl"], r["win_rate"], r["trades"],
            r["MIN_MAGNITUDE"], r["BASE_CONFIDENCE"], r["CONFIDENCE_SLOPE"],
            r["TRAILING_STOP_PCT"] * 100, r["DTE_TARGET"],
        )
    log.info("─" * 60)
    log.info("✅ Done. Open tune_report.html to explore results.")
