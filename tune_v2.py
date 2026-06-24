"""
tune_v2.py — Parameter sweep with:
  • dual_score_cache.json input  (bull or bear side)
  • Black-Scholes option pricing  (via backtest.simulate_option_pnl)
  • MIN/MAX_DAYS_TO_EXPIRY pairs  instead of a single DTE_TARGET
  • 80/20 holdout validation      (tune on first 80%, validate top-5 on last 20%)
  • --mode bull | bear            (which side of dual cache to trade)
  • --end-date / --days           (target any historical regime)

Usage:
    # Bull run — uses dual_score_cache.json (built by rescore_all.py)
    .venv/bin/python tune_v2.py --mode bull >> tune_bull.log 2>&1

    # Bear run — 2022-H1 bear regime, 90-day window, separate cache
    .venv/bin/python tune_v2.py --mode bear \\
        --end-date 2022-06-30 --days 90 \\
        --cache-file bear_dual_cache.json >> tune_bear.log 2>&1

Outputs (per run):
    tune_v2_{mode}_results.csv
    tune_v2_{mode}_report.html
    tune_v2_{mode}_holdout.csv   (top-5 combos evaluated on holdout)

Grid size: 3 × 3 × 3 × 3 × 3 = 243 combos — runs in minutes once bars are cached.
"""

import argparse
import csv
import itertools
import json
import logging
import math
import os
import sys
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path

from dotenv import load_dotenv
load_dotenv()

# ── Logging ───────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

# ── Import backtest + config (we'll monkey-patch module-level vars per combo) ─
import backtest as _bt
import config as _cfg
from backtest import get_stock_bars, get_price_at, is_valid_stock_ticker

# ── In-process bar cache (keyed by ticker → full list of bars for the window) ─
_bar_cache: dict[str, list] = {}

def _cached_get_stock_bars(ticker, start, end):
    if ticker not in _bar_cache:
        _bar_cache[ticker] = get_stock_bars(ticker, start, end)
    bars = _bar_cache[ticker]
    # Make sure start/end are tz-aware before slicing
    if bars and bars[0]["t"].tzinfo is None:
        return [b for b in bars]  # shouldn't happen; return as-is
    if start.tzinfo is None:
        start = start.replace(tzinfo=timezone.utc)
    if end.tzinfo is None:
        end = end.replace(tzinfo=timezone.utc)
    return [b for b in bars if start <= b["t"] <= end]

# Redirect backtest's bar fetching through our cache
_bt.get_stock_bars = _cached_get_stock_bars

# ═══════════════════════════════════════════════════════════════════════════════
# PARAMETER GRID
# ═══════════════════════════════════════════════════════════════════════════════

GRID = {
    # Refocused 2026-06-02 for the reliable-data re-tune: prior optima sat at the
    # grid boundaries (DTE 14-21, Δ0.50, Conf0.65), so probe BEYOND them and drop
    # the consistently-losing low values (Δ0.30, Conf0.45).
    # (MIN_DAYS_TO_EXPIRY, MAX_DAYS_TO_EXPIRY)
    "DTE_PAIR":           [(14, 21), (21, 30), (28, 45)],
    "TARGET_DELTA":       [0.40, 0.50, 0.60],
    "MIN_MAGNITUDE":      [0.25, 0.35, 0.45],
    "BASE_CONFIDENCE":    [0.55, 0.65, 0.70],
    "TRAILING_STOP_PCT":  [0.10, 0.15, 0.20],
}

# ═══════════════════════════════════════════════════════════════════════════════
# LOAD PRE-SCORED ARTICLES FROM dual_score_cache.json
# ═══════════════════════════════════════════════════════════════════════════════

def load_scored_from_dual_cache(
    cache_file: Path,
    mode: str,
    end_dt: datetime,
    days: int,
) -> list[dict]:
    """
    Read pre-scored articles from dual_score_cache.json (or bear_dual_cache.json).
    Filters to the requested date window, returns the bull or bear side.
    Articles are sorted by created_at (oldest first) for deterministic holdout split.
    """
    if not cache_file.exists():
        log.error("Cache file not found: %s", cache_file)
        log.error("Run: .venv/bin/python rescore_all.py --delay 0 [--output-file %s]", cache_file)
        sys.exit(1)

    raw = json.loads(cache_file.read_text())
    start_dt = end_dt - timedelta(days=days)
    side = "bullish" if mode == "bull" else "bearish"
    option_type = "call" if mode == "bull" else "put"

    log.info("📦 Loaded %d entries from %s", len(raw), cache_file)
    log.info("   Filtering %s side  |  window: %s → %s",
             side, start_dt.date(), end_dt.date())

    scored = []
    skipped_date = skipped_failed = skipped_no_ticker = 0

    for entry in raw.values():
        article = entry.get("_article", {})
        created_at_str = article.get("created_at", "")
        if not created_at_str:
            continue
        try:
            created_at = datetime.fromisoformat(
                str(created_at_str).replace("Z", "+00:00")
            )
            if created_at.tzinfo is None:
                created_at = created_at.replace(tzinfo=timezone.utc)
        except Exception:
            continue

        if not (start_dt <= created_at <= end_dt):
            skipped_date += 1
            continue

        signal = entry.get(side, {})
        reasoning = signal.get("reasoning", "")
        if reasoning == "SCORE_FAILED":
            skipped_failed += 1
            continue

        tickers = signal.get("tickers", [])
        if not tickers:
            skipped_no_ticker += 1
            continue

        scored.append({
            "headline":    article.get("headline", ""),
            "created_at":  created_at,
            "magnitude":   float(signal.get("magnitude", 0.0)),
            "confidence":  float(signal.get("confidence", 0.0)),
            "tickers":     tickers,
            "reasoning":   reasoning,
            "option_type": option_type,
        })

    scored.sort(key=lambda x: x["created_at"])
    log.info("   → %d scored articles with %s tickers  "
             "(skipped: %d out-of-window, %d failed, %d no-ticker)",
             len(scored), side, skipped_date, skipped_failed, skipped_no_ticker)
    return scored


# ═══════════════════════════════════════════════════════════════════════════════
# SPLIT  training (first 80%) / holdout (last 20%)
# ═══════════════════════════════════════════════════════════════════════════════

def split_holdout(scored: list[dict], holdout_pct: float = 0.20):
    n_train = max(1, int(len(scored) * (1.0 - holdout_pct)))
    train   = scored[:n_train]
    holdout = scored[n_train:]
    if train:
        log.info("Split: train=%d articles (%s → %s)  holdout=%d articles (%s → %s)",
                 len(train),
                 train[0]["created_at"].date(), train[-1]["created_at"].date(),
                 len(holdout),
                 holdout[0]["created_at"].date() if holdout else "–",
                 holdout[-1]["created_at"].date() if holdout else "–")
    return train, holdout


# ═══════════════════════════════════════════════════════════════════════════════
# PRE-WARM BAR CACHE
# ═══════════════════════════════════════════════════════════════════════════════

def prefetch_bars(scored: list[dict], min_magnitude: float):
    """Fetch all bars once so the grid loop makes zero HTTP calls."""
    # Collect candidate tickers from the full dataset (train + holdout)
    candidate_tickers: set[str] = set()
    for row in scored:
        if row["magnitude"] >= min_magnitude:
            for t in row["tickers"][:2]:
                if t not in ("BTC", "ETH") and is_valid_stock_ticker(t):
                    candidate_tickers.add(t)

    if not candidate_tickers:
        log.warning("No candidate tickers found — check magnitude threshold / cache contents")
        return

    dates = sorted(r["created_at"].date() for r in scored)
    fetch_start = (
        datetime(dates[0].year, dates[0].month, dates[0].day, tzinfo=timezone.utc)
        - timedelta(days=30)          # extra window for base_iv_for() realized-vol calc
    )
    fetch_end = (
        datetime(dates[-1].year, dates[-1].month, dates[-1].day, tzinfo=timezone.utc)
        + timedelta(days=_cfg.MAX_HOLD_DAYS * 2 + 10)
    )

    log.info("📡 Pre-fetching bars for %d candidate tickers (%s → %s)…",
             len(candidate_tickers), fetch_start.date(), fetch_end.date())

    # Also patch backtest's MAX_HOLD_DAYS window here so bars cover it
    for i, ticker in enumerate(sorted(candidate_tickers), start=1):
        if ticker not in _bar_cache:
            _bar_cache[ticker] = get_stock_bars(ticker, fetch_start, fetch_end)
            time.sleep(0.05)          # gentle rate-limit during prefetch
        if i % 20 == 0:
            log.info("  prefetched %d / %d tickers", i, len(candidate_tickers))

    log.info("✅ Bar cache warm — %d tickers, zero HTTP during grid sweep", len(_bar_cache))


# ═══════════════════════════════════════════════════════════════════════════════
# PATCH backtest MODULE for one combo, then simulate all trades
# ═══════════════════════════════════════════════════════════════════════════════

def _apply_combo(params: dict):
    """Monkey-patch backtest module-level globals for this combo."""
    min_dte, max_dte = params["DTE_PAIR"]
    _bt.MIN_DAYS_TO_EXPIRY = min_dte
    _bt.MAX_DAYS_TO_EXPIRY = max_dte
    _bt.DTE_TARGET         = round((min_dte + max_dte) / 2)
    _bt.TARGET_DELTA       = params["TARGET_DELTA"]
    _bt.TRAILING_STOP_PCT  = params["TRAILING_STOP_PCT"]
    # These were imported from config at module load, so patch both places:
    _bt.MIN_MAGNITUDE      = params["MIN_MAGNITUDE"]
    _bt.BASE_CONFIDENCE    = params["BASE_CONFIDENCE"]


def simulate_on_split(scored_split: list[dict], params: dict) -> dict:
    """
    Run one parameter combo against a list of pre-scored articles.
    Returns a dict of performance metrics.
    """
    _apply_combo(params)

    MIN_MAGNITUDE    = params["MIN_MAGNITUDE"]
    BASE_CONFIDENCE  = params["BASE_CONFIDENCE"]
    CONFIDENCE_SLOPE = _cfg.CONFIDENCE_SLOPE   # not swept — use live config value
    TRAILING_STOP    = params["TRAILING_STOP_PCT"]

    def signal_passes(mag, conf) -> bool:
        if mag < MIN_MAGNITUDE:
            return False
        required = BASE_CONFIDENCE + (1.0 - mag) * CONFIDENCE_SLOPE
        return conf >= required

    def scale_usd(mag, conf) -> float:
        return max(_cfg.MAX_POSITION_USD * 0.10,
                   _cfg.MAX_POSITION_USD * mag * conf)

    seen_tickers_by_day: dict[str, set] = {}
    trade_rows   = []
    daily_loss   = {}
    halted_days  = set()
    running_pnl  = 0.0
    equity_curve = []

    for row in scored_split:
        mag    = row["magnitude"]
        conf   = row["confidence"]
        if not signal_passes(mag, conf):
            continue

        created_at = row["created_at"]
        trade_day  = created_at.date().isoformat()
        if trade_day in halted_days:
            continue

        day_set = seen_tickers_by_day.setdefault(trade_day, set())

        for ticker in row["tickers"][:2]:
            if ticker in day_set:
                continue
            if ticker in ("BTC", "ETH"):
                continue
            if not is_valid_stock_ticker(ticker):
                continue
            day_set.add(ticker)

            entry_price = get_price_at(ticker, created_at)
            if not entry_price or entry_price < _cfg.MIN_STOCK_PRICE:
                continue

            position_usd = scale_usd(mag, conf)
            signal = {
                "magnitude":  mag,
                "confidence": conf,
                "reasoning":  row["reasoning"],
            }

            trade = _bt.simulate_option_pnl(
                ticker, created_at, entry_price, position_usd,
                signal, option_type=row["option_type"],
            )
            if not trade:
                continue

            # Daily loss circuit-breaker
            day_loss = daily_loss.get(trade_day, 0.0)
            if trade["pnl_usd"] < 0:
                day_loss += abs(trade["pnl_usd"])
                daily_loss[trade_day] = day_loss
                if day_loss >= _cfg.DAILY_LOSS_LIMIT:
                    halted_days.add(trade_day)

            running_pnl += trade["pnl_usd"]
            trade["cumulative_pnl"] = round(running_pnl, 2)
            trade_rows.append(trade)
            equity_curve.append({"date": trade_day, "cumulative_pnl": round(running_pnl, 2)})

    return _metrics(trade_rows, equity_curve, params)


def _metrics(trade_rows: list, equity_curve: list, params: dict) -> dict:
    n      = len(trade_rows)
    wins   = [t for t in trade_rows if t["pnl_usd"] > 0]
    losses = [t for t in trade_rows if t["pnl_usd"] <= 0]

    total_pnl = round(sum(t["pnl_usd"] for t in trade_rows), 2)
    win_rate  = len(wins) / n * 100 if n else 0
    avg_win   = sum(t["pnl_usd"] for t in wins)   / len(wins)   if wins   else 0.0
    avg_loss  = sum(t["pnl_usd"] for t in losses)  / len(losses) if losses else 0.0
    pf        = round(abs(avg_win / avg_loss), 2)  if avg_loss   else 0.0

    peak_eq = max_dd = 0.0
    for pt in equity_curve:
        eq      = pt["cumulative_pnl"]
        peak_eq = max(peak_eq, eq)
        max_dd  = max(max_dd,  peak_eq - eq)

    if n > 1:
        returns = [t["pnl_pct"] for t in trade_rows]
        mean_r  = sum(returns) / len(returns)
        std_r   = math.sqrt(sum((r - mean_r) ** 2 for r in returns) / len(returns)) or 1e-9
        sharpe  = round((mean_r / std_r) * math.sqrt(252), 2)
    else:
        sharpe = 0.0

    min_dte, max_dte = params["DTE_PAIR"]
    return {
        "MIN_DTE":           min_dte,
        "MAX_DTE":           max_dte,
        "TARGET_DELTA":      params["TARGET_DELTA"],
        "MIN_MAGNITUDE":     params["MIN_MAGNITUDE"],
        "BASE_CONFIDENCE":   params["BASE_CONFIDENCE"],
        "TRAILING_STOP_PCT": params["TRAILING_STOP_PCT"],
        "trades":            n,
        "total_pnl":         total_pnl,
        "win_rate":          round(win_rate, 1),
        "avg_win":           round(avg_win, 2),
        "avg_loss":          round(avg_loss, 2),
        "profit_factor":     pf,
        "max_drawdown":      round(max_dd, 2),
        "sharpe":            sharpe,
        "halted_days":       len({r["date"] for r in equity_curve}
                                  if equity_curve else set()),
    }


# ═══════════════════════════════════════════════════════════════════════════════
# HTML REPORT
# ═══════════════════════════════════════════════════════════════════════════════

REPORT_COLS = [
    ("MIN_DTE",           "Min DTE"),
    ("MAX_DTE",           "Max DTE"),
    ("TARGET_DELTA",      "Delta"),
    ("MIN_MAGNITUDE",     "Min Mag"),
    ("BASE_CONFIDENCE",   "Base Conf"),
    ("TRAILING_STOP_PCT", "Trail Stop"),
    ("trades",            "Trades"),
    ("total_pnl",         "Total P&L"),
    ("win_rate",          "Win %"),
    ("avg_win",           "Avg Win"),
    ("avg_loss",          "Avg Loss"),
    ("profit_factor",     "PF"),
    ("max_drawdown",      "Max DD"),
    ("sharpe",            "Sharpe"),
]

def _row_html(r: dict, rank: int, highlight_style: str = "") -> str:
    pnl_color = "#2ecc71" if r["total_pnl"] >= 0 else "#e74c3c"
    html = f"<tr style='{highlight_style}'><td style='color:#aaa'>{rank}</td>"
    for key, _ in REPORT_COLS:
        val   = r[key]
        style = ""
        if key == "total_pnl":
            style = f"color:{pnl_color};font-weight:600"
            val   = f"${val:+,.0f}"
        elif key == "win_rate":
            val = f"{val}%"
        elif key in ("avg_win", "avg_loss"):
            val = f"${val:+,.0f}"
        elif key == "max_drawdown":
            val   = f"${val:,.0f}"
            style = "color:#e74c3c"
        elif key == "sharpe":
            style = "color:#3498db;font-weight:600"
        elif key == "TRAILING_STOP_PCT":
            val = f"{int(val*100)}%"
        elif key == "TARGET_DELTA":
            val = f"{val:.2f}"
        html += f"<td style='{style}'>{val}</td>"
    html += "</tr>"
    return html


def write_report(
    train_results: list[dict],
    holdout_rows:  list[dict],
    mode: str,
    path: str,
):
    header_html = "<th>Rank</th>" + "".join(f"<th>{lbl}</th>" for _, lbl in REPORT_COLS)

    train_rows_html = ""
    for i, r in enumerate(train_results):
        style = ("background:#1a3a2a;" if i == 0 else
                 "background:#162a3a;" if i == 1 else
                 "background:#2a2a1a;" if i == 2 else "")
        train_rows_html += _row_html(r, i + 1, style)

    holdout_rows_html = ""
    for i, r in enumerate(holdout_rows):
        style = "background:#1a3a2a;" if i == 0 else ""
        holdout_rows_html += _row_html(r, i + 1, style)

    sharpes = [r["sharpe"]    for r in train_results]
    pnls    = [r["total_pnl"] for r in train_results]

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<title>tune_v2 — {mode.upper()} results</title>
<script src="https://cdnjs.cloudflare.com/ajax/libs/Chart.js/4.4.1/chart.umd.min.js"></script>
<style>
  *{{box-sizing:border-box;margin:0;padding:0}}
  body{{font-family:-apple-system,BlinkMacSystemFont,"Segoe UI",sans-serif;
        background:#0f1117;color:#e0e0e0;padding:24px}}
  h1{{font-size:1.5em;margin-bottom:4px;color:#fff}}
  h2{{font-size:1.0em;color:#aaa;margin:28px 0 10px}}
  .subtitle{{color:#888;font-size:.85em;margin-bottom:24px}}
  .charts{{display:grid;grid-template-columns:1fr 1fr;gap:20px;margin-bottom:28px}}
  .chart-box{{background:#1a1d27;border-radius:10px;padding:20px}}
  .chart-box h3{{font-size:.9em;color:#aaa;margin-bottom:14px}}
  table{{width:100%;border-collapse:collapse;background:#1a1d27;
         border-radius:10px;overflow:hidden;font-size:.8em;margin-bottom:28px}}
  th{{background:#252836;padding:8px 10px;text-align:left;
      color:#aaa;font-weight:600;font-size:.75em;text-transform:uppercase;white-space:nowrap}}
  td{{padding:7px 10px;border-bottom:1px solid #1e2130;white-space:nowrap}}
  tr:last-child td{{border-bottom:none}}
  tr:hover td{{background:#21232f}}
  .badge{{display:inline-block;background:#1a4a2a;color:#2ecc71;
           border-radius:4px;padding:2px 8px;font-size:.75em;margin-left:8px}}
</style>
</head>
<body>
<h1>📊 tune_v2 — {mode.upper()} Mode</h1>
<div class="subtitle">{len(train_results)} combinations · ranked by Sharpe on training set</div>

<div class="charts">
  <div class="chart-box">
    <h3>Training Sharpe Distribution</h3>
    <canvas id="sharpeChart"></canvas>
  </div>
  <div class="chart-box">
    <h3>Training P&L Distribution</h3>
    <canvas id="pnlChart"></canvas>
  </div>
</div>

<h2>Training Set — All Combos</h2>
<table>
  <thead><tr>{header_html}</tr></thead>
  <tbody>{train_rows_html}</tbody>
</table>

<h2>Holdout Validation — Top 5 <span class="badge">20% hold-back</span></h2>
<table>
  <thead><tr>{header_html}</tr></thead>
  <tbody>{holdout_rows_html}</tbody>
</table>

<script>
const sharpes={json.dumps(sharpes)};
const pnls={json.dumps(pnls)};
const labels=Array.from({{length:sharpes.length}},(_,i)=>i+1);
new Chart(document.getElementById('sharpeChart'),{{
  type:'bar',data:{{labels,datasets:[{{data:sharpes,
    backgroundColor:sharpes.map(v=>v>=0?'#3498db88':'#e74c3c88'),
    borderColor:sharpes.map(v=>v>=0?'#3498db':'#e74c3c'),borderWidth:1}}]}},
  options:{{responsive:true,plugins:{{legend:{{display:false}}}},
    scales:{{x:{{ticks:{{display:false}},grid:{{color:'#252836'}}}},
             y:{{ticks:{{color:'#666'}},grid:{{color:'#252836'}}}}}}}}
}});
new Chart(document.getElementById('pnlChart'),{{
  type:'bar',data:{{labels,datasets:[{{data:pnls,
    backgroundColor:pnls.map(v=>v>=0?'#2ecc7188':'#e74c3c88'),
    borderColor:pnls.map(v=>v>=0?'#2ecc71':'#e74c3c'),borderWidth:1}}]}},
  options:{{responsive:true,plugins:{{legend:{{display:false}}}},
    scales:{{x:{{ticks:{{display:false}},grid:{{color:'#252836'}}}},
             y:{{ticks:{{color:'#666',callback:v=>'$'+v.toLocaleString()}},
                grid:{{color:'#252836'}}}}}}}}
}});
</script>
</body>
</html>"""

    Path(path).write_text(html)
    log.info("📊 HTML report → %s", path)


# ═══════════════════════════════════════════════════════════════════════════════
# ENTRY POINT
# ═══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="tune_v2 — dual-cache parameter sweep")
    parser.add_argument("--mode",         choices=["bull", "bear"], default="bull",
                        help="bull = long calls on bullish signals; bear = long puts on bearish signals")
    parser.add_argument("--days",         type=int,   default=180,
                        help="Window size in days (default 180)")
    parser.add_argument("--end-date",     type=str,   default=None,
                        help="Window end date YYYY-MM-DD (default: today-5d)")
    parser.add_argument("--cache-file",   type=str,   default=None,
                        help="Score cache file (default: dual_score_cache.json for bull, "
                             "bear_dual_cache.json for bear)")
    parser.add_argument("--holdout-pct",  type=float, default=0.20,
                        help="Fraction of data to hold back for validation (default 0.20)")
    parser.add_argument("--top-n",        type=int,   default=5,
                        help="Number of top combos to validate on holdout (default 5)")
    args = parser.parse_args()

    # ── Resolve defaults ─────────────────────────────────────────────────────
    if args.cache_file:
        cache_file = Path(args.cache_file)
    else:
        cache_file = Path("dual_score_cache.json") if args.mode == "bull" \
                     else Path("bear_dual_cache.json")

    if args.end_date:
        end_dt = datetime.strptime(args.end_date, "%Y-%m-%d").replace(tzinfo=timezone.utc)
    else:
        end_dt = datetime.now(timezone.utc) - timedelta(days=5)

    mode = args.mode
    log.info("═" * 60)
    log.info("tune_v2  MODE=%s  cache=%s  window=%s→%s (%dd)  holdout=%.0f%%",
             mode.upper(), cache_file,
             (end_dt - timedelta(days=args.days)).date(), end_dt.date(),
             args.days, args.holdout_pct * 100)
    log.info("═" * 60)

    # ── Phase 1: Load scored articles ────────────────────────────────────────
    all_scored = load_scored_from_dual_cache(cache_file, mode, end_dt, args.days)
    if len(all_scored) < 10:
        log.error("Too few articles (%d) — is the cache built for this window?", len(all_scored))
        sys.exit(1)

    # ── Phase 2: Holdout split ───────────────────────────────────────────────
    train_scored, holdout_scored = split_holdout(all_scored, args.holdout_pct)

    # ── Phase 3: Pre-warm bar cache (train + holdout, all tickers) ───────────
    min_mag_floor = min(GRID["MIN_MAGNITUDE"])
    prefetch_bars(all_scored, min_mag_floor)

    # ── Phase 4: Grid search on training set ─────────────────────────────────
    keys   = list(GRID.keys())
    combos = list(itertools.product(*[GRID[k] for k in keys]))
    log.info("🔬 Sweeping %d combos on training set (%d articles)…",
             len(combos), len(train_scored))

    train_results = []
    for i, values in enumerate(combos, start=1):
        params = dict(zip(keys, values))
        result = simulate_on_split(train_scored, params)
        train_results.append(result)
        if i % 50 == 0 or i == len(combos):
            log.info("  %d / %d  best Sharpe so far: %.2f",
                     i, len(combos),
                     max(r["sharpe"] for r in train_results))

    train_results.sort(key=lambda r: (r["sharpe"], r["total_pnl"]), reverse=True)

    # ── Phase 5: Validate top-N on holdout ───────────────────────────────────
    log.info("─" * 60)
    log.info("Validating top-%d combos on holdout (%d articles)…",
             args.top_n, len(holdout_scored))

    holdout_results = []
    for r in train_results[:args.top_n]:
        # Rebuild params dict (DTE_PAIR needs to be reconstructed)
        params = {
            "DTE_PAIR":           (r["MIN_DTE"], r["MAX_DTE"]),
            "TARGET_DELTA":       r["TARGET_DELTA"],
            "MIN_MAGNITUDE":      r["MIN_MAGNITUDE"],
            "BASE_CONFIDENCE":    r["BASE_CONFIDENCE"],
            "TRAILING_STOP_PCT":  r["TRAILING_STOP_PCT"],
        }
        h = simulate_on_split(holdout_scored, params)
        h["train_sharpe"] = r["sharpe"]
        h["train_pnl"]    = r["total_pnl"]
        holdout_results.append(h)
        log.info("  [holdout] DTE=%d-%d Δ=%.2f MinMag=%.2f Conf=%.2f Trail=%.0f%%  "
                 "→ Sharpe=%.2f  P&L=$%.0f  (train Sharpe=%.2f)",
                 r["MIN_DTE"], r["MAX_DTE"], r["TARGET_DELTA"],
                 r["MIN_MAGNITUDE"], r["BASE_CONFIDENCE"], r["TRAILING_STOP_PCT"]*100,
                 h["sharpe"], h["total_pnl"], r["sharpe"])

    # ── Phase 6: Write outputs ───────────────────────────────────────────────
    tag = f"tune_v2_{mode}"

    csv_path = f"{tag}_results.csv"
    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=train_results[0].keys())
        w.writeheader()
        w.writerows(train_results)
    log.info("💾 Training results → %s", csv_path)

    holdout_csv = f"{tag}_holdout.csv"
    with open(holdout_csv, "w", newline="") as f:
        fieldnames = list(holdout_results[0].keys()) if holdout_results else []
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(holdout_results)
    log.info("💾 Holdout results  → %s", holdout_csv)

    write_report(train_results, holdout_results, mode, f"{tag}_report.html")

    # ── Summary ──────────────────────────────────────────────────────────────
    log.info("═" * 60)
    log.info("TRAINING TOP 5  (mode=%s)", mode.upper())
    log.info("─" * 60)
    for i, r in enumerate(train_results[:5], start=1):
        log.info("#%d  Sharpe=%.2f  P&L=$%.0f  WinRate=%.1f%%  Trades=%d  "
                 "DTE=%d-%d  Δ=%.2f  MinMag=%.2f  Conf=%.2f  Trail=%.0f%%",
                 i, r["sharpe"], r["total_pnl"], r["win_rate"], r["trades"],
                 r["MIN_DTE"], r["MAX_DTE"], r["TARGET_DELTA"],
                 r["MIN_MAGNITUDE"], r["BASE_CONFIDENCE"],
                 r["TRAILING_STOP_PCT"] * 100)

    log.info("─" * 60)
    log.info("HOLDOUT TOP 5  (mode=%s)", mode.upper())
    log.info("─" * 60)
    for i, r in enumerate(holdout_results, start=1):
        log.info("#%d  Sharpe=%.2f  P&L=$%.0f  WinRate=%.1f%%  Trades=%d  "
                 "DTE=%d-%d  Δ=%.2f  MinMag=%.2f  Conf=%.2f  Trail=%.0f%%  "
                 "(train Sharpe=%.2f)",
                 i, r["sharpe"], r["total_pnl"], r["win_rate"], r["trades"],
                 r["MIN_DTE"], r["MAX_DTE"], r["TARGET_DELTA"],
                 r["MIN_MAGNITUDE"], r["BASE_CONFIDENCE"],
                 r["TRAILING_STOP_PCT"] * 100, r["train_sharpe"])

    log.info("═" * 60)
    log.info("✅ Done. Open %s_report.html to explore.", tag)
