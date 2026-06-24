"""
drill_artifact.py — diagnose the bear-tune $14M artifact.

Reproduces the winning bear combo (DTE 14-21, Δ0.40, Conf0.65, Trail10%) on the
2022 training window, dumps the top-P&L trades, and checks each against the real
underlying move — to confirm whether the blow-up is qty explosion on cheap puts,
bad price data, or both.
"""
from datetime import datetime, timezone, timedelta
from pathlib import Path

import argparse
import backtest as _bt
import tune_v2
from backtest import get_price_at, get_stock_bars, is_valid_stock_ticker
import config as _cfg

ap = argparse.ArgumentParser()
ap.add_argument("--mode", choices=["bull", "bear"], default="bear")
args = ap.parse_args()
MODE = args.mode
OPTION_TYPE = "call" if MODE == "bull" else "put"

if MODE == "bear":
    PARAMS = {"TARGET_DELTA": 0.40, "MIN_MAGNITUDE": 0.25, "BASE_CONFIDENCE": 0.65,
              "TRAILING_STOP_PCT": 0.10}
    MIN_DTE, MAX_DTE = 14, 21
    END_DT = datetime(2022, 6, 30, tzinfo=timezone.utc)
    DAYS = 90
    CACHE = "bear_dual_cache.json"
else:  # bull — winning guarded combo, current window
    PARAMS = {"TARGET_DELTA": 0.50, "MIN_MAGNITUDE": 0.25, "BASE_CONFIDENCE": 0.65,
              "TRAILING_STOP_PCT": 0.15}
    MIN_DTE, MAX_DTE = 14, 21
    END_DT = datetime.now(timezone.utc) - timedelta(days=5)
    DAYS = 180
    CACHE = "dual_score_cache.json"

# Apply combo to backtest globals (same as tune_v2._apply_combo)
_bt.MIN_DAYS_TO_EXPIRY = MIN_DTE
_bt.MAX_DAYS_TO_EXPIRY = MAX_DTE
_bt.DTE_TARGET = round((MIN_DTE + MAX_DTE) / 2)
_bt.TARGET_DELTA = PARAMS["TARGET_DELTA"]
_bt.TRAILING_STOP_PCT = PARAMS["TRAILING_STOP_PCT"]

scored = tune_v2.load_scored_from_dual_cache(
    Path(CACHE), MODE, END_DT, DAYS)
train, _ = tune_v2.split_holdout(scored, 0.20)
print(f"\nReproducing winning {MODE} combo on {len(train)} training articles…\n")

def scale_usd(mag, conf):
    return max(_cfg.MAX_POSITION_USD * 0.10, _cfg.MAX_POSITION_USD * mag * conf)

trades = []
seen = {}
for row in train:
    mag, conf = row["magnitude"], row["confidence"]
    if mag < PARAMS["MIN_MAGNITUDE"]:
        continue
    if conf < PARAMS["BASE_CONFIDENCE"] + (1 - mag) * _cfg.CONFIDENCE_SLOPE:
        continue
    day = row["created_at"].date().isoformat()
    ds = seen.setdefault(day, set())
    for tk in row["tickers"][:2]:
        if tk in ds or tk in ("BTC", "ETH") or not is_valid_stock_ticker(tk):
            continue
        ds.add(tk)
        px = get_price_at(tk, row["created_at"])
        if not px or px < _cfg.MIN_STOCK_PRICE:
            continue
        t = _bt.simulate_option_pnl(tk, row["created_at"], px, scale_usd(mag, conf),
                                    {"magnitude": mag, "confidence": conf, "reasoning": ""},
                                    option_type=OPTION_TYPE)
        if t:
            trades.append(t)

trades.sort(key=lambda t: t["pnl_usd"], reverse=True)
total = sum(t["pnl_usd"] for t in trades)
print(f"Total trades: {len(trades)}  Total P&L: ${total:,.0f}")
print(f"Top 15 P&L share: {sum(t['pnl_usd'] for t in trades[:15])/total*100:.0f}%\n")
print(f"{'ticker':7} {'entry$':>8} {'strike':>8} {'entry_prem':>10} {'qty':>5} {'exit_prem':>9} {'pnl':>12}  realmove")
for t in trades[:15]:
    e = datetime.fromisoformat(t["entry_dt"])
    bars = get_stock_bars(t["ticker"], e, e + timedelta(days=int(MAX_DTE * 1.5) + 7))
    es = t["entry_stock_price"]
    if OPTION_TYPE == "call":  # calls win when stock rises → show peak high
        ext = max((b["c"] for b in bars), default=0)
        move = (ext / es - 1) * 100 if es else 0
        tag = f"high ${ext:.2f} ({move:+.0f}%)"
    else:                       # puts win when stock falls → show low
        ext = min((b["c"] for b in bars), default=0)
        move = (ext / es - 1) * 100 if es else 0
        tag = f"low ${ext:.2f} ({move:+.0f}%)"
    print(f"{t['ticker']:7} {es:8.2f} {t['strike']:8.2f} {t['entry_premium']:10.3f} "
          f"{t['qty']:5d} {t['exit_premium']:9.2f} {t['pnl_usd']:12,.0f}  {tag}")
