"""
voltarget_backtest.py — Volatility-targeted position sizing vs mag*conf sizing.

Current sizing: position_usd = MAX_POSITION_USD * magnitude * confidence
Problem: a low-vol utility and a high-vol biotech get the same notional when scored
identically, but their dollar risk is very different.

Vol-targeting: position_usd = TARGET_RISK_USD / realized_vol
where realized_vol = annualised std of daily returns over the last 20 days.
This normalises risk: each position risks approximately the same dollar amount per
unit of vol, regardless of the underlying's price level or volatility.

Compares mag*conf sizing vs vol-targeting on both stock and PEAD strategies,
bull and bear windows. Best improvement = higher Sharpe at similar or lower maxDD.

Usage:  USE_YAHOO_BARS=1 .venv/bin/python voltarget_backtest.py
"""
import os, math
from datetime import datetime, timezone, timedelta
from pathlib import Path

os.environ.setdefault("USE_YAHOO_BARS", "1")

import tune_v2
from pead_backtest import is_earnings_article, STOCK_SLIPPAGE
from backtest import get_price_at, get_stock_bars, is_valid_stock_ticker, realized_vol
from benchmark import compute_stats
from regime_filter import build_regime
import config as _cfg

WINDOWS = [
    ("bull-meltup", datetime.now(timezone.utc) - timedelta(days=5), 180,
     "dual_score_cache.json"),
    ("2022-bear",   datetime(2022, 6, 30, tzinfo=timezone.utc), 90,
     "bear_dual_cache.json"),
]

VOL_WINDOW       = 20     # days to compute realized vol
TARGET_RISK_USD  = 100    # target $100 of 1-sigma daily risk per position
MAX_POSITION_USD = _cfg.MAX_POSITION_USD
STOCK_TRAIL      = 0.20
MAX_HOLD         = 45


def vol_position_size(ticker, entry_dt):
    """Return vol-targeted position size, capped at MAX_POSITION_USD."""
    bars = get_stock_bars(ticker,
                          entry_dt - timedelta(days=VOL_WINDOW * 2),
                          entry_dt + timedelta(days=1))
    tb = [b for b in bars if b["t"] <= entry_dt]
    if len(tb) < VOL_WINDOW + 1:
        return None
    closes = [b["c"] for b in tb[-VOL_WINDOW - 1:]]
    daily_rets = [(closes[i] / closes[i-1] - 1) for i in range(1, len(closes))]
    if len(daily_rets) < 5:
        return None
    vol = math.sqrt(sum(r**2 for r in daily_rets) / len(daily_rets))  # daily sigma
    if vol <= 0:
        return None
    price = closes[-1]
    # Position size: TARGET_RISK_USD = vol * price * shares → shares = TARGET / (vol * price)
    # position_usd = shares * price = TARGET_RISK_USD / vol
    pos = min(TARGET_RISK_USD / vol, MAX_POSITION_USD)
    return max(MAX_POSITION_USD * 0.10, pos)   # floor at 10% of max


def simulate_stock(ticker, entry_dt, position_usd, trail=STOCK_TRAIL, max_hold=MAX_HOLD):
    bars = get_stock_bars(ticker, entry_dt, entry_dt + timedelta(days=int(max_hold*1.6)+14))
    tb = [b for b in bars if entry_dt < b["t"]]
    if not tb or (tb[0]["t"] - entry_dt).days > 7:
        return None
    p0 = get_price_at(ticker, entry_dt)
    if not p0 or p0 < _cfg.MIN_STOCK_PRICE or p0 > 10_000:
        return None
    prev = p0
    for b in tb:
        if prev > 0 and abs(b["c"]/prev - 1) > 0.5:
            return None
        prev = b["c"]
    entry_fill = p0 * (1 + STOCK_SLIPPAGE/2)
    shares = position_usd / entry_fill
    peak = p0
    exit_price = tb[min(max_hold, len(tb))-1]["c"]
    for b in tb[:max_hold]:
        peak = max(peak, b["c"])
        if b["c"] <= peak*(1-trail):
            exit_price = b["c"]; break
    exit_fill = exit_price * (1 - STOCK_SLIPPAGE/2)
    pnl = (exit_fill - entry_fill) * shares
    return {"ticker": ticker, "entry_dt": entry_dt.isoformat(),
            "pnl_usd": pnl, "pnl_pct": (exit_fill/entry_fill - 1)*100}


def run(scored_rows, regime, sizing="mag_conf", earnings_only=False):
    def scale_mag(mag, conf):
        return max(MAX_POSITION_USD*0.10, MAX_POSITION_USD*mag*conf)

    trades, seen = [], {}
    for row in scored_rows:
        mag, conf = row["magnitude"], row["confidence"]
        req = _cfg.BASE_CONFIDENCE + (1 - mag) * _cfg.CONFIDENCE_SLOPE
        if mag < _cfg.MIN_MAGNITUDE or conf < req:
            continue
        if earnings_only and not is_earnings_article(row.get("headline", "")):
            continue
        d = row["created_at"].date()
        if regime and not regime(d):
            continue
        ds = seen.setdefault(d.isoformat(), set())
        for tk in row["tickers"][:2]:
            if tk in ds or tk in ("BTC", "ETH") or not is_valid_stock_ticker(tk):
                continue
            ds.add(tk)
            if sizing == "vol_target":
                pos = vol_position_size(tk, row["created_at"])
                if pos is None:
                    continue   # insufficient vol data
            else:
                pos = scale_mag(mag, conf)
            t = simulate_stock(tk, row["created_at"], pos)
            if t:
                trades.append(t)
    return trades


def line(tag, s, avg_pos=None):
    extra = f"  avg_pos=${avg_pos:,.0f}" if avg_pos else ""
    return (f"  {tag:40}  n={s['trades']:>4}  Sharpe={s['sharpe']:>6.2f}"
            f"  P&L=${s['total_pnl']:>9,.0f}  win={s['win_rate']:>4.1f}%"
            f"  maxDD=${s['max_dd']:>7,.0f}{extra}")


def main():
    print("VOLATILITY-TARGETING SIZING COMPARISON\n")
    print(f"Vol-target: ${TARGET_RISK_USD} of 1σ daily risk per trade")
    print(f"Mag*conf:   MAX_POSITION_USD × mag × conf\n")

    for label, end_dt, days, cache in WINDOWS:
        if not Path(cache).exists():
            print(f"  [{label}] cache not found — skipping"); continue
        print(f"═══ {label} ═══")
        scored = tune_v2.load_scored_from_dual_cache(Path(cache), "bull", end_dt, days)
        reg    = build_regime(end_dt, days, 200)

        # Stock strategy
        t_mag = run(scored, reg, sizing="mag_conf",   earnings_only=False)
        t_vol = run(scored, reg, sizing="vol_target", earnings_only=False)
        print(line("Stock — mag*conf sizing",    compute_stats(t_mag)))
        print(line("Stock — vol-target sizing",  compute_stats(t_vol)))

        # PEAD earnings only
        t_mag_p = run(scored, reg, sizing="mag_conf",   earnings_only=True)
        t_vol_p = run(scored, reg, sizing="vol_target", earnings_only=True)
        print(line("PEAD  — mag*conf sizing",    compute_stats(t_mag_p)))
        print(line("PEAD  — vol-target sizing",  compute_stats(t_vol_p)))
        print()

    print("═══ INTERPRETATION ═══")
    print("  Vol-targeting improves Sharpe if high-vol names are diluting returns")
    print("  Vol-targeting reduces maxDD if blow-ups come from high-vol names")
    print("  If no improvement: mag*conf is already adequate sizing")


if __name__ == "__main__":
    main()
