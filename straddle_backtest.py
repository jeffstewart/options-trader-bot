"""
straddle_backtest.py — Earnings straddle backtest.

Buy an ATM straddle (call + put, same strike, same expiry) on the earnings
announcement day. Close it 1 day later. Profitable if the actual move exceeds
the implied move priced into the options.

This is the opposite bet from sell-premium: instead of selling the elevated IV,
we BUY the straddle hoping the stock moves MORE than the market implies.

Key insight: our calibrated NEWS_IV_MULTIPLIER = 1.10 means options are priced 10%
above base vol on news days. For straddle buyers to profit, the realised move must
exceed the implied move by enough to overcome the entry/exit spread costs.

Method:
  1. Filter to earnings-related articles (is_earnings_article)
  2. On the article day, price an ATM straddle using BS at elevated entry IV
  3. One trading day later, price it at post-crush IV + intrinsic value
  4. P&L = (exit straddle value − entry straddle cost) × 100 × qty − spreads

Compared against:
  - Long call only (bullish earnings signal)
  - Long put only (bearish earnings signal)
  - Straddle baseline (all articles, no earnings filter)

Usage:  USE_YAHOO_BARS=1 .venv/bin/python straddle_backtest.py
"""
import os
from datetime import datetime, timezone, timedelta
from pathlib import Path

os.environ.setdefault("USE_YAHOO_BARS", "1")

import tune_v2
from pead_backtest import is_earnings_article
from backtest import (get_price_at, get_stock_bars, is_valid_stock_ticker, base_iv_for,
                      _spread_pct, LIQUID_UNDERLYINGS,
                      NEWS_IV_MULTIPLIER, IV_CRUSH_HALFLIFE_DAYS, RISK_FREE_RATE)
from pricing import bs_call_price, bs_put_price, strike_for_delta, iv_crush_path
from benchmark import compute_stats
from regime_filter import build_regime
import config as _cfg

WINDOWS = [
    ("bull-meltup", datetime.now(timezone.utc) - timedelta(days=5), 180,
     "dual_score_cache.json"),
    ("2022-bear",   datetime(2022, 6, 30, tzinfo=timezone.utc), 90,
     "bear_dual_cache.json"),
]

STRADDLE_DTE     = 21       # entry options expiry (3 weeks out)
HOLD_DAYS        = 1        # close 1 trading day after entry
MAX_POSITION_USD = _cfg.MAX_POSITION_USD


def simulate_straddle(ticker, entry_dt, position_usd, hold_days=HOLD_DAYS):
    """
    Buy ATM straddle at entry, close hold_days later.
    Returns pnl_usd or None if insufficient data.
    """
    r = RISK_FREE_RATE
    s0 = get_price_at(ticker, entry_dt)
    if not s0 or s0 < _cfg.MIN_STOCK_PRICE or s0 > 10_000:
        return None

    base_iv  = base_iv_for(ticker, entry_dt)
    entry_iv = base_iv * NEWS_IV_MULTIPLIER
    T0       = STRADDLE_DTE / 365.0
    K        = round(s0, 2)   # ATM strike

    call_mid = bs_call_price(s0, K, T0, entry_iv, r)
    put_mid  = bs_put_price(s0, K, T0, entry_iv, r)
    straddle_mid = call_mid + put_mid
    if straddle_mid <= 0.05:
        return None

    # Entry cost: pay spread on both legs
    hs_c = _spread_pct(call_mid, ticker, 1.0) / 2
    hs_p = _spread_pct(put_mid,  ticker, 1.0) / 2
    entry_cost = call_mid * (1 + hs_c) + put_mid * (1 + hs_p)
    if entry_cost <= 0:
        return None

    max_loss_contract = entry_cost * 100
    qty = max(1, int(position_usd / max_loss_contract))

    # Get next-day price
    bars = get_stock_bars(ticker, entry_dt,
                          entry_dt + timedelta(days=hold_days + 5))
    tb = [b for b in bars if entry_dt < b["t"]]
    if not tb:
        return None
    # Data quality guard
    if abs(tb[0]["c"] / s0 - 1) > 0.5:
        return None

    exit_bar = tb[min(hold_days, len(tb)) - 1]
    s1 = exit_bar["c"]
    dh = max((exit_bar["t"] - entry_dt).days, 1)

    # Exit IV (post-crush)
    exit_iv = iv_crush_path(base_iv, dh, NEWS_IV_MULTIPLIER, IV_CRUSH_HALFLIFE_DAYS)
    T1 = max((STRADDLE_DTE - dh) / 365.0, 0.5 / 365.0)

    exit_call = bs_call_price(s1, K, T1, exit_iv, r)
    exit_put  = bs_put_price(s1, K, T1, exit_iv, r)
    exit_straddle = exit_call + exit_put

    # Exit proceeds: sell spread on both legs
    exit_proceeds = exit_call * (1 - hs_c) + exit_put * (1 - hs_p)

    pnl = (exit_proceeds - entry_cost) * 100 * qty
    pnl_pct = (exit_proceeds / entry_cost - 1) * 100
    move_pct = abs(s1 / s0 - 1) * 100
    implied_move = straddle_mid / s0 * 100

    return {
        "ticker":        ticker,
        "entry_dt":      entry_dt.isoformat(),
        "pnl_usd":       pnl,
        "pnl_pct":       pnl_pct,
        "move_pct":      move_pct,
        "implied_move%": implied_move,
        "beat_implied":  move_pct > implied_move,
    }


def _scale(mag, conf):
    return max(MAX_POSITION_USD * 0.10, MAX_POSITION_USD * mag * conf)


def run(scored_rows, regime=None, earnings_filter=True, min_mag=0.35, min_conf=0.55):
    trades, seen = [], {}
    for row in scored_rows:
        mag, conf = row["magnitude"], row["confidence"]
        req = min_conf + (1 - mag) * _cfg.CONFIDENCE_SLOPE
        if mag < min_mag or conf < req:
            continue
        if earnings_filter and not is_earnings_article(row.get("headline", "")):
            continue
        d = row["created_at"].date()
        if regime and not regime(d):
            continue
        ds = seen.setdefault(d.isoformat(), set())
        for tk in row["tickers"][:2]:
            if tk in ds or tk in ("BTC", "ETH") or not is_valid_stock_ticker(tk):
                continue
            ds.add(tk)
            t = simulate_straddle(tk, row["created_at"], _scale(mag, conf))
            if t:
                trades.append(t)
    return trades


def line(tag, trades):
    if not trades:
        return f"  {tag:42}  n=   0  (no data)"
    s = compute_stats(trades)
    beat = sum(1 for t in trades if t.get("beat_implied", False))
    avg_move    = sum(t.get("move_pct", 0) for t in trades) / len(trades)
    avg_implied = sum(t.get("implied_move%", 0) for t in trades) / len(trades)
    return (f"  {tag:42}  n={s['trades']:>4}  Sharpe={s['sharpe']:>6.2f}"
            f"  P&L=${s['total_pnl']:>9,.0f}  win={s['win_rate']:>4.1f}%"
            f"  beat_implied={beat/max(len(trades),1)*100:.0f}%"
            f"  move={avg_move:.1f}% vs impl={avg_implied:.1f}%")


def main():
    print("EARNINGS STRADDLE BACKTEST")
    print(f"DTE={STRADDLE_DTE}, hold={HOLD_DAYS}d, ATM strike, cross spreads both legs\n")

    for label, end_dt, days, cache in WINDOWS:
        if not Path(cache).exists():
            print(f"  [{label}] cache not found — skipping"); continue
        print(f"═══ {label} ═══")
        scored = tune_v2.load_scored_from_dual_cache(Path(cache), "bull", end_dt, days)
        reg    = build_regime(end_dt, days, 200)

        # Earnings straddles only
        t_earn_ng = run(scored, regime=None, earnings_filter=True)
        t_earn_rg = run(scored, regime=reg,  earnings_filter=True)
        t_all_ng  = run(scored, regime=None, earnings_filter=False)

        print(line("Earnings straddle — NO regime gate", t_earn_ng))
        print(line("Earnings straddle — regime gated",   t_earn_rg))
        print(line("All-signal straddle — no gate",      t_all_ng))

        # Liquid only (tradeable)
        t_liq = [t for t in t_earn_ng if t["ticker"] in LIQUID_UNDERLYINGS]
        print(line("Earnings straddle — LIQUID only",    t_liq))
        print()

    print("═══ INTERPRETATION ═══")
    print("  beat_implied > 50% → straddle buyer wins on average")
    print("  Positive Sharpe → net-of-cost edge")
    print("  move% >> impl% → market consistently underprices earnings moves")


if __name__ == "__main__":
    main()
