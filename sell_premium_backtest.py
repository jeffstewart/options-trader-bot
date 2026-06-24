"""
sell_premium_backtest.py — harvest the IV crush by SELLING defined-risk credit
spreads after news, instead of buying calls.

We proved IV spikes on news then crushes. Buying calls fights that; selling a
BULL PUT credit spread (short a ~0.30-delta put, long a lower put for defined
risk) profits from: IV crush + theta + the stock staying up. This flips the
options weaknesses — use LIQUID names (tight spreads), high win rate, capped risk.

Per trade: sell the spread into the elevated entry IV; mark it daily with BS as
IV decays; take profit at 50% of credit, stop if it doubles, else hold to ~expiry.
Crosses the bid/ask on BOTH legs at entry AND exit (realistic). Risk-based sizing
(max loss ≈ MAX_POSITION_USD). Cross-regime; reports the LIQUID subset (the
tradeable version) and all-names for reference.

Usage:  USE_YAHOO_BARS=1 .venv/bin/python sell_premium_backtest.py
"""
from datetime import datetime, timezone, timedelta
from pathlib import Path

import backtest as _bt
import tune_v2
from backtest import (get_price_at, get_stock_bars, is_valid_stock_ticker, base_iv_for,
                      _spread_pct, LIQUID_UNDERLYINGS,
                      NEWS_IV_MULTIPLIER, IV_CRUSH_HALFLIFE_DAYS, RISK_FREE_RATE)
from pricing import bs_put_price, put_strike_for_delta, iv_crush_path
from benchmark import compute_stats
import config as _cfg

DTE          = 21       # credit-spread tenor (theta-rich, manageable gamma)
SHORT_DELTA  = 0.30     # short put delta (OTM below spot)
WIDTH_PCT    = 0.05     # long put strike = short strike − 5% of spot (defines risk)
PROFIT_TAKE  = 0.50     # close at 50% of credit captured
STOP_MULT    = 2.0      # close if spread value ≥ 2× the credit (≈ −1× credit loss)
MAX_HOLD     = 18       # trading days (< DTE)
WINDOWS = [
    ("bull-meltup", datetime.now(timezone.utc) - timedelta(days=5), 180, "dual_score_cache.json"),
    ("2022-bear",   datetime(2022, 6, 30, tzinfo=timezone.utc),     90,  "bear_dual_cache.json"),
]


def simulate_credit_spread(ticker, entry_dt, position_usd, spread_mult=1.0):
    r = RISK_FREE_RATE
    s0 = get_price_at(ticker, entry_dt)
    if not s0 or s0 < _cfg.MIN_STOCK_PRICE or s0 > 10000:
        return None
    base_iv = base_iv_for(ticker, entry_dt)
    entry_iv = base_iv * NEWS_IV_MULTIPLIER
    T0 = DTE / 365.0
    Ks = round(put_strike_for_delta(s0, T0, entry_iv, SHORT_DELTA, r), 2)   # short put (OTM)
    Kl = round(Ks * (1 - WIDTH_PCT), 2)                                      # long put (lower)
    W = Ks - Kl
    if W <= 0:
        return None
    short_mid = bs_put_price(s0, Ks, T0, entry_iv, r)
    long_mid  = bs_put_price(s0, Kl, T0, entry_iv, r)
    credit_mid = short_mid - long_mid
    if credit_mid <= 0.02:
        return None
    # Entry: SELL short put at bid, BUY long put at ask (cross the spread both legs)
    hs_s = _spread_pct(short_mid, ticker, spread_mult) / 2
    hs_l = _spread_pct(long_mid,  ticker, spread_mult) / 2
    credit_in = short_mid * (1 - hs_s) - long_mid * (1 + hs_l)
    if credit_in <= 0:
        return None
    max_loss_contract = (W - credit_mid) * 100
    if max_loss_contract <= 0:
        return None
    qty = max(1, int(position_usd / max_loss_contract))

    bars = get_stock_bars(ticker, entry_dt, entry_dt + timedelta(days=int(MAX_HOLD * 1.6) + 7))
    tb = [b for b in bars if entry_dt < b["t"]]
    if not tb or (tb[0]["t"] - entry_dt).days > 7:
        return None
    prev = s0
    for b in tb:
        if prev > 0 and abs(b["c"] / prev - 1) > 0.5:
            return None
        prev = b["c"]

    def spread_value(S, days_held, i):
        T = max((DTE - days_held) / 365.0, 1.0 / 365.0)
        iv = iv_crush_path(base_iv, i, NEWS_IV_MULTIPLIER, IV_CRUSH_HALFLIFE_DAYS)
        return (bs_put_price(S, Ks, T, iv, r) - bs_put_price(S, Kl, T, iv, r), iv, T)

    exit_val, reason = None, "max_hold"
    for i, b in enumerate(tb[:MAX_HOLD], start=1):
        dh = max((b["t"] - entry_dt).days, 0)
        sv, iv, T = spread_value(b["c"], dh, i)
        sv = max(0.0, min(sv, W))
        if credit_mid - sv >= credit_mid * PROFIT_TAKE:      # captured enough
            exit_val, reason = sv, "profit_take"; break
        if sv >= credit_mid * STOP_MULT or sv >= W * 0.95:   # stop / near max loss
            exit_val, reason = sv, "stop"; break
    if exit_val is None:
        last = tb[min(MAX_HOLD, len(tb)) - 1]
        dh = max((last["t"] - entry_dt).days, 0)
        exit_val = max(0.0, min(spread_value(last["c"], dh, min(MAX_HOLD, len(tb)))[0], W))

    # Exit: BUY back short at ask, SELL long at bid
    # (approx legs from the spread value via the same proportional friction)
    cost_out = exit_val * (1 + (hs_s + hs_l) / 2)            # pay up to close
    pnl = (credit_in - cost_out) * 100 * qty
    return {"ticker": ticker, "entry_dt": entry_dt.isoformat(),
            "pnl_usd": pnl, "pnl_pct": (credit_in - cost_out) / (W) * 100, "reason": reason}


def run(end_dt, days, cache, regime=None, spread_mult=1.0):
    scored = tune_v2.load_scored_from_dual_cache(Path(cache), "bull", end_dt, days)

    def scale(mag, conf):
        return max(_cfg.MAX_POSITION_USD * 0.10, _cfg.MAX_POSITION_USD * mag * conf)

    trades, seen = [], {}
    for row in scored:
        mag, conf = row["magnitude"], row["confidence"]
        if mag < _cfg.MIN_MAGNITUDE or conf < _cfg.BASE_CONFIDENCE + (1 - mag) * _cfg.CONFIDENCE_SLOPE:
            continue
        d = row["created_at"].date()
        if regime is not None and not regime(d):
            continue
        ds = seen.setdefault(d.isoformat(), set())
        for tk in row["tickers"][:2]:
            if tk in ds or tk in ("BTC", "ETH") or not is_valid_stock_ticker(tk):
                continue
            ds.add(tk)
            t = simulate_credit_spread(tk, row["created_at"], scale(mag, conf), spread_mult)
            if t:
                trades.append(t)
    return trades


def line(tag, s):
    return (f"  {tag:26} trades={s['trades']:>4}  P&L=${s['total_pnl']:>9,.0f}  "
            f"Sharpe={s['sharpe']:>5.2f}  win={s['win_rate']:>4.1f}%  maxDD=${s['max_dd']:>8,.0f}")


def main():
    print(f"SELL-PREMIUM (bull put credit spread) — DTE{DTE}, short Δ{SHORT_DELTA}, "
          f"{WIDTH_PCT*100:.0f}% width, take {PROFIT_TAKE*100:.0f}%, net of spreads\n")
    from regime_filter import build_regime
    for label, end_dt, days, cache in WINDOWS:
        print(f"═══ {label} ═══")
        reg = build_regime(end_dt, days, 200)
        allt = run(end_dt, days, cache, regime=reg, spread_mult=1.0)
        liq = [t for t in allt if t["ticker"] in LIQUID_UNDERLYINGS]
        print(line("all names (gated)", compute_stats(allt)))
        print(line("LIQUID names (gated) ★", compute_stats(liq)))
        print()


if __name__ == "__main__":
    main()
