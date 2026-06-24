"""
news_call_spread_backtest.py — does a BULL CALL SPREAD beat the current naked Δ0.40/DTE10 call
for news_call? The naked call is cost-fragile + IV-crush-bleeding (lost money live). A debit spread
(long Δ0.40, short a higher strike) is cheaper, vega-offset (less crush bleed) and defined-risk —
at the cost of a capped upside. Tests it on the SAME unified_v1 sweet-spot signals (mag≥0.75),
NET of realistic spreads, bull + 2022-bear, with the live exit (40% flat trail, 3d hold).

simulate_spread_pnl mirrors backtest.simulate_option_pnl exactly (BS + IV-crush + tiered bid/ask
spread + data guards) but prices the NET of two calls.

Usage:  USE_YAHOO_BARS=1 .venv/bin/python -u news_call_spread_backtest.py
"""
import os
os.environ.setdefault("USE_YAHOO_BARS", "1")
from datetime import timedelta

import backtest as _bt
import news_call_sweep_unified as nc
from regime_filter import build_regime
from benchmark import compute_stats

MAG, CONF = 0.75, 0.70
LONG_DELTA, DTE, HOLD, TRAIL = 0.40, 10, 3, 0.40
SHORT_DELTAS = [0.25, 0.20, 0.15]      # spread widths to test (lower delta = wider, cheaper, more cap)
r = _bt.RISK_FREE_RATE


def simulate_spread_pnl(ticker, entry_dt, spot, position_usd, signal, short_delta,
                        dte=DTE, max_hold=HOLD, trail=TRAIL, spread_mult=1.0):
    if spot > _bt.MAX_SANE_STOCK_PRICE:
        return None
    base_iv = _bt.base_iv_for(ticker, entry_dt)
    entry_iv = base_iv * _bt.NEWS_IV_MULTIPLIER
    T0 = dte / 365.0
    k_long = round(_bt.strike_for_delta(spot, T0, entry_iv, LONG_DELTA, r), 2)
    k_short = round(_bt.strike_for_delta(spot, T0, entry_iv, short_delta, r), 2)
    if k_short <= k_long:
        return None

    def legs(s, T, iv):
        return _bt.bs_call_price(s, k_long, T, iv, r), _bt.bs_call_price(s, k_short, T, iv, r)

    ll, ss = legs(spot, T0, entry_iv)
    entry_net = ll - ss
    if entry_net < 0.05:
        return None
    if entry_net * 100 > position_usd * _bt.MAX_CONTRACT_BUDGET_MULT:
        return None
    # entry fill: pay ask on long, receive bid on short (both costs work against you)
    if spread_mult > 0:
        entry_fill = ll * (1 + _bt._spread_pct(ll, ticker, spread_mult) / 2) \
            - ss * (1 - _bt._spread_pct(ss, ticker, spread_mult) / 2)
    else:
        entry_fill = entry_net
    if entry_fill < 0.02:
        return None
    qty = max(1, int(position_usd / (entry_fill * 100)))
    cost = entry_fill * 100 * qty

    bars = _bt.get_stock_bars(ticker, entry_dt, entry_dt + timedelta(days=int(max_hold * 1.5) + 7))
    tb = [b for b in bars if entry_dt < b["t"]]
    if not tb or (tb[0]["t"] - entry_dt).days > 7:
        return None
    prev = spot
    for b in tb:
        if prev > 0 and abs(b["c"] / prev - 1) > 0.50:
            return None
        prev = b["c"]

    peak, exit_net, exit_spot, exit_T, exit_iv, reason = entry_net, entry_net, spot, T0, entry_iv, "max_hold"
    for i, b in enumerate(tb[:max_hold], start=1):
        days = max((b["t"] - entry_dt).days, 0)
        T = max((dte - days) / 365.0, 1.0 / 365.0)
        iv = _bt.iv_crush_path(base_iv, i, _bt.NEWS_IV_MULTIPLIER, _bt.IV_CRUSH_HALFLIFE_DAYS)
        cl, cs = legs(b["c"], T, iv)
        cur = max(0.0, cl - cs)
        if cur > peak:
            peak = cur
        if cur <= peak * (1 - trail):
            exit_net, exit_spot, exit_T, exit_iv, reason = cur, b["c"], T, iv, "trail"
            break
    else:
        lb = tb[min(max_hold, len(tb)) - 1]; n = min(max_hold, len(tb))
        days = max((lb["t"] - entry_dt).days, 0)
        exit_T = max((dte - days) / 365.0, 1.0 / 365.0)
        exit_iv = _bt.iv_crush_path(base_iv, n, _bt.NEWS_IV_MULTIPLIER, _bt.IV_CRUSH_HALFLIFE_DAYS)
        exit_spot = lb["c"]
        cl, cs = legs(exit_spot, exit_T, exit_iv)
        exit_net = max(0.0, cl - cs)

    cl, cs = legs(exit_spot, exit_T, exit_iv)
    if spread_mult > 0:                       # sell long at bid, buy back short at ask
        exit_fill = cl * (1 - _bt._spread_pct(cl, ticker, spread_mult) / 2) \
            - cs * (1 + _bt._spread_pct(cs, ticker, spread_mult) / 2)
    else:
        exit_fill = exit_net
    exit_value = max(0.0, exit_fill) * 100 * qty
    pnl = exit_value - cost
    return {"pnl_usd": pnl, "pnl_pct": (pnl / cost * 100) if cost else 0, "cost_basis": cost,
            "entry_dt": entry_dt.isoformat()}


def run_naked(rows, reg, size_fn):
    """Naked Δ0.40/DTE10 call with the live exit (flat 40% trail, 3d hold), via simulate_option_pnl."""
    save = {k: getattr(_bt, k) for k in ("TARGET_DELTA", "DTE_TARGET", "MAX_HOLD_DAYS", "TRAILING_STOP_PCT")}
    _bt.TARGET_DELTA, _bt.DTE_TARGET, _bt.MAX_HOLD_DAYS, _bt.TRAILING_STOP_PCT = LONG_DELTA, DTE, HOLD, TRAIL
    trades, seen = [], set()
    try:
        for tk, dt, mag, conf in rows:
            key = f"{dt.date()}_{tk}"
            if key in seen:
                continue
            seen.add(key)
            sp = _bt.get_price_at(tk, dt)
            if not sp:
                continue
            t = _bt.simulate_option_pnl(tk, dt, sp, size_fn(mag, conf), {"magnitude": mag, "confidence": conf},
                                        option_type="call", exit_rule="trail_premium", spread_mult=1.0)
            if t:
                trades.append(t)
    finally:
        for k, v in save.items():
            setattr(_bt, k, v)
    return trades


def run_spread(rows, reg, size_fn, short_delta):
    trades, seen = [], set()
    for tk, dt, mag, conf in rows:
        key = f"{dt.date()}_{tk}"
        if key in seen:
            continue
        seen.add(key)
        sp = _bt.get_price_at(tk, dt)
        if not sp:
            continue
        t = simulate_spread_pnl(tk, dt, sp, size_fn(mag, conf), {"magnitude": mag, "confidence": conf}, short_delta)
        if t:
            trades.append(t)
    return trades


def max_dd(pnls):
    cum = peak = mdd = 0.0
    for p in pnls:
        cum += p; peak = max(peak, cum); mdd = min(mdd, cum - peak)
    return mdd


def line(tag, trades):
    if not trades:
        return f"  {tag:26} (no trades)"
    s = compute_stats(trades)
    pnls = [t["pnl_usd"] for t in trades]
    avgcost = sum(t["cost_basis"] for t in trades) / len(trades)
    return (f"  {tag:26} n={s['trades']:>3}  P&L {('${:+,.0f}'.format(s['total_pnl'])):>9}  "
            f"Sh{s['sharpe']:>5.2f}  win{s['win_rate']:>3.0f}%  maxDD {('${:+,.0f}'.format(max_dd(pnls))):>9}  "
            f"avg-cost ${avgcost:>5,.0f}")


def gate_rows(window):
    label, end_dt, days, cache = window
    rows = nc.load_scored_from_unified(cache, end_dt, days, "unified_v1")
    reg = build_regime(end_dt, days, 200)
    out = []
    for r_ in rows:
        if r_["magnitude"] < MAG or r_["confidence"] < CONF or not reg(r_["created_at"].date()):
            continue
        for tk in r_["tickers"][:2]:
            if tk not in ("BTC", "ETH") and _bt.is_valid_stock_ticker(tk):
                out.append((tk, r_["created_at"], r_["magnitude"], r_["confidence"]))
    return label, out, reg


def main():
    sizefn = lambda m, c: nc.scale(m, c)
    print("NEWS_CALL: bull-call-spread vs naked Δ0.40/DTE10 call — unified_v1 sweet-spot, NET costs, 40%trail/3d")
    for window in (nc.BULL, nc.BEAR):
        label, rows, reg = gate_rows(window)
        print(f"\n████ {label} ████  ({len(rows)} gated signals)")
        print(line("NAKED call Δ0.40", run_naked(rows, reg, sizefn)))
        for sd in SHORT_DELTAS:
            print(line(f"SPREAD 0.40/{sd:.2f}", run_spread(rows, reg, sizefn, sd)))
    print("\nLook for: the spread CUTS maxDD / lifts Sharpe (cheaper, less crush) without giving up too much P&L.")


if __name__ == "__main__":
    main()
