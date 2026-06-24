"""
regime_gate_sweep.py — backtest ENHANCED regime gates for the long legs (news_call), to stop
buying calls into short pullbacks. Current live gate = SPY >= 200d SMA (slow: SPY can be well
above its 200d while in a multi-day dip). Candidates add a faster short-term condition ON TOP of
the 200d (which stays the primary bull/bear gate).

Runs the unified_v1 sweet-spot news_call sim (mag>=0.75, Δ0.40/DTE10, NET of costs) on the
bull-meltup AND 2022-bear windows, swapping only the regime gate. Reports P&L / Sharpe / win /
maxDD / trades + how many signals each gate skips vs no filter.

Goal: cut pullback loss + maxDD while keeping MOST of the bull P&L vs the current 200d gate.

Usage:  USE_YAHOO_BARS=1 .venv/bin/python -u regime_gate_sweep.py
"""
import os, bisect
os.environ.setdefault("USE_YAHOO_BARS", "1")
from datetime import timedelta

import news_call_sweep_unified as nc
import yahoo_data
from regime_filter import _spy_series

MAG, CONF, DELTA, DTE = 0.75, 0.70, 0.40, 10
ORDER = ["none", "200d", "200d + 3d-mom (live)", "200d + VIX<20", "200d + VIX<25",
         "200d + VIX<VIXma20", "200d + VIX<VIX3M (term)", "200d + mom + VIX<20"]


def make_gates(end_dt, days):
    lb = int((days + 200) * 1.7) + 60
    dates, v = _spy_series(end_dt, lb)

    def yseries(sym):
        bars = yahoo_data.get_yahoo_bars(sym, end_dt - timedelta(days=lb), end_dt + timedelta(days=1))
        return [b["t"].date() for b in bars], [b["c"] for b in bars]
    vd, vv = yseries("^VIX")
    v3d, v3v = yseries("^VIX3M")

    def sma(i, ma):
        return sum(v[i - ma + 1:i + 1]) / ma if i >= ma - 1 else float("inf")

    def vix(day):
        i = bisect.bisect_right(vd, day) - 1
        return vv[i] if i >= 0 else None

    def vix_ma(day, n=20):
        i = bisect.bisect_right(vd, day) - 1
        return sum(vv[i - n + 1:i + 1]) / n if i >= n - 1 else None

    def vix3m(day):
        i = bisect.bisect_right(v3d, day) - 1
        return v3v[i] if i >= 0 else None

    def gate(check):                      # check(i, day) -> bool
        def f(day):
            i = bisect.bisect_right(dates, day) - 1
            if i < 200:
                return False
            return check(i, day)
        return f

    def a200(i):  return v[i] >= sma(i, 200)
    def mom(i):   return v[i] >= v[i - 3]
    def vlt(d, t): x = vix(d); return x is not None and x < t
    def vltma(d):  x, m = vix(d), vix_ma(d); return x is not None and m is not None and x < m
    def vlt3m(d):  x, y = vix(d), vix3m(d); return x is not None and y is not None and x < y

    return {
        "none":                    (lambda day: True),
        "200d":                    gate(lambda i, d: a200(i)),
        "200d + 3d-mom (live)":    gate(lambda i, d: a200(i) and mom(i)),
        "200d + VIX<20":           gate(lambda i, d: a200(i) and vlt(d, 20)),
        "200d + VIX<25":           gate(lambda i, d: a200(i) and vlt(d, 25)),
        "200d + VIX<VIXma20":      gate(lambda i, d: a200(i) and vltma(d)),
        "200d + VIX<VIX3M (term)": gate(lambda i, d: a200(i) and vlt3m(d)),
        "200d + mom + VIX<20":     gate(lambda i, d: a200(i) and mom(i) and vlt(d, 20)),
    }


def max_dd(pnls):
    cum = peak = mdd = 0.0
    for p in pnls:
        cum += p; peak = max(peak, cum); mdd = min(mdd, cum - peak)
    return mdd


def run(window):
    label, end_dt, days, cache = window
    rows = nc.load_scored_from_unified(cache, end_dt, days, "unified_v1")
    gates = make_gates(end_dt, days)
    res = {name: nc.stats(nc.run_gate(rows, reg, MAG, CONF, spread_mult=1.0, delta=DELTA, dte=DTE))
           for name, reg in gates.items()}
    base_n = max(res["none"]["n"], 1)
    print(f"\n████ {label} ████  ({len(rows)} bullish unified_v1 signals · news_call Δ{DELTA}/DTE{DTE} · NET costs)")
    print(f"  {'gate':>16} {'trades':>7} {'P&L':>10} {'Sharpe':>7} {'win%':>5} {'maxDD':>9} {'skip%':>6}")
    for name in ORDER:
        s = res[name]
        mdd = max_dd(s["pnls"])
        skip = (1 - s["n"] / base_n) * 100
        print(f"  {name:>16} {s['n']:>7} {('${:+,.0f}'.format(s['total'])):>10} {s['sharpe']:>7.2f} "
              f"{s['win']:>5.0f} {('${:+,.0f}'.format(mdd)):>9} {skip:>5.0f}%")


def main():
    print("REGIME GATE SWEEP — news_call (unified_v1 sweet-spot) · bull-meltup + 2022-bear")
    run(nc.BULL)
    run(nc.BEAR)
    print("\nPick: cuts maxDD / bear loss MOST while keeping bull P&L close to '200d (current)'.")


if __name__ == "__main__":
    main()
