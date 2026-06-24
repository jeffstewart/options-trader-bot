"""
newsflow_gate_sweep.py — test a NEWS-FLOW regime gate: pause the long legs when the bot's OWN
aggregate news mood is DETERIORATING vs its recent baseline. The raw net-flow is ~always bullish
(only ~2% of days net-bearish), so an absolute ">0" gate is inert — only a DETRENDED/relative form
(today's breadth vs its trailing mean) can filter. Compared vs none / 200d / 200d+momentum on the
unified_v1 sweet-spot news_call sim, bull + 2022-bear, NET of costs.

Daily breadth = (net-bullish − net-bearish article count) / total, from the dual-score cache.

Usage:  USE_YAHOO_BARS=1 .venv/bin/python -u newsflow_gate_sweep.py
"""
import os, bisect, json
os.environ.setdefault("USE_YAHOO_BARS", "1")
from datetime import datetime, timezone, timedelta
from collections import defaultdict

import news_call_sweep_unified as nc
from regime_filter import _spy_series

MAG, CONF, DELTA, DTE = 0.75, 0.70, 0.40, 10
ORDER = ["none", "200d", "200d + 3d-mom (live)", "200d + nflow-abs(>0)",
         "200d + nflow≥trail10", "200d + nflow≥trail-0.1", "200d + mom + nflow", "nflow≥trail only"]


def breadth_series(cache, end_dt, days):
    raw = json.load(open(cache))
    start = end_dt - timedelta(days=days + 40)
    daily = defaultdict(lambda: [0, 0])     # date -> [n_articles, net_lean]
    for v in raw.values():
        if not isinstance(v, dict):
            continue
        ca = (v.get("_article") or {}).get("created_at")
        if not ca:
            continue
        try:
            dt = datetime.fromisoformat(str(ca).replace("Z", "+00:00")).astimezone(timezone.utc)
        except Exception:
            continue
        if not (start <= dt <= end_dt):
            continue
        b, br = v.get("bullish") or {}, v.get("bearish") or {}
        bs = float(b.get("magnitude", 0) or 0) * float(b.get("confidence", 0) or 0)
        rs = float(br.get("magnitude", 0) or 0) * float(br.get("confidence", 0) or 0)
        daily[dt.date()][0] += 1
        daily[dt.date()][1] += (1 if bs > rs else (-1 if rs > bs else 0))
    bdays = sorted(daily)
    breadth = {d: daily[d][1] / max(daily[d][0], 1) for d in bdays}
    return bdays, breadth


def make_gates(cache, end_dt, days):
    bdays, breadth = breadth_series(cache, end_dt, days)
    sdates, sv = _spy_series(end_dt, int((days + 200) * 1.7) + 60)

    def si(day): return bisect.bisect_right(sdates, day) - 1
    def sma(i, ma): return sum(sv[i - ma + 1:i + 1]) / ma if i >= ma - 1 else float("inf")
    def a200(day):
        i = si(day); return i >= 200 and sv[i] >= sma(i, 200)
    def mom(day):
        i = si(day); return i >= 200 and sv[i] >= sv[i - 3]

    def bi(day): return bisect.bisect_right(bdays, day) - 1
    def bval(day):
        i = bi(day); return breadth[bdays[i]] if i >= 0 else None
    def trailmean(day, n=10):
        i = bi(day)
        return sum(breadth[bdays[j]] for j in range(i - n, i)) / n if i >= n else None
    def nflow(day, margin=0.0):              # mood not deteriorating vs trailing baseline
        b, m = bval(day), trailmean(day)
        return True if (b is None or m is None) else b >= m - margin
    def nflow_abs(day):                      # naive absolute (≈ always on — for reference)
        b = bval(day); return b is None or b >= 0

    return {
        "none":                   (lambda day: True),
        "200d":                   a200,
        "200d + 3d-mom (live)":   (lambda day: a200(day) and mom(day)),
        "200d + nflow-abs(>0)":   (lambda day: a200(day) and nflow_abs(day)),
        "200d + nflow≥trail10":   (lambda day: a200(day) and nflow(day, 0.0)),
        "200d + nflow≥trail-0.1": (lambda day: a200(day) and nflow(day, 0.1)),
        "200d + mom + nflow":     (lambda day: a200(day) and mom(day) and nflow(day, 0.0)),
        "nflow≥trail only":       (lambda day: nflow(day, 0.0)),
    }


def max_dd(pnls):
    cum = peak = mdd = 0.0
    for p in pnls:
        cum += p; peak = max(peak, cum); mdd = min(mdd, cum - peak)
    return mdd


def run(window):
    label, end_dt, days, cache = window
    rows = nc.load_scored_from_unified(cache, end_dt, days, "unified_v1")
    gates = make_gates(cache, end_dt, days)
    res = {n: nc.stats(nc.run_gate(rows, g, MAG, CONF, spread_mult=1.0, delta=DELTA, dte=DTE))
           for n, g in gates.items()}
    base = max(res["none"]["n"], 1)
    print(f"\n████ {label} ████  ({len(rows)} signals · news_call Δ{DELTA}/DTE{DTE} · NET costs)")
    print(f"  {'gate':>22} {'trades':>7} {'P&L':>10} {'Sharpe':>7} {'win%':>5} {'maxDD':>9} {'skip%':>6}")
    for n in ORDER:
        s = res[n]
        print(f"  {n:>22} {s['n']:>7} {('${:+,.0f}'.format(s['total'])):>10} {s['sharpe']:>7.2f} "
              f"{s['win']:>5.0f} {('${:+,.0f}'.format(max_dd(s['pnls']))):>9} {(1 - s['n']/base)*100:>5.0f}%")


def main():
    print("NEWS-FLOW GATE SWEEP — news_call (unified_v1 sweet-spot) · bull + 2022-bear")
    run(nc.BULL)
    run(nc.BEAR)
    print("\nDetrended news-flow worth it only if it beats 200d+momentum on Sharpe/maxDD with comparable P&L.")


if __name__ == "__main__":
    main()
