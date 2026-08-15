"""
regime_gate_ma_days_sweep.py — now that the live gate is 200d SMA + 3d-mom + dd10<60% chop
(regime_gate_chop_sweep.py, deployed), is the 200d SMA leg still pulling its weight, or do the
faster mom/chop conditions already do the job? Sweeps the SMA length (including "none": drop the
SMA entirely and keep only mom+chop) on top of / instead of the current live combo.

The 200d leg is a genuine SIMPLE MOVING AVERAGE (mean of the last N daily closes), NOT "SPY
higher than N days ago" -- that "vs N days ago" comparison is what the SEPARATE 3d-mom leg does,
just with a much shorter window. This sweep varies the SMA's N; mom stays fixed at 3d (the live
value) throughout, since that parameter isn't in question here.

Same three tapes as regime_gate_chop_sweep.py, same news_call sim (mag>=0.75, Δ0.40/DTE10, NET of
costs, tiered trail), for direct comparability with the gate currently live:
  bull-meltup  (dual_score_cache, 180d)   — a good gate must KEEP most of this P&L
  2022-bear    (bear_dual_cache, 90d)     — and not give back the bear protection
  live-chop    (regime_decisions.csv, 2026-06-24→) — and SKIP most of this tape

Usage:  USE_YAHOO_BARS=1 .venv/bin/python -u ../research/regime_gate_ma_days_sweep.py   (from data/)
"""
import os, bisect, csv
os.environ.setdefault("USE_YAHOO_BARS", "1")
from datetime import datetime, timezone, timedelta

import news_call_sweep_unified as nc
from regime_filter import _spy_series

MAG, CONF, DELTA, DTE = 0.75, 0.70, 0.40, 10
LIVE_CSV = "regime_decisions.csv"
MOM_DAYS = 3            # live value -- fixed here, not swept
DD_WIN, DD_MAX = 10, 0.60  # live values -- fixed here, not swept
MA_CANDIDATES = [50, 100, 150, 200, 250]

ORDER = ["none", "mom+chop only (no SMA)",
         "50d + mom+chop", "100d + mom+chop", "150d + mom+chop",
         "200d + mom+chop (current live)", "250d + mom+chop",
         "200d alone (no mom/chop)", "mom alone (no SMA/chop)", "chop alone (no SMA/mom)"]


def load_live_rows():
    rows = []
    with open(LIVE_CSV) as f:
        for r in csv.DictReader(f):
            try:
                rows.append({"created_at": datetime.fromisoformat(r["ts"]),
                             "tickers": [r["tickers"]] if r["tickers"] else [],
                             "magnitude": float(r["magnitude"]),
                             "confidence": float(r["confidence"])})
            except (ValueError, KeyError):
                continue
    rows.sort(key=lambda r: r["created_at"])
    return rows


def make_gates(end_dt, days):
    lb = int((days + max(MA_CANDIDATES)) * 1.7) + 60
    dates, v = _spy_series(end_dt, lb)

    def sma(i, ma):
        return sum(v[i - ma + 1:i + 1]) / ma if i >= ma - 1 else float("inf")

    def a_ma(i, ma): return v[i] >= sma(i, ma)
    def mom(i):       return i >= MOM_DAYS and v[i] >= v[i - MOM_DAYS]
    def ddens(i, n=DD_WIN):
        return sum(1 for j in range(i - n + 1, i + 1) if v[j] < v[j - 1]) / n
    def chop_ok(i):   return i >= DD_WIN and ddens(i) < DD_MAX

    def gate(check):                       # check(i) -> bool
        def f(day):
            i = bisect.bisect_right(dates, day) - 1
            if i < max(MA_CANDIDATES):      # enough history for the LONGEST candidate, for a fair comparison
                return False
            return check(i)
        return f

    gates = {
        "none":                    (lambda day: True),
        "mom+chop only (no SMA)":  gate(lambda i: mom(i) and chop_ok(i)),
        "200d alone (no mom/chop)": gate(lambda i: a_ma(i, 200)),
        "mom alone (no SMA/chop)":  gate(lambda i: mom(i)),
        "chop alone (no SMA/mom)":  gate(lambda i: chop_ok(i)),
    }
    for ma in MA_CANDIDATES:
        suffix = " (current live)" if ma == 200 else ""
        gates[f"{ma}d + mom+chop{suffix}"] = \
            gate(lambda i, ma=ma: a_ma(i, ma) and mom(i) and chop_ok(i))
    return gates


def max_dd(pnls):
    cum = peak = mdd = 0.0
    for p in pnls:
        cum += p; peak = max(peak, cum); mdd = min(mdd, cum - peak)
    return mdd


def run(label, end_dt, days, rows):
    gates = make_gates(end_dt, days)
    res = {name: nc.stats(nc.run_gate(rows, reg, MAG, CONF, spread_mult=1.0, delta=DELTA, dte=DTE))
           for name, reg in gates.items()}
    base_n = max(res["none"]["n"], 1)
    print(f"\n████ {label} ████  ({len(rows)} signals · news_call Δ{DELTA}/DTE{DTE} · NET costs)")
    print(f"  {'gate':>32} {'trades':>7} {'P&L':>10} {'Sharpe':>7} {'win%':>5} {'maxDD':>9} {'skip%':>6}")
    for name in ORDER:
        s = res[name]
        mdd = max_dd(s["pnls"])
        skip = (1 - s["n"] / base_n) * 100
        print(f"  {name:>32} {s['n']:>7} {('${:+,.0f}'.format(s['total'])):>10} {s['sharpe']:>7.2f} "
              f"{s['win']:>5.0f} {('${:+,.0f}'.format(mdd)):>9} {skip:>5.0f}%")


def main():
    print("REGIME GATE SMA-LENGTH SWEEP — is 200d still pulling weight given live mom(3d)+chop(dd10<60%)?")
    print(f"(mom={MOM_DAYS}d, dd_window={DD_WIN}d, dd_max={DD_MAX:.0%} held fixed at their live values)")

    label, end_dt, days, cache = nc.BULL
    rows = nc.load_scored_from_unified(cache, end_dt, days, "unified_v1")
    run(label, end_dt, days, rows)

    blabel, bend, bdays, bcache = nc.BEAR
    brows = nc.load_scored_from_unified(bcache, bend, bdays, "unified_v1")
    run(blabel, bend, bdays, brows)

    lrows = load_live_rows()
    if lrows:
        lend = max(r["created_at"] for r in lrows) + timedelta(days=1)
        ldays = (lend - min(r["created_at"] for r in lrows)).days + 1
        run("live-chop 2026-06/07", lend, ldays, lrows)

    print("\nPick: keeps bull P&L + bear protection close to '200d + mom+chop (current live)' while")
    print("simplifying (shorter/no SMA) if a shorter window or dropping it costs little to nothing.")


if __name__ == "__main__":
    main()
