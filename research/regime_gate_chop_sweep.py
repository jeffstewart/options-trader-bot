"""
regime_gate_chop_sweep.py — the Jun–Jul 2026 live window exposed the current regime gate's blind
spot: SPY finished +1.8% (never near its 200d) yet 14/26 sessions were down and news_call bled every
week — the 200d+3d-mom gate measures LEVEL and SIGN, but a 3-day call trade needs FOLLOW-THROUGH.
This sweep tests chop-aware gates on three tapes:

  bull-meltup  (dual_score_cache, 180d)   — a good gate must KEEP most of this P&L
  2022-bear    (bear_dual_cache, 90d)     — and not give back the bear protection
  live-chop    (regime_decisions.csv, 2026-06-24→) — and SKIP most of this tape

Candidates (all on top of the 200d primary):
  ER        Kaufman Efficiency Ratio on SPY 10d: |net move| / Σ|daily moves| — chop→0, trend→1.
            Gated as ER ≥ x AND net direction up.
  dd        down-day density: fraction of down closes in the last 10 sessions < x.
  heat      SELF-REFERENTIAL: trailing N of the strategy's own (ungated, simulated) trades must be
            net-positive once settled (entry+5cd). In production this needs NO shadow-trade infra —
            it can be computed from recent scored signals + the price data the bot already fetches.
            Fail-open during warmup (< N settled trades).

Same news_call sim as regime_gate_sweep.py (mag≥0.75, Δ0.40/DTE10, NET of costs, tiered trail).

Usage:  USE_YAHOO_BARS=1 .venv/bin/python -u ../research/regime_gate_chop_sweep.py   (from data/)
"""
import os, bisect, csv
os.environ.setdefault("USE_YAHOO_BARS", "1")
from datetime import datetime, timezone, timedelta

import news_call_sweep_unified as nc
from regime_filter import _spy_series

MAG, CONF, DELTA, DTE = 0.75, 0.70, 0.40, 10
LIVE_CSV = "regime_decisions.csv"
HEAT_SETTLE_CDAYS = 5      # entry + ~3 trading days ≈ 5 calendar days → trade "settled"

ORDER = ["none", "200d", "200d + 3d-mom (live)",
         "200d + ER10≥0.2↑", "200d + ER10≥0.3↑", "200d + ER10≥0.4↑",
         "200d + dd10<60%", "200d + dd10<50%",
         "200d + mom + ER0.3", "200d + heat20", "200d + heat10", "heat20 alone",
         # robustness checks around the dd winner: parameter sensitivity + momentum combo
         "200d + dd10<65%", "200d + dd15<60%", "200d + mom + dd10<60%"]


def load_live_rows():
    """regime_decisions.csv rows → the same shape news_call_sweep_unified.run_gate consumes.
    Every row here is a signal that already passed the bullish/tradeable bar live."""
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


def make_gates(end_dt, days, heat_trades):
    """heat_trades = [(entry_date, pnl_usd), …] from the UNGATED sim run, date-sorted."""
    lb = int((days + 200) * 1.7) + 60
    dates, v = _spy_series(end_dt, lb)

    def sma(i, ma):
        return sum(v[i - ma + 1:i + 1]) / ma if i >= ma - 1 else float("inf")

    def er(i, n=10):
        den = sum(abs(v[j] - v[j - 1]) for j in range(i - n + 1, i + 1))
        return abs(v[i] - v[i - n]) / den if den else 0.0

    def ddens(i, n=10):
        return sum(1 for j in range(i - n + 1, i + 1) if v[j] < v[j - 1]) / n

    def a200(i): return v[i] >= sma(i, 200)
    def mom(i):  return v[i] >= v[i - 3]
    def up(i, n=10): return v[i] >= v[i - n]

    def heat(day, n):
        settled = [p for d0, p in heat_trades if d0 + timedelta(days=HEAT_SETTLE_CDAYS) <= day]
        if len(settled) < n:
            return True                     # warmup: fail-open (mirrors live regime check)
        return sum(settled[-n:]) > 0

    def gate(check):                        # check(i, day) -> bool
        def f(day):
            i = bisect.bisect_right(dates, day) - 1
            if i < 200:
                return False
            return check(i, day)
        return f

    return {
        "none":                 (lambda day: True),
        "200d":                 gate(lambda i, d: a200(i)),
        "200d + 3d-mom (live)": gate(lambda i, d: a200(i) and mom(i)),
        "200d + ER10≥0.2↑":     gate(lambda i, d: a200(i) and up(i) and er(i) >= 0.2),
        "200d + ER10≥0.3↑":     gate(lambda i, d: a200(i) and up(i) and er(i) >= 0.3),
        "200d + ER10≥0.4↑":     gate(lambda i, d: a200(i) and up(i) and er(i) >= 0.4),
        "200d + dd10<60%":      gate(lambda i, d: a200(i) and ddens(i) < 0.6),
        "200d + dd10<50%":      gate(lambda i, d: a200(i) and ddens(i) < 0.5),
        "200d + mom + ER0.3":   gate(lambda i, d: a200(i) and mom(i) and up(i) and er(i) >= 0.3),
        "200d + heat20":        gate(lambda i, d: a200(i) and heat(d, 20)),
        "200d + heat10":        gate(lambda i, d: a200(i) and heat(d, 10)),
        "heat20 alone":         gate(lambda i, d: heat(d, 20)),
        "200d + dd10<65%":      gate(lambda i, d: a200(i) and ddens(i) < 0.65),
        "200d + dd15<60%":      gate(lambda i, d: a200(i) and ddens(i, 15) < 0.6),
        "200d + mom + dd10<60%": gate(lambda i, d: a200(i) and mom(i) and ddens(i) < 0.6),
    }


def max_dd(pnls):
    cum = peak = mdd = 0.0
    for p in pnls:
        cum += p; peak = max(peak, cum); mdd = min(mdd, cum - peak)
    return mdd


def run(label, end_dt, days, rows):
    # UNGATED sim first — it is both the baseline AND the heat gate's input
    ungated = nc.run_gate(rows, None, MAG, CONF, spread_mult=1.0, delta=DELTA, dte=DTE)
    heat_trades = sorted((datetime.fromisoformat(t["entry_dt"]).date(), t["pnl_usd"])
                         for t in ungated if t.get("entry_dt"))
    gates = make_gates(end_dt, days, heat_trades)

    res = {}
    for name, reg in gates.items():
        if name == "none":
            res[name] = nc.stats(ungated)
        else:
            res[name] = nc.stats(nc.run_gate(rows, reg, MAG, CONF, spread_mult=1.0,
                                             delta=DELTA, dte=DTE))
    base_n = max(res["none"]["n"], 1)
    print(f"\n████ {label} ████  ({len(rows)} signals · news_call Δ{DELTA}/DTE{DTE} · NET costs)")
    print(f"  {'gate':>22} {'trades':>7} {'P&L':>10} {'Sharpe':>7} {'win%':>5} {'maxDD':>9} {'skip%':>6}")
    for name in ORDER:
        s = res[name]
        mdd = max_dd(s["pnls"])
        skip = (1 - s["n"] / base_n) * 100
        print(f"  {name:>22} {s['n']:>7} {('${:+,.0f}'.format(s['total'])):>10} {s['sharpe']:>7.2f} "
              f"{s['win']:>5.0f} {('${:+,.0f}'.format(mdd)):>9} {skip:>5.0f}%")


def main():
    print("CHOP-AWARE REGIME GATE SWEEP — ER / down-day density / self-referential heat")

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

    print("\nPick: skips MOST of live-chop + keeps bear protection, at the least bull-P&L cost")
    print("vs '200d + 3d-mom (live)'. heat gates: fail-open warmup means early trades pass.")


if __name__ == "__main__":
    main()
