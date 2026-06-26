"""
exit_intraday_sweep.py — re-runs the news_call exit-policy comparison on INTRADAY (hourly IEX) bars
instead of daily, with realistic INTRA-BAR fills: a take-profit limit fills when the bar HIGH crosses
the target; a trail/breakeven stop fills when the bar LOW breaches the stop. The daily-bar sweep
(trail_winrate_sweep.py) washed out the breakeven-lock because premiums gap across a whole day — hourly
bars test whether that, and the take-profit edge, hold at finer resolution. Premiums are still BS +
IV-crush synthesized off the underlying (the forward position_paths.csv log will later give REAL quotes).
Entry matches the daily sim (signal-day close, 10 DTE) so ONLY the exit resolution differs.

Usage:  USE_YAHOO_BARS=1 .venv/bin/python -u research/exit_intraday_sweep.py
"""
import os, json, statistics
os.environ.setdefault("USE_YAHOO_BARS", "1")
from datetime import datetime, timezone, timedelta, time as dtime
import exit_sweep_unified as es
import backtest as bt
import pricing as pr
import config as cfg
from alpaca.data.requests import StockBarsRequest
from alpaca.data.timeframe import TimeFrame
from alpaca.data.enums import Adjustment

DTE, DELTA, HOLD, USD = 10, cfg.NEWS_CALL_TARGET_DELTA, 3, cfg.MAX_POSITION_USD
R, ARMED, BARS_PER_DAY = cfg.RISK_FREE_RATE, 0.15, 6.5
TP_FIRST = os.environ.get("TP_FIRST", "") == "1"   # intra-bar: TP limit fills before stop (optimistic)
GATE = lambda m, c: m >= cfg.NEWS_CALL_MIN_MAGNITUDE and es.dyn_gate(m, c)
CACHE = "intraday_bars_cache.json"
_cache = json.load(open(CACHE)) if os.path.exists(CACHE) else {}

INF = float("inf")
# ratchet tiers = [(peak-gain threshold, trail% while below it)], applied off the PEAK gain so the trail
# only ever TIGHTENS (true ratchet). Live EXIT_TIERS thresholds (+100/+300%) almost never engage on a
# 3-day option, so also test ratchets tuned to where news_call lives (+15–50%).
POLICIES = [
    {"name": "flat 40% (LIVE)",       "rule": "trail_premium", "trail": 0.40, "target": None},
    {"name": "flat 30%",              "rule": "trail_premium", "trail": 0.30, "target": None},
    {"name": "flat 25%",              "rule": "trail_premium", "trail": 0.25, "target": None},
    {"name": "flat 20%",              "rule": "trail_premium", "trail": 0.20, "target": None},
    {"name": "flat 15%",              "rule": "trail_premium", "trail": 0.15, "target": None},
    {"name": "flat 12%",              "rule": "trail_premium", "trail": 0.12, "target": None},
    {"name": "flat 10%",              "rule": "trail_premium", "trail": 0.10, "target": None},
    {"name": "ratchet 30/20/13 @+100/300% (live tiers)", "rule": "ratchet", "tiers": [(1.00, 0.30), (3.00, 0.20), (INF, 0.13)], "target": None},
    {"name": "ratchet 40/25/15 @+15/35%", "rule": "ratchet", "tiers": [(0.15, 0.40), (0.35, 0.25), (INF, 0.15)], "target": None},
    {"name": "ratchet 40/20/12 @+20/50%", "rule": "ratchet", "tiers": [(0.20, 0.40), (0.50, 0.20), (INF, 0.12)], "target": None},
    {"name": "ratchet 30/15/10 @+15/40%", "rule": "ratchet", "tiers": [(0.15, 0.30), (0.40, 0.15), (INF, 0.10)], "target": None},
]


def fetch_hourly(tk, start, end):
    key = f"{tk}_{start.date()}"
    if key in _cache:
        return [(datetime.fromisoformat(t), h, l, c) for t, h, l, c in _cache[key]]
    try:
        resp = bt.stock_client.get_stock_bars(StockBarsRequest(symbol_or_symbols=tk,
            timeframe=TimeFrame.Hour, start=start, end=end, feed="iex", adjustment=Adjustment.ALL))
        raw = (resp.data or {}).get(tk, [])
    except Exception:
        raw = []
    out = [(b.timestamp, float(b.high), float(b.low), float(b.close))
           for b in raw if 14 <= b.timestamp.hour < 21]          # RTH only (≈9:30–16:00 ET)
    _cache[key] = [(t.isoformat(), h, l, c) for t, h, l, c in out]
    return out


def sim_intraday(tk, entry_S, entry_dt, bars, base_iv, pol):
    entry_iv = base_iv * cfg.NEWS_IV_MULTIPLIER
    T0 = DTE / 365.0
    strike = round(pr.strike_for_delta(entry_S, T0, entry_iv, DELTA, R), 2)
    entry_prem = pr.bs_call_price(entry_S, strike, T0, entry_iv, R)
    if entry_prem < 0.05:
        return None
    entry_fill = entry_prem * (1 + bt._spread_pct(entry_prem, tk, 1.0) / 2)
    qty = max(1, int(USD / (entry_fill * 100)))
    cost = entry_fill * 100 * qty

    peak, exit_prem, reason = entry_prem, None, "max_hold"
    for idx, (t, h, l, c) in enumerate(bars, start=1):
        if idx > HOLD * 7:
            break
        elapsed_cal = (t - entry_dt).total_seconds() / 86400.0
        if elapsed_cal > HOLD * 1.6:
            break
        T = max((DTE - elapsed_cal) / 365.0, 1.0 / 365.0)
        iv = pr.iv_crush_path(base_iv, idx / BARS_PER_DAY, cfg.NEWS_IV_MULTIPLIER, cfg.IV_CRUSH_HALFLIFE_DAYS)
        ph = pr.bs_call_price(h, strike, T, iv, R)            # premium at the bar HIGH
        pl = pr.bs_call_price(l, strike, T, iv, R)            # at the LOW
        pc = pr.bs_call_price(c, strike, T, iv, R)            # at the close
        # Stop uses the peak from PRIOR bars: this bar's high may NOT tighten a stop that the same bar's
        # low then breaches (that within-bar capture inflated tight trails). This bar's high only raises
        # the peak for the NEXT bar's stop.
        if pol["rule"] == "ratchet":
            tr = next(t for thresh, t in pol["tiers"] if (peak / entry_prem - 1) < thresh)
            stop = peak * (1 - tr)
        else:
            stop = peak * (1 - pol["trail"])
            if pol["rule"] == "trail_breakeven" and (peak / entry_prem - 1) >= pol["be_arm"]:
                stop = max(stop, entry_prem * (1 + pol["be_lock"]))
        hit_stop = pl <= stop
        hit_tp = pol["target"] and ph >= entry_prem * (1 + pol["target"])
        # Intra-bar order unknown; TP_FIRST=1 = optimistic (limit before stop), else pessimistic.
        if hit_tp and (TP_FIRST or not hit_stop):
            exit_prem, reason = entry_prem * (1 + pol["target"]), "take_profit"; break
        if hit_stop:
            exit_prem, reason = stop, pol["rule"]; break
        if ph > peak:                                          # only now: raise peak for the NEXT bar
            peak = ph
        exit_prem = pc                                        # else mark at close, keep holding
    if exit_prem is None:
        exit_prem = entry_prem
    exit_fill = max(0.0, exit_prem) * (1 - bt._spread_pct(exit_prem, tk, 1.0) / 2)
    pnl = exit_fill * 100 * qty - cost
    return {"pnl_usd": pnl, "pnl_pct": pnl / cost * 100 if cost else 0, "peak_ret": peak / entry_prem - 1, "reason": reason}


def report(name, trades):
    n = len(trades)
    if not n:
        print(f"  {name:24} n=0"); return
    pnls = [t["pnl_usd"] for t in trades]
    tot, mean = sum(pnls), sum(pnls) / n
    sd = statistics.pstdev(pnls) if n > 1 else 0
    win = sum(1 for p in pnls if p > 0) / n * 100
    reached = [t for t in trades if t["peak_ret"] >= ARMED]
    faded = [t for t in reached if t["pnl_pct"] < 0]
    rt = len(faded) / n * 100
    cap = (sum(1 for t in reached if t["pnl_pct"] >= 0) / len(reached) * 100) if reached else 0
    print(f"  {name:24} n={n:<4} win{win:>4.0f}%  ${mean:>+5.0f}/tr  tot${tot/1000:>5.0f}k  Sh{(mean/sd if sd else 0):>4.2f}"
          f"   |  +15%→red {rt:>4.1f}%  ({len(reached)} hit +15%, {cap:.0f}% kept green)")


def main():
    _, end, days, cache = es.BULL
    rows = es.load_signals(cache, end, days, es.PROMPT, True)
    reg = es.build_regime(end, days, 200)
    results = {p["name"]: [] for p in POLICIES}
    seen, built, skipped = set(), 0, 0
    print(f"intraday (hourly IEX) exit sweep — gating {len(rows)} news_call signals…", flush=True)
    for r in rows:
        if not GATE(r["magnitude"], r["confidence"]):
            continue
        d = r["dt"].date()
        if reg and not reg(d):
            continue
        for tk in r["tickers"][:2]:
            if not es.is_valid_stock_ticker(tk):
                continue
            key = f"{d}_{tk}"
            if key in seen:
                continue
            seen.add(key)
            entry_S = bt.get_price_at(tk, r["dt"])
            if not entry_S or entry_S > cfg.MAX_SANE_STOCK_PRICE:
                continue
            entry_dt = datetime.combine(d, dtime(21, 0), tzinfo=timezone.utc)   # ≈ session close (match daily entry)
            bars = [b for b in fetch_hourly(tk, entry_dt, entry_dt + timedelta(days=7)) if b[0] > entry_dt]
            if len(bars) < 3:
                skipped += 1; continue
            base_iv = bt.base_iv_for(tk, r["dt"])
            for pol in POLICIES:
                t = sim_intraday(tk, entry_S, entry_dt, bars, base_iv, pol)
                if t:
                    results[pol["name"]].append(t)
            built += 1
            if built % 50 == 0:
                json.dump(_cache, open(CACHE, "w")); print(f"  …{built} trades simulated", flush=True)
    json.dump(_cache, open(CACHE, "w"))
    print(f"\n  ══ news_call exit policies on HOURLY bars ({built} trades, {skipped} skipped for thin data) ══")
    for p in POLICIES:
        report(p["name"], results[p["name"]])
    print("\n  Compare to the DAILY sweep: if take-profit's win-rate edge holds (and BE-lock now separates),")
    print("  finer bars confirm it; if they converge, the daily picture was already adequate.")


if __name__ == "__main__":
    main()
