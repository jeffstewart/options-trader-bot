"""
lotto_entry_cutoff_test.py — jeff's question (2026-07-18): since v2's lotto exit rule force-closes
every position at end-of-day regardless of P&L (LOTTO_SAME_DAY_EXIT), a signal that fires late in
the trading day gives the position almost no runway to develop before that forced exit. Buying and
immediately EOD-selling a contract that never had time to move is pure cost (spread + theta), not
a real bet. This backtests where a cutoff should sit.

Uses the OLLAMA-scored LIVE gate (mag>=0.70, conf>=0.85 -- matches v2/config.py's
LOTTO_MIN_MAGNITUDE/CONFIDENCE) and the EXACT exit rule now wired into v2 (same-day EOD + 20%
entry-anchored stop-loss + 3x hard cap). Same Δ0.20/DTE14 contract geometry and 15-min-bar
Black-Scholes-off-real-stock-bars pricing as every other exit-rule test this cycle (continuity
with the numbers that validated the exit rule and the position sizing, not a new methodology).

Buckets each candidate by HOURS-UNTIL-CLOSE at entry (RTH is 9:30am-4:00pm ET, so max runway at
the open is 6.5h) and reports win%/avg $/total $/Sharpe per bucket, looking for where performance
degrades as runway shrinks -- that's the cutoff.

REAL DATA SURPRISE (worth knowing before rerunning): the live gate's cache is capped regardless of
the `days` window requested (189 gated signals at both 400d and 700d -- the underlying cache
itself is finite, not the date range), and gated signals cluster HEAVILY in the 2pm-6pm ET window
(verified directly against raw timestamps, not a timezone bug) -- most of the "early/mid day"
buckets have ~zero samples. Had to loosen the gate (CUTOFF_MAG/CUTOFF_CONF env vars) to mag>=0.4/
conf>=0.70 for n=106 total, on the reasoning that "does a position have enough runway before a
forced same-day exit" is an intraday-mechanics question that should generalize across gate
thresholds, unlike a signal-quality question.

RESULT (2026-07-18, n=106 across buckets): NOT a clean monotonic "more runway = better" gradient.
1.5-2.0h was the worst bucket by win rate (11%, n=19) while 0.5-1.0h (n=38, the largest bucket)
was actually the BEST (42% win, only net-positive bucket, +$295 total). Read: lotto's convex few-
big-winners structure means per-bucket averages at n=18-38 are noise-dominated -- don't trust the
middle buckets. The one internally-consistent signal: 0.0-0.5h before close (n=7, thin) had both
the worst avg $ (-$18) and a below-average win rate (29%) of any populated bucket, matching the
mechanical expectation (no time for the stop-loss or a real move to matter before the forced
exit). Recommended and shipped: LOTTO_ENTRY_CUTOFF_MIN_BEFORE_CLOSE=30 in v2/config.py -- small
and conservative, deliberately doesn't touch the well-performing 0.5-1.0h window since a larger
cutoff isn't supported by this data. Wired into v2/execution.py's execute_lotto via
market.near_market_close(within_min=...).

Usage:  USE_YAHOO_BARS=1 .venv/bin/python -u lotto_entry_cutoff_test.py   (run from data/)
        CUTOFF_MAG=0.4 CUTOFF_CONF=0.70 CUTOFF_DAYS=400 ... for more statistical power (looser gate)
"""
import os, sys, statistics
os.environ.setdefault("USE_YAHOO_BARS", "1")
from datetime import timedelta
from zoneinfo import ZoneInfo

sys.path.insert(0, "../core")
import config as _cfg
from pricing import bs_call_price, strike_for_delta, iv_crush_path
from backtest import base_iv_for, get_price_at, is_valid_stock_ticker
from regime_filter import build_regime
import news_call_sweep_unified as nc

from alpaca.data.historical import StockHistoricalDataClient
from alpaca.data.requests import StockBarsRequest
from alpaca.data.timeframe import TimeFrame, TimeFrameUnit
from dotenv import load_dotenv
load_dotenv("../.env")

ALPACA_KEY = os.environ["ALPACA_API_KEY"]
ALPACA_SECRET = os.environ["ALPACA_SECRET_KEY"]
hourly_client = StockHistoricalDataClient(ALPACA_KEY, ALPACA_SECRET)

ET = ZoneInfo("America/New_York")
DELTA, DTE = 0.20, 14
NEWS_IV_MULT, IV_HALFLIFE = _cfg.__dict__.get("NEWS_IV_MULTIPLIER", 1.10), 3.0
R = 0.04
BAR_RES = TimeFrame(15, TimeFrameUnit.Minute)
STOP_LOSS_PCT = 0.20
HARD_CAP_MULT = 3.0
LIVE_MAG = float(os.environ.get("CUTOFF_MAG", "0.70"))    # default matches v2's live gate;
LIVE_CONF = float(os.environ.get("CUTOFF_CONF", "0.85"))  # loosen via env for more statistical
                                                            # power -- runway-before-forced-exit is
                                                            # an intraday-mechanics question, not a
                                                            # signal-quality one, so it should
                                                            # generalize across gate thresholds
BUDGET = 100

# (label, lo_hours, hi_hours) -- hours-until-close AT ENTRY. RTH open is 6.5h before the 4pm close.
BUCKETS = [
    ("6.0-6.5h (at the open)", 6.0, 6.5),
    ("5.0-6.0h", 5.0, 6.0),
    ("4.0-5.0h", 4.0, 5.0),
    ("3.0-4.0h", 3.0, 4.0),
    ("2.0-3.0h", 2.0, 3.0),
    ("1.5-2.0h", 1.5, 2.0),
    ("1.0-1.5h", 1.0, 1.5),
    ("0.5-1.0h", 0.5, 1.0),
    ("0.0-0.5h (right before close)", 0.0, 0.5),
]


def get_bars(ticker, start, end):
    try:
        resp = hourly_client.get_stock_bars(StockBarsRequest(
            symbol_or_symbols=ticker, timeframe=BAR_RES, start=start, end=end, feed="iex"))
        bars = (resp.data or {}).get(ticker, [])
        return [{"t": b.timestamp, "c": float(b.close)} for b in bars]
    except Exception:
        return []


def build_path(ticker, entry_dt, entry_price):
    base_iv = base_iv_for(ticker, entry_dt)
    entry_iv = base_iv * NEWS_IV_MULT
    T0 = DTE / 365.0
    strike = round(strike_for_delta(entry_price, T0, entry_iv, DELTA, R), 2)
    entry_premium = bs_call_price(entry_price, strike, T0, entry_iv, R)
    if entry_premium < 0.05:
        return None

    bars = get_bars(ticker, entry_dt, entry_dt + timedelta(hours=10))
    bars = [b for b in bars if b["t"] > entry_dt]
    if not bars:
        return None

    path = []
    for b in bars:
        hrs = (b["t"] - entry_dt).total_seconds() / 3600
        days_held = hrs / 24
        T_now = max((DTE - days_held) / 365.0, 1 / 365.0)
        iv_now = iv_crush_path(base_iv, days_held, NEWS_IV_MULT, IV_HALFLIFE)
        prem = bs_call_price(b["c"], strike, T_now, iv_now, R)
        path.append((hrs, b["t"], prem))
    return {"entry_premium": entry_premium, "path": path}


def exit_v2_live(pv):
    """Exactly v2/execution.py's trailing_stop_monitor lotto logic: hard 3x cap, entry-anchored
    20% stop, else forced same-day EOD exit."""
    entry = pv["entry_premium"]
    entry_day = pv["path"][0][1].date()
    stop = entry * (1 - STOP_LOSS_PCT)
    last = pv["path"][0][2]
    for hrs, t, prem in pv["path"]:
        if t.date() != entry_day:
            break
        if prem >= HARD_CAP_MULT * entry:
            return prem
        if prem <= stop:
            return prem
        last = prem
    return last


def qty_for(entry_premium):
    cost = entry_premium * 100
    if cost > BUDGET * 1.0:
        return None
    return max(1, int(BUDGET / cost))


def hours_until_close(entry_dt):
    et = entry_dt.astimezone(ET)
    close_et = et.replace(hour=16, minute=0, second=0, microsecond=0)
    return (close_et - et).total_seconds() / 3600


def bucket_for(hrs):
    for name, lo, hi in BUCKETS:
        if lo <= hrs < hi or (hi == 6.5 and hrs >= 6.0):
            return name
    return None


def main():
    label, end_dt, _, cache = nc.BULL
    days = int(os.environ.get("CUTOFF_DAYS", "180"))
    rows = nc.load_scored_from_unified(cache, end_dt, days, "unified_v1")
    reg = build_regime(end_dt, days, 200)
    gated = [r for r in rows if r["magnitude"] >= LIVE_MAG and r["confidence"] >= LIVE_CONF]
    gated = [r for r in gated if reg is None or reg(r["created_at"].date())]
    print(f"{label}: live gate mag>={LIVE_MAG} conf>={LIVE_CONF} -> {len(gated)} signals (post-regime)\n")

    results = {name: [] for name, *_ in BUCKETS}
    n_candidates = n_in_rth = n_price_ok = n_priced = 0
    seen = set()

    for r in gated:
        d = r["created_at"].date()
        for tk in r["tickers"][:1]:
            if not is_valid_stock_ticker(tk):
                continue
            key = f"{d}_{tk}"
            if key in seen:
                continue
            seen.add(key)
            n_candidates += 1

            hrs_left = hours_until_close(r["created_at"])
            if hrs_left < 0 or hrs_left > 6.5:
                continue   # outside RTH -- not a live-tradeable entry moment anyway
            bucket = bucket_for(hrs_left)
            if bucket is None:
                continue
            n_in_rth += 1

            sp = get_price_at(tk, r["created_at"])
            if not sp:
                continue
            n_price_ok += 1

            pv = build_path(tk, r["created_at"], sp)
            if not pv:
                continue
            qty = qty_for(pv["entry_premium"])
            if not qty:
                continue
            n_priced += 1

            exit_prem = exit_v2_live(pv)
            pnl = (exit_prem - pv["entry_premium"]) * 100 * qty
            results[bucket].append(pnl)

    print(f"deduped ticker/day candidates: {n_candidates}")
    print(f"  in RTH (0-6.5h before close): {n_in_rth}")
    print(f"  had a stock price: {n_price_ok}")
    print(f"  priced + affordable (final): {n_priced}\n")
    print(f"{'hours-until-close':30} {'n':>4} {'total$':>9} {'avg$':>7} {'win%':>6} {'sharpe':>7}")
    for name, *_ in BUCKETS:
        pnls = results[name]
        if len(pnls) < 3:
            print(f"  {name:28} {len(pnls):>4}  (insufficient sample)")
            continue
        n = len(pnls); tot = sum(pnls); avg = tot / n
        sd = statistics.pstdev(pnls) if n > 1 else 0
        sh = avg / sd if sd else 0
        win = sum(1 for p in pnls if p > 0) / n * 100
        print(f"  {name:28} {n:>4} ${tot:>+8,.0f} ${avg:>+6,.0f} {win:>5.0f}% {sh:>+6.2f}")


if __name__ == "__main__":
    main()
