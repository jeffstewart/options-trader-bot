"""
lotto_intraday_exit_test.py — jeff's two hypotheses on lotto exits, tested at HOURLY granularity
(the current backtest only sees DAILY closes, so it's structurally blind to same-day whipsaws --
exactly the "green then red" pattern jeff is worried about):

  1. Many lotto trades resolve shortly after the open (real v1 data: 7 of 10 non-artifact closes
     land in the 13:30-15:00 UTC / 9:30-11:00am ET window, all overnight holds from a prior-day
     entry) -- does an EOD-of-entry-day or next-morning-early exit rule capture most of the
     eventual move, or leave money on the table vs. the current 3-day hold?
  2. A much tighter trail once a position goes green (lock in gains fast, still let it run) --
     does this actually reduce losses/improve the shape vs. the current wide 40/30/20% tiered
     "let it run" trail, once you can see intraday moves instead of just daily closes?

Approximates option premium hourly via Black-Scholes off hourly stock bars (same pricing model
the daily backtest uses -- pricing.bs_call_price + backtest.base_iv_for/iv_crush_path -- just
evaluated at finer time steps), for the sonnet5 lotto candidate set (mag>=0.3, conf>=0.70 --
n=41, 39% win, the best reasonably-sized cell from the selectivity grid).

Usage:  USE_YAHOO_BARS=1 .venv/bin/python -u lotto_intraday_exit_test.py   (run from data/)

UPDATED 2026-07-31: added exit_trailing_stop (jeff's live v2 rule: stop = max(entry floor,
peak*(1-trail)), same-day EOD boundary), and FIXED a confound -- exit_eod_same_day /
exit_eod_with_stop / exit_eod_with_lock did not apply the 3x hard cap that the live bot and the
other rules do, so their total$ was inflated by riding winners past a level the bot would have
sold at. Every rule now measures against the same live constraint.
"""
import os, sys, statistics
os.environ.setdefault("USE_YAHOO_BARS", "1")
from datetime import datetime, timedelta, timezone

sys.path.insert(0, "../core")
import config as _cfg
from pricing import bs_call_price, strike_for_delta, iv_crush_path
from backtest import base_iv_for, get_price_at, is_valid_stock_ticker
from regime_filter import build_regime
import small_account_sonnet5_sweep as s5

from alpaca.data.historical import StockHistoricalDataClient
from alpaca.data.requests import StockBarsRequest
from alpaca.data.timeframe import TimeFrame
from dotenv import load_dotenv
load_dotenv("../.env")

ALPACA_KEY = os.environ["ALPACA_API_KEY"]
ALPACA_SECRET = os.environ["ALPACA_SECRET_KEY"]
hourly_client = StockHistoricalDataClient(ALPACA_KEY, ALPACA_SECRET)

DELTA, DTE = 0.20, 14
BUDGET = int(os.environ.get("LOTTO_TEST_BUDGET", "100"))
# Live lotto takes profit at LOTTO_HARD_CAP_MULT x entry (3.0). Set LOTTO_TEST_CAP=0 to disable it
# and measure what the cap itself is worth -- it exits 9 of 14 trades here, so it is the single
# most influential rule in the file and every other comparison is conditional on it.
_cap_env = float(os.environ.get("LOTTO_TEST_CAP", "3.0"))
HARD_CAP = _cap_env if _cap_env > 0 else float("inf")
NEWS_IV_MULT, IV_HALFLIFE = _cfg.__dict__.get("NEWS_IV_MULTIPLIER", 1.10), 3.0
R = 0.04
TIERED = [(1.0, 0.40), (3.0, 0.30), (float("inf"), 0.20)]   # current live tiered trail


from alpaca.data.timeframe import TimeFrameUnit
BAR_RES = TimeFrame(15, TimeFrameUnit.Minute)   # v2's real monitor polls every 30s; 15min is the
                                                # finest Alpaca offers short of tick data -- far
                                                # closer to real stop behavior than hourly bars,
                                                # which were shown to understate a stop's real
                                                # protective power (a trade exited "on stop" at
                                                # -96.6% because the next HOURLY check was already
                                                # way past the 20% trigger).


def get_hourly_bars(ticker, start, end):
    try:
        resp = hourly_client.get_stock_bars(StockBarsRequest(
            symbol_or_symbols=ticker, timeframe=BAR_RES, start=start, end=end, feed="iex"))
        bars = (resp.data or {}).get(ticker, [])
        return [{"t": b.timestamp, "c": float(b.close)} for b in bars]
    except Exception:
        return []


def build_path(ticker, entry_dt, entry_price):
    """Hourly (entry->+72h) option-premium path via BS, matching the live pricer's model."""
    base_iv = base_iv_for(ticker, entry_dt)
    entry_iv = base_iv * NEWS_IV_MULT
    T0 = DTE / 365.0
    strike = round(strike_for_delta(entry_price, T0, entry_iv, DELTA, R), 2)
    entry_premium = bs_call_price(entry_price, strike, T0, entry_iv, R)
    if entry_premium < 0.05:
        return None

    bars = get_hourly_bars(ticker, entry_dt, entry_dt + timedelta(hours=90))
    bars = [b for b in bars if b["t"] > entry_dt]
    if not bars:
        return None

    path = []   # (hours_held, stock_px, premium)
    for b in bars:
        hrs = (b["t"] - entry_dt).total_seconds() / 3600
        days_held = hrs / 24
        T_now = max((DTE - days_held) / 365.0, 1 / 365.0)
        iv_now = iv_crush_path(base_iv, days_held, NEWS_IV_MULT, IV_HALFLIFE)
        prem = bs_call_price(b["c"], strike, T_now, iv_now, R)
        path.append((hrs, b["t"], b["c"], prem))
    return {"entry_premium": entry_premium, "path": path, "budget": BUDGET}


def qty_for(entry_premium):
    cost = entry_premium * 100
    if cost > BUDGET * 1.0:   # lotto's live hard cap: never exceed budget
        return None
    return max(1, int(BUDGET / cost))


def exit_tiered_3day(pv):
    """Baseline: current live rule -- tiered 40/30/20% trail, 72h (3d) max hold."""
    entry = pv["entry_premium"]; peak = entry
    for hrs, t, px, prem in pv["path"]:
        if hrs > 72:
            break
        peak = max(peak, prem)
        gain = peak / entry - 1.0
        trail = next(tr for thresh, tr in TIERED if gain < thresh)
        stop = peak * (1 - trail)
        if prem <= stop:
            return prem, hrs, "trail-stop"
        if prem >= HARD_CAP * entry:   # lotto hard profit cap (3x)
            return prem, hrs, "hard-cap"
    last = pv["path"][min(len(pv["path"]), int(72)) - 1] if pv["path"] else None
    final = next((p for h, t, x, p in pv["path"] if h <= 72), pv["path"][-1][3])
    for h, t, x, p in reversed(pv["path"]):
        if h <= 72:
            return p, h, "time-stop"
    return pv["path"][-1][3], pv["path"][-1][0], "time-stop"


def exit_trailing_stop(pv, trail_pct, same_day_eod=True):
    """v2's LIVE rule as of 2026-07-31 (jeff): stop = max(entry floor, peak * (1 - trail_pct)),
    with the same-day EOD exit as the outer boundary.

    This is the variant the original run never tested. Its stop-loss rows were EOD exits with a
    static entry-anchored floor that fired once in 14 trades -- so the sweep measured the EOD rule,
    not the stop. Here the stop moves continuously with the peak, so it should actually bind.
    Floor stays at LOTTO_STOP_LOSS_PCT (0.20) so the worst case is unchanged.
    """
    entry = pv["entry_premium"]
    peak = entry
    floor = entry * (1 - 0.20)
    entry_day = None
    last = (pv["path"][0][3], pv["path"][0][0], "eod") if pv["path"] else (entry, 0, "eod")
    for hrs, t, px, prem in pv["path"]:
        if entry_day is None:
            entry_day = t.date()
        if same_day_eod and t.date() > entry_day:
            return last[0], last[1], "eod-same-day"
        peak = max(peak, prem)
        stop = max(floor, peak * (1 - trail_pct))
        if prem <= stop:
            return prem, hrs, "trail-stop"
        if prem >= HARD_CAP * entry:
            return prem, hrs, "hard-cap"
        last = (prem, hrs, "eod")
    return last

def exit_eod_same_day(pv):
    """Force-exit at the close of the SAME calendar day as entry (first bar >= 20:00 UTC, or the
    last available bar before the next day if the market closes sooner)."""
    entry_day = None
    for hrs, t, px, prem in pv["path"]:
        if entry_day is None:
            entry_day = t.date()
        if t.date() != entry_day:
            # crossed into the next day -- exit at the LAST bar we saw on the entry day
            break
        if prem >= HARD_CAP * pv['entry_premium']:
            return prem, hrs, 'hard-cap'
        last = (prem, hrs, "eod-same-day")
    return last if entry_day else exit_tiered_3day(pv)


def exit_next_morning(pv, minutes_after_open=90):
    """Exit ~90 min after the NEXT trading day's open (matches the real-trade clustering:
    13:30-15:00 UTC). Falls back to the tiered rule if a stop/cap would have fired first."""
    entry_day = pv["path"][0][1].date()
    next_day_open_seen = False
    for hrs, t, px, prem in pv["path"]:
        # tiered stop can still fire first (protects the downside on the way there)
        peak = max((p for h, tt, x, p in pv["path"] if h <= hrs), default=pv["entry_premium"])
        gain = peak / pv["entry_premium"] - 1.0
        trail = next(tr for thresh, tr in TIERED if gain < thresh)
        if prem <= peak * (1 - trail):
            return prem, hrs, "trail-stop(pre-open)"
        if t.date() != entry_day:
            if not next_day_open_seen:
                next_day_open_seen = True
                open_hr = hrs
            if hrs - open_hr >= minutes_after_open / 60:
                return prem, hrs, "next-morning-exit"
    return pv["path"][-1][3], pv["path"][-1][0], "held-to-end"


def exit_eod_with_stop(pv, stop_loss_pct):
    """jeff's actual ask: same-day exit as the default, but with a hard stop-loss anchored to
    ENTRY (not the peak-trailing tiered mechanic) that can fire intraday if the trade goes
    badly wrong -- riding a loser all the way to the close, no matter how far red, is what plain
    exit_eod_same_day does and that's the gap being closed here."""
    entry = pv["entry_premium"]
    entry_day = pv["path"][0][1].date()
    stop = entry * (1 - stop_loss_pct)
    last = (pv["path"][0][3], pv["path"][0][0], "eod")
    for hrs, t, px, prem in pv["path"]:
        if t.date() != entry_day:
            break
        if prem <= stop:
            return prem, hrs, "stop-loss(intraday)"
        if prem >= HARD_CAP * pv['entry_premium']:
            return prem, hrs, 'hard-cap'
        last = (prem, hrs, "eod")
    return last


def exit_breakeven_lock(pv, lock_gain=0.10, tight_trail=0.15):
    """jeff's risk preference: once up `lock_gain`, ratchet the stop to AT LEAST breakeven and
    tighten the trail to `tight_trail` -- so a green trade can't round-trip to red -- but before
    that threshold, use the current wide trail (don't choke off a move before it develops)."""
    entry = pv["entry_premium"]; peak = entry; locked = False
    for hrs, t, px, prem in pv["path"]:
        if hrs > 72:
            break
        peak = max(peak, prem)
        gain = peak / entry - 1.0
        if gain >= lock_gain:
            locked = True
        trail = tight_trail if locked else next(tr for thresh, tr in TIERED if gain < thresh)
        stop = peak * (1 - trail)
        if locked:
            stop = max(stop, entry * 1.01)   # never below breakeven once locked
        if prem <= stop:
            return prem, hrs, "breakeven-lock" if locked else "trail-stop"
        if prem >= HARD_CAP * entry:
            return prem, hrs, "hard-cap"
    for h, t, x, p in reversed(pv["path"]):
        if h <= 72:
            return p, h, "time-stop"
    return pv["path"][-1][3], pv["path"][-1][0], "time-stop"


def exit_eod_with_lock(pv, lock_gain=0.10, tight_trail=0.15):
    """Combines both of jeff's asks: same-day is the hard outer boundary, but a breakeven-lock
    trail runs WITHIN the day so an intraday peak-then-fade doesn't ride all the way to the close."""
    entry = pv["entry_premium"]; peak = entry; locked = False
    entry_day = pv["path"][0][1].date()
    last = (pv["path"][0][3], pv["path"][0][0], "eod-with-lock")
    for hrs, t, px, prem in pv["path"]:
        if t.date() != entry_day:
            break
        if prem >= HARD_CAP * pv['entry_premium']:
            return prem, hrs, 'hard-cap'
        peak = max(peak, prem)
        gain = peak / entry - 1.0
        if gain >= lock_gain:
            locked = True
        trail = tight_trail if locked else next(tr for thresh, tr in TIERED if gain < thresh)
        stop = peak * (1 - trail)
        if locked:
            stop = max(stop, entry * 1.01)
        if prem <= stop:
            return prem, hrs, "breakeven-lock(intraday)" if locked else "trail-stop(intraday)"
        last = (prem, hrs, "eod-with-lock")
    return last


def stat(name, rows):
    if not rows:
        print(f"  {name:28} n=0"); return
    pnls = [r[0] for r in rows]
    n = len(pnls); tot = sum(pnls); avg = tot / n
    sd = statistics.pstdev(pnls) if n > 1 else 0
    sh = avg / sd if sd else 0
    win = sum(1 for p in pnls if p > 0) / n * 100
    losers = [p for p in pnls if p <= 0]
    worst = min(pnls)
    avg_loss = statistics.mean(losers) if losers else 0
    print(f"  {name:28} n={n:<3} Σ${tot:>+8,.0f}  avg${avg:>+6,.0f}  win{win:>4.0f}%  sharpe{sh:>+5.2f}  "
          f"avgLoss${avg_loss:>+6,.0f}  worst${worst:>+7,.0f}")


def main():
    rows = s5.load_sonnet5_rows()
    reg = build_regime(s5.END_DT, s5.DAYS, 200)
    gated = [r for r in rows if r["magnitude"] >= 0.3 and r["confidence"] >= 0.70]
    gated = [r for r in gated if reg is None or reg(r["created_at"].date())]
    print(f"candidate set: mag>=0.3 conf>=0.70 -> {len(gated)} signals (post-regime)")

    STOP_LOSSES = [0.15, 0.20, 0.25, 0.30, 0.35]
    names = (["baseline_3d_tiered", "eod_same_day", "next_morning_90m", "breakeven_lock", "eod_with_lock",
                "trail_15pct", "trail_20pct"]
              + [f"eod_stop_{int(s*100)}pct" for s in STOP_LOSSES])
    results = {n: [] for n in names}
    exit_reasons = {n: [] for n in names}
    n_priced = 0
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
            sp = get_price_at(tk, r["created_at"])
            if not sp:
                continue
            pv = build_path(tk, r["created_at"], sp)
            if not pv:
                continue
            qty = qty_for(pv["entry_premium"])
            if not qty:
                continue
            n_priced += 1
            fns = [("baseline_3d_tiered", exit_tiered_3day),
                   ("eod_same_day", exit_eod_same_day),
                   ("next_morning_90m", exit_next_morning),
                   ("breakeven_lock", exit_breakeven_lock),
                   ("eod_with_lock", exit_eod_with_lock),
                   ("trail_15pct", lambda pv: exit_trailing_stop(pv, 0.15)),
                   ("trail_20pct", lambda pv: exit_trailing_stop(pv, 0.20))]
            for s in STOP_LOSSES:
                fns.append((f"eod_stop_{int(s*100)}pct", lambda pv, s=s: exit_eod_with_stop(pv, s)))
            for name, fn in fns:
                exit_prem, hrs, reason = fn(pv)
                pnl = (exit_prem - pv["entry_premium"]) * 100 * qty
                results[name].append((pnl, hrs, tk))
                exit_reasons[name].append(reason)

    print(f"priced + affordable: {n_priced}\n")
    from collections import Counter
    for name in names:
        stat(name, results[name])
        print(f"    exit reasons: {dict(Counter(exit_reasons[name]))}")
        print()


if __name__ == "__main__":
    main()
