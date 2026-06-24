"""
regime_bypass_backtest.py — the live regime filter pauses ALL long-beta trades whenever SPY is
below its 200d SMA or below its 3-day-ago close. That blocks strong IDIOSYNCRATIC catalysts (e.g.
SCWX relisting, mag 0.85) purely because the market is soft. Question: on DOWN-regime days, do
high-magnitude bullish catalysts still make money (alpha survives), or does beta drag them under?

Reconstructs the exact live regime (SPY 200d SMA AND 3d momentum) as-of each pick date — no
lookahead — then buckets bullish picks by regime × magnitude and compares forward returns. Finally
simulates policies: current (longs only in up-regime) vs high-conviction bypass of the filter.

Usage:  USE_YAHOO_BARS=1 .venv/bin/python -u regime_bypass_backtest.py
"""
import os, json, bisect, random
from datetime import datetime, timezone
os.environ.setdefault("USE_YAHOO_BARS", "1")
import numpy as np
import gemini_lotto_pnl as gl
import bot                       # reuse the live Alpaca data client (Yahoo is sandboxed off)
from alpaca.data.requests import StockBarsRequest
from alpaca.data.timeframe import TimeFrame
from config import REGIME_MA_DAYS, REGIME_MOMENTUM_DAYS

MA, MOM = REGIME_MA_DAYS, REGIME_MOMENTUM_DAYS
HORIZON = "r3"   # bot holds catalyst longs ~1-3 days; r3 is where beta drag shows most


def build_regime():
    """date(trading day) → uptrend bool, replicating market_in_uptrend() exactly, point-in-time."""
    resp = bot.stock_data_client.get_stock_bars(StockBarsRequest(
        symbol_or_symbols="SPY", timeframe=TimeFrame.Day,
        start=datetime(2021, 1, 1, tzinfo=timezone.utc), feed="iex"))   # 2021 start → covers the 2022 bear window
    bars = (resp.data or {}).get("SPY", [])
    closes = [float(b.close) for b in bars]
    days = [b.timestamp.date() for b in bars]
    reg = {}
    for i in range(len(closes)):
        if i < MA:                       # need 200 prior closes for the SMA
            continue
        sma = sum(closes[i - MA + 1:i + 1]) / MA           # closes[-MA:] mean, incl. today
        above = closes[i] >= sma
        mom_ok = (i < MOM) or (closes[i] >= closes[i - MOM])
        reg[days[i]] = bool(above and mom_ok)
    return sorted(reg.keys()), reg


def regime_for(pick_date, sdays, reg):
    """Regime the bot would have used on pick_date = last trading day ≤ pick_date."""
    i = bisect.bisect_right(sdays, pick_date) - 1
    return reg.get(sdays[i]) if i >= 0 else None


def stats(rs):
    if not rs:
        return (0, 0, 0, 0, 0)
    a = np.mean(rs)
    return (len(rs), a, float(np.median(rs)),
            sum(1 for x in rs if x > 0) / len(rs) * 100,      # % positive
            sum(1 for x in rs if x >= 5) / len(rs) * 100)     # hit ≥5%


def main():
    sdays, reg = build_regime()
    print(f"regime: SPY ≥ {MA}d SMA AND ≥ {MOM}d-ago · {len(sdays)} trading days mapped "
          f"({sum(reg.values())} up / {len(reg)-sum(reg.values())} down)\n", flush=True)

    uni = json.load(open("unified_scores.json"))
    cpool = json.load(open("yahoo_candpool_cache.json"))
    gl.SAMPLE = 100000
    picks = []   # (mag, regime_up, rday, r1, r3)
    for c in gl.candidates():
        u = uni.get("unified_v1:" + c["ck"]); fr = cpool.get(f"{c['tk']}_{c['ck']}")
        if not (isinstance(u, dict) and u.get("sentiment") == "bullish" and fr and fr.get("px", 0) >= 5):
            continue
        mag = float(u.get("magnitude", 0) or 0)
        if mag < 0.35:                   # below the live magnitude gate — never a trade anyway
            continue
        up = regime_for(c["dt"].date(), sdays, reg)
        if up is None:
            continue
        picks.append((mag, up, fr["rday"], fr["r1"], fr["r3"]))
    print(f"bullish tradeable picks with regime + forward returns: {len(picks)}\n", flush=True)

    # ── regime × magnitude grid (forward return = {HORIZON}) ──
    tiers = [("0.35–0.60", 0.35, 0.60), ("0.60–0.75", 0.60, 0.75),
             ("0.75–0.85", 0.75, 0.85), ("0.85–1.00", 0.85, 1.01)]
    hi = {"r3": 4, "r1": 3, "rday": 2}[HORIZON]
    print(f"  forward {HORIZON} by regime × magnitude   (avg% · %pos · hit≥5% · n)")
    print(f"  {'magnitude':12} {'UP-regime':>28}   {'DOWN-regime':>28}")
    for name, lo, h in tiers:
        row = {True: [], False: []}
        for mag, up, rday, r1, r3 in picks:
            if lo <= mag < h:
                row[up].append((rday, r1, r3)[hi - 2])
        cells = []
        for up in (True, False):
            n, a, _, pos, hit = stats(row[up])
            cells.append(f"{a:+5.1f}% {pos:3.0f}%pos {hit:3.0f}%hit n={n:<4}" if n else f"{'—':>26} ")
        print(f"  {name:12} {cells[0]:>28}   {cells[1]:>28}")

    # ── policy simulation: total forward-return P&L proxy (equal-weight, {HORIZON}) ──
    def sim(keep):
        rs = [p[hi - 2] for p in picks if keep(p[0], p[1])]
        n, a, _, pos, hit = stats(rs)
        return n, a, sum(rs), pos, hit
    policies = [
        ("Current (longs only in UP-regime)",        lambda mag, up: up),
        ("No regime filter (trade all)",             lambda mag, up: True),
        ("Bypass: UP all + DOWN if mag≥0.75",        lambda mag, up: up or mag >= 0.75),
        ("Bypass: UP all + DOWN if mag≥0.85",        lambda mag, up: up or mag >= 0.85),
        ("Bypass: UP all + DOWN if mag≥0.90",        lambda mag, up: up or mag >= 0.90),
    ]
    print(f"\n  ── policy comparison (equal-weight {HORIZON} forward return) ──")
    print(f"  {'policy':40} {'trades':>6} {'avg':>7} {'Σ ret (P&L proxy)':>18} {'%pos':>5} {'hit':>5}")
    base_sum = None
    for label, keep in policies:
        n, a, tot, pos, hit = sim(keep)
        if base_sum is None:
            base_sum = tot
        delta = f"  (Δ {tot-base_sum:+.1f})" if label != policies[0][0] else ""
        print(f"  {label:40} {n:>6} {a:>+6.1f}% {tot:>+15.1f}%{delta:>11} {pos:>4.0f}% {hit:>4.0f}%")
    print("\n  Σ ret = sum of equal-weight forward returns = relative P&L if every kept pick is sized equally.")
    print("  A bypass that lifts Σ ret above Current = the high-conviction catalysts recover more than they lose.")


if __name__ == "__main__":
    main()
