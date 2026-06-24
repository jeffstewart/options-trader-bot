"""
calibrate_iv.py — anchor the backtest's IV-crush model to real option prints.

The backtest prices long calls with Black-Scholes off the dense stock bars,
inflating implied vol at entry by NEWS_IV_MULTIPLIER to model the post-news IV
spike. This script estimates that multiplier empirically:

  for each historical trade the strategy took:
    1. find the real listed call nearest our target (≈37 DTE, ~0.40 delta)
    2. pull the real option bar closest to the entry date
    3. invert Black-Scholes on that real premium to get the REAL implied vol
    4. compare to the underlying's realized vol (our model's base IV)
  → NEWS_IV_MULTIPLIER ≈ median(real_iv / base_iv)

Historical option prints are sparse, so only a fraction of trades yield a
usable data point — the script reports n and the full distribution so you can
judge how much to trust the result.

Usage:
    .venv/bin/python calibrate_iv.py [--limit N] [--trades backtest_trades.csv]
"""

import argparse
import csv
import os
import statistics
from datetime import datetime, timedelta, timezone

from dotenv import load_dotenv
load_dotenv()

from alpaca.data.historical.stock import StockHistoricalDataClient
from alpaca.data.historical.option import OptionHistoricalDataClient
from alpaca.data.requests import StockBarsRequest, OptionBarsRequest
from alpaca.data.timeframe import TimeFrame
from alpaca.trading.client import TradingClient
from alpaca.trading.requests import GetOptionContractsRequest
from alpaca.trading.enums import ContractType, AssetStatus

from config import (
    ALPACA_KEY, ALPACA_SECRET, RISK_FREE_RATE,
    MIN_DAYS_TO_EXPIRY, MAX_DAYS_TO_EXPIRY, TARGET_DELTA,
)
from pricing import implied_vol_call, strike_for_delta, realized_vol, bs_call_delta

sc  = StockHistoricalDataClient(ALPACA_KEY, ALPACA_SECRET)
odc = OptionHistoricalDataClient(ALPACA_KEY, ALPACA_SECRET)
tc  = TradingClient(ALPACA_KEY, ALPACA_SECRET, paper=True)

DTE_TARGET = round((MIN_DAYS_TO_EXPIRY + MAX_DAYS_TO_EXPIRY) / 2)


def stock_closes(ticker, start, end):
    try:
        resp = sc.get_stock_bars(StockBarsRequest(
            symbol_or_symbols=ticker, timeframe=TimeFrame.Day,
            start=start, end=end, feed="iex"))
        return [(b.timestamp, float(b.close)) for b in (resp.data or {}).get(ticker, [])]
    except Exception:
        return []


def base_iv(ticker, entry_dt):
    closes = stock_closes(ticker, entry_dt - timedelta(days=45), entry_dt)
    rv = realized_vol([c for _, c in closes][-21:])
    return rv


def real_entry_iv(ticker, entry_dt, entry_stock_price):
    """Return (real_iv, base_iv, delta_of_contract) or None if no usable print."""
    biv = base_iv(ticker, entry_dt)
    if biv is None:
        return None

    # Target strike at ~0.40 delta using the realized-vol estimate
    T = DTE_TARGET / 365.0
    target_strike = strike_for_delta(entry_stock_price, T, biv, TARGET_DELTA, RISK_FREE_RATE)

    # Discover the real listed call nearest target strike & expiry window
    try:
        resp = tc.get_option_contracts(GetOptionContractsRequest(
            underlying_symbols=[ticker], type=ContractType.CALL,
            expiration_date_gte=(entry_dt + timedelta(days=MIN_DAYS_TO_EXPIRY)).date().isoformat(),
            expiration_date_lte=(entry_dt + timedelta(days=MAX_DAYS_TO_EXPIRY)).date().isoformat(),
            strike_price_gte=str(round(target_strike * 0.90, 2)),
            strike_price_lte=str(round(target_strike * 1.10, 2)),
            status=AssetStatus.INACTIVE, limit=50,
        ))
    except Exception:
        return None
    contracts = getattr(resp, "option_contracts", []) or []
    if not contracts:
        return None
    contracts.sort(key=lambda c: abs(float(c.strike_price) - target_strike))
    contract = contracts[0]
    K = float(contract.strike_price)
    expiry = datetime.fromisoformat(str(contract.expiration_date)).replace(tzinfo=timezone.utc)

    # Pull real option bars around entry; take the print closest to entry_dt
    try:
        bars_resp = odc.get_option_bars(OptionBarsRequest(
            symbol_or_symbols=contract.symbol, timeframe=TimeFrame.Day,
            start=entry_dt - timedelta(days=3), end=entry_dt + timedelta(days=5)))
        bars = (bars_resp.data or {}).get(contract.symbol, [])
    except Exception:
        return None
    if not bars:
        return None
    bar = min(bars, key=lambda b: abs((b.timestamp - entry_dt).total_seconds()))
    premium = float(bar.close)

    # Stock price on the bar's date (S for the BS inversion)
    sclose = stock_closes(ticker, bar.timestamp - timedelta(days=4), bar.timestamp + timedelta(days=1))
    if not sclose:
        return None
    S = min(sclose, key=lambda t: abs((t[0] - bar.timestamp).total_seconds()))[1]

    T_real = max((expiry - bar.timestamp).days / 365.0, 1.0 / 365.0)
    iv = implied_vol_call(premium, S, K, T_real, RISK_FREE_RATE)
    if iv is None:
        return None
    delta = bs_call_delta(S, K, T_real, iv, RISK_FREE_RATE)
    return iv, biv, delta


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--trades", default="backtest_trades.csv")
    ap.add_argument("--limit", type=int, default=250,
                    help="max trades to sample (API calls are the bottleneck)")
    args = ap.parse_args()

    with open(args.trades, newline="") as f:
        rows = list(csv.DictReader(f))
    # Spread the sample across the whole window rather than the first N
    if len(rows) > args.limit:
        step = len(rows) / args.limit
        rows = [rows[int(i * step)] for i in range(args.limit)]

    print(f"Calibrating against {len(rows)} sampled trades "
          f"(target ≈{DTE_TARGET} DTE, {TARGET_DELTA:.2f} delta)…\n")

    ratios, ivs, deltas = [], [], []
    usable = 0
    for i, row in enumerate(rows, 1):
        ticker = row.get("ticker", "")
        try:
            entry_dt = datetime.fromisoformat(row["entry_dt"])
            if entry_dt.tzinfo is None:
                entry_dt = entry_dt.replace(tzinfo=timezone.utc)
            S = float(row["entry_stock_price"])
        except Exception:
            continue
        if "/" in ticker:  # any stray crypto
            continue
        res = real_entry_iv(ticker, entry_dt, S)
        if res:
            iv, biv, delta = res
            if biv > 0:
                ratios.append(iv / biv)
                ivs.append(iv)
                deltas.append(delta)
                usable += 1
        if i % 25 == 0:
            print(f"  …{i}/{len(rows)} scanned, {usable} usable prints so far")

    print(f"\n{'='*60}\nRESULTS\n{'='*60}")
    print(f"  Trades sampled        : {len(rows)}")
    print(f"  Usable real prints    : {usable}  ({usable/max(len(rows),1)*100:.0f}%)")
    if usable < 5:
        print("\n  ⚠️  Too few usable prints to calibrate reliably.")
        print("     Historical option data is too sparse for these names/dates.")
        print("     Keep the current NEWS_IV_MULTIPLIER default and lean on")
        print("     forward paper-trading to validate. ")
        return
    ratios.sort()
    print(f"  real_iv / base_iv (the NEWS_IV_MULTIPLIER):")
    print(f"     median : {statistics.median(ratios):.3f}   ← suggested value")
    print(f"     mean   : {statistics.mean(ratios):.3f}")
    print(f"     p25–p75: {ratios[len(ratios)//4]:.3f} – {ratios[3*len(ratios)//4]:.3f}")
    print(f"     min/max: {ratios[0]:.3f} / {ratios[-1]:.3f}")
    print(f"  real entry IV (abs)   : median {statistics.median(ivs):.3f}")
    print(f"  contract deltas seen  : median {statistics.median(deltas):.3f} "
          f"(target {TARGET_DELTA:.2f})")
    print(f"\n  → Set NEWS_IV_MULTIPLIER = {statistics.median(ratios):.2f} in config.py "
          f"if the sample size feels adequate.")


if __name__ == "__main__":
    main()
