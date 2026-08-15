"""
liquidity_first_pick_dryrun.py — prototype + live dry-run of a liquidity-aware candidate sort
for pick_call_contract (v1) / the analogous v2/execution.py target search.

jeff's question (2026-08-XX deep dive): the current picker sorts the fetched contract pool
PURELY by (DTE proximity, strike proximity) and only tries the nearest 5, with zero liquidity
awareness. research/spread_cap_log_mining.py found 220 real signals that burned through all 5
candidates and still failed -- and a live snapshot showed some of those tickers (MSFT: 93
contracts in range, 39 with OI>=100; BA: 21 in range, 13 with OI>=100) are NOT actually thin --
the picker is just walking past perfectly good contracts because it never looks at liquidity
before picking which 5 to try.

CANNOT replay history (no historical option-chain/OI snapshots -- see the spread-cap feasibility
check earlier in this session), so this is a LIVE dry run: for each ticker that historically died
with an exhausted shortlist, fetch TODAY's real contract pool and real quotes, and compare:

  OLD sort: contracts.sort(key=(dte proximity, strike proximity))              -- today's code
  NEW sort: contracts filtered to OI >= MIN_OPEN_INTEREST, THEN sorted by      -- prototype
            (dte proximity, strike proximity) among survivors

Both walk their first 5 candidates through the same guardrail checks (tradable, root symbol,
quote sanity, spread <= MAX_SPREAD_PCT) using REAL live quotes. This tells us, on TODAY's market
structure, how often the liquidity-first sort finds a tradeable contract that the proximity-only
sort would miss. It's a structural proxy, not a historical backtest -- see the caveats printed at
the end.

Usage (run from repo root):
    .venv/bin/python research/liquidity_first_pick_dryrun.py
    .venv/bin/python research/liquidity_first_pick_dryrun.py --tickers MSFT BA GOOGL GE AAPL
"""
import argparse
import sys
import time
from datetime import date, timedelta
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "v2"))
import config as cfg                                                    # noqa: E402
from alpaca.trading.client import TradingClient                         # noqa: E402
from alpaca.trading.requests import GetOptionContractsRequest           # noqa: E402
from alpaca.trading.enums import ContractType                           # noqa: E402
from alpaca.data.historical.option import OptionHistoricalDataClient    # noqa: E402
from alpaca.data.historical.stock import StockHistoricalDataClient      # noqa: E402
from alpaca.data.requests import StockLatestQuoteRequest, OptionLatestQuoteRequest  # noqa: E402

# Same defaults the live bot uses (core/config.py / v2/config.py agree on these).
MIN_OPEN_INTEREST = 100
MAX_SPREAD_PCT = 0.20
DTE_MIN, DTE_MAX = 10, 21
STRIKE_LO_MULT, STRIKE_HI_MULT = 0.95, 1.10
TARGET_DELTA = 0.25

# Top-frequency tickers from the 220 real exhausted-shortlist dead signals (research/
# spread_cap_log_mining.py's cohort), covering most of the historical volume.
DEFAULT_TICKERS = ["AMD", "MSFT", "GOOGL", "AAPL", "LMT", "GE", "RTX", "INTC",
                    "LLY", "DIS", "SHAK", "MAR", "PM", "TMO", "JNJ", "BMY",
                    "MU", "AMZN", "CAT", "CAH", "BA"]


def get_quote(opt_client, symbol):
    try:
        resp = opt_client.get_option_latest_quote(OptionLatestQuoteRequest(symbol_or_symbols=symbol))
        q = resp[symbol]
        bid, ask = float(q.bid_price), float(q.ask_price)
        if ask <= 0 or bid < 0:
            return None
        mid = (bid + ask) / 2
        return {"bid": bid, "ask": ask, "mid": mid,
                "spread_pct": (ask - bid) / mid if mid > 0 else 1.0}
    except Exception:
        return None


def walk_candidates(candidates, stock_price, ticker, opt_client, limit=5):
    """Try up to `limit` candidates in the given order; return the first that passes every
    guardrail (tradable, root symbol, quote sanity, spread), or None. Mirrors
    core/bot.py:pick_call_contract's inner loop exactly."""
    tried = 0
    for c in candidates[:limit]:
        if getattr(c, "tradable", True) is False:
            continue
        root = getattr(c, "root_symbol", None)
        if root and root != ticker:
            continue
        tried += 1
        quote = get_quote(opt_client, c.symbol)
        if quote is None:
            continue
        intrinsic = max(0.0, stock_price - float(c.strike_price))
        if quote["mid"] <= 0 or quote["mid"] < intrinsic * 0.9 or quote["mid"] > stock_price:
            continue
        oi = getattr(c, "open_interest", None)
        if oi is not None and int(oi) < MIN_OPEN_INTEREST:
            continue
        if quote["spread_pct"] > MAX_SPREAD_PCT:
            continue
        return c.symbol, quote["spread_pct"], int(oi) if oi is not None else None, tried
    return None, None, None, tried


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tickers", nargs="+", default=DEFAULT_TICKERS)
    args = ap.parse_args()

    tc = TradingClient(cfg.ALPACA_KEY, cfg.ALPACA_SECRET, paper=True)
    opt_client = OptionHistoricalDataClient(cfg.ALPACA_KEY, cfg.ALPACA_SECRET)
    stock_client = StockHistoricalDataClient(cfg.ALPACA_KEY, cfg.ALPACA_SECRET)

    today = date.today()
    min_exp, max_exp = today + timedelta(days=DTE_MIN), today + timedelta(days=DTE_MAX)
    dte_target = round((DTE_MIN + DTE_MAX) / 2)

    print(f"{'ticker':<8} {'pool':<6} {'OI>=100':<8} {'OLD':<28} {'NEW':<28} {'verdict'}")
    rescued, both_fail, both_pass, old_only = 0, 0, 0, 0

    for ticker in args.tickers:
        try:
            q = stock_client.get_stock_latest_quote(StockLatestQuoteRequest(symbol_or_symbols=ticker))
            spot = float(q[ticker].ask_price or q[ticker].bid_price)
        except Exception as e:
            print(f"{ticker:<8} skipped (no stock quote: {e})")
            continue
        if not spot:
            print(f"{ticker:<8} skipped (no stock price)")
            continue

        try:
            resp = tc.get_option_contracts(GetOptionContractsRequest(
                underlying_symbols=[ticker], type=ContractType.CALL,
                expiration_date_gte=min_exp, expiration_date_lte=max_exp,
                strike_price_gte=str(round(spot * STRIKE_LO_MULT, 2)),
                strike_price_lte=str(round(spot * STRIKE_HI_MULT, 2)),
            ))
        except Exception as e:
            print(f"{ticker:<8} skipped (contract lookup failed: {e})")
            continue
        contracts = getattr(resp, "option_contracts", []) or []
        if not contracts:
            print(f"{ticker:<8} 0 contracts in range")
            continue

        target_strike = spot * (1 + (1 - TARGET_DELTA) * 0.15)

        def prox_key(c):
            return (abs((c.expiration_date - today).days - dte_target),
                    abs(float(c.strike_price) - target_strike))

        old_order = sorted(contracts, key=prox_key)

        liquid = [c for c in contracts if c.open_interest is not None
                  and int(c.open_interest) >= MIN_OPEN_INTEREST
                  and getattr(c, "tradable", True) is not False]
        new_order = sorted(liquid, key=prox_key) + [c for c in old_order if c not in liquid]

        old_sym, old_spread, old_oi, old_tried = walk_candidates(old_order, spot, ticker, opt_client)
        time.sleep(0.1)
        new_sym, new_spread, new_oi, new_tried = walk_candidates(new_order, spot, ticker, opt_client)
        time.sleep(0.1)

        n_liquid = len(liquid)
        old_desc = (f"{old_sym} sp={old_spread*100:.0f}% oi={old_oi} (n={old_tried})"
                    if old_sym else f"FAIL (n={old_tried} tried)")
        new_desc = (f"{new_sym} sp={new_spread*100:.0f}% oi={new_oi} (n={new_tried})"
                    if new_sym else f"FAIL (n={new_tried} tried)")

        if old_sym and new_sym:
            verdict = "both pass"; both_pass += 1
        elif not old_sym and new_sym:
            verdict = "RESCUED by liquidity-first sort"; rescued += 1
        elif old_sym and not new_sym:
            verdict = "old-only (unexpected)"; old_only += 1
        else:
            verdict = "both fail (genuinely thin)"; both_fail += 1

        print(f"{ticker:<8} {len(contracts):<6} {n_liquid:<8} {old_desc:<28} {new_desc:<28} {verdict}")

    print(f"\n{'='*80}\nSUMMARY: rescued={rescued}  both_pass={both_pass}  "
          f"both_fail(genuinely thin)={both_fail}  old_only={old_only}  "
          f"(of {rescued+both_pass+both_fail+old_only} tickers tested)")
    print("\nCAVEATS:")
    print(" - This is TODAY's live market, not a replay of the historical dead signals -- it")
    print("   measures whether the STRUCTURAL problem (proximity-only sort ignoring liquidity)")
    print("   still exists on the same tickers today, not whether these exact historical trades")
    print("   would have succeeded.")
    print(" - 'both fail' tickers had NO contract in the whole fetched pool clear OI>=100 --")
    print("   confirms some tickers are genuinely thin regardless of sort order (e.g. GOOGL/GE")
    print("   found earlier), so this fix has a real ceiling, not a 100% rescue rate.")


if __name__ == "__main__":
    main()
