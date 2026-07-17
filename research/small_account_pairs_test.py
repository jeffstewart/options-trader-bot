"""
small_account_pairs_test.py — does pairs (market-neutral L/S) survive being sized down from the
current $83k paper account to a real $500-1000 account? Three questions pairs_backtest.py doesn't
answer at any size: (1) does Sharpe/win hold up as leg size shrinks, (2) how many pairs positions
accumulate concurrently (pairs enters ~1x/day, live has NO time cap — trailing stop is the only
exit, so a wide 25%/15% trail can let positions stack for weeks; is a 3-5 position budget enough?),
(3) can a small leg budget even AFFORD 1 share of the tickers the model actually picks (GS $1121,
LMT $523 appeared in the live book — a $50-150 leg can't buy 1 share of either).

Usage:  USE_YAHOO_BARS=1 .venv/bin/python -u small_account_pairs_test.py   (run from data/)
"""
import os, statistics
os.environ.setdefault("USE_YAHOO_BARS", "1")
from datetime import datetime, timezone, timedelta

import config as _cfg
import pairs_backtest as pb
from benchmark import compute_stats
from backtest import get_price_at

WINDOW = pb.WINDOWS[0]   # bull-meltup, 180d — same tape pairs was originally validated on

# Candidate small-account leg budgets. pairs_backtest sizes each leg via
# _scale(mag,conf) = max(MAX_POSITION_USD*0.10, MAX_POSITION_USD*mag*conf), so MAX_POSITION_USD is
# the dial; a mag=0.85,conf=0.82 signal (typical) scales to ~0.70x MAX_POSITION_USD per leg.
ACCOUNT_SIZES = [500, 750, 1000]
LEG_FRACTION_OF_ACCOUNT = 0.15   # cap: no single leg > 15% of equity (conservative for options-adjacent risk)


def main():
    label, end_dt, days, bull_cache, bear_cache = WINDOW
    bull_rows, bear_rows = pb.load_both_sides(bull_cache, end_dt, days)
    bull_daily = pb.build_daily_signals(bull_rows, "bull")
    bear_daily = pb.build_daily_signals(bear_rows, "bear")
    common_days = sorted(set(bull_daily) & set(bear_daily))
    print(f"{label}: {len(common_days)} days with both a bull and bear signal (max 1 pair/day)\n")

    print("── 1) SHARPE/WIN AT SCALED-DOWN SIZE ──")
    print(f"  {'MAX_POSITION_USD':>18} {'~leg $':>8} {'n':>4} {'Sharpe':>7} {'win%':>6} {'total$':>9} {'maxDD$':>8}")
    for equity in ACCOUNT_SIZES:
        max_pos = equity * LEG_FRACTION_OF_ACCOUNT / 0.70   # back out MAX_POSITION_USD so a TYPICAL
        # (mag .85 conf .82 -> scale .70) signal lands near the leg-fraction cap, not the floor
        save = _cfg.MAX_POSITION_USD
        _cfg.MAX_POSITION_USD = max_pos
        pb._scale.__globals__["_cfg"].MAX_POSITION_USD = max_pos   # _scale reads _cfg at call time
        bd = pb.build_daily_signals(bull_rows, "bull")
        be = pb.build_daily_signals(bear_rows, "bear")
        trades = pb.run_pairs(bd, be)
        _cfg.MAX_POSITION_USD = save
        s = compute_stats(trades)
        avg_leg = statistics.mean(t["pnl_usd"] / (t["pnl_pct"] / 100) for t in trades if t["pnl_pct"]) if trades else 0
        print(f"  ${equity:<5} account   ${avg_leg:>6,.0f}  {s['trades']:>4} {s['sharpe']:>7.2f} "
              f"{s['win_rate']:>5.1f}% ${s['total_pnl']:>8,.0f} ${s['max_dd']:>7,.0f}")

    print(f"\n  reference — current live sizing (MAX_POSITION_USD=${_cfg.MAX_POSITION_USD:,.0f}):")
    trades_ref = pb.run_pairs(bull_daily, bear_daily)
    s = compute_stats(trades_ref)
    print(f"  {'(live)':>18} {'-':>8} {s['trades']:>4} {s['sharpe']:>7.2f} {s['win_rate']:>5.1f}% "
          f"${s['total_pnl']:>8,.0f} ${s['max_dd']:>7,.0f}")

    print("\n── 2) HOLD-DURATION → concurrent-position estimate (no live time cap, trail-stop only exit) ──")
    # simulate_pead/sim_short don't expose exit date, so recompute the same trail logic here,
    # tracking the exit bar's timestamp directly against get_stock_bars (same source both use).
    from backtest import get_stock_bars

    def hold_days(ticker, entry_dt, trail, is_short):
        bars = get_stock_bars(ticker, entry_dt, entry_dt + timedelta(days=int(pb.MAX_HOLD * 1.6) + 14))
        tb = [b for b in bars if entry_dt < b["t"]]
        if not tb:
            return None
        extreme = get_price_at(ticker, entry_dt) or tb[0]["c"]
        for b in tb[:pb.MAX_HOLD]:
            if is_short:
                extreme = min(extreme, b["c"])
                if b["c"] >= extreme * (1 + trail):
                    return (b["t"] - entry_dt).days
            else:
                extreme = max(extreme, b["c"])
                if b["c"] <= extreme * (1 - trail):
                    return (b["t"] - entry_dt).days
        return (tb[min(pb.MAX_HOLD, len(tb)) - 1]["t"] - entry_dt).days if tb else None

    holds = []
    for d in common_days:
        bull_sig, bear_sig = bull_daily[d], bear_daily[d]
        if bull_sig["ticker"] == bear_sig["ticker"]:
            continue
        for sig, trail, is_short in [(bull_sig, pb.LONG_TRAIL, False), (bear_sig, pb.SHORT_TRAIL, True)]:
            try:
                h = hold_days(sig["ticker"], sig["created_at"], trail, is_short)
                if h is not None and h >= 0:
                    holds.append(h)
            except Exception:
                pass
    if holds:
        holds.sort()
        print(f"  n={len(holds)} leg-exits  median hold {statistics.median(holds)}d  "
              f"p75 {holds[int(len(holds)*0.75)]}d  p90 {holds[int(len(holds)*0.90)]}d  "
              f"max {holds[-1]}d  (backtest caps at {pb.MAX_HOLD}d; LIVE HAS NO CAP)")
        # rough concurrent estimate: entries/day (~1 pair = 2 legs when both sides fire) x median hold
        entry_rate = len(common_days) / days
        print(f"  entry rate: {len(common_days)}/{days} days ({entry_rate:.2f} pairs/day) x "
              f"median hold {statistics.median(holds)}d ≈ {entry_rate*statistics.median(holds)*2:.1f} "
              f"concurrent legs in steady state (Little's-law approximation)")

    print("\n── 3) SHARE AFFORDABILITY at small leg budgets (can we even buy 1 share?) ──")
    for equity in ACCOUNT_SIZES:
        leg_budget = equity * LEG_FRACTION_OF_ACCOUNT
        unaffordable = 0
        checked = 0
        for d in common_days[:60]:   # sample for speed
            for sig, side in [(bull_daily[d], "bull"), (bear_daily[d], "bear")]:
                px = get_price_at(sig["ticker"], sig["created_at"])
                if px:
                    checked += 1
                    if px > leg_budget:
                        unaffordable += 1
        pct = unaffordable / checked * 100 if checked else 0
        print(f"  ${equity} account (leg budget ${leg_budget:.0f}): "
              f"{unaffordable}/{checked} picks priced above budget for even 1 share ({pct:.0f}%)")


if __name__ == "__main__":
    main()
