"""
lotto_mispricing_test.py — jeff's contract-selection idea (2026-07-18): rather than always taking
the contract nearest our target delta/DTE, look at NEARBY contracts (same expiry, adjacent
strikes) and see if the market has left one relatively underpriced right after a news catalyst,
before the whole chain's IV has caught up.

DATA AVAILABILITY (confirmed live): Alpaca has no historical option BID/ASK quote endpoint --
only get_option_chain (current/live snapshot only) and get_option_bars/get_option_trades (real
historical OHLCV, any date range). get_option_bars returns real hourly bars WITH VOLUME for
neighboring strikes at a real past expiry (AMZN/APLD/RIVN all checked -- 6-7 bars each around
real historical lotto entries). So this uses REAL historical trade-derived prices for entry AND
exit, not another synthetic BS-off-stock-bars reconstruction -- with the tradeoff that OCC symbols
for expired contracts have to be reconstructed heuristically (nearest Friday expiry + a guessed
strike increment); any that don't return real data get dropped rather than guessed at.

METHODOLOGY, v2 (v1 was wrong -- see git history / project_news_source_quality-adjacent memory):
v1 compared each nearby contract's real price to a "fair value" from ONE shared IV across the
whole strike range. That doesn't measure mispricing -- it measures the ordinary VOLATILITY SMILE
(further-OTM strikes structurally trade at higher effective IV than a flat-vol model assumes,
every day, with or without news). Richness climbed monotonically with distance from the money in
every single case checked -- a smile artifact, not a finding.

Fix: back out each contract's OWN implied vol from its real price (pricing.implied_vol_call,
already existed for calibration), fit a local quadratic curve of IV vs. strike across the neighbor
set AT THAT MOMENT, and flag a contract as cheap only if its actual IV sits BELOW the fitted
curve -- a residual from the expected smile shape, which is what "the market hasn't caught up
yet" would actually look like, rather than deviation from an unrealistic flat baseline.

RESULT (2026-07-18, 270-day window, +/-3 neighbor strikes, $100 budget, live gate mag>=0.70/
conf>=0.85): funnel is 189 gated signals -> 47 priced (real bars found) -> 41 smile-fit attempted
-> 15 usable after the affordability filter (both the target AND the switched pick must fit the
budget for a fair head-to-head). Directionally consistent: switching to the lowest-smile-residual
contract beat always-taking-the-target on win rate (27%->33%), avg $/trade (-$9->-$3), and total $
(-$140->-$43) -- switched in 12 of 15 cases. avg residual gap ~0.044 IV points in aggregate, but
spot-checking individual cases shows most target-vs-cheapest gaps are actually tiny (0.001-0.02,
i.e. 0.1-2 vol points) -- comparable in size to what quadratic-fit noise off 5-7 single-hourly-
close prices could plausibly produce on its own. HONEST READ: this is a promising direction, not
a validated edge -- n=15 is too small and the per-case residuals too close to the noise floor to
be confident this is real exploitable mispricing rather than fit noise that happened to tilt one
way in this sample. Don't wire this live off this evidence alone. Next step if pursued further:
either a much bigger sample (gather over time via a shadow -- log what the smile-cheapest pick
would have been alongside every real trade, no execution change, let real forward data
accumulate) or a less noisy per-contract price estimate (average multiple intraday prints instead
of one hourly close) before trusting the residual signal enough to act on it.

Usage:  USE_YAHOO_BARS=1 MISPRICE_DAYS=270 .venv/bin/python -u lotto_mispricing_test.py   (run from data/)
"""
import os, sys, statistics
os.environ.setdefault("USE_YAHOO_BARS", "1")
from datetime import timedelta

import numpy as np

sys.path.insert(0, "../core")
from pricing import implied_vol_call
from backtest import get_price_at, is_valid_stock_ticker
from regime_filter import build_regime
import news_call_sweep_unified as nc

from alpaca.data.historical.option import OptionHistoricalDataClient
from alpaca.data.requests import OptionBarsRequest
from alpaca.data.timeframe import TimeFrame
from dotenv import load_dotenv
load_dotenv("../.env")

ALPACA_KEY = os.environ["ALPACA_API_KEY"]
ALPACA_SECRET = os.environ["ALPACA_SECRET_KEY"]
opt_client = OptionHistoricalDataClient(ALPACA_KEY, ALPACA_SECRET)

DELTA, DTE = 0.20, 14      # same geometry as every other lotto exit-rule test this session
R = 0.04
STOP_LOSS_PCT = 0.20
HARD_CAP_MULT = 3.0
LIVE_MAG, LIVE_CONF = 0.70, 0.85   # matches v2/config.py's LOTTO_MIN_MAGNITUDE/CONFIDENCE
BUDGET = 100
N_NEIGHBORS_EACH_SIDE = 3
MIN_POINTS_FOR_SMILE = 4   # need > (quadratic's 3 params) or the fit has zero residual everywhere


def nearest_friday(dt, target_dte):
    target = dt + timedelta(days=target_dte)
    offset = (4 - target.weekday()) % 7   # Friday = weekday 4
    return (target + timedelta(days=offset)).date()


def strike_increment(stock_price):
    if stock_price < 25:
        return 1.0
    if stock_price < 100:
        return 2.5
    if stock_price < 500:
        return 5.0
    return 10.0


def occ_symbol(ticker, expiry_date, strike):
    strike_int = round(strike * 1000)
    return f"{ticker}{expiry_date.strftime('%y%m%d')}C{strike_int:08d}"


def fetch_entry_day_bars(symbol, entry_dt):
    try:
        resp = opt_client.get_option_bars(OptionBarsRequest(
            symbol_or_symbols=symbol, timeframe=TimeFrame.Hour,
            start=entry_dt - timedelta(hours=2), end=entry_dt + timedelta(hours=20)))
        bars = (resp.data or {}).get(symbol, [])
        return [{"t": b.timestamp, "close": float(b.close)} for b in bars
                if b.timestamp >= entry_dt - timedelta(minutes=90)]
    except Exception:
        return []


def exit_same_day_stop(bars, entry_price):
    """Same rule as v2's live monitor: hard 3x cap, then 20% stop anchored to entry, else EOD."""
    entry_day = bars[0]["t"].date()
    stop = entry_price * (1 - STOP_LOSS_PCT)
    last = bars[0]["close"]
    for b in bars:
        if b["t"].date() != entry_day:
            break
        if b["close"] >= HARD_CAP_MULT * entry_price:
            return b["close"], "lotto_cap"
        if b["close"] <= stop:
            return b["close"], "lotto_stop_loss"
        last = b["close"]
    return last, "lotto_eod"


def build_candidate(ticker, expiry_date, strike, entry_dt, stock_price):
    symbol = occ_symbol(ticker, expiry_date, strike)
    bars = fetch_entry_day_bars(symbol, entry_dt)
    if len(bars) < 2:
        return None
    entry_price = bars[0]["close"]
    if entry_price < 0.05:
        return None
    dte_days = (expiry_date - entry_dt.date()).days
    T = max(dte_days, 1) / 365.0
    iv = implied_vol_call(entry_price, stock_price, strike, T, R)
    if iv is None or iv <= 0.02 or iv >= 4.9:   # solver-boundary / degenerate values, not real
        return None
    return {"symbol": symbol, "strike": strike, "entry_price": entry_price,
            "iv": iv, "bars": bars}


def fit_smile_residuals(candidates):
    """Quadratic fit of IV vs strike across this signal's neighbor set; returns each candidate
    with a 'residual' = actual_iv - fitted_iv added (negative = cheap relative to the LOCAL
    smile shape, not a flat baseline)."""
    strikes = np.array([c["strike"] for c in candidates], dtype=float)
    ivs = np.array([c["iv"] for c in candidates], dtype=float)
    coeffs = np.polyfit(strikes, ivs, deg=2)
    fitted = np.polyval(coeffs, strikes)
    for c, f in zip(candidates, fitted):
        c["fitted_iv"] = float(f)
        c["residual"] = c["iv"] - float(f)
    return candidates


def main():
    label, end_dt, _, cache = nc.BULL
    days = int(os.environ.get("MISPRICE_DAYS", "180"))
    rows = nc.load_scored_from_unified(cache, end_dt, days, "unified_v1")
    reg = build_regime(end_dt, days, 200)
    gated = [r for r in rows if r["magnitude"] >= LIVE_MAG and r["confidence"] >= LIVE_CONF]
    gated = [r for r in gated if reg is None or reg(r["created_at"].date())]
    print(f"{label}: live gate mag>={LIVE_MAG} conf>={LIVE_CONF} -> {len(gated)} signals (post-regime)\n")

    n_attempted = 0
    n_priced = 0     # >=2 real candidates (old bar-availability check)
    n_smile_fit = 0  # >= MIN_POINTS_FOR_SMILE valid IVs -- smile fit actually attempted
    baseline_pnls, switched_pnls = [], []
    switch_count = 0
    residual_gaps = []
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
            n_attempted += 1

            expiry = nearest_friday(r["created_at"], DTE)
            inc = strike_increment(sp)
            target_strike = round(sp * (1 + (1 - DELTA) * 0.15) / inc) * inc

            strikes = [target_strike + i * inc for i in range(-N_NEIGHBORS_EACH_SIDE, N_NEIGHBORS_EACH_SIDE + 1)]
            candidates = []
            for strike in strikes:
                if strike <= 0:
                    continue
                c = build_candidate(tk, expiry, strike, r["created_at"], sp)
                if c:
                    candidates.append(c)

            target_candidates = [c for c in candidates if abs(c["strike"] - target_strike) < 0.01]
            if not target_candidates or len(candidates) < 2:
                continue
            n_priced += 1
            if len(candidates) < MIN_POINTS_FOR_SMILE:
                continue   # not enough points for a meaningful quadratic residual
            n_smile_fit += 1

            candidates = fit_smile_residuals(candidates)
            baseline = next(c for c in candidates if abs(c["strike"] - target_strike) < 0.01)
            cheapest = min(candidates, key=lambda c: c["residual"])

            qty_base = max(1, int(BUDGET / (baseline["entry_price"] * 100))) if baseline["entry_price"] * 100 <= BUDGET * 1.15 else None
            qty_cheap = max(1, int(BUDGET / (cheapest["entry_price"] * 100))) if cheapest["entry_price"] * 100 <= BUDGET * 1.15 else None
            if not qty_base or not qty_cheap:
                continue

            base_exit, _ = exit_same_day_stop(baseline["bars"], baseline["entry_price"])
            cheap_exit, _ = exit_same_day_stop(cheapest["bars"], cheapest["entry_price"])

            base_pnl = (base_exit - baseline["entry_price"]) * 100 * qty_base
            cheap_pnl = (cheap_exit - cheapest["entry_price"]) * 100 * qty_cheap

            baseline_pnls.append(base_pnl)
            switched_pnls.append(cheap_pnl)
            residual_gaps.append(baseline["residual"] - cheapest["residual"])
            if cheapest["symbol"] != baseline["symbol"]:
                switch_count += 1

    print(f"signals attempted (had entry stock price): {n_attempted}")
    print(f"priced (real bars for target + >=1 real neighbor): {n_priced}")
    print(f"smile-fit attempted (>= {MIN_POINTS_FOR_SMILE} real IVs in the neighbor set): {n_smile_fit}")
    print(f"usable for P&L comparison (also affordable at ${BUDGET} budget): {len(baseline_pnls)}")
    print(f"strategy would have switched off the target contract: {switch_count}/{len(baseline_pnls) or 1}\n")

    if not baseline_pnls:
        print("No usable candidates -- can't compare. See coverage note above.")
        return

    def stat(name, pnls):
        n = len(pnls); tot = sum(pnls); avg = tot / n
        sd = statistics.pstdev(pnls) if n > 1 else 0
        sh = avg / sd if sd else 0
        win = sum(1 for p in pnls if p > 0) / n * 100
        print(f"  {name:34} n={n:<3} Σ${tot:>+8,.0f}  avg${avg:>+6,.0f}  win{win:>4.0f}%  sharpe{sh:>+5.2f}")

    stat("baseline (always target contract)", baseline_pnls)
    stat("switched (lowest smile residual)", switched_pnls)
    print(f"\navg residual gap (baseline - cheapest, in IV points): {statistics.mean(residual_gaps):+.4f}"
          f"  (positive = the alternative really was below the local smile, not just far OTM)")


if __name__ == "__main__":
    main()
