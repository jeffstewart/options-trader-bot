"""
sell_premium_tune.py — Grid search over credit-spread params with 80/20 holdout.

The first pass (sell_premium_backtest.py) flopped: −$34K all-names, −$990 liquid,
because the calibrated NEWS_IV_MULTIPLIER=1.10 gives little juice to harvest and
the spread's negative skew requires >65% win rate to profit.

This tune tests whether any parameter set survives costs, with strict holdout
validation.  Reports both ALL-names (full universe) and LIQUID-names (the
tradeable subset) on train and holdout.

Runs sequentially after pead_backtest.py to avoid concurrent Yahoo-cache writes.

Usage:  USE_YAHOO_BARS=1 .venv/bin/python sell_premium_tune.py
"""
import itertools
import sys
from datetime import datetime, timezone, timedelta
from pathlib import Path

import backtest as _bt
import tune_v2
from backtest import (get_price_at, get_stock_bars, is_valid_stock_ticker,
                      base_iv_for, _spread_pct, LIQUID_UNDERLYINGS,
                      NEWS_IV_MULTIPLIER, IV_CRUSH_HALFLIFE_DAYS, RISK_FREE_RATE)
from pricing import bs_put_price, put_strike_for_delta, iv_crush_path
from benchmark import compute_stats
from regime_filter import build_regime
import config as _cfg

# Only tune on the bull window — it has enough trades for a meaningful split.
# 2022 bear is run as a cross-regime stress-test on the best bull combo.
BULL_END  = datetime.now(timezone.utc) - timedelta(days=5)
BULL_DAYS = 180
BULL_CACHE = Path("dual_score_cache.json")
BEAR_END  = datetime(2022, 6, 30, tzinfo=timezone.utc)
BEAR_DAYS = 90
BEAR_CACHE = Path("bear_dual_cache.json")

# ── Parameterised simulator ───────────────────────────────────────────────────

def simulate_credit_spread(ticker, entry_dt, position_usd,
                            dte, short_delta, width_pct,
                            profit_take, stop_mult):
    r  = RISK_FREE_RATE
    s0 = get_price_at(ticker, entry_dt)
    if not s0 or s0 < _cfg.MIN_STOCK_PRICE or s0 > 10000:
        return None
    base_iv  = base_iv_for(ticker, entry_dt)
    entry_iv = base_iv * NEWS_IV_MULTIPLIER
    T0       = dte / 365.0
    Ks = round(put_strike_for_delta(s0, T0, entry_iv, short_delta, r), 2)
    Kl = round(Ks * (1 - width_pct), 2)
    W  = Ks - Kl
    if W <= 0:
        return None
    short_mid = bs_put_price(s0, Ks, T0, entry_iv, r)
    long_mid  = bs_put_price(s0, Kl, T0, entry_iv, r)
    credit_mid = short_mid - long_mid
    if credit_mid <= 0.02:
        return None
    hs_s = _spread_pct(short_mid, ticker, 1.0) / 2
    hs_l = _spread_pct(long_mid,  ticker, 1.0) / 2
    credit_in = short_mid * (1 - hs_s) - long_mid * (1 + hs_l)
    if credit_in <= 0:
        return None
    max_loss_contract = (W - credit_mid) * 100
    if max_loss_contract <= 0:
        return None
    qty = max(1, int(position_usd / max_loss_contract))
    max_hold = max(3, dte - 3)

    bars = get_stock_bars(ticker, entry_dt,
                          entry_dt + timedelta(days=int(max_hold * 1.6) + 7))
    tb = [b for b in bars if entry_dt < b["t"]]
    if not tb or (tb[0]["t"] - entry_dt).days > 7:
        return None
    prev = s0
    for b in tb:
        if prev > 0 and abs(b["c"] / prev - 1) > 0.5:
            return None
        prev = b["c"]

    def spread_value(S, days_held, i):
        T  = max((dte - days_held) / 365.0, 1.0 / 365.0)
        iv = iv_crush_path(base_iv, i, NEWS_IV_MULTIPLIER, IV_CRUSH_HALFLIFE_DAYS)
        return bs_put_price(S, Ks, T, iv, r) - bs_put_price(S, Kl, T, iv, r)

    exit_val = None
    for i, b in enumerate(tb[:max_hold], start=1):
        dh = max((b["t"] - entry_dt).days, 0)
        sv = max(0.0, min(spread_value(b["c"], dh, i), W))
        if credit_mid - sv >= credit_mid * profit_take:
            exit_val = sv; break
        if sv >= credit_mid * stop_mult or sv >= W * 0.95:
            exit_val = sv; break
    if exit_val is None:
        last = tb[min(max_hold, len(tb)) - 1]
        dh   = max((last["t"] - entry_dt).days, 0)
        exit_val = max(0.0, min(spread_value(last["c"], dh, min(max_hold, len(tb))), W))

    cost_out = exit_val * (1 + (hs_s + hs_l) / 2)
    pnl = (credit_in - cost_out) * 100 * qty
    pnl_pct = (credit_in - cost_out) / W * 100
    return {"ticker": ticker, "entry_dt": entry_dt.isoformat(),
            "pnl_usd": pnl, "pnl_pct": pnl_pct,
            "is_liquid": ticker in LIQUID_UNDERLYINGS}


def run_sp(scored_rows, regime, params):
    dte         = params["dte"]
    short_delta = params["short_delta"]
    width_pct   = params["width_pct"]
    profit_take = params["profit_take"]
    stop_mult   = params["stop_mult"]
    min_mag     = params.get("min_mag", _cfg.MIN_MAGNITUDE)

    def conf_floor(mag):
        return _cfg.BASE_CONFIDENCE + (1 - mag) * _cfg.CONFIDENCE_SLOPE

    def scale(mag, conf):
        return max(_cfg.MAX_POSITION_USD * 0.10, _cfg.MAX_POSITION_USD * mag * conf)

    trades, seen = [], {}
    for row in scored_rows:
        mag, conf = row["magnitude"], row["confidence"]
        if mag < min_mag or conf < conf_floor(mag):
            continue
        d = row["created_at"].date()
        if regime is not None and not regime(d):
            continue
        ds = seen.setdefault(d.isoformat(), set())
        for tk in row["tickers"][:2]:
            if tk in ds or tk in ("BTC", "ETH") or not is_valid_stock_ticker(tk):
                continue
            ds.add(tk)
            t = simulate_credit_spread(
                tk, row["created_at"], scale(mag, conf),
                dte, short_delta, width_pct, profit_take, stop_mult
            )
            if t:
                trades.append(t)
    return trades


# ── Tune grid ─────────────────────────────────────────────────────────────────

GRID = {
    "dte":         [14, 21, 28],
    "short_delta": [0.20, 0.25, 0.30, 0.35],
    "width_pct":   [0.03, 0.05, 0.08],
    "profit_take": [0.25, 0.40, 0.50, 0.65],
    "stop_mult":   [1.5, 2.0, 3.0],
}


def main():
    if not BULL_CACHE.exists():
        print(f"Cache not found: {BULL_CACHE}"); sys.exit(1)

    # ── Split training / holdout ──────────────────────────────────────────────
    all_scored = tune_v2.load_scored_from_dual_cache(BULL_CACHE, "bull",
                                                     BULL_END, BULL_DAYS)
    n     = len(all_scored)
    split = int(n * 0.80)
    train_scored, hold_scored = all_scored[:split], all_scored[split:]
    train_end = train_scored[-1]["created_at"]
    hold_end  = all_scored[-1]["created_at"]
    train_reg = build_regime(train_end, BULL_DAYS, 200)
    hold_reg  = build_regime(hold_end,  BULL_DAYS, 200)

    print(f"SELL-PREMIUM TUNE  (bull window 80/20 holdout)\n")
    print(f"  Train: {len(train_scored)} articles | Holdout: {len(hold_scored)} articles")

    keys   = list(GRID.keys())
    combos = list(itertools.product(*[GRID[k] for k in keys]))
    print(f"  {len(combos)} combos ...\n", flush=True)

    results = []
    for i, vals in enumerate(combos):
        params = dict(zip(keys, vals))
        if (i + 1) % 100 == 0:
            print(f"  {i+1}/{len(combos)} combos done", flush=True)

        trades    = run_sp(train_scored, train_reg, params)
        liq       = [t for t in trades if t.get("is_liquid")]
        ts_all    = compute_stats(trades)
        ts_liq    = compute_stats(liq)
        results.append({
            **params,
            "train_sharpe_all":  ts_all["sharpe"],
            "train_pnl_all":     ts_all["total_pnl"],
            "train_trades_all":  ts_all["trades"],
            "train_win_all":     ts_all["win_rate"],
            "train_sharpe_liq":  ts_liq["sharpe"],
            "train_pnl_liq":     ts_liq["total_pnl"],
            "train_trades_liq":  ts_liq["trades"],
        })

    # ── Report top 10 by all-names Sharpe ────────────────────────────────────
    results.sort(key=lambda r: r["train_sharpe_all"], reverse=True)
    top10 = results[:10]

    print("═══ TOP 10 by TRAIN Sharpe (all names) ═══")
    hdr = (f"  {'DTE':>3} {'Δ':>4} {'wid':>4} {'tp':>4} {'stop':>4}  "
           f"{'Tr.Sh(all)':>10} {'Tr.P&L(all)':>11} {'n':>4}  "
           f"{'Tr.Sh(liq)':>10} {'Tr.P&L(liq)':>11} {'n':>4}")
    print(hdr)
    for r in top10:
        print(f"  {r['dte']:>3} {r['short_delta']:>4.2f} {r['width_pct']:>4.2f} "
              f"{r['profit_take']:>4.2f} {r['stop_mult']:>4.1f}  "
              f"{r['train_sharpe_all']:>10.2f} ${r['train_pnl_all']:>10,.0f} {r['train_trades_all']:>4}  "
              f"{r['train_sharpe_liq']:>10.2f} ${r['train_pnl_liq']:>10,.0f} {r['train_trades_liq']:>4}")

    # ── Holdout validation (top 10) ───────────────────────────────────────────
    print("\n═══ HOLDOUT validation (top 10 by train Sharpe) ═══")
    print(hdr.replace("Tr.", "Ho."))
    holdout_results = []
    for r in top10:
        params = {k: r[k] for k in keys}
        trades = run_sp(hold_scored, hold_reg, params)
        liq    = [t for t in trades if t.get("is_liquid")]
        hs_all = compute_stats(trades)
        hs_liq = compute_stats(liq)
        holdout_results.append({**r,
            "hold_sharpe_all": hs_all["sharpe"],  "hold_pnl_all": hs_all["total_pnl"],
            "hold_trades_all": hs_all["trades"],   "hold_win_all": hs_all["win_rate"],
            "hold_sharpe_liq": hs_liq["sharpe"],  "hold_pnl_liq": hs_liq["total_pnl"],
            "hold_trades_liq": hs_liq["trades"],
        })
        print(f"  {r['dte']:>3} {r['short_delta']:>4.2f} {r['width_pct']:>4.2f} "
              f"{r['profit_take']:>4.2f} {r['stop_mult']:>4.1f}  "
              f"{hs_all['sharpe']:>10.2f} ${hs_all['total_pnl']:>10,.0f} {hs_all['trades']:>4}  "
              f"{hs_liq['sharpe']:>10.2f} ${hs_liq['total_pnl']:>10,.0f} {hs_liq['trades']:>4}")

    # ── Best holdout combo: cross-regime stress-test ──────────────────────────
    best = max(holdout_results, key=lambda r: r["hold_sharpe_all"])
    print(f"\n  Best holdout combo: DTE={best['dte']} Δ={best['short_delta']} "
          f"width={best['width_pct']} tp={best['profit_take']} stop={best['stop_mult']}")

    if BEAR_CACHE.exists():
        print("\n═══ CROSS-REGIME STRESS (2022 bear) — best holdout combo ═══")
        bear_scored = tune_v2.load_scored_from_dual_cache(BEAR_CACHE, "bull",
                                                          BEAR_END, BEAR_DAYS)
        bear_reg = build_regime(BEAR_END, BEAR_DAYS, 200)
        params   = {k: best[k] for k in keys}
        trades   = run_sp(bear_scored, bear_reg, params)
        liq      = [t for t in trades if t.get("is_liquid")]
        bs_all   = compute_stats(trades)
        bs_liq   = compute_stats(liq)
        print(f"  all names:   Sharpe={bs_all['sharpe']:>5.2f}  "
              f"P&L=${bs_all['total_pnl']:>9,.0f}  win={bs_all['win_rate']:.1f}%  "
              f"n={bs_all['trades']}")
        print(f"  LIQUID only: Sharpe={bs_liq['sharpe']:>5.2f}  "
              f"P&L=${bs_liq['total_pnl']:>9,.0f}  win={bs_liq['win_rate']:.1f}%  "
              f"n={bs_liq['trades']}")
    else:
        print(f"\n  [2022 cache not found — skip bear stress test]")

    # ── Verdict ───────────────────────────────────────────────────────────────
    print("\n═══ VERDICT ═══")
    best_ho_sharpe = best["hold_sharpe_all"]
    best_ho_liq    = best["hold_sharpe_liq"]
    if best_ho_sharpe > 0.5:
        print(f"  ✅ Holdout all-names Sharpe={best_ho_sharpe:.2f} > 0.5 — some edge survives OOS")
    else:
        print(f"  ❌ Best holdout all-names Sharpe={best_ho_sharpe:.2f} ≤ 0.5 — no reliable edge")
    if best_ho_liq > 0.5:
        print(f"  ✅ Holdout LIQUID Sharpe={best_ho_liq:.2f} > 0.5 — tradeable edge possible")
    else:
        print(f"  ❌ Holdout LIQUID Sharpe={best_ho_liq:.2f} ≤ 0.5 — not tradeable")
    note = ("  Root-cause note: NEWS_IV_MULTIPLIER=1.10 (calibrated) gives ~10% IV overpricing.\n"
            "  Credit-spread negative skew requires ~65-70% win rate to profit at 5% width.\n"
            "  If no config passes: the IV bump is the hard constraint, not the params.")
    print(note)


if __name__ == "__main__":
    main()
