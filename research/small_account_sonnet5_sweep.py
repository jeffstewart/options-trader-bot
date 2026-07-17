"""
small_account_sonnet5_sweep.py — does the small-account selectivity failure (stock AND options both
flat/tail-driven no matter the mag/conf threshold, small_account_news_call_sweep.py +
stock_selectivity_sweep.py) come from an uncalibrated SIGNAL rather than the wrong THRESHOLD? sonnet5
already beat Ollama on matched-selectivity pick quality (+4.07%/trade vs +3.09%, win 71% vs 68%) --
this reruns the SAME small-account tests (stock selectivity grid, lotto selectivity + affordability)
on sonnet5's scores instead of Ollama's, same simulators (stock_backtest.simulate_stock,
backtest.simulate_option_pnl), same bull-meltup tape, for a direct comparison.

Also tests jeff's stated expectation that LOTTO (small $250 base, deep-OTM) is the natural fit for a
$500-1000 account -- flagging up front that lotto is DESIGNED as a low-win-rate convex bet (see
lotto_backtest.py's own doc: "Low win% is expected; it only works if a few big winners pay for many
small losers"), which is philosophically the OPPOSITE of "fewer trades that actually win" -- small
size ≠ small variance. Worth testing whether sonnet5 changes that shape, not just assuming it will.

Usage:  USE_YAHOO_BARS=1 .venv/bin/python -u small_account_sonnet5_sweep.py   (run from data/)
"""
import json, os, statistics
os.environ.setdefault("USE_YAHOO_BARS", "1")
from datetime import datetime, timezone, timedelta

import backtest as _bt
import config as _cfg
from benchmark import compute_stats
from regime_filter import build_regime
from stock_backtest import simulate_stock
import stock_backtest as _sb

ANTHROPIC_CACHE = "anthropic_backtest_cache.json"
DUAL_CACHE      = "dual_score_cache.json"
END_DT = datetime.now(timezone.utc) - timedelta(days=5)
DAYS   = 180

MAG_FLOORS  = [0.20, 0.30, 0.40, 0.50, 0.60]   # sonnet5 runs LOWER on its own magnitude scale
CONF_FLOORS = [0.60, 0.70, 0.80]


def load_sonnet5_rows():
    """sonnet5-scored bullish rows, same shape tune_v2.load_scored_from_dual_cache returns
    (created_at, magnitude, confidence, tickers) -- built from the article metadata in
    dual_score_cache.json (created_at/headline) joined to sonnet5's own score/ticker output."""
    anth = json.load(open(ANTHROPIC_CACHE))
    dual = json.load(open(DUAL_CACHE))
    start_dt = END_DT - timedelta(days=DAYS)
    rows = []
    for k, v in anth.items():
        if not k.startswith("sonnet5:") or not v or v.get("sent") != "bullish":
            continue
        ck = k.split(":", 1)[1]
        dv = dual.get(ck)
        if not isinstance(dv, dict):
            continue
        a = dv.get("_article", {}) or {}
        ca = a.get("created_at")
        if not ca:
            continue
        try:
            created_at = datetime.fromisoformat(str(ca).replace("Z", "+00:00"))
            if created_at.tzinfo is None:
                created_at = created_at.replace(tzinfo=timezone.utc)
        except Exception:
            continue
        if not (start_dt <= created_at <= END_DT):
            continue
        tickers = [t for t in (v.get("tick") or []) if isinstance(t, str) and t not in ("BTC", "ETH")]
        if not tickers:
            continue
        rows.append({"created_at": created_at, "magnitude": float(v.get("mag", 0) or 0),
                     "confidence": float(v.get("conf", 0) or 0), "tickers": tickers})
    rows.sort(key=lambda r: r["created_at"])
    return rows


def stock_sweep(rows, reg):
    print("── STOCK selectivity, sonnet5 scores (vs Ollama best cell: mag≥0.75 Sharpe -0.29, win 40.1%, "
          "NO cell beat -0.53) ──")
    print(f"  {'gate':>16} {'n':>4} {'total':>9} {'mean$/tr':>9} {'sharpe':>7} {'win%':>6}")
    save = (_sb.STOCK_TRAIL, _sb.MAX_HOLD)
    for mf in MAG_FLOORS:
        for cf in CONF_FLOORS:
            gated = [r for r in rows if r["magnitude"] >= mf and r["confidence"] >= cf]
            trades, seen = [], set()
            for r in gated:
                d = r["created_at"].date()
                if reg and not reg(d):
                    continue
                for tk in r["tickers"][:1]:
                    if not _bt.is_valid_stock_ticker(tk):
                        continue
                    key = f"{d}_{tk}"
                    if key in seen:
                        continue
                    seen.add(key)
                    t = simulate_stock(tk, r["created_at"], 250)
                    if t:
                        trades.append(t)
            if len(trades) < 10:
                print(f"  mag≥{mf:.2f} c≥{cf:.2f}  {len(trades):>4}  (insufficient sample)")
                continue
            s = compute_stats(trades)
            print(f"  mag≥{mf:.2f} c≥{cf:.2f}  {s['trades']:>4} ${s['total_pnl']:>+8,.0f} "
                  f"${s['total_pnl']/s['trades']:>+8,.0f} {s['sharpe']:>+6.2f} {s['win_rate']:>5.1f}%")


def lotto_sweep(rows, reg):
    print("\n── LOTTO selectivity + small-budget affordability, sonnet5 scores ──")
    print("  (lotto is DESIGNED low-win-rate/convex -- judge on win% AND tail concentration, not just Sharpe)")
    tiered = {"tiers": [(1.0, 0.40), (3.0, 0.30), (float("inf"), 0.20)]}
    save = {k: getattr(_bt, k) for k in
            ("TARGET_DELTA", "DTE_TARGET", "MAX_HOLD_DAYS", "TRAILING_STOP_PCT", "EXIT_PARAMS")}
    save_mult = _cfg.MAX_CONTRACT_BUDGET_MULT
    try:
        _bt.DTE_TARGET, _bt.MAX_HOLD_DAYS, _bt.TRAILING_STOP_PCT, _bt.EXIT_PARAMS = 14, 7, 0.30, tiered
        for mf, cf in [(0.30, 0.70), (0.40, 0.70), (0.40, 0.80), (0.50, 0.80)]:
            gated = [r for r in rows if r["magnitude"] >= mf and r["confidence"] >= cf]
            print(f"\n  gate mag>={mf} conf>={cf}: {len(gated)} signals")
            for budget in (100, 150, 250):
                _cfg.MAX_CONTRACT_BUDGET_MULT = 1.15
                _bt.TARGET_DELTA = 0.20   # deep-OTM lotto geometry
                trades, seen = [], set()
                for r in gated:
                    d = r["created_at"].date()
                    if reg and not reg(d):
                        continue
                    for tk in r["tickers"][:1]:
                        if not _bt.is_valid_stock_ticker(tk):
                            continue
                        key = f"{d}_{tk}"
                        if key in seen:
                            continue
                        seen.add(key)
                        sp = _bt.get_price_at(tk, r["created_at"])
                        if not sp:
                            continue
                        t = _bt.simulate_option_pnl(tk, r["created_at"], sp, budget,
                                                    {"magnitude": r["magnitude"], "confidence": r["confidence"]},
                                                    option_type="call", exit_rule="tiered_profit",
                                                    spread_mult=1.0)
                        if t:
                            trades.append(t)
                universe_n = sum(1 for r in gated if reg is None or reg(r["created_at"].date()))
                if len(trades) < 5:
                    print(f"    ${budget:>4} leg  n={len(trades):<3} (insufficient sample, "
                          f"skip {(1-len(trades)/max(universe_n,1))*100:.0f}%)")
                    continue
                trades.sort(key=lambda t: -t["pnl_usd"])
                tot = sum(t["pnl_usd"] for t in trades)
                top3 = sum(t["pnl_usd"] for t in trades[:3])
                s = compute_stats(trades)
                skip = (1 - len(trades) / max(universe_n, 1)) * 100
                top3_pct = (top3 / tot * 100) if tot else 0
                print(f"    ${budget:>4} leg  n={s['trades']:<3} ${s['total_pnl']:>+7,.0f} "
                      f"win={s['win_rate']:>4.1f}%  sharpe={s['sharpe']:>+5.2f}  "
                      f"top3={top3_pct:>4.0f}%ofP&L  skip={skip:>3.0f}%")
    finally:
        for k, v in save.items():
            setattr(_bt, k, v)
        _cfg.MAX_CONTRACT_BUDGET_MULT = save_mult


def main():
    rows = load_sonnet5_rows()
    print(f"sonnet5 bullish rows: {len(rows)}\n")
    reg = build_regime(END_DT, DAYS, 200)
    stock_sweep(rows, reg)
    lotto_sweep(rows, reg)


if __name__ == "__main__":
    main()
