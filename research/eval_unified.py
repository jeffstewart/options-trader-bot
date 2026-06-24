"""
eval_unified.py — does the UNIFIED prompt's single pass replace the materiality 2nd pass?

Reads unified_scores.json (full dicts) + reconstructs each article's date, then tests the unified
prompt AS THE BOT WOULD USE IT:
  • LOTTO: unified's OWN high-conviction set (mag≥0.70 & conf≥0.85, bullish, has ticker), gated by
    its `catalyst` field — sweep the catalyst threshold θ, simulate lotto (Δ0.25/DTE14/hold3/3×cap).
    Success = matches/beats the live TWO-PASS (main mag/conf + separate v2 materiality = +$15,686).
  • STOCK: unified `confidence` selectivity → per-trade quality vs the conf≥0.85 cliff we found.
Runs on whatever is cached so far → directional read while scoring, definitive when complete.

Usage:  USE_YAHOO_BARS=1 .venv/bin/python eval_unified.py
"""
import os, json, random, statistics
os.environ.setdefault("USE_YAHOO_BARS", "1")
from datetime import datetime, timezone, timedelta
from pathlib import Path

import backtest as _bt
import config as cfg
from benchmark import compute_stats
from backtest import is_valid_stock_ticker
from stock_backtest import simulate_stock
import score_unified as su

INF = float("inf")
DELTA, DTE, HOLD, POS = 0.25, 14, 3, cfg.LOTTO_POSITION_USD
EXIT = {"tiers": [(1.0, 0.40), (3.0, 0.30), (INF, 0.20)], "hard_target": 3.0}
HC_MAG, HC_CONF = cfg.LOTTO_MIN_MAGNITUDE, cfg.LOTTO_MIN_CONFIDENCE
PROMPT = os.environ.get("UNI_PROMPT", "unified_v1")
TWO_PASS_REF = 15686            # live two-pass v2 materiality lotto P&L (threshold_gate_test)
CAT_THRESHOLDS = [0.0, 0.10, 0.15, 0.20, 0.30, 0.50]
CONF_FLOORS = [0.70, 0.80, 0.85, 0.90]
N_BOOT = 10000
random.seed(20260612)


def load_joined():
    """Join unified scores with each article's date. Returns list of dicts with u + dt + tk."""
    sc = json.loads(Path("unified_scores.json").read_text())
    rows = []
    for c in su.candidates():
        u = sc.get(f"{PROMPT}:{c['ck']}")
        if not u:
            continue
        tks = [t for t in (u.get("tickers") or []) if t not in ("BTC", "ETH")]
        rows.append({"dt": c["dt"], "u": u, "tk": tks[0] if tks else None})
    return rows


def lotto_sim(passers):
    save = {k: getattr(_bt, k) for k in ("TARGET_DELTA", "DTE_TARGET", "MAX_HOLD_DAYS", "TRAILING_STOP_PCT", "EXIT_PARAMS")}
    _bt.TARGET_DELTA, _bt.DTE_TARGET, _bt.MAX_HOLD_DAYS = DELTA, DTE, HOLD
    _bt.TRAILING_STOP_PCT, _bt.EXIT_PARAMS = 0.30, EXIT
    trades = []
    try:
        for tk, dt in passers:
            if not is_valid_stock_ticker(tk):
                continue
            sp = _bt.get_price_at(tk, dt)
            if not sp:
                continue
            t = _bt.simulate_option_pnl(tk, dt, sp, POS, {"magnitude": 0.8, "confidence": 0.9},
                                        option_type="call", exit_rule="tiered_profit", spread_mult=1.0)
            if t:
                trades.append(t)
    finally:
        for k, v in save.items():
            setattr(_bt, k, v)
    return trades


def pctl(xs, p):
    xs = sorted(xs); i = max(0, min(len(xs) - 1, int(round(p / 100 * (len(xs) - 1)))))
    return xs[i]


def main():
    rows = load_joined()
    print(f"═══ EVAL unified [{PROMPT}] — {len(rows)} scored articles cached ═══\n")

    # ── LOTTO: unified's own high-conviction set, gated by catalyst ──
    hc = [(r["tk"], r["dt"], float(r["u"].get("catalyst", 0)))
          for r in rows if r["tk"] and r["u"].get("sentiment") == "bullish"
          and r["u"].get("magnitude", 0) >= HC_MAG and r["u"].get("confidence", 0) >= HC_CONF]
    # dedupe by date_ticker
    seen, hcd = set(), []
    for tk, dt, cat in hc:
        k = f"{dt.date()}_{tk}"
        if k not in seen:
            seen.add(k); hcd.append((tk, dt, cat))
    print(f"LOTTO — unified high-conviction set (mag≥{HC_MAG} & conf≥{HC_CONF}): {len(hcd)} candidates")
    print(f"  catalyst-gate sweep (vs live two-pass v2 materiality = +${TWO_PASS_REF:,}):")
    print(f"  {'θ':>6} {'pass':>5} {'trades':>7} {'P&L':>10} {'Sharpe':>7} {'2x+':>4} {'4x+':>4}")
    best = None
    for th in CAT_THRESHOLDS:
        passers = [(tk, dt) for tk, dt, cat in hcd if cat >= th]
        tr = lotto_sim(passers)
        if not tr:
            print(f"  {th:>6.2f} {len(passers):>5} {0:>7}  (no trades)"); continue
        s = compute_stats(tr)
        x2 = sum(1 for t in tr if t["pnl_pct"] >= 100); x4 = sum(1 for t in tr if t["pnl_pct"] >= 300)
        mark = "  ✓≥two-pass" if s["total_pnl"] >= TWO_PASS_REF else ""
        print(f"  {th:>6.2f} {len(passers):>5} {len(tr):>7} {'${:+,.0f}'.format(s['total_pnl']):>10} "
              f"{s['sharpe']:>7.2f} {x2:>4} {x4:>4}{mark}")
        if best is None or s["total_pnl"] > best[1]["total_pnl"]:
            best = (th, s, tr)

    if best:
        th, s, tr = best
        verdict = "REPLACES the 2nd pass ✓" if s["total_pnl"] >= TWO_PASS_REF else "below two-pass — keep gate / iterate prompt"
        print(f"\n  best θ={th:.2f}: ${s['total_pnl']:+,.0f}/Sh{s['sharpe']:.2f} on {len(tr)} trades  → {verdict}")
        pnls = [t["pnl_usd"] for t in tr]
        boots = [sum(random.choices(pnls, k=len(pnls))) for _ in range(N_BOOT)]
        print(f"    bootstrap total P&L 95% CI [${pctl(boots,2.5):+,.0f}, ${pctl(boots,97.5):+,.0f}]")

    # ── STOCK: unified confidence selectivity ──
    print(f"\nSTOCK — unified confidence selectivity (per-trade quality; cap binds):")
    print(f"  {'conf≥':>6} {'trades':>7} {'mean/tr':>9} {'Sharpe':>7} {'win%':>6}")
    def scale(m, c): return max(cfg.MAX_POSITION_USD * 0.10, cfg.MAX_POSITION_USD * m * c)
    for cf in CONF_FLOORS:
        sd, trd = set(), []
        for r in rows:
            u = r["u"]
            if not (r["tk"] and u.get("sentiment") == "bullish" and u.get("confidence", 0) >= cf):
                continue
            key = f"{r['dt'].date()}_{r['tk']}"
            if key in sd or not is_valid_stock_ticker(r["tk"]):
                continue
            sd.add(key)
            t = simulate_stock(r["tk"], r["dt"], scale(u.get("magnitude", 0.5), u.get("confidence", 0.7)))
            if t:
                trd.append(t)
        if trd:
            s = compute_stats(trd)
            print(f"  {cf:>6.2f} {s['trades']:>7} {('${:+,.0f}'.format(s['total_pnl']/s['trades'])):>9} "
                  f"{s['sharpe']:>7.2f} {s['win_rate']:>6.1f}")
        else:
            print(f"  {cf:>6.2f} {0:>7}  (no trades)")
    print("\n  (compare to current main-scorer: $4/tr loose, $12/tr @ conf≥0.85 from stock_selectivity_sweep)")


if __name__ == "__main__":
    main()
