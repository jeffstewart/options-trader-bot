"""
mat_iv_gate_test.py — does combining the MATERIALITY gate with the LOW-IV finding compound?

Two independent lotto-quality signals validated separately:
  • materiality (news-surprise score, llama3.2/materiality_fewshot ≥ MATERIALITY_GATE)
  • low entry IV (greek_screen_backtest: the big winners came from cheap, low-IV entries)
This tests them alone and TOGETHER on the gated lotto signals (Δ0.25, 14 DTE, 7d hold) to see
whether the combination beats either alone (compounds) or just overlaps.

Materiality scores are READ from the existing prompt_exp_scores.json cache (scored 2026-06-08
on local llama3.2 — NO new scoring / NO Groq). Entry IV comes from the same BS model the sim
prices with. We restrict to the subset that HAS a materiality score so all gates share a base.

Usage:  USE_YAHOO_BARS=1 .venv/bin/python mat_iv_gate_test.py
"""
import os, json, statistics
os.environ.setdefault("USE_YAHOO_BARS", "1")
from datetime import datetime, timezone, timedelta
from pathlib import Path

import backtest as _bt
import tune_v2, config as cfg
from backtest import cache_key
from pricing import strike_for_delta
from benchmark import compute_stats
from regime_filter import build_regime

DELTA, DTE, HOLD, POS = cfg.LOTTO_TARGET_DELTA, 14, cfg.LOTTO_MAX_HOLD_DAYS, cfg.LOTTO_POSITION_USD
GATE_MAG, GATE_CONF = cfg.LOTTO_MIN_MAGNITUDE, cfg.LOTTO_MIN_CONFIDENCE
MAT_GATE = cfg.MATERIALITY_GATE                      # 0.15
TIERED = {"tiers": [(1.0, 0.40), (3.0, 0.30), (float("inf"), 0.20)]}
CACHES = ["dual_score_cache.json", "bear_dual_cache.json"]
WINDOWS = [
    ("bull-meltup", datetime.now(timezone.utc) - timedelta(days=5), 180, "dual_score_cache.json"),
    ("2022-bear",   datetime(2022, 6, 30, tzinfo=timezone.utc), 90, "bear_dual_cache.json"),
]


def build_materiality_map():
    """headline → materiality score, from the raw caches + the cached llama3.2 scores."""
    scores = json.loads(Path("prompt_exp_scores.json").read_text())
    m = {}
    for cache in CACHES:
        for v in json.load(open(cache)).values():
            if not isinstance(v, dict):
                continue
            a = v.get("_article", {}) or {}
            h = a.get("headline")
            if not h:
                continue
            key = f"llama3.2:materiality_fewshot:{cache_key(h, a.get('summary', ''))}"
            s = scores.get(key)
            if s is not None:
                m[h] = float(s)
    return m


def collect(rows, regime, mat_map):
    save = {k: getattr(_bt, k) for k in
            ("TARGET_DELTA", "DTE_TARGET", "MAX_HOLD_DAYS", "TRAILING_STOP_PCT", "EXIT_PARAMS")}
    _bt.TARGET_DELTA, _bt.DTE_TARGET, _bt.MAX_HOLD_DAYS = DELTA, DTE, HOLD
    _bt.TRAILING_STOP_PCT, _bt.EXIT_PARAMS = 0.30, TIERED
    T = DTE / 365.0
    trades, seen, have_mat = [], set(), 0
    try:
        for r in rows:
            d = r["created_at"].date()
            if regime and not regime(d):
                continue
            for tk in r["tickers"][:1]:
                if tk in ("BTC", "ETH") or not _bt.is_valid_stock_ticker(tk):
                    continue
                k = f"{d}_{tk}"
                if k in seen:
                    continue
                seen.add(k)
                mat = mat_map.get(r["headline"])
                if mat is None:
                    continue                              # no materiality score → exclude (fair base)
                have_mat += 1
                sp = _bt.get_price_at(tk, r["created_at"])
                if not sp:
                    continue
                entry_iv = _bt.base_iv_for(tk, r["created_at"]) * _bt.NEWS_IV_MULTIPLIER
                t = _bt.simulate_option_pnl(
                    tk, r["created_at"], sp, max(POS * 0.5, POS * r["magnitude"] * r["confidence"]),
                    {"magnitude": r["magnitude"], "confidence": r["confidence"]},
                    option_type="call", exit_rule="tiered_profit", spread_mult=1.0)
                if not t:
                    continue
                t.update(materiality=mat, entry_iv=entry_iv)
                trades.append(t)
    finally:
        for kk, vv in save.items():
            setattr(_bt, kk, vv)
    return trades, have_mat


def stat_line(tag, trades, base_n):
    if not trades:
        return f"  {tag:32} n=   0"
    s = compute_stats(trades)
    x2 = sum(1 for t in trades if t["pnl_pct"] >= 100)
    x4 = sum(1 for t in trades if t["pnl_pct"] >= 300)
    kept = f"{len(trades)}/{base_n} ({len(trades)/base_n*100:.0f}%)"
    return (f"  {tag:32} n={kept:>12}  P&L=${s['total_pnl']:>8,.0f}  Sharpe={s['sharpe']:>5.2f}  "
            f"win={s['win_rate']:>4.1f}%  2x+={x2:>3} 4x+={x4:>3}")


def main():
    print(f"═══ MATERIALITY × LOW-IV combined lotto gate (Δ{DELTA}, {DTE}DTE, {HOLD}d) ═══")
    print(f"materiality gate ≥{MAT_GATE} (cached llama3.2) · IV gate = drop top-20% (≤p80)\n")
    mat_map = build_materiality_map()
    print(f"materiality map: {len(mat_map):,} headlines scored\n")
    for label, end_dt, days, cache in WINDOWS:
        if not Path(cache).exists():
            print(f"[{label}] cache missing — skip"); continue
        scored = tune_v2.load_scored_from_dual_cache(Path(cache), "bull", end_dt, days)
        gated = [r for r in scored if r["magnitude"] >= GATE_MAG and r["confidence"] >= GATE_CONF]
        reg = build_regime(end_dt, days, 200)
        trades, have_mat = collect(gated, reg, mat_map)
        base_n = len(trades)
        print(f"═══ {label} ═══  gated={len(gated)}  with-materiality+sim={base_n}")
        if base_n < 15:
            print("  (too few)\n"); continue
        iv_p80 = sorted(t["entry_iv"] for t in trades)[int(0.80 * (base_n - 1))]
        gates = {
            "BASELINE (all scored)":     trades,
            "MATERIALITY ≥ gate":        [t for t in trades if t["materiality"] >= MAT_GATE],
            "LOW-IV (≤p80)":             [t for t in trades if t["entry_iv"] <= iv_p80],
            "BOTH (mat≥gate & IV≤p80)":  [t for t in trades if t["materiality"] >= MAT_GATE and t["entry_iv"] <= iv_p80],
        }
        for tag, ts in gates.items():
            print(stat_line(tag, ts, base_n))
        print()
    print("Read: 'BOTH' should beat each single gate (higher P&L/Sharpe, winners kept) to be worth")
    print("stacking. If BOTH ≈ the better single gate, they overlap — stacking adds nothing.")


if __name__ == "__main__":
    main()
