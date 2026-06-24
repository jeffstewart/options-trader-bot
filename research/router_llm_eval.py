"""
router_llm_eval.py — does the LLM route to the RIGHT strategy?

Reads router_llm_cache.json (produced by router_rescore.py: per article the LLM
picked a strategy ∈ {long_calls, lotto_calls, long_stock, pead, bear_short, none}).

For the bullish-tradeable universe (every article the LLM tagged bullish with a
real strategy), it compares:
  • LLM-routed  — trade each article with the strategy the LLM chose
  • always_X    — trade EVERY one of those articles with a single fixed strategy
The LLM router only adds value if LLM-routed beats the best always_X. Separately
reports the bearish picks (bear_short) and the LLM's pick distribution.

Long strategies are 200d-SMA regime-gated (as the live bot does). Net of spreads.

Run AFTER the overnight rescore finishes:
    USE_YAHOO_BARS=1 .venv/bin/python router_llm_eval.py
    USE_YAHOO_BARS=1 .venv/bin/python router_llm_eval.py --cache router_llm_bear_cache.json \
        --end-date 2022-06-30 --days 90   # cross-regime, once a 2022 rescore exists
"""
import os, argparse, json
from collections import defaultdict
from datetime import datetime, timezone, timedelta
from pathlib import Path

os.environ.setdefault("USE_YAHOO_BARS", "1")

import backtest as _bt
from pead_backtest import simulate_pead
from pead_bear import sim_short
from benchmark import compute_stats
from regime_filter import build_regime
import config as _cfg

LOTTO_DELTA, STD_DELTA = 0.25, 0.50
LOTTO_TIERS = {"tiers": [(1.0, 0.40), (3.0, 0.30), (float("inf"), 0.20)]}


def _scale(mag, conf, base=_cfg.MAX_POSITION_USD):
    return max(base * 0.10, base * mag * conf)


def load_router(cache_file, end_dt, days):
    raw = json.loads(Path(cache_file).read_text())
    start = end_dt - timedelta(days=days)
    rows = []
    for v in raw.values():
        if v.get("reasoning") == "SCORE_FAILED" or v.get("strategy") == "none":
            continue
        tk = v.get("ticker", "")
        if not tk or tk in ("BTC", "ETH") or not _bt.is_valid_stock_ticker(tk):
            continue
        a = v.get("_article", {})
        try:
            ts = datetime.fromisoformat(str(a.get("created_at", "")).replace("Z", "+00:00"))
            if ts.tzinfo is None: ts = ts.replace(tzinfo=timezone.utc)
        except Exception:
            continue
        if not (start <= ts <= end_dt):
            continue
        rows.append({"ticker": tk, "strategy": v["strategy"], "sentiment": v.get("sentiment", ""),
                     "mag": float(v.get("magnitude", 0)), "conf": float(v.get("confidence", 0)),
                     "swing": float(v.get("swing", 0)), "created_at": ts})
    return rows


def sim_call(tk, dt, pos, delta, exit_rule="tiered_profit", exit_params=None):
    save = {k: getattr(_bt, k) for k in ("TARGET_DELTA", "DTE_TARGET", "MAX_HOLD_DAYS",
                                         "TRAILING_STOP_PCT", "EXIT_PARAMS")}
    _bt.TARGET_DELTA = delta
    _bt.DTE_TARGET = 21; _bt.MAX_HOLD_DAYS = 21; _bt.TRAILING_STOP_PCT = 0.30
    _bt.EXIT_PARAMS = exit_params or LOTTO_TIERS
    try:
        sp = _bt.get_price_at(tk, dt)
        if not sp: return None
        return _bt.simulate_option_pnl(tk, dt, sp, pos, {"magnitude": 0.7, "confidence": 0.8},
                                       option_type="call", exit_rule=exit_rule, spread_mult=1.0)
    finally:
        for k, v in save.items(): setattr(_bt, k, v)


def trade(strategy, tk, dt, pos):
    """Simulate one trade for a given strategy. Returns dict with pnl_usd or None."""
    if strategy == "lotto_calls":
        return sim_call(tk, dt, pos, LOTTO_DELTA)
    if strategy == "long_calls":
        return sim_call(tk, dt, pos, STD_DELTA)
    if strategy == "long_stock":
        return simulate_pead(tk, dt, pos, trail=0.10, max_hold=30)
    if strategy == "pead":
        return simulate_pead(tk, dt, pos, trail=0.20, max_hold=45)
    if strategy == "bear_short":
        return sim_short(tk, dt, pos, trail=0.15, max_hold=45)
    return None


def stat_line(tag, trades):
    s = compute_stats(trades)
    return (f"  {tag:24}  n={s['trades']:>4}  Sharpe={s['sharpe']:>6.2f}"
            f"  P&L=${s['total_pnl']:>9,.0f}  win={s['win_rate']:>4.1f}%")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", default="router_llm_cache.json")
    ap.add_argument("--end-date", default=None)
    ap.add_argument("--days", type=int, default=180)
    args = ap.parse_args()

    if not Path(args.cache).exists():
        print(f"Cache {args.cache} not found — run router_rescore.py first."); return
    end_dt = (datetime.strptime(args.end_date, "%Y-%m-%d").replace(tzinfo=timezone.utc)
              if args.end_date else datetime.now(timezone.utc) - timedelta(days=5))

    rows = load_router(args.cache, end_dt, args.days)
    regime = build_regime(end_dt, args.days, 200)
    print("═══ LLM STRATEGY ROUTER — EVALUATION ═══\n")
    mix = defaultdict(int)
    for r in rows: mix[r["strategy"]] += 1
    print(f"  Tradeable LLM picks: {len(rows)}")
    print("  Pick distribution:", "  ".join(f"{k}={v}" for k, v in sorted(mix.items())), "\n")

    BULLISH = {"long_calls", "lotto_calls", "long_stock", "pead"}
    bull_rows = [r for r in rows if r["strategy"] in BULLISH and r["sentiment"] == "bullish"]

    # LLM-routed (each its own pick) vs always-one-strategy, on the SAME bullish set.
    # Long strategies are regime-gated (live behavior).
    routed, always = [], defaultdict(list)
    for r in bull_rows:
        if regime and not regime(r["created_at"].date()):
            continue
        pos = _scale(r["mag"], r["conf"])
        t = trade(r["strategy"], r["ticker"], r["created_at"], pos)
        if t: routed.append(t)
        for fixed in ("long_calls", "lotto_calls", "long_stock", "pead"):
            tf = trade(fixed, r["ticker"], r["created_at"], pos)
            if tf: always[fixed].append(tf)

    print("  ── Bullish universe (regime-gated): LLM routing vs fixed ──")
    print(stat_line("LLM-ROUTED", routed))
    for fixed in ("long_calls", "lotto_calls", "long_stock", "pead"):
        print(stat_line(f"always {fixed}", always[fixed]))

    # Bearish picks
    bear_rows = [r for r in rows if r["strategy"] == "bear_short"]
    bear_trades = []
    for r in bear_rows:
        t = trade("bear_short", r["ticker"], r["created_at"], _scale(r["mag"], r["conf"]))
        if t: bear_trades.append(t)
    print("\n  ── Bearish picks (no regime gate) ──")
    print(stat_line("bear_short", bear_trades))

    # Verdict
    routed_pnl = compute_stats(routed)["total_pnl"]
    best_fixed = max(("long_calls", "lotto_calls", "long_stock", "pead"),
                     key=lambda k: compute_stats(always[k])["total_pnl"])
    best_pnl = compute_stats(always[best_fixed])["total_pnl"]
    print(f"\n  Best fixed bullish strategy: always {best_fixed} (${best_pnl:,.0f})")
    verdict = "BEATS" if routed_pnl > best_pnl else "does NOT beat"
    print(f"  → LLM routing {verdict} the best fixed strategy "
          f"(${routed_pnl:,.0f} vs ${best_pnl:,.0f}).")
    print("\n  (Realized backtest, net of spreads, one regime per run. Run with a 2022")
    print("   rescore for the cross-regime read.)")


if __name__ == "__main__":
    main()
