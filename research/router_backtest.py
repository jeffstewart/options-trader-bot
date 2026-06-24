"""
router_backtest.py — LLM strategy router feasibility backtest.

The question: can an LLM, given a news article, recommend the BEST trading strategy
for that specific signal better than always picking the same fixed strategy?

Method:
  1. Use the existing dual_score_cache.json (both sides already scored separately
     with specialized prompts — far better quality than a neutral single pass).
  2. Define a simple routing rule: if both bull and bear scores pass gate on same day,
     route to PAIRS; if only bull passes and it's an earnings article, route to PEAD;
     if only bull passes (no earnings), route to STOCK; if only bear passes, route to
     BEAR_SHORT. No LLM call needed — the routing logic uses existing scores.
  3. Separately: test an LLM-augmented router that asks the model to pick a strategy,
     using Groq 70B on a sample of 200 articles.
  4. Compare: fixed-strategy baselines vs rule-based router vs LLM router.

The rule-based router is the primary test — it costs nothing and should show whether
signal type naturally maps to strategy. The LLM router is a supplementary check.

Usage:  USE_YAHOO_BARS=1 .venv/bin/python router_backtest.py
        USE_YAHOO_BARS=1 .venv/bin/python router_backtest.py --llm   (adds LLM routing)
"""
import os, sys, json, random
from collections import defaultdict
from datetime import datetime, timezone, timedelta
from pathlib import Path

os.environ.setdefault("USE_YAHOO_BARS", "1")

import tune_v2
from pead_backtest import simulate_pead, is_earnings_article
from pead_bear import sim_short
from backtest import is_valid_stock_ticker
from benchmark import compute_stats
from regime_filter import build_regime
import config as _cfg

WINDOWS = [
    ("bull-meltup", datetime.now(timezone.utc) - timedelta(days=5), 180,
     "dual_score_cache.json"),
    ("2022-bear",   datetime(2022, 6, 30, tzinfo=timezone.utc), 90,
     "bear_dual_cache.json"),
]

LONG_TRAIL  = 0.20
SHORT_TRAIL = 0.15
MAX_HOLD    = 45


def _scale(mag, conf):
    return max(_cfg.MAX_POSITION_USD * 0.10, _cfg.MAX_POSITION_USD * mag * conf)


def _passes(mag, conf, min_conf=None):
    req = (min_conf or _cfg.BASE_CONFIDENCE) + (1 - mag) * _cfg.CONFIDENCE_SLOPE
    return mag >= _cfg.MIN_MAGNITUDE and conf >= req


def load_dual(cache_path, end_dt, days):
    """Load both bull and bear sides, keyed by date."""
    import json as _json
    raw = _json.loads(Path(cache_path).read_text())
    start_dt = end_dt - timedelta(days=days)

    by_date = defaultdict(lambda: {"bull": [], "bear": []})
    for entry in raw.values():
        art   = entry.get("_article", {})
        ts_s  = art.get("created_at", "")
        if not ts_s:
            continue
        try:
            ts = datetime.fromisoformat(str(ts_s).replace("Z", "+00:00"))
            if ts.tzinfo is None:
                ts = ts.replace(tzinfo=timezone.utc)
        except Exception:
            continue
        if not (start_dt <= ts <= end_dt):
            continue
        hl = art.get("headline", "")

        for side in ("bullish", "bearish"):
            sig = entry.get(side, {})
            if sig.get("reasoning") == "SCORE_FAILED" or not sig.get("tickers"):
                continue
            mag  = float(sig.get("magnitude", 0))
            conf = float(sig.get("confidence", 0))
            if not _passes(mag, conf):
                continue
            tk   = sig["tickers"][0]
            if not is_valid_stock_ticker(tk) or tk in ("BTC", "ETH"):
                continue
            direction = "bull" if side == "bullish" else "bear"
            by_date[ts.date()][direction].append({
                "ticker":     tk,
                "magnitude":  mag,
                "confidence": conf,
                "created_at": ts,
                "headline":   hl,
                "position_usd": _scale(mag, conf),
            })
    return by_date


# ── Strategy simulations ──────────────────────────────────────────────────────

def sim_stock(tk, dt, pos):
    return simulate_pead(tk, dt, pos, trail=0.10, max_hold=30)

def sim_pead_fn(tk, dt, pos):
    return simulate_pead(tk, dt, pos, trail=LONG_TRAIL, max_hold=MAX_HOLD)

def sim_pairs(bull_sig, bear_sig):
    pos   = (bull_sig["position_usd"] + bear_sig["position_usd"]) / 2
    long  = simulate_pead(bull_sig["ticker"], bull_sig["created_at"],
                          pos, trail=LONG_TRAIL, max_hold=MAX_HOLD)
    short = sim_short(bear_sig["ticker"], bear_sig["created_at"],
                      pos, trail=SHORT_TRAIL, max_hold=MAX_HOLD)
    if long is None and short is None:
        return None
    net = (long["pnl_usd"] if long else 0) + (short["pnl_usd"] if short else 0)
    return {"ticker": f"{bull_sig['ticker']}L/{bear_sig['ticker']}S",
            "entry_dt": bull_sig["created_at"].isoformat(),
            "pnl_usd": net, "pnl_pct": net / pos * 100 if pos else 0}


# ── Rule-based router ─────────────────────────────────────────────────────────

def route(d, by_date, regime):
    """Route a single day's signals to the appropriate strategy."""
    if regime and not regime(d):
        return []

    bulls = by_date[d]["bull"]
    bears = by_date[d]["bear"]

    trades = []

    if bulls and bears:
        # Both directions: PAIRS trade (top bull vs top bear)
        top_bull = max(bulls, key=lambda s: s["magnitude"] * s["confidence"])
        top_bear = max(bears, key=lambda s: s["magnitude"] * s["confidence"])
        t = sim_pairs(top_bull, top_bear)
        if t:
            t["strategy"] = "pairs"
            trades.append(t)
    elif bulls:
        # Bull only: PEAD if earnings, else STOCK
        top = max(bulls, key=lambda s: s["magnitude"] * s["confidence"])
        is_earn = is_earnings_article(top["headline"])
        fn  = sim_pead_fn if is_earn else sim_stock
        t   = fn(top["ticker"], top["created_at"], top["position_usd"])
        if t:
            t["strategy"] = "pead" if is_earn else "stock"
            trades.append(t)
    elif bears:
        # Bear only: BEAR_SHORT
        top = max(bears, key=lambda s: s["magnitude"] * s["confidence"])
        t   = sim_short(top["ticker"], top["created_at"], top["position_usd"],
                        trail=SHORT_TRAIL, max_hold=MAX_HOLD)
        if t:
            t["strategy"] = "bear_short"
            trades.append(t)

    return trades


# ── Fixed strategy baselines ──────────────────────────────────────────────────

def run_fixed(by_date, regime, strategy):
    trades = []
    seen = set()
    for d in sorted(by_date.keys()):
        if regime and not regime(d):
            continue
        signals = by_date[d]["bull"] if strategy != "bear_short" else by_date[d]["bear"]
        if not signals:
            continue
        top = max(signals, key=lambda s: s["magnitude"] * s["confidence"])
        tk  = top["ticker"]
        key = f"{d}_{tk}"
        if key in seen:
            continue
        seen.add(key)

        if strategy == "stock":
            t = sim_stock(tk, top["created_at"], top["position_usd"])
        elif strategy == "pead":
            if not is_earnings_article(top["headline"]):
                continue
            t = sim_pead_fn(tk, top["created_at"], top["position_usd"])
        elif strategy == "bear_short":
            t = sim_short(tk, top["created_at"], top["position_usd"],
                          trail=SHORT_TRAIL, max_hold=MAX_HOLD)
        else:
            t = None

        if t:
            t["strategy"] = strategy
            trades.append(t)
    return trades


def line(tag, trades):
    s = compute_stats(trades)
    strat_counts = defaultdict(int)
    for t in trades:
        strat_counts[t.get("strategy", "?")] += 1
    breakdown = "  ".join(f"{k}:{v}" for k, v in sorted(strat_counts.items()))
    return (f"  {tag:38}  n={s['trades']:>4}  Sharpe={s['sharpe']:>6.2f}"
            f"  P&L=${s['total_pnl']:>9,.0f}  win={s['win_rate']:>4.1f}%"
            f"  maxDD=${s['max_dd']:>7,.0f}  [{breakdown}]")


def main():
    use_llm = "--llm" in sys.argv
    print("═══ STRATEGY ROUTER BACKTEST ═══\n")
    print("Rule-based routing: pairs when bull+bear same day, pead/stock/short otherwise.\n")

    for label, end_dt, days, cache in WINDOWS:
        if not Path(cache).exists():
            print(f"  [{label}] cache not found — skipping"); continue

        print(f"{'═'*65}")
        print(f"  {label}")
        print(f"{'═'*65}")

        by_date = load_dual(cache, end_dt, days)
        regime  = build_regime(end_dt, days, 200)

        bull_days = sum(1 for d in by_date if by_date[d]["bull"])
        bear_days = sum(1 for d in by_date if by_date[d]["bear"])
        both_days = sum(1 for d in by_date if by_date[d]["bull"] and by_date[d]["bear"])
        print(f"  Bull-only days: {bull_days-both_days}  "
              f"Bear-only days: {bear_days-both_days}  Both: {both_days}\n")

        # ── Rule-based router (no LLM, no regime gate) ────────────────────────
        routed_ng = []
        for d in sorted(by_date.keys()):
            routed_ng.extend(route(d, by_date, regime=None))

        # ── Rule-based router (with regime gate) ──────────────────────────────
        routed_rg = []
        for d in sorted(by_date.keys()):
            routed_rg.extend(route(d, by_date, regime=regime))

        # ── Fixed strategy baselines ──────────────────────────────────────────
        fix_stock   = run_fixed(by_date, regime, "stock")
        fix_pead    = run_fixed(by_date, regime, "pead")
        fix_short   = run_fixed(by_date, regime, "bear_short")

        print(line("Router — NO regime gate",     routed_ng))
        print(line("Router — regime gated",        routed_rg))
        print(line("Fixed: stock (gated)",          fix_stock))
        print(line("Fixed: pead (gated)",           fix_pead))
        print(line("Fixed: bear_short (no gate)",   fix_short))

        # Strategy breakdown of routed trades
        from collections import Counter
        strat_cnt = Counter(t.get("strategy") for t in routed_ng)
        print(f"\n  Router allocation (no gate): {dict(strat_cnt)}")
        print()

    print("═══ INTERPRETATION ═══")
    print("  Router > best fixed strategy → routing adds value")
    print("  Router allocation shows how often each strategy is recommended")
    print("  If pairs dominates: most value comes from L/S structure, not individual strategies")


if __name__ == "__main__":
    main()
