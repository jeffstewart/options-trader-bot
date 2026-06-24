"""
threshold_gate_test.py — validates the materiality prompt AS THE BOT ACTUALLY USES IT:
a THRESHOLD gate (materiality ≥ MATERIALITY_GATE) on the HIGH-CONVICTION subset (baseline
mag≥0.70 & conf≥0.85), trading EVERYTHING that passes — not a top-N ranking on the broad pool.

For each materiality prompt, reports: how many high-conviction candidates PASS the gate (pass
rate), and the realized lotto P&L of ALL passers (Δ0.25/DTE14/hold3/3× cap). Compares to NO gate
(trade all high-conviction). Answers: is the 0.15 threshold well-calibrated for this prompt, and
does the actual gated set make money? Uses cached scores → no new scoring.

Usage:  USE_YAHOO_BARS=1 .venv/bin/python threshold_gate_test.py
"""
import os, json
os.environ.setdefault("USE_YAHOO_BARS", "1")
from datetime import datetime, timezone, timedelta
from pathlib import Path

import backtest as _bt
import config as cfg
from backtest import cache_key
from benchmark import compute_stats
from regime_filter import build_regime

INF = float("inf")
DELTA, DTE, HOLD, POS = 0.25, 14, 3, cfg.LOTTO_POSITION_USD
EXIT = {"tiers": [(1.0, 0.40), (3.0, 0.30), (INF, 0.20)], "hard_target": 3.0}
GATE = cfg.MATERIALITY_GATE                       # 0.15
HC_MAG, HC_CONF = cfg.LOTTO_MIN_MAGNITUDE, cfg.LOTTO_MIN_CONFIDENCE   # 0.70 / 0.85
END = datetime.now(timezone.utc) - timedelta(days=5)
PROMPTS = ["materiality_fewshot", "materiality_fewshot_v2", "materiality_fewshot_v3"]


def high_conviction_pool():
    """The lotto candidate set the bot actually considers: baseline mag≥0.70 & conf≥0.85,
    regime-gated. Returns list of (ticker, dt, cache_key)."""
    raw = json.load(open("dual_score_cache.json"))
    reg = build_regime(END, 180, 200)
    out, seen = [], set()
    for v in raw.values():
        if not isinstance(v, dict):
            continue
        a, b = v.get("_article", {}) or {}, v.get("bullish", {}) or {}
        tks = [t for t in (b.get("tickers") or []) if t not in ("BTC", "ETH")]
        h, ca = a.get("headline"), a.get("created_at")
        if not tks or not h or not ca:
            continue
        if float(b.get("magnitude", 0)) < HC_MAG or float(b.get("confidence", 0)) < HC_CONF:
            continue
        try:
            dt = datetime.fromisoformat(str(ca).replace("Z", "+00:00")).astimezone(timezone.utc)
        except Exception:
            continue
        if not (END - timedelta(days=180) <= dt <= END) or (reg and not reg(dt.date())):
            continue
        k = f"{dt.date()}_{tks[0]}"
        if k in seen:
            continue
        seen.add(k)
        out.append((tks[0], dt, cache_key(h, a.get("summary", ""))))
    return out


def simulate(passers):
    save = {k: getattr(_bt, k) for k in ("TARGET_DELTA", "DTE_TARGET", "MAX_HOLD_DAYS", "TRAILING_STOP_PCT", "EXIT_PARAMS")}
    _bt.TARGET_DELTA, _bt.DTE_TARGET, _bt.MAX_HOLD_DAYS = DELTA, DTE, HOLD
    _bt.TRAILING_STOP_PCT, _bt.EXIT_PARAMS = 0.30, EXIT
    trades = []
    try:
        for tk, dt in passers:
            if not _bt.is_valid_stock_ticker(tk):
                continue
            sp = _bt.get_price_at(tk, dt)
            if not sp:
                continue
            t = _bt.simulate_option_pnl(tk, dt, sp, POS, {"magnitude": 0.8, "confidence": 0.9},
                                        option_type="call", exit_rule="tiered_profit", spread_mult=1.0)
            if t:
                trades.append(t)
    finally:
        for kk, vv in save.items():
            setattr(_bt, kk, vv)
    return trades


def report(tag, n_cand, passers):
    trades = simulate(passers)
    if not trades:
        print(f"  {tag:34} passed {len(passers):>4}/{n_cand}  → 0 trades"); return
    s = compute_stats(trades)
    x2 = sum(1 for t in trades if t["pnl_pct"] >= 100); x4 = sum(1 for t in trades if t["pnl_pct"] >= 300)
    pr = len(passers) / n_cand * 100
    print(f"  {tag:34} passed {len(passers):>4}/{n_cand} ({pr:>3.0f}%)  trades={len(trades):>4}  "
          f"P&L={'${:+,.0f}'.format(s['total_pnl']):>9}  Sharpe={s['sharpe']:>5.2f}  2x+={x2:>3} 4x+={x4:>3}")


def main():
    sc = json.loads(Path("prompt_exp_scores.json").read_text())
    pool = high_conviction_pool()
    n = len(pool)
    print(f"═══ THRESHOLD-GATE TEST (bot's real logic: materiality ≥{GATE} on high-conviction set) ═══")
    print(f"high-conviction lotto candidates (baseline mag≥{HC_MAG} & conf≥{HC_CONF}, regime-gated): {n}\n")
    # reference: trade ALL high-conviction (no materiality gate)
    report("NO GATE (all high-conviction)", n, [(tk, dt) for tk, dt, ck in pool])
    print()
    for prompt in PROMPTS:
        pre = f"llama3.2:{prompt}:"
        covered = [(tk, dt, sc[pre + ck]) for tk, dt, ck in pool if (pre + ck) in sc and sc[pre + ck] is not None]
        passers = [(tk, dt) for tk, dt, score in covered if score >= GATE]
        cov = len(covered)
        tag = f"{prompt} ≥{GATE}" + ("" if cov == n else f" [{cov}/{n} scored]")
        report(tag, cov, passers)
    print(f"\nRead: this is what the bot ACTUALLY trades. Watch the PASS RATE — if v3 passes far fewer")
    print(f"than baseline, the 0.15 threshold may be mis-calibrated for v3 (it scores lower) → lotto")
    print(f"fires too rarely. Compare gated P&L/trade-count/Sharpe to NO-GATE and across prompts.")


if __name__ == "__main__":
    main()
