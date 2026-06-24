"""
bootstrap_gate_test.py — is the materiality gate's effect on lotto P&L REAL or noise?

The gated set is a strict SUBSET of the no-gate set (gate = high-conviction trades whose v2
materiality ≥0.15). So:  no-gate P&L − gate P&L  ==  total P&L of the GATED-OUT trades.
⇒ "does the gate help?" = "are the trades it DISCARDS net-positive (gate costs money) or
net-negative (gate earns its keep)?"  We bootstrap that difference + per-trade quality, and
jackknife the gated-out P&L to check it isn't one lucky multibagger.

Uses cached v2 scores → no ollama scoring (safe to run with the market open).
Usage:  USE_YAHOO_BARS=1 .venv/bin/python bootstrap_gate_test.py
"""
import os, json, random, statistics
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
GATE = cfg.MATERIALITY_GATE
HC_MAG, HC_CONF = cfg.LOTTO_MIN_MAGNITUDE, cfg.LOTTO_MIN_CONFIDENCE
END = datetime.now(timezone.utc) - timedelta(days=5)
PROMPT = "materiality_fewshot_v2"          # the LIVE prompt
N_BOOT = 10000
random.seed(12345)


def pool_with_scores(sc):
    """High-conviction lotto candidates (the bot's real lotto set) + their v2 materiality score."""
    raw = json.load(open("dual_score_cache.json"))
    reg = build_regime(END, 180, 200)
    out, seen = [], set()
    pre = f"llama3.2:{PROMPT}:"
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
        key = pre + cache_key(h, a.get("summary", ""))
        if key not in sc or sc[key] is None:
            continue
        seen.add(k)
        out.append((tks[0], dt, float(sc[key])))
    return out


def simulate(pool):
    """Simulate EVERY high-conviction candidate once; tag each resulting trade passed_gate."""
    save = {k: getattr(_bt, k) for k in ("TARGET_DELTA", "DTE_TARGET", "MAX_HOLD_DAYS", "TRAILING_STOP_PCT", "EXIT_PARAMS")}
    _bt.TARGET_DELTA, _bt.DTE_TARGET, _bt.MAX_HOLD_DAYS = DELTA, DTE, HOLD
    _bt.TRAILING_STOP_PCT, _bt.EXIT_PARAMS = 0.30, EXIT
    trades = []
    try:
        for tk, dt, score in pool:
            if not _bt.is_valid_stock_ticker(tk):
                continue
            sp = _bt.get_price_at(tk, dt)
            if not sp:
                continue
            t = _bt.simulate_option_pnl(tk, dt, sp, POS, {"magnitude": 0.8, "confidence": 0.9},
                                        option_type="call", exit_rule="tiered_profit", spread_mult=1.0)
            if t:
                t["passed"] = score >= GATE
                trades.append(t)
    finally:
        for kk, vv in save.items():
            setattr(_bt, kk, vv)
    return trades


def pctl(xs, p):
    xs = sorted(xs); i = max(0, min(len(xs) - 1, int(round(p / 100 * (len(xs) - 1)))))
    return xs[i]


def main():
    sc = json.loads(Path("prompt_exp_scores.json").read_text())
    pool = pool_with_scores(sc)
    trades = simulate(pool)
    gin = [t for t in trades if t["passed"]]
    gout = [t for t in trades if not t["passed"]]
    pin, pout = [t["pnl_usd"] for t in gin], [t["pnl_usd"] for t in gout]
    nogate_tot, gate_tot, gout_tot = sum(t["pnl_usd"] for t in trades), sum(pin), sum(pout)

    print(f"═══ BOOTSTRAP: materiality gate vs NO gate (v2, {len(trades)} high-conviction lotto trades) ═══\n")
    si, so, sa = compute_stats(gin), compute_stats(gout), compute_stats(trades)
    print(f"  GATE ON  (v2≥{GATE}) : {len(gin):>3} trades  P&L {'${:+,.0f}'.format(gate_tot):>9}  Sharpe {si['sharpe']:>5.2f}  mean/trade ${statistics.mean(pin):>+7,.0f}")
    print(f"  NO GATE  (all HC)   : {len(trades):>3} trades  P&L {'${:+,.0f}'.format(nogate_tot):>9}  Sharpe {sa['sharpe']:>5.2f}  mean/trade ${statistics.mean([t['pnl_usd'] for t in trades]):>+7,.0f}")
    print(f"  GATED-OUT (discarded): {len(gout):>3} trades  P&L {'${:+,.0f}'.format(gout_tot):>9}  Sharpe {so['sharpe']:>5.2f}  mean/trade ${statistics.mean(pout):>+7,.0f}")
    print(f"\n  no-gate − gate = gated-out total = ${gout_tot:+,.0f}  (>0 ⇒ gate DISCARDS profit; <0 ⇒ gate HELPS)\n")

    # ── Bootstrap 1: total P&L of the gated-out (discarded) trades — resample those trades ──
    boots_out = [sum(random.choices(pout, k=len(pout))) for _ in range(N_BOOT)]
    lo, hi = pctl(boots_out, 2.5), pctl(boots_out, 97.5)
    frac_pos = sum(1 for x in boots_out if x > 0) / N_BOOT
    print(f"  ▸ Gated-out total P&L  : {'${:+,.0f}'.format(gout_tot)}  95% CI [${lo:+,.0f}, ${hi:+,.0f}]")
    print(f"    P(discarded trades net profitable) = {frac_pos*100:.1f}%   "
          f"→ {'CI excludes 0: gate robustly COSTS P&L' if lo > 0 else ('CI excludes 0: gate robustly HELPS' if hi < 0 else 'CI straddles 0: effect is NOISE')}")

    # ── Bootstrap 2: per-trade quality, gated-in mean − gated-out mean ──
    diffs = [statistics.mean(random.choices(pin, k=len(pin))) - statistics.mean(random.choices(pout, k=len(pout))) for _ in range(N_BOOT)]
    dlo, dhi = pctl(diffs, 2.5), pctl(diffs, 97.5)
    obs = statistics.mean(pin) - statistics.mean(pout)
    frac_better = sum(1 for d in diffs if d > 0) / N_BOOT
    print(f"\n  ▸ Per-trade quality (gate − gatedout mean): ${obs:+,.0f}/trade  95% CI [${dlo:+,.0f}, ${dhi:+,.0f}]")
    print(f"    P(gate picks higher-$/trade) = {frac_better*100:.1f}%   "
          f"→ {'gate picks SIGNIFICANTLY better trades' if dlo > 0 else 'per-trade edge not significant'}")

    # ── Jackknife: is the gated-out total driven by ONE lucky trade? ──
    jk = [gout_tot - p for p in pout]   # leave-one-out totals
    big = max(pout); print(f"\n  ▸ Jackknife gated-out total (leave-one-out): range [${min(jk):+,.0f}, ${max(jk):+,.0f}]")
    print(f"    biggest single gated-out trade = ${big:+,.0f}  → drop it and gated-out total = ${gout_tot-big:+,.0f}")
    flip = "FLIPS NEGATIVE (gate's 'cost' was one multibagger)" if (gout_tot > 0 and gout_tot - big < 0) else "stays same sign"
    print(f"    removing the single best discarded trade: {flip}")

    print(f"\n  READ: lotto is convex with uniform small size, so RAW total P&L is the objective. If the")
    print(f"  gated-out CI straddles 0 / flips on one trade, the gate is ~P&L-neutral and only buys")
    print(f"  Sharpe — weak justification. The live shadow A/B (gated-out paper trades) is the forward check.")


if __name__ == "__main__":
    main()
