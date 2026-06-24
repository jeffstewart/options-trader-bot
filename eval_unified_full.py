"""
eval_unified_full.py — rigorous eval of a unified prompt on ONE window (bull or bear).
Reads unified_scores.json (full dicts) + the window's cache, joins by date, and reports:
  • LOTTO: prompt's high-conviction set (mag≥0.70 & conf≥0.85, bullish, has ticker), optional
    catalyst gate, simulate (Δ0.25/DTE14/hold3/3×cap) → P&L/Sharpe/multibaggers + BOOTSTRAP (total
    P&L 95% CI) + JACKKNIFE (drop each trade → range, biggest single trade's share).
  • STOCK: confidence-selectivity sweep → per-trade quality.
Window via env so the overnight driver can call it for bull and the 2022-bear out-of-regime set.
The regime decision is baked in at SCORING time (bull scored regime-on, bear regime-off), so this
just evaluates whatever was scored.

Usage:  UNI_PROMPT=unified_v3 UNI_CACHE=dual_score_cache.json UNI_LABEL=bull \
        USE_YAHOO_BARS=1 .venv/bin/python eval_unified_full.py
"""
import os, json, random, statistics
os.environ.setdefault("USE_YAHOO_BARS", "1")
from datetime import datetime, timezone, timedelta
from pathlib import Path

import backtest as _bt
import config as cfg
from benchmark import compute_stats
from backtest import is_valid_stock_ticker, cache_key
from stock_backtest import simulate_stock

INF = float("inf")
DELTA, DTE, HOLD, POS = 0.25, 14, 3, cfg.LOTTO_POSITION_USD
EXIT = {"tiers": [(1.0, 0.40), (3.0, 0.30), (INF, 0.20)], "hard_target": 3.0}
HC_MAG, HC_CONF = cfg.LOTTO_MIN_MAGNITUDE, cfg.LOTTO_MIN_CONFIDENCE
PROMPT = os.environ.get("UNI_PROMPT", "unified_v1")
CACHE = os.environ.get("UNI_CACHE", "dual_score_cache.json")
DAYS = int(os.environ.get("UNI_DAYS", "180"))
_E = os.environ.get("UNI_END", "")
END = datetime.fromisoformat(_E).replace(tzinfo=timezone.utc) if _E else datetime.now(timezone.utc) - timedelta(days=5)
LABEL = os.environ.get("UNI_LABEL", "window")
TWO_PASS_REF = 15686
CAT_THRESHOLDS = [0.0, 0.15, 0.30, 0.50]
CONF_FLOORS = [0.70, 0.85, 0.90]
N_BOOT = 10000
random.seed(20260612)


def candidates():
    raw = json.load(open(CACHE))
    out, seen = [], set()
    for v in raw.values():
        if not isinstance(v, dict):
            continue
        a = v.get("_article", {}) or {}
        h, ca, body = a.get("headline"), a.get("created_at"), a.get("summary", "")
        if not h or not ca:
            continue
        try:
            dt = datetime.fromisoformat(str(ca).replace("Z", "+00:00")).astimezone(timezone.utc)
        except Exception:
            continue
        if not (END - timedelta(days=DAYS) <= dt <= END):
            continue
        ck = cache_key(h, body)
        if ck in seen:
            continue
        seen.add(ck)
        out.append({"dt": dt, "ck": ck})
    return out


def load_joined():
    sc = json.loads(Path("unified_scores.json").read_text())
    rows = []
    for c in candidates():
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
    has_cat = any("catalyst" in r["u"] for r in rows)
    print(f"\n████ {PROMPT} — {LABEL} ████  ({len(rows)} scored articles, catalyst field: {has_cat})")

    # high-conviction lotto set
    hc, seen = [], set()
    for r in rows:
        u = r["u"]
        if not (r["tk"] and u.get("sentiment") == "bullish"
                and u.get("magnitude", 0) >= HC_MAG and u.get("confidence", 0) >= HC_CONF):
            continue
        k = f"{r['dt'].date()}_{r['tk']}"
        if k in seen:
            continue
        seen.add(k); hc.append((r["tk"], r["dt"], float(u.get("catalyst", 1.0))))
    print(f"  LOTTO high-conviction candidates: {len(hc)}")
    ths = CAT_THRESHOLDS if has_cat else [0.0]
    best = None
    for th in ths:
        passers = [(tk, dt) for tk, dt, cat in hc if cat >= th]
        tr = lotto_sim(passers)
        if not tr:
            print(f"    catθ={th:.2f}: {len(passers)} pass → 0 trades"); continue
        s = compute_stats(tr); x2 = sum(1 for t in tr if t["pnl_pct"] >= 100); x4 = sum(1 for t in tr if t["pnl_pct"] >= 300)
        tag = " ✓≥2pass" if s["total_pnl"] >= TWO_PASS_REF else ""
        print(f"    catθ={th:.2f}: {len(tr):>3} trades  P&L {'${:+,.0f}'.format(s['total_pnl']):>10}  Sh {s['sharpe']:>5.2f}  2x+={x2} 4x+={x4}{tag}")
        if best is None or s["total_pnl"] > best[1]:
            best = (th, s["total_pnl"], tr)
    if best:
        th, tot, tr = best
        pnls = [t["pnl_usd"] for t in tr]
        boots = [sum(random.choices(pnls, k=len(pnls))) for _ in range(N_BOOT)]
        big = max(pnls); jk = tot - big
        print(f"    ▸ best catθ={th:.2f}: ${tot:+,.0f} on {len(tr)} trades")
        print(f"      BOOTSTRAP total P&L 95% CI [${pctl(boots,2.5):+,.0f}, ${pctl(boots,97.5):+,.0f}]  (P>0 = {sum(1 for b in boots if b>0)/N_BOOT*100:.0f}%)")
        print(f"      JACKKNIFE: biggest single trade ${big:+,.0f} ({big/tot*100:.0f}% of total) → drop it = ${jk:+,.0f} {'(still positive)' if jk>0 else '(FLIPS NEG — one-trade-driven)'}")

    # stock
    def scale(m, c): return max(cfg.MAX_POSITION_USD * 0.10, cfg.MAX_POSITION_USD * m * c)
    print(f"  STOCK conf-selectivity:")
    for cf in CONF_FLOORS:
        sd, trd = set(), []
        for r in rows:
            u = r["u"]
            if not (r["tk"] and u.get("sentiment") == "bullish" and u.get("confidence", 0) >= cf):
                continue
            k = f"{r['dt'].date()}_{r['tk']}"
            if k in sd or not is_valid_stock_ticker(r["tk"]):
                continue
            sd.add(k)
            t = simulate_stock(r["tk"], r["dt"], scale(u.get("magnitude", 0.5), u.get("confidence", 0.7)))
            if t:
                trd.append(t)
        if trd:
            s = compute_stats(trd)
            print(f"    conf≥{cf:.2f}: {s['trades']:>4} tr  ${s['total_pnl']/s['trades']:+,.0f}/tr  Sh {s['sharpe']:>5.2f}  win {s['win_rate']:.0f}%")
        else:
            print(f"    conf≥{cf:.2f}: 0 trades")


if __name__ == "__main__":
    main()
