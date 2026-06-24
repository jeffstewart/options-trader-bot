"""
exit_sweep_unified.py — sweep EXIT criteria (trailing-stop × max-hold) for EACH active long leg
on the unified_v1 score distribution, at each leg's CURRENT gate + geometry, NET of realistic
costs (options). Answers: given the new prompt + new geometries, what trail width / hold cap is
right per leg, and does it beat the current live exit?

Legs (each its own population / sim / exit mechanic):
  • stock      — simulate_stock, flat STOCK_TRAIL × MAX_HOLD (live 10% / 2d). Dynamic-floor gate.
  • news_call  — Δ0.40/DTE10 call, flat premium trail × hold (live: tiered EXIT_TIERS / no cap).
                 Sweet-spot gate mag≥0.75. NET of spreads.
  • lotto      — Δ0.25/DTE14 OTM call, trail × hold (live: tiered+3× cap / 3d). Gate mag≥0.70&conf≥0.85.
  • qqq_macro  — QQQ Δ0.60/DTE10 call on NO-ticker macro-bullish signals (≤1/day). trail × hold.
  • pead       — earnings-filtered stock, flat trail × max_hold (live tiered / 45d). Dynamic-floor gate.

Each leg prints a trail×hold grid (total net P&L / Sharpe), the CURRENT live-exit benchmark, and
the best robust cell (bootstrap CI + 2022-bear guard). Cached scores + Yahoo bars only.

Usage:  USE_YAHOO_BARS=1 .venv/bin/python -u exit_sweep_unified.py            # all legs
        EXIT_LEG=news_call .venv/bin/python -u exit_sweep_unified.py          # one leg
"""
import os, json, random
os.environ.setdefault("USE_YAHOO_BARS", "1")
from datetime import datetime, timezone, timedelta
from pathlib import Path

import backtest as _bt
import stock_backtest as _sb
import pead_backtest as _pb
import config as cfg
from benchmark import compute_stats
from regime_filter import build_regime
from backtest import is_valid_stock_ticker, cache_key

PROMPT = os.environ.get("UNI_PROMPT", "unified_v1")
ONLY   = os.environ.get("EXIT_LEG", "")            # "" = all legs
BULL = ("bull", datetime.now(timezone.utc) - timedelta(days=5), 180, "dual_score_cache.json")
BEAR = ("2022-bear", datetime(2022, 6, 30, tzinfo=timezone.utc), 90, "bear_dual_cache.json")
N_BOOT = 10000
random.seed(20260614)


def _scale(mag, conf, base):
    return max(base * 0.10, base * mag * conf)


def dyn_gate(mag, conf):
    return mag >= cfg.MIN_MAGNITUDE and conf >= cfg.BASE_CONFIDENCE + (1 - mag) * cfg.CONFIDENCE_SLOPE


def load_signals(cache, end_dt, days, prompt, want_ticker=True):
    """Join unified_scores.json[<prompt>:<ck>] to article metadata. Returns rows
    {dt, headline, magnitude, confidence, tickers, sentiment}. want_ticker=False keeps
    only NO-ticker bullish rows (for qqq_macro)."""
    sc = json.loads(Path("unified_scores.json").read_text())
    raw = json.loads(Path(cache).read_text())
    start = end_dt - timedelta(days=days)
    rows, seen = [], set()
    for e in raw.values():
        if not isinstance(e, dict):
            continue
        a = e.get("_article", {}) or {}
        h, ca, body = a.get("headline"), a.get("created_at"), a.get("summary", "")
        if not h or not ca:
            continue
        try:
            dt = datetime.fromisoformat(str(ca).replace("Z", "+00:00"))
            dt = dt.replace(tzinfo=timezone.utc) if dt.tzinfo is None else dt.astimezone(timezone.utc)
        except Exception:
            continue
        if not (start <= dt <= end_dt):
            continue
        ck = cache_key(h, body)
        if ck in seen:
            continue
        u = sc.get(f"{prompt}:{ck}")
        if not isinstance(u, dict) or u.get("sentiment") != "bullish":
            continue
        tks = [t for t in (u.get("tickers") or []) if isinstance(t, str) and t not in ("BTC", "ETH")]
        if want_ticker and not tks:
            continue
        if not want_ticker and tks:
            continue
        seen.add(ck)
        rows.append({"dt": dt, "headline": h, "magnitude": float(u.get("magnitude", 0) or 0),
                     "confidence": float(u.get("confidence", 0) or 0), "tickers": tks})
    rows.sort(key=lambda r: r["dt"])
    return rows


def stat(trades):
    if not trades:
        return dict(n=0, total=0.0, mean=0.0, sharpe=0.0, win=0.0, x2=0, pnls=[])
    s = compute_stats(trades)
    return dict(n=s["trades"], total=s["total_pnl"], mean=s["total_pnl"] / s["trades"],
                sharpe=s["sharpe"], win=s["win_rate"],
                x2=sum(1 for t in trades if t.get("pnl_pct", 0) >= 100),
                pnls=[t["pnl_usd"] for t in trades])


def pctl(xs, p):
    xs = sorted(xs); return xs[max(0, min(len(xs) - 1, int(round(p / 100 * (len(xs) - 1)))))]


# ── per-leg trade simulators (return list of trade dicts for a given trail, hold) ──

def sim_stock(rows, reg, trail, hold, gate, earnings=False):
    save = (_sb.STOCK_TRAIL, _sb.MAX_HOLD)
    _sb.STOCK_TRAIL, _sb.MAX_HOLD = trail, hold
    trades, seen = [], {}
    try:
        for r in rows:
            if not gate(r["magnitude"], r["confidence"]):
                continue
            if earnings and not _pb.is_earnings_article(r["headline"]):
                continue
            d = r["dt"].date()
            if reg and not reg(d):
                continue
            ds = seen.setdefault(d.isoformat(), set())
            for tk in r["tickers"][:2]:
                if tk in ds or not is_valid_stock_ticker(tk):
                    continue
                ds.add(tk)
                t = _sb.simulate_stock(tk, r["dt"], _scale(r["magnitude"], r["confidence"], cfg.MAX_POSITION_USD))
                if t:
                    trades.append(t)
    finally:
        _sb.STOCK_TRAIL, _sb.MAX_HOLD = save
    return trades


def sim_pead(rows, reg, trail, hold, gate):
    trades, seen = [], {}
    for r in rows:
        if not gate(r["magnitude"], r["confidence"]) or not _pb.is_earnings_article(r["headline"]):
            continue
        d = r["dt"].date()
        if reg and not reg(d):
            continue
        ds = seen.setdefault(d.isoformat(), set())
        for tk in r["tickers"][:2]:
            if tk in ds or not is_valid_stock_ticker(tk):
                continue
            ds.add(tk)
            t = _pb.simulate_pead(tk, r["dt"], _scale(r["magnitude"], r["confidence"], cfg.MAX_POSITION_USD),
                                  trail=trail, max_hold=hold)
            if t:
                trades.append(t)
    return trades


def sim_option(rows, reg, trail, hold, gate, delta, dte, base_usd, exit_rule="trail_premium",
               exit_params=None, qqq=False):
    save = {k: getattr(_bt, k) for k in
            ("TARGET_DELTA", "DTE_TARGET", "MAX_HOLD_DAYS", "TRAILING_STOP_PCT", "EXIT_PARAMS")}
    _bt.TARGET_DELTA, _bt.DTE_TARGET, _bt.MAX_HOLD_DAYS = delta, dte, hold
    _bt.TRAILING_STOP_PCT = trail
    if exit_params is not None:
        _bt.EXIT_PARAMS = exit_params
    trades, seen = [], set()
    try:
        for r in rows:
            if not gate(r["magnitude"], r["confidence"]):
                continue
            d = r["dt"].date()
            if reg and not reg(d):
                continue
            tickers = ["QQQ"] if qqq else r["tickers"][:2]
            for tk in tickers:
                if not is_valid_stock_ticker(tk):
                    continue
                key = f"{d}_{tk}"
                if key in seen:
                    continue
                seen.add(key)
                sp = _bt.get_price_at(tk, r["dt"])
                if not sp:
                    continue
                t = _bt.simulate_option_pnl(tk, r["dt"], sp, _scale(r["magnitude"], r["confidence"], base_usd),
                                            {"magnitude": r["magnitude"], "confidence": r["confidence"]},
                                            option_type="call", exit_rule=exit_rule, spread_mult=1.0)
                if t:
                    trades.append(t)
    finally:
        for k, v in save.items():
            setattr(_bt, k, v)
    return trades


# ── leg specs: (name, sim_fn(rows,reg,trail,hold), TRAILS, HOLDS, gate, want_ticker, current(trail,hold,label,fn)) ──
INF = float("inf")

def leg_specs():
    lotto_gate = lambda m, c: m >= cfg.LOTTO_MIN_MAGNITUDE and c >= cfg.LOTTO_MIN_CONFIDENCE
    specs = {}
    specs["stock"] = dict(
        sim=lambda rows, reg, tr, h: sim_stock(rows, reg, tr, h, dyn_gate),
        trails=[0.06, 0.08, 0.10, 0.12, 0.15], holds=[1, 2, 3, 5, 10, 20],
        want_ticker=True, cur=(0.10, 2, "live 10%/2d"),
        cur_sim=lambda rows, reg: sim_stock(rows, reg, 0.10, 2, dyn_gate))
    specs["news_call"] = dict(
        sim=lambda rows, reg, tr, h: sim_option(rows, reg, tr, h,
            lambda m, c: m >= cfg.NEWS_CALL_MIN_MAGNITUDE and dyn_gate(m, c),
            cfg.NEWS_CALL_TARGET_DELTA, 10, cfg.MAX_POSITION_USD),
        trails=[0.15, 0.20, 0.30, 0.40, 0.50], holds=[3, 5, 7, 10],
        want_ticker=True, cur=(None, 10, "live tiered EXIT_TIERS/DTE-hold"),
        cur_sim=lambda rows, reg: sim_option(rows, reg, 0.30, 10,
            lambda m, c: m >= cfg.NEWS_CALL_MIN_MAGNITUDE and dyn_gate(m, c),
            cfg.NEWS_CALL_TARGET_DELTA, 10, cfg.MAX_POSITION_USD,
            exit_rule="tiered_trail", exit_params={"tiers": list(cfg.EXIT_TIERS)}))
    specs["lotto"] = dict(
        sim=lambda rows, reg, tr, h: sim_option(rows, reg, tr, h, lotto_gate,
            cfg.LOTTO_TARGET_DELTA, 14, cfg.LOTTO_POSITION_USD),
        trails=[0.20, 0.30, 0.40, 0.50], holds=[2, 3, 5, 7],
        want_ticker=True, cur=(None, 3, "live tiered+3× cap/3d"),
        cur_sim=lambda rows, reg: sim_option(rows, reg, 0.30, 3, lotto_gate,
            cfg.LOTTO_TARGET_DELTA, 14, cfg.LOTTO_POSITION_USD,
            exit_rule="tiered_profit", exit_params={"tiers": list(cfg.LOTTO_EXIT_TIERS), "hard_target": 3.0}))
    specs["qqq_macro"] = dict(
        sim=lambda rows, reg, tr, h: sim_option(rows, reg, tr, h, dyn_gate,
            cfg.QQQ_MACRO_TARGET_DELTA, 10, cfg.MAX_POSITION_USD, qqq=True),
        trails=[0.15, 0.20, 0.30, 0.40], holds=[3, 5, 7, 10],
        want_ticker=False, cur=(0.30, 10, "live tiered/no cap"),
        cur_sim=lambda rows, reg: sim_option(rows, reg, 0.30, 10, dyn_gate,
            cfg.QQQ_MACRO_TARGET_DELTA, 10, cfg.MAX_POSITION_USD, qqq=True,
            exit_rule="tiered_trail", exit_params={"tiers": list(cfg.EXIT_TIERS)}))
    specs["pead"] = dict(
        sim=lambda rows, reg, tr, h: sim_pead(rows, reg, tr, h, dyn_gate),
        trails=[0.08, 0.10, 0.12, 0.15, 0.20], holds=[10, 20, 30, 45],
        want_ticker=True, cur=(0.10, 45, "live ~0.10 tiered/45d"),
        cur_sim=lambda rows, reg: sim_pead(rows, reg, 0.10, 45, dyn_gate))
    return specs


def run_leg(name, spec):
    blabel, bend, bdays, bcache = BEAR
    _, end_dt, days, cache = BULL
    rows  = load_signals(cache, end_dt, days, PROMPT, spec["want_ticker"])
    brows = load_signals(bcache, bend, bdays, PROMPT, spec["want_ticker"])
    reg, breg = build_regime(end_dt, days, 200), build_regime(bend, bdays, 200)
    print(f"\n{'='*78}\n██ LEG: {name}  ({len(rows)} signals, NET of costs) ██")
    cur = stat(spec["cur_sim"](rows, reg))
    ct, ch, cl = spec["cur"]
    print(f"  CURRENT exit [{cl}]: n={cur['n']} total=${cur['total']:,.0f} ${cur['mean']:+,.0f}/tr "
          f"Sh{cur['sharpe']:.2f} win{cur['win']:.0f}%")
    print(f"\n  total net P&L / Sharpe  (rows=trail, cols=max-hold days):")
    print(f"  {'trail \\ hold':>12}" + "".join(f"{h:>13}" for h in spec["holds"]))
    grid = {}
    for tr in spec["trails"]:
        cells = []
        for h in spec["holds"]:
            s = stat(spec["sim"](rows, reg, tr, h))
            grid[(tr, h)] = s
            cells.append(f"${s['total']/1000:>4.0f}k/{s['sharpe']:>4.2f}")
        print(f"  {('%.0f%%' % (tr*100)):>12}" + "".join(f"{c:>13}" for c in cells), flush=True)
    elig = [(k, s) for k, s in grid.items() if s["n"] >= 30 and s["total"] > 0]
    if not elig:
        print("  → no trail/hold cell net-positive with ≥30 trades."); return
    (bt_, bh), best = max(elig, key=lambda kv: kv[1]["sharpe"])
    pnls = best["pnls"]; B = sorted(sum(random.choices(pnls, k=len(pnls))) for _ in range(N_BOOT))
    lo, hi = B[int(.025*N_BOOT)], B[int(.975*N_BOOT)]
    bb = stat(spec["sim"](brows, breg, bt_, bh))
    dpnl, dsh = best["total"]-cur["total"], best["sharpe"]-cur["sharpe"]
    print(f"\n  BEST-Sharpe cell: trail {bt_*100:.0f}% / hold {bh}d → n={best['n']} ${best['total']:,.0f} "
          f"${best['mean']:+,.0f}/tr Sh{best['sharpe']:.2f} win{best['win']:.0f}%")
    print(f"    bootstrap total P&L CI [${lo:,.0f}, ${hi:,.0f}] P>0={sum(1 for b in B if b>0)/N_BOOT*100:.0f}%")
    print(f"    vs current: P&L {dpnl:+,.0f} / Sharpe {dsh:+.2f}   |   bear guard: n={bb['n']} ${bb['total']:,.0f} Sh{bb['sharpe']:.2f}")
    print(f"    → {'CHANGE candidate (beats current + bear-ok)' if dpnl>0 and dsh>=0 and bb['total']>=cur['total']*0.0 else 'prefer current / not clearly better — inspect grid'}")


def main():
    specs = leg_specs()
    legs = [ONLY] if ONLY else list(specs.keys())
    print(f"EXIT SWEEP (trail × max-hold) on {PROMPT} — legs: {', '.join(legs)}")
    for name in legs:
        run_leg(name, specs[name])


if __name__ == "__main__":
    main()
