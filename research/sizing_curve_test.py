"""
sizing_curve_test.py — is our position-sizing curve optimal?

Live sizing is LINEAR: position_usd = MAX_POSITION_USD × magnitude × confidence (floored at
10%). We never validated that shape. This tests several sizing curves on the STOCK leg (the
always-on workhorse) at the live 2-day hold, holding TOTAL deployed capital constant so it's
apples-to-apples — only the DISTRIBUTION of capital across signals changes.

The whole thing hinges on one question: does realized return actually rise with mag×conf?
  • If yes  → a convex curve (bet more on the top signals) beats flat.
  • If no   → flat (equal $) is best and our linear curve is just adding variance.
So we first print the return-by-conviction-quintile + rank correlation (the diagnostic), then
the P&L/Sharpe of each curve.

Usage:  USE_YAHOO_BARS=1 .venv/bin/python sizing_curve_test.py
"""
import os, statistics
os.environ.setdefault("USE_YAHOO_BARS", "1")
from datetime import datetime, timezone, timedelta
from pathlib import Path

import stock_backtest as sb
import tune_v2, config as cfg
from regime_filter import build_regime

sb.MAX_HOLD = cfg.NEWS_STOCK_MAX_HOLD_DAYS        # live 2-day hold (was 30 in stock_backtest)
BASE = 1000.0                                     # avg $/trade (normalised so curves match)

WINDOWS = [
    ("bull-meltup", datetime.now(timezone.utc) - timedelta(days=5), 180, "dual_score_cache.json"),
    ("2022-bear",   datetime(2022, 6, 30, tzinfo=timezone.utc), 90,  "bear_dual_cache.json"),
]


def gate(mag, conf):
    return mag >= cfg.MIN_MAGNITUDE and conf >= cfg.BASE_CONFIDENCE + (1 - mag) * cfg.CONFIDENCE_SLOPE


def collect(rows, regime):
    """Per-signal realized 2-day stock return + its mag/conf. Return list of (mag, conf, ret_pct)."""
    out, seen = [], {}
    for r in rows:
        m, c = r["magnitude"], r["confidence"]
        if not gate(m, c):
            continue
        d = r["created_at"].date()
        if regime and not regime(d):
            continue
        ds = seen.setdefault(d.isoformat(), set())
        for tk in r["tickers"][:2]:
            if tk in ds or tk in ("BTC", "ETH") or not sb.is_valid_stock_ticker(tk):
                continue
            ds.add(tk)
            try:
                t = sb.simulate_stock(tk, r["created_at"], BASE)
            except Exception:
                continue                      # one bad ticker/data row must not kill the run
            if t:
                out.append((m, c, t["pnl_pct"]))
    return out


def spearman(xs, ys):
    def rank(v):
        order = sorted(range(len(v)), key=lambda i: v[i])
        rk = [0.0] * len(v)
        for pos, i in enumerate(order):
            rk[i] = pos
        return rk
    rx, ry = rank(xs), rank(ys)
    n = len(xs)
    mx, my = sum(rx) / n, sum(ry) / n
    cov = sum((a - mx) * (b - my) for a, b in zip(rx, ry))
    sx = (sum((a - mx) ** 2 for a in rx)) ** 0.5
    sy = (sum((b - my) ** 2 for b in ry)) ** 0.5
    return cov / (sx * sy) if sx and sy else 0.0


def portfolio(weights, rets):
    """Normalise weights to mean 1 (→ same total capital = BASE×N), return (total_pnl, sharpe)."""
    mw = sum(weights) / len(weights)
    wn = [w / mw for w in weights]
    pnl = [BASE * w * (r / 100.0) for w, r in zip(wn, rets)]
    tot = sum(pnl)
    sd = statistics.pstdev(pnl)
    sharpe = (statistics.mean(pnl) / sd) if sd else 0.0
    return tot, sharpe


def quintile_diag(data):
    """Return-by-conviction-quintile: does ret rise with mag×conf?"""
    ranked = sorted(data, key=lambda x: x[0] * x[1])
    n = len(ranked)
    print(f"  conviction (mag×conf) quintiles — does return rise with conviction?")
    for q in range(5):
        chunk = ranked[q * n // 5:(q + 1) * n // 5]
        rets = [r for _, _, r in chunk]
        mc = [m * c for m, c, _ in chunk]
        win = sum(1 for r in rets if r > 0) / len(rets) * 100
        print(f"    Q{q+1} (mag×conf {min(mc):.2f}-{max(mc):.2f})  n={len(chunk):>4}  "
              f"mean_ret={statistics.mean(rets):>+5.2f}%  win={win:>4.1f}%")
    rho = spearman([m * c for m, c, _ in data], [r for _, _, r in data])
    print(f"  Spearman rank corr( mag×conf , 2-day return ) = {rho:+.3f}   "
          f"({'concentrate' if rho > 0.03 else 'go flat' if rho < -0.03 else 'weak/none'})")


def main():
    print(f"═══ SIZING-CURVE TEST — STOCK leg, {sb.MAX_HOLD}d hold, equal total capital ═══\n")
    for label, end_dt, days, cache in WINDOWS:
        if not Path(cache).exists():
            print(f"[{label}] cache missing — skip"); continue
        scored = tune_v2.load_scored_from_dual_cache(Path(cache), "bull", end_dt, days)
        reg = build_regime(end_dt, days, 200)
        data = collect(scored, reg)
        print(f"═══ {label} ═══  trades={len(data)}")
        if len(data) < 30:
            print("  (too few)\n"); continue
        quintile_diag(data)
        rets = [r for _, _, r in data]
        med = sorted([m * c for m, c, _ in data])[len(data) // 2]
        curves = {
            "flat (equal $)":          [1.0 for _ in data],
            "linear mag×conf (LIVE)":  [max(0.10, m * c) for m, c, _ in data],
            "convex (mag×conf)^2":     [max(0.01, (m * c) ** 2) for m, c, _ in data],
            "steep (mag×conf)^3":      [max(0.001, (m * c) ** 3) for m, c, _ in data],
            "top-half only (2× each)": [1.0 if m * c >= med else 0.0 for m, c, _ in data],
        }
        print(f"  {'curve':26} {'total P&L':>12} {'per-trade Sharpe':>18}")
        for name, w in curves.items():
            if sum(w) == 0:
                continue
            tot, sh = portfolio(w, rets)
            print(f"  {name:26} {f'${tot:+,.0f}':>12} {sh:>18.3f}")
        print()
    print("Read: if Sharpe/PnL rise from flat→convex AND the quintile returns climb with")
    print("conviction, concentrate capital on the top signals. If flat wins, the model's")
    print("mag×conf does NOT rank dollar outcomes → equal-weight and stop adding variance.")


if __name__ == "__main__":
    main()
