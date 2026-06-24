"""
stock_selectivity_sweep.py — sweep the STOCK strategy's entry gate (magnitude × confidence)
to make it MORE SELECTIVE. The stock leg is the always-on, loosest-gate entry (mag≥0.35 +
dynamic conf floor) — it fires on nearly every bullish signal first-come-first-served and fills
the 30-position cap with mediocre names early in the day. That gate was NEVER swept (stock_backtest
validated the stock-vs-option approach + regime gate, but inherited the generic threshold).

Because the 30-slot cap BINDS, the right objective is PER-TRADE QUALITY (mean $/trade, Sharpe,
win-rate) — a tighter gate that raises quality-per-slot beats a loose one that floods the cap.
Total P&L mechanically falls as the gate tightens (fewer trades) and is NOT the cap-aware metric.

Reuses stock_backtest.simulate_stock + tune_v2 signal loading. Cached signals + stock bars only
(NO ollama / Alpaca) → safe to run alongside the live bot. Bootstraps the winner vs current.

Usage:  USE_YAHOO_BARS=1 .venv/bin/python -u stock_selectivity_sweep.py
"""
import os, json, random, statistics
os.environ.setdefault("USE_YAHOO_BARS", "1")
from datetime import datetime, timezone, timedelta
from pathlib import Path

import tune_v2
import config as _cfg
from benchmark import compute_stats
from regime_filter import build_regime
from backtest import is_valid_stock_ticker, cache_key
from stock_backtest import simulate_stock

# When UNI_PROMPT is set (e.g. "unified_v1"), load the UNIFIED prompt's scores from
# unified_scores.json (joined to article metadata) instead of the retired two-pass
# dual_score_cache. This recalibrates the stock gate on the LIVE prompt's own score
# distribution. Unset → original behavior (old dual-cache scores).
UNI_PROMPT = os.environ.get("UNI_PROMPT", "")


def load_scored_from_unified(cache_file, end_dt, days, prompt):
    """Join unified_scores.json["<prompt>:<ck>"] to the article metadata in cache_file,
    yielding rows in the SAME shape as tune_v2.load_scored_from_dual_cache (bullish side,
    has-ticker, in-window). The unified score dict supplies magnitude/confidence/tickers/
    sentiment; created_at comes from the article. cache_file is only the metadata source
    (same one score_unified.py scored against), so the cache_key join lines up exactly."""
    sc = json.loads(Path("unified_scores.json").read_text())
    raw = json.loads(Path(cache_file).read_text())
    start_dt = end_dt - timedelta(days=days)
    rows, seen = [], set()
    for entry in raw.values():
        if not isinstance(entry, dict):
            continue
        a = entry.get("_article", {}) or {}
        h, ca, body = a.get("headline"), a.get("created_at"), a.get("summary", "")
        if not h or not ca:
            continue
        try:
            dt = datetime.fromisoformat(str(ca).replace("Z", "+00:00"))
            if dt.tzinfo is None:
                dt = dt.replace(tzinfo=timezone.utc)
            dt = dt.astimezone(timezone.utc)
        except Exception:
            continue
        if not (start_dt <= dt <= end_dt):
            continue
        ck = cache_key(h, body)
        if ck in seen:
            continue
        u = sc.get(f"{prompt}:{ck}")
        if not isinstance(u, dict) or u.get("sentiment") != "bullish":
            continue
        tickers = [t for t in (u.get("tickers") or []) if isinstance(t, str)]
        if not tickers:
            continue
        seen.add(ck)
        rows.append({
            "headline":   h,
            "created_at": dt,
            "magnitude":  float(u.get("magnitude", 0.0) or 0),
            "confidence": float(u.get("confidence", 0.0) or 0),
            "tickers":    tickers,
        })
    rows.sort(key=lambda x: x["created_at"])
    return rows


def load_window(cache_file, end_dt, days):
    """Unified loader when UNI_PROMPT is set, else the original dual-cache loader."""
    if UNI_PROMPT:
        return load_scored_from_unified(cache_file, end_dt, days, UNI_PROMPT)
    return tune_v2.load_scored_from_dual_cache(Path(cache_file), "bull", end_dt, days)

BULL = ("bull-meltup", datetime.now(timezone.utc) - timedelta(days=5), 180, "dual_score_cache.json")
BEAR = ("2022-bear",   datetime(2022, 6, 30, tzinfo=timezone.utc),     90,  "bear_dual_cache.json")
MAG_FLOORS  = [0.35, 0.45, 0.55, 0.65, 0.75]
CONF_FLOORS = [0.70, 0.80, 0.85, 0.90]
MIN_TRADES  = 50          # keep enough volume to keep the 30-slot cap fed over the window
N_BOOT = 10000
random.seed(20260612)


def scale(mag, conf):
    return max(_cfg.MAX_POSITION_USD * 0.10, _cfg.MAX_POSITION_USD * mag * conf)


def run_gate(scored, regime, gate):
    """gate(mag,conf)->bool. Returns list of stock trades (1 per ticker/day, ≤2 tickers/signal)."""
    trades, seen = [], {}
    for row in scored:
        mag, conf = row["magnitude"], row["confidence"]
        if not gate(mag, conf):
            continue
        d = row["created_at"].date()
        if regime is not None and not regime(d):
            continue
        ds = seen.setdefault(d.isoformat(), set())
        for tk in row["tickers"][:2]:
            if tk in ds or tk in ("BTC", "ETH") or not is_valid_stock_ticker(tk):
                continue
            ds.add(tk)
            t = simulate_stock(tk, row["created_at"], scale(mag, conf))
            if t:
                trades.append(t)
    return trades


def current_gate(mag, conf):
    return mag >= _cfg.MIN_MAGNITUDE and conf >= _cfg.BASE_CONFIDENCE + (1 - mag) * _cfg.CONFIDENCE_SLOPE


def stats_row(trades):
    if not trades:
        return dict(n=0, total=0.0, mean=0.0, sharpe=0.0, win=0.0)
    s = compute_stats(trades)
    return dict(n=s["trades"], total=s["total_pnl"], mean=s["total_pnl"] / s["trades"],
                sharpe=s["sharpe"], win=s["win_rate"], pnls=[t["pnl_usd"] for t in trades])


def pctl(xs, p):
    xs = sorted(xs); i = max(0, min(len(xs) - 1, int(round(p / 100 * (len(xs) - 1)))))
    return xs[i]


def main():
    label, end_dt, days, cache = BULL
    scored = load_window(cache, end_dt, days)
    reg = build_regime(end_dt, days, 200)
    srcdesc = f"unified_scores.json[{UNI_PROMPT}]" if UNI_PROMPT else f"dual_score_cache[{cache}]"
    print(f"STOCK selectivity sweep — {label}, {len(scored)} scored signals from {srcdesc}, "
          f"SPY>200d regime-gated, sized like live, 10% trail ≤30d\n")

    cur = stats_row(run_gate(scored, reg, current_gate))
    print(f"  CURRENT live gate (mag≥{_cfg.MIN_MAGNITUDE} & conf≥dyn): "
          f"n={cur['n']}  total=${cur['total']:,.0f}  mean=${cur['mean']:+,.0f}/tr  "
          f"Sharpe={cur['sharpe']:.2f}  win={cur['win']:.1f}%\n")

    grid = {}
    print(f"  {'gate':>16} {'n':>5} {'total':>11} {'mean/tr':>9} {'Sharpe':>7} {'win%':>6}")
    for mf in MAG_FLOORS:
        for cf in CONF_FLOORS:
            r = stats_row(run_gate(scored, reg, lambda m, c, mf=mf, cf=cf: m >= mf and c >= cf))
            grid[(mf, cf)] = r
            flag = "  <cap-starved" if 0 < r["n"] < MIN_TRADES else ""
            print(f"  mag≥{mf} c≥{cf:.2f}  {r['n']:>5} {('${:,.0f}'.format(r['total'])):>11} "
                  f"{('${:+,.0f}'.format(r['mean'])):>9} {r['sharpe']:>7.2f} {r['win']:>6.1f}{flag}", flush=True)

    # Candidate = best Sharpe among gates that (a) keep ≥MIN_TRADES and (b) beat current mean $/trade
    elig = [(k, r) for k, r in grid.items()
            if r["n"] >= MIN_TRADES and r["mean"] > cur["mean"] and r["sharpe"] >= cur["sharpe"]]
    if not elig:
        print("\n  No gate beats current on mean $/trade AND Sharpe while keeping the cap fed → "
              "current gate holds (or selectivity gains too few trades). Inspect grid above.")
        return
    cand_key, cand = max(elig, key=lambda kv: kv[1]["sharpe"])
    print(f"\n  CANDIDATE: mag≥{cand_key[0]} & conf≥{cand_key[1]:.2f} → "
          f"n={cand['n']}  mean=${cand['mean']:+,.0f}/tr (vs ${cur['mean']:+,.0f})  "
          f"Sharpe={cand['sharpe']:.2f} (vs {cur['sharpe']:.2f})  win={cand['win']:.1f}% (vs {cur['win']:.1f}%)\n")

    # ── Bootstrap per-trade mean (unpaired: different signal sets) ──
    cb = [statistics.mean(random.choices(cand["pnls"], k=len(cand["pnls"]))) for _ in range(N_BOOT)]
    ub = [statistics.mean(random.choices(cur["pnls"],  k=len(cur["pnls"])))  for _ in range(N_BOOT)]
    diff = [a - b for a, b in zip(cb, ub)]
    lo, hi = pctl(diff, 2.5), pctl(diff, 97.5)
    frac = sum(1 for d in diff if d > 0) / N_BOOT
    print(f"  Bootstrap mean $/trade (candidate − current): ${cand['mean']-cur['mean']:+,.0f}  "
          f"95% CI [${lo:+,.0f}, ${hi:+,.0f}]   P(candidate better/trade) = {frac*100:.1f}%")

    # ── Bear-window regime guard: does the tighter gate also behave in a bear? ──
    blabel, bend, bdays, bcache = BEAR
    bscored = load_window(bcache, bend, bdays)
    breg = build_regime(bend, bdays, 200)
    bcur = stats_row(run_gate(bscored, breg, current_gate))
    bcand = stats_row(run_gate(bscored, breg, lambda m, c: m >= cand_key[0] and c >= cand_key[1]))
    print(f"\n  Bear-window guard ({blabel}, regime-gated): "
          f"current n={bcur['n']} ${bcur['total']:,.0f}/Sh{bcur['sharpe']:.2f}  |  "
          f"candidate n={bcand['n']} ${bcand['total']:,.0f}/Sh{bcand['sharpe']:.2f}")

    robust = lo > 0 and frac >= 0.95
    print(f"\n  VERDICT: {'ROBUST selectivity gain → worth tightening' if robust else 'marginal / not robust → discuss before changing'}")


if __name__ == "__main__":
    main()
