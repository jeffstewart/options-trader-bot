"""
anthropic_backtest.py — P&L backtest for Anthropic model scores vs Ollama baseline,
with a confidence threshold sweep to find the sweet spot for each model.

Usage (run from data/ directory):
    USE_YAHOO_BARS=1 ../.venv/bin/python ../research/anthropic_backtest.py

    # Report only (no new scoring):
    USE_YAHOO_BARS=1 ../.venv/bin/python ../research/anthropic_backtest.py --report-only

    # Sweep only (skip the fixed-threshold A/B comparison):
    USE_YAHOO_BARS=1 ../.venv/bin/python ../research/anthropic_backtest.py --sweep-only

Outputs to stdout. Run with | tee anthropic_backtest_report.txt to save.
"""
import os, json, statistics, argparse
os.environ.setdefault("USE_YAHOO_BARS", "1")

import gemini_lotto_pnl as gl

# ── Config ─────────────────────────────────────────────────────────────────────
ANTHROPIC_CACHE = "anthropic_backtest_cache.json"
UNIFIED_SCORES  = "unified_scores.json"
CANDPOOL        = "yahoo_candpool_cache.json"
HORIZON         = "r3"    # 3-day forward return — matches groq_vs_ollama_backtest.py
OLLAMA_MAG      = 0.75    # Ollama's current live gate

# Models in Anthropic cache to evaluate
ANTHROPIC_MODELS = ["haiku", "sonnet5"]

# Confidence sweep range
CONF_SWEEP  = [round(x * 0.05, 2) for x in range(0, 21)]   # 0.00 to 1.00 in 0.05 steps
# Magnitude sweep range (for Anthropic models which score lower than Ollama)
MAG_SWEEP   = [round(x * 0.05, 2) for x in range(0, 21)]

# ── Data loading ───────────────────────────────────────────────────────────────

def load_universe():
    """Same universe as groq_vs_ollama_backtest.py — Ollama-scored articles with forward returns."""
    uni = json.load(open(UNIFIED_SCORES))
    cp  = json.load(open(CANDPOOL))
    gl.SAMPLE = 100000
    out = []
    for c in gl.candidates():
        u  = uni.get("unified_v1:" + c["ck"])
        fr = cp.get(f"{c['tk']}_{c['ck']}")
        if isinstance(u, dict) and fr and fr.get("px", 0) >= 5 and c.get("h"):
            out.append((c, u, fr))
    out.sort(key=lambda x: x[0]["ck"])
    return out

def load_anthropic_scores() -> dict:
    if not os.path.exists(ANTHROPIC_CACHE):
        raise FileNotFoundError(f"{ANTHROPIC_CACHE} not found — run anthropic_scorer.py first")
    return json.load(open(ANTHROPIC_CACHE))

# ── Trade decision helpers ─────────────────────────────────────────────────────

def ollama_trades(u) -> bool:
    """Ollama baseline: bullish AND magnitude >= 0.75."""
    return (u.get("sentiment") == "bullish"
            and float(u.get("magnitude", 0) or 0) >= OLLAMA_MAG)

def anthropic_trades(score, min_mag: float = 0.0, min_conf: float = 0.0) -> bool:
    """Anthropic model: bullish AND magnitude >= min_mag AND confidence >= min_conf."""
    if not score or not isinstance(score, dict):
        return False
    return (score.get("sent") == "bullish"
            and float(score.get("mag", 0) or 0) >= min_mag
            and float(score.get("conf", 0) or 0) >= min_conf)

# ── Stats helper ───────────────────────────────────────────────────────────────

def stat(name: str, xs: list, indent: int = 2) -> dict:
    pad = " " * indent
    if not xs:
        print(f"{pad}{name:42} n=0")
        return {}
    n   = len(xs)
    tot = sum(xs)
    avg = tot / n
    sd  = statistics.pstdev(xs) if n > 1 else 0
    sh  = avg / sd if sd else 0
    win = sum(1 for x in xs if x > 0) / n * 100
    print(f"{pad}{name:42} n={n:<4} Σ{tot:>+8.1f}%  avg{avg:>+6.2f}%  win{win:>4.0f}%  sharpe{sh:>+5.2f}")
    return {"n": n, "total": tot, "avg": avg, "win_rate": win, "sharpe": sh}

# ══════════════════════════════════════════════════════════════════════════════
# SECTION A — Fixed threshold comparison (mirrors groq_vs_ollama_backtest.py)
# ══════════════════════════════════════════════════════════════════════════════

def report_fixed(univ, sc):
    """A/B comparison at fixed thresholds — shows calibration offset vs Ollama."""
    o_pnl = [fr[HORIZON] for c, u, fr in univ if ollama_trades(u)]

    print(f"\n  ══ A) FIXED THRESHOLD COMPARISON ══")
    print(f"  universe {len(univ)} articles · Ollama gate: bullish & mag≥{OLLAMA_MAG}")
    print(f"  Anthropic gate: bullish & mag≥0.45 & conf≥0.60  (current bot thresholds)")
    print()
    stat("OLLAMA baseline (mag≥0.75)", o_pnl)

    for model in ANTHROPIC_MODELS:
        scored = [(c, u, fr) for c, u, fr in univ if sc.get(f"{model}:{c['ck']}")]
        cov    = len(scored)
        if not scored:
            print(f"\n  [{model}] no scores found")
            continue

        # Use current bot thresholds (MIN_MAGNITUDE=0.45, BASE_CONFIDENCE=0.60)
        a_pnl  = [fr[HORIZON] for c, u, fr in scored
                  if anthropic_trades(sc[f"{model}:{c['ck']}"], min_mag=0.45, min_conf=0.60)]
        o_only = [fr[HORIZON] for c, u, fr in scored
                  if ollama_trades(u) and not anthropic_trades(sc[f"{model}:{c['ck']}"], 0.45, 0.60)]
        a_only = [fr[HORIZON] for c, u, fr in scored
                  if anthropic_trades(sc[f"{model}:{c['ck']}"], 0.45, 0.60) and not ollama_trades(u)]

        print(f"\n  [{model}]  coverage {cov}/{len(univ)} ({cov/len(univ)*100:.0f}%)")
        stat(f"{model} (mag≥0.45, conf≥0.60)", a_pnl)
        stat(f"  └ {model}-only (Ollama skips → MISSED?)", a_only, indent=4)
        stat(f"  └ Ollama-only ({model} skips → AVOIDED?)", o_only, indent=4)

# ══════════════════════════════════════════════════════════════════════════════
# SECTION B — Matched selectivity (same N trades, whose picks are better?)
# ══════════════════════════════════════════════════════════════════════════════

def report_matched(univ, sc):
    print(f"\n  ══ B) MATCHED SELECTIVITY (top-N by magnitude, N = Ollama trade count) ══")
    print(f"     Fair pick-quality test — equal trade count, best avg return wins.")

    for model in ANTHROPIC_MODELS:
        scored = [(c, u, fr) for c, u, fr in univ if sc.get(f"{model}:{c['ck']}")]
        if not scored:
            continue
        n       = sum(1 for c, u, fr in scored if ollama_trades(u))
        o_match = [fr[HORIZON] for c, u, fr in scored if ollama_trades(u)]
        a_bull  = [(c, u, fr) for c, u, fr in scored
                   if sc[f"{model}:{c['ck']}"] and sc[f"{model}:{c['ck']}"].get("sent") == "bullish"]
        a_bull.sort(key=lambda x: float(sc[f"{model}:{x[0]['ck']}"].get("mag", 0) or 0), reverse=True)
        take    = min(n, len(a_bull))
        a_top   = [fr[HORIZON] for c, u, fr in a_bull[:take]]
        thr     = float(sc[f"{model}:{a_bull[take-1][0]['ck']}"].get("mag", 0)) if take > 0 else 0
        note    = f"≈mag≥{thr:.2f}" if len(a_bull) >= n else f"only {len(a_bull)} bullish picks vs N={n}"

        print(f"\n  [{model}]  matched N={n} (common scored set {len(scored)})")
        stat("  Ollama top-N (mag≥0.75)", o_match, indent=4)
        stat(f"  {model} top-N ({note})", a_top, indent=4)

# ══════════════════════════════════════════════════════════════════════════════
# SECTION C — Confidence threshold sweep
# ══════════════════════════════════════════════════════════════════════════════

def sweep_confidence(univ, sc):
    print(f"\n  ══ C) CONFIDENCE THRESHOLD SWEEP (mag≥0.40 fixed, conf swept) ══")
    print(f"     Find the confidence cutoff that maximises each metric.")
    print(f"     Ollama baseline: n={sum(1 for c,u,fr in univ if ollama_trades(u))}  "
          f"avg={sum(fr[HORIZON] for c,u,fr in univ if ollama_trades(u))/max(1,sum(1 for c,u,fr in univ if ollama_trades(u))):+.2f}%")

    for model in ANTHROPIC_MODELS:
        scored = [(c, u, fr) for c, u, fr in univ if sc.get(f"{model}:{c['ck']}")]
        if not scored:
            continue
        print(f"\n  [{model}]  (mag≥0.40 fixed)")
        print(f"  {'conf≥':>6}  {'n':>5}  {'total%':>8}  {'avg%':>7}  {'win%':>6}  {'sharpe':>7}")
        print(f"  {'-'*46}")

        best = {"avg": None, "total": None, "sharpe": None}
        rows = []
        for conf in CONF_SWEEP:
            xs = [fr[HORIZON] for c, u, fr in scored
                  if anthropic_trades(sc[f"{model}:{c['ck']}"], min_mag=0.40, min_conf=conf)]
            if not xs:
                rows.append((conf, 0, 0, 0, 0, 0))
                continue
            n   = len(xs)
            tot = sum(xs)
            avg = tot / n
            sd  = statistics.pstdev(xs) if n > 1 else 0
            sh  = avg / sd if sd else 0
            win = sum(1 for x in xs if x > 0) / n * 100
            rows.append((conf, n, tot, avg, win, sh))
            if best["avg"] is None or avg > best["avg"]:   best["avg"]    = conf
            if best["total"] is None or tot > best["total"]: best["total"] = conf
            if best["sharpe"] is None or sh > best["sharpe"]: best["sharpe"] = conf

        for conf, n, tot, avg, win, sh in rows:
            markers = []
            if conf == best["avg"]:    markers.append("← best avg/trade")
            if conf == best["total"]:  markers.append("← best total P&L")
            if conf == best["sharpe"]: markers.append("← best Sharpe")
            flag = "  " + " | ".join(markers) if markers else ""
            print(f"  {conf:>6.2f}  {n:>5}  {tot:>+8.1f}%  {avg:>+7.2f}%  {win:>5.0f}%  {sh:>+7.2f}{flag}")

# ══════════════════════════════════════════════════════════════════════════════
# SECTION D — Magnitude threshold sweep
# ══════════════════════════════════════════════════════════════════════════════

def sweep_magnitude(univ, sc):
    print(f"\n  ══ D) MAGNITUDE THRESHOLD SWEEP (conf≥0.60 fixed, mag swept) ══")
    print(f"     Find where the magnitude bar maximises quality without killing volume.")

    for model in ANTHROPIC_MODELS:
        scored = [(c, u, fr) for c, u, fr in univ if sc.get(f"{model}:{c['ck']}")]
        if not scored:
            continue
        print(f"\n  [{model}]  (conf≥0.60 fixed)")
        print(f"  {'mag≥':>6}  {'n':>5}  {'total%':>8}  {'avg%':>7}  {'win%':>6}  {'sharpe':>7}")
        print(f"  {'-'*46}")

        best = {"avg": None, "total": None, "sharpe": None}
        rows = []
        for mag in MAG_SWEEP:
            xs = [fr[HORIZON] for c, u, fr in scored
                  if anthropic_trades(sc[f"{model}:{c['ck']}"], min_mag=mag, min_conf=0.60)]
            if not xs:
                rows.append((mag, 0, 0, 0, 0, 0))
                continue
            n   = len(xs)
            tot = sum(xs)
            avg = tot / n
            sd  = statistics.pstdev(xs) if n > 1 else 0
            sh  = avg / sd if sd else 0
            win = sum(1 for x in xs if x > 0) / n * 100
            rows.append((mag, n, tot, avg, win, sh))
            if best["avg"] is None or avg > best["avg"]:     best["avg"]    = mag
            if best["total"] is None or tot > best["total"]: best["total"]  = mag
            if best["sharpe"] is None or sh > best["sharpe"]: best["sharpe"] = mag

        for mag, n, tot, avg, win, sh in rows:
            markers = []
            if mag == best["avg"]:    markers.append("← best avg/trade")
            if mag == best["total"]:  markers.append("← best total P&L")
            if mag == best["sharpe"]: markers.append("← best Sharpe")
            flag = "  " + " | ".join(markers) if markers else ""
            print(f"  {mag:>6.2f}  {n:>5}  {tot:>+8.1f}%  {avg:>+7.2f}%  {win:>5.0f}%  {sh:>+7.2f}{flag}")

# ══════════════════════════════════════════════════════════════════════════════
# SECTION E — 2D sweep: find optimal (mag, conf) pair
# ══════════════════════════════════════════════════════════════════════════════

def sweep_2d(univ, sc):
    print(f"\n  ══ E) 2D SWEEP — optimal (magnitude, confidence) pair per model ══")
    print(f"     Exhaustive grid search. Best cell per metric shown.")

    mag_steps  = [round(x * 0.05, 2) for x in range(4, 16)]   # 0.20 to 0.75
    conf_steps = [round(x * 0.05, 2) for x in range(4, 16)]   # 0.20 to 0.75

    for model in ANTHROPIC_MODELS:
        scored = [(c, u, fr) for c, u, fr in univ if sc.get(f"{model}:{c['ck']}")]
        if not scored:
            continue

        best_avg = best_total = best_sharpe = None
        best_avg_val = best_total_val = best_sharpe_val = float("-inf")

        for mag in mag_steps:
            for conf in conf_steps:
                xs = [fr[HORIZON] for c, u, fr in scored
                      if anthropic_trades(sc[f"{model}:{c['ck']}"], min_mag=mag, min_conf=conf)]
                if len(xs) < 10:   # need minimum sample for meaningful stats
                    continue
                tot = sum(xs)
                avg = tot / len(xs)
                sd  = statistics.pstdev(xs) if len(xs) > 1 else 0
                sh  = avg / sd if sd else 0
                if avg   > best_avg_val:    best_avg_val = avg;    best_avg    = (mag, conf, len(xs), avg, sh)
                if tot   > best_total_val:  best_total_val = tot;  best_total  = (mag, conf, len(xs), avg, sh)
                if sh    > best_sharpe_val: best_sharpe_val = sh;  best_sharpe = (mag, conf, len(xs), avg, sh)

        print(f"\n  [{model}]  (minimum 10 trades per cell)")
        for label, result in [("best avg/trade", best_avg),
                               ("best total P&L", best_total),
                               ("best Sharpe",    best_sharpe)]:
            if result:
                mag, conf, n, avg, sh = result
                print(f"    {label:18}  mag≥{mag:.2f}  conf≥{conf:.2f}  "
                      f"n={n}  avg={avg:+.2f}%  sharpe={sh:+.2f}")
            else:
                print(f"    {label:18}  insufficient data")

# ══════════════════════════════════════════════════════════════════════════════
# SECTION F — Bootstrap confidence intervals
# ══════════════════════════════════════════════════════════════════════════════

def bootstrap(univ, sc, n_boot=2000, seed=42):
    """
    Bootstrap resampling to get confidence intervals on avg/trade, total P&L,
    win rate, and Sharpe for each model at its optimal and current thresholds.
    Resamples the full universe WITH replacement n_boot times.
    """
    import random
    rng = random.Random(seed)

    print(f"\n  ══ F) BOOTSTRAP VALIDATION (n_boot={n_boot}, 95% CI) ══")
    print(f"     Tests whether sweep-optimal thresholds are stable or overfit.")
    print(f"     Wide CI relative to point estimate → overfit / insufficient data.")

    # Thresholds to validate: current bot, sweep-optimal per metric, and Ollama baseline
    CONFIGS = {
        "ollama":           lambda c, u, fr: ollama_trades(u),
        "haiku_current":    lambda c, u, fr: anthropic_trades(sc.get(f"haiku:{c['ck']}"),    0.45, 0.60),
        "haiku_best_avg":   lambda c, u, fr: anthropic_trades(sc.get(f"haiku:{c['ck']}"),    0.20, 0.75),
        "haiku_best_sharpe":lambda c, u, fr: anthropic_trades(sc.get(f"haiku:{c['ck']}"),    0.55, 0.20),
        "s5_current":       lambda c, u, fr: anthropic_trades(sc.get(f"sonnet5:{c['ck']}"),  0.45, 0.60),
        "s5_best_avg":      lambda c, u, fr: anthropic_trades(sc.get(f"sonnet5:{c['ck']}"),  0.45, 0.65),
        "s5_best_sharpe":   lambda c, u, fr: anthropic_trades(sc.get(f"sonnet5:{c['ck']}"),  0.20, 0.75),
    }

    def compute(rows):
        """Compute stats on a list of (traded, pnl) tuples."""
        xs = [pnl for traded, pnl in rows if traded]
        if len(xs) < 3:
            return None
        n   = len(xs)
        avg = sum(xs) / n
        sd  = statistics.pstdev(xs)
        sh  = avg / sd if sd else 0
        win = sum(1 for x in xs if x > 0) / n * 100
        return {"n": n, "avg": avg, "sharpe": sh, "win": win}

    # Pre-extract per-row decisions and returns for each config
    data = {name: [] for name in CONFIGS}
    for c, u, fr in univ:
        pnl = fr.get(HORIZON, 0)
        for name, fn in CONFIGS.items():
            try:
                traded = fn(c, u, fr)
            except Exception:
                traded = False
            data[name].append((traded, pnl))

    print(f"\n  {'Config':<22} {'n':>5}  {'avg%':>7}  {'95% CI avg':>18}  "
          f"{'sharpe':>7}  {'95% CI sharpe':>16}  {'win%':>6}")
    print(f"  {'-'*90}")

    for name, rows in data.items():
        point = compute(rows)
        if not point:
            print(f"  {name:<22}  insufficient data")
            continue

        # Bootstrap
        boot_avgs, boot_sharpes, boot_wins = [], [], []
        n_rows = len(rows)
        for _ in range(n_boot):
            sample = [rows[rng.randint(0, n_rows-1)] for _ in range(n_rows)]
            s = compute(sample)
            if s:
                boot_avgs.append(s["avg"])
                boot_sharpes.append(s["sharpe"])
                boot_wins.append(s["win"])

        if not boot_avgs:
            print(f"  {name:<22}  bootstrap failed")
            continue

        boot_avgs.sort(); boot_sharpes.sort()
        lo_a, hi_a = boot_avgs[int(0.025*len(boot_avgs))], boot_avgs[int(0.975*len(boot_avgs))]
        lo_s, hi_s = boot_sharpes[int(0.025*len(boot_sharpes))], boot_sharpes[int(0.975*len(boot_sharpes))]

        # Flag if CI crosses zero (not reliably positive)
        flag = "  ⚠️  CI crosses zero" if lo_a < 0 else ""

        print(f"  {name:<22} {point['n']:>5}  {point['avg']:>+7.2f}%"
              f"  [{lo_a:>+6.2f}%, {hi_a:>+6.2f}%]"
              f"  {point['sharpe']:>+7.2f}"
              f"  [{lo_s:>+5.2f}, {hi_s:>+5.2f}]"
              f"  {point['win']:>5.0f}%{flag}")

# ══════════════════════════════════════════════════════════════════════════════
# SECTION G — Jackknife stability test
# ══════════════════════════════════════════════════════════════════════════════

def jackknife(univ, sc):
    """
    Leave-one-out jackknife on avg/trade and Sharpe.
    High jackknife standard error relative to point estimate → result is
    driven by a small number of influential trades → fragile.
    Also runs a time-based split: first half vs second half of the dataset
    (sorted by article date) to check for temporal consistency.
    """
    print(f"\n  ══ G) JACKKNIFE STABILITY + TEMPORAL SPLIT ══")
    print(f"     Jackknife SE: how much each single trade moves the avg.")
    print(f"     Temporal split: does performance hold across time periods?")

    CONFIGS = {
        "ollama":        (None,       None,  None),
        "haiku_current": ("haiku",    0.45,  0.60),
        "haiku_best_avg":("haiku",    0.20,  0.75),
        "s5_current":    ("sonnet5",  0.45,  0.60),
        "s5_best_avg":   ("sonnet5",  0.45,  0.65),
        "s5_best_sharpe":("sonnet5",  0.20,  0.75),
    }

    # Sort universe by article date for temporal split
    def get_date(c):
        return c.get("dt", "")
    univ_sorted = sorted(univ, key=lambda x: get_date(x[0]))
    mid = len(univ_sorted) // 2
    halves = {"first_half": univ_sorted[:mid], "second_half": univ_sorted[mid:]}

    for name, (model, mag, conf) in CONFIGS.items():
        def trades(c, u, fr):
            if model is None:
                return ollama_trades(u)
            return anthropic_trades(sc.get(f"{model}:{c['ck']}"), mag, conf)

        xs = [fr[HORIZON] for c, u, fr in univ if trades(c, u, fr)]
        if len(xs) < 10:
            print(f"\n  {name}: insufficient data (n={len(xs)})")
            continue

        n   = len(xs)
        avg = sum(xs) / n

        # Leave-one-out jackknife
        jk_avgs = [(sum(xs) - x) / (n - 1) for x in xs]
        jk_mean = sum(jk_avgs) / n
        jk_se   = ((n - 1) / n * sum((a - jk_mean)**2 for a in jk_avgs)) ** 0.5

        # Jackknife bias
        jk_bias = (n - 1) * (jk_mean - avg)

        # Temporal split
        h1 = [fr[HORIZON] for c, u, fr in halves["first_half"]  if trades(c, u, fr)]
        h2 = [fr[HORIZON] for c, u, fr in halves["second_half"] if trades(c, u, fr)]
        h1_avg = sum(h1)/len(h1) if h1 else float("nan")
        h2_avg = sum(h2)/len(h2) if h2 else float("nan")

        # Consistency flag
        both_positive = (h1_avg > 0 and h2_avg > 0)
        consistent    = abs(h1_avg - h2_avg) < 2.0   # within 2% of each other
        flag = "✅ consistent" if (both_positive and consistent) else \
               "⚠️  both positive but drift" if both_positive else "❌ sign flip"

        print(f"\n  {name}  (n={n})")
        print(f"    point avg:      {avg:>+6.2f}%")
        print(f"    jackknife SE:   {jk_se:>+6.3f}%  (SE/avg = {abs(jk_se/avg)*100:.0f}% — "
              f"{'stable' if abs(jk_se/avg) < 0.15 else 'moderate' if abs(jk_se/avg) < 0.30 else 'fragile'})")
        print(f"    jackknife bias: {jk_bias:>+6.3f}%")
        print(f"    first half:     {h1_avg:>+6.2f}%  n={len(h1)}")
        print(f"    second half:    {h2_avg:>+6.2f}%  n={len(h2)}")
        print(f"    temporal:       {flag}")

# ══════════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════════

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--report-only",     action="store_true", help="Run A+B sections only")
    ap.add_argument("--sweep-only",      action="store_true", help="Run C+D+E sections only")
    ap.add_argument("--validation-only", action="store_true", help="Run F+G sections only (bootstrap+jackknife)")
    args = ap.parse_args()

    print("Loading universe…")
    univ = load_universe()
    print(f"  {len(univ)} articles with Ollama scores + forward returns")

    print("Loading Anthropic scores…")
    sc = load_anthropic_scores()
    for model in ANTHROPIC_MODELS:
        n = sum(1 for k in sc if k.startswith(f"{model}:") and sc[k])
        print(f"  {model}: {n} scored")

    ollama_n = sum(1 for c, u, fr in univ if ollama_trades(u))
    print(f"\nOllama baseline trades (mag≥{OLLAMA_MAG}): {ollama_n}")

    if not args.sweep_only and not args.validation_only:
        report_fixed(univ, sc)
        report_matched(univ, sc)

    if not args.report_only and not args.validation_only:
        sweep_confidence(univ, sc)
        sweep_magnitude(univ, sc)
        sweep_2d(univ, sc)

    if not args.report_only:
        bootstrap(univ, sc)
        jackknife(univ, sc)

if __name__ == "__main__":
    main()

