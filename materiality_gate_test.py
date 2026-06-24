"""
materiality_gate_test.py — Stage-1.5 screen: is materiality useful as a BINARY GATE?

The rank-IC tests (prompt_exp.py / signal_lab.py) asked "does the graded materiality
score RANK-ORDER returns?" — answer across 5 model families: no (IC ≈ 0). But qwen3-32b's
top-quintile LIFT CI cleared 1.0, which is the signature of a *binary* effect: a coarse
"material vs not" split may still select a better subset even with no graded ordering.

This script tests that directly, on the SAME frozen dev sample + market-excess labels +
bootstrap CIs as the rest of signal_lab, and answers the two decision-relevant questions:

  A. STANDALONE gate — sweep τ; does {materiality ≥ τ} have a higher winner-rate / mean
     excess return than the full population? CI on the DIFFERENCE (gated − base).
  B. INCREMENTAL gate — among events the live bot ALREADY trades (mag≥0.35 & conf≥0.70),
     does additionally requiring {materiality ≥ τ} raise the winner-rate? CI on the
     difference (gated∩live − live). This is the only result that would justify a code
     change: materiality must add separation *on top of* the existing signal, not just
     correlate with it.

Promote only if a difference CI clears 0. None (parse-fail) scores are reported and
EXCLUDED (can't gate on a missing score) — a high None rate is itself a deployability
strike against the prompt.

Usage:
  USE_YAHOO_BARS=1 .venv/bin/python materiality_gate_test.py
  USE_YAHOO_BARS=1 .venv/bin/python materiality_gate_test.py --model qwen/qwen3-32b --win 5 --fwd 5
"""
import argparse, json, os, statistics
from pathlib import Path

os.environ.setdefault("USE_YAHOO_BARS", "1")
import backtest as _bt
import signal_lab as SL
from prompt_exp import load_dev_sample

SCORES = Path("prompt_exp_scores.json")
LIVE_MIN_MAG, LIVE_MIN_CONF = 0.35, 0.70   # the bot's actual entry gate (config.py)


def diff_ci(rows_a, rows_b, stat, n=2000, seed=1):
    """Bootstrap 95% CI on stat(A) - stat(B), resampling each group independently."""
    import random
    rng = random.Random(seed)
    if not rows_a or not rows_b:
        return (float("nan"), float("nan"))
    vals = []
    for _ in range(n):
        sa = [rows_a[rng.randrange(len(rows_a))] for _ in range(len(rows_a))]
        sb = [rows_b[rng.randrange(len(rows_b))] for _ in range(len(rows_b))]
        vals.append(stat(sa) - stat(sb))
    vals.sort()
    return (vals[int(0.025 * len(vals))], vals[int(0.975 * len(vals))])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="qwen/qwen3-32b",
                    help="which cached model's materiality_fewshot scores to gate on")
    ap.add_argument("--prompt", default="materiality_fewshot")
    ap.add_argument("--limit", type=int, default=1000)   # must match the scoring run
    ap.add_argument("--fwd", type=int, default=5)
    ap.add_argument("--win", type=float, default=5.0)
    ap.add_argument("--split", choices=["dev", "test", "all"], default="dev",
                    help="which frozen split to evaluate (must match the split that was scored)")
    ap.add_argument("--cache-file", default="dual_score_cache.json",
                    help="source article/score cache (e.g. bear_dual_cache.json for the 2022 window)")
    args = ap.parse_args()

    print(f"Rebuilding the exact {args.split} sample (limit {args.limit}, fwd {args.fwd}d)…", flush=True)
    sample = load_dev_sample(args.limit, args.fwd, split=args.split, cache_file=args.cache_file)   # base_score (mag×conf) + ex
    print(f"{args.split} sample with usable excess returns: {len(sample)}", flush=True)

    sc = json.loads(SCORES.read_text())
    # map cache_key -> (mag, conf) from the same cache so we can apply the EXACT live gate
    dual = json.load(open(args.cache_file))
    magconf = {}
    for v in dual.values():
        if not isinstance(v, dict):
            continue
        a = v.get("_article", {}) or {}
        b = v.get("bullish", {}) or {}
        if not a.get("headline"):
            continue
        k = _bt.cache_key(a["headline"], a.get("summary", ""))
        magconf[k] = (float(b.get("magnitude", 0) or 0), float(b.get("confidence", 0) or 0))

    rows, n_none, n_nokey = [], 0, 0
    for e in sample:
        ck = _bt.cache_key(e["headline"], e["body"])
        mscore = sc.get(f"{args.model}:{args.prompt}:{ck}", "MISSING")
        if mscore == "MISSING":
            n_nokey += 1
            continue
        mag, conf = magconf.get(ck, (0.0, 0.0))
        rows.append({"ex": e["ex"], "mat": mscore, "mag": mag, "conf": conf,
                     "live": (mag >= LIVE_MIN_MAG and conf >= LIVE_MIN_CONF)})

    usable = [r for r in rows if r["mat"] is not None]
    n_none = sum(1 for r in rows if r["mat"] is None)
    print(f"\n=== materiality BINARY GATE  model={args.model}  prompt={args.prompt}  "
          f"split={args.split}  fwd={args.fwd}d  win=+{args.win}% excess ===")
    print(f"  joined={len(rows)}  (no cached score for {n_nokey})   "
          f"usable score={len(usable)}  None/parse-fail={n_none} "
          f"({100*n_none/max(1,len(rows)):.0f}% — these can't be gated, deployability strike)")
    if len(usable) < 60:
        print("  too few usable for a stable read."); return

    win = lambda s: (sum(1 for r in s if r["ex"] >= args.win) / len(s)) if s else 0.0
    mean = lambda s: statistics.mean(r["ex"] for r in s) if s else 0.0
    base_w, base_m = win(usable), mean(usable)
    print(f"\n  FULL population (n={len(usable)}): winner-rate {base_w*100:.1f}%   mean excess {base_m:+.2f}%")

    # ---- A. STANDALONE gate sweep ----
    print(f"\n  A. STANDALONE materiality gate (vs full population):")
    print(f"     {'τ (≥)':>6} {'kept':>5} {'%kept':>6} {'win%':>6} {'Δwin% [95% CI]':>22} {'meanEx':>7} {'Δmean [95% CI]':>22}")
    for tau in [0.0001, 0.1, 0.15, 0.2, 0.3]:
        keep = [r for r in usable if r["mat"] >= tau]
        if len(keep) < 15:
            print(f"     {tau:>6.3f} {len(keep):>5}   (too few)"); continue
        dwc = diff_ci(keep, usable, win)
        dmc = diff_ci(keep, usable, mean)
        flag = "★" if dwc[0] > 0 else " "
        print(f"   {flag} {tau:>6.3f} {len(keep):>5} {100*len(keep)/len(usable):>5.0f}% "
              f"{win(keep)*100:>5.1f}% [{dwc[0]*100:+5.1f},{dwc[1]*100:+5.1f}]pp "
              f"{mean(keep):>+6.2f}% [{dmc[0]:+5.2f},{dmc[1]:+5.2f}]")

    # ---- B. INCREMENTAL over the live bot gate ----
    live = [r for r in usable if r["live"]]
    print(f"\n  B. INCREMENTAL over live gate (mag≥{LIVE_MIN_MAG} & conf≥{LIVE_MIN_CONF}):")
    if len(live) < 40:
        print(f"     only {len(live)} events pass the live gate — too few; reporting anyway.")
    print(f"     LIVE-gate alone (n={len(live)}): winner-rate {win(live)*100:.1f}%   mean excess {mean(live):+.2f}%")
    print(f"     {'τ (≥)':>6} {'kept':>5} {'win%':>6} {'Δwin vs live [95% CI]':>26} {'meanEx':>7}")
    for tau in [0.0001, 0.1, 0.15, 0.2]:
        keep = [r for r in live if r["mat"] >= tau]
        if len(keep) < 12:
            print(f"     {tau:>6.3f} {len(keep):>5}   (too few)"); continue
        dwc = diff_ci(keep, live, win)
        flag = "★" if dwc[0] > 0 else " "
        print(f"   {flag} {tau:>6.3f} {len(keep):>5} {win(keep)*100:>5.1f}% "
              f"[{dwc[0]*100:+5.1f},{dwc[1]*100:+5.1f}]pp   {mean(keep):>+6.2f}%")

    print("\n  ★ = difference CI clears 0 (real, not noise). PROMOTE only if B shows a ★")
    print("  (materiality must add separation ON TOP of the existing mag×conf gate).")


if __name__ == "__main__":
    main()
