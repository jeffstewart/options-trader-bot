"""
prompt_exp.py — Stage-1 prompt/model experiments graded by the signal_lab yardstick.

Scores a FIXED dev-split sample with candidate prompts on a hosted model (Groq, so
no Ollama contention with the live bot), caches scores, then grades each
(model, prompt) with signal_lab's metric: rank-IC vs MARKET-EXCESS return (95%
bootstrap CI), top-quintile LIFT, and bottom-quintile winner-rate (noise-exclusion).
The llama3.2 baseline (current dual-cache mag×conf) is graded on the SAME sample
as the control. Promote a cell only if its IC CI clears the baseline and lift CI > 1.

Cache: prompt_exp_scores.json  key = "{model}:{prompt}:{cache_key}"  (resumable).

Usage:
  USE_YAHOO_BARS=1 .venv/bin/python prompt_exp.py \
      --model llama-3.3-70b-versatile --prompts baseline,direct_return,materiality \
      --limit 300 --fwd 5 --win 5 --delay 0.3
  # other free Groq models: openai/gpt-oss-120b, qwen3-32b  (one --model per run)
"""
import argparse, json, os, re, time
from datetime import datetime, timezone
from pathlib import Path

from dotenv import load_dotenv; load_dotenv()
from openai import OpenAI

os.environ.setdefault("USE_YAHOO_BARS", "1")
import backtest as _bt
from prompt_lab import PROMPTS, score_article          # reuse prompt set + scorer (retry/backoff)
import signal_lab as SL
from signal_quality import spearman

SCORES = Path("prompt_exp_scores.json")
_TK = re.compile(r"^[A-Z]{1,5}$")


def load_dev_sample(limit, fwd, split="dev", cache_file="dual_score_cache.json"):
    raw = json.load(open(cache_file))
    evs = []
    for v in raw.values():
        if not isinstance(v, dict):
            continue
        a = v.get("_article", {}) or {}
        b = v.get("bullish", {}) or {}
        if b.get("reasoning") == "SCORE_FAILED":
            continue
        cands = [t for t in (b.get("tickers", []) or []) if _TK.match(t) and t not in ("BTC", "ETH")]
        if not cands or not a.get("headline") or not a.get("created_at"):
            continue
        try:
            dt = datetime.fromisoformat(str(a["created_at"]).replace("Z", "+00:00")).astimezone(timezone.utc)
        except Exception:
            continue
        if split != "all" and SL.split_of(cands[0], dt) != split:
            continue
        evs.append({"ticker": cands[0], "dt": dt, "headline": a["headline"],
                    "body": a.get("summary", ""),
                    "base_score": float(b.get("magnitude", 0)) * float(b.get("confidence", 0))})
    evs.sort(key=lambda e: e["dt"])
    if len(evs) > limit * 3:                       # stride so the sample spans the window
        evs = evs[::max(1, len(evs) // (limit * 3))]
    sample = []
    for e in evs:
        if len(sample) >= limit:
            break
        ex = SL.excess_return(e["ticker"], e["dt"], fwd)
        if ex is not None:
            e["ex"] = ex
            sample.append(e)
    return sample


def grade(name, rows, win):
    rows = [r for r in rows if r.get("score") is not None]
    if len(rows) < 40:
        return f"  {name:34} n={len(rows):>4}   (too few usable)"
    sc = [r["score"] for r in rows]; ex = [r["ex"] for r in rows]
    ic = spearman(sc, ex)
    ic_ci = SL.boot_ci(lambda s: spearman([x["score"] for x in s], [x["ex"] for x in s]), rows)
    order = sorted(rows, key=lambda r: r["score"]); q = len(order) // 5
    topq, botq = order[4*q:], order[:q]
    base = sum(1 for r in rows if r["ex"] >= win) / len(rows)
    wr = lambda s: (sum(1 for x in s if x["ex"] >= win) / len(s)) if s else 0
    lift = wr(topq) / base if base else float("nan")
    lift_ci = SL.boot_ci(lambda s: (wr(s) / base) if base else None, topq)
    flag = "★" if (ic_ci[0] > 0 and lift_ci[0] > 1) else " "
    return (f" {flag}{name:33} n={len(rows):>4}  IC={ic:+.3f} CI[{ic_ci[0]:+.2f},{ic_ci[1]:+.2f}]"
            f"  lift={lift:.2f}x CI[{lift_ci[0]:.2f},{lift_ci[1]:.2f}]"
            f"  topQ={wr(topq)*100:.0f}% botQ={wr(botq)*100:.0f}% (base {base*100:.0f}%)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--provider", choices=["groq", "ollama"], default="groq",
                    help="groq = hosted (LAB_HOSTED_*); ollama = local (no rate limit/cost)")
    ap.add_argument("--model", default=os.environ.get("LAB_HOSTED_MODEL", "llama-3.3-70b-versatile"))
    ap.add_argument("--prompts", default="baseline,direct_return,materiality")
    ap.add_argument("--limit", type=int, default=300)
    ap.add_argument("--fwd", type=int, default=5)
    ap.add_argument("--win", type=float, default=5.0)
    ap.add_argument("--delay", type=float, default=0.3)
    ap.add_argument("--split", choices=["dev","test","all"], default="dev",
                    help="which split to score (default: dev — test is held out for validation)")
    ap.add_argument("--cache-file", default="dual_score_cache.json",
                    help="source article/score cache (e.g. bear_dual_cache.json for the 2022 window)")
    args = ap.parse_args()

    if args.provider == "ollama":
        client = OpenAI(base_url=os.environ.get("OLLAMA_BASE_URL", "http://localhost:11434/v1"),
                        api_key="ollama")
    else:
        client = OpenAI(base_url=os.environ["LAB_HOSTED_BASE_URL"], api_key=os.environ["LAB_HOSTED_KEY"])
    prompts = [p for p in args.prompts.split(",") if p.strip()]

    print(f"Building {args.split}-split sample (limit {args.limit}, fwd {args.fwd}d, excess-return labels)…", flush=True)
    sample = load_dev_sample(args.limit, args.fwd, split=args.split, cache_file=args.cache_file)
    print(f"Dev sample with usable excess returns: {len(sample)}\n", flush=True)

    sc = json.loads(SCORES.read_text()) if SCORES.exists() else {}

    # Baseline control: current llama3.2 mag×conf on the SAME sample
    base_rows = [{"score": e["base_score"], "ex": e["ex"], "ticker": e["ticker"]} for e in sample]
    print(f"=== signal_lab grade  model={args.model}  dev n={len(sample)}  fwd={args.fwd}d  win=+{args.win}% ===")
    print(grade("llama3.2 BASELINE (mag×conf)", base_rows, args.win))

    for pname in prompts:
        if pname not in PROMPTS:
            print(f"  unknown prompt '{pname}' — skip"); continue
        system = PROMPTS[pname]
        rows = []
        scored_n = 0
        for e in sample:
            key = f"{args.model}:{pname}:{_bt.cache_key(e['headline'], e['body'])}"
            if key in sc:
                s = sc[key]
            else:
                s = score_article(client, args.model, system, e["headline"], e["body"])
                sc[key] = s
                scored_n += 1
                if scored_n % 25 == 0:
                    SCORES.write_text(json.dumps(sc))
                    print(f"   …{pname}: scored {scored_n} new", flush=True)
                if args.delay:
                    time.sleep(args.delay)
            rows.append({"score": s, "ex": e["ex"], "ticker": e["ticker"]})
        SCORES.write_text(json.dumps(sc))
        print(grade(f"{args.model.split('/')[-1]} / {pname}", rows, args.win))

    print("\n  ★ = IC CI clears 0 AND lift CI > 1 (promote candidate). Confirm survivors on")
    print("  --split test + 2022 + the Stage-3 net-of-cost P&L gate before trusting.")


if __name__ == "__main__":
    main()
