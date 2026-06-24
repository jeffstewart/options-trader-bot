"""
grade_scores.py — re-grade any cached (model, prompt) at any fwd/win threshold.

Reads prompt_exp_scores.json (key = "model:prompt:article_hash", value = float score),
matches to dual_score_cache.json for ticker+date, computes market-excess returns via
Yahoo, then grades with signal_lab metrics.  No LLM calls — pure price lookup + math.

Useful for:
  • multi-horizon check (fwd=1/3/5) on existing scores without re-running LLM
  • comparing the same scores at different win thresholds
  • adding new candidate prompts and checking how the distribution shifts

Usage:
  USE_YAHOO_BARS=1 .venv/bin/python grade_scores.py          # grade all cached combos
  USE_YAHOO_BARS=1 .venv/bin/python grade_scores.py --fwd 1 3 5 --model llama3.1:8b
  USE_YAHOO_BARS=1 .venv/bin/python grade_scores.py --show-dist  # score distribution per combo
"""
import argparse, hashlib, json, re
from datetime import datetime, timezone
from collections import defaultdict

import signal_lab as SL
from signal_quality import spearman, MEGACAPS

os_env_import = __import__("os").environ.setdefault("USE_YAHOO_BARS", "1")

_TK = re.compile(r"^[A-Z]{1,5}$")


def cache_key(headline, body):
    return hashlib.md5(f"{headline}||{body[:500]}".encode()).hexdigest()


def load_article_map():
    """Build hash → {ticker, dt, headline} from the dual cache."""
    raw = json.load(open("dual_score_cache.json"))
    amap = {}
    for v in raw.values():
        a  = v.get("_article", {}) or {}
        b  = v.get("bullish", {}) or {}
        hl = a.get("headline", "")
        bd = a.get("summary", "")
        if not hl:
            continue
        tks = [t for t in (b.get("tickers", []) or []) if _TK.match(t) and t not in ("BTC", "ETH")]
        if not tks:
            continue
        try:
            dt = datetime.fromisoformat(str(a["created_at"]).replace("Z", "+00:00")).astimezone(timezone.utc)
        except Exception:
            continue
        key = cache_key(hl, bd)
        amap[key] = {"ticker": tks[0], "dt": dt, "headline": hl}
    return amap


def grade_combo(label, rows, fwd, win, show_dist=False):
    """Grade a list of {score, ex, ticker} at the given threshold."""
    rows = [r for r in rows if r.get("score") is not None]
    if len(rows) < 40:
        return f"  {label:<46}  n={len(rows):>4}  (too few)"

    sc = [r["score"] for r in rows]
    ex = [r["ex"] for r in rows]
    ic = spearman(sc, ex)
    ic_ci = SL.boot_ci(lambda s: spearman([x["score"] for x in s], [x["ex"] for x in s]), rows)
    base = sum(1 for r in rows if r["ex"] >= win) / len(rows)
    order = sorted(rows, key=lambda r: r["score"])
    q = len(order) // 5
    topq, botq = order[4*q:], order[:q]
    wr = lambda s: sum(1 for x in s if x["ex"] >= win) / len(s) if s else 0
    lift = wr(topq) / base if base else float("nan")
    lift_ci = SL.boot_ci(lambda s: wr(s) / base if base else None, topq)
    flag = "★" if ic_ci[0] > 0 and lift_ci[0] > 1 else " "
    top_liq = sum(1 for r in sorted(rows, key=lambda r: r["ex"], reverse=True)[:max(10, len(rows)//10)]
                  if r["ticker"] in MEGACAPS)
    line = (f" {flag}{label:<45}  n={len(rows):>4}  "
            f"IC={ic:+.3f} CI[{ic_ci[0]:+.2f},{ic_ci[1]:+.2f}]  "
            f"lift={lift:.2f}x CI[{lift_ci[0]:.2f},{lift_ci[1]:.2f}]  "
            f"topQ={wr(topq)*100:.0f}% botQ={wr(botq)*100:.0f}% (base {base*100:.0f}%)")
    if show_dist:
        import statistics
        p25 = sorted(sc)[len(sc)//4]; p75 = sorted(sc)[3*len(sc)//4]
        line += f"  score_p25={p25:.2f} p75={p75:.2f} mean={statistics.mean(sc):.2f}"
    return line


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fwd", type=int, nargs="+", default=[1, 3, 5])
    ap.add_argument("--win", type=float, default=5.0)
    ap.add_argument("--model", nargs="*", default=None, help="filter to these models (space or comma separated)")
    ap.add_argument("--prompt", nargs="*", default=None, help="filter to these prompts (space or comma separated)")
    ap.add_argument("--show-dist", action="store_true")
    ap.add_argument("--split", choices=["dev", "test", "all"], default="dev")
    args = ap.parse_args()
    # Allow comma-separated values from shell scripts (e.g. --prompt a,b,c)
    if args.model:
        args.model = [m for raw in args.model for m in raw.split(",") if m]
    if args.prompt:
        args.prompt = [p for raw in args.prompt for p in raw.split(",") if p]

    print("Loading article map from dual_score_cache.json…", flush=True)
    amap = load_article_map()
    scores = json.load(open("prompt_exp_scores.json"))

    # Known prompt names — used to correctly parse keys where the model name
    # itself contains a colon (e.g. "llama3.1:8b:materiality:hash" has 4 parts;
    # naive split gives model="llama3.1", prompt="8b" which is wrong).
    KNOWN_PROMPTS = {
        "baseline", "direct_return", "materiality", "catalyst_typed",
        "binary_gate", "materiality_fewshot", "surprise_score",
        "lotto_swing",
    }

    def _parse_key(key):
        """Return (model, prompt, hash) handling colons in model names."""
        parts = key.split(":")
        if len(parts) < 3:
            return None, None, None
        # If parts[1] is a known prompt, model = parts[0]
        if parts[1] in KNOWN_PROMPTS:
            return parts[0], parts[1], ":".join(parts[2:])
        # Otherwise model spans parts[0]:parts[1] (e.g. "llama3.1:8b")
        if len(parts) >= 4:
            return f"{parts[0]}:{parts[1]}", parts[2], ":".join(parts[3:])
        return None, None, None

    # Group score entries by (model, prompt) → {hash: score}
    combos = defaultdict(dict)
    for key, score in scores.items():
        model_part, prompt_part, hash_part = _parse_key(key)
        if not model_part:
            continue
        if args.model and model_part not in args.model:
            continue
        if args.prompt and prompt_part not in args.prompt:
            continue
        combos[(model_part, prompt_part)][hash_part] = score

    print(f"Combos in cache: {len(combos)}  |  fwd horizons: {args.fwd}d  |  split: {args.split}")
    print(f"Article map: {len(amap)} entries\n")

    for fwd in args.fwd:
        print(f"{'═'*80}")
        print(f"  fwd={fwd}d  win=+{args.win}%  split={args.split}  (market-excess return)")
        print(f"{'═'*80}")

        # Baseline: llama3.2 mag×conf from signal_lab
        print("  Building excess-return labels for this horizon…", flush=True)

        for (model, prompt), hash_scores in sorted(combos.items()):
            rows = []
            for h, score in hash_scores.items():
                art = amap.get(h)
                if not art:
                    continue
                if args.split != "all" and SL.split_of(art["ticker"], art["dt"]) != args.split:
                    continue
                ex = SL.excess_return(art["ticker"], art["dt"], fwd)
                if ex is not None:
                    rows.append({"score": score, "ex": ex, "ticker": art["ticker"]})
            label = f"{model} / {prompt}"
            print(grade_combo(label, rows, fwd, args.win, args.show_dist))
        print()

    print("  ★ = IC CI excludes 0 AND lift CI > 1  (the promotion bar)")
    print("  Promote survivors to --split test, then 2022 window, then net-of-cost P&L gate.")


if __name__ == "__main__":
    main()
