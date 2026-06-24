"""
model_calibration.py — which model's unified_v1 SCORE best RANKS forward returns? (rank-IC).
This is the statistically-powered model comparison the noisy top-35 lotto P&L couldn't give us.
Uses the cached per-model scores (prompt_exp_scores.json, value = magnitude×confidence) over the
backtest candidate pool, joins each candidate to its forward move (Yahoo, cached), and reports
per-model Spearman rank-IC at the event-day, 1d, and 3d horizons + on the strict shared set.

Usage:  USE_YAHOO_BARS=1 .venv/bin/python -u model_calibration.py
"""
import os, json
os.environ.setdefault("USE_YAHOO_BARS", "1")
from datetime import timedelta
from collections import defaultdict
import gemini_lotto_pnl as gl
import yahoo_data
from prompt_lab import extract_score
from missed_movers import spearman

CACHE = "yahoo_candpool_cache.json"
MODELS = ["llama3.2", "gemini-2.5-flash", "gemini-3.5-flash", "gemini-3.1-flash-lite",
          "gpt-5.4-mini", "openai/gpt-oss-120b", "mistral-medium-latest",
          "mistral-small-latest", "llama-3.3-70b-versatile"]


def main():
    sc = json.load(open(gl.SCORES_PATH))
    uni = json.load(open("unified_scores.json"))
    gl.SAMPLE = 100000
    cands = gl.candidates()
    byck = {c["ck"]: c for c in cands}
    for c in cands:                                  # seed llama3.2 from the live unified cache
        u = uni.get("unified_v1:" + c["ck"])
        if isinstance(u, dict):
            sc["llama3.2:unified_v1:" + c["ck"]] = extract_score(u)

    # restrict to candidates a NON-llama model scored (llama is seeded everywhere, so it would
    # balloon the pool). The frontier/groq/mistral models only scored the ~533 bake-off pool —
    # that shared set is the fair comparison ground and keeps the fetch small.
    NONLLAMA = [m for m in MODELS if m != "llama3.2"]
    need = [c for c in cands if any(sc.get(f"{m}:unified_v1:" + c["ck"]) is not None for m in NONLLAMA)]
    cache = json.load(open(CACHE)) if os.path.exists(CACHE) else {}
    by_tk = defaultdict(list)
    for c in need:
        if f"{c['tk']}_{c['ck']}" not in cache:
            by_tk[c["tk"]].append(c)
    todo = sum(len(v) for v in by_tk.values())
    print(f"candidates: {len(need)} | forward returns to fetch: {len(by_tk)} tickers ({todo}) …", flush=True)
    for n, (tk, cs) in enumerate(by_tk.items()):
        try:
            ds = sorted(c["dt"] for c in cs)
            bars = sorted(yahoo_data.get_yahoo_bars(tk, ds[0] - timedelta(days=5), ds[-1] + timedelta(days=10)),
                          key=lambda b: b["t"])
            for c in cs:
                d0 = c["dt"].date()
                prior = [b for b in bars if b["t"].date() < d0]
                fwd = [b for b in bars if b["t"].date() >= d0]
                if len(fwd) >= 2 and fwd[0]["c"] and prior:
                    px = fwd[0]["c"]
                    cache[f"{tk}_{c['ck']}"] = {
                        "px": round(px, 2),
                        "rday": round((px / prior[-1]["c"] - 1) * 100, 2),
                        "r1": round((fwd[1]["c"] / px - 1) * 100, 2),
                        "r3": round((max(b["c"] for b in fwd[1:4]) / px - 1) * 100, 2)}
                else:
                    cache[f"{tk}_{c['ck']}"] = None
        except Exception:
            for c in cs:
                cache[f"{tk}_{c['ck']}"] = None
        if n % 50 == 49:
            json.dump(cache, open(CACHE, "w"))
    json.dump(cache, open(CACHE, "w"))

    # liquid only (px>=5)
    def ret(ck, h):
        v = cache.get(f"{byck[ck]['tk']}_{ck}")
        return v[h] if v and v.get("px", 0) >= 5 else None

    print(f"\n══ per-model rank-IC of score vs forward return (liquid, n=coverage) ══")
    print(f"  {'model':24} {'n':>5} {'IC event-day':>13} {'IC 1d':>8} {'IC 3d':>8}")
    res = []
    for m in MODELS:
        pre = f"{m}:unified_v1:"
        trip = [(sc[pre + ck], ret(ck, "rday"), ret(ck, "r1"), ret(ck, "r3"))
                for ck in byck if sc.get(pre + ck) is not None and ret(ck, "rday") is not None]
        if len(trip) < 20:
            print(f"  {m:24} {len(trip):>5}   (insufficient)"); continue
        s = [t[0] for t in trip]
        icd = spearman(s, [t[1] for t in trip]); ic1 = spearman(s, [t[2] for t in trip]); ic3 = spearman(s, [t[3] for t in trip])
        res.append((m, len(trip), icd, ic1, ic3))
    for m, n, icd, ic1, ic3 in sorted(res, key=lambda r: -(r[2] or -9)):
        print(f"  {m:24} {n:>5} {icd:>+13.3f} {ic1:>+8.3f} {ic3:>+8.3f}")
    print(f"\n  Noise floor ≈ ±{1/ (min(r[1] for r in res)**0.5):.3f} (1/√n). An IC must clear that to be real.")
    print("  Higher IC = that model's score genuinely ranks winners — i.e. a usable magnitude.")


if __name__ == "__main__":
    main()
