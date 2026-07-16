"""
anthropic_gate_bakeoff.py — slots sonnet5/haiku into the SAME confirm/veto ranking that originally
selected mistral-large as the live gate (overnight_confirm_veto.py: eligible() sample = 346 Ollama
bullish&mag>=0.75 picks, outcome = fr["rday"] forward stock return, confirm rule = bullish & mag>=CM).

Pure cache read — no new API calls. Anthropic scores already cover this sample (anthropic_backtest_cache.json,
same ck hash space as everything else here — see anthropic_scorer.py docstring). Prints two tables:
  1) FIXED CM (current live gate bar, config.GEMINI_CONFIRM_MIN_MAGNITUDE) — apples-to-apples with the
     bar mistral-large actually runs at live.
  2) PER-MODEL BEST CM — each model's own magnitude scale is different (Anthropic runs lower than
     Mistral's), so also report each model's own edge-maximizing threshold, same tuning exercise that
     moved mistral's bar 0.40->0.50.

Usage:  USE_YAHOO_BARS=1 .venv/bin/python -u anthropic_gate_bakeoff.py   (run from data/)
"""
import json, random
import numpy as np
import gemini_lotto_pnl as gl
from config import GEMINI_CONFIRM_MIN_MAGNITUDE as LIVE_CM

OVERNIGHT_CACHE  = "overnight_cv_cache.json"
ANTHROPIC_CACHE  = "anthropic_backtest_cache.json"

GATE_MODELS = ["groq/llama-4-scout", "mistral-large", "groq/gpt-oss-120b", "groq/qwen3.6-27b"]
ANTHROPIC_MODELS = ["haiku", "sonnet5"]

CM_SWEEP = [round(x * 0.05, 2) for x in range(4, 18)]   # 0.20 .. 0.85


def eligible():
    """Same sample overnight_confirm_veto.py validated the live gate on."""
    uni = json.load(open("unified_scores.json")); cpool = json.load(open("yahoo_candpool_cache.json"))
    gl.SAMPLE = 100000
    out = []
    for c in gl.candidates():
        u = uni.get("unified_v1:" + c["ck"]); fr = cpool.get(f"{c['tk']}_{c['ck']}")
        if (isinstance(u, dict) and u.get("sentiment") == "bullish" and float(u.get("magnitude", 0) or 0) >= 0.75
                and fr and fr.get("px", 0) >= 5 and c.get("h")):
            out.append((c, fr))
    out.sort(key=lambda x: x[0]["ck"])
    return out


def boot_edge(cf, vt, n=5000, seed=0):
    rng = random.Random(seed)
    d = [sum(rng.choices(cf, k=len(cf))) / len(cf) - sum(rng.choices(vt, k=len(vt))) / len(vt) for _ in range(n)]
    return np.percentile(d, 5), np.percentile(d, 95)


def edge_at(rows, cm):
    """rows = [(sent, mag, rday), ...]. Returns stats dict or None if too small a split."""
    cf = [r for s, m, r in rows if s == "bullish" and m >= cm]
    vt = [r for s, m, r in rows if not (s == "bullish" and m >= cm)]
    if len(cf) < 5 or len(vt) < 5:
        return None
    ca, va = sum(cf) / len(cf), sum(vt) / len(vt)
    lo, hi = boot_edge(cf, vt)
    ch = sum(1 for x in cf if x >= 5) / len(cf) * 100
    vh = sum(1 for x in vt if x >= 5) / len(vt) * 100
    return {"n": len(cf) + len(vt), "veto_pct": len(vt) / (len(cf) + len(vt)) * 100,
            "confirm_avg": ca, "veto_avg": va, "edge": ca - va, "ci": (lo, hi),
            "confirm_hit": ch, "veto_hit": vh}


def print_row(name, s):
    sig = "✓sig" if s["ci"][0] > 0 else "~ns"
    print(f"  {name:22} {s['n']:>4} {s['veto_pct']:>4.0f}% {s['confirm_avg']:>+7.1f}% {s['veto_avg']:>+6.1f}% "
          f"{s['edge']:>+5.1f}% [{s['ci'][0]:>+4.1f},{s['ci'][1]:>+4.1f}]{sig:>5} {s['confirm_hit']:>4.0f}% {s['veto_hit']:>4.0f}%")


def main():
    sample = eligible()
    overnight = json.load(open(OVERNIGHT_CACHE))
    anthropic = json.load(open(ANTHROPIC_CACHE))

    print(f"sample: {len(sample)} picks (same set overnight_confirm_veto.py used to pick mistral-large)")
    print(f"live gate bar: config.GEMINI_CONFIRM_MIN_MAGNITUDE = {LIVE_CM}\n")

    # Build (sent, mag, rday) rows per model
    rows_by_model = {}
    for disp in GATE_MODELS:
        rows = [(overnight[f"{disp}:{c['ck']}"]["sent"], overnight[f"{disp}:{c['ck']}"]["mag"], fr["rday"])
                 for c, fr in sample if overnight.get(f"{disp}:{c['ck']}")]
        rows_by_model[disp] = rows
    for disp in ANTHROPIC_MODELS:
        rows = [(anthropic[f"{disp}:{c['ck']}"]["sent"], float(anthropic[f"{disp}:{c['ck']}"].get("mag", 0) or 0), fr["rday"])
                 for c, fr in sample if anthropic.get(f"{disp}:{c['ck']}")]
        rows_by_model[disp] = rows

    print(f"  ══ 1) FIXED CM={LIVE_CM} (the bar the live gate actually runs at) ══")
    print(f"  {'model':22} {'n':>4} {'veto%':>5} {'CONFIRM':>8} {'VETO':>7} {'edge':>6} {'boot 5-95% edge CI':>20} {'Chit':>5} {'Vhit':>5}")
    res = []
    for disp in GATE_MODELS + ANTHROPIC_MODELS:
        s = edge_at(rows_by_model[disp], LIVE_CM)
        if s:
            res.append((disp, s))
        else:
            n = len(rows_by_model[disp])
            print(f"  {disp:22} n={n} (insufficient split at this CM)")
    for disp, s in sorted(res, key=lambda r: -r[1]["edge"]):
        print_row(disp, s)

    print(f"\n  ══ 2) PER-MODEL BEST CM (own magnitude-scale-tuned threshold, edge-maximizing) ══")
    print(f"  {'model':22} {'best CM':>7} {'n':>4} {'veto%':>5} {'CONFIRM':>8} {'VETO':>7} {'edge':>6} {'boot 5-95% edge CI':>20} {'Chit':>5} {'Vhit':>5}")
    res2 = []
    for disp in GATE_MODELS + ANTHROPIC_MODELS:
        best_cm, best_s = None, None
        for cm in CM_SWEEP:
            s = edge_at(rows_by_model[disp], cm)
            if s and (best_s is None or s["edge"] > best_s["edge"]):
                best_cm, best_s = cm, s
        if best_s:
            res2.append((disp, best_cm, best_s))
        else:
            print(f"  {disp:22}  insufficient data at any threshold")
    for disp, cm, s in sorted(res2, key=lambda r: -r[2]["edge"]):
        sig = "✓sig" if s["ci"][0] > 0 else "~ns"
        print(f"  {disp:22} {cm:>7.2f} {s['n']:>4} {s['veto_pct']:>4.0f}% {s['confirm_avg']:>+7.1f}% "
              f"{s['veto_avg']:>+6.1f}% {s['edge']:>+5.1f}% [{s['ci'][0]:>+4.1f},{s['ci'][1]:>+4.1f}]{sig:>5} "
              f"{s['confirm_hit']:>4.0f}% {s['veto_hit']:>4.0f}%")

    print("\n  edge = avg forward return of CONFIRMed picks minus VETOed picks (rday, next-session close-to-close).")
    print("  CI entirely > 0 ⇒ that model's veto significantly separates winners from losers at this sample size.")
    print("  Table 1 is the fair live-equivalent comparison; table 2 shows headroom if each model's bar were tuned")
    print("  the way mistral-large's was (0.40→0.50, config.py note).")


if __name__ == "__main__":
    main()
