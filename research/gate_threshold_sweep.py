"""
gate_threshold_sweep.py — does raising the confirm/veto gate's magnitude threshold (live
GEMINI_CONFIRM_MIN_MAGNITUDE = 0.50) prune MARGINAL confirms that underperform? Motivated by BA: the
gate confirmed "Boeing wins $2B contract" at mag 0.65 → −34%. Uses the gate model's (mistral-large)
historical scores (overnight_cv_cache) joined to forward stock r3 (yahoo_candpool), sweeps the bar, and
reports the KEPT vs REMOVED set at each level with a bootstrap CI on the removed (marginal) set. If the
removed set is net-weak while KEPT stays strong, raising helps. (Stock r3 is a conservative proxy — on
leveraged short-DTE calls a weak underlying move is worse after theta, so an edge here understates it.)

Usage:  USE_YAHOO_BARS=1 .venv/bin/python -u research/gate_threshold_sweep.py
"""
import os, json, random
os.environ.setdefault("USE_YAHOO_BARS", "1")
import config
os.chdir(config.DATA_DIR)
import numpy as np
import gemini_lotto_pnl as gl

GATE = "mistral-large"
BASE = 0.50                                   # current GEMINI_CONFIRM_MIN_MAGNITUDE


def load():
    d = json.load(open("overnight_cv_cache.json"))
    ml = {k.split(":", 1)[1]: v for k, v in d.items() if k.startswith(GATE + ":") and isinstance(v, dict)}
    cp = json.load(open("yahoo_candpool_cache.json"))
    gl.SAMPLE = 100000
    rows = []
    for c in gl.candidates():
        v = ml.get(c["ck"]); fr = cp.get(f"{c['tk']}_{c['ck']}")
        if v and v.get("sent") == "bullish" and fr and fr.get("px", 0) >= 5:
            rows.append((float(v.get("mag", 0) or 0), fr["r3"]))
    return rows


def stat(rs):
    if not rs:
        return "n=0            "
    a = sum(rs) / len(rs); w = sum(1 for x in rs if x > 0) / len(rs) * 100
    return f"n={len(rs):<3} avg{a:>+6.2f}% win{w:>3.0f}%"


def boot(rs, n=5000):
    if not rs:
        return (0.0, 0.0)
    rng = random.Random(0)
    m = [sum(rng.choices(rs, k=len(rs))) / len(rs) for _ in range(n)]
    return float(np.percentile(m, 5)), float(np.percentile(m, 95))


def main():
    rows = load()
    bull = [(m, r) for m, r in rows if m >= BASE]      # confirmed under the current 0.50 bar
    allr = [r for m, r in bull]
    print(f"gate={GATE} · confirmed-at-{BASE} bullish trades with forward r3: {len(bull)}")
    print(f"  ALL confirms (current 0.50): {stat(allr)}")

    print(f"\n  ── raise the bar 0.50 → T: REMOVED (0.50≤mag<T, would now be VETOED) vs KEPT (mag≥T) ──")
    for T in (0.55, 0.60, 0.65, 0.70):
        removed = [r for m, r in bull if BASE <= m < T]
        kept    = [r for m, r in bull if m >= T]
        lo, hi = boot(removed)
        sig = "✓removes weak (CI≤0)" if hi <= 0 else ("✗cuts WINNERS (CI>0)" if lo > 0 else "~mixed")
        print(f"   T={T:.2f}  REMOVED {stat(removed)} CI[{lo:+.1f},{hi:+.1f}] {sig}   KEPT {stat(kept)}")

    print(f"\n  ── forward r3 by gate-mag bucket (does the marginal band underperform?) ──")
    for lo, hi in [(0.50, 0.55), (0.55, 0.65), (0.65, 0.75), (0.75, 1.01)]:
        rs = [r for m, r in bull if lo <= m < hi]
        print(f"   gate-mag [{lo:.2f},{hi:.2f}): {stat(rs)}")
    print("\n  Raise the bar only if REMOVED is net-weak (CI≤0) AND KEPT holds — else we'd cut real trades.")


if __name__ == "__main__":
    main()
