"""
sonnet5_mistral_pipeline.py — if sonnet5 replaced Ollama as PRIMARY scorer, how often would the live
confirm/veto gate (mistral-large) confirm vs veto its picks, and what would that do to P&L?

Universe = sonnet5's own trade picks at current bot thresholds (bullish & mag>=0.45 & conf>=0.60,
matching anthropic_backtest.py). mistral-large scores are looked up from overnight_cv_cache.json
(346 cached scores from the original gate bake-off) — coverage is PARTIAL because that cache was
built on Ollama's own top picks, not sonnet5's, so it only overlaps where the two scorers agree on
a candidate. Picks with no cached mistral score are reported separately (not silently dropped).

Usage:  USE_YAHOO_BARS=1 .venv/bin/python -u sonnet5_mistral_pipeline.py   (run from data/)
"""
import json, statistics
import gemini_lotto_pnl as gl
from config import GEMINI_CONFIRM_MIN_MAGNITUDE as CM

HORIZON = "r3"
S5_MIN_MAG, S5_MIN_CONF = 0.45, 0.60


def universe():
    uni = json.load(open("unified_scores.json")); cp = json.load(open("yahoo_candpool_cache.json"))
    gl.SAMPLE = 100000
    out = []
    for c in gl.candidates():
        u = uni.get("unified_v1:" + c["ck"]); fr = cp.get(f"{c['tk']}_{c['ck']}")
        if isinstance(u, dict) and fr and fr.get("px", 0) >= 5 and c.get("h"):
            out.append((c, u, fr))
    return out


def stat(name, xs, indent=2):
    pad = " " * indent
    if not xs:
        print(f"{pad}{name:46} n=0")
        return
    n, tot = len(xs), sum(xs)
    avg = tot / n
    sd = statistics.pstdev(xs) if n > 1 else 0
    sh = avg / sd if sd else 0
    win = sum(1 for x in xs if x > 0) / n * 100
    print(f"{pad}{name:46} n={n:<4} Σ{tot:>+8.1f}%  avg{avg:>+6.2f}%  win{win:>4.0f}%  sharpe{sh:>+5.2f}")


def main():
    anthropic = json.load(open("anthropic_backtest_cache.json"))
    overnight = json.load(open("overnight_cv_cache.json"))
    univ = universe()

    def s5(ck):
        return anthropic.get(f"sonnet5:{ck}")

    def s5_trades(ck):
        v = s5(ck)
        return v and v.get("sent") == "bullish" and float(v.get("mag", 0) or 0) >= S5_MIN_MAG \
            and float(v.get("conf", 0) or 0) >= S5_MIN_CONF

    picks = [(c, u, fr) for c, u, fr in univ if s5_trades(c["ck"])]
    print(f"sonnet5 primary-scorer picks (bullish, mag≥{S5_MIN_MAG}, conf≥{S5_MIN_CONF}): n={len(picks)}\n")

    covered, uncovered = [], []
    for c, u, fr in picks:
        mg = overnight.get(f"mistral-large:{c['ck']}")
        (covered if mg else uncovered).append((c, u, fr, mg))

    print(f"mistral-large gate coverage: {len(covered)}/{len(picks)} already scored "
          f"({len(uncovered)} would need fresh scoring — sonnet5 flags candidates Ollama's own")
    print(f"bar never reached, so the original gate bake-off never scored them with mistral-large)\n")

    confirmed = [(c, u, fr) for c, u, fr, mg in covered
                 if mg["sent"] == "bullish" and float(mg["mag"] or 0) >= CM]
    vetoed = [(c, u, fr) for c, u, fr, mg in covered
              if not (mg["sent"] == "bullish" and float(mg["mag"] or 0) >= CM)]

    print(f"  ══ Confirm/veto split of sonnet5's picks (live gate bar mag≥{CM}), n={len(covered)} covered ══")
    if covered:
        print(f"  confirmed: {len(confirmed)}/{len(covered)} ({len(confirmed)/len(covered)*100:.0f}%)   "
              f"vetoed: {len(vetoed)}/{len(covered)} ({len(vetoed)/len(covered)*100:.0f}%)\n")

    stat("sonnet5 ALL picks (no gate)", [fr[HORIZON] for c, u, fr in picks])
    stat("  └ covered by mistral cache", [fr[HORIZON] for c, u, fr, mg in covered], indent=4)
    stat("      ├ mistral CONFIRMS", [fr[HORIZON] for c, u, fr in confirmed], indent=6)
    stat("      └ mistral VETOES", [fr[HORIZON] for c, u, fr in vetoed], indent=6)
    stat("  └ NOT covered (unscored, pass-through)", [fr[HORIZON] for c, u, fr, mg in uncovered], indent=4)

    print(f"\n  ══ Pipeline comparison (on the {len(covered)}-pick covered subset only) ══")
    stat("sonnet5 alone", [fr[HORIZON] for c, u, fr, mg in covered])
    stat("sonnet5 + mistral gate (keep confirms only)", [fr[HORIZON] for c, u, fr in confirmed])
    print(f"\n  Gate would have thrown out {len(vetoed)} of sonnet5's {len(covered)} covered picks "
          f"({len(vetoed)/len(covered)*100:.0f}%).")
    vet_pnl = sum(fr[HORIZON] for c, u, fr in vetoed)
    print(f"  Those vetoed picks totaled {vet_pnl:+.1f}% combined forward return "
          f"({'gate would have been WRONG to cut them' if vet_pnl > 0 else 'gate would have correctly avoided a net loser'}).")


if __name__ == "__main__":
    main()
