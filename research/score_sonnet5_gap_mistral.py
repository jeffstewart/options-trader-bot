"""
score_sonnet5_gap_mistral.py — fills the mistral-large coverage gap for sonnet5's unique picks
(candidates below Ollama's own bar, so never scored in the original overnight_confirm_veto.py
bake-off). Free tier, so paced conservatively — the live bot is hitting the same mistral-large
free-tier quota for its live confirm/veto gate while the market is open, and this must not starve
it. Writes into overnight_cv_cache.json (same schema/keys as the original bake-off, "mistral-large:
{ck}" -> {"sent","mag"}) so sonnet5_mistral_pipeline.py picks the new scores up automatically.

Usage: USE_YAHOO_BARS=1 .venv/bin/python -u score_sonnet5_gap_mistral.py   (run from data/)
"""
import os, json, time
os.environ.setdefault("USE_YAHOO_BARS", "1")
from openai import OpenAI
import gemini_lotto_pnl as gl
import confirm_veto_sweep as cvs

CACHE = "overnight_cv_cache.json"
S5_MIN_MAG, S5_MIN_CONF = 0.45, 0.60
PACE = 12   # seconds between calls -- conservative; live bot shares this free-tier quota right now


def universe():
    uni = json.load(open("unified_scores.json")); cp = json.load(open("yahoo_candpool_cache.json"))
    gl.SAMPLE = 100000
    out = []
    for c in gl.candidates():
        u = uni.get("unified_v1:" + c["ck"]); fr = cp.get(f"{c['tk']}_{c['ck']}")
        if isinstance(u, dict) and fr and fr.get("px", 0) >= 5 and c.get("h"):
            out.append((c, u, fr))
    return out


def main():
    anthropic = json.load(open("anthropic_backtest_cache.json"))
    sc = json.load(open(CACHE)) if os.path.exists(CACHE) else {}
    univ = universe()

    def s5_trades(ck):
        v = anthropic.get(f"sonnet5:{ck}")
        return v and v.get("sent") == "bullish" and float(v.get("mag", 0) or 0) >= S5_MIN_MAG \
            and float(v.get("conf", 0) or 0) >= S5_MIN_CONF

    picks = [c for c, u, fr in univ if s5_trades(c["ck"])]
    todo = [c for c in picks if f"mistral-large:{c['ck']}" not in sc]
    print(f"sonnet5 picks: {len(picks)} · already cached: {len(picks) - len(todo)} · to score: {len(todo)}", flush=True)
    if not todo:
        print("nothing to do"); return

    cli = OpenAI(base_url=cvs.PROV["mistral"][0], api_key=cvs.PROV["mistral"][1])
    done = 0
    for c in todo:
        for att in range(5):
            try:
                sc[f"mistral-large:{c['ck']}"] = cvs.score_one(
                    cli, "mistral-large-latest", "mistral", c["h"], c.get("body", ""))
                break
            except Exception as ex:
                if ("rate" in repr(ex).lower() or "429" in repr(ex)) and att < 4:
                    print(f"  rate-limited, backing off 30s ({att + 1}/5)", flush=True)
                    time.sleep(30); continue
                print(f"  error on {c['ck'][:8]}: {ex}", flush=True)
                sc[f"mistral-large:{c['ck']}"] = None
                break
        done += 1
        json.dump(sc, open(CACHE, "w"))   # checkpoint every call -- n=55, cheap to be safe
        print(f"  [{done}/{len(todo)}] {c['h'][:60]}", flush=True)
        time.sleep(PACE)

    print(f"\ndone — {done} scored, cache saved to {CACHE}", flush=True)


if __name__ == "__main__":
    main()
