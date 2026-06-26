"""
groq_vs_ollama_backtest.py — the live A/B (groq_vs_ollama_pnl.py) needs ~2 weeks to settle. This is
the BACKTEST version: same question (would PAID Groq score live signals as well as / better than
local Ollama, as the PRIMARY scorer?) answered NOW on the historical pool.

Universe = every candidate article with an Ollama score (unified_scores) AND a forward return
(yahoo_candpool) — incl. Ollama-neutral/bearish ones, so "Groq-only" trades (Ollama skipped, Groq
would buy) are measurable, not just the bullish overlap. Each model "trades" an article when it
scores bullish & magnitude ≥ 0.75 (the live news_call gate); P&L = the trade's forward STOCK return.
Scores the universe with each GROQ model single-article + faithful (bot.SYSTEM_PROMPT), reusing
prior caches. Incremental cache → resumable. Report runs on whatever is scored (REPORT_ONLY=1).

Usage:  USE_YAHOO_BARS=1 .venv/bin/python -u groq_vs_ollama_backtest.py
        REPORT_ONLY=1 USE_YAHOO_BARS=1 .venv/bin/python -u groq_vs_ollama_backtest.py
"""
import os, json, time, statistics, threading
os.environ.setdefault("USE_YAHOO_BARS", "1")
import numpy as np
from openai import OpenAI
import gemini_lotto_pnl as gl
import confirm_veto_sweep as cvs

# (display, model_id, pace_seconds, extra_params) — pace tuned to each model's RPM headroom.
# llama-4-scout DROPPED 2026-06-24 (Groq deprecating it); its 1,287 cached scores stay for reference.
# llama-3.1-8b DROPPED 2026-06-25 (Groq deprecating it); replaced by gpt-oss-20b (Groq's recommended
# successor). Its prior scores stay cached but are no longer gathered/reported.
# gpt-oss-120b ADDED 2026-06-24 with reasoning_effort=low → 100% clean parse @ ~565ms (validated).
MODELS = [("llama-3.3-70b", "llama-3.3-70b-versatile", 3, None),                   # ~1k/day, low RPM
          ("gpt-oss-20b",   "openai/gpt-oss-20b",      8, {"reasoning_effort": "low"}),  # 8b replacement
          ("gpt-oss-120b",  "openai/gpt-oss-120b",     8, {"reasoning_effort": "low"})]
          # gpt-oss limits: 30 RPM, 1k RPD, 8k TPM, 200k TPD · ~967 tok/req → TPM binds at ~8/min (8s
          # pace) and TPD caps at ~207/day → full ~1.7k-article set ≈ 8-9 days of resets via scheduler.
CACHE = "groq_primary_backtest_cache.json"
MAG, HORIZON = 0.75, "r3"


def universe():
    uni = json.load(open("unified_scores.json")); cp = json.load(open("yahoo_candpool_cache.json"))
    gl.SAMPLE = 100000
    out = []
    for c in gl.candidates():
        u = uni.get("unified_v1:" + c["ck"]); fr = cp.get(f"{c['tk']}_{c['ck']}")
        if isinstance(u, dict) and fr and fr.get("px", 0) >= 5 and c.get("h"):
            out.append((c, u, fr))
    out.sort(key=lambda x: x[0]["ck"])
    return out


def seed(sc):
    """Reuse Groq scores already computed by prior runs to cut fresh calls."""
    def take(path, prefix, model):
        if not os.path.exists(path):
            return
        for k, v in json.load(open(path)).items():
            if v and k.startswith(prefix):
                sc.setdefault(f"{model}:{k.split(':', 1)[1]}", v)
    take("overnight_cv_cache.json",       "groq/llama-4-scout:",      "llama-4-scout")
    take("confirm_veto_sweep_cache.json", "groq/llama-4-scout:",      "llama-4-scout")
    take("confirm_veto_sweep_cache.json", "groq/llama-3.3-70b:",      "llama-3.3-70b")
    take("groq_value_cache.json",         "llama-3.3-70b-versatile:", "llama-3.3-70b")
    take("confirm_veto_sweep_cache.json", "groq/gpt-oss-20b:",        "gpt-oss-20b")   # 8b replacement
    take("overnight_cv_cache.json",       "groq/gpt-oss-20b:",        "gpt-oss-20b")
    take("overnight_cv_cache.json",       "groq/gpt-oss-120b:",       "gpt-oss-120b")
    take("confirm_veto_sweep_cache.json", "groq/gpt-oss-120b:",       "gpt-oss-120b")
    take("groq_value_cache.json",         "openai/gpt-oss-120b:",     "gpt-oss-120b")
    return sc


def ot(u):   # ollama trade decision (unified_scores schema)
    return u.get("sentiment") == "bullish" and float(u.get("magnitude", 0) or 0) >= MAG


def gt(g):   # groq trade decision (cvs.score_one schema {sent,mag})
    return g and g.get("sent") == "bullish" and float(g.get("mag", 0) or 0) >= MAG


def stat(name, xs):
    if not xs:
        print(f"  {name:34} n=0"); return
    n = len(xs); tot = sum(xs); avg = tot / n
    sd = statistics.pstdev(xs) if n > 1 else 0
    sh = avg / sd if sd else 0
    win = sum(1 for x in xs if x > 0) / n * 100
    print(f"  {name:34} n={n:<4} Σ{tot:>+8.1f}% avg{avg:>+6.2f}% win{win:>4.0f}% sharpe{sh:>+5.2f}")


def report(univ, sc):
    o_pnl = [fr[HORIZON] for c, u, fr in univ if ot(u)]
    print(f"\n  ══ PRIMARY-SCORER BACKTEST — forward {HORIZON} P&L of each scorer's trades (mag≥{MAG}) ══")
    print(f"  universe {len(univ)} articles · trade = bullish & mag≥{MAG}")
    print(f"\n  ── A) FIXED threshold (both at mag≥{MAG}) — shows the magnitude-scale offset ──")
    stat("OLLAMA-primary (baseline)", o_pnl)
    for disp, mid, *_ in MODELS:
        scored = [(c, u, fr) for c, u, fr in univ if sc.get(f"{disp}:{c['ck']}")]
        cov = len(scored)
        g_pnl = [fr[HORIZON] for c, u, fr in scored if gt(sc[f"{disp}:{c['ck']}"])]
        o_only = [fr[HORIZON] for c, u, fr in scored if ot(u) and not gt(sc[f"{disp}:{c['ck']}"])]
        g_only = [fr[HORIZON] for c, u, fr in scored if gt(sc[f"{disp}:{c['ck']}"]) and not ot(u)]
        print(f"\n  [{disp}]  scored {cov}/{len(univ)} ({cov/len(univ)*100:.0f}%)")
        stat(f"GROQ-primary [{disp}]", g_pnl)
        stat(f"  └ {disp}-only (Ollama skips → MISSED?)", g_only)
        stat(f"  └ Ollama-only ({disp} skips → AVOIDED?)", o_only)

    # B) MATCHED selectivity: each scorer trades its OWN top-N bullish picks by magnitude, N = the
    #    Ollama trade count within the COMMON scored set → isolates pick QUALITY from the mag-scale offset.
    print(f"\n  ── B) MATCHED selectivity (each scorer's top-N bullish by its OWN magnitude) ──")
    print(f"     (fair pick-quality test; meaningful once coverage ≈100% — needs the full universe scored)")
    for disp, mid, *_ in MODELS:
        scored = [(c, u, fr) for c, u, fr in univ if sc.get(f"{disp}:{c['ck']}")]
        if not scored:
            continue
        n = sum(1 for c, u, fr in scored if ot(u))            # match Ollama's count on the common set
        o_match = [fr[HORIZON] for c, u, fr in scored if ot(u)]
        def gmag(c):
            return float(sc[f"{disp}:{c['ck']}"].get("mag", 0) or 0)
        g_bull = [(c, u, fr) for c, u, fr in scored if sc[f"{disp}:{c['ck']}"].get("sent") == "bullish"]
        g_bull.sort(key=lambda x: gmag(x[0]), reverse=True)
        g_top = [fr[HORIZON] for c, u, fr in g_bull[:n]]
        take = min(n, len(g_bull))
        thr = gmag(g_bull[take - 1][0]) if take > 0 else 0
        note = f"≈mag≥{thr:.2f}" if len(g_bull) >= n else f"only {len(g_bull)} bullish picks vs N={n}"
        print(f"\n  [{disp}]  matched N={n} trades (common scored set {len(scored)})")
        stat("Ollama top-N (mag≥0.75)", o_match)
        stat(f"{disp} top-N ({note})", g_top)
    print("\n  Read A) for the magnitude-scale offset (Groq scores lower → skips most Ollama trades at a")
    print("  fixed bar; if Ollama-only is POSITIVE, Groq's caution would forgo good trades). Read B) for")
    print("  the real question: at equal trade COUNT, whose picks earn more? Higher = better stock picker.")


def main():
    univ = universe()
    sc = json.load(open(CACHE)) if os.path.exists(CACHE) else {}
    sc = seed(sc); json.dump(sc, open(CACHE, "w"))
    have = sum(1 for c, u, fr in univ for d, *_ in MODELS if sc.get(f"{d}:{c['ck']}"))
    print(f"universe {len(univ)} · model-scores cached {have}/{len(univ)*len(MODELS)} "
          f"(seeded from prior runs)", flush=True)

    if os.environ.get("REPORT_ONLY") == "1":
        report(univ, sc); return

    # Groq rate-limits PER MODEL, so score both models CONCURRENTLY (one thread each) — each model's
    # request stream stays under its own limit instead of one model hogging while the other idles.
    lock = threading.Lock()

    def worker(disp, mid, pace, extra):
        cli = OpenAI(base_url=cvs.PROV["groq"][0], api_key=cvs.PROV["groq"][1])  # own client per thread
        todo = [(c, u, fr) for c, u, fr in univ if f"{disp}:{c['ck']}" not in sc]
        print(f"▶ {disp}: {len(todo)} to score (pace {pace}s, concurrent)", flush=True)
        consec_rl, dry_rounds, done = 0, 0, 0
        for c, u, fr in todo:
            res, rl = None, False
            for att in range(5):
                try:
                    res = cvs.score_one(cli, mid, "groq", c["h"], c.get("body", ""), extra); break
                except Exception as ex:
                    if "rate" in repr(ex).lower() or "429" in repr(ex):
                        rl = True
                        if att < 4:
                            time.sleep(15); continue
                    res = None; break
            if res is None and rl:                  # rate-limited → leave UNSCORED, but PERSIST (don't exit)
                consec_rl += 1
                if consec_rl >= 12:                 # hit the per-minute (TPM) wall → long backoff, KEEP GOING
                    dry_rounds += 1                 # consecutive backoffs with NO success = daily cap (TPD) gone
                    if dry_rounds >= 8:             # ~16 min of pure throttle → cap really exhausted → stop
                        print(f"⏸ {disp}: daily cap exhausted ({done} scored this run) — rest UNSCORED "
                              f"until next reset", flush=True)
                        break
                    with lock:
                        json.dump(sc, open(CACHE, "w"))      # checkpoint progress before the wait
                    print(f"  ⏳ {disp}: throttle wall (backoff {dry_rounds}/8, {done} scored) — pausing 120s "
                          f"for the TPM window to free, then resuming…", flush=True)
                    time.sleep(120)
                    consec_rl = 0
                continue
            consec_rl, dry_rounds = 0, 0            # a SUCCESS resets both → captures quota in bursts all day
            with lock:
                sc[f"{disp}:{c['ck']}"] = res       # real score, or a genuine (non-rate-limit) parse-fail None
                done += 1
                if done % 20 == 0:
                    json.dump(sc, open(CACHE, "w")); print(f"    {disp} {done} scored this run", flush=True)
            time.sleep(pace)
        with lock:
            json.dump(sc, open(CACHE, "w"))
        print(f"✓ {disp}: worker exit ({done} scored this run)", flush=True)

    threads = [threading.Thread(target=worker, args=(d, m, p, e), daemon=True) for d, m, p, e in MODELS]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    report(univ, sc)


if __name__ == "__main__":
    main()
