"""
prompt_v2_test.py — can task-RESHAPING beat the unified_v1 float-anchoring failure on local llama3.2?

The v1 audit (2026-07-16) showed the model echoes the prompt's schema examples (77% of confidence
outputs are literally 0.82, magnitude modes on 0.75), the catalyst float is uninformative (winners
0.860 = losers 0.860), and free ticker generation invents/mismatches symbols. v2 reshapes each field
into a task a 3B model might actually do:
  ticker    → SELECTION from the feed's own symbols metadata (multiple choice, "NONE" allowed)
  magnitude → categorical low/medium/high (no float to anchor on)
  resolved  → forced binary (no float to hedge into the middle)
  confidence → DROPPED (was an echo)

Sample = (A) the ~139 articles where sonnet5 and v1-Ollama disagreed outright on ticker (the hard
attribution cases; sonnet5 = reference standard, its pick sits in the feed candidates 98% of the
time) + (B) ~200 random articles with forward returns (r3) for the resolved/magnitude validation.

Validation:
  ticker    — agreement with sonnet5 on set A (v1 scored ~0% by construction) + overall
  resolved  — r3 of bullish resolved=true vs resolved=false (v1's catalyst float: no separation)
  magnitude — r3 monotonicity across low/medium/high
Plus parse-fail rate + latency. Scores cached incrementally (data/prompt_v2_cache.json, "v2:{ck}").

Usage:  .venv/bin/python -u ../research/prompt_v2_test.py            (from data/; scores + report)
        REPORT_ONLY=1 .venv/bin/python -u ../research/prompt_v2_test.py
"""
import ast, json, os, random, re, statistics, time

from openai import OpenAI

OLLAMA_URL = os.environ.get("OLLAMA_BASE_URL", "http://localhost:11434/v1")
MODEL      = os.environ.get("OLLAMA_MODEL", "llama3.2")
CACHE      = "prompt_v2_cache.json"
N_RANDOM   = 200
SEED       = 20260716

V2_PROMPT = """You are a financial news analyst. Respond with a JSON object ONLY — no markdown, no explanation.

Schema:
{
  "ticker":    "<one symbol from CANDIDATES, or NONE>",
  "sentiment": "bullish" | "bearish" | "neutral",
  "magnitude": "low" | "medium" | "high",
  "resolved":  true | false
}

ticker: the ONE symbol from CANDIDATES whose company is the PRIMARY SUBJECT and the party that
  BENEFITS (or suffers) most directly from this news. If the news is not mainly about any candidate,
  answer "NONE". Never output a symbol that is not in CANDIDATES.
sentiment: direction for the chosen ticker's stock. If ticker is NONE, use "neutral".
magnitude:
  low    = routine or noise (minor updates, opinions, recaps, listicles, analyst ratings)
  medium = meaningful catalyst (solid earnings beat, notable partnership, guidance change)
  high   = major, company-defining event (transformative deal, landmark approval, huge beat)
resolved: true ONLY if the event has ALREADY HAPPENED and removed uncertainty — earnings REPORTED,
  approval GRANTED, deal SIGNED, contract WON, data RELEASED. false for anything anticipated,
  speculative, or opinion — previews, "ahead of", "in talks", "could/may/plans to", analyst views,
  price targets, predictions."""


# ── Ticker-only DEPLOYMENT variant (TICKER_ONLY=1) ────────────────────────────────────────────────
# The 85% result above came from the 4-field prompt; the live post-score check would ask ONLY the
# ticker question (dead fields dropped, "losing side" made explicit — the BEAM arbitration case).
# Scored under separate cache keys ("v2t:") so both variants coexist for comparison.
V2T_PROMPT = """You are a financial news analyst. Respond with a JSON object ONLY — no markdown, no explanation.

Schema:
{ "ticker": "<one symbol from CANDIDATES, or NONE>" }

ticker: the ONE symbol from CANDIDATES whose company is the PRIMARY SUBJECT of this news AND the
party that most directly benefits or suffers from it. A company that is merely mentioned, or that
is on the LOSING side of the event (lost the lawsuit, is being outcompeted, is the acquirer paying
a premium), is NOT the answer. If the news is not mainly about any candidate, answer "NONE".
Never output a symbol that is not in CANDIDATES."""


def score_v2t(cli, art):
    t0 = time.time()
    raw = cli.chat.completions.create(
        model=MODEL, temperature=0.1, max_tokens=60,
        messages=[{"role": "system", "content": V2T_PROMPT},
                  {"role": "user", "content": f"Headline: {art['h']}\n\nBody: {art['body']}\n\n"
                                              f"CANDIDATES: {art['syms']}\n\nRespond with JSON only."}],
    ).choices[0].message.content.strip()
    lat = time.time() - t0
    if raw.startswith("```"):
        raw = raw.split("```")[1]
        if raw.startswith("json"):
            raw = raw[4:]
    m = re.search(r"\{.*\}", raw, re.S)
    if not m:
        return None, lat
    try:
        o = json.loads(m.group(0))
    except json.JSONDecodeError:
        return None, lat
    tk = o.get("ticker")
    tk = tk.strip().upper() if isinstance(tk, str) else "NONE"
    if tk != "NONE" and tk not in art["syms"]:
        tk = f"INVALID:{tk}"
    return {"tk": tk}, lat


def report_ticker_only(set_a, sc):
    print("\n══ TICKER-ONLY deployment prompt vs 4-field tested prompt (set A hard cases) ══")
    rows = [(a, sc.get(f"v2t:{a['ck']}"), sc.get(f"v2:{a['ck']}")) for a in set_a]
    rows = [(a, t, f) for a, t, f in rows if t]
    agree_t = sum(1 for a, t, f in rows if t["tk"] == a["s5_tk"])
    agree_f = sum(1 for a, t, f in rows if f and f["tk"] == a["s5_tk"])
    none_t = sum(1 for a, t, f in rows if t["tk"] == "NONE")
    inval_t = sum(1 for a, t, f in rows if str(t["tk"]).startswith("INVALID"))
    print(f"  n={len(rows)}")
    print(f"  ticker-only matches sonnet5: {agree_t}/{len(rows)} ({agree_t/len(rows)*100:.0f}%)")
    print(f"  4-field   matches sonnet5:   {agree_f}/{len(rows)} ({agree_f/len(rows)*100:.0f}%)")
    print(f"  ticker-only NONE: {none_t} · invalid: {inval_t}")
    flips_bad = [(a, t, f) for a, t, f in rows if f and f["tk"] == a["s5_tk"] and t["tk"] != a["s5_tk"]]
    flips_good = [(a, t, f) for a, t, f in rows if f and f["tk"] != a["s5_tk"] and t["tk"] == a["s5_tk"]]
    print(f"  flips: 4-field-right→ticker-only-wrong {len(flips_bad)} · wrong→right {len(flips_good)}")
    for a, t, f in flips_bad[:6]:
        print(f"    lost:  v2t={str(t['tk']):10} s5={a['s5_tk']:6} {a['h'][:60]}")
    lats = [v for k, v in sc.items() if k.startswith("latt:")]
    if lats:
        print(f"  latency: avg {statistics.mean(lats):.1f}s  p50 {statistics.median(lats):.1f}s")


def load_full_sets():
    """FULL_TEST sets: every article with feed candidates + a sonnet5 reference.
    easy  = v1-Ollama already agreed with sonnet5 (selection must not BREAK these)
    hard  = outright disagreement (set A superset — includes s5-pick-not-in-candidates)
    none  = sonnet5 emitted NO ticker (selection must answer NONE, not force-pick)"""
    dual = json.load(open("dual_score_cache.json"))
    anth = json.load(open("anthropic_backtest_cache.json"))
    easy, hard, none_set = [], [], []
    for k, v in anth.items():
        if not k.startswith("sonnet5:") or not v:
            continue
        ck = k.split(":", 1)[1]
        dv = dual.get(ck)
        if not isinstance(dv, dict):
            continue
        a = dv.get("_article", {}) or {}
        b = dv.get("bullish", {}) or {}
        h = (a.get("headline") or "").strip()
        try:
            syms = ast.literal_eval(a.get("symbols")) if isinstance(a.get("symbols"), str) else (a.get("symbols") or [])
        except Exception:
            syms = []
        syms = [s for s in syms if isinstance(s, str) and s.isupper() and 1 <= len(s) <= 5]
        if not h or not syms:
            continue
        o_tks = [t for t in (b.get("tickers") or []) if t not in ("BTC", "ETH")]
        s5_ticks = [t for t in (v.get("tick") or []) if isinstance(t, str)]
        art = {"ck": ck, "h": h, "body": (a.get("summary") or "")[:1200], "syms": syms[:8],
               "o_tk": o_tks[0] if o_tks else None, "s5_tk": s5_ticks[0] if s5_ticks else None}
        if not s5_ticks:
            none_set.append(art)
        elif art["s5_tk"] not in art["syms"]:
            continue                        # no in-candidates reference → unmeasurable, skip
        elif o_tks and o_tks[0] == art["s5_tk"]:
            easy.append(art)
        else:
            hard.append(art)
    rng = random.Random(SEED)
    none_sample = rng.sample(none_set, min(250, len(none_set)))
    return easy, hard, none_sample


def report_full(easy, hard, none_sample, sc):
    print("\n══ FULL TEST — ticker-only deployment prompt vs sonnet5 reference ══")
    for name, arts in (("EASY (v1 already right)", easy), ("HARD (v1 wrong)", hard)):
        rows = [(a, sc.get(f"v2t:{a['ck']}")) for a in arts]
        rows = [(a, t) for a, t in rows if t]
        if not rows:
            print(f"  {name}: n=0")
            continue
        agree = sum(1 for a, t in rows if t["tk"] == a["s5_tk"])
        none_r = sum(1 for a, t in rows if t["tk"] == "NONE")
        print(f"  {name:24} n={len(rows):<4} matches sonnet5 {agree}/{len(rows)} ({agree/len(rows)*100:.0f}%)  NONE={none_r}")
        misses = [(a, t) for a, t in rows if t["tk"] != a["s5_tk"] and t["tk"] != "NONE"][:5]
        for a, t in misses:
            print(f"      miss: v2t={str(t['tk']):8} s5={a['s5_tk']:6} v1={str(a['o_tk']):6} {a['h'][:58]}")
    rows = [(a, sc.get(f"v2t:{a['ck']}")) for a in none_sample]
    rows = [(a, t) for a, t in rows if t]
    if rows:
        none_r = sum(1 for a, t in rows if t["tk"] == "NONE")
        print(f"  {'NO-TICKER (s5 said none)':24} n={len(rows):<4} v2t answered NONE {none_r}/{len(rows)} ({none_r/len(rows)*100:.0f}%)")
        forced = [(a, t) for a, t in rows if t["tk"] != "NONE"][:6]
        for a, t in forced:
            print(f"      force-picked: v2t={str(t['tk']):8} {a['h'][:64]}")
    lats = [v for k, v in sc.items() if k.startswith("latt:")]
    if lats:
        print(f"  latency: avg {statistics.mean(lats):.1f}s  p50 {statistics.median(lats):.1f}s")


def load_sample():
    dual = json.load(open("dual_score_cache.json"))
    anth = json.load(open("anthropic_backtest_cache.json"))
    cp   = json.load(open("yahoo_candpool_cache.json"))

    arts = {}
    for ck, v in dual.items():
        if not isinstance(v, dict):
            continue
        a = v.get("_article", {}) or {}
        b = v.get("bullish", {}) or {}
        h = (a.get("headline") or "").strip()
        try:
            syms = ast.literal_eval(a.get("symbols")) if isinstance(a.get("symbols"), str) else (a.get("symbols") or [])
        except Exception:
            syms = []
        syms = [s for s in syms if isinstance(s, str) and s.isupper() and 1 <= len(s) <= 5]
        if not h or not syms:
            continue
        o_tks = [t for t in (b.get("tickers") or []) if t not in ("BTC", "ETH")]
        s5 = anth.get(f"sonnet5:{ck}")
        s5_t = None
        if s5 and isinstance(s5.get("tick"), list) and s5["tick"]:
            s5_t = s5["tick"][0] if isinstance(s5["tick"][0], str) else None
        r3 = None
        if o_tks:
            fr = cp.get(f"{o_tks[0]}_{ck}")
            if isinstance(fr, dict):
                r3 = fr.get("r3")
        arts[ck] = {"ck": ck, "h": h, "body": (a.get("summary") or "")[:1200], "syms": syms[:8],
                    "o_tk": o_tks[0] if o_tks else None, "s5_tk": s5_t, "r3": r3,
                    "v1_mag": float(b.get("magnitude", 0) or 0)}

    # Set A: sonnet5-vs-v1 outright ticker disagreements where sonnet5's pick is a valid candidate
    set_a = [a for a in arts.values()
             if a["o_tk"] and a["s5_tk"] and a["s5_tk"] != a["o_tk"] and a["s5_tk"] in a["syms"]]
    # Set B: random articles with a forward return (for resolved/magnitude validation)
    pool_b = [a for a in arts.values() if a["r3"] is not None and a not in set_a]
    rng = random.Random(SEED)
    set_b = rng.sample(pool_b, min(N_RANDOM, len(pool_b)))
    return set_a, set_b


def score_v2(cli, art):
    t0 = time.time()
    raw = cli.chat.completions.create(
        model=MODEL, temperature=0.1, max_tokens=200,
        messages=[{"role": "system", "content": V2_PROMPT},
                  {"role": "user", "content": f"Headline: {art['h']}\n\nBody: {art['body']}\n\n"
                                              f"CANDIDATES: {art['syms']}\n\nRespond with JSON only."}],
    ).choices[0].message.content.strip()
    lat = time.time() - t0
    if raw.startswith("```"):
        raw = raw.split("```")[1]
        if raw.startswith("json"):
            raw = raw[4:]
    m = re.search(r"\{.*\}", raw, re.S)
    if not m:
        return None, lat
    try:
        o = json.loads(m.group(0))
    except json.JSONDecodeError:
        return None, lat
    tk = o.get("ticker")
    tk = tk.strip().upper() if isinstance(tk, str) else "NONE"
    if tk != "NONE" and tk not in art["syms"]:
        tk = f"INVALID:{tk}"          # emitted a non-candidate — count it, don't hide it
    res = o.get("resolved")
    if isinstance(res, str):
        res = res.strip().lower() == "true"
    return {"tk": tk, "sent": str(o.get("sentiment", "")).lower(),
            "mag": str(o.get("magnitude", "")).lower(), "res": bool(res)}, lat


def report(set_a, set_b, sc):
    print("\n══ 1) TICKER — selection-from-candidates vs sonnet5 reference (set A: v1 hard cases) ══")
    rows = [(a, sc[f"v2:{a['ck']}"]) for a in set_a if sc.get(f"v2:{a['ck']}")]
    agree = sum(1 for a, v in rows if v["tk"] == a["s5_tk"])
    none_r = sum(1 for a, v in rows if v["tk"] == "NONE")
    inval = sum(1 for a, v in rows if str(v["tk"]).startswith("INVALID"))
    v1_agree = sum(1 for a, v in rows if a["o_tk"] == a["s5_tk"])   # 0 by construction
    print(f"  n={len(rows)} hard cases (v1-Ollama agreed with sonnet5 on {v1_agree} — 0 by construction)")
    if rows:
        print(f"  v2-Ollama matches sonnet5: {agree}/{len(rows)} ({agree/len(rows)*100:.0f}%)")
        print(f"  v2 answered NONE: {none_r} · emitted non-candidate: {inval}")
        print(f"\n  sample of remaining misses:")
        n = 0
        for a, v in rows:
            if v["tk"] != a["s5_tk"] and n < 8:
                n += 1
                print(f"    v2={str(v['tk']):14} s5={a['s5_tk']:6} v1={a['o_tk']:6} {a['h'][:62]}")

    print("\n══ 2) RESOLVED binary — forward r3 separation (set B bullish; v1 catalyst float = none) ══")
    rows_b = [(a, sc[f"v2:{a['ck']}"]) for a in set_b if sc.get(f"v2:{a['ck']}")]
    bull = [(a, v) for a, v in rows_b if v["sent"] == "bullish" and a["r3"] is not None]
    for label, flt in [("resolved=TRUE", lambda v: v["res"]), ("resolved=FALSE", lambda v: not v["res"])]:
        xs = [a["r3"] for a, v in bull if flt(v)]
        if xs:
            win = sum(1 for x in xs if x > 0) / len(xs) * 100
            print(f"  {label:16} n={len(xs):<4} avg r3 {statistics.mean(xs):+.2f}%  med {statistics.median(xs):+.2f}%  win {win:.0f}%")
        else:
            print(f"  {label:16} n=0")

    print("\n══ 3) MAGNITUDE categorical — r3 monotonicity (set B bullish) ══")
    for m in ("low", "medium", "high"):
        xs = [a["r3"] for a, v in bull if v["mag"] == m]
        if xs:
            win = sum(1 for x in xs if x > 0) / len(xs) * 100
            print(f"  {m:8} n={len(xs):<4} avg r3 {statistics.mean(xs):+.2f}%  med {statistics.median(xs):+.2f}%  win {win:.0f}%")
        else:
            print(f"  {m:8} n=0")

    print("\n══ 4) hygiene ══")
    all_rows = rows + rows_b
    sents = statistics.mean(1 for _ in all_rows) if all_rows else 0
    fails = sum(1 for a in set_a + set_b if sc.get(f"v2:{a['ck']}", "miss") is None)
    print(f"  parse failures: {fails}/{len(set_a)+len(set_b)}")
    lats = [v for k, v in sc.items() if k.startswith("lat:")]
    if lats:
        print(f"  latency: avg {statistics.mean(lats):.1f}s  p50 {statistics.median(lats):.1f}s")


def main():
    set_a, set_b = load_sample()
    print(f"set A (ticker hard cases): {len(set_a)} · set B (random w/ r3): {len(set_b)}")
    sc = json.load(open(CACHE)) if os.path.exists(CACHE) else {}

    if os.environ.get("FULL_TEST") == "1":
        easy, hard, none_sample = load_full_sets()
        print(f"FULL TEST — easy: {len(easy)} · hard: {len(hard)} · no-ticker sample: {len(none_sample)}")
        if os.environ.get("REPORT_ONLY") != "1":
            cli = OpenAI(base_url=OLLAMA_URL, api_key="ollama")
            todo = [a for a in easy + hard + none_sample if f"v2t:{a['ck']}" not in sc]
            print(f"to score: {len(todo)}")
            for i, art in enumerate(todo):
                try:
                    res, lat = score_v2t(cli, art)
                except Exception as e:
                    print(f"  [{i+1}] error: {e}", flush=True)
                    continue
                sc[f"v2t:{art['ck']}"] = res
                sc[f"latt:{art['ck']}"] = round(lat, 2)
                if (i + 1) % 50 == 0:
                    json.dump(sc, open(CACHE, "w"))
                    print(f"  {i+1}/{len(todo)}", flush=True)
            json.dump(sc, open(CACHE, "w"))
        report_full(easy, hard, none_sample, sc)
        return

    if os.environ.get("TICKER_ONLY") == "1":
        if os.environ.get("REPORT_ONLY") != "1":
            cli = OpenAI(base_url=OLLAMA_URL, api_key="ollama")
            todo = [a for a in set_a if f"v2t:{a['ck']}" not in sc]
            print(f"ticker-only variant — to score: {len(todo)}")
            for i, art in enumerate(todo):
                try:
                    res, lat = score_v2t(cli, art)
                except Exception as e:
                    print(f"  [{i+1}] error: {e}", flush=True)
                    continue
                sc[f"v2t:{art['ck']}"] = res
                sc[f"latt:{art['ck']}"] = round(lat, 2)
                if (i + 1) % 25 == 0:
                    json.dump(sc, open(CACHE, "w"))
                    print(f"  {i+1}/{len(todo)}", flush=True)
            json.dump(sc, open(CACHE, "w"))
        report_ticker_only(set_a, sc)
        return

    if os.environ.get("REPORT_ONLY") != "1":
        cli = OpenAI(base_url=OLLAMA_URL, api_key="ollama")
        todo = [a for a in set_a + set_b if f"v2:{a['ck']}" not in sc]
        print(f"to score: {len(todo)} (llama3.2 via {OLLAMA_URL})")
        for i, art in enumerate(todo):
            try:
                res, lat = score_v2(cli, art)
            except Exception as e:
                print(f"  [{i+1}] error: {e}", flush=True)
                continue
            sc[f"v2:{art['ck']}"] = res
            sc[f"lat:{art['ck']}"] = round(lat, 2)
            if (i + 1) % 20 == 0:
                json.dump(sc, open(CACHE, "w"))
                print(f"  {i+1}/{len(todo)} scored ({lat:.1f}s/call)", flush=True)
        json.dump(sc, open(CACHE, "w"))

    report(set_a, set_b, sc)


if __name__ == "__main__":
    main()
