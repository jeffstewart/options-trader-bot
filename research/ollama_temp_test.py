"""
ollama_temp_test.py — is llama3.2's confidence collapse (0.82 on 97% of bullish signals) a
property of the MODEL, or an artifact of sampling it at temperature 0.1?

jeff's question (2026-07-29). It matters because llama is free: if temp 1.0 unlocks a usable
confidence dial, the local model becomes viable as a gate again. If it only adds jitter, temp 1.0
is strictly worse than 0.1 -- noisy AND uninformative -- and the "llama has no dial" verdict is
final.

WHY FOUR ARMS AND NOT ONE
Raising temperature GUARANTEES more distinct values; that alone proves nothing. The question is
whether the extra spread is INFORMATIVE (tracks something about the article) or JITTER (random
wobble around the mode). Those are indistinguishable from a single pass -- you need repeat
measurements of the SAME article to separate them:

  A  t=0.1  full corpus     prompt-matched baseline (see prompt note below)
  B  t=1.0  full corpus     the temperature question, isolated against A
  C  t=1.0  repeat subset   within-article noise floor at t=1.0
  D  t=0.1  repeat subset   within-article noise floor at t=0.1 (so the comparison is symmetric --
                            without this we would not know whether 0.1 is itself noisy)

The payoff statistic is TEST-RETEST CORRELATION between paired passes (B vs C, A vs D). r near 1
means a re-scored article lands in the same place, so between-article spread is real signal. r near
0 means the spread is noise regardless of how wide it looks.

PROMPT NOTE -- this run also fixes a confound in the existing cross-model table. The
`ollama:unified_v1` scores in data/unified_scores.json were produced with the unified_v1 prompt
(2,435 chars, with a `catalyst` output field) while haiku/sonnet-4-6/sonnet5/opus/kimi were all
scored with anthropic_scorer's prompt (1,100 chars, no catalyst field). That row therefore differed
from every other row on PROMPT as well as temperature. All arms here use anthropic_scorer's prompt,
so arm A is the row that belongs in the comparison table -- and A vs the old unified_v1 scores
isolates the prompt effect, which nobody has measured.

Free (local Ollama) and ~13 scores/min, so ~2.2h per full arm, ~5h for all four. Overnight job.

Usage (run from data/):
    python ../research/ollama_temp_test.py --preflight            # check Ollama is up
    python ../research/ollama_temp_test.py --arm A --sample 25    # smoke test one arm
    python ../research/ollama_temp_test.py --all                  # all four arms, overnight
    python ../research/ollama_temp_test.py --analyze              # the verdict
"""
import argparse
import json
import os
import math
import statistics
import sys
import time
from collections import Counter
from pathlib import Path

from openai import OpenAI

sys.path.insert(0, str(Path(__file__).resolve().parent))
# Same corpus + prompt as the Anthropic/Kimi runs -- imported, never copied.
from anthropic_scorer import SYSTEM_PROMPT, load_universe, _parse_raw  # noqa: E402

DATA_DIR   = Path("/Users/jeff/Claude/Trader/data")
CACHE_FILE = DATA_DIR / "ollama_temp_cache.json"
BASE_URL   = os.environ.get("OLLAMA_BASE_URL", "http://localhost:11434/v1")
MODEL      = os.environ.get("OLLAMA_MODEL", "llama3.2")

REPEAT_N = int(os.environ.get("REPEAT_N", "300"))   # subset size for the noise-floor arms

# arm -> (temperature, cache_prefix, n_articles or None for full)
ARMS = {
    "A": (0.1, "t01",   None),
    "B": (1.0, "t10",   None),
    "C": (1.0, "t10r2", REPEAT_N),   # paired with B
    "D": (0.1, "t01r2", REPEAT_N),   # paired with A
}
PAIRS = {"t=1.0": ("t10", "t10r2"), "t=0.1": ("t01", "t01r2")}


def load_cache() -> dict:
    return json.loads(CACHE_FILE.read_text()) if CACHE_FILE.exists() else {}


def save_cache(cache: dict) -> None:
    CACHE_FILE.write_text(json.dumps(cache, indent=2))


def _client() -> OpenAI:
    return OpenAI(base_url=BASE_URL, api_key="ollama", timeout=120.0)


def preflight() -> bool:
    """Ollama must be serving AND have the model pulled. A silent failure here would waste the
    whole overnight window."""
    try:
        c = _client()
        ids = [m.id for m in c.models.list().data]
        print(f"✅ Ollama reachable at {BASE_URL}; {len(ids)} models: {', '.join(ids[:6])}")
        if not any(MODEL in i for i in ids):
            print(f"❌ model {MODEL!r} not present — run: ollama pull {MODEL}")
            return False
        t0 = time.time()
        art = load_universe()[0]
        r = c.chat.completions.create(
            model=MODEL, max_tokens=200, temperature=0.1,
            messages=[{"role": "system", "content": SYSTEM_PROMPT},
                      {"role": "user", "content": f"Headline: {art['headline']}\n\n"
                                                  f"Body: {art['summary']}\n\nRespond with JSON only."}])
        parsed = _parse_raw(r.choices[0].message.content or "")
        dt = time.time() - t0
        print(f"✅ test score in {dt:.1f}s, parsed={parsed is not None} → "
              f"ETA ~{1683 * dt / 60:.0f} min per full arm")
        return parsed is not None
    except Exception as e:
        print(f"❌ Ollama unreachable ({type(e).__name__}): {e}\n   Start it with: ollama serve")
        return False


def score_one(client: OpenAI, article: dict, temp: float) -> "tuple[dict | None, float]":
    t0 = time.time()
    try:
        r = client.chat.completions.create(
            model=MODEL, max_tokens=200, temperature=temp,
            messages=[{"role": "system", "content": SYSTEM_PROMPT},
                      {"role": "user", "content": f"Headline: {article['headline']}\n\n"
                                                  f"Body: {article['summary']}\n\n"
                                                  f"Respond with JSON only."}])
        return _parse_raw(r.choices[0].message.content or ""), time.time() - t0
    except Exception as e:
        print(f"  ⚠️  {type(e).__name__}: {str(e)[:120]}")
        return None, time.time() - t0


def run_arm(arm: str, limit: "int | None" = None) -> None:
    temp, prefix, arm_n = ARMS[arm]
    universe = load_universe()
    # The repeat arms must cover the SAME articles as their full-arm partner, so take the first
    # REPEAT_N of the hash-sorted universe -- deterministic, and identical across C and D.
    pool = universe[:arm_n] if arm_n else universe
    if limit:
        pool = pool[:limit]
    cache = load_cache()
    client = _client()
    todo = [a for a in pool if f"{prefix}:{a['hash']}" not in cache]
    print(f"\n── arm {arm}: temp={temp} prefix={prefix} · {len(todo)} to score "
          f"({len(pool) - len(todo)} cached of {len(pool)} in scope)")
    lat, ok, failed = [], 0, 0
    for i, art in enumerate(todo, 1):
        parsed, dt = score_one(client, art, temp)
        lat.append(dt)
        if parsed is not None:
            ok += 1
            cache[f"{prefix}:{art['hash']}"] = {
                "sent": parsed.get("sentiment", ""), "mag": parsed.get("magnitude", 0),
                "conf": parsed.get("confidence", 0), "tick": parsed.get("tickers", []),
                "why": parsed.get("reasoning", ""), "latency": round(dt, 3),
            }
        else:
            failed += 1
        if i % 25 == 0 or i == len(todo):
            save_cache(cache)
            print(f"   {i}/{len(todo)} ok={ok} parse-fail={failed} "
                  f"median={statistics.median(lat):.1f}s")
    save_cache(cache)
    if lat:
        print(f"   arm {arm} done: {ok} scored, {failed} parse failures "
              f"({100 * failed / max(len(todo), 1):.1f}%), median {statistics.median(lat):.1f}s")


def _finite(vals) -> list:
    """llama3.2 at t=1.0 occasionally emits a non-numeric confidence that json parses to nan
    (2 of 289 in arm B). One nan poisons every downstream statistic -- pstdev raises outright --
    so filter here rather than at each call site."""
    out = []
    for x in vals:
        try:
            f = float(x)
        except (TypeError, ValueError):
            continue
        if math.isfinite(f):
            out.append(f)
    return out


def _dist(vals: list) -> str:
    v = sorted(_finite(vals))
    if not v:
        return "n=0"
    mode, n_mode = Counter(round(x, 2) for x in v).most_common(1)[0]
    return (f"n={len(v):<5} med={statistics.median(v):.2f} "
            f"[{v[0]:.2f}-{v[-1]:.2f}] sd={statistics.pstdev(v):.3f} "
            f"distinct={len(set(v)):<3} mode={mode} holds {100 * n_mode / len(v):.0f}%")


def analyze() -> None:
    cache = load_cache()
    if not cache:
        return print("nothing cached yet")
    by_prefix: dict[str, dict[str, dict]] = {}
    for k, v in cache.items():
        p, _, h = k.partition(":")
        if v:
            by_prefix.setdefault(p, {})[h] = v

    print(f"\n{'=' * 90}\nARM DISTRIBUTIONS (llama3.2, anthropic_scorer prompt)\n{'=' * 90}")
    for arm, (temp, prefix, _) in ARMS.items():
        d = by_prefix.get(prefix, {})
        if not d:
            print(f"arm {arm} (t={temp}): nothing cached")
            continue
        bull = [v for v in d.values() if v.get("sent") == "bullish"]
        print(f"\narm {arm} (t={temp}, {prefix}): {len(d)} scored, {len(bull)} bullish "
              f"({100 * len(bull) / len(d):.0f}%)")
        if bull:
            print(f"   confidence  {_dist([float(v['conf'] or 0) for v in bull])}")
            print(f"   magnitude   {_dist([float(v['mag'] or 0) for v in bull])}")

    print(f"\n{'=' * 90}\nTEST-RETEST — is the spread signal or jitter?\n{'=' * 90}")
    for label, (pa, pb) in PAIRS.items():
        da, db = by_prefix.get(pa, {}), by_prefix.get(pb, {})
        common = sorted(set(da) & set(db))
        if len(common) < 30:
            print(f"{label}: only {len(common)} paired articles — need >=30, run arms "
                  f"{'B and C' if pa == 't10' else 'A and D'}")
            continue
        pairs = []
        for h in common:
            a, b = _finite([da[h].get("conf")]), _finite([db[h].get("conf")])
            if a and b:
                pairs.append((a[0], b[0]))
        dropped = len(common) - len(pairs)
        if dropped:
            print(f"   (dropped {dropped} pairs with a non-finite confidence)")
        if len(pairs) < 30:
            print(f"{label}: only {len(pairs)} usable pairs after filtering — skipping")
            continue
        x = [p[0] for p in pairs]
        y = [p[1] for p in pairs]
        common = [h for h in common if _finite([da[h].get("conf")]) and _finite([db[h].get("conf")])]
        mad = statistics.mean(abs(a - b) for a, b in zip(x, y))
        within = statistics.mean((a - b) ** 2 / 2 for a, b in zip(x, y)) ** 0.5
        between = statistics.pstdev(x)
        try:
            r = statistics.correlation(x, y)
        except Exception:
            r = float("nan")          # zero variance in one pass -> undefined, and that IS the answer
        agree = sum(1 for h in common if da[h]["sent"] == db[h]["sent"]) / len(common)
        print(f"\n{label}  (n={len(common)} paired)")
        print(f"   test-retest r        {r:+.3f}   (→1 reproducible · →0 pure noise)")
        print(f"   mean |Δconfidence|   {mad:.3f}")
        print(f"   within-article sd    {within:.3f}")
        print(f"   between-article sd   {between:.3f}")
        print(f"   signal-to-noise      {between / within:.2f}x" if within else
              "   signal-to-noise      ∞ (identical on repeat)")
        print(f"   sentiment agreement  {100 * agree:.0f}%")
    print("\nVERDICT GUIDE: temp 1.0 only 'unlocks a dial' if its between/within ratio AND its\n"
          "test-retest r beat temp 0.1's. A wider distribution with r≈0 is noise wearing a\n"
          "costume — that would settle the llama question against it for good.")

    old = DATA_DIR / "unified_scores.json"
    if old.exists() and by_prefix.get("t01"):
        uni = json.loads(old.read_text())
        d = by_prefix["t01"]
        common = [h for h in d if f"unified_v1:{h}" in uni]
        if len(common) >= 30:
            ap = [float(d[h]["conf"] or 0) for h in common if d[h].get("sent") == "bullish"]
            uv = [float((uni[f"unified_v1:{h}"] or {}).get("confidence") or 0) for h in common
                  if (uni[f"unified_v1:{h}"] or {}).get("sentiment") == "bullish"]
            print(f"\n{'=' * 90}\nPROMPT EFFECT (bonus): same model, same temp 0.1, different prompt"
                  f"\n{'=' * 90}")
            print(f"   anthropic_scorer prompt  {_dist(ap)}")
            print(f"   unified_v1 prompt        {_dist(uv)}")
            print("   (n differs because the two prompts disagree on WHICH articles are bullish)")


def main():
    ap = argparse.ArgumentParser()
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--preflight", action="store_true")
    g.add_argument("--arm", choices=list(ARMS))
    g.add_argument("--all", action="store_true")
    g.add_argument("--analyze", action="store_true")
    ap.add_argument("--sample", type=int, help="cap articles (smoke testing)")
    args = ap.parse_args()

    if args.preflight:
        sys.exit(0 if preflight() else 1)
    if args.analyze:
        return analyze()
    if not preflight():
        sys.exit("aborting — Ollama not ready")
    for arm in (list(ARMS) if args.all else [args.arm]):
        run_arm(arm, args.sample)
    analyze()


if __name__ == "__main__":
    main()
