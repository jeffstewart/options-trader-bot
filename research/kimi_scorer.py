"""
kimi_scorer.py — score the SAME backtest article universe with Moonshot's Kimi models, so
Kimi/Ollama/Groq/Anthropic results are directly comparable.

WHY THIS IMPORTS FROM anthropic_scorer INSTEAD OF COPYING IT
jeff's requirement (2026-07-29) was an HONEST cross-model comparison, not a new dataset. Every
apples-to-apples property depends on three things being byte-identical to the Anthropic run:
the article universe, the prompt, and the cache-key scheme. Re-implementing any of them invites
silent drift (a slightly different noise filter, a reworded prompt, a different price floor) that
would show up as a "model difference". So this module imports `load_universe`, `SYSTEM_PROMPT`,
`cache_key` and `_parse_raw` from anthropic_scorer rather than restating them. If that file's
universe filter changes, this scorer follows automatically.

Deliberately NOT the unified_v1 prompt that v2/scoring.py uses live: the historical corpus was
scored with anthropic_scorer's prompt (no `catalyst` field), and matching the comparison set beats
matching production here. A separate live-prompt run is a different experiment.

Output: data/kimi_backtest_cache.json, keyed `{alias}:{article_hash}` with the same value schema
as anthropic_backtest_cache.json ({sent, mag, conf, tick, why, latency}), so existing loaders
(e.g. small_account_sonnet5_sweep.load_sonnet5_rows) work by swapping the cache path + prefix.

Kimi's API is OpenAI-compatible (https://api.moonshot.ai/v1), so this uses the `openai` SDK
already in the venv -- no new dependency.

Usage (run from data/, same convention as anthropic_scorer.py):
    python ../research/kimi_scorer.py --dry-run                    # cost estimate only
    python ../research/kimi_scorer.py --sample 50                  # go/no-go on a small sample
    python ../research/kimi_scorer.py --all                        # score everything uncached
    python ../research/kimi_scorer.py --report                     # stats on what's cached
    python ../research/kimi_scorer.py --sample 50 --models k2.6 k3
    python ../research/kimi_scorer.py --all --thinking      # reasoning mode (slow, ~12x cost)
    python ../research/kimi_scorer.py --compare             # vs Anthropic + Ollama

Requires MOONSHOT_API_KEY in .env (sign up: https://platform.moonshot.ai — prepaid credits).
"""
import argparse
import json
import os
import statistics
import sys
import time
from pathlib import Path

from dotenv import load_dotenv
from openai import OpenAI

sys.path.insert(0, str(Path(__file__).resolve().parent))
# Single source of truth for corpus + prompt -- see module docstring.
from anthropic_scorer import (SYSTEM_PROMPT, cache_key, load_universe,  # noqa: E402
                              _parse_raw)

for _env in (Path(".env"), Path("../.env"), Path(__file__).resolve().parent.parent / ".env"):
    if _env.exists():
        load_dotenv(_env)
        break

DATA_DIR   = Path("/Users/jeff/Claude/Trader/data")
CACHE_FILE = DATA_DIR / "kimi_backtest_cache.json"
BASE_URL   = "https://api.moonshot.ai/v1"

# (model_id, $/1M input, $/1M output). The ID list is what `GET /v1/models` returned for jeff's
# actual key on 2026-07-29 -- NOT what the docs advertise. `kimi-k2.5` is unavailable (404 "Not
# found the model kimi-k2.5 or Permission denied") even though it is current on the public pricing
# pages, so always trust the models endpoint over documentation here. k2.6 is the cheapest
# general-purpose option; the k2.7 variants are CODE-specialised (wrong tool for news sentiment,
# listed for completeness). Prices are list rates at that date -- re-check
# https://platform.kimi.ai/docs/pricing before trusting a cost estimate; they move fast.
# Prompt caching DOES engage on this workload (measured cached_tokens=341 of 341 input on repeat
# calls), so real input cost runs below the estimator's conservative no-cache assumption.
MODELS = {
    "k2.6":       ("kimi-k2.6",                    0.60,  3.41),
    "k2.7-code":  ("kimi-k2.7-code",               0.95,  4.00),
    "k2.7-fast":  ("kimi-k2.7-code-highspeed",     0.95,  4.00),
    "k3":         ("kimi-k3",                      3.00, 15.00),
}
# Per-model probe results, 2026-07-30 (3 articles each, measured not assumed):
#   k2.6       thinking-disabled OK · ~3.2s · ~419 tok  -> the only practical option; full corpus
#              ran in ~90 min for ~$0.80. This is what the cached k2.6 scores came from.
#   k3         thinking-disabled OK (reason=0, parses) BUT 18-40s per call and roughly 1 in 3
#              requests returns `429 engine_overloaded_error` -- Moonshot capacity, not our RPM
#              (we were pacing at 3.3s). Full corpus extrapolates to 14-20h wall-clock for ~$3.30.
#              Released 2026-07-27, so this is almost certainly launch congestion; retry in a few
#              weeks rather than burning a day on retries now.
#   k2.7-code  REJECTS thinking-disabled outright: `400 invalid thinking: only type=enabled is
#              allowed for this model`. Forced reasoning means ~1,650 tok/article -> ~$9.36 and
#              10h+ for the full corpus, on a model trained for CODE rather than news sentiment.
#              Not worth it; k2.7-code-highspeed presumably shares the constraint (untested).
DEFAULT_MODELS = ["k2.6"]        # cheapest general-purpose model the account can reach

# ── Thinking: OFF by default, and it matters enormously ───────────────────────────────────────
# kimi-k2.6 is a REASONING model. Left alone it spends ~1,240 reasoning tokens per article before
# answering (measured), which with max_tokens=200 meant it never emitted the JSON at all --
# `finish_reason: length`, empty content, 0/8 parsed. Measured per article on this corpus:
#     thinking ON : ~1,650 tokens, 15-69s   -> full corpus ~$7.80, EXCEEDS Tier 0's 1.5M TPD, ~12h
#     thinking OFF:   ~389 tokens,   2.8s   -> full corpus ~$0.62, ~47% of TPD, ~84 min
# `extra_body={"thinking": {"type": "disabled"}}` is what works; `enable_thinking: False` and
# `reasoning_effort: "low"` are both ACCEPTED but ignored (still burned 895 / 1,404 reasoning
# tokens respectively) -- a silent no-op, so don't trust them.
#
# Fairness note: thinking-off matches how haiku/sonnet-4-6/opus-4-8 were scored on this corpus
# (anthropic_scorer.py passes no thinking param, which means no thinking on those models). It does
# NOT match sonnet5, which runs adaptive thinking when the param is omitted -- so the existing
# corpus already mixes both, and that is a pre-existing property of the dataset, not something
# introduced here. Scores are cached under a separate `-think` alias when thinking is on, so the
# two never contaminate one comparison.
THINKING_MAX_TOKENS = 3000       # needs headroom for reasoning + the answer
PLAIN_MAX_TOKENS    = 300        # measured answer is ~48 tokens; this is generous

# Measured on this corpus 2026-07-29 (not guessed): system prompt 1,100 chars ~275 tok; articles
# (headline+summary) mean 182 chars ~45 tok. So ~340 in + ~80 out = ~420 tokens/article, and the
# full 1,683-article corpus is ~707k tokens -- under half of Tier 0's daily budget. TPD is NOT the
# constraint here; RPM is.
EST_INPUT_TOKENS  = 340
EST_OUTPUT_TOKENS = 80

# ── Rate limits ──────────────────────────────────────────────────────────────────────────────
# Moonshot Tier 0 (until $10 accumulated recharge): concurrency 3, RPM 20, TPM 500k, TPD 1.5M.
# With thinking OFF (the default) per-article latency is ~2.8s, under the 3.3s pacing interval, so
# RPM 20 is the binding limit: ~84 minutes for the full corpus, and concurrency would NOT help --
# 3 parallel requests just reach the same 20/min ceiling sooner and earn 429s. Serial + paced is
# both sufficient and simpler, and it keeps cached latency numbers meaningful rather than
# queue-distorted. (With --thinking, latency is ~25s and serial throughput drops to ~2.4 RPM, far
# under the cap -- that mode IS latency-bound and would benefit from the tier's 3 concurrent
# slots. Not implemented: the mode costs ~12x and exceeds Tier 0's daily tokens anyway.)
# Override for Tier 1+ via env: KIMI_RPM / KIMI_TPD.
RPM_LIMIT = int(os.environ.get("KIMI_RPM", "20"))
TPD_LIMIT = int(os.environ.get("KIMI_TPD", "1500000"))
RPM_SAFETY = 0.90                # aim for 90% of the stated RPM; bursts near the ceiling 429
MIN_INTERVAL_S = 60.0 / (RPM_LIMIT * RPM_SAFETY)

MAX_RETRIES = 5
RETRY_BACKOFF_S = 2.0


def load_cache() -> dict:
    return json.loads(CACHE_FILE.read_text()) if CACHE_FILE.exists() else {}


def save_cache(cache: dict) -> None:
    CACHE_FILE.write_text(json.dumps(cache, indent=2))


def _client() -> OpenAI:
    key = os.environ.get("MOONSHOT_API_KEY") or os.environ.get("KIMI_API_KEY")
    if not key:
        sys.exit("❌ MOONSHOT_API_KEY not set. Sign up at https://platform.moonshot.ai, add "
                 "prepaid credits, create a key, then put MOONSHOT_API_KEY=... in .env")
    return OpenAI(base_url=BASE_URL, api_key=key, timeout=60.0)


class _Pacer:
    """Enforces a minimum gap between request STARTS so we stay under RPM. Sleeps only for the
    remaining gap, so a slow API response (which already consumed wall-clock) costs no extra
    wait -- the limit is requests per minute, not idle time per minute."""

    def __init__(self, min_interval_s: float):
        self.min_interval_s = min_interval_s
        self._last = 0.0

    def wait(self) -> None:
        gap = time.monotonic() - self._last
        if gap < self.min_interval_s:
            time.sleep(self.min_interval_s - gap)
        self._last = time.monotonic()


def score_article(client: OpenAI, model_id: str, article: dict,
                  pacer: "_Pacer | None" = None,
                  thinking: bool = False) -> "tuple[dict | None, float, int]":
    """One article -> (parsed dict | None, latency_seconds, tokens_used). Never caches a failure:
    a persistent failure returns None so the article stays uncached and a later run retries it,
    rather than silently shrinking the comparison set."""
    t0 = time.time()
    for attempt in range(MAX_RETRIES):
        if pacer:
            pacer.wait()
        try:
            # ⚠️ NO temperature parameter. kimi-k2.6 rejects anything but 1 with
            # `400 invalid temperature: only 1 is allowed for this model` (live test 2026-07-29).
            # This is FINE for the comparison that matters: anthropic_scorer.py passes no
            # temperature either, so haiku/sonnet-4-6/sonnet5/opus were all scored at the Anthropic
            # default of 1.0 -- Kimi is temperature-MATCHED to them. Only Ollama (v2/scoring.py)
            # and Groq (groq_value.py) used 0.1, making those two the low-variance outliers rather
            # than Kimi the high-variance one.
            # Why temperature matters here at all: run-to-run variance is noise that attenuates any
            # real score-vs-return correlation AND inflates the "distinct values emitted" statistic
            # this comparison leans on -- more distinct confidences can mean finer discrimination OR
            # just sampling jitter. Matched temperature is what makes that column readable.
            resp = client.chat.completions.create(
                model=model_id,
                max_tokens=THINKING_MAX_TOKENS if thinking else PLAIN_MAX_TOKENS,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content":
                     f"Headline: {article['headline']}\n\nBody: {article['summary']}\n\n"
                     f"Respond with JSON only."},
                ],
                **({} if thinking else
                   {"extra_body": {"thinking": {"type": "disabled"}}}),
            )
            parsed = _parse_raw(resp.choices[0].message.content or "")
            used = getattr(getattr(resp, "usage", None), "total_tokens", 0) or 0
            return (parsed if isinstance(parsed, dict) else None), time.time() - t0, used
        except Exception as e:
            msg = str(e)
            is_429 = "429" in msg or "rate limit" in msg.lower()
            # A 4xx that isn't 429 is a PERMANENT client error (bad param, bad model id, auth) --
            # retrying it just burns wall-clock and, at 3.3s pacing, RPM budget. Caught by the
            # live test: a rejected `temperature` was retried 5x per article for 35s each.
            status = getattr(e, "status_code", None)
            if not is_429 and isinstance(status, int) and 400 <= status < 500:
                print(f"  ⚠️  permanent {status}, not retrying: {msg[:180]}")
                return None, time.time() - t0, 0
            if attempt == MAX_RETRIES - 1:
                print(f"  ⚠️  {'rate-limited' if is_429 else 'API error'} after {MAX_RETRIES} "
                      f"attempts: {msg[:160]}")
                return None, time.time() - t0, 0
            # A 429 means the pacing estimate is wrong for current conditions, so back off much
            # harder than for a transient network blip -- and give the RPM window time to roll.
            delay = (max(60.0 / RPM_LIMIT * 3, RETRY_BACKOFF_S * (2 ** attempt)) if is_429
                     else RETRY_BACKOFF_S * (2 ** attempt))
            if is_429:
                print(f"  ⏳ rate-limited, backing off {delay:.0f}s "
                      f"(attempt {attempt + 1}/{MAX_RETRIES})")
            time.sleep(delay)
    return None, time.time() - t0, 0


def _alias(alias: str, thinking: bool) -> str:
    """Cache alias. Thinking and non-thinking scores are DIFFERENT signals from the same model --
    keeping them under one key would silently blend them into one 'Kimi' column."""
    return f"{alias}-think" if thinking else alias


def run_scoring(aliases: list[str], limit: "int | None", thinking: bool = False) -> None:
    universe = load_universe()
    cache = load_cache()
    client = _client()
    print(f"Universe: {len(universe)} articles (same filter as anthropic_scorer.load_universe)")
    print(f"Thinking: {'ON — expect ~1,650 tok and 15-69s per article' if thinking else 'OFF'}")

    pacer = _Pacer(MIN_INTERVAL_S)
    print(f"Pacing: {RPM_LIMIT} RPM limit → 1 request every {MIN_INTERVAL_S:.1f}s "
          f"(serial; concurrency won't help under an RPM cap)")
    tokens_today = 0

    for alias in aliases:
        model_id, pin, pout = MODELS[alias]
        ca = _alias(alias, thinking)
        uncached = [a for a in universe if cache_key(ca, a["hash"]) not in cache]
        todo = uncached[:limit] if limit else uncached
        # With thinking ON, per-article latency (~25s) far exceeds the 3.3s pacing interval, so
        # wall-clock is latency-bound, not RPM-bound -- the ETA has to reflect whichever dominates.
        per_article_s = max(MIN_INTERVAL_S, 25.0 if thinking else 3.0)
        eta_min = len(todo) * per_article_s / 60
        print(f"\n── {ca} ({model_id}): scoring {len(todo)} of {len(uncached)} uncached "
              f"({len(universe) - len(uncached)} already cached) · ETA ~{eta_min:.0f} min")
        lat, ok = [], 0
        for i, art in enumerate(todo, 1):
            if tokens_today >= TPD_LIMIT:
                print(f"   🛑 daily token budget reached ({tokens_today:,}/{TPD_LIMIT:,}) — "
                      f"stopping. Cache is saved; rerun tomorrow to continue where this left off.")
                break
            result, latency, used = score_article(client, model_id, art, pacer, thinking)
            tokens_today += used
            lat.append(latency)
            if result is not None:
                ok += 1
                cache[cache_key(ca, art["hash"])] = {
                    "sent":    result.get("sentiment", ""),
                    "mag":     result.get("magnitude", 0),
                    "conf":    result.get("confidence", 0),
                    "tick":    result.get("tickers", []),
                    "why":     result.get("reasoning", ""),
                    "latency": round(latency, 3),
                }
            if i % 25 == 0 or i == len(todo):
                save_cache(cache)                 # crash/Ctrl-C safe: progress every 25
                per_art = tokens_today / max(i, 1)
                print(f"   {i}/{len(todo)}  ok={ok}  median latency={statistics.median(lat):.2f}s"
                      f"  tokens={tokens_today:,} ({per_art:.0f}/article)")
        save_cache(cache)
        if lat:
            # Bill off MEASURED tokens where we have them; fall back to the estimate otherwise.
            spend = ((tokens_today / 1e6 * (pin + pout) / 2) if tokens_today else
                     len(todo) * (EST_INPUT_TOKENS / 1e6 * pin + EST_OUTPUT_TOKENS / 1e6 * pout))
            print(f"   done: {ok}/{len(todo)} parsed · median {statistics.median(lat):.2f}s "
                  f"· p90 {sorted(lat)[int(len(lat) * 0.9) - 1]:.2f}s "
                  f"· {tokens_today:,} tokens · ~${spend:.2f}")


def dry_run(aliases: list[str], limit: "int | None", thinking: bool = False) -> None:
    universe = load_universe()
    cache = load_cache()
    print(f"Universe: {len(universe)} articles\n")
    print(f"{'model':<12} {'to score':>9} {'est input$':>11} {'est output$':>12} {'est total$':>11}")
    for alias in aliases:
        model_id, pin, pout = MODELS[alias]
        n = len([a for a in universe
                 if cache_key(_alias(alias, thinking), a["hash"]) not in cache])
        if limit:
            n = min(n, limit)
        out_tok = 1300 if thinking else EST_OUTPUT_TOKENS   # measured, see MODELS notes
        ci = n * EST_INPUT_TOKENS / 1e6 * pin
        co = n * out_tok / 1e6 * pout
        print(f"{alias:<12} {n:>9} {ci:>11.2f} {co:>12.2f} {ci + co:>11.2f}")
    print("\n(no cache discount assumed; verify live rates at platform.kimi.ai/docs/pricing)")


def report(aliases: list[str]) -> None:
    """Distribution stats per model — the fields the strategy actually gates on. The point of the
    comparison is DISCRIMINATION, not averages: llama3.2 scored 0.82 confidence on 97% of its
    bullish signals, which made the confidence gate a no-op regardless of where it was set."""
    cache = load_cache()
    if not cache:
        return print("no Kimi scores cached yet")
    for alias in aliases:
        rows = [v for k, v in cache.items()
                if k.split(":", 1)[0] in (alias, f"{alias}-think") and v]
        if not rows:
            print(f"\n{alias}: nothing cached")
            continue
        bull = [r for r in rows if r.get("sent") == "bullish"]
        print(f"\n── {alias}: {len(rows)} scored, {len(bull)} bullish "
              f"({100 * len(bull) / len(rows):.0f}%)")
        for field, label in (("conf", "confidence"), ("mag", "magnitude")):
            vals = sorted(float(r.get(field) or 0) for r in bull)
            if not vals:
                continue
            uniq = len(set(vals))
            print(f"   {label:<11} min={vals[0]:.2f} p25={vals[len(vals)//4]:.2f} "
                  f"med={statistics.median(vals):.2f} p75={vals[3*len(vals)//4]:.2f} "
                  f"max={vals[-1]:.2f} · {uniq} distinct values")
        lats = [r["latency"] for r in rows if r.get("latency")]
        if lats:
            print(f"   latency     median={statistics.median(lats):.2f}s "
                  f"p90={sorted(lats)[int(len(lats)*0.9)-1]:.2f}s")


def compare() -> None:
    """Side-by-side discrimination on the SAME article hashes: Kimi vs Anthropic vs Ollama.
    Only hashes scored by every model present are counted, so no model gets an easier subset."""
    kimi = load_cache()
    anth = json.loads((DATA_DIR / "anthropic_backtest_cache.json").read_text())
    uni  = json.loads((DATA_DIR / "unified_scores.json").read_text())

    cols: dict[str, dict[str, dict]] = {}
    for k, v in kimi.items():
        alias, _, h = k.partition(":")
        if v:
            cols.setdefault(f"kimi:{alias}", {})[h] = {"sent": v["sent"], "conf": v["conf"], "mag": v["mag"]}
    for k, v in anth.items():
        alias, _, h = k.partition(":")
        if v:
            cols.setdefault(alias, {})[h] = {"sent": v["sent"], "conf": v["conf"], "mag": v["mag"]}
    # unified_scores keys are "<prompt>:<hash>" with the live field names
    for k, v in uni.items():
        prompt, _, h = k.partition(":")
        if v and prompt == "unified_v1":
            cols.setdefault("ollama:unified_v1", {})[h] = {
                "sent": v.get("sentiment", ""), "conf": v.get("confidence", 0),
                "mag": v.get("magnitude", 0)}

    cols = {name: d for name, d in cols.items() if len(d) >= 50}
    if len(cols) < 2:
        return print("need >=2 models with >=50 scores cached to compare")
    common = set.intersection(*(set(d) for d in cols.values()))
    print(f"\n{'=' * 96}\nCROSS-MODEL DISCRIMINATION — {len(common)} article hashes scored by all "
          f"{len(cols)} models\n{'=' * 96}")
    # Coverage matters: haiku/sonnet/opus were only ever SAMPLED (~115 articles each) while
    # sonnet5 and ollama cover the full corpus, so the strict all-model intersection is small.
    # Printing total coverage next to the shared-subset stats stops that from looking like a
    # like-for-like n when it isn't.
    print("coverage (total scored, full corpus = %d): %s" % (
        len(load_universe()),
        ", ".join(f"{n}={len(d)}" for n, d in sorted(cols.items()))))
    print(f"\n{'model':<22} {'bull%':>6} {'conf med':>9} {'conf uniq':>10} {'mag med':>8} "
          f"{'mag uniq':>9} {'pass 0.35/0.70':>15}")
    for name, d in sorted(cols.items()):
        rows = [d[h] for h in common]
        bull = [r for r in rows if r["sent"] == "bullish"]
        if not bull:
            continue
        cf = sorted(float(r["conf"] or 0) for r in bull)
        mg = sorted(float(r["mag"] or 0) for r in bull)
        passed = sum(1 for r in bull
                     if float(r["mag"] or 0) >= 0.35 and float(r["conf"] or 0) >= 0.70)
        print(f"{name:<22} {100*len(bull)/len(rows):>5.0f}% {statistics.median(cf):>9.2f} "
              f"{len(set(cf)):>10} {statistics.median(mg):>8.2f} {len(set(mg)):>9} "
              f"{passed:>7} ({100*passed/len(bull):>3.0f}%)")
    print("\n'uniq' is the count of DISTINCT values emitted — the single best tell for whether a\n"
          "model's field is a usable dial or a constant. llama3.2's confidence collapses here.\n"
          "'pass' applies v2's live gate (mag>=0.35 AND conf>=0.70) to each model's own scale.")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+", default=DEFAULT_MODELS, choices=list(MODELS))
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--dry-run", action="store_true")
    g.add_argument("--sample", type=int, metavar="N")
    g.add_argument("--all", action="store_true")
    g.add_argument("--report", action="store_true")
    g.add_argument("--compare", action="store_true",
                   help="side-by-side vs Anthropic + Ollama on shared hashes")
    ap.add_argument("--thinking", action="store_true",
                    help="let the model reason first (~4x tokens, ~10x latency, "
                         "cached separately under a -think alias)")
    args = ap.parse_args()

    if args.dry_run:
        dry_run(args.models, args.sample, args.thinking)
    elif args.compare:
        compare()
    elif args.report:
        report(args.models)
    else:
        run_scoring(args.models, None if args.all else args.sample, args.thinking)
        report(args.models)


if __name__ == "__main__":
    main()
