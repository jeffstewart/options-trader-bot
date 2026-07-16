"""
anthropic_scorer.py — score the backtest article universe using Anthropic models.
Results are stored in data/anthropic_backtest_cache.json and are directly
comparable to groq_primary_backtest_cache.json (same article hashes, same schema).

Usage:
    # Dry run — estimate cost only, score nothing
    python anthropic_scorer.py --dry-run

    # Score a small sample first (go/no-go decision)
    python anthropic_scorer.py --sample 50

    # Score all remaining articles (skips already-cached)
    python anthropic_scorer.py --all

    # Run report on cached scores (no new scoring)
    python anthropic_scorer.py --report

    # Use Batch API (50% cheaper, ~24hr turnaround — backtest only)
    python anthropic_scorer.py --all --batch

Models tested (set via --models flag, default: all four):
    haiku     claude-haiku-4-5
    sonnet    claude-sonnet-4-6
    sonnet5   claude-sonnet-5   (intro pricing through Aug 31 2026)
    opus      claude-opus-4-8
    fable     claude-fable-5    (frontier tier — expensive, optional)

Example — sample with haiku and sonnet only:
    python anthropic_scorer.py --sample 50 --models haiku sonnet
"""

import argparse
import json
import os
import random
import re
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import anthropic
from dotenv import load_dotenv

# Load .env from project root (one level up from data/ where script runs)
for _env in [Path(".env"), Path("../.env"), Path(__file__).parent.parent / ".env"]:
    if _env.exists():
        load_dotenv(_env)
        break

# ── Paths ─────────────────────────────────────────────────────────────────────
BASE_DIR   = Path("/Users/jeff/Claude/Trader")
DATA_DIR   = BASE_DIR / "data"
CACHE_FILE = DATA_DIR / "anthropic_backtest_cache.json"
DUAL_CACHE = DATA_DIR / "dual_score_cache.json"
GROQ_CACHE = DATA_DIR / "groq_primary_backtest_cache.json"

# ── Model registry ─────────────────────────────────────────────────────────────
MODELS = {
    "haiku":   ("claude-haiku-4-5-20251001", 1.00,  5.00),   # date suffix required
    "sonnet":  ("claude-sonnet-4-6",         3.00, 15.00),
    "sonnet5": ("claude-sonnet-5",           2.00, 10.00),   # intro pricing through Aug 31 2026
    "opus":    ("claude-opus-4-8",           5.00, 25.00),
    "fable":   ("claude-fable-5",           10.00, 50.00),
}

DEFAULT_MODELS = ["haiku", "sonnet", "sonnet5", "opus"]   # fable excluded by default

# ── Token estimates (per article) ──────────────────────────────────────────────
EST_INPUT_TOKENS  = 600   # system prompt ~400 + headline/summary ~200
EST_OUTPUT_TOKENS = 80    # JSON response

# ── Scoring prompt (mirrors groq_vs_ollama_backtest.py exactly) ────────────────
SYSTEM_PROMPT = """\
You are a quantitative equity trading signal generator.
Respond with a JSON object ONLY — no markdown, no code fences, no explanation.

Schema:
{
  "tickers":    ["AAPL"],
  "sentiment":  "bullish",
  "confidence": 0.82,
  "magnitude":  0.75,
  "reasoning":  "one sentence"
}

sentiment: "bullish" | "bearish" | "neutral"
confidence: float 0.0–1.0 — how certain you are about the sentiment direction
magnitude:  float 0.0–1.0 — how likely this news is to meaningfully move the stock price

Magnitude scale:
  0.0–0.2  Noise or irrelevant
  0.2–0.4  Routine news (in-line earnings, minor upgrades)
  0.4–0.6  Meaningful catalyst (earnings beat, notable partnership)
  0.6–0.8  Strong catalyst (major acquisition, landmark FDA approval)
  0.8–1.0  Transformative event (company-defining deal, paradigm shift)

Rules:
- Only include tickers you are highly confident about.
- General macro news with no specific company → empty tickers list.
- Be conservative: confidence > 0.7 only for materially significant news.
- Only flag bullish sentiment — we trade long calls only.
- Return ONLY the JSON object."""


# ── Pre-score noise filter (mirrors core/config.py PRESCORE_SKIP_PATTERNS) ──────
PRESCORE_SKIP_PATTERNS = (
    "stock futures", "dow futures", "nasdaq futures", "s&p futures", "futures rise", "futures fall",
    "futures point", "futures slip", "premarket", "pre-market", "market wrap", "this week in markets",
    "stock market today", "closing bell", "opening bell", "stocks close higher", "stocks close lower",
    "treasury yield", "10-year yield",
    "stocks to watch", "stocks to buy", "stocks to avoid", "stocks to consider",
    "best stocks to", "top stocks to",
    "what's going on with", "whats going on with",
    "russell 3000", "russell 2000", "russell 1000", "russell microcap",
    "s&p 600", "s&p 400", "s&p smallcap", "s&p midcap",
    "dow jones industrial average", "nasdaq biotechnology index",
    "to join the s&p", "joins the s&p", "joins s&p", "added to the s&p",
    "to join the russell", "joins the russell", "added to the russell", "added to russell",
    "to join the nasdaq", "join the nasdaq-100", "join the dow jones", "joins the dow jones",
)
PRESCORE_SKIP_REGEXES = (
    re.compile(r"\bif you (?:had )?invested\b", re.I),
    re.compile(r"\$[\d,]+\s+invested\s+in\b", re.I),
    re.compile(r"\d+\s+years?\s+ago\s+would\s+be\s+worth", re.I),
    re.compile(r"\boutperformed\b[^.]{0,40}\bover the (?:past|last)\s+\d+\s+(?:year|month)", re.I),
)

def is_noise(headline: str) -> bool:
    """Return True if this headline should be skipped before scoring."""
    h = (headline or "").lower()
    if any(p in h for p in PRESCORE_SKIP_PATTERNS):
        return True
    if any(r.search(headline or "") for r in PRESCORE_SKIP_REGEXES):
        return True
    return False


def load_cache() -> dict:
    if CACHE_FILE.exists():
        return json.loads(CACHE_FILE.read_text())
    return {}


def save_cache(cache: dict):
    CACHE_FILE.write_text(json.dumps(cache, indent=2))


def load_universe() -> list[dict]:
    """
    Load the same filtered universe the live bot acts on:
      - dual_score_cache.json: article must have at least one valid ticker
        from the bullish scorer (filters junk like "If you invested $X...")
      - yahoo_candpool_cache.json: underlying stock price must be >= $5
    This mirrors the candidates() filter in gemini_lotto_pnl.py.
    """
    cp   = json.loads((DATA_DIR / "yahoo_candpool_cache.json").read_text())
    dual = json.loads(DUAL_CACHE.read_text())

    # Build set of content-hashes that have a valid price (px >= 5)
    valid_cks = {
        k.split("_", 1)[1]
        for k, v in cp.items()
        if isinstance(v, dict) and v.get("px", 0) >= 5
    }

    articles = []
    seen_cks = set()
    for ck, v in dual.items():          # dual cache key IS the article hash
        if not isinstance(v, dict):
            continue
        article = v.get("_article", {}) or {}
        bullish = v.get("bullish", {}) or {}

        # Must have a headline
        h = article.get("headline", "").strip()
        if not h:
            continue

        # Must have at least one non-crypto ticker identified by bullish scorer
        tickers = [t for t in (bullish.get("tickers") or [])
                   if t not in ("BTC", "ETH")]
        if not tickers:
            continue

        # Must have valid price data and no duplicates
        if ck not in valid_cks or ck in seen_cks:
            continue
        seen_cks.add(ck)

        # Apply same pre-score noise filter as the live bot
        body = article.get("summary", "") or ""
        if is_noise(h):
            continue

        articles.append({
            "hash":     ck,
            "headline": h,
            "summary":  body[:1200],
            "ticker":   tickers[0],
        })

    articles.sort(key=lambda x: x["hash"])
    return articles


def cache_key(model_alias: str, article_hash: str) -> str:
    return f"{model_alias}:{article_hash}"


def _parse_raw(raw: str) -> dict | None:
    """Parse JSON from model response, stripping markdown fences."""
    raw = raw.strip()
    if raw.startswith("```"):
        raw = raw.split("```")[1]
        if raw.startswith("json"):
            raw = raw[4:]
    try:
        return json.loads(raw.strip())
    except json.JSONDecodeError:
        return None


def _make_messages(article: dict) -> list:
    """Build messages array with prompt caching on the system prompt."""
    return [{"role": "user", "content": f"Headline: {article['headline']}\n\nBody: {article['summary']}\n\nRespond with JSON only."}]


def _cached_system() -> list:
    """System prompt as a content block with cache_control for prompt caching.
    Anthropic caches blocks marked ephemeral for ~5 minutes — the system prompt
    is identical across all articles so it will be cache-hit on every call after
    the first, saving ~67% of input token costs."""
    return [{"type": "text", "text": SYSTEM_PROMPT, "cache_control": {"type": "ephemeral"}}]


def score_article(client: anthropic.Anthropic, model_id: str, article: dict) -> tuple[dict | None, float]:
    """Score one article with prompt caching enabled. Returns (result_dict, latency_seconds)."""
    t0 = time.time()
    try:
        resp = client.messages.create(
            model=model_id,
            max_tokens=200,
            system=_cached_system(),
            messages=_make_messages(article),
            betas=["prompt-caching-2024-07-31"],
        )
        latency = time.time() - t0
        raw_parsed = _parse_raw(resp.content[0].text)
        result  = raw_parsed if isinstance(raw_parsed, dict) else None
        if result is None:
            print(f"  ⚠️  JSON parse failed")
        return result, latency
    except Exception as e:
        print(f"  ⚠️  API error: {e}")
        return None, time.time() - t0


# ── Batch API implementation ───────────────────────────────────────────────────

def submit_batch(client: anthropic.Anthropic, model_id: str, alias: str,
                 articles: list[dict], cache: dict) -> str:
    """Submit a Message Batch for all unscored articles for one model.
    Returns the batch ID. Results are polled separately."""
    requests = []
    for article in articles:
        k = f"{alias}:{article['hash']}"
        if k in cache and cache[k] is not None:
            continue
        requests.append({
            "custom_id": f"{alias}__{article['hash']}",
            "params": {
                "model":      model_id,
                "max_tokens": 200,
                "system":     _cached_system(),
                "messages":   _make_messages(article),
            }
        })
    if not requests:
        print(f"  {alias}: nothing to submit")
        return ""

    batch = client.beta.messages.batches.create(requests=requests,
                                                 betas=["message-batches-2024-09-24",
                                                        "prompt-caching-2024-07-31"])
    print(f"  {alias}: submitted {len(requests)} requests → batch {batch.id}")
    return batch.id


def poll_batch(client: anthropic.Anthropic, batch_id: str, alias: str,
               cache: dict, poll_interval: int = 60) -> int:
    """Poll a batch until complete, writing results into cache. Returns success count."""
    print(f"  Polling batch {batch_id} for {alias}…")
    while True:
        batch = client.beta.messages.batches.retrieve(batch_id,
                                                       betas=["message-batches-2024-09-24"])
        counts = batch.request_counts
        print(f"    {alias}: processing={counts.processing}  succeeded={counts.succeeded}"
              f"  errored={counts.errored}  expired={counts.expired}")
        if batch.processing_status == "ended":
            break
        time.sleep(poll_interval)

    # Collect results
    done = 0
    for result in client.beta.messages.batches.results(batch_id,
                                                        betas=["message-batches-2024-09-24"]):
        k = result.custom_id.replace("__", ":", 1)
        if result.result.type == "succeeded":
            raw  = result.result.message.content[0].text
            parsed = _parse_raw(raw)
            if isinstance(parsed, list): parsed = parsed[0] if parsed else None
            if parsed:
                cache[k] = {
                    "sent":    parsed.get("sentiment", ""),
                    "mag":     parsed.get("magnitude", 0),
                    "conf":    parsed.get("confidence", 0),
                    "tick":    parsed.get("tickers", []),
                    "why":     parsed.get("reasoning", ""),
                    "latency": None,   # batch doesn't provide per-request latency
                }
                done += 1
            else:
                cache[k] = None
        else:
            cache[k] = None
    return done


def estimate_cost(n_articles: int, model_aliases: list[str]) -> float:
    total = 0.0
    inp_M = (n_articles * EST_INPUT_TOKENS)  / 1_000_000
    out_M = (n_articles * EST_OUTPUT_TOKENS) / 1_000_000
    for alias in model_aliases:
        _, inp_rate, out_rate = MODELS[alias]
        total += inp_M * inp_rate + out_M * out_rate
    return total


def print_cost_table(n_articles: int, model_aliases: list[str], use_batch: bool = False):
    discount = 0.5 if use_batch else 1.0
    mode     = "Batch API (50% off)" if use_batch else "Standard"
    print(f"\n  Cost estimate — {n_articles} articles — {mode}")
    print(f"  {'Model':<14} {'$/article':>10} {'Total':>10}")
    print(f"  {'-'*36}")
    grand = 0.0
    inp_M = (n_articles * EST_INPUT_TOKENS)  / 1_000_000
    out_M = (n_articles * EST_OUTPUT_TOKENS) / 1_000_000
    for alias in model_aliases:
        model_id, inp_rate, out_rate = MODELS[alias]
        cost  = (inp_M * inp_rate + out_M * out_rate) * discount
        per_a = cost / n_articles if n_articles else 0
        grand += cost
        print(f"  {alias:<14} ${per_a:>9.5f} ${cost:>9.4f}")
    print(f"  {'TOTAL':<14} {'':>10} ${grand:>9.4f}")
    return grand


def run_scoring(args):
    client   = anthropic.Anthropic()
    universe = load_universe()
    cache    = load_cache()
    aliases  = args.models

    print(f"\n📰 Universe: {len(universe)} articles")
    print(f"🤖 Models:   {', '.join(aliases)}")
    print(f"💾 Cache:    {len(cache)} existing scores")

    # Determine which articles still need scoring per model
    todo = []
    for article in universe:
        for alias in aliases:
            k = f"{alias}:{article['hash']}"
            # Retry None entries (previous auth failures, parse errors etc.)
            if k not in cache or cache[k] is None:
                todo.append((alias, article))

    if args.sample:
        all_hashes = list({a["hash"] for _, a in todo})
        random.shuffle(all_hashes)
        sample_hashes = set(all_hashes[:args.sample])
        hash_to_article = {a["hash"]: a for _, a in todo}
        todo = [
            (alias, hash_to_article[h])
            for h in sample_hashes
            for alias in aliases
            if f"{alias}:{h}" not in cache or cache[f"{alias}:{h}"] is None
        ]
        print(f"🎲 Sample mode: {args.sample} articles × {len(aliases)} models = {len(todo)} calls")
    else:
        print(f"📋 To score: {len(todo)} article-model pairs")

    if args.batch and not args.sample:
        # ── Batch API path (50% cheaper, async, ~minutes to hours turnaround) ──
        print(f"\n📦 Batch API mode — submitting {len(aliases)} batches…")
        batch_ids = {}
        for alias in aliases:
            model_id   = MODELS[alias][0]
            alias_articles = [a for _, a in todo if _ == alias] if args.sample else universe
            bid = submit_batch(client, model_id, alias, alias_articles, cache)
            if bid:
                batch_ids[alias] = bid

        if not batch_ids:
            print("  Nothing to submit.")
            return

        # Save batch IDs so we can resume polling if interrupted
        bid_file = DATA_DIR / "anthropic_batch_ids.json"
        existing = json.loads(bid_file.read_text()) if bid_file.exists() else {}
        existing.update(batch_ids)
        bid_file.write_text(json.dumps(existing, indent=2))
        print(f"\n💾 Batch IDs saved to {bid_file}")
        print(f"   Poll results with: --poll")
        print(f"   Or wait here (polls every 60s)…\n")

        total_done = 0
        for alias, bid in batch_ids.items():
            done = poll_batch(client, bid, alias, cache)
            total_done += done
            save_cache(cache)
            print(f"  ✅ {alias}: {done} results saved")

        print(f"\n✅ Batch complete — {total_done} total scored")
        print(f"💾 Cache saved to {CACHE_FILE}")
        return

    if not todo:
        print("✅ Nothing to score — all articles already cached.")
        return

    # Cost estimate + confirmation
    n_unique = len(set(a["hash"] for _, a in todo))
    print_cost_table(n_unique, aliases, use_batch=False)

    if args.dry_run:
        print("\n  --dry-run: no API calls made.")
        return

    if not args.yes:
        ans = input("\n  Proceed? [y/N] ").strip().lower()
        if ans != "y":
            print("  Aborted.")
            return

    # Score
    print(f"\n  Scoring {len(todo)} pairs…\n")
    done = 0
    errors = 0
    t0 = time.time()
    # Per-model latency tracking
    model_latencies: dict[str, list[float]] = {alias: [] for alias in aliases}

    for alias, article in todo:
        model_id = MODELS[alias][0]
        k = f"{alias}:{article['hash']}"
        result, latency = score_article(client, model_id, article)
        if result:
            cache[k] = {
                "sent":    result.get("sentiment", ""),
                "mag":     result.get("magnitude", 0),
                "conf":    result.get("confidence", 0),
                "tick":    result.get("tickers", []),
                "why":     result.get("reasoning", ""),
                "latency": round(latency, 3),
            }
            model_latencies[alias].append(latency)
            done += 1
        else:
            cache[k] = None
            errors += 1

        elapsed = time.time() - t0
        rate    = done / elapsed if elapsed > 0 else 0
        remain  = (len(todo) - done - errors)
        eta     = remain / rate if rate > 0 else 0
        mag_str = f"{cache[k]['mag']:4}" if cache[k] else "-   "
        lat_str = f"{latency:.1f}s" if result else "ERR "
        print(f"  [{done+errors:>4}/{len(todo)}] {alias:<9}  "
              f"sent={cache[k]['sent'] if cache[k] else 'ERR':8}  "
              f"mag={mag_str}  lat={lat_str}  "
              f"ETA {eta/60:.1f}min  {article['headline'][:55]}")

        if (done + errors) % 10 == 0:
            save_cache(cache)

        time.sleep(0.2)

    save_cache(cache)
    elapsed = time.time() - t0
    print(f"\n✅ Done — {done} scored, {errors} errors in {elapsed/60:.1f}min")

    # Latency summary per model
    print(f"\n  Latency summary:")
    print(f"  {'Model':<12} {'n':>4} {'avg':>7} {'p50':>7} {'p95':>7} {'max':>7}")
    print(f"  {'-'*46}")
    for alias in aliases:
        lats = sorted(model_latencies[alias])
        if not lats: continue
        avg = sum(lats) / len(lats)
        p50 = lats[len(lats)//2]
        p95 = lats[int(len(lats)*0.95)]
        mx  = lats[-1]
        print(f"  {alias:<12} {len(lats):>4} {avg:>6.2f}s {p50:>6.2f}s {p95:>6.2f}s {mx:>6.2f}s")

    print(f"\n💾 Cache saved to {CACHE_FILE}")


def run_report(args):
    """Print signal quality summary for all cached Anthropic scores."""
    cache  = load_cache()
    if not cache:
        print("No cached scores found. Run --sample or --all first.")
        return

    # Count by model
    from collections import Counter
    import statistics

    by_model: dict[str, list] = {}
    for k, v in cache.items():
        if not v or not isinstance(v, dict):
            continue
        alias = k.split(":")[0]
        by_model.setdefault(alias, []).append(v)

    print(f"\n{'='*60}")
    print(f"ANTHROPIC SCORER REPORT  —  {len(cache)} cached entries")
    print(f"{'='*60}")

    for alias in sorted(by_model.keys()):
        entries  = by_model[alias]
        bullish  = [e for e in entries if e.get("sent") == "bullish"]
        mags     = [float(e.get("mag", 0) or 0) for e in bullish]
        confs    = [float(e.get("conf", 0) or 0) for e in bullish]
        print(f"\n  {alias}  ({len(entries)} scored, {len(bullish)} bullish)")
        if mags:
            print(f"  magnitude: min={min(mags):.2f}  "
                  f"median={statistics.median(mags):.2f}  "
                  f"mean={sum(mags)/len(mags):.2f}  max={max(mags):.2f}")
            print(f"  confidence: mean={sum(confs)/len(confs):.2f}")
            # Distribution
            buckets = [0.0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0]
            for lo, hi in zip(buckets, buckets[1:]):
                n   = sum(1 for m in mags if lo <= m < hi)
                bar = "█" * (n // 3)
                print(f"    {lo:.1f}-{hi:.1f}  {n:3d}  {bar}")


def main():
    parser = argparse.ArgumentParser(description="Score backtest articles with Anthropic models")
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--dry-run",  action="store_true", help="Show cost estimate only, no API calls")
    mode.add_argument("--sample",   type=int, metavar="N", help="Score N articles per model (go/no-go)")
    mode.add_argument("--all",      action="store_true", help="Score all unscored articles")
    mode.add_argument("--report",   action="store_true", help="Print report on cached scores")
    parser.add_argument("--models", nargs="+", choices=list(MODELS.keys()),
                        default=DEFAULT_MODELS, help="Which models to score (default: all except fable)")
    parser.add_argument("--batch",  action="store_true", help="Use Batch API (cheaper, async, backtest only)")
    parser.add_argument("-y", "--yes", action="store_true", help="Skip confirmation prompt")
    args = parser.parse_args()

    if args.report:
        run_report(args)
    elif args.dry_run:
        universe = load_universe()
        cache    = load_cache()
        todo     = sum(1 for a in universe for alias in args.models
                       if cache_key(alias, a["hash"]) not in cache)
        print(f"\n📰 Universe: {len(universe)} articles")
        print(f"💾 Cached:   {len(cache)} existing scores")
        print(f"📋 To score: {todo} article-model pairs")
        print_cost_table(len(universe), args.models)
        # Production cost
        print(f"\n  Production cost (41 articles/day, live scoring with prompt cache):")
        print(f"  {'Model':<14} {'$/day':>8} {'$/month':>10}")
        print(f"  {'-'*34}")
        for alias in args.models:
            _, inp, out = MODELS[alias]
            inp_M = (41 * EST_INPUT_TOKENS)  / 1_000_000
            out_M = (41 * EST_OUTPUT_TOKENS) / 1_000_000
            cached_frac = 400 / EST_INPUT_TOKENS
            day = (inp_M * cached_frac * inp * 0.10 +
                   inp_M * (1 - cached_frac) * inp +
                   out_M * out)
            print(f"  {alias:<14} ${day:>7.4f}  ${day*30:>9.4f}")
    else:
        run_scoring(args)


if __name__ == "__main__":
    main()