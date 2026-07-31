"""
scoring.py (v2) — the primary scorer + the ticker-selection corrector.

Primary scorer is Sonnet 5 on anthropic_scorer's prompt (both swapped 2026-07-29 — see
config.py SCORER_MODEL and the SYSTEM_PROMPT comment below for the evidence and the
calibration-linkage argument). Kept as ONE function (score_article) so a future provider swap
touches only this file; that isolation is what made today's change a two-file edit.
v1 (core/bot.py) is untouched and still runs local Ollama on unified_v1.

Ticker-selection corrector (validated 2026-07-16, research/prompt_v2_test.py): a second, cheap
Ollama call that picks the primary beneficiary from the feed's own `symbols` metadata instead of
trusting the primary scorer's free-generated ticker. 85-96% agreement with sonnet5 on the hard
cases across 850 articles tested. Deliberately NEVER used as a trade/no-trade gate — tested at
96% force-pick rate on articles with no real beneficiary, so it only corrects the SYMBOL on
signals that already passed every other gate.
"""

import csv
import json
import logging
import os
import re
import time
from datetime import datetime, timezone

import anthropic
from openai import OpenAI

import config as cfg

log = logging.getLogger(__name__)

# Ollama client is now ONLY for correct_ticker() -- the primary scorer is Anthropic.
ollama_client = OpenAI(base_url=cfg.OLLAMA_BASE_URL, api_key="ollama")

# max_retries covers the container's real failure mode: DNS resolution inside Docker drops in
# short bursts (v2/logs/bot.log shows 32 such failures in two tight clusters, 07-24 11:11-11:12
# and 07-29 11:04-11:06, hitting www.sec.gov AND data.alpaca.markets alike -- transient host
# network changes, not a per-host misconfiguration). The SDK retries connection errors, 408, 409,
# 429 and 5xx with exponential backoff, so a burst that lasts seconds is absorbed rather than
# silently losing the signal. Timeout is per attempt; worst case is roughly
# SCORER_TIMEOUT_S * (SCORER_MAX_RETRIES + 1).
anthropic_client = anthropic.Anthropic(
    api_key=cfg.ANTHROPIC_API_KEY or None,      # None -> SDK's own env/profile resolution
    timeout=cfg.SCORER_TIMEOUT_S,
    max_retries=cfg.SCORER_MAX_RETRIES,
)

SCORER_LATENCY_CSV = "scorer_latency.csv"

# ── Primary scorer prompt — anthropic_scorer's, NOT unified_v1 (swapped 2026-07-29) ──────────
# jeff's call, and the decisive reason is CALIBRATION LINKAGE, not prompt quality:
#   v2's live gate (MIN_MAGNITUDE 0.35 / MIN_CONFIDENCE 0.70) was derived from
#   research/lotto_sonnet5_selectivity_grid.py, whose sonnet5 scores came from THIS prompt.
#   Running unified_v1 live meant gating a score distribution nobody had ever measured with
#   cutoffs fitted to a different one. Switching the prompt is the free half of that fix.
# Sonnet-on-THIS-prompt is the MEASURED combination (1,682 cached scores, conf med 0.65,
# mode 0.55 holding 35%); Sonnet-on-unified_v1 was the extrapolated one. So this direction
# reduces what we are guessing about, it does not add to it.
# Supporting evidence (llama3.2, temp 0.1, same 1,683 articles, prompt the ONLY variable --
# research/ollama_temp_test.py + scorer_pnl_matched.py --prompt-ab):
#   distribution: this prompt 32 distinct conf values, mode holds 39%, sd 0.230
#                 unified_v1   12 distinct conf values, mode holds 71%, sd 0.103
#   P&L @ n=50 matched trades: avg $+223/trade vs $+195 (also ahead at n=100 and on a
#                 magnitude-only ranking) -- consistent direction, but bootstrap CIs overlap
#                 almost entirely. Treat as "no evidence of harm", NOT as "proven better".
# TRADE-OFF ACCEPTED: this prompt drops unified_v1's `catalyst` output field and its few-shot
# examples separating resolved binary events from analyst opinion. Verified safe for v2 --
# nothing here consumes the scored catalyst value (v2/filters.py's soft_catalyst_hit is an
# unrelated headline regex that runs BEFORE scoring). v1 keeps unified_v1 and is unaffected.
SYSTEM_PROMPT = """You are a quantitative equity trading signal generator.
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


# Bumped from 1200 -> 2000 (2026-07-18) to fit the SEC EDGAR fix's real filing/press-release text
# (previously just an empty body) -- Alpaca/NewsAPI bodies are short blurbs well under either
# limit, so this only meaningfully changes what SEC-sourced signals see.
SCORE_BODY_CHARS = 2000


# Structured output schema -- replaces the old parse-the-text-and-hope path. With
# output_config.format the API CONSTRAINS generation to this schema, so the markdown-fence
# stripping and json.JSONDecodeError branch the Ollama version needed are gone: malformed JSON is
# no longer a reachable failure mode. `catalyst` is intentionally ABSENT: the schema mirrors the
# prompt exactly, and forcing a field the prompt never defines would make the model emit an
# undefined number rather than a meaningful one.
_SCORE_SCHEMA = {
    "type": "object",
    "properties": {
        "tickers":    {"type": "array", "items": {"type": "string"}},
        "sentiment":  {"type": "string", "enum": ["bullish", "bearish", "neutral"]},
        "confidence": {"type": "number"},
        "magnitude":  {"type": "number"},
        "reasoning":  {"type": "string"},
    },
    "required": ["tickers", "sentiment", "confidence", "magnitude", "reasoning"],
    "additionalProperties": False,
}


def _log_latency(ms: float, usage, ok: bool, model: str) -> None:
    """Per-call latency + token usage, to the log line's caller AND a CSV for analysis.
    Latency is the number that decides whether the adaptive-thinking default stays: this is a
    hot path (news edge decays in minutes) and it replaced a ~1-11s local call."""
    try:
        new = not os.path.exists(SCORER_LATENCY_CSV)
        with open(SCORER_LATENCY_CSV, "a", newline="") as f:
            w = csv.writer(f)
            if new:
                w.writerow(["ts", "model", "ok", "latency_ms", "input_tokens", "output_tokens",
                            "cache_read_tokens"])
            w.writerow([datetime.now(timezone.utc).isoformat(), model, int(ok), round(ms, 1),
                        getattr(usage, "input_tokens", "") if usage else "",
                        getattr(usage, "output_tokens", "") if usage else "",
                        getattr(usage, "cache_read_input_tokens", "") if usage else ""])
    except Exception as e:                       # never let telemetry break scoring
        log.debug("latency log failed: %s", e)


def preflight() -> bool:
    """Startup reachability + auth + model-access check for the scorer, run once before any news
    is processed. Exists because of how the container fails: DNS drops in bursts and the old
    Ollama scorer failed *silently* per-article (log.warning at most). Without this, a missing
    ANTHROPIC_API_KEY or a blocked egress path presents as "the bot is running but never trades" --
    the exact ambiguity that hid v2's zero-trade problem for a week. models.retrieve is free (no
    tokens) and validates DNS, TLS, credentials, and model entitlement in one call."""
    if not cfg.ANTHROPIC_API_KEY and not os.environ.get("ANTHROPIC_AUTH_TOKEN"):
        log.error("❌ ANTHROPIC_API_KEY is not set — the scorer cannot run. Add it to v2/.env "
                  "(docker-compose injects that file via env_file) and restart.")
        return False
    try:
        t0 = time.monotonic()
        m = anthropic_client.models.retrieve(cfg.SCORER_MODEL)
        log.info("✅ scorer reachable: %s (%s) in %.0fms", m.id, m.display_name,
                 (time.monotonic() - t0) * 1000)
        return True
    except anthropic.AuthenticationError:
        log.error("❌ scorer auth failed — ANTHROPIC_API_KEY is present but rejected.")
    except anthropic.NotFoundError:
        log.error("❌ scorer model %r not found or not entitled to this key.", cfg.SCORER_MODEL)
    except anthropic.APIConnectionError as e:
        log.error("❌ cannot reach api.anthropic.com from this container: %s. Check DNS/egress "
                  "(the container has had name-resolution bursts affecting sec.gov and "
                  "data.alpaca.markets too).", e)
    except Exception as e:
        log.error("❌ scorer preflight failed (%s): %s", type(e).__name__, e)
    return False


def score_article(headline: str, body: str, source: str = "") -> "dict | None":
    prefix = f"[Source: {source}]\n" if source else ""
    thinking = ({"type": "disabled"} if cfg.SCORER_THINKING == "disabled"
                else {"type": "adaptive"})
    t0 = time.monotonic()
    usage, ok = None, False
    try:
        resp = anthropic_client.messages.create(
            model=cfg.SCORER_MODEL,
            max_tokens=cfg.SCORER_MAX_TOKENS,
            system=SYSTEM_PROMPT,
            thinking=thinking,
            output_config={"effort": cfg.SCORER_EFFORT,
                           "format": {"type": "json_schema", "schema": _SCORE_SCHEMA}},
            messages=[{"role": "user", "content":
                       f"{prefix}Headline: {headline}\n\nBody: {body[:SCORE_BODY_CHARS]}"}],
        )
        usage = resp.usage
        # Safety classifiers can decline a request (HTTP 200 + stop_reason "refusal"), and
        # max_tokens can truncate -- either way `content` may be empty or partial, so never index
        # into it blindly.
        if resp.stop_reason == "refusal":
            log.warning("scorer refused (%s) — treating as unscorable",
                        getattr(resp.stop_details, "category", None))
            return None
        text = next((b.text for b in resp.content if b.type == "text"), None)
        if not text:
            log.warning("scorer returned no text (stop_reason=%s)", resp.stop_reason)
            return None
        signal = json.loads(text)
        ok = True
        return signal
    except anthropic.APIStatusError as e:
        log.warning("scorer API error %s: %s", e.status_code, e.message)
        return None
    except anthropic.APIConnectionError as e:
        # Already retried SCORER_MAX_RETRIES times by the SDK -- a burst outlasted the backoff.
        log.warning("scorer unreachable after retries: %s", e)
        return None
    except Exception as e:
        log.warning("scoring failed: %s", e)
        return None
    finally:
        ms = (time.monotonic() - t0) * 1000
        log.info("  ⏱️  scorer %s: %.0fms%s", cfg.SCORER_MODEL, ms,
                 f" ({usage.input_tokens}in/{usage.output_tokens}out)" if usage else " (failed)")
        _log_latency(ms, usage, ok, cfg.SCORER_MODEL)


# ── Ticker-selection corrector (validated research/prompt_v2_test.py) ───────────────────────
TICKER_CORRECTOR_PROMPT = """You are a financial news analyst. Respond with a JSON object ONLY — no markdown, no explanation.

Schema:
{ "ticker": "<one symbol from CANDIDATES, or NONE>" }

ticker: the ONE symbol from CANDIDATES whose company is the PRIMARY SUBJECT of this news AND the
party that most directly benefits or suffers from it. A company that is merely mentioned, or that
is on the LOSING side of the event (lost the lawsuit, is being outcompeted, is the acquirer paying
a premium), is NOT the answer. If the news is not mainly about any candidate, answer "NONE".
Never output a symbol that is not in CANDIDATES."""


def correct_ticker(headline: str, body: str, candidates: list[str]) -> "str | None":
    """Returns a corrected ticker if the model picks a valid candidate, else None (meaning:
    keep the primary scorer's own ticker — this function only ever CORRECTS, never gates)."""
    if not candidates:
        return None
    try:
        resp = ollama_client.chat.completions.create(
            model=cfg.OLLAMA_MODEL,
            messages=[
                {"role": "system", "content": TICKER_CORRECTOR_PROMPT},
                {"role": "user", "content": f"Headline: {headline}\n\nBody: {body[:1200]}\n\n"
                                            f"CANDIDATES: {candidates}\n\nRespond with JSON only."},
            ],
            temperature=0.1,
            max_tokens=60,
        )
        raw = resp.choices[0].message.content.strip()
        if raw.startswith("```"):
            raw = raw.split("```")[1]
            if raw.startswith("json"):
                raw = raw[4:]
        m = re.search(r"\{.*\}", raw, re.S)
        if not m:
            return None
        tk = json.loads(m.group(0)).get("ticker")
        tk = tk.strip().upper() if isinstance(tk, str) else None
        return tk if tk and tk in candidates else None
    except Exception as e:
        log.debug("ticker corrector failed: %s", e)
        return None
