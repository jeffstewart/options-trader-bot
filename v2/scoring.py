"""
scoring.py (v2) — the primary scorer + the ticker-selection corrector.

Primary scorer stays local Ollama, unified_v1 prompt (char-identical to v1's core/bot.py --
this exact text is what's been validated all session; changing it is a fresh experiment, not
a port). Kept as ONE function (score_article) so swapping providers later — the sonnet5
decision is explicitly deferred, see config.py — only touches this file.

Ticker-selection corrector (validated 2026-07-16, research/prompt_v2_test.py): a second, cheap
Ollama call that picks the primary beneficiary from the feed's own `symbols` metadata instead of
trusting the primary scorer's free-generated ticker. 85-96% agreement with sonnet5 on the hard
cases across 850 articles tested. Deliberately NEVER used as a trade/no-trade gate — tested at
96% force-pick rate on articles with no real beneficiary, so it only corrects the SYMBOL on
signals that already passed every other gate.
"""

import json
import logging
import re

from openai import OpenAI

import config as cfg

log = logging.getLogger(__name__)

ollama_client = OpenAI(base_url=cfg.OLLAMA_BASE_URL, api_key="ollama")

# ── Primary scorer prompt -- unified_v1, unchanged from v1 ──────────────────────────────────
SYSTEM_PROMPT = """You are a quantitative equity trading signal generator.
Respond with a JSON object ONLY — no markdown, no code fences, no explanation.

Schema:
{
  "tickers":    ["AAPL"],
  "sentiment":  "bullish",
  "confidence": 0.82,
  "magnitude":  0.75,
  "catalyst":   0.60,
  "reasoning":  "one sentence"
}

sentiment:  "bullish" | "bearish" | "neutral"
confidence: float 0.0–1.0 — how certain you are about the sentiment direction
magnitude:  float 0.0–1.0 — how likely this news is to meaningfully move the stock price
catalyst:   float 0.0–1.0 — probability this is a RESOLVED, UNPRICED, binary event likely to trigger a
            LARGE, FAST repricing (a gap up or a trading halt) — NOT a slow drift and NOT an opinion

Magnitude scale:
  0.0–0.2  Noise or irrelevant (routine filings, minor analyst reiterations, fluff)
  0.2–0.4  Routine news (in-line earnings, small contract wins, minor upgrades)
  0.4–0.6  Meaningful catalyst (solid earnings beat, notable partnership, meaningful guidance raise)
  0.6–0.8  Strong catalyst (significant beat, major acquisition, landmark FDA approval for key drug)
  0.8–1.0  Transformative event (company-defining deal, paradigm-shifting approval, massive revision)

Catalyst scale — ONLY a RESOLVED binary event that just REMOVED uncertainty earns high catalyst:
  earnings PRINTED (big beat/miss), FDA decision MADE, deal SIGNED/announced, a trading HALT, a
  guidance change, a major contract WON → catalyst 0.7–0.95.
  Anticipation, speculation, opinion, ANALYST ratings/price targets, "could/may/plans to",
  partnerships, expansions, routine product launches, incremental updates → catalyst < 0.2 EVEN WHEN
  BULLISH. Macro / no specific company → catalyst 0.0.
  Examples: "Reports Q3 EPS $2.10 vs $1.60 est, Raises Guidance"→0.85 · "Shares Halted, Circuit
  Breaker To The Upside"→0.90 · "Wins Surprise FDA Approval"→0.85 · "To Acquire X at 40% Premium"→0.92
  · "Wells Fargo Maintains Overweight, Raises PT"→0.10 · "Morgan Stanley Upgrades to Buy"→0.18 ·
  "Announces Partnership With Beta Corp"→0.15 · "CEO Presents at Tech Conference"→0.05

Rules:
- Only include tickers you are highly confident about.
- General macro news with no specific company → empty tickers list.
- Most news is routine — be conservative; reserve magnitude 0.7+ AND catalyst 0.7+ for genuinely
  exceptional, RESOLVED events.
- Only flag bullish sentiment — we trade long calls only.
- Return ONLY the JSON object."""


def score_article(headline: str, body: str, source: str = "") -> "dict | None":
    prefix = f"[Source: {source}]\n" if source else ""
    try:
        resp = ollama_client.chat.completions.create(
            model=cfg.OLLAMA_MODEL,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": f"{prefix}Headline: {headline}\n\nBody: {body[:1200]}\n\nRespond with JSON only."},
            ],
            temperature=0.1,
        )
        raw = resp.choices[0].message.content.strip()
        if raw.startswith("```"):
            raw = raw.split("```")[1]
            if raw.startswith("json"):
                raw = raw[4:]
        return json.loads(raw.strip())
    except json.JSONDecodeError as e:
        log.warning("JSON parse failed: %s", e)
        return None
    except Exception as e:
        log.warning("Ollama scoring failed: %s", e)
        return None


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
