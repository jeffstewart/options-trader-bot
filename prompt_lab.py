"""
prompt_lab.py — compare PROMPT × MODEL combos on signal quality (score → realized
5-day return), against the current baseline (Spearman +0.055, top-quintile +2.56%).

Re-scores a fixed sample of bullish articles (headline+body from the cache, no
re-fetch) with each prompt/model, then measures how well the resulting score
predicts the realized 5-day stock move. Scores are cached (LLM calls are slow),
so re-runs and added variants are cheap.

Models: local Ollama by name. To add a free-hosted model, set in .env e.g.
  LAB_HOSTED_BASE_URL=https://api.groq.com/openai/v1
  LAB_HOSTED_KEY=...      LAB_HOSTED_MODEL=llama-3.3-70b-versatile
and pass --hosted.

Usage:  USE_YAHOO_BARS=1 .venv/bin/python prompt_lab.py [--limit 80] [--models llama3.2] [--hosted]
"""
import argparse
import json
import os
from pathlib import Path

from dotenv import load_dotenv; load_dotenv()
from openai import OpenAI

import backtest as _bt
from signal_quality import load_events, fwd_return, spearman
import statistics

OLLAMA = OpenAI(base_url=os.environ.get("OLLAMA_BASE_URL", "http://localhost:11434/v1"), api_key="ollama")
SCORE_CACHE = Path("prompt_lab_scores.json")

# ── Prompt variants ───────────────────────────────────────────────────────────
BASELINE = _bt  # we'll read backtest.SYSTEM_PROMPT
PROMPTS = {
    "baseline": __import__("backtest").SYSTEM_PROMPT,

    "direct_return": """You are an equity analyst. Given news about a stock, estimate the stock's
% price change over the NEXT 5 TRADING DAYS. Respond JSON ONLY:
{"tickers":["AAPL"], "expected_return_pct": 3.5, "confidence": 0.7, "reasoning":"one sentence"}
expected_return_pct: your best estimate of the 5-day % move (negative allowed). Be realistic —
most news barely moves a stock (0–1%); reserve >5% for genuinely major, UNPRICED catalysts.
Only include tickers the news is genuinely about. Return ONLY the JSON object.""",

    "materiality": """You are a trading signal generator. The ONLY thing that matters is whether this
news is NEW, SURPRISING information the market has NOT already priced, and will move the stock UP over
the next few days. Respond JSON ONLY:
{"tickers":["AAPL"], "sentiment":"bullish", "confidence":0.7, "magnitude":0.6, "reasoning":"one sentence"}
magnitude (0–1): how much UNPRICED, market-moving surprise this contains (0 = already known/routine,
1 = major surprise that will move the stock). confidence (0–1): certainty it's genuinely bullish.
Be harsh: most news is already priced → magnitude < 0.3. Only tickers the news is truly about.
Return ONLY the JSON object.""",

    "catalyst_typed": """You are an equity catalyst classifier. Identify the catalyst and how reliably
that TYPE of catalyst moves a stock up over days. Respond JSON ONLY:
{"tickers":["AAPL"], "catalyst":"earnings_beat", "move_score":0.7, "confidence":0.7, "reasoning":"one sentence"}
catalyst: earnings_beat | guidance_raise | M&A | analyst | product | regulatory | macro | other.
move_score (0–1): historically, how strongly this specific catalyst + its size tends to move the stock
up over 5 days (M&A/guidance_raise/major beats high; analyst/routine low). confidence (0–1): certainty.
Only tickers the news is truly about. Return ONLY the JSON object.""",

    # ── New prompts targeting binary exclusion + calibrated surprise ──────────
    # Grounded in finding: LLM's edge is NOISE EXCLUSION, not graded ranking.
    # These push the model toward a conservative binary gate rather than a
    # continuous score, so that high scores genuinely mean something.

    "binary_gate": """You are a strict trading signal filter. Your job is to identify the rare news
events that will cause a stock to move MORE THAN 5% in the next week. The vast majority of news
(>85%) does NOT qualify — analyst reiterations, routine filings, minor updates, macro commentary,
and already-anticipated events are all NOISE. Only flag genuine surprises.
Respond JSON ONLY:
{"tickers":["AAPL"], "tradeable": true, "direction":"bullish", "confidence":0.8, "reasoning":"one sentence"}
tradeable: true ONLY if you believe this has >30% chance of a 5%+ move. false for everything else.
direction: "bullish" | "bearish" | "neutral"
confidence: 0.0–1.0 certainty about the direction IF tradeable (ignore if tradeable=false)
Calibration: a typical week has 1–2 truly tradeable events per 50 articles. Be that selective.
Return ONLY the JSON object.""",

    "materiality_fewshot": """You are a trading signal generator. Score news by how much UNPRICED
SURPRISE it contains — information the market has NOT already priced in.

EXAMPLES of HIGH magnitude (0.7+): unexpected earnings beat 30% above estimates, surprise FDA
approval, company being acquired at premium, major guidance raise nobody expected.
EXAMPLES of LOW magnitude (<0.2): analyst reiterates price target, in-line quarterly result,
company announces routine buyback, CEO gives speech at conference.
EXAMPLES of ZERO (no trade): macro/political news, no specific company, price-target cut on
routine basis, "investors should watch" articles.

Respond JSON ONLY:
{"tickers":["AAPL"], "sentiment":"bullish", "confidence":0.7, "magnitude":0.6, "reasoning":"one sentence"}
magnitude (0–1): proportion of this news that was NOT already priced by the market.
confidence (0–1): certainty about the direction.
Most news is already priced → magnitude <0.2. Return ONLY the JSON object.""",

    "materiality_fewshot_v2": """You are a trading signal generator. Score news by how much UNPRICED
SURPRISE it contains — information the market has NOT already priced in. Most news is routine and
already priced (magnitude <0.2). Only a genuine, unexpected EVENT earns a high magnitude.

CRITICAL RULE: an analyst rating, price-target change, or "maintains/reiterates" note is NOT
surprise — it is routine opinion the market already discounts. Score magnitude LOW even when
bullish. The surprise that MOVES a stock is a concrete EVENT (earnings beat, M&A, FDA, a trading
halt, a guidance change), not someone's opinion about the stock.

Few-shot examples (headline → the magnitude/confidence you should output):
- "Wells Fargo Maintains Overweight on JM Smucker, Raises Price Target to $130" → magnitude 0.12, confidence 0.85 (routine analyst note — already priced)
- "Guggenheim Reiterates Buy on Zscaler, Maintains $214 Target" → magnitude 0.10, confidence 0.85 (no new event)
- "Acme Corp Reports Q3 EPS $2.10 vs $1.60 est, Raises FY Guidance" → magnitude 0.82, confidence 0.90 (large unexpected beat + raise)
- "XYZ Shares Halted, Circuit Breaker To The Upside" → magnitude 0.88, confidence 0.85 (a violent unpriced move underway)
- "BioCo Receives Surprise FDA Approval for Lead Drug" → magnitude 0.85, confidence 0.90 (binary catalyst resolved favorably)
- "MegaCorp to Acquire SmallCo for $50/share, 40% Premium" → magnitude 0.90, confidence 0.95 (M&A at premium)
- "Company Announces Routine $1B Buyback" → magnitude 0.12, confidence 0.80 (expected capital return)
- "5 Stocks Investors Should Watch" / macro / political → magnitude 0.0, confidence 0.0, sentiment neutral (no specific unpriced event)

Respond JSON ONLY:
{"tickers":["AAPL"], "sentiment":"bullish", "confidence":0.7, "magnitude":0.6, "reasoning":"one sentence"}
magnitude (0–1): proportion of this news NOT already priced. confidence (0–1): direction certainty.
Return ONLY the JSON object.""",

    "materiality_fewshot_v3": """You score news for a LOTTERY-TICKET options strategy: you only care
about news likely to trigger a LARGE, FAST repricing — a gap up or a trading halt — NOT a slow drift
and NOT an opinion. magnitude = the probability this is a genuine, UNPRICED, market-moving EVENT.
Most news is incremental and already priced → magnitude <0.2.

ONLY a RESOLVED binary catalyst earns high magnitude — an event that just REMOVED uncertainty:
earnings PRINTED (big beat/miss), FDA decision MADE, deal SIGNED/announced, a trading HALT, a
guidance change, a major contract WON. Anticipation, speculation, opinion, analyst ratings,
"could/may/plans to", partnerships, expansions, routine product launches, and incremental updates
are NOT catalysts → magnitude LOW even when bullish.

Few-shot (headline → magnitude, confidence):
HIGH (resolved, unpriced, violent-move event):
- "Acme Reports Q3 EPS $2.10 vs $1.60 est, Raises FY Guidance" → 0.82, 0.90
- "XYZ Shares Halted, Circuit Breaker To The Upside" → 0.90, 0.85
- "BioCo Wins Surprise FDA Approval for Lead Drug" → 0.85, 0.90
- "MegaCorp to Acquire SmallCo for $50/share, 40% Premium" → 0.92, 0.95
- "DefenseCo Wins $5B Pentagon Contract, Doubles Backlog" → 0.75, 0.85
LOW (opinion / incremental / vague / already-priced):
- "Wells Fargo Maintains Overweight on Smucker, Raises Price Target" → 0.10, 0.85
- "Morgan Stanley Upgrades XYZ to Buy" → 0.18, 0.80
- "Acme Announces Partnership With Beta Corp" → 0.15, 0.70
- "XYZ Expands Into European Market" → 0.15, 0.70
- "CEO Presents at Tech Conference" → 0.05, 0.60
- "Company Completes Routine $1B Buyback" → 0.12, 0.75
ZERO: macro / political / "stocks to watch" / no specific company → magnitude 0.0, confidence 0.0, neutral.

Respond JSON ONLY:
{"tickers":["AAPL"], "sentiment":"bullish", "confidence":0.7, "magnitude":0.6, "reasoning":"one sentence"}
Return ONLY the JSON object.""",

    "surprise_score": """You are an equity surprise detector. For each news item, estimate the
SURPRISE factor — how different this news is from what the market expected.
Respond JSON ONLY:
{"tickers":["AAPL"], "sentiment":"bullish", "surprise":0.7, "magnitude":0.6, "confidence":0.7, "reasoning":"one sentence"}
surprise (0–1): 0 = market fully expected this, 1 = complete shock to the market.
magnitude (0–1): how much will the stock move in the next 1–5 days.
confidence (0–1): certainty about the direction.
Key insight: SURPRISE × MAGNITUDE is what drives returns, not magnitude alone.
A big event that was expected (surprise=0.1) won't move the stock much.
A small event nobody expected (surprise=0.9) can move it a lot.
Only include tickers this news is directly about. Return ONLY the JSON object.""",

    # ── Lotto-specific: targets large-swing binary events ──────────────────────
    # The lotto strategy buys cheap OTM calls — it only pays off on a LARGE,
    # FAST move (20%+ in 1-2 days). A different question than "will this move 5%."
    # This prompt explicitly asks about extreme-move probability so we can test
    # whether the LLM can identify lotto-grade catalysts vs ordinary catalysts.
    # Graded with --win 15 or --win 20 at fwd=1 in grade_scores.py.
    "lotto_swing": """You are an options trader focused on binary catalysts. Identify news that has
a meaningful chance of causing a LARGE, FAST stock move (20%+ in 1-2 trading days).
These are rare: FDA binary decisions, blowout earnings beats 3×+ above estimates,
surprise M&A announcements, short squeeze catalysts, major unexpected contract wins.
Respond JSON ONLY:
{"tickers":["AAPL"], "sentiment":"bullish", "swing_prob":0.15, "confidence":0.8, "reasoning":"one sentence"}
swing_prob (0–1): estimated probability of a 20%+ move in 1-2 days. For most news this is <0.05.
Realistic calibration: a genuine binary catalyst (FDA readout, surprise takeover) might be 0.20–0.40.
An earnings beat might be 0.05–0.15 depending on magnitude. Routine upgrades/partnerships: <0.03.
confidence (0–1): certainty about the direction (bullish/bearish).
Only include tickers this news is directly about. Return ONLY the JSON object.""",

    # ── UNIFIED prompt: the live main scorer + a `catalyst` field that absorbs the
    # materiality judgment, so a SINGLE pass yields everything the routing rules need
    # (goal: delete the separate materiality 2nd pass; rules route by mag/conf/catalyst).
    "unified_v1": """You are a quantitative equity trading signal generator.
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
- Return ONLY the JSON object.""",

    # ── v2 = unified_v1 + REASONING-FIRST (chain-of-thought): reasoning is the FIRST field so the
    # 3B model reasons before committing numbers. Keeps catalyst → isolates the CoT effect.
    "unified_v2": """You are a quantitative equity trading signal generator.
Respond with a JSON object ONLY — no markdown, no code fences, no explanation.

Schema — fill "reasoning" FIRST (state the event and why it would/wouldn't move the stock), THEN
assign every score CONSISTENT with that reasoning:
{
  "reasoning":  "one sentence: the event and why it moves / doesn't move the stock",
  "tickers":    ["AAPL"],
  "sentiment":  "bullish",
  "confidence": 0.82,
  "magnitude":  0.75,
  "catalyst":   0.60
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
  earnings PRINTED, FDA decision MADE, deal SIGNED, a trading HALT, a guidance change, a major
  contract WON → 0.7–0.95. Anticipation, opinion, ANALYST ratings/price targets, "could/may/plans
  to", partnerships, expansions, routine launches → catalyst < 0.2 EVEN WHEN BULLISH. Macro → 0.0.
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
- Return ONLY the JSON object.""",

    # ── v3 = STRIP catalyst (proven vestigial) + migrate its teaching into explicit MAGNITUDE
    # few-shot (headline→magnitude). Tests whether a simpler schema sharpens mag/conf. reasoning last.
    "unified_v3": """You are a quantitative equity trading signal generator.
Respond with a JSON object ONLY — no markdown, no code fences, no explanation.

Schema:
{
  "tickers":    ["AAPL"],
  "sentiment":  "bullish",
  "confidence": 0.82,
  "magnitude":  0.75,
  "reasoning":  "one sentence"
}

sentiment:  "bullish" | "bearish" | "neutral"
confidence: float 0.0–1.0 — how certain you are about the sentiment direction
magnitude:  float 0.0–1.0 — how likely this news is to meaningfully move the stock price

Magnitude scale:
  0.0–0.2  Noise or irrelevant (routine filings, minor analyst reiterations, fluff)
  0.2–0.4  Routine news (in-line earnings, small contract wins, minor upgrades)
  0.4–0.6  Meaningful catalyst (solid earnings beat, notable partnership, meaningful guidance raise)
  0.6–0.8  Strong catalyst (significant beat, major acquisition, landmark FDA approval for key drug)
  0.8–1.0  Transformative event (company-defining deal, paradigm-shifting approval, massive revision)

Magnitude few-shot (headline → magnitude) — ONLY a RESOLVED binary event that just REMOVED
uncertainty earns high magnitude; opinion / analyst notes / anticipation stay LOW even when bullish:
- "Reports Q3 EPS $2.10 vs $1.60 est, Raises Guidance" → 0.85
- "Shares Halted, Circuit Breaker To The Upside" → 0.90
- "Wins Surprise FDA Approval" → 0.85
- "To Acquire X at 40% Premium" → 0.92
- "Wins $5B Contract, Doubles Backlog" → 0.75
- "Wells Fargo Maintains Overweight, Raises PT" → 0.10
- "Morgan Stanley Upgrades to Buy" → 0.18
- "Announces Partnership With Beta Corp" → 0.15
- "Expands Into European Market" → 0.15
- "CEO Presents at Tech Conference" → 0.05

Rules:
- Only include tickers you are highly confident about.
- General macro news with no specific company → empty tickers list.
- Most news is routine — be conservative; reserve magnitude 0.7+ for genuinely exceptional, RESOLVED events.
- Only flag bullish sentiment — we trade long calls only.
- Return ONLY the JSON object.""",

    # ── v4 = v2 + v3: reasoning-FIRST AND strip-catalyst + magnitude few-shot. Combined best-guess.
    "unified_v4": """You are a quantitative equity trading signal generator.
Respond with a JSON object ONLY — no markdown, no code fences, no explanation.

Schema — fill "reasoning" FIRST (state the event and why it would/wouldn't move the stock), THEN
assign every score CONSISTENT with that reasoning:
{
  "reasoning":  "one sentence: the event and why it moves / doesn't move the stock",
  "tickers":    ["AAPL"],
  "sentiment":  "bullish",
  "confidence": 0.82,
  "magnitude":  0.75
}

sentiment:  "bullish" | "bearish" | "neutral"
confidence: float 0.0–1.0 — how certain you are about the sentiment direction
magnitude:  float 0.0–1.0 — how likely this news is to meaningfully move the stock price

Magnitude scale:
  0.0–0.2  Noise or irrelevant (routine filings, minor analyst reiterations, fluff)
  0.2–0.4  Routine news (in-line earnings, small contract wins, minor upgrades)
  0.4–0.6  Meaningful catalyst (solid earnings beat, notable partnership, meaningful guidance raise)
  0.6–0.8  Strong catalyst (significant beat, major acquisition, landmark FDA approval for key drug)
  0.8–1.0  Transformative event (company-defining deal, paradigm-shifting approval, massive revision)

Magnitude few-shot (headline → magnitude) — ONLY a RESOLVED binary event that just REMOVED
uncertainty earns high magnitude; opinion / analyst notes / anticipation stay LOW even when bullish:
- "Reports Q3 EPS $2.10 vs $1.60 est, Raises Guidance" → 0.85
- "Shares Halted, Circuit Breaker To The Upside" → 0.90
- "Wins Surprise FDA Approval" → 0.85
- "To Acquire X at 40% Premium" → 0.92
- "Wins $5B Contract, Doubles Backlog" → 0.75
- "Wells Fargo Maintains Overweight, Raises PT" → 0.10
- "Morgan Stanley Upgrades to Buy" → 0.18
- "Announces Partnership With Beta Corp" → 0.15
- "Expands Into European Market" → 0.15
- "CEO Presents at Tech Conference" → 0.05

Rules:
- Only include tickers you are highly confident about.
- General macro news with no specific company → empty tickers list.
- Most news is routine — be conservative; reserve magnitude 0.7+ for genuinely exceptional, RESOLVED events.
- Only flag bullish sentiment — we trade long calls only.
- Return ONLY the JSON object.""",
}


def extract_score(obj):
    """Uniform score from any variant's JSON — always returns a float in [0,1]
    (or a signed float for direct_return) that ranks articles by expected alpha.

    Priority order reflects what each prompt is designed to measure:
      binary_gate       → tradeable (bool) × confidence → 0 or conf
      surprise_score    → surprise × magnitude × confidence
      direct_return     → expected_return_pct × confidence (signed, natural rank)
      materiality*      → magnitude × confidence (unpriced surprise × certainty)
      move_score        → move_score × confidence (catalyst_typed)
      fallback          → magnitude × confidence
    """
    try:
        conf = float(obj.get("confidence", 0.5) or 0.5)
        # lotto_swing: swing_prob is the primary score
        if "swing_prob" in obj:
            return float(obj["swing_prob"]) * conf
        # binary_gate: tradeable=true/false is the primary gate
        if "tradeable" in obj:
            return conf if obj.get("tradeable") else 0.0
        # surprise_score: surprise × magnitude captures the cross-term
        if "surprise" in obj:
            s = float(obj.get("surprise", 0))
            m = float(obj.get("magnitude", 0))
            return s * m * conf
        if "expected_return_pct" in obj:
            return float(obj["expected_return_pct"]) * conf
        if "move_score" in obj:
            return float(obj["move_score"]) * conf
        return float(obj.get("magnitude", 0)) * conf
    except Exception:
        return None


def score_article(client, model, system, headline, body):
    import time
    for attempt in range(3):
        try:
            resp = client.chat.completions.create(
                model=model, temperature=0.1,
                messages=[{"role": "system", "content": system},
                          {"role": "user", "content": f"Headline: {headline}\n\nBody: {body[:1200]}\n\nJSON only."}])
            raw = resp.choices[0].message.content.strip()
            # Strip reasoning-model <think>...</think> blocks (e.g. qwen3)
            if "<think>" in raw:
                raw = raw.split("</think>")[-1].strip()
            if raw.startswith("```"):
                raw = raw.split("```")[1].lstrip("json").strip()
            return extract_score(json.loads(raw))
        except Exception as e:
            # rate limit / transient → back off and retry; parse error → give up
            if "429" in str(e) or "rate" in str(e).lower():
                time.sleep(2.0 * (attempt + 1))
                continue
            return None
    return None


def load_score_cache():
    return json.loads(SCORE_CACHE.read_text()) if SCORE_CACHE.exists() else {}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=80)
    ap.add_argument("--fwd", type=int, default=5)
    ap.add_argument("--models", default="llama3.2")
    ap.add_argument("--prompts", default="baseline,direct_return,materiality")
    ap.add_argument("--hosted", action="store_true")
    ap.add_argument("--hosted-models", default="",
                    help="comma-sep Groq model ids to compare (default: LAB_HOSTED_MODEL)")
    args = ap.parse_args()

    models = [("ollama", m) for m in args.models.split(",") if m.strip()]
    if args.hosted:
        hm = [m.strip() for m in (args.hosted_models or os.environ.get("LAB_HOSTED_MODEL", "")).split(",") if m.strip()]
        models += [("hosted", m) for m in hm]
    prompts = args.prompts.split(",")

    # Build sample directly from the cache so each event carries headline+body
    # (needed to re-score) + ticker/dt, then attach the realized forward return.
    import re
    from datetime import datetime, timezone
    _TK = re.compile(r"^[A-Z]{1,5}$")
    cache = json.load(open("dual_score_cache.json"))
    raw = []
    for v in cache.values():
        if not isinstance(v, dict):
            continue
        a = v.get("_article", {}) or {}
        b = v.get("bullish", {}) or {}
        cands = [t for t in (b.get("tickers", []) or []) if _TK.match(t) and t not in ("BTC", "ETH")]
        if not cands or not a.get("headline") or not a.get("created_at"):
            continue
        try:
            dt = datetime.fromisoformat(str(a["created_at"]).replace("Z", "+00:00")).astimezone(timezone.utc)
        except Exception:
            continue
        raw.append({"ticker": cands[0], "dt": dt, "headline": a["headline"],
                    "body": a.get("summary", "")})
    raw.sort(key=lambda e: e["dt"])
    step = max(1, len(raw) // (args.limit * 3))
    print(f"Building sample with realized {args.fwd}d returns from {len(raw)} candidates…")
    sample = []
    for e in raw[::step]:
        if len(sample) >= args.limit:
            break
        r = fwd_return(e["ticker"], e["dt"], args.fwd)
        if r is not None:
            e["_ret"] = r
            sample.append(e)
    print(f"Sample: {len(sample)} events with {args.fwd}d returns")

    sc = load_score_cache()
    hosted_client = None
    if args.hosted:
        hkey = os.environ.get("LAB_HOSTED_TEST_KEY") or os.environ.get("LAB_HOSTED_KEY")
        hosted_client = OpenAI(base_url=os.environ["LAB_HOSTED_BASE_URL"], api_key=hkey)
        print(f"  (hosted via {'TEST' if os.environ.get('LAB_HOSTED_TEST_KEY') else 'LIVE'} key)")

    print(f"\n{'model':18} {'prompt':16} {'n':>4} {'corr':>7} {'Q5 ret%':>8} {'Q1 ret%':>8}")
    for kind, model in models:
        client = hosted_client if kind == "hosted" else OLLAMA
        for pname in prompts:
            system = PROMPTS[pname]
            pairs = []
            for e in sample:
                key = f"{kind}:{model}:{pname}:{_bt.cache_key(e['headline'], e['body'])}"
                if key in sc:
                    s = sc[key]
                else:
                    s = score_article(client, model, system, e["headline"], e["body"])
                    sc[key] = s
                    SCORE_CACHE.write_text(json.dumps(sc))
                if s is not None:
                    pairs.append((s, e["_ret"]))
            if len(pairs) < 20:
                print(f"  {model:18} {pname:16} {len(pairs):>4}  (too few)")
                continue
            ss = [p[0] for p in pairs]; rr = [p[1] for p in pairs]
            order = sorted(pairs, key=lambda x: x[0]); q = len(order) // 5
            q5 = statistics.mean([x[1] for x in order[-q:]])
            q1 = statistics.mean([x[1] for x in order[:q]])
            print(f"  {model:18} {pname:16} {len(pairs):>4} {spearman(ss, rr):>+7.3f} {q5:>+7.2f}% {q1:>+7.2f}%")


if __name__ == "__main__":
    main()
