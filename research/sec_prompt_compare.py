"""
sec_prompt_compare.py — research-only (2026-07-23): compares the current SYSTEM_PROMPT against a
candidate SEC-specific addendum on the real filing content behind today's SEC-sourced bullish
signals, plus the big-movers found earlier (ADTN, SMCI) that never reached scoring.

v2 of this script: the first version fetched raw HTML and stripped tags itself instead of
reusing sec_edgar.fetch_8k_content -- that missed the skip_to_item logic (every 8-K opens with
~2KB of identical SEC cover-page boilerplate before the Item narrative; sec_edgar.py's own
docstring notes this was already root-caused 2026-07-18) and fed the model mostly boilerplate.
Re-running the UNCHANGED current prompt on that bad input gave a completely different (neutral)
result vs the live bullish call on the SAME filing -- 5 repeat runs confirmed it wasn't model
randomness, it was different input. This version imports sec_edgar.py directly so the fetched
body is byte-for-byte what the live bot actually saw.

Addendum targets three gaps identified from critically reading the PSKY filing (Paramount
Skydance / Warner Bros. Discovery merger, EU clearance 2026-07-22, scored bullish mag=0.75 but
fell 12-20% within minutes):
  1. No distinction between a FRESH catalyst and one step in an ALREADY-KNOWN multi-step process.
  2. No acquirer-vs-target awareness for M&A filings.
  3. No source-level skepticism that SEC 8-Ks often formalize news financial media already
     covered hours earlier.

Usage:  .venv/bin/python research/sec_prompt_compare.py
"""
import asyncio
import csv
import json
import os
import re
import sys
import time

import aiohttp
import requests
from dotenv import load_dotenv
from openai import OpenAI

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "core"))
import sec_edgar

load_dotenv()

SEC_UA = "Trader Bot research admin@example.com"
OLLAMA_MODEL = os.environ.get("OLLAMA_MODEL", "llama3.2")
SCORE_BODY_CHARS = 2000

ollama_client = OpenAI(base_url=os.environ.get("OLLAMA_BASE_URL", "http://localhost:11434/v1"), api_key="ollama")

CURRENT_PROMPT = """You are a quantitative equity trading signal generator.
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

SEC_ADDENDUM = """

Additional guidance for SEC EDGAR 8-K filings specifically:
- SEC filings often FORMALIZE news that financial media already reported hours or days earlier
  (press releases, analyst coverage). Unless the filing is clearly the FIRST public disclosure of
  genuinely new information, treat magnitude more conservatively than you would for a fresh news
  article — the market has likely already priced in widely-covered news by the time the formal
  filing appears.
- For M&A / regulatory-approval filings: identify whether the SCORED TICKER is the ACQUIRER or the
  TARGET of the transaction. Approval news is a much more reliable catalyst for the TARGET (price
  mechanically converges toward the deal price). For the ACQUIRER, routine deal-progress news (one
  of several required regulatory approvals) is usually a MUCH WEAKER catalyst — it mainly removes
  deal-break risk rather than adding new value, and can carry integration/overpayment concerns.
  Only score high magnitude for the acquirer if this is explicitly the FINAL outstanding condition
  for deal completion, or the filing reveals genuinely new economic terms.
- Look for signals that a disclosed event is ONE STEP in an ALREADY-KNOWN, MULTI-STEP process (the
  filing mentions prior related approvals/dates, says "as previously disclosed", or states the
  transaction "remains subject to other conditions"). This is a strong signal the event was ALREADY
  EXPECTED by the market and should generally lower both magnitude and catalyst, even though the
  event itself is technically "resolved.\""""

NEW_PROMPT = CURRENT_PROMPT + SEC_ADDENDUM


def score(prompt: str, headline: str, body: str) -> dict:
    resp = ollama_client.chat.completions.create(
        model=OLLAMA_MODEL,
        messages=[
            {"role": "system", "content": prompt},
            {"role": "user", "content": f"[Source: SEC EDGAR 8-K]\nHeadline: {headline}\n\nBody: {body[:SCORE_BODY_CHARS]}\n\nRespond with JSON only."},
        ],
        temperature=0.1,
    )
    raw = resp.choices[0].message.content.strip()
    if raw.startswith("```"):
        raw = raw.split("```")[1]
        if raw.startswith("json"):
            raw = raw[4:]
    try:
        return json.loads(raw.strip())
    except Exception as e:
        return {"error": str(e), "raw": raw[:200]}


async def fetch_real_content(cik: int, ticker: str) -> dict:
    """Byte-for-byte what the live bot would have fetched: latest 8-K's RSS summary (item codes)
    + fetch_8k_content's primary-doc-skip-to-item + exhibit, exactly as sec_rss_poller builds it
    (core/bot.py:3578-3590)."""
    r = requests.get(
        f"https://www.sec.gov/cgi-bin/browse-edgar?action=getcompany&CIK={cik:010d}&type=8-K&dateb=&owner=include&count=1&output=atom",
        headers={"User-Agent": SEC_UA}, timeout=15)
    m_href = re.search(r"<filing-href>(.+?)</filing-href>", r.text)
    m_title = re.search(r"<conformed-name>(.+?)</conformed-name>", r.text)
    m_items = re.search(r"<items-desc>(.+?)</items-desc>", r.text)
    m_date = re.search(r"<filing-date>(.+?)</filing-date>", r.text)
    if not m_href:
        return {}
    href = m_href.group(1)
    cik2, accession = sec_edgar.extract_cik_accession(href)
    if not cik2:
        # index href form: /Archives/edgar/data/{cik}/{accession-dashed}/{accession-dashed}-index.htm
        m = re.search(r"/data/(\d+)/(\d{18})", href.replace("-", ""))
        if m:
            cik2, accession = int(m.group(1)), m.group(2)
    company = m_title.group(1) if m_title else ""
    items_desc = m_items.group(1) if m_items else ""
    summary = f"Item {items_desc}" if items_desc else ""

    full_text = ""
    if cik2 and accession:
        async with aiohttp.ClientSession() as session:
            try:
                full_text = await sec_edgar.fetch_8k_content(session, cik2, accession, SEC_UA)
            except Exception as e:
                print(f"    fetch_8k_content failed: {e}")

    body = f"{summary}\n\n{full_text}" if full_text else summary
    return {"company": company, "body": body, "date": m_date.group(1) if m_date else "", "items": items_desc}


ITEMS = [
    ("FSBC", 1275168), ("MNSB", 1693577), ("STLD", 1022671), ("TEX", 97216),
    ("FATN", 1993400), ("HGIT", 1585101), ("DHI", 882184), ("MLI", 89439),
    ("RRC", 315852), ("CHCO", 726854), ("GCBC", 1070524), ("PSKY", 2041610),
    ("ADTN", 926282), ("SMCI", 1375365),
]


async def main():
    rows = []
    for ticker, cik in ITEMS:
        print(f"fetching {ticker} (CIK {cik})...")
        content = await fetch_real_content(cik, ticker)
        if not content.get("body") or not content["body"].strip():
            print(f"  no content, skipping")
            continue
        headline = f"8-K - {content['company']} ({cik:010d}) (Filer) [{ticker}]"
        body = content["body"]

        cur = score(CURRENT_PROMPT, headline, body)
        new = score(NEW_PROMPT, headline, body)
        rows.append({"ticker": ticker, "date": content.get("date"), "items": content.get("items"),
                     "current": cur, "new": new})
        print(f"  filed {content.get('date')}, items {content.get('items')}")
        print(f"  current: mag={cur.get('magnitude')} cat={cur.get('catalyst')} sent={cur.get('sentiment')}  {cur.get('reasoning','')}")
        print(f"      new: mag={new.get('magnitude')} cat={new.get('catalyst')} sent={new.get('sentiment')}  {new.get('reasoning','')}")
        time.sleep(0.3)

    out_path = os.path.join(os.path.dirname(__file__), "..", "data", "sec_prompt_compare.csv")
    with open(out_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["ticker", "filed", "items", "cur_sentiment", "cur_mag", "cur_cat", "cur_reasoning",
                    "new_sentiment", "new_mag", "new_cat", "new_reasoning"])
        for r in rows:
            c, n = r["current"], r["new"]
            w.writerow([r["ticker"], r.get("date"), r.get("items"),
                       c.get("sentiment"), c.get("magnitude"), c.get("catalyst"), c.get("reasoning", ""),
                       n.get("sentiment"), n.get("magnitude"), n.get("catalyst"), n.get("reasoning", "")])

    print(f"\n{len(rows)} comparisons written to {out_path}")


if __name__ == "__main__":
    asyncio.run(main())
