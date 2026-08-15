"""
sec_backtest.py — research-only (2026-07-23): broader historical backtest of SEC 8-K prompt
scoring, built on the now-reproducible fetch path (sec_rss_poller no longer prefixes the
ephemeral RSS summary -- the body sent to the model is 100% derivable from (cik, accession)
alone, confirmed by re-fetching a known filing and getting byte-identical content).

Uses SEC's full-text search (efts.sec.gov) to find a broad, diverse sample of material 8-Ks
across many companies over a ~2 week window -- much more efficient than polling company-by-
company. For each: fetch the exact self-contained body via sec_edgar.fetch_8k_content (the SAME
function the live bot uses), get the stock's same-day price move as ground truth, and score with
multiple prompt variants to see which one's magnitude/catalyst actually correlates with real
outcomes -- not just internal consistency.

Usage:  .venv/bin/python research/sec_backtest.py
"""
import asyncio
import csv
import json
import os
import re
import sys
import time
from datetime import datetime, timedelta, timezone

import aiohttp
import requests
from dotenv import load_dotenv
from openai import OpenAI

os.environ.setdefault("USE_YAHOO_BARS", "1")
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "core"))
import sec_edgar
import backtest as _bt
import config as _cfg

load_dotenv()

SEC_UA = "Trader Bot research admin@example.com"
OLLAMA_MODEL = os.environ.get("OLLAMA_MODEL", "llama3.2")
SCORE_BODY_CHARS = 2000
N_TARGET = 40

ollama_client = OpenAI(base_url=os.environ.get("OLLAMA_BASE_URL", "http://localhost:11434/v1"), api_key="ollama")

MATERIAL_ITEMS = sec_edgar.MATERIAL_ITEMS

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

FULL_ADDENDUM = """

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

# Isolates just the piece that showed a clear, validated win in the smaller comparison (TEX/MNSB:
# suppressing scheduling announcements and routine dividends that scored too high) from the
# unvalidated pieces (staleness, acquirer/target) -- worth testing separately.
ROUTINE_ONLY_ADDENDUM = """

Additional guidance: many 8-K filings are administrative/scheduling in nature (announcing an
UPCOMING earnings call date, a routine recurring dividend at a similar rate to prior quarters,
standard governance items) rather than disclosing a new, resolved, material event. These should
score LOW on both magnitude and catalyst regardless of the underlying company's prospects — score
the INFORMATION CONTENT of this specific filing, not general optimism about the company."""

PROMPTS = {
    "current": CURRENT_PROMPT,
    "full_addendum": CURRENT_PROMPT + FULL_ADDENDUM,
    "routine_only": CURRENT_PROMPT + ROUTINE_ONLY_ADDENDUM,
}


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


def search_material_filings(start: str, end: str, max_results: int) -> list:
    """SEC full-text search across all MATERIAL_ITEMS codes, deduped to one filing per CIK
    (most recent in range) so the sample spans many different companies/situations."""
    seen_ciks = set()
    results = []
    queries = ["%22Item+2.02%22", "%22Item+1.01%22", "%22Item+8.01%22", "%22Item+7.01%22"]
    for q in queries:
        if len(results) >= max_results:
            break
        r = requests.get(
            f"https://efts.sec.gov/LATEST/search-index?q={q}&forms=8-K&startdt={start}&enddt={end}",
            headers={"User-Agent": SEC_UA}, timeout=20)
        if r.status_code != 200:
            continue
        hits = r.json().get("hits", {}).get("hits", [])
        for h in hits:
            s = h["_source"]
            cik = s["ciks"][0].lstrip("0") if s.get("ciks") else None
            if not cik or cik in seen_ciks:
                continue
            items = set(s.get("items", []))
            if not (items & MATERIAL_ITEMS):
                continue
            m_ticker = re.search(r"\(([A-Z]{1,6})(?:,|\))", s.get("display_names", [""])[0])
            ticker = m_ticker.group(1) if m_ticker else None
            if not ticker:
                continue
            seen_ciks.add(cik)
            results.append({
                "ticker": ticker, "cik": int(cik), "accession": s["adsh"].replace("-", ""),
                "company": s.get("display_names", [""])[0].split("  (")[0],
                "date": s.get("file_date"), "items": ",".join(sorted(items)),
            })
            if len(results) >= max_results:
                break
        time.sleep(0.3)
    return results


async def fetch_body(cik: int, accession: str) -> str:
    async with aiohttp.ClientSession() as session:
        try:
            return await sec_edgar.fetch_8k_content(session, cik, accession, SEC_UA)
        except Exception as e:
            print(f"    fetch failed: {e}")
            return ""


# Live news_call geometry (core/config.py), NOT news_call_sweep_unified.py's older 0.50/DTE17
# sweep constants -- this is meant to answer "would TODAY's actual strategy have profited",
# so it needs to match what's actually deployed right now.
NC_DELTA = _cfg.NEWS_CALL_TARGET_DELTA        # 0.40
NC_DTE   = round((_cfg.NEWS_CALL_DTE_MIN + _cfg.NEWS_CALL_DTE_MAX) / 2)  # ~10
EXIT_RULE = "tiered_trail"
EXIT_PARAMS = {**_bt.EXIT_PARAMS, "tiers": list(_cfg.EXIT_TIERS)}


def _passes_live_gate(sentiment, magnitude, confidence) -> bool:
    """The exact chain process_signal + execute_news_call apply live: general MIN_MAGNITUDE +
    dynamic confidence floor, THEN news_call's own stricter per-strategy magnitude bar."""
    if sentiment != "bullish":
        return False
    if magnitude < _cfg.MIN_MAGNITUDE:
        return False
    if confidence < _cfg.BASE_CONFIDENCE + (1 - magnitude) * _cfg.CONFIDENCE_SLOPE:
        return False
    if magnitude < _cfg.NEWS_CALL_MIN_MAGNITUDE:
        return False
    return True


def _scale(mag, conf):
    return max(_cfg.MAX_POSITION_USD * 0.10, _cfg.MAX_POSITION_USD * mag * conf)


def simulate_trade(ticker: str, date: str, sig: dict) -> "dict | None":
    """None if the gate wouldn't have fired live (no trade, not a $0 trade) or if price data
    is unavailable; otherwise the real simulate_option_pnl trade dict (has pnl_usd, pnl_pct)."""
    magnitude, confidence = float(sig.get("magnitude") or 0), float(sig.get("confidence") or 0)
    if not _passes_live_gate(sig.get("sentiment"), magnitude, confidence):
        return None
    entry_dt = datetime.fromisoformat(f"{date}T15:00:00+00:00")   # mid-morning ET, approximate
    sp = _bt.get_price_at(ticker, entry_dt)
    if not sp or not _bt.is_valid_stock_ticker(ticker):
        return None
    save = {k: getattr(_bt, k) for k in ("TARGET_DELTA", "DTE_TARGET", "MAX_HOLD_DAYS", "TRAILING_STOP_PCT", "EXIT_PARAMS")}
    _bt.TARGET_DELTA, _bt.DTE_TARGET, _bt.MAX_HOLD_DAYS = NC_DELTA, NC_DTE, NC_DTE
    _bt.TRAILING_STOP_PCT, _bt.EXIT_PARAMS = _cfg.TRAILING_STOP_PCT, EXIT_PARAMS
    try:
        return _bt.simulate_option_pnl(
            ticker, entry_dt, sp, _scale(magnitude, confidence),
            {"magnitude": magnitude, "confidence": confidence},
            option_type="call", exit_rule=EXIT_RULE, spread_mult=1.0)
    finally:
        for k, v in save.items():
            setattr(_bt, k, v)


# Same bull/bear windows already used elsewhere in this codebase (news_call_sweep_unified.py),
# for consistency with other backtests -- a "choppy 2026" window isn't representative on its own.
#
# End offset must clear simulate_option_pnl's forward-data requirement: MAX_HOLD_DAYS=NC_DTE=10 ->
# window_cal_days = int(10*1.5)+7 = 22 calendar days of bars AFTER entry_dt. A filing dated within
# 22 days of "now" has no future bars yet and would spuriously return "no trade" (confirmed via a
# direct get_stock_bars check returning 0 bars for a same-day entry_dt) -- indistinguishable from a
# real gate miss in the output. Padded to 35 days for weekend/holiday/data-lag margin.
_now = datetime.now(timezone.utc)
REGIMES = [
    ("bull-2026", (_now - timedelta(days=180)).date(), (_now - timedelta(days=35)).date(), 150),
    ("bear-2022", (datetime(2022, 4, 1, tzinfo=timezone.utc)).date(), (datetime(2022, 6, 30, tzinfo=timezone.utc)).date(), 150),
]


async def main():
    all_rows = []
    for label, start, end, n_target in REGIMES:
        print(f"\n{'='*20} {label}: searching material 8-Ks from {start} to {end} (target {n_target}) {'='*20}")
        filings = search_material_filings(start.isoformat(), end.isoformat(), n_target)
        print(f"found {len(filings)} unique companies\n")

        for i, f in enumerate(filings):
            print(f"[{label} {i+1}/{len(filings)}] {f['ticker']:>6} ({f['company'][:35]:35s}) filed {f['date']} items={f['items']}")
            body = await fetch_body(f["cik"], f["accession"])
            if not body.strip():
                print("  no content, skipping")
                continue
            headline = f"8-K - {f['company']} ({f['cik']:010d}) (Filer) [{f['ticker']}]"

            scores = {}
            for name, prompt in PROMPTS.items():
                scores[name] = score(prompt, headline, body)

            row = {"regime": label, "ticker": f["ticker"], "date": f["date"], "items": f["items"]}
            for name, s in scores.items():
                row[f"{name}_mag"] = s.get("magnitude")
                row[f"{name}_cat"] = s.get("catalyst")
                row[f"{name}_sent"] = s.get("sentiment")
                row[f"{name}_conf"] = s.get("confidence")
                trade = simulate_trade(f["ticker"], f["date"], s)
                row[f"{name}_traded"] = trade is not None
                row[f"{name}_pnl_usd"] = trade["pnl_usd"] if trade else ""
                row[f"{name}_pnl_pct"] = trade["pnl_pct"] if trade else ""
            all_rows.append(row)
            print("  " + "  ".join(
                f"{n}: mag={row[n+'_mag']} sent={row[n+'_sent']}"
                + (f" pnl=${row[n+'_pnl_usd']:.0f}" if row[n+"_traded"] else " no-trade")
                for n in PROMPTS))
            time.sleep(0.2)

            # Flush progress every 10 rows so a long run's partial results are inspectable / safe
            # against interruption without losing everything.
            if len(all_rows) % 10 == 0:
                out_path = os.path.join(os.path.dirname(__file__), "..", "data", "sec_backtest.csv")
                with open(out_path, "w", newline="") as fh:
                    w = csv.DictWriter(fh, fieldnames=list(all_rows[0].keys()))
                    w.writeheader()
                    w.writerows(all_rows)

    out_path = os.path.join(os.path.dirname(__file__), "..", "data", "sec_backtest.csv")
    if all_rows:
        with open(out_path, "w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=list(all_rows[0].keys()))
            w.writeheader()
            w.writerows(all_rows)
    print(f"\n{len(all_rows)} total rows written to {out_path}")


if __name__ == "__main__":
    asyncio.run(main())
