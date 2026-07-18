"""
sec_edgar.py — fixes the SEC EDGAR 8-K feed diagnosed 2026-07-18 as effectively non-functional
(research/news_source_attribution.py: 99.7% of signals never resolved to a ticker). v2's old
sec_rss_poller() was even leaner than v1's bug: it scored the RSS filer-index title with an EMPTY
body (no Item-code summary at all), so the model had nothing but a bare company legal name to work
with -- never a ticker symbol.

Two independent fixes, both needed (identical to core/sec_edgar.py -- kept as a separate copy here
since v2 is deliberately self-contained, no import from core/):
  1. Deterministic CIK -> ticker resolution via SEC's own company_tickers.json (cached locally,
     refreshed weekly) -- removes any reliance on the LLM guessing a ticker from a bare legal
     company name it may not recognize.
  2. Real filing content: fetch the actual primary 8-K document + a press-release exhibit
     (EX-99.x) when present, instead of nothing. Gated to a curated set of "material" Item codes
     so SEC request volume stays proportional to genuinely interesting filings.

All requests declare SEC's required User-Agent and go through a bounded semaphore -- SEC's fair-
access limit is 10 req/s; the extra per-filing fetches (index + doc[+exhibit]) are capped in
concurrency to stay well under that even during a filing burst.
"""
import asyncio
import json
import re
import time
from pathlib import Path

import aiohttp
from bs4 import BeautifulSoup

CIK_TICKER_URL   = "https://www.sec.gov/files/company_tickers.json"
CIK_TICKER_CACHE = "sec_cik_tickers.json"   # bare filename -- daemons run with CWD=data/ (config.py)
CIK_TICKER_TTL_S = 7 * 24 * 3600            # CIK<->ticker mappings change rarely; refresh weekly

# Item codes worth the extra fetch (real corporate actions). Deliberately excludes routine/
# administrative-only items (5.03 bylaw amendments, 5.07 vote results, 5.08 shareholder
# nominations, bare 9.01 with nothing else) so fetch volume tracks genuinely interesting filings.
MATERIAL_ITEMS = {
    "1.01", "1.02", "1.03",                  # material agreements / defaults / bankruptcy
    "2.01", "2.02", "2.03", "2.04", "2.05", "2.06",   # acquisitions, earnings, debt, impairment
    "3.01", "3.02", "3.03",                  # delisting, unregistered sales, security rights
    "4.01", "4.02",                          # auditor change, non-reliance on financials
    "5.01", "5.02",                          # change in control, exec departures/appointments
    "7.01", "8.01",                          # Reg FD (often carries a press-release exhibit), other events
}

MAX_PRIMARY_CHARS = 900
MAX_EXHIBIT_CHARS = 900
FETCH_TIMEOUT_S = 8
MAX_CONCURRENT_FETCHES = 4

_sem = asyncio.Semaphore(MAX_CONCURRENT_FETCHES)
_cik_ticker_map: dict[int, str] = {}
_cik_ticker_loaded_at = 0.0

_ITEM_RE = re.compile(r"Item\s+(\d+\.\d+)")
_CIK_ACC_RE = re.compile(r"/data/(\d+)/(\d+)/")


def item_codes_in_summary(summary: str) -> set:
    return set(_ITEM_RE.findall(summary or ""))


def extract_cik_accession(index_url: str):
    """From '.../data/{cik}/{accession_nodashes}/...' -> (cik:int, accession_nodashes:str)."""
    m = _CIK_ACC_RE.search(index_url or "")
    if not m:
        return None, None
    return int(m.group(1)), m.group(2)


async def load_cik_ticker_map(session: aiohttp.ClientSession, user_agent: str) -> dict:
    """In-memory + on-disk cache; only hits SEC's servers once a week."""
    global _cik_ticker_map, _cik_ticker_loaded_at
    if _cik_ticker_map and (time.time() - _cik_ticker_loaded_at) < CIK_TICKER_TTL_S:
        return _cik_ticker_map

    cache_path = Path(CIK_TICKER_CACHE)
    data = None
    if cache_path.exists() and (time.time() - cache_path.stat().st_mtime) < CIK_TICKER_TTL_S:
        try:
            data = json.loads(cache_path.read_text())
        except Exception:
            data = None
    if data is None:
        async with session.get(CIK_TICKER_URL, headers={"User-Agent": user_agent},
                                timeout=aiohttp.ClientTimeout(total=15)) as r:
            data = await r.json(content_type=None)
        try:
            cache_path.write_text(json.dumps(data))
        except Exception:
            pass

    _cik_ticker_map = {int(v["cik_str"]): v["ticker"] for v in data.values()}
    _cik_ticker_loaded_at = time.time()
    return _cik_ticker_map


def _strip_html(html: str, max_chars: int, skip_to_item: bool = False) -> str:
    """skip_to_item: every primary 8-K document opens with ~2KB of identical SEC cover-page
    boilerplate (registrant name/address/CIK header) before the actual Item narrative -- without
    skipping past it, a small max_chars budget captures none of the substance (verified 2026-07-18:
    a 900-char cutoff on the raw doc landed entirely inside the cover page). Exhibits (press
    releases) have no such header, so this only applies to the primary document."""
    soup = BeautifulSoup(html, "html.parser")
    text = re.sub(r"\s+", " ", soup.get_text(" ", strip=True))
    if skip_to_item:
        m = _ITEM_RE.search(text)
        if m:
            text = text[m.start():]
    return text[:max_chars]


async def _find_documents(session, cik: int, accession_nodashes: str, user_agent: str):
    """Parse the filing's -index.htm 'Document Format Files' table for (type, url) pairs."""
    acc_dashed = f"{accession_nodashes[:10]}-{accession_nodashes[10:12]}-{accession_nodashes[12:]}"
    index_url = f"https://www.sec.gov/Archives/edgar/data/{cik}/{accession_nodashes}/{acc_dashed}-index.htm"
    async with _sem:
        async with session.get(index_url, headers={"User-Agent": user_agent},
                                timeout=aiohttp.ClientTimeout(total=FETCH_TIMEOUT_S)) as r:
            html = await r.text()
    soup = BeautifulSoup(html, "html.parser")
    docs = []
    for row in soup.select("table.tableFile tr"):
        cells = row.find_all("td")
        if len(cells) < 4:
            continue
        a = cells[2].find("a")
        if not a or not a.get("href"):
            continue
        form_type = cells[3].get_text(strip=True)
        href = a["href"]
        if href.startswith("/ix?doc="):
            href = href[len("/ix?doc="):]
        if not href.startswith("http"):
            href = "https://www.sec.gov" + href
        docs.append({"type": form_type, "url": href})
    return docs


async def fetch_8k_content(session: aiohttp.ClientSession, cik: int, accession_nodashes: str,
                            user_agent: str) -> str:
    """Real filing text: the primary 8-K document + a press-release exhibit (EX-99.x) if one was
    attached. Returns '' on any failure -- caller falls back to whatever it had before."""
    try:
        docs = await _find_documents(session, cik, accession_nodashes, user_agent)
    except Exception:
        return ""
    primary = next((d for d in docs if d["type"].upper().startswith("8-K")), None)
    exhibit = next((d for d in docs if d["type"].upper().startswith("EX-99")), None)

    parts = []
    for doc, limit, skip_to_item in ((primary, MAX_PRIMARY_CHARS, True), (exhibit, MAX_EXHIBIT_CHARS, False)):
        if not doc:
            continue
        try:
            async with _sem:
                async with session.get(doc["url"], headers={"User-Agent": user_agent},
                                        timeout=aiohttp.ClientTimeout(total=FETCH_TIMEOUT_S)) as r:
                    html = await r.text()
            parts.append(_strip_html(html, limit, skip_to_item=skip_to_item))
        except Exception:
            continue
    return "\n\n".join(p for p in parts if p)
