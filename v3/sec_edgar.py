"""
sec_edgar.py (v3) — SEC EDGAR 8-K real-time filing ingestion and parser.
Features:
  - Deterministic CIK -> ticker resolution via SEC company_tickers.json (cached locally).
  - Material 8-K Item code filtering (M&A, earnings, clinical trials, executive changes, etc.).
  - Bounded concurrency and fair-access rate limiting (SEC requirement: <= 10 req/s).
  - Primary 8-K disclosure text and press-release exhibit (EX-99.x) extraction.
"""

import asyncio
import json
import logging
import re
import time
from pathlib import Path
from typing import Optional, Tuple

import aiohttp
from bs4 import BeautifulSoup

try:
    from v3 import config as cfg
except ImportError:
    import config as cfg

log = logging.getLogger(__name__)

CIK_TICKER_URL = "https://www.sec.gov/files/company_tickers.json"
CIK_TICKER_CACHE = Path(getattr(cfg, "DATA_DIR", "data")) / "sec_cik_tickers.json"
CIK_TICKER_TTL_S = 7 * 24 * 3600  # Refresh weekly

# Material corporate events worth fetching full text for:
MATERIAL_ITEMS = {
    "1.01", "1.02", "1.03",                  # material agreements / defaults / bankruptcy
    "2.01", "2.02", "2.03", "2.04", "2.05", "2.06",   # acquisitions, earnings, debt, impairment
    "3.01", "3.02", "3.03",                  # delisting, unregistered sales, security rights
    "4.01", "4.02",                          # auditor change, non-reliance on financials
    "5.01", "5.02",                          # change in control, exec departures/appointments
    "7.01", "8.01",                          # Reg FD, other events (FDA/clinical updates often land in 8.01)
}

MAX_PRIMARY_CHARS = 1000
MAX_EXHIBIT_CHARS = 1000
FETCH_TIMEOUT_S = 8
MAX_CONCURRENT_FETCHES = 4

_sem = asyncio.Semaphore(MAX_CONCURRENT_FETCHES)
_cik_ticker_map: dict[int, str] = {}
_cik_ticker_loaded_at = 0.0

_ITEM_RE = re.compile(r"Item\s+(\d+\.\d+)")
_CIK_ACC_RE = re.compile(r"/data/(\d+)/(\d+)/")


def item_codes_in_summary(summary: str) -> set:
    """Extract Item codes mentioned in RSS summary."""
    return set(_ITEM_RE.findall(summary or ""))


def extract_cik_accession(index_url: str) -> Tuple[Optional[int], Optional[str]]:
    """From '.../data/{cik}/{accession_nodashes}/...' -> (cik: int, accession_nodashes: str)."""
    m = _CIK_ACC_RE.search(index_url or "")
    if not m:
        return None, None
    try:
        return int(m.group(1)), m.group(2)
    except ValueError:
        return None, None


async def load_cik_ticker_map(session: aiohttp.ClientSession, user_agent: str) -> dict[int, str]:
    """In-memory + on-disk cache of CIK -> primary ticker mapping."""
    global _cik_ticker_map, _cik_ticker_loaded_at
    if _cik_ticker_map and (time.time() - _cik_ticker_loaded_at) < CIK_TICKER_TTL_S:
        return _cik_ticker_map

    cache_path = CIK_TICKER_CACHE
    data = None
    if cache_path.exists() and (time.time() - cache_path.stat().st_mtime) < CIK_TICKER_TTL_S:
        try:
            data = json.loads(cache_path.read_text())
        except Exception:
            data = None

    if data is None:
        try:
            cache_path.parent.mkdir(parents=True, exist_ok=True)
            async with session.get(CIK_TICKER_URL, headers={"User-Agent": user_agent},
                                   timeout=aiohttp.ClientTimeout(total=15)) as r:
                if r.status == 200:
                    data = await r.json(content_type=None)
                    try:
                        cache_path.write_text(json.dumps(data))
                    except Exception as e:
                        log.debug("Could not write CIK cache: %s", e)
        except Exception as e:
            log.warning("Could not fetch SEC company_tickers.json: %s", e)
            return _cik_ticker_map

    if data and isinstance(data, dict):
        _cik_ticker_map = {}
        for v in data.values():
            if isinstance(v, dict) and "cik_str" in v and "ticker" in v:
                # setdefault keeps the first (primary) common stock entry per CIK
                _cik_ticker_map.setdefault(int(v["cik_str"]), str(v["ticker"]).upper())
        _cik_ticker_loaded_at = time.time()

    return _cik_ticker_map


def _strip_html(html: str, max_chars: int, skip_to_item: bool = False) -> str:
    """Extract clean text and optionally skip cover page boilerplate."""
    soup = BeautifulSoup(html, "html.parser")
    text = re.sub(r"\s+", " ", soup.get_text(" ", strip=True))
    if skip_to_item:
        m = _ITEM_RE.search(text)
        if m:
            text = text[m.start():]
    return text[:max_chars]


async def _find_documents(session: aiohttp.ClientSession, cik: int, accession_nodashes: str, user_agent: str) -> list[dict]:
    """Parse the filing's -index.htm table for document URLs."""
    acc_dashed = f"{accession_nodashes[:10]}-{accession_nodashes[10:12]}-{accession_nodashes[12:]}"
    index_url = f"https://www.sec.gov/Archives/edgar/data/{cik}/{accession_nodashes}/{acc_dashed}-index.htm"
    async with _sem:
        async with session.get(index_url, headers={"User-Agent": user_agent},
                               timeout=aiohttp.ClientTimeout(total=FETCH_TIMEOUT_S)) as r:
            if r.status != 200:
                return []
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
    """Fetch primary 8-K document text and press-release exhibit (EX-99)."""
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
                    if r.status == 200:
                        html = await r.text()
                        parts.append(_strip_html(html, limit, skip_to_item=skip_to_item))
        except Exception:
            continue
    return "\n\n".join(p for p in parts if p)
