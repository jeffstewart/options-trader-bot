"""
overnight_8k_backtest.py — do overnight SEC 8-K filings still have tradeable juice AT THE OPEN?

Context (2026-07-23): core/bot.py's market_is_open() check in process_signal() silently dropped
every SEC filing that arrived outside market hours, and sec_rss_poller() had already marked the
eid seen — so pre-market earnings 8-Ks (ADTN +9.47% on 2026-07-22, SMCI +5.42%) never reached
scoring at all. Before wiring a queue-and-replay fix into the live bot, this study answers the
go/no-go question historically: for overnight material 8-Ks, how much of the day's move is the
gap (prior close -> open, NOT capturable by an RTH-only bot) vs the drift (open -> close, which
IS)? If drift is ~zero conditional on the gap, the replay path should stay shadow-only.

Pipeline (all data sources free):
  1. EDGAR daily form.idx (1 request per trading day) -> every 8-K filed in the range.
  2. CIK -> ticker filter via SEC company_tickers.json (reuses v2/sec_edgar.py's loader, with
     its first-entry-wins fix), then ONE data.sec.gov/submissions request per unique CIK --
     that JSON carries acceptanceDateTime + item codes for all recent filings, so we never
     fetch per-filing index pages just to timestamp them.
  3. Keep filings with MATERIAL_ITEMS overlap accepted OUTSIDE 9:30-16:00 ET; assign each to
     its next regular session (pre-market -> same day, post-close/weekend -> next trading day).
  4. Fetch real filing bodies with v2/sec_edgar.fetch_8k_content -- same 900+900-char primary+
     exhibit truncation the live bot sees, so any later scoring pass matches production inputs.
  5. Alpaca daily + minute bars -> per event: r_gap, r_drift (open->close), r_open30,
     r_30_to_close. SIP feed with IEX fallback (free tier allows historical SIP; opens matter
     here and IEX opens can be off for thin names).
  6. Optional --score: run the REAL v2 Ollama scorer (unified_v1 prompt) over the fetched
     bodies so the analysis can condition on "would my bot have gone long on this".

SEC fair-access: all EDGAR requests go through one semaphore at ~5 req/s equivalent, half the
10 req/s limit. Everything SEC-side is cached on disk (research/overnight8k_cache/) so reruns
and range extensions only fetch what's new. Alpaca calls are batched (multi-symbol) and cheap.

Usage:
  .venv/bin/python research/overnight_8k_backtest.py                    # last ~30 days pilot
  .venv/bin/python research/overnight_8k_backtest.py --start 2026-06-20 --end 2026-07-21
  .venv/bin/python research/overnight_8k_backtest.py --score            # adds Ollama scores
"""
import argparse
import asyncio
import csv
import json
import os
import sys
import time
from collections import Counter
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from zoneinfo import ZoneInfo

import aiohttp
from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parent.parent
load_dotenv(ROOT / ".env")            # worktrees don't carry .env; the canonical one does
load_dotenv("/Users/jeff/Claude/Trader/.env")

sys.path.insert(0, str(ROOT / "v2"))
from sec_edgar import MATERIAL_ITEMS, fetch_8k_content, load_cik_ticker_map  # noqa: E402

from alpaca.data.historical import StockHistoricalDataClient  # noqa: E402
from alpaca.data.enums import Adjustment                      # noqa: E402
from alpaca.data.requests import StockBarsRequest             # noqa: E402
from alpaca.data.timeframe import TimeFrame                   # noqa: E402

ET = ZoneInfo("America/New_York")
USER_AGENT = os.environ.get("SEC_USER_AGENT", "Jeff Stewart research jeffstewart.ca@gmail.com")
CACHE = ROOT / "research" / "overnight8k_cache"
CACHE.mkdir(exist_ok=True)

# One shared throttle for ALL EDGAR hosts (www.sec.gov + data.sec.gov): 5 concurrent with a
# small delay ~= 5 req/s, half of SEC's 10 req/s fair-access limit.
_sec_sem = asyncio.Semaphore(5)


async def _sec_get(session: aiohttp.ClientSession, url: str) -> str:
    async with _sec_sem:
        async with session.get(url, headers={"User-Agent": USER_AGENT},
                                timeout=aiohttp.ClientTimeout(total=20)) as r:
            r.raise_for_status()
            text = await r.text()
        await asyncio.sleep(0.2)
    return text


# ── Stage 1: enumerate 8-Ks from daily form indexes ─────────────────────────────────────────
def _qtr(d: date) -> str:
    return f"QTR{(d.month - 1) // 3 + 1}"


async def daily_8ks(session, d: date) -> list[dict]:
    """Parse form.idx for one day -> [{cik, accession, filer}] for exact form '8-K' (amendments
    excluded -- an 8-K/A is stale news by definition). Cached; 404 (holiday) caches empty."""
    cache_f = CACHE / f"idx_{d.isoformat()}.json"
    if cache_f.exists():
        return json.loads(cache_f.read_text())
    url = f"https://www.sec.gov/Archives/edgar/daily-index/{d.year}/{_qtr(d)}/form.{d.strftime('%Y%m%d')}.idx"
    try:
        text = await _sec_get(session, url)
    except aiohttp.ClientResponseError as e:
        # EDGAR answers 403 (not 404) for daily-index files that don't exist, i.e. weekends
        # and market holidays.
        if e.status in (403, 404):
            cache_f.write_text("[]")
            return []
        raise
    rows = []
    for line in text.splitlines():
        # form.idx is fixed-width but column offsets drift across years; split-on-whitespace
        # from the right is stable: last two fields are date + file path, third-from-last is CIK.
        if not line.startswith("8-K "):
            continue
        parts = line.split()
        path = parts[-1]                      # edgar/data/{cik}/{accession}.txt
        cik = int(parts[-3])
        accession = path.rsplit("/", 1)[-1].removesuffix(".txt")
        rows.append({"cik": cik, "accession": accession})
    cache_f.write_text(json.dumps(rows))
    return rows


# ── Stage 2: acceptance time + item codes via per-CIK submissions JSON ──────────────────────
async def cik_submissions(session, cik: int) -> dict:
    """accession -> {accepted, items} from data.sec.gov. 'recent' covers >=1yr, plenty here."""
    cache_f = CACHE / f"sub_{cik}.json"
    if cache_f.exists():
        return json.loads(cache_f.read_text())
    url = f"https://data.sec.gov/submissions/CIK{cik:010d}.json"
    try:
        data = json.loads(await _sec_get(session, url))
        recent = data.get("filings", {}).get("recent", {})
        out = {}
        for acc, accepted, items, form in zip(recent.get("accessionNumber", []),
                                              recent.get("acceptanceDateTime", []),
                                              recent.get("items", []),
                                              recent.get("form", [])):
            if form == "8-K":
                out[acc] = {"accepted": accepted, "items": items}
    except Exception:
        out = {}
    cache_f.write_text(json.dumps(out))
    return out


def classify_session(accepted_utc: datetime, trading_days: list[date]) -> tuple[str, date] | None:
    """-> (category, trade_day). 'intraday' = accepted during RTH with >=30 min left before the
    close (matching v2's late-day entry cutoff) -- the live bot's actual lane, entry at
    acceptance + latency. Everything else trades the NEXT session's open. None = untradeable
    (after the 15:30 cutoff but before the close, or no next session in range)."""
    t = accepted_utc.astimezone(ET)
    d = t.date()
    is_tday = d in trading_days
    if is_tday and t.time() < datetime.strptime("09:30", "%H:%M").time():
        return "premarket", d
    if is_tday and t.time() < datetime.strptime("15:30", "%H:%M").time():
        return "intraday", d
    if is_tday and t.time() < datetime.strptime("16:00", "%H:%M").time():
        return None                                          # inside the entry cutoff
    nxt = next((td for td in trading_days if td > d), None)  # post-close, weekend, holiday
    if nxt is None:
        return None
    return ("postclose" if is_tday else "closed_day"), nxt


# ── Stage 4/5: Alpaca bars ──────────────────────────────────────────────────────────────────
def _chunks(seq, n):
    for i in range(0, len(seq), n):
        yield seq[i:i + n]


def _clamp_recent(dt: datetime) -> datetime:
    """Free tier allows historical SIP but rejects the most recent 15 min; clamping request
    ends keeps us on SIP (IEX opens are unreliable for exactly the thin names we study)."""
    return min(dt, datetime.now(timezone.utc) - timedelta(minutes=16))


def _alp(sym: str) -> str:
    """SEC writes class shares 'BF-A'; Alpaca rejects the dash and wants 'BF.A'."""
    return sym.replace("-", ".")


def fetch_daily_bars(client, symbols, start: date, end: date, feed: str) -> dict:
    """{symbol: {date: (open, close)}} via multi-symbol requests, 100 tickers per call.
    Keys are the SEC-style symbols the caller passed in."""
    out = {}
    for chunk in _chunks(sorted(symbols), 100):
        back = {_alp(s): s for s in chunk}
        # adjustment=ALL: without it a reverse split reads as a massive overnight "gap"
        # (caught on the first full run: mean gap in the >+5% bucket was +72%).
        resp = client.get_stock_bars(StockBarsRequest(
            symbol_or_symbols=list(back), timeframe=TimeFrame.Day,
            start=datetime.combine(start, datetime.min.time(), tzinfo=timezone.utc),
            end=_clamp_recent(datetime.combine(end + timedelta(days=1), datetime.min.time(),
                                               tzinfo=timezone.utc)),
            adjustment=Adjustment.ALL, feed=feed))
        for sym, bars in resp.data.items():
            out[back.get(sym, sym)] = {b.timestamp.astimezone(ET).date(): (b.open, b.close)
                                       for b in bars}
    return out


def fetch_minute_marks(client, day_events: dict, feed: str) -> dict:
    """{(symbol, day): (open_930, px_at_10am_ET)} -- the 9:30 bar's open and the last 1-min bar
    close at/before 10:00 ET. Both legs of the first-30-min return come from the SAME (raw)
    minute series: mixing adjusted daily opens with raw minute prices corrupts split names."""
    out = {}
    for day, syms in sorted(day_events.items()):
        start = datetime.combine(day, datetime.strptime("09:30", "%H:%M").time(), tzinfo=ET)
        end = datetime.combine(day, datetime.strptime("10:01", "%H:%M").time(), tzinfo=ET)
        for chunk in _chunks(sorted(syms), 100):
            back = {_alp(s): s for s in chunk}
            try:
                resp = client.get_stock_bars(StockBarsRequest(
                    symbol_or_symbols=list(back), timeframe=TimeFrame.Minute,
                    start=start, end=_clamp_recent(end), feed=feed))
            except Exception as e:
                print(f"  minute bars failed for {day}: {e}")
                continue
            for alp_sym, bars in resp.data.items():
                sym = back.get(alp_sym, alp_sym)
                marks = [b for b in bars if b.timestamp.astimezone(ET).time()
                         <= datetime.strptime("10:00", "%H:%M").time()]
                if marks:
                    out[(sym, day)] = (marks[0].open, marks[-1].close)
    return out


ENTRY_LAG_MIN = 2   # RSS poll + body fetch + scoring + order placement latency for RTH filings


def fetch_intraday_marks(client, events: list[dict], feed: str) -> None:
    """For 'intraday' events: entry = open of the first minute bar >= acceptance + ENTRY_LAG_MIN,
    plus the last bar close <= entry+30m and the last bar close of the day (same raw series, so
    every intraday ratio is split-consistent). Sets r_pre_entry / r_entry_close / r_entry_30."""
    for e in events:
        e.update({"r_pre_entry": None, "r_entry_close": None, "r_entry_30": None})
    todo = [e for e in events if e["category"] == "intraday"]
    by_day: dict[date, list] = {}
    for e in todo:
        by_day.setdefault(date.fromisoformat(e["trade_day"]), []).append(e)
    for day, evs in sorted(by_day.items()):
        start = datetime.combine(day, datetime.strptime("09:30", "%H:%M").time(), tzinfo=ET)
        end = datetime.combine(day, datetime.strptime("16:01", "%H:%M").time(), tzinfo=ET)
        for chunk in _chunks(sorted({e["ticker"] for e in evs}), 100):
            back = {_alp(s): s for s in chunk}
            try:
                resp = client.get_stock_bars(StockBarsRequest(
                    symbol_or_symbols=list(back), timeframe=TimeFrame.Minute,
                    start=start, end=_clamp_recent(end), feed=feed))
            except Exception as ex:
                print(f"  intraday minute bars failed for {day}: {ex}")
                continue
            bars_by_sym = {back.get(s, s): b for s, b in resp.data.items()}
            for e in evs:
                bars = bars_by_sym.get(e["ticker"])
                if not bars:
                    continue
                entry_t = (datetime.fromisoformat(e["accepted"])
                           + timedelta(minutes=ENTRY_LAG_MIN)).astimezone(ET)
                entry = next((b for b in bars if b.timestamp.astimezone(ET) >= entry_t), None)
                if entry is None or not entry.open:
                    continue
                upto30 = [b for b in bars if b.timestamp.astimezone(ET)
                          <= entry_t + timedelta(minutes=30)]
                e["r_entry_close"] = bars[-1].close / entry.open - 1
                if upto30:
                    e["r_entry_30"] = upto30[-1].close / entry.open - 1
                # how much had already moved by entry, off the same day's session open
                e["r_pre_entry"] = entry.open / bars[0].open - 1


# ── Optional: score bodies with the real v2 scorer ──────────────────────────────────────────
def score_events(events: list[dict], limit: int) -> None:
    import scoring  # v2/scoring.py -- imports v2/config, needs .env loaded (done above)
    todo = [e for e in events if e["body"]][:limit]
    print(f"Scoring {len(todo)} bodies with Ollama ({len(events) - len(todo)} skipped/empty)...")
    for i, e in enumerate(todo, 1):
        cache_f = CACHE / f"score_{e['accession']}.json"
        if cache_f.exists():
            s = json.loads(cache_f.read_text())
        else:
            headline = f"8-K filing ({e['items']}) — {e['ticker']}"
            s = scoring.score_article(headline, e["body"], source="SEC EDGAR 8-K") or {}
            if s:                       # don't cache failures (Ollama down/JSON parse miss)
                cache_f.write_text(json.dumps(s))
        e.update({"sentiment": s.get("sentiment", ""), "confidence": s.get("confidence", ""),
                  "magnitude": s.get("magnitude", ""), "catalyst": s.get("catalyst", "")})
        if i % 50 == 0:
            print(f"  {i}/{len(todo)}")


# ── Analysis ────────────────────────────────────────────────────────────────────────────────
def pct(x):
    return f"{100 * x:+.2f}%"


def summarize_intraday(events: list[dict]) -> None:
    """RTH-filed events, the live bot's lane: entry at acceptance + ENTRY_LAG_MIN."""
    priced = [e for e in events if e["category"] == "intraday"
              and e.get("r_entry_close") is not None]
    print(f"\n{'=' * 78}\nINTRADAY 8-K — entry {ENTRY_LAG_MIN} min after acceptance — "
          f"{len(priced)} priced events")

    def stats(rows, label):
        if not rows:
            print(f"{label:<30} n=0")
            return
        rets = [e["r_entry_close"] for e in rows]
        r30 = [e["r_entry_30"] for e in rows if e.get("r_entry_30") is not None]
        print(f"{label:<30} n={len(rows):<5} entry→close μ={pct(sum(rets) / len(rets)):>8} "
              f"med={pct(sorted(rets)[len(rets) // 2]):>8} "
              f"P(>+3%)={100 * sum(r > .03 for r in rets) / len(rets):.1f}% "
              f"P(>+5%)={100 * sum(r > .05 for r in rets) / len(rets):.1f}% "
              f"P(<-5%)={100 * sum(r < -.05 for r in rets) / len(rets):.1f}%"
              + (f"  [entry→30m μ={pct(sum(r30) / len(r30))}]" if r30 else ""))

    stats(priced, "ALL intraday")
    stats([e for e in priced if "2.02" in e["items"]], "Item 2.02 (earnings)")
    scored = [e for e in priced if e.get("sentiment")]
    if scored:
        stats([e for e in scored if e["sentiment"] == "bullish"], "bullish (any)")
        stats([e for e in scored if e["sentiment"] == "bullish"
               and (e.get("magnitude") or 0) >= 0.6 and (e.get("catalyst") or 0) >= 0.6],
              "bullish mag>=.6 cat>=.6")


def summarize(events: list[dict]) -> None:
    priced = [e for e in events if e.get("r_gap") is not None
              and e["category"] != "intraday"]
    print(f"\n{'=' * 78}\nOVERNIGHT 8-K GAP vs DRIFT — {len(priced)} priced events "
          f"(of {len(events)} qualifying filings)")
    print(Counter(e["category"] for e in priced))
    # Sanity check on the UTC->ET conversion: 8-K acceptances should cluster pre-open (6-9am)
    # and after the close (4-6pm) ET. If this histogram peaks mid-day, the tz handling is wrong.
    hours = Counter(datetime.fromisoformat(e["accepted"]).astimezone(ET).hour for e in events)
    print("acceptance-hour histogram (ET):",
          " ".join(f"{h:02d}:{hours.get(h, 0)}" for h in range(24)))

    def stats(rows, label):
        if not rows:
            print(f"{label:<28} n=0")
            return
        gaps = [e["r_gap"] for e in rows]
        drifts = [e["r_drift"] for e in rows]
        agree = [e for e in rows if e["r_gap"] * e["r_drift"] > 0]
        line = (f"{label:<28} n={len(rows):<5} gap μ={pct(sum(gaps) / len(gaps)):>8} "
                f"drift μ={pct(sum(drifts) / len(drifts)):>8} "
                f"drift med={pct(sorted(drifts)[len(drifts) // 2]):>8} "
                f"continuation={100 * len(agree) / len(rows):.0f}%")
        m30 = [e for e in rows if e.get("r_open30") is not None]
        if m30:
            line += (f"  [first30 μ={pct(sum(e['r_open30'] for e in m30) / len(m30))} "
                     f"rest μ={pct(sum(e['r_30_to_close'] for e in m30) / len(m30))}]")
        print(line)

    stats(priced, "ALL")
    stats([e for e in priced if "2.02" in e["items"]], "Item 2.02 (earnings)")
    print("\nBy gap bucket (the tradeable question: given the open you SEE, what follows?)")
    for lo, hi, label in [(-99, -.05, "gap < -5%"), (-.05, -.02, "-5% .. -2%"),
                          (-.02, .02, "-2% .. +2%"), (.02, .05, "+2% .. +5%"),
                          (.05, 99, "gap > +5%")]:
        stats([e for e in priced if lo <= e["r_gap"] < hi], f"  {label}")
    scored = [e for e in priced if e.get("sentiment")]
    if scored:
        print("\nScored subset — the bot's hypothetical longs:")
        stats([e for e in scored if e["sentiment"] == "bullish"], "  bullish (any)")
        stats([e for e in scored if e["sentiment"] == "bullish"
               and (e.get("magnitude") or 0) >= 0.6 and (e.get("catalyst") or 0) >= 0.6],
              "  bullish mag>=.6 cat>=.6")


# ── Main ────────────────────────────────────────────────────────────────────────────────────
async def build_events(start: date, end: date, trading_days: list[date], limit: int) -> list[dict]:
    async with aiohttp.ClientSession() as session:
        cik_map = await load_cik_ticker_map(session, USER_AGENT)

        idx_days = [start + timedelta(days=i) for i in range((end - start).days + 1)]
        raw = []
        for d, rows in zip(idx_days, await asyncio.gather(*(daily_8ks(session, d) for d in idx_days))):
            raw.extend(rows)
        # Joint filings show one form.idx row per co-filer; keep one row per accession.
        seen_acc = set()
        with_ticker = [r for r in raw if r["cik"] in cik_map
                       and not (r["accession"] in seen_acc or seen_acc.add(r["accession"]))]
        print(f"8-Ks filed {start}..{end}: {len(raw)}; with a resolvable ticker: {len(with_ticker)}")

        ciks = sorted({r["cik"] for r in with_ticker})
        subs: dict[int, dict] = {}
        for i, chunk in enumerate(_chunks(ciks, 200)):
            got = await asyncio.gather(*(cik_submissions(session, c) for c in chunk))
            subs.update(dict(zip(chunk, got)))
            print(f"  submissions metadata: {min((i + 1) * 200, len(ciks))}/{len(ciks)} CIKs")

        events = []
        for r in with_ticker:
            acc_dashed = r["accession"]
            meta = subs.get(r["cik"], {}).get(acc_dashed)
            if not meta or not meta["accepted"]:
                continue
            items = set((meta["items"] or "").split(","))
            if not (items & MATERIAL_ITEMS):
                continue
            accepted = datetime.fromisoformat(meta["accepted"].replace("Z", "+00:00"))
            cls = classify_session(accepted, trading_days)
            if cls is None:
                continue
            category, trade_day = cls
            events.append({
                "ticker": cik_map[r["cik"]], "cik": r["cik"],
                "accession": acc_dashed.replace("-", ""),
                "items": meta["items"], "accepted": accepted.isoformat(),
                "category": category, "trade_day": trade_day.isoformat(),
            })
        print(f"Material-item filings accepted outside RTH: {len(events)}")
        if limit:
            events = events[:limit]

        # fetch_8k_content's own semaphore (4 concurrent, no pacing) is tuned for the live
        # bot's trickle of filings; gathering 100 at once here saturated it at >10 req/s and
        # SEC's fair-access limiter 403'd ~90% of the fetches (silently -- the module returns
        # '' on failure by design). Throttle to 3 in-flight requests + 2 filings at a time,
        # and treat a cached EMPTY body as a failed fetch to retry, not a result.
        # Even ~6-8 req/s sustained for minutes trips EDGAR's 10-minute block (second full run:
        # 94% fill for the first ~600 filings, then 0 new bodies for the rest). One filing at a
        # time + 0.25s pacing ~= 4 req/s. Reruns retry only the still-empty ones, so chaining
        # passes (with the block expiring in between) converges on full coverage.
        import sec_edgar as _se
        _se._sem = asyncio.Semaphore(2)

        async def body_for(e):
            cache_f = CACHE / f"body_{e['accession']}.txt"
            if cache_f.exists() and cache_f.stat().st_size > 0:
                e["body"] = cache_f.read_text()
                return
            e["body"] = await fetch_8k_content(session, e["cik"], e["accession"], USER_AGENT)
            if e["body"]:
                cache_f.write_text(e["body"])
            await asyncio.sleep(0.25)

        for i, e in enumerate(events, 1):
            await body_for(e)
            if i % 100 == 0:
                print(f"  bodies: {i}/{len(events)} "
                      f"({sum(1 for x in events[:i] if x.get('body'))} non-empty)")
    return events


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--start", type=date.fromisoformat,
                    default=date.today() - timedelta(days=31))
    ap.add_argument("--end", type=date.fromisoformat, default=date.today() - timedelta(days=1))
    ap.add_argument("--score", action="store_true", help="run the v2 Ollama scorer over bodies")
    ap.add_argument("--score-limit", type=int, default=10 ** 9)
    ap.add_argument("--limit", type=int, default=0, help="cap events (debug)")
    ap.add_argument("--out", default=str(ROOT / "research" / "overnight8k_events.csv"))
    args = ap.parse_args()

    client = StockHistoricalDataClient(os.environ["ALPACA_API_KEY"],
                                       os.environ["ALPACA_SECRET_KEY"])
    # Trading calendar from SPY daily bars; padded a week each side so every filing in range
    # has a prior close and a next session.
    cal_start, cal_end = args.start - timedelta(days=7), args.end + timedelta(days=7)
    feed = "sip"
    try:
        spy = fetch_daily_bars(client, ["SPY"], cal_start, cal_end, feed)
    except Exception as e:
        print(f"SIP feed unavailable ({e}); falling back to IEX")
        feed = "iex"
        spy = fetch_daily_bars(client, ["SPY"], cal_start, cal_end, feed)
    trading_days = sorted(spy["SPY"].keys())
    print(f"Feed: {feed}; {len(trading_days)} trading days in padded range")

    events = asyncio.run(build_events(args.start, args.end, trading_days, args.limit))

    symbols = {e["ticker"] for e in events}
    daily = fetch_daily_bars(client, symbols, cal_start, cal_end, feed)
    day_events: dict[date, set] = {}
    for e in events:
        day_events.setdefault(date.fromisoformat(e["trade_day"]), set()).add(e["ticker"])
    marks = fetch_minute_marks(client, day_events, feed)

    for e in events:
        td = date.fromisoformat(e["trade_day"])
        sym_bars = daily.get(e["ticker"], {})
        prior = [d for d in trading_days if d < td and d in sym_bars]
        e.update({"r_gap": None, "r_drift": None, "r_open30": None, "r_30_to_close": None})
        if td not in sym_bars or not prior:
            continue
        prior_close = sym_bars[prior[-1]][1]
        o, c = sym_bars[td]
        if not prior_close or not o:
            continue
        e["r_gap"] = o / prior_close - 1
        e["r_drift"] = c / o - 1
        m = marks.get((e["ticker"], td))
        if m and m[0]:
            # first-30-min return from minute bars only; rest-of-day derived from it and the
            # daily open->close drift (both ratios are same-day, hence split-invariant).
            e["r_open30"] = m[1] / m[0] - 1
            e["r_30_to_close"] = (1 + e["r_drift"]) / (1 + e["r_open30"]) - 1

    fetch_intraday_marks(client, events, feed)

    if args.score:
        score_events(events, args.score_limit)

    cols = ["ticker", "cik", "accession", "items", "accepted", "category", "trade_day",
            "r_gap", "r_drift", "r_open30", "r_30_to_close",
            "r_pre_entry", "r_entry_close", "r_entry_30",
            "sentiment", "confidence", "magnitude", "catalyst", "body"]
    with open(args.out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols, extrasaction="ignore")
        w.writeheader()
        w.writerows(events)
    print(f"\nWrote {len(events)} events -> {args.out}")
    summarize(events)
    summarize_intraday(events)


if __name__ == "__main__":
    main()
