"""
missed_movers.py — RECALL & CALIBRATION AUDIT of the live scorer/prompt.

For every stock that appeared in our news feed (Jun 9–18), join what WE scored it to its actual
forward move (Yahoo, cached), then:
  (A) BUCKET the big winners we did NOT trade — prompt miss vs gate/cap miss — to see where the
      misses come from (prompt under-rating vs execution blocking).
  (B) CALIBRATION — does our score actually rank forward returns? (directional validity +
      magnitude rank-IC). This is the statistically-powered scorer/prompt metric the noisy
      top-35 lotto P&L lacked — use it to evaluate prompt/scorer changes.

Scored record:
  - router_decisions.csv  → dated BULLISH signals (mag/conf/is_earnings/regime/headline).
  - bot.log 🤖 lines      → the FULL record incl. NEUTRAL/BEARISH (time-only; dated by matching
    bullish lines to router_decisions + carry-forward within 🚀-delimited sessions).

Forward return = entry-day close → max close over the next FWD trading days (long-call proxy).
Yahoo returns cached in yahoo_ret_cache.json so re-runs are instant.

Usage:  USE_YAHOO_BARS=1 .venv/bin/python -u missed_movers.py
"""
import os, re, csv, json, math
os.environ.setdefault("USE_YAHOO_BARS", "1")
from datetime import datetime, timezone, timedelta
from collections import defaultdict
import yahoo_data
import config as cfg

HORIZON = int(os.environ.get("HORIZON", "1"))   # forward trading days (1 = the known news-signal horizon)
BIG = 5.0                     # % forward move that counts as a "winner" we'd have wanted
LIQ_MIN_PX = 5.0              # liquidity/optionability filter: drop sub-$5 names (penny pumps, no real options)
CACHE = "yahoo_movers_cache.json"   # per ticker-day: {px: entry close, r1: 1d ret%, r3: 3d-max ret%}
SKIP = {"BTC", "ETH", "QQQ", "SPY"}

BOT_RE = re.compile(r"(\d{2}:\d{2}:\d{2}).+🤖 \w[\w.]*: (\w+)\s+conf=([\d.]+)\s+mag=([\d.]+)\s+tickers=\[([^\]]*)\]")


# ── dated scored record ────────────────────────────────────────────────────────
def load_router():
    rows = []
    for r in csv.DictReader(open("router_decisions.csv")):
        tk = (r.get("ticker") or "").strip()
        if not tk or tk in SKIP:
            continue
        rows.append({"date": r["timestamp"][:10], "tk": tk,
                     "mag": round(float(r.get("magnitude") or 0), 2),
                     "conf": round(float(r.get("confidence") or 0), 2),
                     "earn": str(r.get("is_earnings")).lower() in ("true", "1"),
                     "up": str(r.get("regime_uptrend")).lower() in ("true", "1"),
                     "head": r.get("headline", "")})
    return rows


def parse_botlog_dated(router_rows):
    """Walk bot.log 🤖 lines; date each by matching bullish lines to router_decisions
    (ticker+mag+conf) and carrying the date forward within each 🚀 session."""
    anchor = {}                                  # (tk,mag,conf) -> date  (from router)
    for r in router_rows:
        anchor[(r["tk"], r["mag"], r["conf"])] = r["date"]
    out = []                                     # dicts: date, tk, sent, mag, conf
    cur_date = None
    for line in open("bot.log", errors="ignore"):
        if "🚀 News trading bot starting" in line:
            cur_date = None                      # new session — re-anchor
            continue
        m = BOT_RE.search(line)
        if not m:
            continue
        _t, sent, conf, mag, tks = m.groups()
        conf = round(float(conf), 2); mag = round(float(mag), 2)
        tickers = [t.strip().strip("'\"") for t in tks.split(",") if t.strip()]
        for tk in tickers:
            if tk in SKIP:
                continue
            d = anchor.get((tk, mag, conf))
            if d:                                # bullish line we can date exactly
                cur_date = d
            out.append({"date": cur_date, "tk": tk, "sent": sent, "mag": mag, "conf": conf})
    # backfill leading lines (before first anchor in a session) with the next known date
    nxt = None
    for r in reversed(out):
        if r["date"]:
            nxt = r["date"]
        elif nxt:
            r["date"] = nxt
    return [r for r in out if r["date"]]


def best_scored():
    """(tk,date) -> best read we had that day {sent,mag,conf,earn,head}. Prefer the
    highest |mag×conf| entry; enrich earnings/headline from router."""
    router = load_router()
    rmap = {}
    for r in router:
        rmap.setdefault((r["tk"], r["date"]), r)
    rec = {}
    for r in parse_botlog_dated(router):
        k = (r["tk"], r["date"])
        sc = r["mag"] * r["conf"] * (1 if r["sent"] == "bullish" else (-1 if r["sent"] == "bearish" else 0))
        prev = rec.get(k)
        if prev is None or abs(sc) > abs(prev["score"]):
            ro = rmap.get(k, {})
            rec[k] = {"sent": r["sent"], "mag": r["mag"], "conf": r["conf"], "score": sc,
                      "earn": ro.get("earn", False), "head": ro.get("head", "")}
    for r in router:                             # ensure router bullish ticker-days present
        k = (r["tk"], r["date"])
        if k not in rec:
            rec[k] = {"sent": "bullish", "mag": r["mag"], "conf": r["conf"],
                      "score": r["mag"] * r["conf"], "earn": r["earn"], "head": r["head"]}
    return rec


def traded_sets():
    real, shadow = set(), set()
    for r in csv.DictReader(open("trades.csv")):
        tk, ts = r.get("source_ticker"), r.get("timestamp", "")
        if tk and ts:
            real.add((tk, ts[:10]))
    if os.path.exists("shadow_trades.csv"):
        for r in csv.DictReader(open("shadow_trades.csv")):
            if r.get("action") == "open" and r.get("underlying") and r.get("timestamp"):
                shadow.add((r["underlying"], r["timestamp"][:10]))
    return real, shadow


# ── forward returns (cached, per-ticker for speed) ─────────────────────────────
def fwd_returns(keys, cache):
    """Fetch each unique ticker's bars ONCE, slice for all its dates. Caches entry price +
    1-day and 3-day-max forward returns per ticker-day so any horizon/liquidity cut is free."""
    by_tk = defaultdict(list)
    for tk, date in keys:
        c = cache.get(f"{tk}_{date}", "MISS")
        if c == "MISS" or (isinstance(c, dict) and "rday" not in c):   # refetch entries lacking event-day move
            by_tk[tk].append(date)
    todo = sum(len(v) for v in by_tk.values())
    print(f"  forward returns: {len(keys)-todo} cached, fetching {len(by_tk)} tickers ({todo} ticker-days) …", flush=True)
    for n, (tk, dates) in enumerate(by_tk.items()):
        try:
            ds = sorted(dates)
            lo = datetime.fromisoformat(ds[0]).replace(tzinfo=timezone.utc) - timedelta(days=5)
            hi = datetime.fromisoformat(ds[-1]).replace(tzinfo=timezone.utc) + timedelta(days=10)
            bars = sorted(yahoo_data.get_yahoo_bars(tk, lo, hi), key=lambda b: b["t"])
            for date in dates:
                d0 = datetime.fromisoformat(date).date()
                prior = [b for b in bars if b["t"].date() < d0]
                fwd = [b for b in bars if b["t"].date() >= d0]
                if len(fwd) >= 2 and fwd[0]["c"]:
                    px = fwd[0]["c"]                                   # news-day close
                    pc = prior[-1]["c"] if prior else None            # prior trading-day close
                    rec = {"px": round(px, 2),
                           "r1": round((fwd[1]["c"] / px - 1) * 100, 2),
                           "r3": round((max(b["c"] for b in fwd[1:4]) / px - 1) * 100, 2),
                           "rintra": round((px / fwd[0]["o"] - 1) * 100, 2) if fwd[0].get("o") else None,
                           "rday": round((px / pc - 1) * 100, 2) if pc else None}   # event-day move
                    cache[f"{tk}_{date}"] = rec
                else:
                    cache[f"{tk}_{date}"] = None
        except Exception:
            for date in dates:
                cache[f"{tk}_{date}"] = None
        if n % 50 == 49:
            json.dump(cache, open(CACHE, "w"))
    json.dump(cache, open(CACHE, "w"))
    return cache


# ── calibration (no scipy) ─────────────────────────────────────────────────────
def spearman(xs, ys):
    n = len(xs)
    if n < 5:
        return None
    def ranks(v):
        order = sorted(range(n), key=lambda i: v[i])
        rk = [0.0] * n
        i = 0
        while i < n:
            j = i
            while j + 1 < n and v[order[j + 1]] == v[order[i]]:
                j += 1
            avg = (i + j) / 2.0
            for k in range(i, j + 1):
                rk[order[k]] = avg
            i = j + 1
        return rk
    rx, ry = ranks(xs), ranks(ys)
    mx, my = sum(rx) / n, sum(ry) / n
    cov = sum((rx[i] - mx) * (ry[i] - my) for i in range(n))
    sx = math.sqrt(sum((r - mx) ** 2 for r in rx)); sy = math.sqrt(sum((r - my) ** 2 for r in ry))
    return cov / (sx * sy) if sx and sy else None


def main():
    rec = best_scored()
    real, shadow = traded_sets()
    cache = json.load(open(CACHE)) if os.path.exists(CACHE) else {}
    cache = fwd_returns(list(rec.keys()), cache)
    rk = {0: "rday", 1: "r1", 3: "r3"}.get(HORIZON, "r3")   # 0 = news-day move (prior close → news close)
    label = {"rday": "news-DAY move (prior-close→close)", "r1": "1-day fwd", "r3": "3-day-max fwd"}[rk]
    rows, illiq = [], 0
    for (tk, date), s in rec.items():
        c = cache.get(f"{tk}_{date}")
        if not c or c.get(rk) is None:
            continue
        if c["px"] < LIQ_MIN_PX:                 # liquidity/optionability filter
            illiq += 1
            continue
        rows.append(dict(s, tk=tk, date=date, ret=c[rk], px=c["px"],
                         traded=(tk, date) in real, shadowed=(tk, date) in shadow))
    print(f"\nMETRIC={label} · liquidity≥${LIQ_MIN_PX:.0f} (dropped {illiq} sub-${LIQ_MIN_PX:.0f} names)")
    print(f"{len(rows)} scored ticker-days "
          f"({sum(r['sent']=='bullish' for r in rows)} bull / "
          f"{sum(r['sent']=='neutral' for r in rows)} neut / "
          f"{sum(r['sent']=='bearish' for r in rows)} bear)\n")

    # ── (A) recall buckets on the big winners ──────────────────────────────────
    NC = cfg.NEWS_CALL_MIN_MAGNITUDE
    def bucket(r):
        if r["traded"]:   return "1. TRADED"
        if r["shadowed"]: return "2. shadow-only (no capital)"
        if r["sent"] == "bullish" and (r["mag"] >= NC or r["earn"]):
            return "3. GATE/CAP miss (scored strong, not traded)"
        if r["sent"] == "bullish":
            return "4. PROMPT under-score (weak bullish, mag<%.2f)" % NC
        if r["sent"] == "neutral":
            return "5. PROMPT miss (scored NEUTRAL)"
        return "6. PROMPT wrong-way (scored BEARISH, ran up)"
    winners = [r for r in rows if r["ret"] >= BIG]
    buckets = defaultdict(list)
    for r in winners:
        buckets[bucket(r)].append(r)
    print(f"═══ BIG WINNERS (≥{BIG}% in {HORIZON}d) among names we saw: {len(winners)} ═══")
    for b in sorted(buckets):
        rs = sorted(buckets[b], key=lambda r: -r["ret"])
        avg = sum(r["ret"] for r in rs) / len(rs)
        ex = "  ".join(f"{r['tk']}+{r['ret']:.0f}%({r['sent'][:4]} {r['mag']:.2f}/{r['conf']:.2f})" for r in rs[:4])
        print(f"  {b:50} n={len(rs):>3}  avg+{avg:>4.1f}%   {ex}")

    # ── (B) calibration ────────────────────────────────────────────────────────
    print("\n═══ CALIBRATION (scorer/prompt quality — statistically powered) ═══")
    by_sent = defaultdict(list)
    for r in rows:
        by_sent[r["sent"]].append(r["ret"])
    for s in ("bullish", "neutral", "bearish"):
        v = sorted(by_sent.get(s, []))
        if v:
            print(f"  {s:8} n={len(v):>4}  avg fwd {sum(v)/len(v):>+5.1f}%  median {v[len(v)//2]:>+5.1f}%  "
                  f"hit≥{BIG}%: {sum(x>=BIG for x in v)/len(v)*100:>4.0f}%")
    bull = [r for r in rows if r["sent"] == "bullish"]
    ic = spearman([r["mag"] * r["conf"] for r in bull], [r["ret"] for r in bull])
    ic_all = spearman([r["score"] for r in rows], [r["ret"] for r in rows])
    if ic is not None:
        print(f"\n  magnitude rank-IC (bullish mag×conf vs fwd return): {ic:+.3f}  (n={len(bull)})")
    if ic_all is not None:
        print(f"  signed-score rank-IC (all signals):                 {ic_all:+.3f}  (n={len(rows)})")
    tradedw = sum(1 for r in winners if r["traded"])
    print(f"\n  RECALL: of {len(winners)} winners we saw, we traded {tradedw} ({tradedw/max(len(winners),1)*100:.0f}%).")
    print("  Read: bullish avg should beat neutral/bearish (direction works); positive mag-IC means")
    print("  higher magnitude picks bigger movers. Re-run after a prompt change to see IC move.")


if __name__ == "__main__":
    main()
