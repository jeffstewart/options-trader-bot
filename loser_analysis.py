"""
loser_analysis.py — POST-MORTEM on the bot's REAL closed trades to find tunable patterns in our
LOSERS (so we can gate/prompt them out). Joins closed_trades.csv (realized P&L) to trades.csv
(score, spread, the LLM reasoning) and asks:
  (A) CATALYST TYPE — categorize each trade's reasoning (analyst/PT, M&A-acquirer, technical,
      macro-pinned, product, earnings, …) and show win-rate + P&L per type. Which catalyst kinds
      systematically lose? → prompt/gating targets.
  (B) DIRECTION — did the underlying go UP or DOWN over the hold? down = selection error (prompt
      picked wrong); up-but-lost = structure (IV-crush/theta/spread → geometry/exit, not prompt).
  (C) SCORE/SPREAD — are losers distinguishable by mag/conf/entry-spread?

Usage:  USE_YAHOO_BARS=1 .venv/bin/python -u loser_analysis.py
"""
import os, csv, json
os.environ.setdefault("USE_YAHOO_BARS", "1")
from datetime import datetime, timezone, timedelta
from collections import defaultdict
import yahoo_data

UCACHE = "yahoo_hold_cache.json"   # "tk_entry_exit" -> underlying % move over the hold

# catalyst categories by reasoning keyword (priority order — first match wins)
CATS = [
    ("analyst/PT",      ["price target", "analyst", "upgrade", "downgrade", "rating", "reiterat",
                          "initiat", "fab five", "raises forecast", "raise their", "pt hike", "boost their"]),
    ("M&A-acquirer",    ["to buy", "to acquire", "acquir", "in discussions to buy", "takeover bid", "buying"]),
    ("M&A-target",      ["acquired by", "takeover of", "to be acquired", "buyout of"]),
    ("technical/signal",["trading signal", "flashed", "breakout", "support", "resistance", "chart",
                          "momentum", "gain", "rally for", "surges", "soars", "jumps"]),
    ("macro/sector",    ["peace deal", "fed ", "tariff", "rate cut", "sector", "stocks are up",
                          "tech rally", "market rally", "broader market", "inflation"]),
    ("earnings",        ["earnings", "beat", "eps", "quarterly", "revenue", "q1 ", "q2 ", "q3 ", "q4 "]),
    ("guidance",        ["guidance", "outlook", "raises full-year", "forecast"]),
    ("FDA/approval",    ["fda", "approval", "phase", "trial", "clinical"]),
    ("product/deal",    ["launch", "introduces", "unveils", "signs", "deal", "partnership", "contract"]),
]


def categorize(text):
    t = (text or "").lower()
    for name, kws in CATS:
        if any(k in t for k in kws):
            return name
    return "vague/other"


def load_joined():
    opens = list(csv.DictReader(open("trades.csv")))
    osym = {o["option_symbol"]: o for o in opens if o.get("option_symbol") and o["option_symbol"] != "-"}
    # stock: ticker+date -> open row
    sday = {}
    for o in opens:
        if o.get("option_symbol") in (None, "-"):
            sday[(o["source_ticker"], o["timestamp"][:10])] = o
    rows = []
    for c in csv.DictReader(open("closed_trades.csv")):
        o = osym.get(c["symbol"]) or sday.get((c["underlying"], c["timestamp"][:10]))
        rows.append({
            "tk": c["underlying"], "strat": c["strategy"], "atype": c["asset_type"],
            "pnl": float(c["pnl_usd"]), "reason_exit": c["reason"],
            "close_date": c["timestamp"][:10],
            "mag": float(o["magnitude"]) if o and o.get("magnitude") else None,
            "conf": float(o["confidence"]) if o and o.get("confidence") else None,
            "spread": o.get("spread_pct") if o else None,
            "reasoning": o.get("reasoning", "") if o else "",
            "open_date": o["timestamp"][:10] if o else c["timestamp"][:10],
        })
    return rows


def underlying_move(rows):
    cache = json.load(open(UCACHE)) if os.path.exists(UCACHE) else {}
    for r in rows:
        ck = f"{r['tk']}_{r['open_date']}_{r['close_date']}"
        if ck not in cache:
            try:
                d0 = datetime.fromisoformat(r["open_date"]).replace(tzinfo=timezone.utc)
                d1 = datetime.fromisoformat(r["close_date"]).replace(tzinfo=timezone.utc)
                bars = sorted(yahoo_data.get_yahoo_bars(r["tk"], d0 - timedelta(days=3), d1 + timedelta(days=2)),
                              key=lambda b: b["t"])
                e = next((b["c"] for b in bars if b["t"].date() >= d0.date()), None)
                x = next((b["c"] for b in reversed(bars) if b["t"].date() <= max(d1.date(), d0.date())), None)
                cache[ck] = round((x / e - 1) * 100, 2) if e and x else None
            except Exception:
                cache[ck] = None
        r["umove"] = cache[ck]
    json.dump(cache, open(UCACHE, "w"))
    return rows


def main():
    rows = underlying_move(load_joined())
    tot = sum(r["pnl"] for r in rows)
    losers = [r for r in rows if r["pnl"] < 0]
    print(f"{len(rows)} closed trades · {len(losers)} losers · realized ${tot:+,.0f}\n")

    # ── (A) catalyst type (where reasoning is available) ───────────────────────
    have = [r for r in rows if r["reasoning"]]
    by = defaultdict(list)
    for r in have:
        by[categorize(r["reasoning"])].append(r)
    print(f"═══ (A) by CATALYST TYPE (n={len(have)} with reasoning) ═══")
    print(f"  {'catalyst':18} {'n':>3} {'win%':>5} {'totP&L':>9} {'avgP&L':>8}  example losers")
    for cat in sorted(by, key=lambda c: sum(r['pnl'] for r in by[c])):
        rs = by[cat]; w = sum(1 for r in rs if r["pnl"] > 0)
        exl = ", ".join(f"{r['tk']}${r['pnl']:+.0f}" for r in sorted(rs, key=lambda r: r["pnl"])[:3] if r["pnl"] < 0)
        print(f"  {cat:18} {len(rs):>3} {w/len(rs)*100:>4.0f}% {sum(r['pnl'] for r in rs):>+9.0f} "
              f"{sum(r['pnl'] for r in rs)/len(rs):>+8.0f}  {exl}")

    # ── (B) direction: selection vs structure ──────────────────────────────────
    md = [r for r in losers if r["umove"] is not None]
    down = [r for r in md if r["umove"] < -0.5]
    up = [r for r in md if r["umove"] > 0.5]
    print(f"\n═══ (B) LOSER DIRECTION (n={len(md)} with underlying move) ═══")
    print(f"  underlying DOWN (selection error): {len(down):>2}  (${sum(r['pnl'] for r in down):+,.0f})")
    print(f"  underlying UP   (structure: IV-crush/theta/spread): {len(up):>2}  (${sum(r['pnl'] for r in up):+,.0f})")
    print(f"  flat: {len(md)-len(down)-len(up)}")

    # ── (C) score / spread / strategy / exit ───────────────────────────────────
    def avg(xs): return sum(xs) / len(xs) if xs else 0
    win = [r for r in rows if r["pnl"] > 0]
    lm = [r["mag"] for r in losers if r["mag"] is not None]; wm = [r["mag"] for r in win if r["mag"] is not None]
    lc = [r["conf"] for r in losers if r["conf"] is not None]; wc = [r["conf"] for r in win if r["conf"] is not None]
    print(f"\n═══ (C) SCORE/STRATEGY ═══")
    print(f"  avg magnitude  losers {avg(lm):.2f}  vs winners {avg(wm):.2f}")
    print(f"  avg confidence losers {avg(lc):.2f}  vs winners {avg(wc):.2f}")
    bys = defaultdict(lambda: [0, 0.0])
    for r in rows:
        bys[r["strat"]][0] += 1; bys[r["strat"]][1] += r["pnl"]
    print("  by strategy: " + " · ".join(f"{k} ${v[1]:+,.0f}(n{v[0]})" for k, v in sorted(bys.items(), key=lambda x: x[1][1])))

    print("\n  Biggest losers:")
    for r in sorted(losers, key=lambda r: r["pnl"])[:8]:
        um = f"{r['umove']:+.1f}%" if r["umove"] is not None else "?"
        print(f"    {r['tk']:6} ${r['pnl']:+7.0f} [{r['strat']}] mag={r['mag']} undr {um} "
              f"{categorize(r['reasoning']):14} :: {r['reasoning'][:60]}")


if __name__ == "__main__":
    main()
