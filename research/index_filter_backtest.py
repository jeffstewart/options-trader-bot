"""
index_filter_backtest.py — should INDEX-INCLUSION/REPLACEMENT headlines ("X to join/replace Y in the
S&P/Dow/Nasdaq/Russell") be dropped pre-score? They're weak catalysts the model over-scores (GOOG
'joins the DJIA' → mag 0.85 → tripped the regime bypass → −36%). Same method as the soft-catalyst
validation (prefilter_pnl.py): reuse the cached simulated news_call option trades, split by the
pattern, and bootstrap the P&L of the trades the filter would REMOVE. Removed set net-negative
(CI < 0) = the filter helps. Also scans the LIVE trade log for the same pattern.

Usage:  .venv/bin/python -u index_filter_backtest.py
"""
import os, json, re, csv, random
import numpy as np

# Candidate filter pattern — index add/drop/rebalance news.
IDX = r"(s&amp;p|s&p|dow jones|dow industrial|nasdaq-?100|russell|midcap|smallcap)"
PAT = re.compile(
    r"(to replace .{0,45}? in the .{0,25}" + IDX + r"|"
    r"to join the .{0,25}" + IDX + r"|"
    r"\bjoins?\b.{0,35}(dow jones|" + IDX + r" ?\d|russell|nasdaq)|"
    r"added to .{0,25}" + IDX + r"|"
    r"selected to join|"
    r"in the dow jones industrial average)", re.I)


def boot(pnls, n=5000):
    if not pnls:
        return (0, 0, 0)
    rng = random.Random(0)
    tot = [sum(rng.choices(pnls, k=len(pnls))) for _ in range(n)]
    return sum(pnls), float(np.percentile(tot, 5)), float(np.percentile(tot, 95))


def stat(name, pnls):
    if not pnls:
        print(f"  {name:34} n=0"); return
    n = len(pnls); tot = sum(pnls); win = sum(1 for p in pnls if p > 0)
    s, lo, hi = boot(pnls)
    sig = "SIG<0 ✓" if hi < 0 else ("SIG>0" if lo > 0 else "spans 0")
    print(f"  {name:34} n={n:<4} total ${tot:>+8.0f} · avg ${tot/n:>+6.0f} · win {win}/{n} · boot CI [${lo:>+7.0f},${hi:>+7.0f}] {sig}")


def main():
    print("══ PATTERN SANITY (must match these, the real index headlines) ══")
    for h in ["Selected to join the Dow Jones Industrial Average",
              "Alphabet To Replace Verizon In Dow Jones Industrial Average",
              "Toast To Replace TopBuild Corp. In The S&amp;P Midcap 400",
              "Silver Bow Mining To Join The Russell 3000 And Russell MicroCap Indexes",
              "National Health Investors To Replace Apollo Commercial Real Estate Finance In The S&amp;P SmallCap 600"]:
        print(f"  [{'HIT ' if PAT.search(h) else 'MISS'}] {h[:70]}")
    # must NOT match normal catalysts:
    print("  -- should NOT match (real catalysts): --")
    for h in ["Upstart announces $600 million investment", "Broadcom unveils Jalapeño AI inference chip",
              "Qualcomm to exit FY26 at $6B automotive revenue", "Nvidia beats earnings, raises guidance"]:
        print(f"  [{'HIT(!)' if PAT.search(h) else 'ok  '}] {h[:70]}")

    print("\n══ BACKTEST — P&L of the news_call trades this filter would REMOVE ══")
    if not os.path.exists("prefilter_pnl_trades.json"):
        print("  (no prefilter_pnl_trades.json — run prefilter_pnl.py first)"); return
    trades = json.load(open("prefilter_pnl_trades.json"))
    matched = [t for t in trades if PAT.search(t.get("h", "") or "")]
    kept = [t for t in trades if not PAT.search(t.get("h", "") or "")]
    print(f"  simulated news_call universe: {len(trades)} trades")
    stat("ALL (current, no index filter)", [t["pnl"] for t in trades])
    stat("REMOVED by index filter", [t["pnl"] for t in matched])
    stat("KEPT (after index filter)", [t["pnl"] for t in kept])
    if matched:
        print("\n  index-inclusion trades it would remove (P&L · headline):")
        for t in sorted(matched, key=lambda x: x["pnl"]):
            print(f"    ${t['pnl']:>+7.0f}  {(t['h'] or '')[:78]}")

    print("\n══ LIVE history — index-inclusion trades actually taken (trades.csv ⋈ closed_trades.csv) ══")
    live = [r for r in csv.DictReader(open("trades.csv"))] if os.path.exists("trades.csv") else []
    closed = {}
    if os.path.exists("closed_trades.csv"):
        for r in csv.DictReader(open("closed_trades.csv")):
            closed.setdefault(r.get("underlying") or r.get("symbol"), []).append(r)
    hits = [r for r in live if PAT.search(r.get("reasoning", "") or "")]
    print(f"  live trades whose catalyst matches the index pattern: {len(hits)}")
    for r in hits:
        tk = r.get("source_ticker", "")
        pnl = next((float(c["pnl_usd"]) for c in closed.get(tk, []) if c["timestamp"] >= r["timestamp"]), None)
        print(f"    {r['timestamp'][:16]} {tk:<6} {r.get('strategy',''):<10} mag={r.get('magnitude','')} "
              f"realized={'$%+.0f' % pnl if pnl is not None else 'open'}  :: {(r.get('reasoning','') or '')[:60]}")


if __name__ == "__main__":
    main()
