"""
lotto_materiality_eval.py — forward A/B readout for the lotto materiality shadow.

The live bot scores every lotto entry's materiality (lotto_materiality.csv) but trades
ALL of them (baseline arm preserved). This joins those scores to realized P&L in
closed_trades.csv (by option symbol) and compares:

  • BASELINE  = all lotto trades (current behavior)
  • FILTERED  = lotto trades with materiality ≥ MATERIALITY_GATE (the candidate gate)
  • EXCLUDED  = the trades the gate would have skipped (what we'd give up)

If forward FILTERED beats BASELINE (higher avg/total, similar-or-better win rate) like
the backtest said, promote materiality to a hard live gate on lotto.

Usage:  .venv/bin/python lotto_materiality_eval.py
"""
import csv, os
from collections import defaultdict
from config import MATERIALITY_GATE

MAT_FILE, CLOSED = "lotto_materiality.csv", "closed_trades.csv"


def load_materiality():
    m = {}
    if not os.path.exists(MAT_FILE):
        return m
    with open(MAT_FILE) as f:
        for r in csv.DictReader(f):
            v = r.get("materiality", "")
            m[r["option_symbol"]] = float(v) if v not in ("", None) else None
    return m


def load_closed_lotto():
    out = []
    if not os.path.exists(CLOSED):
        return out
    with open(CLOSED) as f:
        for r in csv.DictReader(f):
            if r.get("strategy") == "lotto":
                try:
                    out.append((r["symbol"], float(r["pnl_usd"])))
                except Exception:
                    continue
    return out


def stats(rows):
    n = len(rows)
    if not n:
        return "n=0"
    pnls = [p for _, p in rows]
    tot = sum(pnls)
    wins = sum(1 for p in pnls if p > 0)
    return (f"n={n:>3}  total=${tot:>8,.0f}  avg=${tot/n:>7,.0f}  "
            f"win={wins/n*100:>4.1f}%  best=${max(pnls):>7,.0f}  worst=${min(pnls):>7,.0f}")


def main():
    mat = load_materiality()
    closed = load_closed_lotto()
    joined = [(s, p, mat.get(s)) for s, p in closed]
    scored = [(s, p, m) for s, p, m in joined if m is not None]
    unscored = len(joined) - len(scored)

    print(f"LOTTO materiality forward A/B  (gate = materiality ≥ {MATERIALITY_GATE})")
    print(f"  closed lotto trades: {len(closed)}   with a materiality score: {len(scored)}   "
          f"(no score yet: {unscored})\n")
    if not scored:
        print("  No scored+closed lotto trades yet — check back after some lotto positions close.")
        if closed:
            print(f"  (there are {len(closed)} closed lotto trades but none joined a materiality row —")
            print("   they likely opened before the shadow filter went live.)")
        return

    baseline = [(s, p) for s, p, m in scored]
    filtered = [(s, p) for s, p, m in scored if m >= MATERIALITY_GATE]
    excluded = [(s, p) for s, p, m in scored if m < MATERIALITY_GATE]

    print(f"  BASELINE (all lotto)        {stats(baseline)}")
    print(f"  FILTERED (mat ≥ {MATERIALITY_GATE})       {stats(filtered)}")
    print(f"  EXCLUDED (mat < {MATERIALITY_GATE})       {stats(excluded)}")
    print("\n  → Promote to a hard gate when FILTERED's avg/total beats BASELINE on a")
    print("    meaningful sample AND EXCLUDED is net-negative (the gate drops losers).")


if __name__ == "__main__":
    main()
