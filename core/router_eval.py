"""
router_eval.py — evaluate the SHADOW router against fixed single strategies.

The live bot logs the router's per-signal pick to router_decisions.csv but opens
no extra orders — every strategy already trades. This script attributes the
realized P&L of those actual trades (closed_trades.csv) to the router's choices,
then compares the router's cumulative P&L to each always-one-strategy baseline.

Attribution (per-event, greedy, time-ordered):
  • For a router pick of stock / pead / bear_short / qqq_macro: claim the nearest
    not-yet-claimed closed trade with the same (strategy, underlying) whose close
    is at/after the decision time.
  • For a "pairs" pick: claim the pairs_long(ticker) + pairs_short(partner) closes.
  • "Best fixed" = the single strategy with the highest realized total. The router
    only adds value if its total beats that.

Note: only CLOSED trades count (realized). Run it any time — it fills in as the
forward test accumulates. Until trades close it will report nothing matched.

Usage:  .venv/bin/python router_eval.py
"""
import csv
from collections import defaultdict
from datetime import datetime
from pathlib import Path

ROUTER_CSV = Path("router_decisions.csv")
CLOSED_CSV = Path("closed_trades.csv")


def _ts(s):
    try:
        return datetime.fromisoformat(str(s).replace("Z", "+00:00"))
    except Exception:
        return None


def load_closed():
    """Return list of closed trades as dicts, sorted by close time."""
    if not CLOSED_CSV.exists():
        return []
    rows = []
    with open(CLOSED_CSV, newline="") as f:
        for r in csv.DictReader(f):
            r["_ts"]  = _ts(r.get("timestamp"))
            r["_pnl"] = float(r.get("pnl_usd", 0) or 0)
            rows.append(r)
    rows.sort(key=lambda r: r["_ts"] or datetime.min)
    return rows


def load_decisions():
    if not ROUTER_CSV.exists():
        return []
    with open(ROUTER_CSV, newline="") as f:
        rows = list(csv.DictReader(f))
    for r in rows:
        r["_ts"] = _ts(r.get("timestamp"))
    rows.sort(key=lambda r: r["_ts"] or datetime.min)
    return rows


def claim(rows, used, strategy, underlying, after_ts):
    """Greedily claim the nearest unused row matching (strategy, underlying)
    at/after after_ts. Rows with no timestamp (open positions) are always
    eligible. Returns pnl or None."""
    best_i, best = None, None
    for i, c in enumerate(rows):
        if i in used:
            continue
        if c.get("strategy") != strategy:
            continue
        if c.get("underlying") != underlying:
            continue
        if after_ts and c["_ts"] and c["_ts"] < after_ts:
            continue
        if best is None or (c["_ts"] and best["_ts"] and c["_ts"] < best["_ts"]):
            best, best_i = c, i
    if best_i is not None:
        used.add(best_i)
        return best["_pnl"]
    return None


def attribute(decisions, rows):
    """Attribute each router decision to a matching trade row (realized closed
    trades OR open positions — both normalized to {strategy, underlying, _ts,
    _pnl}). Greedy, time-ordered, each row claimed at most once.
    Returns (total_pnl, matched_flags, wins, losses) where matched_flags is a
    list aligned to `decisions` (True if that decision claimed a row)."""
    used = set()
    total, wins, losses = 0.0, 0, 0
    flags = []
    for d in decisions:
        choice, tk, after = d["router_choice"], d["ticker"], d.get("_ts")
        if choice == "pairs":
            lp = claim(rows, used, "pairs_long",  tk,                  after)
            sp = claim(rows, used, "pairs_short", d.get("pair_partner"), after)
            got = (lp or 0) + (sp or 0) if (lp is not None or sp is not None) else None
        else:
            got = claim(rows, used, choice, tk, after)
        flags.append(got is not None)
        if got is not None:
            total += got
            if got > 0: wins += 1
            else:       losses += 1
    return total, flags, wins, losses


def main():
    decisions = load_decisions()
    closed    = load_closed()

    print("═══ SHADOW ROUTER EVALUATION ═══\n")
    print(f"  Router decisions logged : {len(decisions)}")
    print(f"  Closed trades on book   : {len(closed)}")
    if not decisions:
        print("\n  No router decisions yet — bot logs them live as signals arrive.")
        return

    # Decision mix
    mix = defaultdict(int)
    for d in decisions:
        mix[d["router_choice"]] += 1
    print("  Router pick distribution:",
          "  ".join(f"{k}={v}" for k, v in sorted(mix.items())))

    if not closed:
        print("\n  No closed trades yet — router P&L fills in as positions close.")
        print("  (Decisions are being tagged now so the forward test can run later.)")
        return

    # ── Attribute realized P&L to router picks ────────────────────────────────
    router_pnl, flags, _wins, _losses = attribute(decisions, closed)
    matched   = sum(flags)
    unmatched = len(flags) - matched

    # ── Fixed-strategy baselines (all realized P&L per strategy) ──────────────
    fixed = defaultdict(float)
    fixed_n = defaultdict(int)
    for c in closed:
        fixed[c.get("strategy", "?")] += c["_pnl"]
        fixed_n[c.get("strategy", "?")] += 1

    print(f"\n  Router matched {matched}/{len(decisions)} decisions to closed "
          f"trades ({unmatched} not yet closed/no trade).")
    print(f"\n  {'Policy':<16}{'Realized P&L':>14}{'Trades':>9}")
    print("  " + "─" * 39)
    print(f"  {'ROUTER':<16}{router_pnl:>14,.2f}{matched:>9}")
    for s in sorted(fixed, key=lambda k: fixed[k], reverse=True):
        print(f"  {s:<16}{fixed[s]:>14,.2f}{fixed_n[s]:>9}")

    if fixed:
        best = max(fixed, key=lambda k: fixed[k])
        verdict = ("BEATS" if router_pnl > fixed[best] else "does NOT beat")
        print(f"\n  Best fixed strategy: {best} (${fixed[best]:,.2f})")
        print(f"  → Router {verdict} the best fixed strategy "
              f"(${router_pnl:,.2f} vs ${fixed[best]:,.2f}).")
    print("\n  Note: realized only; small samples are noisy. Let it accumulate.")


if __name__ == "__main__":
    main()
