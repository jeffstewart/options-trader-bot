"""
fix_closed_trades_strategy.py — one-time (re-runnable) backfill of the strategy tag
in closed_trades.csv for OPTION trades that closed under the restart bug.

Bug (fixed 2026-06-08): on restart, reloaded option positions were hardcoded to
strategy="news_call", so lotto/qqq_macro options that later closed were logged to
closed_trades.csv as news_call → the dashboard's realized strategy P&L was wrong.

trades.csv is the authoritative, append-only entry record (option_symbol → strategy).
This re-attributes each OPTION row in closed_trades.csv to its true strategy. Stock
rows (key = TICKER__strategy) were never corrupted and are left untouched. Only
unambiguous symbols (one strategy in trades.csv) are changed. Idempotent: re-running
sets the same authoritative value. Backs up the original first; atomic replace.

Usage:  .venv/bin/python fix_closed_trades_strategy.py
"""
import csv, os, shutil
from collections import defaultdict
from datetime import datetime, timezone

TRADES, CLOSED = "trades.csv", "closed_trades.csv"


def main():
    # authoritative option_symbol → strategy (only if unambiguous)
    seen = defaultdict(set)
    for r in csv.DictReader(open(TRADES)):
        s = r.get("option_symbol")
        if s:
            seen[s].add(r.get("strategy"))
    auth = {s: next(iter(v)) for s, v in seen.items() if len(v) == 1}

    rows = list(csv.DictReader(open(CLOSED)))
    if not rows:
        print("closed_trades.csv empty — nothing to do"); return
    header = list(rows[0].keys())

    changes = []
    for r in rows:
        if r.get("asset_type") != "option":
            continue                      # stocks keep their (correct) tag
        true = auth.get(r.get("symbol"))
        if true and true != r.get("strategy"):
            changes.append((r.get("symbol"), r.get("strategy"), true, r.get("pnl_usd")))
            r["strategy"] = true

    if not changes:
        print("No mis-attributed option rows — closed_trades.csv already correct.")
        return

    # backup + atomic rewrite
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    backup = f"{CLOSED}.bak_strategyfix_{stamp}"
    shutil.copy2(CLOSED, backup)
    tmp = CLOSED + ".tmp"
    with open(tmp, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=header)
        w.writeheader()
        w.writerows(rows)
    os.replace(tmp, CLOSED)

    print(f"Backed up original → {backup}")
    print(f"Re-attributed {len(changes)} option row(s):")
    for sym, old, new, pnl in changes:
        print(f"  {sym:28} {old:10} → {new:10}  (pnl ${pnl})")


if __name__ == "__main__":
    main()
