"""
pead_options_test.py — PEAD currently trades the STOCK (10-day hold, flat 20% trail). Would CALL OPTIONS
pay off, or does theta eat the slow ~10-day drift? Runs the SAME earnings-bullish (PEAD) signals through
(a) the current stock leg and (b) calls across delta × DTE — held 10 days with a premium trail, NET of
costs (Black-Scholes + IV-crush + spreads). Key tension: a 10-day hold needs DTE well above 10 (or the
option expires mid-drift) and higher delta (ITM = less theta) for the slow drift to survive. If no
geometry beats the stock, theta wins → keep PEAD as stock. If ITM/longer-DTE cells win, it's worth a
forward pead_options shadow.

Usage:  USE_YAHOO_BARS=1 .venv/bin/python -u research/pead_options_test.py

RESULT (2026-06-30) — NOT a clean win; DO NOT trust the headline total. Every option cell "beats" the
stock ($1,886 → up to $122,186) BUT: the MEDIAN option trade loses −24% (theta eats the slow drift, as
theorized); the total is pure tail (24/94 trades >+100%, e.g. SNOW $7.25→$67.70 = +780%); and it's
inflated by trades the live bot would REJECT — entry premiums down to $0.06 bought at qty 92 (the sim
applies no MAX_SPREAD_PCT / open-interest / contract-budget guards). So options convert PEAD from a
steady, high-Sharpe (3.61), 53%-win stock strategy into a lotto-style convex bet (42% win, median −24%).
UPDATE (2026-06-30, after a contract-selection sweep): GEOMETRY is decisive, and the totals misled —
look at the MEDIAN + win% (robust to the tail). ATM/short (Δ0.50/21d) loses (median −24%, theta wins),
but ITM/longer-DTE (Δ0.70/45d) flips POSITIVE: median +12%, 57% win — a BROAD-based edge (majority of
trades win), not a tail mirage. Higher delta (stock-like, less theta) + longer expiry let the slow drift
survive. Excluding junk contracts (prem≥$1, qty≤20) barely moved it, so it's not just the execution
artifact. Verdict: PROMISING at ITM/longer-DTE — worth a forward pead_options SHADOW at ~Δ0.70/45d (NOT
the ATM/short geometry). Totals stay tail-skewed + synthetic-priced, so the shadow on REAL quotes is the
confirmation. (Keep the live PEAD leg as stock until the shadow confirms.)
"""
import os
os.environ.setdefault("USE_YAHOO_BARS", "1")
import exit_sweep_unified as es
import config as cfg
import bot

HOLD = 10                                   # PEAD_MAX_HOLD_DAYS


def main():
    _, end, days, cache = es.BULL
    rows = es.load_signals(cache, end, days, es.PROMPT, want_ticker=True)
    pead = [r for r in rows if bot._is_earnings_headline(r["headline"])]
    reg = es.build_regime(end, days, 200)
    gate = es.dyn_gate
    print(f"PEAD options test — {len(pead)} earnings-bullish signals · {HOLD}d hold · NET of costs")

    base = es.stat(es.sim_pead(pead, reg, 0.20, HOLD, gate))
    print(f"\n  BASELINE — PEAD STOCK (20% trail): n={base['n']}  ${base['total']:,.0f}  "
          f"${base['mean']:+.0f}/tr  win {base['win']:.0f}%  Sharpe {base['sharpe']:+.2f}")

    print(f"\n  ── PEAD as CALLS (40% premium trail, {HOLD}d hold) — DTE must clear the 10d hold ──")
    print(f"  {'geometry':12}{'n':>5}{'total':>11}{'$/tr':>8}{'win%':>6}{'sharpe':>8}{'2x+':>5}")
    best = base["total"]
    for dte in (21, 30, 45):
        for delta in (0.50, 0.60, 0.70):
            s = es.stat(es.sim_option(pead, reg, 0.40, HOLD, gate, delta, dte, cfg.MAX_POSITION_USD))
            flag = "  ✓ beats stock" if s["total"] > base["total"] else ""
            best = max(best, s["total"])
            print(f"  Δ{delta:.2f}/{dte}d  {s['n']:>5}{s['total']:>+11,.0f}{s['mean']:>+8.0f}"
                  f"{s['win']:>6.0f}{s['sharpe']:>+8.2f}{s['x2']:>5}{flag}")
    print(f"\n  stock baseline ${base['total']:,.0f} vs best option ${best:,.0f}. "
          f"{'Options add → worth a forward shadow.' if best > base['total'] else 'Theta eats the slow drift → keep PEAD as stock.'}")


if __name__ == "__main__":
    main()
