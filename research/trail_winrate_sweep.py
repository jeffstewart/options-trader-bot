"""
trail_winrate_sweep.py — the live news_call exit is a flat 40% premium trail + 3-day hold. It lets
convex winners run but lets a +15% peak round-trip deep into the red before firing (theta + reversal).
This sweeps exit policies optimising for WIN RATE (accepting some P&L), reusing exit_sweep_unified's
news_call signal set + option sim (Black-Scholes + IV-crush + realistic spreads). Each policy reports
win%, $/trade, total P&L, Sharpe AND the round-trip pathology: % of trades that peaked ≥ +15% then
closed red, and of those that reached +15%, how many were still held to a green exit.

Usage:  USE_YAHOO_BARS=1 .venv/bin/python -u research/trail_winrate_sweep.py
"""
import os
os.environ.setdefault("USE_YAHOO_BARS", "1")
import exit_sweep_unified as es
import config as cfg

HOLD = 3                                                  # live news_call max-hold (the cap does most cutting)
DELTA, DTE, USD = cfg.NEWS_CALL_TARGET_DELTA, 10, cfg.MAX_POSITION_USD
GATE = lambda m, c: m >= cfg.NEWS_CALL_MIN_MAGNITUDE and es.dyn_gate(m, c)
ARMED = 0.15                                              # "winner" = peaked ≥ +15% (the user's threshold)


def metrics(trades):
    s = es.stat(trades)
    if not s["n"]:
        return s, 0.0, 0.0, 0
    faded   = [t for t in trades if t["peak_ret"] >= ARMED and t["pnl_pct"] < 0]   # +15%→red round-trips
    reached = [t for t in trades if t["peak_ret"] >= ARMED]
    cap     = (sum(1 for t in reached if t["pnl_pct"] >= 0) / len(reached) * 100) if reached else 0.0
    return s, len(faded) / s["n"] * 100, cap, len(reached)


def run(label, rows, reg, exit_rule, trail, params):
    trades = es.sim_option(rows, reg, trail, HOLD, GATE, DELTA, DTE, USD, exit_rule=exit_rule, exit_params=params)
    s, rt, cap, reached = metrics(trades)
    print(f"  {label:30} n={s['n']:<4} win{s['win']:>4.0f}%  ${s['mean']:>+5.0f}/tr  tot${s['total']/1000:>5.0f}k  "
          f"Sh{s['sharpe']:>4.2f}   |  +15%→red {rt:>4.1f}%  ({reached} hit +15%, {cap:.0f}% kept green)")
    return s


def main():
    _, end, days, cache = es.BULL
    rows = es.load_signals(cache, end, days, es.PROMPT, True)
    reg  = es.build_regime(end, days, 200)
    print(f"news_call EXIT-POLICY win-rate sweep — {len(rows)} signals · {HOLD}d hold · NET of costs")
    print("── flat trail (current family) ──")
    for tr in (0.40, 0.30, 0.25, 0.20):
        run(f"flat {tr*100:.0f}% trail" + (" ← LIVE" if tr == 0.40 else ""), rows, reg, "trail_premium", tr, {})
    print("── breakeven lock: 40% trail, once peak ≥ arm the stop floor = entry ──")
    for arm in (0.10, 0.15, 0.20):
        run(f"40% trail + BE-lock @+{arm*100:.0f}%", rows, reg, "trail_breakeven", 0.40, {"be_arm": arm, "be_lock": 0.0})
    print("── breakeven + small locked profit ──")
    run("40% + lock +5% @+20%", rows, reg, "trail_breakeven", 0.40, {"be_arm": 0.20, "be_lock": 0.05})
    print("── hard take-profit (40% trail + cap) ──")
    for tgt in (0.25, 0.40):
        run(f"take-profit @+{tgt*100:.0f}%", rows, reg, "profit_target", 0.40, {"target_pct": tgt})

    bl, bend, bdays, bcache = es.BEAR
    brows = es.load_signals(bcache, bend, bdays, es.PROMPT, True)
    breg  = es.build_regime(bend, bdays, 200)
    print(f"\n── BEAR guard ({bl}) ──")
    run("LIVE flat 40%", brows, breg, "trail_premium", 0.40, {})
    run("40% + BE-lock @+15%", brows, breg, "trail_breakeven", 0.40, {"be_arm": 0.15, "be_lock": 0.0})
    print("\n  Read: BE-lock should raise win% and cut the +15%→red rate; the cost is $/tr (faded-then-")
    print("  recovered winners get scratched). Pick the point where win% jumps most per $ of P&L given up.")


if __name__ == "__main__":
    main()
