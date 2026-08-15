"""
spread_cap_log_mining.py — mine bot.log for the FULL candidate-contract attempt sequence behind
each lotto/news_call entry, to answer a question trade_review_202608.py's simple spread-cap sweep
can't: when a signal's accepted contract would fail a TIGHTER spread cap, would the bot actually
have found NO trade at all, or fallen through to one of the other (never-queried) candidates in
its own 5-contract shortlist?

jeff's question (2026-08-XX): "do we actually have enough data to make this [spread cap] testable
in a backtest?" No, not fully -- see the conversation for the three gaps (contract-fallback
behavior, core/backtest.py's synthetic (non-market) spread model, small n). This script closes
part of gap 1 using data we already have for free: both bots log every REJECTED candidate's real
spread_pct as plain text ("spread too wide: X% > Y% max — skipping"), one line per candidate,
immediately before either a "📋 ... spread=Z%" + "✅ OPTION BUY" (accepted) or a "no liquid
contracts found" / "no contracts found" (fully dead) terminal line for that ticker.

WHAT THIS CAN AND CAN'T TELL US:
- Every candidate BEFORE the accepted one already failed SOME guardrail under TODAY's settings --
  so it would fail again under any tighter spread cap too (tightening only adds restrictions).
  Those lines add no new information about a tighter cap's effect on THIS signal.
- What matters is whether there's ROOM LEFT in the 5-contract shortlist after the accepted one.
  Both bots' contract pickers do `for c in contracts[:5]: ... if passes: break` (core/bot.py
  pick_call_contract, v2/execution.py find target) -- so if the accepted contract was e.g. the
  2nd one tried, candidates 3-5 were NEVER QUERIED and we have zero data on them.
- REVISED 2026-08-XX: the per-candidate loop rejects for FOUR distinct reasons, not just spread --
  spread-too-wide, open interest below MIN_OPEN_INTEREST, non-standard/adjusted contract root, and
  a bid/ask-vs-intrinsic sanity check (core/bot.py:1061-1092, v2/execution.py
  _passes_contract_checks). An earlier version of this script only recognized spread rejections as
  valid "candidate tried" evidence -- an OI/root/bad-data rejection for a ticker's OWN candidate
  search was invisible to it and hit the generic "any other line resets the buffer" branch,
  DISCARDING that ticker's own already-accumulated spread rejects mid-sequence and undercounting
  n_prior_rejects whenever one signal's search mixed rejection reasons across its up to 5
  candidates. Caught by hand-tracing a GLD "died after 6 spread-rejects" result from an earlier,
  cruder ad hoc query against these same logs -- the real log showed GLD's candidates were
  rejected for LOW OPEN INTEREST (OI=2, OI=3, OI=22, all under the 100 minimum), not spread; the
  200%-spread figures that query attributed to GLD actually belonged to SPY's immediately
  preceding, separate search. All four reasons are now tracked as first-class attempts so
  sequences are scoped correctly and reasons are attributed accurately.

Usage (run from repo root):
    .venv/bin/python research/spread_cap_log_mining.py
    .venv/bin/python research/spread_cap_log_mining.py --caps 15 12 10
"""
import argparse
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

# Order matters only for readability; each candidate line matches exactly one of these.
SPREAD_RE = re.compile(r"spread too wide:\s*([\d.]+)%")
OI_RE = re.compile(r"(\S+) OI=(\S+) below min (\d+), skipping")
ROOT_MISMATCH_RE = re.compile(r"(\S+) is a non-standard/adjusted contract \(root (\S+)\) — skipping")
BAD_DATA_RE = re.compile(r"(\S+) quote \$[\d.]+ inconsistent w/ stock")
ACCEPT_RE = re.compile(r"📋\s+(\S+)\s+strike=.*?spread=([\d.]+)%")
BUY_RE = re.compile(r"✅ OPTION BUY\s+(\S+)")
DEAD_RE = re.compile(r"no liquid contracts found for (\S+) after guardrail checks"
                      r"|no contracts found for (\S+) in range")


def mine_log(path: Path, bot: str) -> list[dict]:
    """One record per completed candidate-search attempt (ends in either an accepted buy or a
    dead end). `attempts` is the ordered list of (reason, detail) tuples for every REJECTED
    candidate tried before the outcome; `n_prior_rejects` = len(attempts), a LOWER BOUND on how
    many of the 5 shortlist slots were consumed before the outcome (a handful of guardrail checks,
    e.g. `tradable` / `mid <= 0`, produce no log line at all -- see _passes_contract_checks)."""
    if not path.exists():
        return []
    records = []
    pending: list[tuple] = []          # [(reason, detail_float_or_str), ...]
    pending_accept_spread = None

    def reset():
        nonlocal pending, pending_accept_spread
        pending, pending_accept_spread = [], None

    with open(path, errors="replace") as f:
        for line in f:
            m = SPREAD_RE.search(line)
            if m:
                pending.append(("spread", float(m.group(1))))
                continue
            m = OI_RE.search(line)
            if m:
                pending.append(("low_oi", m.group(2)))
                continue
            m = ROOT_MISMATCH_RE.search(line)
            if m:
                pending.append(("bad_root", m.group(2)))
                continue
            m = BAD_DATA_RE.search(line)
            if m:
                pending.append(("bad_data", None))
                continue
            m = ACCEPT_RE.search(line)
            if m:
                pending_accept_spread = float(m.group(2))
                continue
            m = BUY_RE.search(line)
            if m:
                records.append({
                    "bot": bot, "symbol": m.group(1), "outcome": "accepted",
                    "attempts": pending[:], "n_prior_rejects": len(pending),
                    "accepted_spread": pending_accept_spread,
                })
                reset()
                continue
            m = DEAD_RE.search(line)
            if m:
                if pending:    # only care about dead-ends that tried at least one real candidate
                    ticker = m.group(1) or m.group(2)
                    records.append({
                        "bot": bot, "symbol": ticker, "outcome": "dead",
                        "attempts": pending[:], "n_prior_rejects": len(pending),
                        "accepted_spread": None,
                    })
                reset()
                continue
            # Any other line (headline, scorer output, unrelated log) closes out an in-progress
            # sequence -- these lines are tight (same-second) in practice, so this is a safe reset.
            # This is also what fixes the GLD mis-attribution: an OI-only sequence for one ticker
            # no longer inherits a leftover spread-only buffer from a PRECEDING ticker's sequence,
            # because every recognized line type (including these four) is tracked in the same
            # `pending` list and reset together at each terminal marker.
            if pending or pending_accept_spread is not None:
                reset()
    return records


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--caps", type=float, nargs="+", default=[15, 12, 10, 8])
    args = ap.parse_args()

    all_records = (mine_log(ROOT / "logs" / "bot.log", "v1")
                   + mine_log(ROOT / "v2" / "logs" / "bot.log", "v2"))
    accepted = [r for r in all_records if r["outcome"] == "accepted"]
    dead = [r for r in all_records if r["outcome"] == "dead"]
    print(f"mined {len(all_records)} completed candidate-search attempts "
          f"({len(accepted)} accepted a contract, {len(dead)} died with no trade)\n")

    print("=" * 100)
    print("1. HOW OFTEN DOES THE ACCEPTED CONTRACT COME AFTER A SPREAD-REJECTED CANDIDATE?")
    print("=" * 100)
    from collections import Counter
    pos_counts = Counter(r["n_prior_rejects"] for r in accepted)
    for n in sorted(pos_counts):
        pct = 100 * pos_counts[n] / len(accepted)
        room_left = "no fallback room left (5-slot shortlist exhausted)" if n >= 4 else \
                    f"{4 - n} untested candidate(s) still in the shortlist"
        print(f"  {n} prior spread-rejects  n={pos_counts[n]:<4} ({pct:>4.1f}%)  {room_left}")

    print("\n" + "=" * 100)
    print("2. FOR EACH TIGHTER CAP: split accepted trades into CERTAIN vs AMBIGUOUS outcomes")
    print("=" * 100)
    print(" CERTAIN-would-die   = accepted spread > cap AND n_prior_rejects >= 4 (shortlist")
    print("                       exhausted -- no untested candidate could have saved it)")
    print(" AMBIGUOUS           = accepted spread > cap AND n_prior_rejects < 4 (there WAS")
    print("                       fallback room the bot never had to use -- unknown outcome)")
    print(" unaffected          = accepted spread <= cap (same trade either way)\n")
    for cap in sorted(args.caps, reverse=True):
        unaffected = [r for r in accepted if r["accepted_spread"] is not None and r["accepted_spread"] <= cap]
        certain_die = [r for r in accepted if r["accepted_spread"] is not None and r["accepted_spread"] > cap
                       and r["n_prior_rejects"] >= 4]
        ambiguous = [r for r in accepted if r["accepted_spread"] is not None and r["accepted_spread"] > cap
                     and r["n_prior_rejects"] < 4]
        print(f" cap={cap:>4}%:  unaffected={len(unaffected):<4} certain-die={len(certain_die):<4} "
              f"ambiguous={len(ambiguous):<4}  (of {len(accepted)} total accepted trades)")

    print("\n" + "=" * 100)
    print("3. THE AMBIGUOUS TRADES AT cap=12% -- these are exactly what a simple spread-cap sweep")
    print("   on trades.csv CANNOT tell you: real trades where tightening the cap wouldn't")
    print("   necessarily kill the signal, just redirect it to an untested contract")
    print("=" * 100)
    cap = 12.0
    ambiguous = [r for r in accepted if r["accepted_spread"] is not None and r["accepted_spread"] > cap
                 and r["n_prior_rejects"] < 4]
    for r in sorted(ambiguous, key=lambda r: -r["accepted_spread"]):
        print(f"  {r['bot']:<3} {r['symbol']:<24} accepted_spread={r['accepted_spread']:>5.1f}%  "
              f"prior_rejects={r['n_prior_rejects']} (each of those was >20%, so a tighter cap "
              f"doesn't change their fate) but {4 - r['n_prior_rejects']} LATER candidate(s) in "
              f"the 5-slot shortlist were never even queried -- unknown whether one of them "
              f"would have cleared a tighter cap")

    print("\n" + "=" * 100)
    print("4. DEAD SIGNALS (no trade under current 20% cap) -- how many candidates did they burn")
    print("   through before giving up?")
    print("=" * 100)
    dead_counts = Counter(r["n_prior_rejects"] for r in dead)
    for n in sorted(dead_counts):
        print(f"  died after {n} rejects  n={dead_counts[n]}")

    print("\n" + "=" * 100)
    print("5. WHY DID CANDIDATES ACTUALLY GET REJECTED? (all attempts, accepted + dead signals)")
    print("=" * 100)
    reason_counts = Counter()
    spread_by_reason = []
    for r in all_records:
        for reason, detail in r["attempts"]:
            reason_counts[reason] += 1
            if reason == "spread":
                spread_by_reason.append(detail)
    total = sum(reason_counts.values())
    for reason, n in reason_counts.most_common():
        print(f"  {reason:<12} n={n:<5} ({100*n/total:>4.1f}%)")
    if spread_by_reason:
        no_bid = sum(1 for v in spread_by_reason if v >= 200)
        wide = sum(1 for v in spread_by_reason if 100 <= v < 200)
        print(f"\n  of the {len(spread_by_reason)} spread rejects specifically:")
        print(f"    exactly 200% (bid=0, literally no market) : {no_bid} ({100*no_bid/len(spread_by_reason):.1f}%)")
        print(f"    100-199% (essentially no two-sided market): {wide} ({100*wide/len(spread_by_reason):.1f}%)")

    print("\n" + "=" * 100)
    print("6. DEAD SIGNALS THAT EXHAUSTED THE SHORTLIST (n_prior_rejects >= 5) -- reason mix")
    print("=" * 100)
    exhausted = [r for r in dead if r["n_prior_rejects"] >= 5]
    print(f"  n={len(exhausted)} of {len(dead)} dead signals\n")
    exhausted_reasons = Counter()
    for r in exhausted:
        for reason, _ in r["attempts"]:
            exhausted_reasons[reason] += 1
    total_ex = sum(exhausted_reasons.values())
    for reason, n in exhausted_reasons.most_common():
        print(f"  {reason:<12} n={n:<5} ({100*n/total_ex:>4.1f}% of rejections within this cohort)")
    print("\n  sample tickers (bot, symbol, [reason:detail, ...]):")
    for r in exhausted[:15]:
        seq = [f"{reason}:{detail}" for reason, detail in r["attempts"]]
        print(f"    {r['bot']:<3} {r['symbol']:<8} {seq}")


if __name__ == "__main__":
    main()
