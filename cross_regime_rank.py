"""
cross_regime_rank.py — pick parameters that are robust ACROSS regimes.

Joins the holdout results of the bull-melt-up tune and the 2022-bear tune (both
on real Yahoo data) by parameter combo, then ranks. The regime-gated strategy
trades mostly in uptrends, so the bull-window holdout is the primary objective;
the 2022 holdout is a tail-risk check (how bad if the filter lags into a bear).

Reports:
  • top combos by BULL holdout Sharpe (with their 2022 numbers alongside)
  • top combos by a ROBUSTNESS score (bull + bear holdout Sharpe)
  • the current live config row for comparison

Usage:
    .venv/bin/python cross_regime_rank.py
"""
import csv

# Full grid results (all combos) — the holdout CSVs only store top-5 per window
# so they barely overlap. These are TRAINING metrics (in-sample); use them for
# the cross-regime robustness SHAPE across all combos.
BULL = "tune_v2_bull_meltup_yahoo.csv"
BEAR = "tune_v2_bull_2022_yahoo.csv"

KEY = ("MIN_DTE", "MAX_DTE", "TARGET_DELTA", "MIN_MAGNITUDE", "BASE_CONFIDENCE", "TRAILING_STOP_PCT")
LIVE = ("28", "45", "0.4", "0.35", "0.55", "0.15")  # current config.py


def load(path):
    out = {}
    for r in csv.DictReader(open(path)):
        k = tuple(r[c] for c in KEY)
        out[k] = {"sharpe": float(r["sharpe"]), "pnl": float(r["total_pnl"]),
                  "win": float(r["win_rate"]), "trades": int(r["trades"])}
    return out


def combo_str(k):
    return f"DTE{k[0]}-{k[1]} Δ{k[2]} mag{k[3]} conf{k[4]} trail{k[5]}"


def main():
    bull, bear = load(BULL), load(BEAR)
    common = [k for k in bull if k in bear]
    print(f"Joined {len(common)} combos across both windows (real Yahoo data).\n")

    def row(k):
        b, r = bull[k], bear[k]
        return (f"  {combo_str(k):46}  bull: S={b['sharpe']:5.2f} ${b['pnl']:>9,.0f}"
                f"   2022: S={r['sharpe']:6.2f} ${r['pnl']:>8,.0f}")

    print("═══ TOP 8 by BULL holdout Sharpe (primary — where we trade) ═══")
    for k in sorted(common, key=lambda k: bull[k]["sharpe"], reverse=True)[:8]:
        print(row(k))

    print("\n═══ TOP 8 by ROBUSTNESS (bull Sharpe + 2022 Sharpe) ═══")
    def robust(k): return bull[k]["sharpe"] + bear[k]["sharpe"]
    for k in sorted(common, key=robust, reverse=True)[:8]:
        print(row(k) + f"   [robust={robust(k):.2f}]")

    print("\n═══ Most bull-profit while 2022 P&L ≥ 0 (ideal: profit both regimes) ═══")
    both_pos = [k for k in common if bear[k]["pnl"] >= 0]
    if both_pos:
        for k in sorted(both_pos, key=lambda k: bull[k]["pnl"], reverse=True)[:5]:
            print(row(k))
    else:
        print("  (none profitable in 2022 — expected; bear is hard for long calls)")
        # show least-bad 2022 among strong bull combos
        strong = sorted(common, key=lambda k: bull[k]["sharpe"], reverse=True)[:40]
        for k in sorted(strong, key=lambda k: bear[k]["pnl"], reverse=True)[:5]:
            print(row(k))

    print("\n═══ Current live config ═══")
    if LIVE in bull and LIVE in bear:
        print(row(LIVE))
    else:
        print(f"  {combo_str(LIVE)} — not in joined set "
              f"(bull:{LIVE in bull} bear:{LIVE in bear})")


if __name__ == "__main__":
    main()
