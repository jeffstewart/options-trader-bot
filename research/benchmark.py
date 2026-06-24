"""
benchmark.py — does the news/LLM signal beat market beta?

The backtest window is a one-directional bull melt-up (SPY +10.4%, QQQ +18.2%,
~zero drawdown), so a strong long-call P&L proves little on its own. This
script holds EVERYTHING constant except ticker selection and asks whether the
strategy's name-picking beats naive alternatives:

  • strategy  — the real tickers the LLM signal chose (sanity re-sim)
  • QQQ / SPY — buy an index call at the same moments & sizing (pure beta)
  • random    — buy a call on a random name from the same traded universe,
                ignoring the signal (averaged over several seeds)

Every arm reuses the validated BS pricing engine (backtest.simulate_option_pnl)
with identical entry dates and position sizes — only the underlying changes.
If the strategy doesn't beat QQQ/random on risk-adjusted terms, the signal
adds no alpha over being long the market.

Usage:
    .venv/bin/python benchmark.py [--seeds 5] [--trades backtest_trades.csv]
"""

import argparse
import csv
import math
import random
from datetime import datetime, timezone

import backtest as bt


def compute_stats(trades: list[dict]) -> dict:
    """Replicates backtest.py's stats math exactly for a fair comparison."""
    n = len(trades)
    if n == 0:
        return {"trades": 0, "total_pnl": 0, "win_rate": 0, "avg_win": 0,
                "avg_loss": 0, "profit_factor": 0, "max_dd": 0, "sharpe": 0}
    wins   = [t for t in trades if t["pnl_usd"] > 0]
    losses = [t for t in trades if t["pnl_usd"] <= 0]
    total  = sum(t["pnl_usd"] for t in trades)
    avg_win  = sum(t["pnl_usd"] for t in wins) / len(wins) if wins else 0
    avg_loss = sum(t["pnl_usd"] for t in losses) / len(losses) if losses else 0

    # Max drawdown from the equity curve (trades in entry-date order)
    ordered = sorted(trades, key=lambda t: t["entry_dt"])
    run = peak = max_dd = 0.0
    for t in ordered:
        run += t["pnl_usd"]
        peak = max(peak, run)
        max_dd = max(max_dd, peak - run)

    # Sharpe (same simplified per-trade method as the backtest)
    rets = [t["pnl_pct"] for t in trades]
    mean_r = sum(rets) / len(rets)
    std_r  = math.sqrt(sum((r - mean_r) ** 2 for r in rets) / len(rets)) or 1
    sharpe = (mean_r / std_r) * math.sqrt(252)

    return {
        "trades": n,
        "total_pnl": total,
        "win_rate": len(wins) / n * 100,
        "avg_win": avg_win,
        "avg_loss": avg_loss,
        "profit_factor": abs(avg_win / avg_loss) if avg_loss else 0,
        "max_dd": max_dd,
        "sharpe": sharpe,
    }


def resim(ticker: str, entry_dt: datetime, position_usd: float, signal: dict):
    """Re-price one trade with a swapped underlying, same date & sizing."""
    px = bt.get_price_at(ticker, entry_dt)
    if not px:
        return None
    return bt.simulate_option_pnl(ticker, entry_dt, px, position_usd, signal)


def run_arm(strat_trades, ticker_fn):
    """ticker_fn(row) → underlying to use for that trade (None to skip)."""
    out = []
    for row in strat_trades:
        tk = ticker_fn(row)
        if not tk:
            continue
        sig = {"magnitude": float(row.get("magnitude", 0) or 0),
               "confidence": float(row.get("confidence", 0) or 0),
               "reasoning": ""}
        res = resim(tk, row["_entry_dt"], float(row["position_usd"]), sig)
        if res:
            out.append(res)
    return out


def fmt(s):
    return (f"  {s['arm']:<16} trades={s['trades']:>4}  P&L=${s['total_pnl']:>11,.0f}  "
            f"Sharpe={s['sharpe']:>5.2f}  win={s['win_rate']:>4.1f}%  "
            f"PF={s['profit_factor']:>4.2f}  maxDD=${s['max_dd']:>9,.0f}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--trades", default="backtest_trades.csv")
    ap.add_argument("--seeds", type=int, default=5, help="random-arm seeds to average")
    ap.add_argument("--sample", type=int, default=0,
                    help="compare arms on N trades spread across the window (0 = all)")
    args = ap.parse_args()

    with open(args.trades, newline="") as f:
        strat = list(csv.DictReader(f))
    # All arms use the SAME (optionally sampled) trade set & dates → fair compare
    if args.sample and len(strat) > args.sample:
        step = len(strat) / args.sample
        strat = [strat[int(i * step)] for i in range(args.sample)]
    for r in strat:
        dt = datetime.fromisoformat(r["entry_dt"])
        r["_entry_dt"] = dt if dt.tzinfo else dt.replace(tzinfo=timezone.utc)

    universe = sorted({r["ticker"] for r in strat if "/" not in r["ticker"]})
    print(f"Loaded {len(strat)} strategy trades over {len(universe)} distinct names.")
    print("Re-pricing each arm with identical entry dates & sizing "
          "(only the underlying changes)…\n")

    results = []

    # Arm 0: re-sim the real tickers (sanity — should ≈ the backtest total)
    s = compute_stats(run_arm(strat, lambda r: r["ticker"]))
    s["arm"] = "strategy"; results.append(s); print(fmt(s))

    # Arms 1-2: index calls (pure beta)
    for idx in ("QQQ", "SPY"):
        s = compute_stats(run_arm(strat, lambda r, t=idx: t))
        s["arm"] = f"{idx} (beta)"; results.append(s); print(fmt(s))

    # Arm 3: random names from the same universe, averaged over seeds
    rand_stats = []
    for seed in range(args.seeds):
        rnd = random.Random(seed)
        arm = run_arm(strat, lambda r: rnd.choice(universe))
        rand_stats.append(compute_stats(arm))
    avg = {"arm": f"random (×{args.seeds})",
           "trades": round(sum(s["trades"] for s in rand_stats) / len(rand_stats)),
           "total_pnl": sum(s["total_pnl"] for s in rand_stats) / len(rand_stats),
           "win_rate": sum(s["win_rate"] for s in rand_stats) / len(rand_stats),
           "avg_win": 0, "avg_loss": 0,
           "profit_factor": sum(s["profit_factor"] for s in rand_stats) / len(rand_stats),
           "max_dd": sum(s["max_dd"] for s in rand_stats) / len(rand_stats),
           "sharpe": sum(s["sharpe"] for s in rand_stats) / len(rand_stats)}
    results.append(avg); print(fmt(avg))

    # Verdict
    strat_s = results[0]
    print(f"\n{'='*70}")
    beta_best = max(r["total_pnl"] for r in results if "beta" in r["arm"])
    rand_pnl  = avg["total_pnl"]
    print("VERDICT")
    print(f"  Strategy P&L      : ${strat_s['total_pnl']:,.0f}  (Sharpe {strat_s['sharpe']:.2f})")
    print(f"  Best index (beta) : ${beta_best:,.0f}")
    print(f"  Random names      : ${rand_pnl:,.0f}")
    if strat_s["total_pnl"] > beta_best and strat_s["total_pnl"] > rand_pnl:
        print("  → Strategy beats both beta and random selection. Possible alpha "
              "in name-picking (verify across regimes).")
    else:
        print("  → Strategy does NOT clearly beat beta/random. The signal may add "
              "little over just being long the market in this regime.")


if __name__ == "__main__":
    main()
