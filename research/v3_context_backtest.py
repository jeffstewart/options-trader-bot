"""
v3_context_backtest.py — test harness to evaluate context-aware scoring on historical grid data.
Features:
  - Replays historical signals from data/contract_grid_snapshots.csv.
  - Queries point-in-time prior news from SQLite archive (strictly before signal_ts).
  - Evaluates different lookback windows (e.g. 0d isolated vs. 1d vs. 3d vs. 7d).
  - Pluggable across scorers (local Ollama llama3.2, Gemini, Anthropic, Kimi).
  - Evaluates echo/recap detection rate and trade P&L under Delta 0.50 + fixed 30m exit.

Usage:
  python3 research/v3_context_backtest.py --sample 30 --lookback 7 --provider ollama --model llama3.2
"""

import argparse
import logging
import os
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import pandas as pd

# Add repo root to path
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from v3 import config as cfg
from v3 import news_db
from v3 import scoring

logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(levelname)-8s  %(message)s")
log = logging.getLogger("v3_backtest")


def load_unique_signals(grid_csv_path: str, sample_n: int = 50, random_state: int = 42) -> pd.DataFrame:
    """Load unique historical signals with first-snapshot option data and price paths."""
    log.info("Reading signals from %s ...", grid_csv_path)
    usecols = [
        "signal_id", "signal_ts", "ticker", "headline", "minutes_since_signal",
        "symbol", "strike", "expiry", "oi_at_signal", "spread_pct", "delta", "ask", "bid"
    ]
    df = pd.read_csv(grid_csv_path, usecols=usecols, low_memory=False, parse_dates=["signal_ts"])

    # First snapshot per (signal_id, symbol)
    entry_rows = df[df["minutes_since_signal"] <= 1.0].copy()

    # Filter to candidate contracts meeting Delta ~0.50 and liquidity
    valid = entry_rows[
        (entry_rows["oi_at_signal"].fillna(0) >= 10) &
        (entry_rows["spread_pct"] <= 0.08) &
        (entry_rows["delta"].notna())
    ].copy()
    valid["delta_dist"] = (valid["delta"] - 0.50).abs()

    # Pick best contract per signal
    best_picks = valid.loc[valid.groupby("signal_id")["delta_dist"].idxmin()]

    # Simulate 30-minute timed exit return for each picked contract
    picked_pairs = set(zip(best_picks["signal_id"], best_picks["symbol"]))
    paths = df[df.set_index(["signal_id", "symbol"]).index.isin(picked_pairs)].copy()

    # Compute entry price (first ask) and 30m exit price (first bid >= 30m, else last bid)
    exits = []
    for (sig_id, sym), grp in paths.groupby(["signal_id", "symbol"]):
        grp = grp.sort_values("minutes_since_signal")
        entry_ask = grp["ask"].iloc[0]
        if entry_ask <= 0:
            continue
        hit = grp[grp["minutes_since_signal"] >= 30.0]
        exit_bid = hit["bid"].iloc[0] if len(hit) > 0 else grp["bid"].iloc[-1]
        pnl_pct = (exit_bid / entry_ask) - 1.0
        exits.append({"signal_id": sig_id, "symbol": sym, "entry_ask": entry_ask,
                      "exit_bid": exit_bid, "pnl_pct": pnl_pct})

    exit_df = pd.DataFrame(exits)
    signals = best_picks.merge(exit_df, on=["signal_id", "symbol"])

    log.info("Identified %d qualified signals with valid option trajectories.", len(signals))

    if sample_n > 0 and len(signals) > sample_n:
        signals = signals.sample(n=sample_n, random_state=random_state).sort_values("signal_ts")
        log.info("Sampled %d signals for evaluation.", len(signals))

    return signals


def run_evaluation(signals: pd.DataFrame, lookback_days: int,
                   provider: str, model: str, db_path: str = None) -> pd.DataFrame:
    """Score each signal with given lookback window and record metrics."""
    results = []
    log.info("Running evaluation: Provider=%s, Model=%s, Lookback=%d days, N=%d ...",
             provider, model, lookback_days, len(signals))

    for idx, (_, row) in enumerate(signals.iterrows(), 1):
        sig_id = row["signal_id"]
        ticker = row["ticker"]
        headline = row["headline"]
        sig_ts = row["signal_ts"]
        pnl = row["pnl_pct"]

        # Call context-aware scorer
        score = scoring.score_article(
            headline=headline,
            body="",
            primary_ticker=ticker,
            article_ts=sig_ts,
            lookback_days=lookback_days,
            provider=provider,
            model=model,
            db_path=db_path,
        )

        if not score:
            log.warning("[%d/%d] Scoring failed for %s", idx, len(signals), ticker)
            continue

        sentiment = score.get("sentiment", "neutral")
        conf = score.get("confidence", 0.0)
        mag = score.get("magnitude", 0.0)
        is_echo = score.get("is_stale_echo", False)
        reasoning = score.get("reasoning", "")

        # Passes gate: bullish & conf >= 0.70 & mag >= 0.35 & not echo
        passes_gate = (sentiment == "bullish" and conf >= 0.70 and mag >= 0.35 and not is_echo)

        log.info("[%d/%d] %s: sent=%s mag=%.2f conf=%.2f echo=%s gate=%s -> P&L=%+.1f%%",
                 idx, len(signals), ticker, sentiment, mag, conf, is_echo, passes_gate, pnl * 100)

        results.append({
            "signal_id": sig_id,
            "signal_ts": sig_ts,
            "ticker": ticker,
            "headline": headline,
            "lookback_days": lookback_days,
            "sentiment": sentiment,
            "confidence": conf,
            "magnitude": mag,
            "is_stale_echo": is_echo,
            "passes_gate": passes_gate,
            "pnl_pct": pnl,
            "reasoning": reasoning,
        })

    return pd.DataFrame(results)


def summarize_results(df: pd.DataFrame, label: str = "") -> dict:
    """Compute summary statistics for an evaluation run."""
    n_total = len(df)
    if n_total == 0:
        return {}

    gated = df[df["passes_gate"]]
    n_gated = len(gated)

    base_win_rate = (df["pnl_pct"] > 0).mean()
    base_avg_pnl = df["pnl_pct"].mean()

    gated_win_rate = (gated["pnl_pct"] > 0).mean() if n_gated > 0 else 0.0
    gated_avg_pnl = gated["pnl_pct"].mean() if n_gated > 0 else 0.0
    echo_count = df["is_stale_echo"].sum()

    print("\n" + "=" * 60)
    print(f"EVALUATION SUMMARY: {label}")
    print(f"Total signals evaluated: {n_total}")
    print(f"Stale echo/recap rejected count: {echo_count} ({echo_count/n_total:.1%})")
    print(f"Baseline (All signals): Win Rate = {base_win_rate:.1%}, Avg P&L = {base_avg_pnl:.2%}")
    print(f"Gated signals kept:     {n_gated}/{n_total} ({n_gated/n_total:.1%})")
    print(f"Gated Performance:      Win Rate = {gated_win_rate:.1%}, Avg P&L = {gated_avg_pnl:.2%}")
    print("=" * 60 + "\n")

    return {
        "label": label,
        "n_total": n_total,
        "n_gated": n_gated,
        "echo_count": int(echo_count),
        "base_win_rate": base_win_rate,
        "base_avg_pnl": base_avg_pnl,
        "gated_win_rate": gated_win_rate,
        "gated_avg_pnl": gated_avg_pnl,
    }


def main():
    parser = argparse.ArgumentParser(description="v3 Context-Aware News Backtest Harness")
    parser.add_argument("--grid-csv", default="data/contract_grid_snapshots.csv", help="Path to contract grid CSV")
    parser.add_argument("--db-path", default=str(cfg.NEWS_DB_PATH), help="Path to SQLite news database")
    parser.add_argument("--provider", default=cfg.SCORER_PROVIDER, help="LLM provider: ollama, gemini, anthropic, moonshot")
    parser.add_argument("--model", default=cfg.SCORER_MODEL, help="LLM model name")
    parser.add_argument("--sample", type=int, default=25, help="Number of signals to evaluate (0 = all)")
    parser.add_argument("--lookback", type=int, default=7, help="Lookback window in days (default: 7)")
    parser.add_argument("--compare-baseline", action="store_true", help="Also run 0-day baseline for comparison")
    parser.add_argument("--backfill-days", type=int, default=0, help="Backfill N days of news before running if > 0")
    args = parser.parse_args()

    news_db.init_db(args.db_path)

    # Optional backfill
    if args.backfill_days > 0:
        log.info("Backfilling %d days of news...", args.backfill_days)
        news_db.backfill_news_from_alpaca(cfg.ALPACA_KEY, cfg.ALPACA_SECRET, days=args.backfill_days, db_path=args.db_path)

    signals = load_unique_signals(args.grid_csv, sample_n=args.sample)
    if len(signals) == 0:
        log.error("No valid signals found.")
        return

    # Run evaluation with configured lookback
    eval_df = run_evaluation(signals, lookback_days=args.lookback,
                             provider=args.provider, model=args.model, db_path=args.db_path)
    summarize_results(eval_df, label=f"{args.provider}/{args.model} ({args.lookback}d lookback)")

    safe_model = args.model.replace("/", "_").replace(":", "_")
    out_csv = f"research/v3_backtest_{args.provider}_{safe_model}_{args.lookback}d.csv"
    eval_df.to_csv(out_csv, index=False)
    log.info("Saved detailed results to %s", out_csv)

    # Optional 0-day baseline comparison
    if args.compare_baseline:
        log.info("Running 0-day (isolated article) baseline comparison ...")
        base_df = run_evaluation(signals, lookback_days=0,
                                 provider=args.provider, model=args.model, db_path=args.db_path)
        summarize_results(base_df, label=f"{args.provider}/{args.model} (0d isolated baseline)")
        base_csv = f"research/v3_backtest_{args.provider}_{safe_model}_0d.csv"
        base_df.to_csv(base_csv, index=False)
        log.info("Saved baseline results to %s", base_csv)


if __name__ == "__main__":
    main()

