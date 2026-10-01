"""
research/rescore_signals.py

Batch rescoring utility for historical signals in contract_grid_snapshots.csv.
Uses the v3 context-aware prompt (v3.scoring) with point-in-time ticker news history
from v3/data/news.db to determine:
  - v3_sentiment: 'bullish' | 'bearish' | 'neutral'
  - v3_confidence: float [0, 1]
  - v3_magnitude: float [0, 1]
  - v3_is_stale_echo: bool (True if the headline merely echoes/recaps prior 1-7 day news)
  - v3_reasoning: explanation from LLM

Results are saved incrementally to an on-disk CSV so runs can be safely interrupted
and resumed without re-scoring already completed signals.
"""

import argparse
import csv
import logging
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd

# Add project root to path
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from v3.scoring import score_article
from research.contract_strategy_tuner import load_entry_snapshot_and_stats

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)

DEFAULT_CSV = "data/contract_grid_snapshots.csv"
DEFAULT_OUTPUT = "data/v3_rescored_signals.csv"
DEFAULT_NEWS_DB = "v3/data/news.db"


def load_cached_signals(output_path: str) -> Dict[str, dict]:
    """Load previously scored signals from output CSV if it exists."""
    cached = {}
    if os.path.exists(output_path):
        try:
            df = pd.read_csv(output_path)
            for _, row in df.iterrows():
                sid = str(row["signal_id"])
                cached[sid] = row.to_dict()
            log.info("Loaded %d already-scored signals from %s", len(cached), output_path)
        except Exception as e:
            log.warning("Could not read existing cache from %s: %s", output_path, e)
    return cached


def append_result(output_path: str, record: dict, fieldnames: List[str]):
    """Append a single scored record to output CSV and flush immediately."""
    file_exists = os.path.exists(output_path) and os.path.getsize(output_path) > 0
    with open(output_path, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if not file_exists:
            writer.writeheader()
        writer.writerow(record)
        f.flush()


def score_single_signal(sig: dict, provider: str, model: str, news_db_path: str) -> dict:
    """Evaluate one signal with the v3 context-aware prompt."""
    sid = sig["signal_id"]
    ticker = str(sig.get("ticker", "")).strip()
    headline = str(sig.get("headline", "")).strip()
    sig_ts = sig.get("signal_ts")

    if isinstance(sig_ts, str):
        try:
            article_ts = datetime.fromisoformat(sig_ts.replace("Z", "+00:00"))
        except Exception:
            article_ts = datetime.now(timezone.utc)
    elif isinstance(sig_ts, datetime):
        article_ts = sig_ts
    else:
        article_ts = datetime.now(timezone.utc)

    t0 = time.time()
    try:
        score = score_article(
            headline=headline,
            body="",
            primary_ticker=ticker,
            article_ts=article_ts,
            provider=provider,
            model=model,
            db_path=news_db_path,
        )
    except Exception as e:
        log.warning("Scoring failed for signal %s (%s): %s", sid, ticker, e)
        score = None

    elapsed = time.time() - t0

    if score is None:
        return {
            "signal_id": sid,
            "signal_ts": str(sig_ts),
            "ticker": ticker,
            "headline": headline,
            "v3_sentiment": "neutral",
            "v3_confidence": 0.0,
            "v3_magnitude": 0.0,
            "v3_is_stale_echo": False,
            "v3_reasoning": "Error or timeout during scoring",
            "score_duration_sec": round(elapsed, 2),
        }

    return {
        "signal_id": sid,
        "signal_ts": str(sig_ts),
        "ticker": ticker,
        "headline": headline,
        "v3_sentiment": str(score.get("sentiment", "neutral")).lower(),
        "v3_confidence": round(float(score.get("confidence", 0.0)), 4),
        "v3_magnitude": round(float(score.get("magnitude", 0.0)), 4),
        "v3_is_stale_echo": bool(score.get("is_stale_echo", False)),
        "v3_reasoning": str(score.get("reasoning", "")),
        "score_duration_sec": round(elapsed, 2),
    }


def main():
    parser = argparse.ArgumentParser(description="Batch rescore contract grid signals with historical news context")
    parser.add_argument("--csv", default=DEFAULT_CSV, help="Path to contract_grid_snapshots.csv")
    parser.add_argument("--output", default=DEFAULT_OUTPUT, help="Path to write v3_rescored_signals.csv")
    parser.add_argument("--news-db", default=DEFAULT_NEWS_DB, help="Path to news.db")
    parser.add_argument("--provider", default="ollama", help="LLM provider: ollama, gemini, anthropic")
    parser.add_argument("--model", default="llama3.2", help="Model name (e.g. llama3.2, gemini-2.5-flash)")
    parser.add_argument("--workers", type=int, default=2, help="Concurrent worker threads (default 2)")
    parser.add_argument("--max-signals", type=int, default=None, help="Limit number of signals to score (for testing)")
    args = parser.parse_args()

    log.info("Loading entry signals from %s ...", args.csv)
    entry_df, total_rows, _ = load_entry_snapshot_and_stats(args.csv)

    # Extract one record per signal_id
    info_cols = ["signal_id", "signal_ts", "ticker", "headline", "magnitude", "confidence", "catalyst"]
    available_cols = [c for c in info_cols if c in entry_df.columns]
    signals_df = entry_df.groupby("signal_id")[available_cols].first().reset_index(drop=True)
    all_signals = signals_df.to_dict("records")
    log.info("Found %d unique signals in %s", len(all_signals), args.csv)

    cached = load_cached_signals(args.output)
    to_score = [s for s in all_signals if s["signal_id"] not in cached]

    if args.max_signals is not None and args.max_signals > 0:
        to_score = to_score[:args.max_signals]
        log.info("Limiting to %d signals via --max-signals", len(to_score))

    log.info("Signals to score: %d (already cached: %d)", len(to_score), len(cached))
    if not to_score:
        log.info("All signals are already scored in %s. Nothing to do!", args.output)
        return

    fieldnames = [
        "signal_id", "signal_ts", "ticker", "headline",
        "v3_sentiment", "v3_confidence", "v3_magnitude", "v3_is_stale_echo",
        "v3_reasoning", "score_duration_sec"
    ]

    scored_count = 0
    t_start = time.time()

    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        future_map = {
            executor.submit(score_single_signal, sig, args.provider, args.model, args.news_db): sig
            for sig in to_score
        }

        for future in as_completed(future_map):
            res = future.result()
            append_result(args.output, res, fieldnames)
            scored_count += 1

            echo_flag = " [STALE ECHO]" if res["v3_is_stale_echo"] else ""
            log.info(
                "[%d/%d] %s (%s): %s, mag=%.2f, conf=%.2f%s (%.1fs)",
                scored_count, len(to_score), res["ticker"], res["signal_id"][:12],
                res["v3_sentiment"], res["v3_magnitude"], res["v3_confidence"],
                echo_flag, res["score_duration_sec"]
            )

    elapsed_tot = time.time() - t_start
    log.info("Successfully finished scoring %d signals in %.1f seconds (avg %.2fs/signal).",
             scored_count, elapsed_tot, elapsed_tot / max(1, scored_count))
    log.info("Results saved to: %s", args.output)


if __name__ == "__main__":
    main()
