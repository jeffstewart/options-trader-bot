#!/usr/bin/env bash
# run_tuning_queue.sh — sequential queue (no concurrent Yahoo writers).
#   1. wait for the running bull-melt-up tune, save its results
#   2. 2022-bear-window tune (Yahoo)
#   3. cross-regime parameter ranking
#   4. exit-rule comparison (cross-regime)
cd "$(dirname "$0")"
PY=".venv/bin/python"
echo "=== QUEUE START $(date) ==="

echo "[1/4] waiting for bull-melt-up tune…"
while pgrep -f 'tune_v2.py --mode bull --days 180' >/dev/null; do sleep 30; done
cp tune_v2_bull_results.csv tune_v2_bull_meltup_yahoo.csv 2>/dev/null
cp tune_v2_bull_holdout.csv  tune_v2_bull_meltup_yahoo_holdout.csv 2>/dev/null
echo "      bull-melt-up saved."

echo "[2/4] 2022-bear-window tune (Yahoo) $(date)…"
USE_YAHOO_BARS=1 $PY tune_v2.py --mode bull --end-date 2022-06-30 --days 90 \
    --cache-file bear_dual_cache.json >> tune_bull_2022.log 2>&1
cp tune_v2_bull_results.csv tune_v2_bull_2022_yahoo.csv 2>/dev/null
cp tune_v2_bull_holdout.csv  tune_v2_bull_2022_yahoo_holdout.csv 2>/dev/null

echo "[3/4] cross-regime ranking $(date)…"
$PY cross_regime_rank.py > cross_regime_rank.out 2>&1

echo "[4/4] exit-rule comparison $(date)…"
USE_YAHOO_BARS=1 $PY exit_rules_test.py > exit_rules_test.out 2>&1

echo "=== QUEUE DONE $(date) ==="
