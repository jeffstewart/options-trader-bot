#!/bin/bash
# Screen the unified prompt variants on a shared sample (bull window) → unified_scores.json.
# v1 is already fully scored (6285); this scores v2/v3/v4 on the same 600-article sample for
# an apples-to-apples directional rank before the overnight full run.
cd /Users/jeff/Claude/Trader
for p in unified_v2 unified_v3 unified_v4; do
  echo "=== scoring $p (sample 600) $(date +%H:%M:%S) ==="
  UNI_PROMPT=$p UNI_SAMPLE=600 USE_YAHOO_BARS=1 .venv/bin/python -u score_unified.py
done
echo "=== SCREEN DONE $(date +%H:%M:%S) ==="
