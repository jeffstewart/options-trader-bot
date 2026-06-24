#!/bin/bash
# Overnight rigorous eval: finalists v3 (clean catalyst-strip) + v4 (CoT+strip) vs v1 incumbent,
# on FULL bull (6285) + 2022-bear out-of-regime (2000), bootstrap + jackknife (eval_unified_full).
# v3+v1 FIRST (faster) so v1-vs-v3 lands by morning; v4 (slow CoT) completes Saturday.
# score_unified is resumable → screening's 600 reused. Results → overnight_eval.log.
cd /Users/jeff/Claude/Trader
PY=".venv/bin/python"
log(){ echo "=== $(date +%H:%M:%S) $* ==="; }
score_bear(){ UNI_PROMPT=$1 UNI_CACHE=bear_dual_cache.json UNI_END=2022-06-30 UNI_DAYS=90 \
  UNI_REGIME=0 UNI_SAMPLE=2000 USE_YAHOO_BARS=1 $PY -u score_unified.py; }
eval_bull(){ UNI_PROMPT=$1 UNI_LABEL=bull UNI_CACHE=dual_score_cache.json USE_YAHOO_BARS=1 $PY eval_unified_full.py; }
eval_bear(){ UNI_PROMPT=$1 UNI_LABEL=bear-OOR UNI_CACHE=bear_dual_cache.json UNI_END=2022-06-30 UNI_DAYS=90 \
  USE_YAHOO_BARS=1 $PY eval_unified_full.py; }

log "START overnight"
# Phase A: v3 + v1 — finishes by morning
log "score v3 bull (full)";  UNI_PROMPT=unified_v3 USE_YAHOO_BARS=1 $PY -u score_unified.py
log "score v3 bear";         score_bear unified_v3
log "score v1 bear";         score_bear unified_v1
log "EVAL v1 + v3"
{ echo "########## PHASE A: v1 vs v3  $(date) ##########"
  for P in unified_v1 unified_v3; do eval_bull $P; eval_bear $P; done
} >> overnight_eval.log 2>&1
log "PHASE A DONE (v1 vs v3 ready)"
# Phase B: v4 — completes Saturday
log "score v4 bull (full)";  UNI_PROMPT=unified_v4 USE_YAHOO_BARS=1 $PY -u score_unified.py
log "score v4 bear";         score_bear unified_v4
log "EVAL v4"
{ echo "########## PHASE B: v4  $(date) ##########"
  eval_bull unified_v4; eval_bear unified_v4
} >> overnight_eval.log 2>&1
log "ALL OVERNIGHT DONE"
