#!/bin/bash
# Overnight prompt refinement run — after market close (Ollama free).
# Scores the full dev set (1000 articles) for:
#   • Top existing candidates (materiality on both models) — widen to get tight CIs
#   • New prompts: binary_gate, materiality_fewshot, surprise_score
# All on llama3.2 (fast, 2GB, no swap) first, then llama3.1:8b for the best survivor.
# Results graded by grade_scores.py at 1/3/5d horizons.
cd /Users/jeff/Claude/Trader

ALL_PROMPTS="materiality,binary_gate,materiality_fewshot,surprise_score,lotto_swing"

echo "$(date '+%H:%M:%S') Starting prompt refinement sweep (llama3.2)…"
USE_YAHOO_BARS=1 .venv/bin/python prompt_exp.py \
  --provider ollama --model llama3.2 \
  --prompts $ALL_PROMPTS \
  --limit 1000 --delay 0 \
  > prompt_exp_llama32_v2.log 2>&1
echo "$(date '+%H:%M:%S') llama3.2 done"

echo "$(date '+%H:%M:%S') Grading llama3.2 — stock signals (fwd=1d, win=5%)…"
USE_YAHOO_BARS=1 .venv/bin/python grade_scores.py \
  --fwd 1 3 5 --win 5 --model llama3.2 \
  --prompt materiality,binary_gate,materiality_fewshot,surprise_score \
  > grade_llama32_v2.out 2>&1

echo "$(date '+%H:%M:%S') Grading llama3.2 — lotto signals (fwd=1d, win=15%)…"
USE_YAHOO_BARS=1 .venv/bin/python grade_scores.py \
  --fwd 1 --win 15 --model llama3.2 \
  --prompt lotto_swing,surprise_score,materiality \
  >> grade_llama32_v2.out 2>&1
echo "$(date '+%H:%M:%S') llama3.2 grading done"

echo "$(date '+%H:%M:%S') Running llama3.1:8b on all prompts…"
USE_YAHOO_BARS=1 .venv/bin/python prompt_exp.py \
  --provider ollama --model llama3.1:8b \
  --prompts $ALL_PROMPTS \
  --limit 1000 --delay 0 \
  > prompt_exp_llama318b_v2.log 2>&1
ollama stop llama3.1:8b
echo "$(date '+%H:%M:%S') llama3.1:8b done"

echo "$(date '+%H:%M:%S') Grading llama3.1:8b — stock + lotto…"
USE_YAHOO_BARS=1 .venv/bin/python grade_scores.py \
  --fwd 1 3 5 --win 5 --model llama3.1:8b \
  --prompt materiality,binary_gate,materiality_fewshot,surprise_score \
  > grade_llama318b_v2.out 2>&1
USE_YAHOO_BARS=1 .venv/bin/python grade_scores.py \
  --fwd 1 --win 15 --model llama3.1:8b \
  --prompt lotto_swing,surprise_score,materiality \
  >> grade_llama318b_v2.out 2>&1

echo "$(date '+%H:%M:%S') ✅ PROMPT REFINEMENT COMPLETE"
echo "Check: grade_llama32_v2.out  grade_llama318b_v2.out"
echo "★ cells = cleared IC CI>0 AND lift CI>1 → candidates for live deployment"
