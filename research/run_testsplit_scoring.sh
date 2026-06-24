#!/bin/bash
# Score the HELD-OUT TEST split with materiality_fewshot on Groq, to confirm the
# dev-split binary-gate finding (2026-06-07). Sequential (qwen then gpt-oss) to avoid
# concurrent Yahoo-cache corruption. gpt-oss runs only after qwen completes; if Groq
# daily quota is exhausted by then it will error into its own log, leaving qwen intact.
cd /Users/jeff/Claude/Trader

echo "$(date '+%Y-%m-%d %H:%M') starting qwen3-32b / materiality_fewshot (TEST split)"
USE_YAHOO_BARS=1 .venv/bin/python prompt_exp.py \
    --provider groq --model qwen/qwen3-32b \
    --prompts materiality_fewshot \
    --split test --limit 1000 --delay 2 \
    > prompt_exp_groq_qwen32b_test.log 2>&1
echo "$(date '+%Y-%m-%d %H:%M') qwen3-32b TEST done (exit $?)"

echo "$(date '+%Y-%m-%d %H:%M') starting gpt-oss-20b / materiality_fewshot (TEST split)"
USE_YAHOO_BARS=1 .venv/bin/python prompt_exp.py \
    --provider groq --model openai/gpt-oss-20b \
    --prompts materiality_fewshot \
    --split test --limit 1000 --delay 2 \
    > prompt_exp_groq_gpt20b_test.log 2>&1
echo "$(date '+%Y-%m-%d %H:%M') gpt-oss-20b TEST done (exit $?)"
echo "TEST-SPLIT SCORING COMPLETE"
