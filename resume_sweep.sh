#!/bin/bash
# Resume the local prompt sweep for the two models not yet done (llama3.2 finished
# pre-reboot). Runs sequentially, unloads each model before the next. qwen2.5:7b
# is already pulled. Keeps the bot up (idle overnight → light; market closed).
cd /Users/jeff/Claude/Trader

echo "$(date '+%H:%M:%S') sweeping llama3.1:8b"
USE_YAHOO_BARS=1 .venv/bin/python prompt_exp.py --provider ollama --model llama3.1:8b \
  --prompts baseline,direct_return,materiality,catalyst_typed --limit 250 --delay 0 \
  > prompt_exp_llama31_8b.log 2>&1
ollama stop llama3.1:8b

echo "$(date '+%H:%M:%S') sweeping qwen2.5:7b (different family)"
USE_YAHOO_BARS=1 .venv/bin/python prompt_exp.py --provider ollama --model qwen2.5:7b \
  --prompts baseline,direct_return,materiality,catalyst_typed --limit 250 --delay 0 \
  > prompt_exp_qwen25_7b.log 2>&1
ollama stop qwen2.5:7b

echo "$(date '+%H:%M:%S') RESUME SWEEP COMPLETE"
