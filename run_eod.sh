#!/bin/bash
# End-of-day trader report — invoked by cron weekdays ~15min after the US close.
cd /Users/jeff/Claude/Trader || exit 1
mkdir -p eod_reports
OUT="eod_reports/eod_$(date +%Y-%m-%d).txt"
/Users/jeff/Claude/Trader/.venv/bin/python eod_report.py > "$OUT" 2>&1
echo "" >> eod_reports/eod_history.log
cat "$OUT" >> eod_reports/eod_history.log
# Accumulate Gemini unified_v1 scores toward the 1000-pool comparison (cache-aware; stops at the
# daily free cap, resumes tomorrow). Finishes coverage over a few days, then just re-prints the compare.
UNI_SAMPLE=1000 USE_YAHOO_BARS=1 /Users/jeff/Claude/Trader/.venv/bin/python -u gemini_batch_unified.py >> eod_reports/gemini_unified_daily.log 2>&1
