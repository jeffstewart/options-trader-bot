#!/bin/bash
# End-of-day trader report — invoked by cron weekdays ~15min after the US close.
# Post-2026-06-24 restructure: code in research/, data in data/. eod_report.py opens bare data-file
# refs (trades.csv, gemini_decisions.csv, …) so it MUST run with CWD=data/; imports (config, yahoo_data)
# resolve via the venv _trader_paths.pth. Reports are written under eod_reports/ at the repo root.
ROOT=/Users/jeff/Claude/Trader
PY=$ROOT/.venv/bin/python
mkdir -p "$ROOT/eod_reports"
OUT="$ROOT/eod_reports/eod_$(date +%Y-%m-%d).txt"
( cd "$ROOT/data" && "$PY" "$ROOT/research/eod_report.py" ) > "$OUT" 2>&1
echo "" >> "$ROOT/eod_reports/eod_history.log"
cat "$OUT" >> "$ROOT/eod_reports/eod_history.log"
# Accumulate Gemini unified_v1 scores toward the 1000-pool comparison (cache-aware; stops at the
# daily free cap, resumes tomorrow). Finishes coverage over a few days, then just re-prints the compare.
( cd "$ROOT/data" && UNI_SAMPLE=1000 USE_YAHOO_BARS=1 "$PY" -u "$ROOT/research/gemini_batch_unified.py" ) \
    >> "$ROOT/eod_reports/gemini_unified_daily.log" 2>&1
# Daily off-machine backup to iCloud (code bundle + data + eod_reports). No remote / Time Machine.
"$ROOT/manage.sh" backup >> "$ROOT/logs/watchdog.log" 2>&1
