#!/bin/bash
# Shim: the crontab still calls this root path (pre-2026-06-24-restructure location). The real EOD
# script now lives in research/. Keep this delegating shim so the existing cron keeps working without
# editing the crontab; the canonical script is research/run_eod.sh.
exec /Users/jeff/Claude/Trader/research/run_eod.sh "$@"
