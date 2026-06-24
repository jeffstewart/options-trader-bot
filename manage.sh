#!/bin/zsh
# manage.sh — start / stop / status the trader daemons.
# Post-2026-06-24 restructure: code in core/+research/, data in data/, logs in logs/.
# Daemons run via core/spawn_daemon.py which sets CWD=data/ (so bare data-file refs resolve there)
# and detaches to ppid 1. Imports resolve via .venv/.../site-packages/_trader_paths.pth (core/+research/).
set -e
ROOT=/Users/jeff/Claude/Trader
PY=$ROOT/.venv/bin/python
spawn() { $PY $ROOT/core/spawn_daemon.py "$1" "$2" "${@:3}"; }   # PIDFILE LOGFILE CMD...

start_bot()   { spawn $ROOT/data/bot.pid                $ROOT/logs/bot.log                $PY -u $ROOT/core/bot.py; }
start_dash()  { spawn $ROOT/data/dashboard.pid          $ROOT/logs/dashboard.log          $PY -u $ROOT/core/dashboard.py; }
start_sched() { spawn $ROOT/data/resume_after_reset.pid $ROOT/logs/resume_after_reset.log $PY -u $ROOT/research/resume_after_reset.py; }

case "$1" in
  start)
    start_bot; start_dash; sleep 2
    echo "started bot + dashboard (scheduler: ./manage.sh sched)"; "$0" status ;;
  sched) start_sched; echo "started backtest scheduler" ;;
  stop)
    pkill -f 'core/bot.py' 2>/dev/null || true
    pkill -f 'core/dashboard.py' 2>/dev/null || true
    pkill -f 'research/resume_after_reset.py' 2>/dev/null || true
    pkill -f 'research/groq_vs_ollama_backtest.py' 2>/dev/null || true
    echo "stopped all daemons" ;;
  restart-bot) pkill -f 'core/bot.py' 2>/dev/null || true; sleep 2; start_bot; echo "bot restarted" ;;
  status)
    for n in bot dashboard resume_after_reset groq_backtest; do
      p=$(cat $ROOT/data/$n.pid 2>/dev/null)
      if [[ -n "$p" ]] && kill -0 "$p" 2>/dev/null; then echo "  $n: UP (pid $p)"; else echo "  $n: down"; fi
    done ;;
  test) $PY -m pytest ;;
  *) echo "usage: ./manage.sh {start|sched|stop|restart-bot|status|test}" ;;
esac
