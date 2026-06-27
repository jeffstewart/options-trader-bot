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

WDLOG=$ROOT/logs/watchdog.log
wlog() { echo "[$(date '+%F %T')] $1" >> $WDLOG; }

is_up() { local p=$(cat $ROOT/data/$1.pid 2>/dev/null); [[ -n "$p" ]] && kill -0 "$p" 2>/dev/null; }

# Restart a daemon if its pidfile process is gone. $1=pidfile-name $2=start-fn
ensure() { if ! is_up "$1"; then wlog "$1 DOWN → restarting"; $2 || wlog "$1 restart FAILED"; fi; }

# Insurance against SYSTEM sleep (would freeze trading + data capture) in case the pmset 'sleep 0'
# setting ever gets reset. Uses -i only (idle SYSTEM sleep) — NOT -d/-u, so the display still turns
# off and the lock screen still engages. Persists past any Claude session.
ensure_caffeinate() { pgrep -x caffeinate >/dev/null 2>&1 || { nohup caffeinate -i >/dev/null 2>&1 & wlog "started caffeinate -i (system-sleep insurance; display/lock unaffected)"; }; }

# Copy-truncate growing logs to the last 50k lines once they exceed 150MB (Python append-mode → safe).
rotate_logs() {
  for lf in bot dashboard ollama groq_backtest; do
    local f=$ROOT/logs/$lf.log
    [[ -f $f ]] || continue
    local sz=$(stat -f%z "$f" 2>/dev/null || echo 0)
    if (( sz > 157286400 )); then
      tail -n 50000 "$f" > "$f.tmp" 2>/dev/null && cp "$f.tmp" "$f" && rm -f "$f.tmp" && wlog "rotated $lf.log (was $((sz/1048576))MB)"
    fi
  done
}

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
  restart-dash) pkill -f 'core/dashboard.py' 2>/dev/null || true; sleep 2; start_dash; echo "dashboard restarted" ;;
  backup)
    # Off-machine backup to iCloud Drive (no remote / Time Machine). Code as a git bundle (full history,
    # ~5MB), data/ + eod_reports mirrored. Run on demand or daily via run_eod.sh. ~67MB total.
    BK="$HOME/Library/Mobile Documents/com~apple~CloudDocs/TraderBackup"
    mkdir -p "$BK/data" "$BK/eod_reports"
    git -C $ROOT bundle create "$BK/trader-code.bundle" --all >/dev/null 2>&1 && bundle_ok=1 || bundle_ok=0
    rsync -a --delete --exclude '*.pid' $ROOT/data/ "$BK/data/" 2>/dev/null
    rsync -a $ROOT/eod_reports/ "$BK/eod_reports/" 2>/dev/null
    date '+%F %T' > "$BK/LAST_BACKUP.txt"
    wlog "backup → iCloud (code bundle=$bundle_ok, data+eod mirrored)"
    echo "backed up to $BK (code bundle=$bundle_ok)" ;;
  watchdog)
    # Run every few minutes (LaunchAgent / cron) for unattended operation: restart any down daemon,
    # keep the Mac awake, and cap log growth. Idempotent — only acts when something is wrong.
    ensure bot start_bot
    ensure dashboard start_dash
    ensure resume_after_reset start_sched
    ensure_caffeinate
    rotate_logs ;;
  report) ( cd $ROOT/data && USE_YAHOO_BARS=1 $PY $ROOT/research/status.py ) ;;
  test) $PY -m pytest ;;
  *) echo "usage: ./manage.sh {start|sched|stop|restart-bot|restart-dash|watchdog|backup|report|status|test}" ;;
esac
