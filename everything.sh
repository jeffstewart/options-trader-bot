#!/bin/zsh
# everything.sh — single entry point to start/stop/status the FULL stack:
#   ollama (native, host)  ·  v1 bot+dashboard (native, host, via manage.sh)
#   v2 bot (docker)        ·  v2 dashboard (native, host, NOT containerized — see v2/docker-compose.yml)
#
# Built 2026-08-03 after both bots + ollama got shut off over a weekend to free RAM and had to be
# restarted piecemeal by hand. This wraps that into one command so nothing gets left half-up.
#
# Usage: ./everything.sh {start|stop|restart|status}
set -e
ROOT=/Users/jeff/Claude/Trader
PY=$ROOT/.venv/bin/python
OLLAMA_BIN=/opt/homebrew/bin/ollama

is_up() { local p=$(cat "$1" 2>/dev/null); [[ -n "$p" ]] && kill -0 "$p" 2>/dev/null; }

# Send SIGTERM, wait up to 10s for the pidfile's process to actually exit, SIGKILL if it doesn't.
# Plain `pkill` (what manage.sh uses) fires-and-forgets; this confirms the process is really gone
# before the script reports it stopped and moves to the next component.
wait_for_death() {
  local pidfile=$1 name=$2
  is_up "$pidfile" || { echo "  $name: already down"; return; }
  local pid=$(cat "$pidfile")
  kill "$pid" 2>/dev/null || true
  for _ in $(seq 1 10); do
    kill -0 "$pid" 2>/dev/null || { rm -f "$pidfile"; echo "  $name: stopped"; return; }
    sleep 1
  done
  echo "  $name: still up after 10s SIGTERM, sending SIGKILL"
  kill -9 "$pid" 2>/dev/null || true
  rm -f "$pidfile"
  echo "  $name: force-stopped"
}

start_docker() {
  # Docker daemon here is Colima (not Docker Desktop — switched off 2026-08-03 to kill its ~2GB
  # idle VM overhead). Colima does NOT survive a machine restart, so after a reboot `docker compose
  # up` fails outright unless something brings it back first.
  if docker info >/dev/null 2>&1; then echo "  docker: already up"; return; fi
  echo "  docker: daemon not reachable — starting colima…"
  colima start >/dev/null 2>&1
  for _ in $(seq 1 30); do
    docker info >/dev/null 2>&1 && { echo "  docker: started (colima)"; return; }
    sleep 1
  done
  echo "  docker: FAILED to start — run 'colima start' manually and check its output"
}

stop_docker() {
  # Frees Colima's whole VM (~1GiB) -- worth it when nothing needs docker for a while (e.g. iOS
  # dev over a weekend). Must run AFTER `docker compose down` above, which needs the daemon up.
  if ! colima status >/dev/null 2>&1; then echo "  docker (colima): already down"; return; fi
  colima stop >/dev/null 2>&1 && echo "  docker (colima): stopped" \
    || echo "  docker (colima): FAILED to stop — check 'colima status' manually"
}

start_ollama() {
  if is_up $ROOT/data/ollama.pid; then echo "  ollama: already up"; return; fi
  $PY $ROOT/core/spawn_daemon.py $ROOT/data/ollama.pid $ROOT/logs/ollama.log $OLLAMA_BIN serve
  sleep 2
  is_up $ROOT/data/ollama.pid && echo "  ollama: started" || echo "  ollama: FAILED to start — check logs/ollama.log"
}

start_v2_dash() {
  if is_up $ROOT/v2/data/dashboard.pid; then echo "  v2-dashboard: already up"; return; fi
  # spawn_daemon.py's WORKDIR defaults to $ROOT/data (v1's!) unless TRADER_DATA_DIR says otherwise
  # — without this, the v2 dashboard's bare Path("trades.csv") silently reads v1's trades.
  TRADER_DATA_DIR=$ROOT/v2/data TRADER_LOG_DIR=$ROOT/v2/logs \
    $PY $ROOT/core/spawn_daemon.py $ROOT/v2/data/dashboard.pid $ROOT/v2/logs/dashboard.log $PY -u $ROOT/v2/dashboard.py
  sleep 1
  is_up $ROOT/v2/data/dashboard.pid && echo "  v2-dashboard: started" || echo "  v2-dashboard: FAILED to start — check v2/logs/dashboard.log"
}

status_all() {
  echo "── docker (colima) ──"
  docker info >/dev/null 2>&1 && echo "  UP" || echo "  down"
  echo "── ollama ──"
  is_up $ROOT/data/ollama.pid && echo "  UP (pid $(cat $ROOT/data/ollama.pid))" || echo "  down"
  echo "── v1 (bot + dashboard) ──"
  $ROOT/manage.sh status
  echo "── v2-bot (docker) ──"
  if docker info >/dev/null 2>&1; then
    ( cd $ROOT/v2 && docker compose ps )
  else
    echo "  down (docker daemon not running)"
  fi
  echo "── v2-dashboard ──"
  is_up $ROOT/v2/data/dashboard.pid && echo "  UP (pid $(cat $ROOT/v2/data/dashboard.pid))" || echo "  down"
}

case "$1" in
  start)
    echo "Starting full stack…"
    start_docker                                    # v2-bot needs this before `docker compose up`
    start_ollama                                    # bots' ticker-corrector calls this — bring it up first
    $ROOT/manage.sh start >/dev/null && echo "  v1 (bot+dashboard): started"
    ( cd $ROOT/v2 && docker compose up -d >/dev/null ) && echo "  v2-bot (docker): started"
    start_v2_dash
    sleep 2
    echo
    status_all ;;

  stop)
    # Order: trading engines first (so nothing keeps trying to open positions while the rest of
    # the stack goes down), monitoring dashboards next, ollama next, docker/colima LAST (it must
    # still be up for `docker compose down` above to work at all).
    echo "Stopping full stack (graceful)…"
    if docker info >/dev/null 2>&1; then
      ( cd $ROOT/v2 && docker compose down ) && echo "  v2-bot (docker): stopped (SIGTERM, graceful)"
    else
      echo "  v2-bot (docker): already down (docker daemon not running)"
    fi
    wait_for_death $ROOT/data/bot.pid "v1-bot"
    wait_for_death $ROOT/data/dashboard.pid "v1-dashboard"
    wait_for_death $ROOT/v2/data/dashboard.pid "v2-dashboard"
    wait_for_death $ROOT/data/ollama.pid "ollama"
    stop_docker
    echo "All stopped." ;;

  restart)
    "$0" stop
    sleep 2
    "$0" start ;;

  status) status_all ;;

  *) echo "usage: $0 {start|stop|restart|status}" ;;
esac
