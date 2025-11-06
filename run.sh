#!/usr/bin/env bash
set -euo pipefail

# configuration (can be overridden by env)
: "${PORT:=8080}"
PROXY_CMD=${PROXY_CMD:-"uvicorn app:app --host 0.0.0.0 --port ${PORT} --proxy-headers"}
BOT_CMD=${BOT_CMD:-"python -u bot.py"}
CORE_CMD=${CORE_CMD:-"python -u core.py"}

LOGDIR=${LOGDIR:-/tmp/logs}
mkdir -p "${LOGDIR}"

# helper to stop children
children=()
term_handler() {
  echo "Received termination signal, stopping children..."
  for pid in "${children[@]}"; do
    if kill -0 "$pid" 2>/dev/null; then
      kill -TERM "$pid" 2>/dev/null || true
    fi
  done
  # give them time to exit
  sleep 2
  exit 0
}
trap term_handler SIGINT SIGTERM

# start process helper
start_proc() {
  name="$1"
  shift
  cmd="$*"
  logfile="${LOGDIR}/${name}.log"
  echo "Starting ${name}: ${cmd} (log -> ${logfile})"
  # start in background, redirect stdout/stderr
  bash -lc "${cmd}" >> "${logfile}" 2>&1 &
  pid=$!
  children+=("$pid")
  echo "${name} started as PID ${pid}"
}

# Start proxy first (binds port)
start_proc "proxy" ${PROXY_CMD}
sleep 1

# Start bot
start_proc "bot" ${BOT_CMD}
sleep 1

# Start core
start_proc "core" ${CORE_CMD}

# Wait for children - if any exits, propagate exit
while true; do
  for pid in "${children[@]}"; do
    if ! kill -0 "$pid" 2>/dev/null; then
      echo "Process PID ${pid} exited; shutting down other children."
      term_handler
    fi
  done
  sleep 2
done
