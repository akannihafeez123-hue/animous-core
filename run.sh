#!/usr/bin/env bash
set -euo pipefail

# runtime configuration (can be overridden with env)
: "${PORT:=8080}"
PROXY_CMD=${PROXY_CMD:-"uvicorn app:app --host 0.0.0.0 --port ${PORT} --proxy-headers"}
BOT_CMD=${BOT_CMD:-"python -u bot.py"}
CORE_CMD=${CORE_CMD:-"python -u core.py"}

LOGDIR=${LOGDIR:-/tmp/logs}
mkdir -p "${LOGDIR}"

children=()

# Graceful shutdown handler: forward TERM to children
term_handler() {
  echo "Received termination signal, stopping children..."
  for pid in "${children[@]}"; do
    if kill -0 "$pid" 2>/dev/null; then
      kill -TERM "$pid" 2>/dev/null || true
    fi
  done
  # give processes a moment to exit
  sleep 2
  exit 0
}
trap term_handler SIGINT SIGTERM

start_proc() {
  name="$1"; shift
  cmd="$*"
  logfile="${LOGDIR}/${name}.log"
  echo "Starting ${name}: ${cmd} (log -> ${logfile})"
  # start in background, redirect stdout/stderr to logfile
  bash -lc "${cmd}" >> "${logfile}" 2>&1 &
  pid=$!
  children+=("$pid")
  echo "${name} started as PID ${pid}"
}

# Start proxy first so it binds the port
start_proc "proxy" ${PROXY_CMD}
sleep 1

# Start bot
start_proc "bot" ${BOT_CMD}
sleep 1

# Start core worker
start_proc "core" ${CORE_CMD}

# Monitor children; if any exits, shut everything down
while true; do
  for pid in "${children[@]}"; do
    if ! kill -0 "$pid" 2>/dev/null; then
      echo "Process PID ${pid} exited; shutting down other children."
      term_handler
    fi
  done
  sleep 2
done
