#!/usr/bin/env python3
"""
bot.py — single-entry Python launcher that starts proxy, bot code and core worker.

Usage (default):
  python -u bot.py

Environment overrides:
  PORT (default 8080)
  PROXY_CMD (default: uvicorn app:app --host 0.0.0.0 --port {PORT} --proxy-headers)
  BOT_CMD   (default: python -u bot.py  -> changed below to run local bot function)
  CORE_CMD  (default: python -u core.py)
  LOGDIR    (default: /tmp/logs)
"""

import os
import shlex
import signal
import subprocess
import sys
import threading
import time
from typing import List

# === Configuration from env ===
PORT = os.environ.get("PORT", "8080")
LOGDIR = os.environ.get("LOGDIR", "/tmp/logs")
os.makedirs(LOGDIR, exist_ok=True)

# Default commands; BOT_CMD is placeholder because we run bot logic in-process below
DEFAULT_PROXY_CMD = f"uvicorn app:app --host 0.0.0.0 --port {PORT} --proxy-headers"
DEFAULT_BOT_CMD = None  # we will run local run_bot() function if present
DEFAULT_CORE_CMD = "python -u core.py"

PROXY_CMD = os.environ.get("PROXY_CMD", DEFAULT_PROXY_CMD)
BOT_CMD = os.environ.get("BOT_CMD", "")  # if provided, will spawn as subprocess
CORE_CMD = os.environ.get("CORE_CMD", DEFAULT_CORE_CMD)

CHILDREN: List[subprocess.Popen] = []
SHUTTING_DOWN = False
LOCK = threading.Lock()


def stream_process_output(proc: subprocess.Popen, name: str) -> None:
    """Stream subprocess stdout/stderr to main stdout with prefix."""
    assert proc.stdout is not None
    for line in iter(proc.stdout.readline, b""):
        sys.stdout.write(f"[{name}] {line.decode(errors='replace')}")
    proc.stdout.close()


def start_subprocess(cmd: str, name: str) -> subprocess.Popen:
    """Start a subprocess with combined stdout/stderr and non-blocking streaming."""
    logfile = os.path.join(LOGDIR, f"{name}.log")
    print(f"Starting {name}: {cmd} (log -> {logfile})", flush=True)
    # Use shell=False for safety; split the command
    proc = subprocess.Popen(
        shlex.split(cmd),
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        cwd=os.getcwd(),
        env=os.environ.copy(),
    )
    # Start thread to stream output
    t = threading.Thread(target=stream_process_output, args=(proc, name), daemon=True)
    t.start()
    with LOCK:
        CHILDREN.append(proc)
    print(f"{name} started PID {proc.pid}", flush=True)
    return proc


def terminate_children():
    global SHUTTING_DOWN
    with LOCK:
        if SHUTTING_DOWN:
            return
        SHUTTING_DOWN = True
    print("Shutting down children...", flush=True)
    # send TERM to each child
    for proc in list(CHILDREN):
        try:
            proc.terminate()
        except Exception:
            pass
    # wait briefly, then kill remaining
    deadline = time.time() + 5
    for proc in list(CHILDREN):
        try:
            while proc.poll() is None and time.time() < deadline:
                time.sleep(0.1)
            if proc.poll() is None:
                proc.kill()
        except Exception:
            pass
    print("Children terminated.", flush=True)


def signal_handler(signum, frame):
    print(f"Received signal {signum}; initiating shutdown.", flush=True)
    terminate_children()
    # allow main to exit
    sys.exit(0)


# === If you have inline bot code, run it here ===
def run_bot_inline():
    """
    Replace this stub with your bot's existing logic if it runs in-process.
    If you prefer to spawn bot as a subprocess, set BOT_CMD env to the command
    that starts your bot and leave this function as a no-op.
    """
    # Example: import your bot module and call its run loop function
    try:
        # from mybotmodule import main as bot_main
        # bot_main()
        # For now, keep a simple idle loop so the process doesn't exit.
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        pass


def main():
    # Setup signal handlers
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

    procs = []

    # 1) Start proxy
    proxy_proc = start_subprocess(PROXY_CMD, "proxy")
    procs.append(proxy_proc)
    # small delay to let proxy bind the port
    time.sleep(1)

    # 2) Start bot
    if BOT_CMD:
        bot_proc = start_subprocess(BOT_CMD, "bot")
        procs.append(bot_proc)
    else:
        # Run inline bot in a background thread to keep main thread free for signal handling
        bot_thread = threading.Thread(target=run_bot_inline, name="bot-inline", daemon=True)
        bot_thread.start()
        print("Bot running inline in-thread.", flush=True)

    time.sleep(1)

    # 3) Start core
    core_proc = start_subprocess(CORE_CMD, "core")
    procs.append(core_proc)

    # monitor children: if any exits, shutdown everything
    try:
        while True:
            with LOCK:
                for p in list(CHILDREN):
                    if p.poll() is not None:
                        print(f"Child PID {p.pid} exited with code {p.returncode}; shutting down.", flush=True)
                        terminate_children()
                        # exit after shutdown
                        sys.exit(0)
            time.sleep(1)
    except SystemExit:
        raise
    except Exception as exc:
        print(f"Launcher encountered exception: {exc}", flush=True)
        terminate_children()
        sys.exit(1)


if __name__ == "__main__":
    main()
