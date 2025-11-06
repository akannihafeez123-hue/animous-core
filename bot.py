#!/usr/bin/env python3
"""
bot.py — integrated launcher that starts:
  - proxy (uvicorn app:app)
  - bot logic (inline or subprocess)
  - core worker (core.py)

Run with: python -u bot.py

Environment variables (all optional; defaults shown):
  PORT=8080
  LOGDIR=/tmp/logs
  PROXY_CMD="uvicorn app:app --host 0.0.0.0 --port {PORT} --proxy-headers"
  BOT_CMD=""  (if empty, run inline run_bot_inline())
  CORE_CMD="python -u core.py"
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

DEFAULT_PROXY_CMD = f"uvicorn app:app --host 0.0.0.0 --port {PORT} --proxy-headers"
DEFAULT_CORE_CMD = "python -u core.py"

PROXY_CMD = os.environ.get("PROXY_CMD", DEFAULT_PROXY_CMD)
BOT_CMD = os.environ.get("BOT_CMD", "")  # if set, launcher will spawn as subprocess
CORE_CMD = os.environ.get("CORE_CMD", DEFAULT_CORE_CMD)

CHILDREN: List[subprocess.Popen] = []
CHILD_LOCK = threading.Lock()
SHUTTING_DOWN = False


def _stream_process_output(proc: subprocess.Popen, name: str) -> None:
    """Read subprocess stdout/stderr and write to main stdout with prefix."""
    assert proc.stdout is not None
    try:
        for raw in iter(proc.stdout.readline, b""):
            if not raw:
                break
            line = raw.decode(errors="replace")
            sys.stdout.write(f"[{name}] {line}")
            sys.stdout.flush()
    except Exception:
        pass
    finally:
        try:
            proc.stdout.close()
        except Exception:
            pass


def start_subprocess(cmd: str, name: str) -> subprocess.Popen:
    """Start a subprocess and stream its combined output to main stdout."""
    logfile = os.path.join(LOGDIR, f"{name}.log")
    print(f"Starting {name}: {cmd} (log -> {logfile})", flush=True)
    # Use shlex.split so we avoid a shell; works with simple commands
    parts = shlex.split(cmd)
    proc = subprocess.Popen(
        parts,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        cwd=os.getcwd(),
        env=os.environ.copy(),
    )
    t = threading.Thread(target=_stream_process_output, args=(proc, name), daemon=True)
    t.start()
    with CHILD_LOCK:
        CHILDREN.append(proc)
    print(f"{name} started PID {proc.pid}", flush=True)
    return proc


def terminate_children(timeout: float = 5.0) -> None:
    """Gracefully terminate children, escalate to kill if needed."""
    global SHUTTING_DOWN
    with CHILD_LOCK:
        if SHUTTING_DOWN:
            return
        SHUTTING_DOWN = True

    print("Shutting down children...", flush=True)
    procs = []
    with CHILD_LOCK:
        procs = list(CHILDREN)

    for p in procs:
        try:
            p.terminate()
        except Exception:
            pass

    deadline = time.time() + timeout
    for p in procs:
        try:
            while p.poll() is None and time.time() < deadline:
                time.sleep(0.1)
            if p.poll() is None:
                p.kill()
        except Exception:
            pass

    print("All child processes terminated.", flush=True)


def _signal_handler(signum, frame) -> None:
    print(f"Received signal {signum}; shutting down.", flush=True)
    terminate_children()
    # allow a brief pause for cleanup
    time.sleep(0.2)
    # exit main process
    sys.exit(0)


def run_bot_inline() -> None:
    """
    Place your bot startup code here.
    If you prefer to run the bot as a subprocess, set BOT_CMD environment variable.
    Example: BOT_CMD="python -u mybot.py"
    """
    try:
        # If you have an existing bot module with entrypoint, import and run it here.
        # Example:
        # from mybotmodule import main as bot_main
        # bot_main()
        #
        # Placeholder idle loop to keep the thread alive if no inline bot provided.
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        pass
    except Exception as ex:
        print(f"Inline bot encountered error: {ex}", flush=True)


def main() -> None:
    # register signal handlers
    signal.signal(signal.SIGINT, _signal_handler)
    signal.signal(signal.SIGTERM, _signal_handler)

    # 1) Start proxy first (binds port)
    try:
        proxy_proc = start_subprocess(PROXY_CMD, "proxy")
    except Exception as e:
        print(f"Failed to start proxy: {e}", flush=True)
        terminate_children()
        sys.exit(1)

    # give proxy a moment to bind the port
    time.sleep(1)

    # 2) Start bot (inline or subprocess)
    if BOT_CMD:
        try:
            bot_proc = start_subprocess(BOT_CMD, "bot")
        except Exception as e:
            print(f"Failed to start bot subprocess: {e}", flush=True)
            terminate_children()
            sys.exit(1)
    else:
        bot_thread = threading.Thread(target=run_bot_inline, name="bot-inline", daemon=True)
        bot_thread.start()
        print("Bot running inline (daemon thread).", flush=True)

    # 3) Start core worker
    try:
        core_proc = start_subprocess(CORE_CMD, "core")
    except Exception as e:
        print(f"Failed to start core: {e}", flush=True)
        terminate_children()
        sys.exit(1)

    # monitor children: if any exits, shut everything down
    try:
        while True:
            with CHILD_LOCK:
                for p in list(CHILDREN):
                    if p.poll() is not None:
                        print(f"Child PID {p.pid} exited with code {p.returncode}; shutting down.", flush=True)
                        terminate_children()
                        sys.exit(0)
            time.sleep(1)
    except KeyboardInterrupt:
        terminate_children()
        sys.exit(0)
    except SystemExit:
        raise
    except Exception as exc:
        print(f"Launcher error: {exc}", flush=True)
        terminate_children()
        sys.exit(1)


if __name__ == "__main__":
    main()
