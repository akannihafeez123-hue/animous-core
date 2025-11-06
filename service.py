#!/usr/bin/env python3
"""
service.py — unified single-file service that includes:
  - memory (simple persistent store)
  - app (FastAPI proxy with /health and /v1/generate)
  - bot (threaded bot loop, integrate Telegram or other)
  - core (threaded core worker)
  - executor (task runner)
  - reader (request parsing helpers)
  - synthesizer (TTS placeholder)
  - llm_utils (upstream calling helpers)
  - rotate_api_key (update .env and optionally notify)
  - write_dockerfile (writes a recommended Dockerfile)
Usage:
  python -u service.py
Keep requirements.txt and .env in repo root.
"""
from __future__ import annotations

import os
import sys
import time
import json
import signal
import threading
import logging
import shlex
from pathlib import Path
from typing import Optional, Dict, Any, List
from contextlib import suppress

from fastapi import FastAPI, Request, HTTPException
import uvicorn
from pydantic import BaseSettings
from dotenv import load_dotenv, set_key, dotenv_values

# Load .env if present
ROOT = Path(__file__).parent
ENV_PATH = ROOT / ".env"
if ENV_PATH.exists():
    load_dotenv(dotenv_path=ENV_PATH)

# === Settings ===
class Settings(BaseSettings):
    PORT: int = int(os.getenv("PORT", 8080))
    LOGDIR: str = os.getenv("LOGDIR", "/tmp/logs")
    BOT_PROXY_KEY: Optional[str] = os.getenv("BOT_PROXY_KEY")
    UPSTREAM_TEXT_URL: Optional[str] = os.getenv("UPSTREAM_TEXT_URL")
    ROTATE_NOTIFY_URL: Optional[str] = os.getenv("ROTATE_NOTIFY_URL")
    TELEGRAM_BOT_TOKEN: Optional[str] = os.getenv("TELEGRAM_BOT_TOKEN")
    BOT_POLL_INTERVAL: float = float(os.getenv("BOT_POLL_INTERVAL", 1.0))
    CORE_SLEEP_SECONDS: float = float(os.getenv("CORE_SLEEP_SECONDS", 5.0))
    DEBUG: bool = os.getenv("DEBUG", "0") in ("1", "true", "True")
    MEMORY_PERSIST: Optional[str] = os.getenv("MEMORY_PERSIST")  # path to persist memory.json

settings = Settings()

# === Logging ===
os.makedirs(settings.LOGDIR, exist_ok=True)
logger = logging.getLogger("service")
logger.setLevel(logging.DEBUG if settings.DEBUG else logging.INFO)
stream_h = logging.StreamHandler(sys.stdout)
stream_h.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s"))
logger.addHandler(stream_h)
file_h = logging.FileHandler(Path(settings.LOGDIR) / "service.log")
file_h.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s"))
logger.addHandler(file_h)

# === memory.py (simple persistent in-file memory) ===
class Memory:
    def __init__(self, persist_path: Optional[str] = None):
        self._lock = threading.Lock()
        self._data: Dict[str, Any] = {}
        self.persist_path = persist_path or (Path(settings.LOGDIR) / "memory.json")
        self._load()

    def _load(self):
        try:
            p = Path(self.persist_path)
            if p.exists():
                with p.open("r", encoding="utf-8") as f:
                    self._data = json.load(f)
        except Exception:
            logger.exception("Failed to load memory from %s", self.persist_path)

    def _persist(self):
        try:
            p = Path(self.persist_path)
            with p.open("w", encoding="utf-8") as f:
                json.dump(self._data, f)
        except Exception:
            logger.exception("Failed to persist memory to %s", self.persist_path)

    def get(self, key: str, default=None):
        with self._lock:
            return self._data.get(key, default)

    def set(self, key: str, value: Any):
        with self._lock:
            self._data[key] = value
            self._persist()

    def delete(self, key: str):
        with self._lock:
            if key in self._data:
                del self._data[key]
                self._persist()

    def all(self) -> Dict[str, Any]:
        with self._lock:
            return dict(self._data)

memory = Memory(persist_path=str(settings.MEMORY_PERSIST) if settings.MEMORY_PERSIST else str(Path(settings.LOGDIR) / "memory.json"))

# === reader.py (payload parsing helpers) ===
def read_prompt_from_payload(payload: Any) -> str:
    if isinstance(payload, dict):
        if "prompt" in payload:
            return payload["prompt"]
        if "input" in payload:
            return payload["input"]
        return json.dumps(payload)
    return str(payload)

# === llm_utils.py (upstream LLM call wrapper) ===
def call_upstream_text(prompt: str, timeout: float = 10.0) -> str:
    """
    Minimal wrapper: if UPSTREAM_TEXT_URL is provided, do a POST with prompt.
    If not configured, return a deterministic echo.
    Note: requests is optional; ensure included in requirements if you enable upstream.
    """
    url = settings.UPSTREAM_TEXT_URL
    if not url:
        return f"echo: {prompt}"
    try:
        import requests
        payload = {"prompt": prompt}
        r = requests.post(url, json=payload, timeout=timeout)
        r.raise_for_status()
        try:
            data = r.json()
            # expected field heuristics
            for k in ("generated_text", "text", "output"):
                if k in data:
                    return data[k]
            return json.dumps(data)
        except Exception:
            return r.text
    except Exception:
        logger.exception("Failed calling upstream LLM; falling back to echo")
        return f"echo: {prompt}"

# === synthesizer.py (TTS placeholder) ===
def synthesize_text_to_audio_url(text: str) -> str:
    """
    Placeholder synthesizer: returns a data URL containing hex of text.
    Replace with real TTS integration that returns a hosted audio URL.
    """
    hexed = text.encode("utf-8").hex()
    short = hexed[:2048]
    return f"data:audio/wav;base64,{short}"

# === executor.py (task runner) ===
def execute_task(task_name: str, payload: Dict[str, Any]) -> Dict[str, Any]:
    logger.info("Executing task %s with payload keys=%s", task_name, list(payload.keys()) if isinstance(payload, dict) else None)
    # placeholder behaviors
    if task_name == "ping":
        return {"status": "ok", "ts": time.time()}
    if task_name == "memory_dump":
        return {"memory": memory.all()}
    # general echo
    return {"task": task_name, "payload": payload, "result": "done"}

# === app.py (FastAPI) ===
app = FastAPI(title="UnifiedService")

@app.get("/health")
async def health():
    return {"status": "ok"}

@app.post("/v1/generate")
async def generate(req: Request):
    key = req.headers.get("x-bot-proxy-key")
    if settings.BOT_PROXY_KEY and key != settings.BOT_PROXY_KEY:
        raise HTTPException(status_code=401, detail="unauthorized")
    payload = await req.json()
    prompt = read_prompt_from_payload(payload)
    logger.info("Received generate request prompt_len=%d", len(prompt) if prompt else 0)
    # record last prompt
    memory.set("last_prompt", {"prompt": prompt, "ts": time.time()})
    text = call_upstream_text(prompt)
    audio = synthesize_text_to_audio_url(text)
    return {"text": text, "audio_url": audio}

@app.post("/v1/execute")
async def api_execute(req: Request):
    key = req.headers.get("x-bot-proxy-key")
    if settings.BOT_PROXY_KEY and key != settings.BOT_PROXY_KEY:
        raise HTTPException(status_code=401, detail="unauthorized")
    payload = await req.json()
    task = payload.get("task", "default")
    data = payload.get("payload", {})
    result = execute_task(task, data)
    return {"result": result}

@app.get("/v1/memory")
async def api_memory(req: Request):
    key = req.headers.get("x-bot-proxy-key")
    if settings.BOT_PROXY_KEY and key != settings.BOT_PROXY_KEY:
        raise HTTPException(status_code=401, detail="unauthorized")
    return {"memory": memory.all()}

# === bot.py (threaded bot loop) ===
stop_event = threading.Event()
_threads: List[threading.Thread] = []

def bot_thread_loop():
    """
    Replace the internals to integrate with real Telegram or other bot clients.
    This thread demonstrates periodic work and safe shutdown.
    """
    logger.info("Bot thread started")
    try:
        while not stop_event.is_set():
            # Example: read last prompt and optionally act
            last = memory.get("last_prompt")
            if last:
                logger.debug("Bot observed last prompt ts=%s", last.get("ts"))
            # If using Telegram, you'd poll or react here.
            time.sleep(settings.BOT_POLL_INTERVAL)
    except Exception:
        logger.exception("Bot thread exception")
    finally:
        logger.info("Bot thread stopping")

# === core.py (threaded core worker) ===
def core_thread_loop():
    logger.info("Core thread started")
    try:
        while not stop_event.is_set():
            # Example periodic maintenance tasks
            # - compact memory when it grows too large
            mem = memory.all()
            if len(mem) > 2000:
                logger.info("Compacting memory from %d entries", len(mem))
                keep = {"last_prompt": mem.get("last_prompt")}
                for k in list(mem.keys()):
                    if k != "last_prompt":
                        memory.delete(k)
                memory.set("compacted_at", time.time())
            time.sleep(settings.CORE_SLEEP_SECONDS)
    except Exception:
        logger.exception("Core thread exception")
    finally:
        logger.info("Core thread stopping")

# === rotate_api_key utility ===
def rotate_api_key(new_key: str, propagate_url: Optional[str] = None) -> None:
    """
    Update BOT_PROXY_KEY in .env and optionally POST the new key to propagate_url.
    """
    env_file = str(ENV_PATH)
    # Use set_key from python-dotenv for robust quoting
    try:
        set_key(env_file, "BOT_PROXY_KEY", new_key)
        logger.info("Updated BOT_PROXY_KEY in %s", env_file)
    except Exception:
        logger.exception("Failed to write BOT_PROXY_KEY with set_key; attempting manual rewrite")
        vals = dict(dotenv_values(env_file))
        vals["BOT_PROXY_KEY"] = new_key
        with open(env_file, "w", encoding="utf-8") as f:
            f.write("\n".join(f"{k}={v}" for k, v in vals.items()))
        logger.info("Wrote BOT_PROXY_KEY manually to %s", env_file)
    if propagate_url:
        try:
            import requests
            r = requests.post(propagate_url, json={"key": new_key}, timeout=10)
            if r.status_code >= 300:
                logger.warning("Propagation POST to %s returned %s", propagate_url, r.status_code)
            else:
                logger.info("Propagated new API key to %s", propagate_url)
        except Exception:
            logger.exception("Failed to propagate new API key to %s", propagate_url)

# === write_dockerfile utility ===
DOCKERFILE_TEMPLATE = """# Generated Dockerfile for unified service.py
FROM python:3.11-slim
RUN apt-get update && apt-get install -y --no-install-recommends \\
    build-essential ca-certificates curl \\
  && rm -rf /var/lib/apt/lists/*
WORKDIR /app
COPY requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r /app/requirements.txt
COPY . /app
RUN test -f /app/service.py && chmod +x /app/service.py
EXPOSE 8080
CMD ["python", "-u", "service.py"]
"""

def write_dockerfile(path: Optional[str] = None) -> str:
    target = Path(path or ROOT / "Dockerfile")
    target.write_text(DOCKERFILE_TEMPLATE, encoding="utf-8")
    logger.info("Dockerfile written to %s", target)
    return str(target)

# === Graceful shutdown helpers and main ===
def _signal_handler(sig, frame):
    logger.info("Received signal %s; shutting down", sig)
    stop_event.set()

def start_threads():
    t_bot = threading.Thread(target=bot_thread_loop, name="bot-thread", daemon=True)
    t_core = threading.Thread(target=core_thread_loop, name="core-thread", daemon=True)
    _threads.extend([t_bot, t_core])
    t_bot.start()
    t_core.start()
    logger.info("Started bot and core threads")

def stop_threads(timeout: float = 3.0):
    stop_event.set()
    for t in _threads:
        with suppress(Exception):
            t.join(timeout=timeout)
    logger.info("Threads stopped")

def main():
    signal.signal(signal.SIGINT, _signal_handler)
    signal.signal(signal.SIGTERM, _signal_handler)

    # ensure Dockerfile exists for reference but do not override if present
    if not (ROOT / "Dockerfile").exists():
        write_dockerfile()

    start_threads()

    # Run uvicorn programmatically so server runs inside this single PID
    config = uvicorn.Config(app, host="0.0.0.0", port=settings.PORT, log_level="debug" if settings.DEBUG else "info")
    server = uvicorn.Server(config)
    try:
        logger.info("Starting uvicorn on port %s", settings.PORT)
        server.run()
    except Exception:
        logger.exception("Uvicorn server failed")
    finally:
        logger.info("Server stopping; stopping threads")
        stop_threads()
        logger.info("Shutdown complete")

if __name__ == "__main__":
    main()
