# core.py
import time
import os
import signal
import traceback
from reader import parse_platform
from synthesizer import generate_plugin
from executor import execute_task
from llm_utils import get_embeddings

# Environment
WALLET_ADDRESS = os.getenv("WALLET_ADDRESS", "")
CORE_SLEEP_SECONDS = int(os.getenv("CORE_SLEEP_SECONDS", "300"))

# Memory / embeddings (uses sentence-transformers fallback via llm_utils)
embeddings = get_embeddings()

# Try to use FAISS vector store if available; otherwise use an in-memory fallback.
try:
    from langchain_community.vectorstores import FAISS
    from langchain.docstore.document import Document

    class MemoryDB:
        def __init__(self, embedder):
            self.embedder = embedder
            docs = [Document(page_content="initial memory", metadata={"source": "bootstrap"})]
            self.store = FAISS.from_documents(docs, self.embedder)

        def add_documents(self, docs):
            self.store.add_documents(docs)

        def similarity_search(self, query, k=3):
            try:
                return self.store.similarity_search(query, k=k)
            except Exception:
                return []

    memory_db = MemoryDB(embeddings)

except Exception:
    class FallbackMemory:
        def __init__(self, embedder):
            self.embedder = embedder
            self.docs = []  # list of dicts: {"content": str, "meta": {...}}

        def add_documents(self, docs):
            for d in docs:
                content = getattr(d, "page_content", str(d)) if hasattr(d, "page_content") else str(d)
                meta = getattr(d, "metadata", {}) if hasattr(d, "metadata") else {}
                self.docs.append({"content": content, "meta": meta})

        def similarity_search(self, query, k=3):
            try:
                scored = sorted(self.docs, key=lambda x: abs(len(x["content"]) - len(query)))
                results = []
                for item in scored[:k]:
                    class D:
                        def __init__(self, text, meta):
                            self.page_content = text
                            self.metadata = meta
                    results.append(D(item["content"], item["meta"]))
                return results
            except Exception:
                return []

    memory_db = FallbackMemory(embeddings)

# In-memory registry
platforms = {}
plugin_scores = {}

def store_platform(name, plugin):
    """
    Save plugin to registry and persist a representation into memory_db.
    """
    try:
        platforms[name] = plugin
        try:
            from langchain.docstore.document import Document
            doc = Document(page_content=str(plugin), metadata={"source": name})
            memory_db.add_documents([doc])
        except Exception:
            class SimpleDoc:
                def __init__(self, text, meta):
                    self.page_content = text
                    self.metadata = meta
            memory_db.add_documents([SimpleDoc(str(plugin), {"source": name})])
        print(f"🧠 Stored plugin for {name}")
    except Exception:
        traceback.print_exc()

def search_memory(query):
    """
    Return up to 3 similar Document-like objects from memory_db.
    """
    try:
        return memory_db.similarity_search(query, k=3)
    except Exception:
        traceback.print_exc()
        return []

def update_plugin_score(name, success, payout):
    score = plugin_scores.get(name, {"success": 0, "fail": 0, "earned": 0})
    if success:
        score["success"] += 1
        score["earned"] += payout
    else:
        score["fail"] += 1
    plugin_scores[name] = score

# Core loop flags (configurable via env)
auto_learn_enabled = os.getenv("AUTO_LEARN_ENABLED", "true").lower() in ("1", "true", "yes")
auto_retrain_enabled = os.getenv("AUTO_RETRAIN_ENABLED", "true").lower() in ("1", "true", "yes")

_shutdown = False

def _signal_handler(signum, frame):
    global _shutdown
    print("Received shutdown signal.")
    _shutdown = True

signal.signal(signal.SIGINT, _signal_handler)
signal.signal(signal.SIGTERM, _signal_handler)

def heartbeat():
    print(f"💓 Animous Core is alive. Wallet: {WALLET_ADDRESS}")
    print(f"🧠 Learned platforms: {len(platforms)} | Plugin scores: {len(plugin_scores)}")

def _parse_platforms_env():
    env_list = os.getenv("PLATFORMS_TO_LEARN", "")
    parsed = []
    if env_list:
        for item in env_list.split(","):
            if "|" in item:
                n, u = item.split("|", 1)
                parsed.append({"name": n.strip(), "url": u.strip()})
    return parsed

def main_loop():
    # Default sample platforms (replace with real targets via env)
    platforms_to_learn = [
        {"name": "LabelNet", "url": "https://labelnet.ai/docs"},
        {"name": "TaskHive", "url": "https://taskhive.io/help"},
    ]
    env_parsed = _parse_platforms_env()
    if env_parsed:
        platforms_to_learn = env_parsed

    while not _shutdown:
        try:
            heartbeat()
            for p in platforms_to_learn:
                if _shutdown:
                    break
                name = p.get("name")
                url = p.get("url")
                try:
                    doc = parse_platform(url)
                    if not doc or not doc.get("content"):
                        print(f"⚠️ No content for {name} ({url})")
                        continue
                    if auto_learn_enabled:
                        plugin = generate_plugin(doc)
                        if plugin:
                            store_platform(name, plugin)
                            if auto_retrain_enabled:
                                tasks = plugin.get("tasks", []) if isinstance(plugin, dict) else []
                                for task in tasks:
                                    if _shutdown:
                                        break
                                    try:
                                        # Ensure task has required fields
                                        task_obj = {
                                            "title": task.get("title", "unnamed"),
                                            "description": task.get("description", ""),
                                            "pay": task.get("pay", 0),
                                            "ai_solvable": task.get("ai_solvable", True),
                                            "platform": name,
                                        }
                                        result = execute_task(task_obj)
                                        update_plugin_score(name, result.get("status") == "completed", result.get("pay", 0))
                                    except Exception:
                                        traceback.print_exc()
                except Exception:
                    traceback.print_exc()
            # sleep between cycles, but check shutdown flag periodically
            slept = 0
            while slept < CORE_SLEEP_SECONDS and not _shutdown:
                time.sleep(1)
                slept += 1
        except Exception:
            traceback.print_exc()
            time.sleep(5)

    print("Main loop exiting gracefully.")

if __name__ == "__main__":
    main_loop()
