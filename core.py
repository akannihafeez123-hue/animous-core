# core.py
import time
import os
import traceback
from reader import parse_platform
from synthesizer import generate_plugin
from executor import execute_task
from llm_utils import get_embeddings

# Environment
WALLET_ADDRESS = os.getenv("WALLET_ADDRESS")

# Memory / embeddings (uses sentence-transformers fallback via llm_utils)
embeddings = get_embeddings()

# Try to use FAISS vector store if available; otherwise use an in-memory fallback.
try:
    from langchain_community.vectorstores import FAISS
    from langchain.docstore.document import Document

    class MemoryDB:
        def __init__(self, embedder):
            self.embedder = embedder
            # bootstrap with a single doc so FAISS creation is straightforward
            docs = [Document(page_content="initial memory", metadata={"source": "bootstrap"})]
            vectors = self.embedder.embed_documents([d.page_content for d in docs])
            self.store = FAISS.from_documents(docs, self.embedder)
        def add_documents(self, docs):
            # docs: list[Document]
            self.store.add_documents(docs)
        def similarity_search(self, query, k=3):
            try:
                return self.store.similarity_search(query, k=k)
            except Exception:
                return []

    memory_db = MemoryDB(embeddings)

except Exception:
    # Fallback simple memory: keep raw texts and do naive length-based similarity
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
                # cheap heuristic: sort by length similarity and return top-k as pseudo-documents
                scored = sorted(self.docs, key=lambda x: abs(len(x["content"]) - len(query)))
                results = []
                for item in scored[:k]:
                    # create a lightweight Document-like object for compatibility
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
    plugin may be any JSON-serializable object; we store its string form.
    """
    try:
        platforms[name] = plugin
        # Create a minimal Document-like object if FAISS is used, otherwise pass a simple object
        try:
            from langchain.docstore.document import Document
            doc = Document(page_content=str(plugin), metadata={"source": name})
            memory_db.add_documents([doc])
        except Exception:
            # fallback: pass a simple object with attributes consumed by fallback memory
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
    Each result has .page_content and .metadata attributes.
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

# Core loop flags
auto_learn_enabled = True
auto_retrain_enabled = True

def heartbeat():
    print(f"💓 Animous Core is alive. Wallet: {WALLET_ADDRESS}")
    print(f"🧠 Memory has {len(platforms)} learned platforms. Plugin scores tracked: {len(plugin_scores)}")

def main_loop():
    # Example platforms to learn; replace with dynamic discovery in production
    platforms_to_learn = [
        {"name": "LabelNet", "url": "https://labelnet.ai/docs"},
        {"name": "TaskHive", "url": "https://taskhive.io/help"},
    ]
    while True:
        try:
            heartbeat()
            for p in platforms_to_learn:
                doc = parse_platform(p["url"])
                if not doc or not doc.get("content"):
                    print(f"⚠️ No content for {p['name']} ({p['url']})")
                    continue
                if auto_learn_enabled:
                    plugin = generate_plugin(doc)
                    if plugin:
                        store_platform(p["name"], plugin)
                        if auto_retrain_enabled:
                            for task in plugin.get("tasks", []):
                                try:
                                    result = execute_task({**task, "platform": p["name"]})
                                    update_plugin_score(p["name"], result.get("status") == "completed", result.get("pay", 0))
                                except Exception:
                                    traceback.print_exc()
            # sleep between cycles
            time.sleep(300)
        except KeyboardInterrupt:
            print("Shutting down main loop.")
            break
        except Exception:
            traceback.print_exc()
            # short sleep on unexpected error to avoid tight failure loop
            time.sleep(10)

if __name__ == "__main__":
    main_loop()
