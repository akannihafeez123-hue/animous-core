# synthesizer.py
import json
import traceback
from llm_utils import get_llm, get_embeddings
from reader import parse_platform

# LLM and embeddings (safe initialization via llm_utils)
llm = get_llm()
embeddings = get_embeddings()

# Try to use FAISS if available for similarity search, otherwise fall back to an in-memory list
try:
    from langchain_community.vectorstores import FAISS
    from langchain.docstore.document import Document

    class MemoryIndex:
        def __init__(self, embedder):
            self.embedder = embedder
            # bootstrap with a minimal document so FAISS can initialize cleanly
            docs = [Document(page_content="initial memory", metadata={"source": "bootstrap"})]
            self.store = FAISS.from_documents(docs, embedder)

        def add_documents(self, docs):
            self.store.add_documents(docs)

        def similarity_search(self, query, k=3):
            try:
                return self.store.similarity_search(query, k=k)
            except Exception:
                return []

    memory_index = MemoryIndex(embeddings)

except Exception:
    # Simple fallback memory index: store raw texts and return naive top-k
    class FallbackIndex:
        def __init__(self, embedder):
            self.embedder = embedder
            self.items = []  # list of dicts: {"content": str, "meta": {...}}

        def add_documents(self, docs):
            for d in docs:
                content = getattr(d, "page_content", str(d)) if hasattr(d, "page_content") else str(d)
                meta = getattr(d, "metadata", {}) if hasattr(d, "metadata") else {}
                self.items.append({"content": content, "meta": meta})

        def similarity_search(self, query, k=3):
            # naive heuristic: return shortest distance by length
            scored = sorted(self.items, key=lambda x: abs(len(x["content"]) - len(query)))
            results = []
            for item in scored[:k]:
                class D:
                    def __init__(self, text, meta):
                        self.page_content = text
                        self.metadata = meta
                results.append(D(item["content"], item["meta"]))
            return results

    memory_index = FallbackIndex(embeddings)


def search_memory(query, k=3):
    """
    Return up to k Document-like objects with attributes .page_content and .metadata.
    """
    try:
        return memory_index.similarity_search(query, k=k)
    except Exception:
        traceback.print_exc()
        return []


def _ask_llm_for_plugin(doc_text, context_text):
    """
    Ask the LLM to return a JSON object describing the plugin.
    The JSON schema expected:
    {
      "platform": "ShortName",
      "tasks": [
        {"title": "...", "description": "...", "pay": 0.0, "ai_solvable": true}
      ],
      "executor": "Short description or null"
    }
    """
    prompt = f"""
You are Animous Core. Read the platform documentation and produce a JSON object matching this schema:

{{
  "platform": "Short name for the platform",
  "tasks": [
    {{
      "title": "Short task title",
      "description": "Short description of the task steps",
      "pay": 0.0,
      "ai_solvable": true
    }}
  ],
  "executor": "Optional: short description of how to submit results or null"
}}

Platform documentation (truncated to first 8000 chars):
{doc_text[:8000]}

Context from similar platforms:
{context_text}

Respond ONLY with valid JSON and nothing else.
"""
    try:
        return llm.predict(prompt)
    except Exception:
        traceback.print_exc()
        return ""


def generate_plugin(doc):
    """
    Given a parsed platform doc (dict with 'url' and 'content'), generate a structured
    plugin description (as a dict). Returns None on failure.
    """
    try:
        # search memory for similar platforms to provide context
        similar = search_memory("task format: " + (doc.get("content", "")[:200]))
        context = "\n\n".join([s.page_content for s in similar])

        raw = _ask_llm_for_plugin(doc.get("content", ""), context)
        if not raw:
            return None

        # Prefer strict JSON parsing; attempt to extract a JSON substring if needed
        try:
            parsed = json.loads(raw)
            return parsed
        except Exception:
            try:
                start = raw.index("{")
                end = raw.rindex("}") + 1
                parsed = json.loads(raw[start:end])
                return parsed
            except Exception:
                # If LLM returned Python-like dict literal, attempt safe eval via json-style replacements
                try:
                    cleaned = raw.replace("'", '"')
                    parsed = json.loads(cleaned)
                    return parsed
                except Exception:
                    traceback.print_exc()
                    return None
    except Exception:
        traceback.print_exc()
        return None
