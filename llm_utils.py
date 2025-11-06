# llm_utils.py
import os
import traceback
import requests

def _chatgpt5_via_http(prompt: str, timeout=20):
    endpoint = os.getenv("CHATGPT5_ENDPOINT")
    if not endpoint:
        raise RuntimeError("CHATGPT5_ENDPOINT not set")
    headers = {"Content-Type": "application/json"}
    auth = os.getenv("CHATGPT5_AUTH_TOKEN")
    if auth:
        headers["Authorization"] = f"Bearer {auth}"
    try:
        r = requests.post(endpoint, json={"prompt": prompt}, headers=headers, timeout=timeout)
        r.raise_for_status()
        payload = r.json()
        return payload.get("response", "")
    except Exception:
        traceback.print_exc()
        return ""

def get_llm():
    """
    Return an object exposing .predict(prompt: str) -> str.
    Priority:
      1. Platform-managed ChatGPT5 via CHATGPT5_ENDPOINT
      2. Anthropic ChatAnthropic via langchain_anthropic when ANTHROPIC_API_KEY set
      3. Deterministic DummyLLM
    """
    # 1) ChatGPT5 via HTTP endpoint
    if os.getenv("CHATGPT5_ENDPOINT"):
        class ChatGPT5HTTP:
            def predict(self, prompt: str) -> str:
                return _chatgpt5_via_http(prompt)
        return ChatGPT5HTTP()

    # 2) Anthropic via LangChain if configured
    try:
        if os.getenv("ANTHROPIC_API_KEY"):
            from langchain_anthropic import ChatAnthropic
            return ChatAnthropic(model="claude-sonnet-4.5", temperature=0.3)
    except Exception:
        traceback.print_exc()

    # 3) Dummy fallback
    class DummyLLM:
        def predict(self, prompt: str) -> str:
            return "FALLBACK_LLM: no production LLM configured."
    return DummyLLM()

def get_embeddings():
    """
    Return embeddings provider with embed_documents(list[str]) -> list[list[float]].
    Priority:
      1. sentence-transformers local model
      2. Identity fallback
    """
    try:
        from sentence_transformers import SentenceTransformer
        class STEmbeddings:
            def __init__(self, model_name="all-MiniLM-L6-v2"):
                self.model = SentenceTransformer(model_name)
            def embed_documents(self, docs):
                return self.model.encode(docs).tolist()
        return STEmbeddings()
    except Exception:
        traceback.print_exc()

    class IdentityEmbeddings:
        def embed_documents(self, docs):
            return [[float(len(d))] for d in docs]
    return IdentityEmbeddings()

def can_use_voice_transcription():
    """
    True if platform endpoint or flag for ChatGPT5 voice transcription is configured.
    Set CHATGPT5_VOICE=1 or provide CHATGPT5_VOICE_ENDPOINT for audio-to-text.
    """
    return bool(os.getenv("CHATGPT5_VOICE")) or bool(os.getenv("CHATGPT5_VOICE_ENDPOINT"))
