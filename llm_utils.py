# llm_utils.py
import os
import time
import base64
import threading
import traceback
import requests
from typing import Optional, List

_token_lock = threading.Lock()
_token_cache = {"token": None, "expires_at": 0}


# --- Token helpers (client_credentials autotoken support) ---
def _fetch_oauth_token(token_url: str, client_id: str, client_secret: str, scope: Optional[str] = None, timeout: int = 10):
    auth = base64.b64encode(f"{client_id}:{client_secret}".encode()).decode()
    headers = {"Authorization": f"Basic {auth}", "Content-Type": "application/x-www-form-urlencoded"}
    data = {"grant_type": "client_credentials"}
    if scope:
        data["scope"] = scope
    r = requests.post(token_url, headers=headers, data=data, timeout=timeout)
    r.raise_for_status()
    j = r.json()
    return j["access_token"], int(j.get("expires_in", 3600))


def _get_cached_token():
    static = os.getenv("CHATGPT5_AUTH_TOKEN")
    if static:
        return static

    token_url = os.getenv("CHATGPT5_TOKEN_URL")
    client_id = os.getenv("CHATGPT5_CLIENT_ID")
    client_secret = os.getenv("CHATGPT5_CLIENT_SECRET")
    if not (token_url and client_id and client_secret):
        return None

    with _token_lock:
        now = time.time()
        if _token_cache["token"] and now < _token_cache["expires_at"]:
            return _token_cache["token"]
        try:
            token, expires_in = _fetch_oauth_token(token_url, client_id, client_secret)
            _token_cache["token"] = token
            _token_cache["expires_at"] = now + expires_in - 60
            return token
        except Exception:
            traceback.print_exc()
            return None


# --- Low-level proxy forwarders (for upstream URLs) ---
def forward_to_upstream_json(endpoint: str, payload: dict, timeout: int = 30) -> dict:
    token = os.getenv("UPSTREAM_API_KEY") or os.getenv("CHATGPT5_AUTH_TOKEN")
    headers = {"Content-Type": "application/json"}
    if token:
        headers["Authorization"] = f"Bearer {token}"
    r = requests.post(endpoint, json=payload, headers=headers, timeout=timeout)
    r.raise_for_status()
    return r.json()


def forward_to_upstream_files(endpoint: str, files: dict, timeout: int = 60) -> dict:
    token = os.getenv("UPSTREAM_API_KEY") or os.getenv("CHATGPT5_AUTH_TOKEN")
    headers = {}
    if token:
        headers["Authorization"] = f"Bearer {token}"
    r = requests.post(endpoint, files=files, headers=headers, timeout=timeout)
    r.raise_for_status()
    return r.json()


# --- Provider adapters (unified interface used by other modules) ---
def provider_generate(prompt: str, max_tokens: int = 300, temperature: float = 0.2) -> str:
    """
    Generate text from the configured provider.
    Priority:
      1) OPENAI_API_KEY -> OpenAI ChatCompletion
      2) ANTHROPIC_API_KEY -> Anthropic completions
      3) CHATGPT5_PROXY/CHATGPT5_ENDPOINT -> platform-managed endpoint
    """
    # OpenAI
    openai_key = os.getenv("OPENAI_API_KEY")
    if openai_key:
        try:
            import openai
            openai.api_key = openai_key
            model = os.getenv("OPENAI_MODEL", "gpt-4")
            resp = openai.ChatCompletion.create(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                temperature=temperature,
                max_tokens=max_tokens,
            )
            try:
                return resp["choices"][0]["message"]["content"].strip()
            except Exception:
                return str(resp)
        except Exception:
            traceback.print_exc()

    # Anthropic
    anthropic_key = os.getenv("ANTHROPIC_API_KEY")
    if anthropic_key:
        try:
            from anthropic import Anthropic, HUMAN_PROMPT, AI_PROMPT
            client = Anthropic(api_key=anthropic_key)
            model = os.getenv("ANTHROPIC_MODEL", "claude-2")
            prompt_text = HUMAN_PROMPT + prompt + AI_PROMPT
            resp = client.create_completion(model=model, prompt=prompt_text, max_tokens_to_sample=max_tokens, temperature=temperature)
            return resp.get("completion", "").strip()
        except Exception:
            traceback.print_exc()

    # Proxy / platform endpoint
    proxy = os.getenv("CHATGPT5_PROXY") or os.getenv("CHATGPT5_ENDPOINT")
    if proxy:
        try:
            token = _get_cached_token()
            headers = {"Content-Type": "application/json"}
            if token:
                headers["Authorization"] = f"Bearer {token}"
            r = requests.post(proxy, json={"prompt": prompt}, headers=headers, timeout=30)
            r.raise_for_status()
            j = r.json()
            if isinstance(j, dict):
                if "response" in j:
                    return j["response"]
                if "output" in j:
                    return j["output"]
                if "choices" in j and j["choices"]:
                    first = j["choices"][0]
                    if isinstance(first, dict):
                        if "message" in first and "content" in first["message"]:
                            return first["message"]["content"]
                        if "text" in first:
                            return first["text"]
            return str(j)
        except Exception:
            traceback.print_exc()
            return ""

    raise RuntimeError("No provider configured. Set OPENAI_API_KEY, ANTHROPIC_API_KEY, or CHATGPT5_PROXY/CHATGPT5_ENDPOINT.")


def provider_transcribe(audio_bytes: bytes, filename: Optional[str] = None) -> str:
    """
    Transcribe audio. Prefers direct provider (if they support ASR) or a configured voice proxy endpoint.
    """
    # If a voice proxy is provided, prefer it (supports provider-side ASR)
    voice_proxy = os.getenv("CHATGPT5_VOICE_PROXY") or os.getenv("CHATGPT5_VOICE_ENDPOINT")
    if voice_proxy:
        try:
            token = _get_cached_token()
            headers = {}
            if token:
                headers["Authorization"] = f"Bearer {token}"
            files = {"file": (filename or "audio", audio_bytes)}
            r = requests.post(voice_proxy, files=files, headers=headers, timeout=120)
            r.raise_for_status()
            j = r.json()
            return j.get("transcript") or j.get("text") or str(j)
        except Exception:
            traceback.print_exc()

    # OpenAI direct (no voice proxy)
    openai_key = os.getenv("OPENAI_API_KEY")
    if openai_key:
        try:
            import openai
            openai.api_key = openai_key
            # Some OpenAI SDKs provide a transcription endpoint; many deployments use a separate API.
            # If you have a Whisper/Transcribe endpoint, prefer proxy; otherwise raise.
            raise RuntimeError("OpenAI transcription requires a voice proxy or specific endpoint. Configure CHATGPT5_VOICE_PROXY.")
        except Exception:
            traceback.print_exc()

    # Anthropic direct fallback (rare); prefer proxy as above
    anthropic_key = os.getenv("ANTHROPIC_API_KEY")
    if anthropic_key:
        raise RuntimeError("Anthropic transcription requires a voice proxy endpoint. Configure CHATGPT5_VOICE_PROXY.")

    raise RuntimeError("No transcription provider configured. Set CHATGPT5_VOICE_PROXY or a provider API key.")


# --- LLM wrapper used by bot.get_llm() pattern (keeps .predict for compatibility) ---
class ProviderLLM:
    def predict(self, prompt: str) -> str:
        try:
            return provider_generate(prompt, max_tokens=300, temperature=0.0)
        except Exception:
            traceback.print_exc()
            return "ERROR: provider_generate failed."


def get_llm():
    return ProviderLLM()


# --- Embeddings provider (sentence-transformers fallback) ---
def get_embeddings():
    try:
        from sentence_transformers import SentenceTransformer

        class STEmbedder:
            def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
                self.model = SentenceTransformer(model_name)

            def embed_documents(self, docs: List[str]) -> List[List[float]]:
                return self.model.encode(docs).tolist()

        return STEmbedder()
    except Exception:
        traceback.print_exc()

    class IdentityEmbedder:
        def embed_documents(self, docs: List[str]) -> List[List[float]]:
            return [[float(len(d))] for d in docs]

    return IdentityEmbedder()


# --- Voice availability helper ---
def can_use_voice_transcription() -> bool:
    return bool(os.getenv("CHATGPT5_VOICE")) or bool(os.getenv("CHATGPT5_VOICE_PROXY")) or bool(os.getenv("CHATGPT5_VOICE_ENDPOINT"))
