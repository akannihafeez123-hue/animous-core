# app.py
import os, time, base64, requests
from fastapi import FastAPI, Request, UploadFile, File, HTTPException

from llm_utils import get_autotoken, forward_to_upstream_json, forward_to_upstream_files

app = FastAPI()

# If you want the proxy to forward to an upstream provider directly,
# set UPSTREAM_TEXT_URL and UPSTREAM_VOICE_URL in Choreo env.
UPSTREAM_TEXT_URL = os.getenv("UPSTREAM_TEXT_URL", "")
UPSTREAM_VOICE_URL = os.getenv("UPSTREAM_VOICE_URL", "")
BOT_PROXY_KEY = os.getenv("BOT_PROXY_KEY", "")  # optional API key for incoming bot calls

def require_proxy_key(incoming_key):
    if BOT_PROXY_KEY and incoming_key != BOT_PROXY_KEY:
        raise HTTPException(status_code=401, detail="invalid proxy key")

@app.post("/v1/generate")
async def generate(request: Request):
    incoming_key = request.headers.get("X-Bot-Proxy-Key", "")
    require_proxy_key(incoming_key)
    body = await request.json()
    prompt = body.get("prompt")
    if not prompt:
        raise HTTPException(status_code=400, detail="prompt required")

    # If UPSTREAM_TEXT_URL is set, forward raw to that URL
    if UPSTREAM_TEXT_URL:
        try:
            return forward_to_upstream_json(UPSTREAM_TEXT_URL, {"prompt": prompt})
        except Exception as e:
            raise HTTPException(status_code=502, detail=str(e))

    # Otherwise try to call provider via llm_utils' unified helper
    try:
        resp = {}
        # llm_utils exports a helper that will use configured provider
        from llm_utils import provider_generate
        text = provider_generate(prompt)
        return {"response": text}
    except Exception as e:
        raise HTTPException(status_code=502, detail=str(e))


@app.post("/v1/transcribe")
async def transcribe(file: UploadFile = File(...), request: Request = None):
    incoming_key = request.headers.get("X-Bot-Proxy-Key", "") if request else ""
    require_proxy_key(incoming_key)
    if UPSTREAM_VOICE_URL:
        content = await file.read()
        files = {"file": (file.filename, content, file.content_type)}
        try:
            return forward_to_upstream_files(UPSTREAM_VOICE_URL, files)
        except Exception as e:
            raise HTTPException(status_code=502, detail=str(e))

    # Otherwise use provider transcription helper if available
    try:
        from llm_utils import provider_transcribe
        # save to memory then call provider
        tmp = await file.read()
        # provider_transcribe expects bytes
        text = provider_transcribe(tmp, filename=file.filename)
        return {"transcript": text}
    except Exception as e:
        raise HTTPException(status_code=502, detail=str(e))
