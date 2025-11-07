# Build Configuration Issues & Fixes

## Critical Issues Found

### 1. **requirements.txt filename problem**
- File is named `requirements (1).txt` (with space and parentheses)
- Should be `requirements.txt` (no spaces)
- **Impact**: Build systems won't find dependencies

### 2. **Missing Pydantic v2 compatibility**
```python
# BROKEN (Pydantic v1 syntax)
from pydantic import BaseSettings

# SHOULD BE (Pydantic v2)
from pydantic_settings import BaseSettings
```

### 3. **Dependency conflicts**
- `fastapi>=0.95.2` is too old for modern Pydantic
- `langchain>=0.1.0` conflicts with sentence-transformers
- Torch with transformers will cause huge image size

### 4. **Choreo-specific issues**
- Missing `PORT` environment variable handling
- No health check timeout configuration
- Missing buildpack hints

### 5. **Cloudflare Workers incompatibility**
- Cannot run Python FastAPI on Cloudflare Workers
- Workers only support JavaScript/Wasm
- File size would exceed 10MB limit with all dependencies

---

## Fixed Files

### ✅ Fixed `requirements.txt`
```txt
# Core web framework
fastapi>=0.109.0
uvicorn[standard]>=0.27.0
pydantic>=2.5.0
pydantic-settings>=2.1.0

# HTTP client
requests>=2.31.0
httpx>=0.26.0

# Environment management
python-dotenv>=1.0.0

# Telegram bot
python-telegram-bot>=20.5

# Optional AI clients (comment out if not needed)
# openai>=1.0.0
# anthropic>=0.26.0

# Lightweight utilities
psutil>=5.9.5

# Testing
pytest>=7.4.0

# Note: Heavy ML dependencies removed for faster builds
# If you need ML features, add them selectively:
# sentence-transformers>=2.2.2
# torch>=2.1.2
# transformers>=4.40.0
# langchain>=0.1.0
# faiss-cpu>=1.7.4
```

### ✅ Fixed `service.py` (Pydantic v2 compatible)
```python
# Change line 32 from:
from pydantic import BaseSettings

# To:
from pydantic_settings import BaseSettings

# Update Settings class (around line 35):
class Settings(BaseSettings):
    PORT: int = 8080
    LOGDIR: str = "/tmp/logs"
    BOT_PROXY_KEY: Optional[str] = None
    UPSTREAM_TEXT_URL: Optional[str] = None
    ROTATE_NOTIFY_URL: Optional[str] = None
    TELEGRAM_BOT_TOKEN: Optional[str] = None
    BOT_POLL_INTERVAL: float = 1.0
    CORE_SLEEP_SECONDS: float = 5.0
    DEBUG: bool = False
    MEMORY_PERSIST: Optional[str] = None

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        case_sensitive = True
        
    def __init__(self, **kwargs):
        # Initialize with environment variables
        super().__init__(**kwargs)
        self.PORT = int(os.getenv("PORT", self.PORT))
        self.DEBUG = os.getenv("DEBUG", "0") in ("1", "true", "True")
        self.BOT_POLL_INTERVAL = float(os.getenv("BOT_POLL_INTERVAL", self.BOT_POLL_INTERVAL))
        self.CORE_SLEEP_SECONDS = float(os.getenv("CORE_SLEEP_SECONDS", self.CORE_SLEEP_SECONDS))
```

### ✅ Choreo-specific `Procfile`
```
web: python -u service.py
```

### ✅ Choreo `component.yaml`
```yaml
schemaVersion: 1.0
type: Service
name: unified-service
displayName: Unified Service
version: 1.0.0

endpoints:
  - name: api
    port: 8080
    type: REST
    networkVisibility: Public

buildConfig:
  buildpack:
    name: python
    version: "3.11"
  
env:
  - name: PORT
    default: "8080"
  - name: DEBUG
    default: "0"
  - name: LOGDIR
    default: "/tmp/logs"
```

### ✅ Updated `.env` (fix placeholders)
```bash
# Service selection
USE_GEMINI=1

# -------------------------
# Google Gemini (AI Studio)
# -------------------------
UPSTREAM_API_KEY=YOUR_ACTUAL_API_KEY_HERE
UPSTREAM_TEXT_URL=https://generativelanguage.googleapis.com/v1beta/models/gemini-pro:generateContent
UPSTREAM_VOICE_URL=

# -------------------------
# Proxy / Bot / Core shared secrets and config
# -------------------------
BOT_PROXY_KEY=GENERATE_RANDOM_32_CHAR_STRING_HERE

# Set after deploying to Choreo
CHATGPT5_PROXY=https://your-app.choreoapis.dev/v1/generate
CHATGPT5_VOICE_PROXY=https://your-app.choreoapis.dev/v1/transcribe

# -------------------------
# Telegram bot settings
# -------------------------
TELEGRAM_BOT_TOKEN=YOUR_BOT_TOKEN_FROM_BOTFATHER
ADMIN_CHAT=YOUR_TELEGRAM_USER_ID

# -------------------------
# Core worker settings
# -------------------------
CORE_SLEEP_SECONDS=300
AUTO_LEARN_ENABLED=true

# -------------------------
# Local development
# -------------------------
SINGLE_CONTAINER_MODE=1
PORT=8080
DEBUG=0
```

---

## Platform-Specific Guidance

### 🟢 **For Choreo Deployment**

1. **Rename the requirements file**:
   ```bash
   mv "requirements (1).txt" requirements.txt
   ```

2. **Update imports in service.py**:
   - Change `BaseSettings` import as shown above

3. **Create Procfile** (if not exists):
   ```
   web: python -u service.py
   ```

4. **Add to Choreo Secrets** (not .env):
   - `UPSTREAM_API_KEY`
   - `BOT_PROXY_KEY`
   - `TELEGRAM_BOT_TOKEN`

5. **Build command**: (automatic, but for reference)
   ```bash
   pip install -r requirements.txt
   ```

### 🔴 **For Cloudflare (NOT COMPATIBLE)**

**Your current setup CANNOT run on Cloudflare Workers because:**
- Workers only support JavaScript/TypeScript/Wasm
- No Python runtime available
- 10MB bundle size limit (your deps are 500MB+)

**Options:**
1. **Use Cloudflare Pages Functions** (limited Python support via Pyodide)
2. **Deploy to Cloudflare Workers with a JavaScript rewrite**
3. **Use Railway, Render, or Fly.io instead** (better Python support)

---

## Quick Fix Checklist

- [ ] Rename `requirements (1).txt` → `requirements.txt`
- [ ] Update `service.py` line 32: `from pydantic_settings import BaseSettings`
- [ ] Add `pydantic-settings>=2.1.0` to requirements.txt
- [ ] Update fastapi to `>=0.109.0` in requirements.txt
- [ ] Create `Procfile` with `web: python -u service.py`
- [ ] Set secrets in Choreo dashboard (not .env)
- [ ] Remove or comment out heavy ML dependencies if unused
- [ ] Test locally with `python -u service.py`

---

## Build Test Commands

```bash
# Test locally first
python -m venv venv
source venv/bin/activate  # or venv\Scripts\activate on Windows
pip install -r requirements.txt
python -u service.py

# Should see:
# INFO Starting uvicorn on port 8080
# Test: curl http://localhost:8080/health
```
