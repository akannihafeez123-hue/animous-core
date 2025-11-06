# Dockerfile — single-container image that runs proxy + bot + core via bot.py launcher
FROM python:3.11-slim

# Install minimal system deps needed for many Python packages
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential git curl ca-certificates \
  && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy dependency manifest first for layer caching
COPY requirements.txt /app/requirements.txt

# Install Python dependencies
RUN pip install --no-cache-dir -r /app/requirements.txt

# Copy application source
COPY . /app

# Ensure the launcher script exists and is executable
RUN test -f /app/bot.py && chmod +x /app/bot.py

ENV PORT=8080 \
    PYTHONUNBUFFERED=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    SINGLE_CONTAINER_MODE=1

EXPOSE 8080

# Run python directly (exec form) — avoids shell/run.sh dependency
CMD ["python", "-u", "bot.py"]
