# Dockerfile — single-container image that runs proxy + bot + core via run.sh
FROM python:3.11-slim

# Install system dependencies required for building wheels and runtime
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    git \
    curl \
    ca-certificates \
    openssl \
  && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy dependency manifest first for efficient layer caching
COPY requirements.txt /app/requirements.txt

# Install Python dependencies
RUN pip install --no-cache-dir -r /app/requirements.txt

# Copy application code and launcher
COPY . /app

# Ensure launcher is executable
RUN chmod +x /app/run.sh

# Sensible runtime environment variables (override in Choreo)
ENV PORT=8080 \
    PYTHONUNBUFFERED=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    SINGLE_CONTAINER_MODE=1

EXPOSE 8080

# Default entrypoint: launcher starts proxy (uvicorn), bot, and core
CMD ["bash", "/app/run.sh"]
