# Dockerfile — single-container image that runs proxy + bot + core via bot.py launcher
FROM python:3.11-slim

# System deps
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential git curl ca-certificates openssl \
  && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy dependency manifest first for caching
COPY requirements.txt /app/requirements.txt

# Install Python dependencies
RUN pip install --no-cache-dir -r /app/requirements.txt

# Copy application code (ensure bot.py is at repo root)
COPY . /app

# Make sure bot.py is executable (convenience)
RUN chmod +x /app/bot.py

ENV PORT=8080 \
    PYTHONUNBUFFERED=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    SINGLE_CONTAINER_MODE=1

EXPOSE 8080

# Start the integrated launcher inside bot.py
CMD ["bash", "-lc", "python -u bot.py"]
