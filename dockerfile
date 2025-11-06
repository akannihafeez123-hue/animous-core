# Dockerfile — single-container entry for proxy + bot + core via run.sh
FROM python:3.11-slim

# Install system dependencies required to build Python packages and for runtime
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    git \
    curl \
    ca-certificates \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy dependency manifest first for efficient layer caching
COPY requirements.txt /app/requirements.txt

# Install Python dependencies
RUN pip install --no-cache-dir -r /app/requirements.txt

# Copy application code
COPY . /app

# Ensure run script is executable
RUN chmod +x /app/run.sh

# Recommended runtime environment variables (can be overridden in Choreo)
ENV PORT=8080 \
    PYTHONUNBUFFERED=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# Expose port for the proxy (Choreo will map this)
EXPOSE 8080

# Use the single-entrypoint launcher created earlier
CMD ["bash", "/app/run.sh"]
