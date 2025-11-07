# =========================================================
# 🧩 Institutional AI Trading Bot — Choreo Ready
# =========================================================
FROM python:3.11-slim

# Prevent Python from writing .pyc files & buffering stdout
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# Set work directory
WORKDIR /app

# Install OS dependencies
RUN apt-get update && apt-get install -y \
    gcc g++ libatlas-base-dev liblapack-dev gfortran \
    && rm -rf /var/lib/apt/lists/*

# Copy dependency list
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy all project files
COPY . .

# Expose Flask port for Choreo health check
EXPOSE 8080

# Default command — runs your main trading bot
CMD ["python", "bot.py"]
