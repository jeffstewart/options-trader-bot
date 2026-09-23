# v3 trading bot — single always-on container on Colima
FROM python:3.13-slim

WORKDIR /app

# Install deps first to leverage Docker layer caching
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application source code
COPY . .

ENV TRADER_V3_DATA_DIR=/app/data
ENV TRADER_V3_LOG_DIR=/app/logs
ENV PYTHONPATH=/app

RUN mkdir -p /app/data /app/logs
WORKDIR /app/data

# Graceful SIGTERM shutdown is wired into bot.py's signal handler
CMD ["sh", "-c", "python -u /app/bot.py 2>&1 | tee -a /app/logs/bot.log"]
