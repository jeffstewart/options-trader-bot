# v2 trading bot — single always-on container (docs/HANDOFF.md's go-live plan: a stateful
# singleton, no k8s/serverless — reproducibility + managed auto-restart is the point, not scaling).
# Builds bot.py only; the dashboard is meant to keep running on the host against the same
# bind-mounted data/logs volumes (see docker-compose.yml), not inside this container.
FROM python:3.13-slim

WORKDIR /app

# Install deps first so this layer only rebuilds when requirements.txt actually changes.
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# Bare-filename data writes (trades.csv, bot_state.json, sec_cik_tickers.json, ...) resolve
# relative to CWD, matching the same convention v1's spawn_daemon.py uses on the host (CWD=data/).
# TRADER_DATA_DIR/TRADER_LOG_DIR are set here for the pieces of config.py that DO read them
# explicitly; the CWD below is what actually matters for the bare-filename call sites.
ENV TRADER_DATA_DIR=/app/data
ENV TRADER_LOG_DIR=/app/logs
RUN mkdir -p /app/data /app/logs
WORKDIR /app/data

# Graceful SIGTERM shutdown is already wired into bot.py's main() (closes both data-stream
# connections cleanly before exit) -- this is exactly what lets `docker stop`/a platform restart
# avoid the "connection limit exceeded" cooldown that an ungraceful kill causes.
CMD ["sh", "-c", "python -u /app/bot.py 2>&1 | tee -a /app/logs/bot.log"]
