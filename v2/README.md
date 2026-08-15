# Trader Bot v2

A lean rebuild for the eventual $500-1000 real-money deploy. Runs independently of `core/`
(v1) — v1 keeps running unmodified as the ongoing data source / safety net while v2 is free to
diverge, on its own separate paper account. See
`~/.claude/projects/-Users-jeff-Claude-Trader/memory/project_v2_small_account_architecture.md`
for the full research behind every decision below.

## What's different from v1

- **No confirm/veto gate.** Net-negative even on Ollama's own picks on rerun, and a clear
  net-negative once sonnet5's picks were tested (cuts P&L ~14% by vetoing 80%-win-rate trades).
  Also removes the biggest latency source in the pipeline (~20s gate timeout).
- **No pairs (market-neutral L/S).** Structurally incompatible with a small account: ~66
  concurrent legs needed in steady state vs. any realistic 3-5 position budget, and 61-76% of its
  own picks are priced above what a small leg budget can afford for even 1 share.
- **Regime gate runs before any scoring call**, not after. With pairs/bear_short/qqq_macro all
  gone, nothing left needs scoring during a downtrend, so this is a clean gate rather than a
  two-stage compromise. Same 200d-SMA + 3d-momentum + chop-brake logic as v1.
- **Equity-relative position sizing**, not a fixed dollar constant (`LOTTO_POSITION_FRAC_OF_EQUITY`
  in `config.py`). v1's `MAX_POSITION_USD=$1,000` was ~the entire target account in one trade.
  Also fixes a real bug: v1's daily-loss-breaker fail-safe floor ($2,000) exceeded the whole target
  account, so if the equity fetch itself failed at exactly that moment the breaker gave zero
  protection.
- **Ticker-selection corrector** (`TICKER_CORRECTOR_ENABLED`): a second, cheap Ollama call that
  picks the primary beneficiary from the feed's own `symbols` metadata instead of trusting the
  primary scorer's free-generated ticker. Fixes the invented-ticker and wrong-beneficiary failure
  modes directly implicated in real live losses. Never used as a trade/no-trade gate.
- **Contract-mispricing switch** (`MISPRICING_SWITCH_ENABLED`, `execution.py`): if a same-expiry
  neighbor's implied vol sits meaningfully (`MISPRICING_MIN_RESIDUAL_GAP`) below a local IV-smile
  fit, the bot switches the target contract to the cheaper neighbor instead of the original pick.
  Wired live (previously shadow-only) — see `[[project_lotto_contract_mispricing]]` memory for the
  backing evidence and its caveats (small sample, not fully validated).
- **Lotto entry cutoff**: no new lotto positions opened inside `LOTTO_ENTRY_CUTOFF_MIN_BEFORE_CLOSE`
  (30 min) of the close — checked both when scoring the signal and again before order placement.
- **Currently lotto-only.** `NEWS_CALL_ENABLED=False` and `PEAD_ENABLED=False` in `config.py` —
  the small-account sweep found no positive-Sharpe cell for either at any threshold on the scorers
  tested. "Stock" as its own strategy was never implemented in v2. Revisit alongside future scorer
  or sizing work, not before.
- **Scorer: kimi-k2.6** (Moonshot API, OpenAI-compatible), not local Ollama. Switched from Ollama →
  sonnet5 → kimi-k2.6 as Anthropic credits ran out; matched-trade-count testing found no P&L
  separation between sonnet5 and kimi-k2.6 on the same prompt (see `[[project_kimi_scorer]]`
  memory), so the swap needed no threshold retuning. `scoring.py` isolates the scorer call to one
  function — swapping back to Anthropic or to Ollama is a one-line `SCORER_MODEL` change. Ollama is
  still used locally for the ticker corrector above.
- **No shadow-trade infrastructure by default.** Most shadow branches were superseded by
  backtesting the historical cache directly (faster, no weeks-long wait). Add a narrow one only for
  a question backtesting genuinely can't answer.

## Not yet finalized

Position-size fraction and max open positions (`MAX_OPEN_POSITIONS=3` today) are working values,
not a final small-account backtest pass. Revisit before any real-money move.

## Running it

```bash
pip install -r requirements.txt
cp .env .env.bak   # if you already put real keys in .env, back it up first
# edit .env: fill in a FRESH Alpaca PAPER account's ALPACA_API_KEY / ALPACA_SECRET_KEY
#            (not v1's account — this runs side-by-side on its own book)
#            + MOONSHOT_API_KEY for the kimi-k2.6 scorer
ollama serve &
ollama pull llama3.2   # used by the ticker corrector only, not the primary scorer
python bot.py
```

Tests: `pytest` (self-contained — own `pytest.ini`, own `tests/conftest.py`).

## Running it in a container

Only `bot.py` is containerized; `dashboard.py` runs on the host with the normal venv
(`../.venv/bin/python dashboard.py`) — it reads the same bind-mounted `data/`/`logs/` files, so
nothing about it changes. Both v1 and v2 (plus Ollama, plus the Docker/Colima backend itself) can
be brought up together with `../everything.sh start` from the repo root, which wraps the commands
below.

```bash
ollama serve &                 # Ollama stays on the HOST, not in the container
ollama pull llama3.2
docker compose up -d --build   # build the image, start detached, auto-restart on crash/reboot
docker compose logs -f         # follow live output (also written to ./logs/bot.log)
docker compose down            # SIGTERM -> bot.py's graceful shutdown -> stop
```

Gotchas specific to running in a container (not present when running `python bot.py` directly):
- **Ollama must be reached via `host.docker.internal`, not `localhost`** — "localhost" inside the
  container means the container itself. `docker-compose.yml` already sets
  `OLLAMA_BASE_URL=http://host.docker.internal:11434/v1` and adds the Linux-compatible
  `extra_hosts` mapping (Docker Desktop / Colima on Mac support this hostname natively).
- **`.env` is never baked into the image** (`.dockerignore` excludes it) — real keys are injected
  at `docker compose up` time via `env_file`, not present in the image itself even if it's later
  pushed to a registry.
- **`DRY_RUN=1` for a supervised test run**: `DRY_RUN=1 docker compose up --build` (or add it to
  `environment:` in `docker-compose.yml` temporarily) — logs "would trade" instead of placing real
  orders. Normal operation leaves it unset (real paper trades).
- **Data/logs are bind-mounted, not baked in** (`./data:/app/data`, `./logs:/app/logs`) — deleting
  the container/image never touches `bot_state.json`/`trades.csv`/etc.
- **Docker backend is Colima, not Docker Desktop** (switched to reduce idle RAM overhead — see
  root `everything.sh`). Colima does not survive a machine restart; `everything.sh start` brings it
  back up automatically, a bare `docker compose up` after a reboot will fail until Colima is
  running.

Next step when this actually moves to real money: deploy the same image to a small always-on host
(Fly.io or a small VM with `systemd Restart=always`) and move stops broker-side. Not done yet —
this is still the paper account, still local.
