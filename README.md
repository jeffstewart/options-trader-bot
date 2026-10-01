# Trader Bot: Event-Driven Options Trading System

A modular, multi-generation algorithmic trading framework designed to trade breaking financial news catalysts using the Alpaca Markets API. Paper trading only — built for quantitative research and execution engineering.

### Architecture Generations

- **`v3/` (Current Flagship)** — Context-aware options trading engine running in Docker on Colima. Features real-time Alpaca WebSocket news streaming (`NewsDataStream`), symmetrical LLM directional scoring (bullish Calls and bearish Puts), normalized contract grid telemetry (>90% storage reduction), an embedded SQLite news archive providing point-in-time ticker history and market sentiment flow, empirical contract selection (~0.50 Delta ATM calls, $\le 6.5\%$ spread cap, $\ge 10$ OI floor, 30m timed horizon exits), and a pluggable LLM scorer (local Ollama, Google Gemini, Anthropic Claude, Moonshot Kimi). See [`v3/README.md`](v3/README.md).
- **`v2/`** — Lean containerized options bot focused on far-OTM lotto experiments. See [`v2/README.md`](v2/README.md).
- **`core/` (v1 - SUNSET)** — The foundational engine that captured the initial 8.26-million-row options order-book research dataset (`contract_grid_snapshots.csv`). Now sunset; its research data collection tap has been moved into v3 with relational normalization.

The entire stack is orchestrated via [`everything.sh`](everything.sh).

---

## v1 (`core/`) — how it works

```
Alpaca/Benzinga news stream  +  SEC EDGAR 8-K RSS  +  NewsAPI
        │
        ▼
 (1) stale-news drop          older than MAX_FEED_LAG_SECS (~5 min) → skip (keeps the bot on live news)
        │
 (2) pre-score filters        drop BEFORE scoring — never worth an LLM call:
        │                       • noise: macro/index wraps, listicles ("stocks to buy")
        │                       • soft-catalyst: analyst/price-target, technical/chart, acquirer-side M&A
        │                       • index-inclusion: Russell/S&P/Dow reconstitution (no directional edge)
        ▼
 (3) Ollama scorer            local llama3.2 → {sentiment, magnitude, confidence, tickers}   (free, fast)
        │
 (4) grid collector tap       mag×conf ≥ 0.13 → fire-and-forget contract-grid snapshot (research only,
        │                       see below) — runs BEFORE the trade-eligibility checks below it
        │
 (5) signal checks            magnitude / confidence gates, cooldown, liquidity
        ▼
 (6) regime gate              long-beta paused when SPY < 200d SMA, < 3-day momentum, or the chop
        │                       brake trips (down-day density ≥60% over the last 10 SPY sessions)…
        │                       …EXCEPT the high-conviction BYPASS (currently DISABLED, see below)
        ▼
 (7) confirm/veto gate        mistral-large independently re-scores the SAME article (unified_v1 prompt):
        │                       • confirm (bullish & mag ≥ GEMINI_CONFIRM_MIN_MAGNITUDE) → trade for real
        │                       • veto                                                  → shadow-only, no trade
        │                       • cap / error                                           → fall back to Ollama
        ▼
 (8) option legs               pick ~0.50-delta / 14-21-DTE call → in-process trailing-stop monitor → exit
```

The pre-score filters and the regime/gate logic exist because the model **systematically over-scores
certain weak catalysts** (analyst notes, index inclusions, sector ETFs) to high magnitude. The
deterministic filters drop the known-bad categories cheaply; the **mistral-large gate** is the
learned second opinion that catches the rest. Both were validated with bootstrapped backtests (see
`research/`).

The high-conviction regime **bypass is currently disabled** (`REGIME_BYPASS_MIN_MAGNITUDE=1.01`,
effectively unreachable) — live forward data contradicted the backtest that justified it (see
`docs/HANDOFF.md`).

### Scoring stack (v1)

| Role | Model | Where | Notes |
|---|---|---|---|
| **Primary scorer** | `llama3.2` | local Ollama | scores every article; free, ~3–8s |
| **Confirm/veto gate** | `mistral-large` | Mistral API | re-scores real-trade-eligible bullish signals; ~2.7s |
| **A/B shadow** | Groq (configurable, currently off) | shadow only | logged to `scorer_ab.csv`, never trades |

Magnitude is **uncalibrated** — no model (incl. frontier) ranks it better than local llama. But a
*different-family* model used as a **disagreement gate** does add value (it catches losers the
primary over-hypes). See `docs/HANDOFF.md` for the full findings.

### Strategies (v1)

- **`news_call`** — long calls on bullish single-name catalysts.
- **`lotto`** — cheap far-OTM calls on very high-conviction events; late-day entry cutoff (30 min
  before close).
- **`pead`** — post-earnings-announcement drift (reserved position budget).
- **`pairs`** — market-neutral long/short equity pairs; **runs in all regimes** (carries down days
  when the directional book is paused).
- Shadow/off: `stock` (`STOCK_ENABLED=False`, real entries off), `bear_short`, `qqq_macro`.

### Contract-grid research collector (v1 only)

For every signal that clears `mag×conf ≥ GRID_LOG_THRESHOLD` (0.13 — the closest achievable
approximation of "90% of historically-bullish signals" on Ollama's clustered confidence
distribution; see `core/config.py` comments), regardless of whether it goes on to trade, the bot
snapshots bid/ask/greeks/OI for a wide DTE×strike grid of real contracts and re-polls that grid on
a decaying schedule (30s → 3min → 15min → 30min) through each contract's own expiry. Purely
market-data reads — no orders, no capital. Output: `data/contract_grid_snapshots.csv`. Goal: enough
data to evaluate the contract-picking logic (which of the available strikes/expiries would have
performed best) and the exit logic (how far past today's sell point contracts kept moving),
independent of what the bot actually happened to trade. See `core/grid_collector.py`.

Each row also carries the context needed for a later train/test analysis of what makes a good
contract pick: which scorer model produced the score (`scorer_model`, so results can be
recalibrated across model swaps), the LLM's `catalyst` sub-score, the underlying stock price at
signal time and at every poll, the day's SPY trend/momentum/chop-brake regime state, a VIX proxy
(see below), the news source, and whether the signal cleared each strategy's live threshold
(`passes_news_call_gate` / `passes_lotto_gate`). Timestamps are full datetimes, so day-of-week /
time-of-day are derivable without extra columns.

**VIX proxy**: true CBOE VIX isn't reachable on this Alpaca plan (no indices data endpoint).
`VIXY` (a normal equity, VIX-futures-based) is used as the closest available proxy, logged as its
**percentile rank within its own trailing 20-session window** (`vol_proxy_pctile`) rather than a
raw price or an SMA comparison — VIXY bleeds value over time from contango roll cost and periodic
reverse splits, so its long-run price trend is structural decay, not a volatility signal; the
percentile cancels that drift out.

**Join key to real trades (`grid_signal_id`)**: when a collected signal goes on to actually trade,
`trades.csv`/`closed_trades.csv` carry the same `grid_signal_id` the grid snapshots were logged
under — so analysis can compare what the grid says was the best available contract against what
the bot actually bought, its real fill, and its real exit (mechanism/timing/price), not just a
hypothetical best-in-grid pick with nothing to compare it against.

### Risk & guardrails (v1)

- **In-process trailing stops are the ONLY protection** — this account can't place exchange-held
  stop orders, so **a downed bot leaves positions unprotected.** Keep it running during market hours.
- **Daily-loss circuit breaker** — halts new trades past ~2% of equity/day (floor $2,000), resets
  00:00 UTC.
- **Flat position sizing** — magnitude is uncalibrated, so size is flat (`NONMAG_SIZE_FRAC ×
  MAX_POSITION_USD`), not magnitude-scaled.
- **Max open positions** cap (45); per-ticker cooldown.

---

## v2 (`v2/`) — how it's different

Lean rewrite for an eventual small real-money account: no confirm/veto gate, no pairs, only
`lotto` currently enabled, equity-relative sizing instead of a fixed dollar constant, and a
contract-mispricing switch that can redirect the target contract to a cheaper same-expiry neighbor.
Runs on its own paper account, in a Docker container. Full detail in [`v2/README.md`](v2/README.md).

---

## Project structure

```
core/        v1 runtime + engine — bot.py, config.py, dashboard.py, backtest.py, grid_collector.py,
             pricing.py, yahoo_data.py, router_eval.py, spawn_daemon.py, sec_edgar.py
v2/          v2 runtime + engine — bot.py, config.py, dashboard.py, execution.py, market.py,
             scoring.py, filters.py, pricing.py, sec_edgar.py, Dockerfile, docker-compose.yml
research/    ~180 backtest / analysis / tuning scripts (not needed to run either bot)
tests/       v1 pytest suite (filters + trading math + grid collector)
docs/        HANDOFF.md (full live-state writeup)
data/        v1 caches, CSVs, bot_state.json        ← gitignored, machine-local
v2/data/     v2 caches, CSVs, bot_state.json        ← gitignored, machine-local
logs/        v1 *.log                               ← gitignored
v2/logs/     v2 *.log                                ← gitignored
everything.sh  single entry point: ollama + v1 (native) + v2 (docker) + v2 dashboard (native)
manage.sh      v1-only daemon control (start/stop/status/restart-bot/test/backup/watchdog)
```

Code is imported via a `_trader_paths.pth` in the venv (`core/`+`research/` on `sys.path`), and
v1 daemons run with **CWD=`data/`** so data files resolve there. `config.DATA_DIR` / `LOG_DIR` are
the explicit path anchors. v2 is fully self-contained under `v2/` with its own `.env`, `data/`,
`logs/`, and `pytest.ini`.

---

## Setup

```bash
# 1. Python deps (venv — system pip is blocked on this machine)
.venv/bin/pip install -r requirements.txt
.venv/bin/pip install -r requirements-dev.txt    # pytest (dev only)

# 2. Local model (used by both v1's primary scorer and v2's ticker corrector)
ollama serve &        # then: ollama pull llama3.2

# 3. Credentials → .env  (never committed)
#    ALPACA_API_KEY / ALPACA_SECRET_KEY      paper trading + news + bars (v1 account)
#    MISTRAL_API_KEY / MISTRAL_BASE_URL      v1 confirm/veto gate
#    LAB_HOSTED_* (Groq)                     v1 A/B shadow scorer + research backtests
#
# v2 has its own separate .env under v2/ — see v2/README.md (different Alpaca paper account,
# MOONSHOT_API_KEY for the kimi-k2.6 scorer).
```

8GB-RAM note: only `llama3.2` (2GB) fits alongside the bots — larger local models swap and can
panic the machine. v1's confirm/veto gate is therefore cloud-based (Mistral) by necessity; v2's
primary scorer is cloud-based (kimi-k2.6) for the same reason.

---

## Running

Use `./everything.sh {start|stop|restart|status}` to bring up/down the **whole stack**: Colima
(Docker backend) → Ollama → v1 (bot+dashboard, native) → v3 bot (Docker container).

```bash
./everything.sh start
./everything.sh status
./everything.sh stop
```

For v1-only control (e.g. restarting just the bot after a config change), use `manage.sh` directly:

```bash
./manage.sh start        # start v1 bot + dashboard (daemonized, ppid 1, survive the shell)
./manage.sh sched        # start the backtest scheduler (optional, research)
./manage.sh status       # show which v1 daemons are up
./manage.sh restart-bot  # restart just the v1 bot
./manage.sh stop         # stop v1 only
./manage.sh test         # run the v1 pytest suite
./manage.sh watchdog     # idempotent auto-restart + caffeinate + log rotation (cron target)
./manage.sh backup       # code bundle + data/eod_reports → iCloud Drive
```

For v2-only control, use `docker compose` directly from `v2/` — see [`v2/README.md`](v2/README.md).

- v1 dashboard: `http://localhost:5001` (status, P&L, open positions w/ side, recent trades w/
  close reason, activity feed).
- v2 dashboard: runs natively on the host (not containerized) — see `v2/README.md` for port/run
  command.
- pids in `data/*.pid` (v1) / `v2/data/*.pid` (v2), logs in `logs/` / `v2/logs/`. Process match:
  `pgrep -f 'core/bot.py'`.
- Log timestamps are **local PDT**, not UTC.

---

## Configuration

Everything tunable for v1 lives in **`core/config.py`**; for v2, in **`v2/config.py`**. Each bot
has its own config, its own `.env`, and its own tuned parameters — they do not share state or
config beyond both reading from the same `research/` scripts when doing analysis.

---

## Testing

The project maintains comprehensive test coverage across 269 automated unit tests:

```bash
# v3 Flagship Suite (54 tests — filters, market regime, database, prompt formatting, execution rules)
pytest -c v3/pytest.ini v3/tests/ -v

# v2 Lean Suite (123 tests — lotto pipeline, filters, ratcheting stops, contract picker)
pytest -c v2/pytest.ini v2/tests/ -v

# v1 Core Suite (92 tests — news parsing, regime gate, order execution, sizing math)
pytest -c pytest.ini tests/ -v
```

All suites test deterministic logic, risk controls, and trading math without making external network calls. Run before any production deployment.

---

## Backtesting & research

The `research/` scripts replay history with Black-Scholes + IV-crush option pricing, real Alpaca
historical option trades/bars where available, and bootstrap P&L confidence intervals. They're how
every filter and both gates were validated. They import the live `core/`/`v2/` modules where
possible, so they test close to what actually trades.

---

## Safety

Paper trading only — built for research and education. Do not point either bot at real money. v1's
in-process trailing stop is the sole position protection for that account (no exchange-held stops
available); a downed v1 bot leaves positions unprotected. v2's risk model is documented in
`v2/README.md`.
