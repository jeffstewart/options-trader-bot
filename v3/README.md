# Trader Bot v3: Context-Aware Options Trading System

**Trader Bot v3** is an event-driven algorithmic options trading system designed to trade breaking financial news catalysts via the Alpaca Markets API. 

Unlike conventional single-article sentiment bots that evaluate headlines in isolation, **v3 maintains a point-in-time embedded SQLite news archive**. The scoring LLM receives the trailing sequence of prior news for the candidate ticker over a configurable lookback window (defaulting to 7 days) alongside an aggregate 24-hour market sentiment backdrop.

All execution parameters and contract selection rules are grounded in empirical findings from a **5-week, 7.5-million-row options order-book dataset** (`data/contract_grid_snapshots.csv`).

---

## 1. System Architecture

```text
Alpaca Real-Time News Stream / REST API
                  │
                  ▼
   ┌───────────────────────────────┐
   │ SQLite News DB (v3/news_db.py) │  <── Zero-overhead WAL mode
   │ • Point-in-time historical     │  <── 7-day automatic startup backfill
   │   lookback (zero lookahead)    │  <── 24h market sentiment flow
   └──────────────┬────────────────┘
                  │
                  ▼
   ┌───────────────────────────────┐
   │ Pre-Score Deterministic       │  <── Drops listicles ("10 stocks to buy")
   │ Filters (v3/filters.py)       │  <── Drops macro wraps, reactive recaps
   └──────────────┬────────────────┘  <── Drops analyst PT revisions & chart chatter
                  │
                  ▼
   ┌───────────────────────────────┐
   │ SPY Regime Gate               │  <── 200-day SMA uptrend
   │ (v3/market.py)                │  <── 3-day momentum check
   └──────────────┬────────────────┘  <── 10-day chop brake (<60% down-day density)
                  │
                  ▼
   ┌───────────────────────────────┐
   │ Context-Enriched LLM Scorer   │  <── Pluggable: Ollama (default), Gemini,
   │ (v3/scoring.py)               │      Anthropic, Moonshot
   │                               │  <── Trailing ticker news history context
   │                               │  <── Echo/recap & sympathy trap detection
   └──────────────┬────────────────┘
                  │ [Passes Conviction & Novelty Gate]
                  ▼
   ┌───────────────────────────────┐
   │ Empirical Execution Engine    │  <── ~0.50 Delta (ATM calls)
   │ (v3/execution.py)             │  <── Hard <= 6.5% Bid/Ask Spread Cap
   │                               │  <── >= 10 Open Interest Floor
   │                               │  <── 30-Minute Timed Horizon Exit
   └───────────────────────────────┘
```

---

## 2. Core Innovations in v3

### A. Point-in-Time Context Enrichment (`news_db.py`)
Evaluating a headline in a vacuum leads to false positives on post-earnings recaps, routine analyst commentary, or second-hand summaries. 
* **Zero Lookahead Bias**: SQLite queries enforce `created_at < signal_ts` to ensure only information known at the exact millisecond of the signal is provided.
* **Echo Detection**: If a company announced earnings two days ago, a new article discussing the same numbers is correctly flagged as `is_stale_echo: true` and downgraded to `magnitude: 0.20`.
* **Market Flow Backdrop**: Computes market-wide 24-hour sentiment distribution and extracts macro headlines to gauge broader market risk.

### B. Empirical Contract Selection (`execution.py`)
Developed from quantitative research over 7,586,577 option order-book snapshots across 652 signals (`research/contract_strategy_tuner.py`):
1. **Target Delta ~0.50 (ATM Calls)**: Outperformed far-OTM "lotto" calls across all holding horizons. Lotto calls suffered a 4% win rate and high IV crush.
2. **Hard Spread Cap ($\le 6.5\%$)**: Option bid/ask spread percentage was identified as the single strongest predictor of loss ($\rho = -0.532$, $p = 1.76 \times 10^{-27}$). Rejecting wide spreads cut option losses by ~80%.
3. **Open Interest Floor ($\ge 10$)**: Enforces baseline market liquidity to ensure slippage-free fills.
4. **30-Minute Timed Horizon Exit**: Quantitative analysis demonstrated edge decay after 30 minutes; automatic horizon exits lock in intraday momentum while avoiding multi-hour theta burn.

### C. Multi-Provider Scorer with Pacing & Quota Protection (`scoring.py`)
Supports hot-swapping between local and frontier LLMs via environment variables:
* **`ollama`**: Runs local open-weights models (`llama3.2`) via OpenAI-compatible endpoints at zero marginal cost.
* **`gemini`**: Supports Google Gemini models (`gemini-2.5-flash`, `gemini-1.5-flash`) with automatic pacing ($\le 13$ RPM), exponential backoff, and daily quota safety tracking.
* **`anthropic` & `moonshot`**: Full support for Claude 3.5 Sonnet and Kimi k2.6.

---

## 3. Empirical Research Findings

### Ollama vs. Gemini Head-to-Head Comparison
Replaying historical signals with 7-day news context revealed critical model behavior differences:

| Metric | Ollama (`llama3.2`) | Gemini (`gemini-2.5-flash`) |
| :--- | :---: | :---: |
| **Pass Gate Rate** | 100.0% (11/11) | **27.3% (3/11)** |
| **Rejection Rate** | 0.0% | **72.7% (8/11)** |
| **Cumulative Traded P&L** | -93.94% | **-19.70%** |
| **Losses Avoided** | 0.0% | **-74.24% saved** |

* **Ollama's Conviction Inversion**: Local 3B models tend to over-score retail-hyped sensational headlines ("Federal Reserve of AI", "SpaceX sympathy merger with Tesla") to $\ge 0.80$, taking positions at the peak of IV.
* **Gemini's Context Discrimination**: Gemini successfully identified 6 of the 8 losing trades as **stale echoes** of prior news in the SQLite archive, and flagged 1 trade as an unverified **sympathy trap**, preventing -74.2% in options losses.

---

## 4. Containerized Deployment (Colima / Docker)

The v3 bot is packaged as a single lightweight container managed via Docker Compose:

```bash
# Build and run detached on Colima:
docker compose up -d --build

# Follow live trading logs:
docker compose logs -f

# Gracefully stop (sends SIGTERM, closes active sockets):
docker compose down
```

### Key Deployment Features:
* **Host Ollama Bridge**: Uses `host.docker.internal:host-gateway` to allow the containerized bot to communicate with host Ollama at zero latency.
* **Persistent Bind Mounts**: `./data` and `./logs` are mounted from the host, preserving `news.db` and transaction records across container restarts.
* **DNS Resilience**: Configured with fallback public resolvers (`1.1.1.1`, `8.8.8.8`) with fast timeouts to avoid macOS local DNS proxy flapping.

---

## 5. Testing & Quality Assurance

v3 contains an isolated unit test suite covering filters, market regime math, database operations, prompt formatting, and execution rules:

```bash
# Run the 54 unit tests for v3:
.venv/bin/pytest -c v3/pytest.ini v3/tests/ -v
```

### Test Suite Structure:
* **`test_filters.py`** (33 tests): Validates HTML normalization, listicle detection, macro wraps, reactive recaps, analyst PT revisions, chart chatter, and ensures genuine catalysts pass cleanly.
* **`test_market.py`** (6 tests): Validates SPY 200d SMA, 3-day momentum, 10-day chop brake density, option spread calculations, and the 2% daily loss circuit breaker.
* **`test_execution.py`** (6 tests): Validates contract guardrails (spread cap, OI floor, root symbol), 30-minute timed horizon exit triggers, and state persistence.
* **`test_news_db.py`** (4 tests): Validates SQLite schema, deduplication, point-in-time isolation, and 24h market sentiment aggregation.
* **`test_scoring_prompt.py`** (5 tests): Validates prompt assembly with and without context, JSON extraction with code fences, and Gemini daily quota guardrails.

