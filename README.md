# News-Driven Options Trading Bot

An event-driven bot that reacts to financial news in real time, scores each headline with a **local
LLM**, gates the strongest bullish signals through an **independent cloud confirm/veto model**, and
trades **long call options** on equities + ETFs via **Alpaca paper trading**. Options-only (crypto
removed). Paper account — for research/education, not real money.

---

## How it works

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
 (4) signal checks            magnitude / confidence gates, cooldown, liquidity
        ▼
 (5) regime gate              long-beta paused when SPY < 200d SMA or < 3-day momentum…
        │                       …EXCEPT the high-conviction BYPASS (magnitude ≥ REGIME_BYPASS_MIN_MAGNITUDE)
        ▼
 (6) confirm/veto gate        mistral-large independently re-scores the SAME article (unified_v1 prompt):
        │                       • confirm (bullish & mag ≥ GEMINI_CONFIRM_MIN_MAGNITUDE) → trade for real
        │                       • veto                                                  → shadow-only, no trade
        │                       • cap / error                                           → fall back to Ollama
        ▼
 (7) option legs              pick ~0.40-delta / ~10-DTE call → in-process trailing-stop monitor → exit
```

The pre-score filters and the regime/gate logic exist because the model **systematically over-scores
certain weak catalysts** (analyst notes, index inclusions, sector ETFs) to high magnitude. The
deterministic filters drop the known-bad categories cheaply; the **mistral-large gate** is the
learned second opinion that catches the rest (it independently judged 6 of 7 vetoes correct on its
first live day). Both were validated with bootstrapped backtests (see `research/`).

---

## Scoring stack

| Role | Model | Where | Notes |
|---|---|---|---|
| **Primary scorer** | `llama3.2` | local Ollama | scores every article; free, ~3–8s |
| **Confirm/veto gate** | `mistral-large` | Mistral API | re-scores real-trade-eligible bullish signals; ~2.7s |
| **A/B shadow** | `llama-3.3-70b` (Groq) | shadow only | logged to `scorer_ab.csv`, never trades — data for a possible cloud-primary move |

Why a gate instead of a better primary? Magnitude is **uncalibrated** — no model (incl. frontier)
ranks it better than local llama. But a *different-family* model used as a **disagreement gate** does
add value (it catches losers the primary over-hypes). See `docs/HANDOFF.md` for the full findings.

---

## Strategies

- **`news_call`** — long calls on bullish single-name catalysts (the core strategy).
- **`lotto`** — cheap far-OTM calls on very high-conviction events.
- **`pead`** — post-earnings-announcement drift (reserved position budget).
- **`pairs`** — market-neutral long/short equity pairs; **runs in all regimes** (carries down days when the directional book is paused).
- Shadow/off: `stock`, `bear_short`, `qqq_macro` (paper-tracked, not live).

---

## Risk & guardrails

- **In-process trailing stops are the ONLY protection** — this account can't place exchange-held stop orders, so **a downed bot leaves positions unprotected.** Keep it running during market hours.
- **Daily-loss circuit breaker** — halts new trades past ~2% of equity/day (resets 00:00 UTC).
- **Flat position sizing** — magnitude is uncalibrated, so size is flat (`NONMAG_SIZE_FRAC × MAX_POSITION_USD`), not magnitude-scaled.
- **Max open positions** cap; per-ticker cooldown.

---

## Project structure

```
core/        runtime + engine — bot.py, config.py, dashboard.py, backtest.py,
             pricing.py, yahoo_data.py, router_eval.py, spawn_daemon.py
research/    ~110 backtest / analysis / tuning scripts (not needed to run the bot)
tests/       pytest suite (filters + trading math)
docs/        HANDOFF.md (full live-state writeup), BOT_FLOW.md
data/        caches, CSVs, bot_state.json        ← gitignored, machine-local
logs/        *.log                               ← gitignored
config.py    → core/config.py is the single source of truth for every tunable
```

Code is imported via a `_trader_paths.pth` in the venv (`core/`+`research/` on `sys.path`), and
daemons run with **CWD=`data/`** so data files resolve there. `config.DATA_DIR` / `LOG_DIR` are the
explicit path anchors.

---

## Setup

```bash
# 1. Python deps (venv — system pip is blocked on this machine)
.venv/bin/pip install -r requirements.txt
.venv/bin/pip install -r requirements-dev.txt    # pytest (dev only)

# 2. Local model
ollama serve &        # then: ollama pull llama3.2   (the primary scorer; runs locally)

# 3. Credentials → .env  (never committed)
#    ALPACA_API_KEY / ALPACA_SECRET_KEY      paper trading + news + bars
#    MISTRAL_API_KEY / MISTRAL_BASE_URL      confirm/veto gate
#    LAB_HOSTED_* (Groq)                     A/B shadow scorer + research backtests
```

8GB-RAM note: only `llama3.2` (2GB) fits alongside the bot — larger local models swap and can panic
the machine. The confirm/veto gate is therefore cloud-based by necessity (see HANDOFF).

---

## Running

Use `manage.sh` (wraps the now-verbose absolute-path daemon launches):

```bash
./manage.sh start        # start bot + dashboard (daemonized, ppid 1, survive the shell)
./manage.sh sched        # start the backtest scheduler (optional, research)
./manage.sh status       # show which daemons are up
./manage.sh restart-bot  # restart just the bot (e.g. after a config change)
./manage.sh stop         # stop everything
./manage.sh test         # run the pytest suite
```

- Dashboard: `http://localhost:5001` (status, P&L, open positions w/ side, recent trades w/ close reason, activity feed).
- pids in `data/*.pid`, logs in `logs/`. Process match: `pgrep -f 'core/bot.py'`.
- Log timestamps are **local PDT**, not UTC.

---

## Configuration

Everything tunable lives in **`core/config.py`** (imported by both the live bot and the backtests).
Key knobs: the pre-score pattern lists (`PRESCORE_SKIP_PATTERNS`, `SOFT_CATALYST_PATTERNS`), the
regime filter (`REGIME_MA_DAYS`, `REGIME_MOMENTUM_DAYS`, `REGIME_BYPASS_MIN_MAGNITUDE`), the gate
(`GEMINI_MODEL`, `GEMINI_CONFIRM_MIN_MAGNITUDE`), and sizing/risk limits. Most accept `.env`
overrides — see the comments in `config.py`.

---

## Testing

`pytest` (config in `pytest.ini`) covers the pure-logic core — the pre-score filters and the trading
math (realized P&L incl. short-side sign, flat sizing, hold caps). Run before any change:

```bash
./manage.sh test
```

---

## Backtesting & research

The `research/` scripts replay history with Black-Scholes + IV-crush option pricing and bootstrap
P&L confidence intervals. They're how every filter and the gate were validated (e.g.
`index_event_backtest.py`, `regime_bypass_option_pnl.py`, `overnight_confirm_veto.py`). They import
the live `core/` modules, so they test exactly what trades.

---

## Safety

Paper trading only — built for research and education. Do not point this at real money. The bot will
not trade outside market hours; the in-process trailing stop is the sole position protection (no
exchange-held stops on this account).
