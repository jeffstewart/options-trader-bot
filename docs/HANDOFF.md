# Trader Bot — Session Handoff

**Purpose of this file:** a self-contained context dump so another Claude
session (Cowork or otherwise) with file access to this Mac can pick up the work
without the prior conversation. Last updated **2026-06-23** (live-state block at top; sections below are 2026-06-01 history).

Working directory: `/Users/jeff/Claude/Trader`

---

## ⚠️ CURRENT LIVE STATE (2026-06-24) — read this first; everything below is from 2026-06-01 and is STALE

**🗂 RESTRUCTURED 2026-06-24 — the project is now a git repo with an organized layout (was flat):**
- `core/` runtime+engine (bot.py, config.py, dashboard.py, backtest.py, pricing.py, yahoo_data.py, router_eval.py, spawn_daemon.py) · `research/` ~112 backtest/analysis scripts · `tests/` pytest suite · `docs/` (this file) · `data/` caches+CSVs+`bot_state.json` (gitignored) · `logs/` (gitignored).
- **Imports** work via `_trader_paths.pth` in the venv site-packages (puts `core/`+`research/` on sys.path) — so flat `import bot`/`import config` still work from anywhere. **Daemons run with CWD=`data/`** (set by `spawn_daemon.py` WORKDIR) so bare data-file refs resolve into `data/`. `config.DATA_DIR`/`LOG_DIR` are the explicit anchors.
- **Manage daemons with `./manage.sh {start|sched|stop|restart-bot|status|test}`** (wraps the now-verbose absolute-path launch commands). pids in `data/*.pid`, logs in `logs/`.
- **git:** `main` branch; code+docs tracked, `.env`/data/logs/caches gitignored. Commit going forward.

**Running:** bot, dashboard, `resume_after_reset.py` (backtest scheduler), and the backtest are daemons (ppid 1) via `core/spawn_daemon.py`. Restart the bot: `./manage.sh restart-bot`. Process match is still `pgrep -f 'core/bot.py'` (or `bot.py`). Log timestamps are LOCAL PDT, not UTC. Run tests: `./manage.sh test` (or `.venv/bin/python -m pytest`).

**Scoring flow:** Ollama (`llama3.2`, local, unified_v1 prompt = `SYSTEM_PROMPT`) scores every article. Bullish + passes signal_checks → **(1) pre-score noise + soft-catalyst filters** (drop analyst/PT, technical, acquirer-M&A, AND index-inclusion/reconstitution headlines PRE-score) → **(2) regime gate** (SPY≥200d SMA AND ≥3d-ago momentum) — long-beta paused in downtrend, EXCEPT high-conviction bypass below → **(3) confirm/veto gate** → legs. Index-inclusion filter added 2026-06-24 (`PRESCORE_SKIP_PATTERNS`): backtested on past rebalance windows via Alpaca historical news (`index_event_backtest.py`) — added stocks show NO directional edge (3-day fwd avg +0.1%, ~50% positive) yet over-score to 0.85 → tripped the regime bypass (GOOG/DJIA −36% live). Russell/S&P/Dow reconstitution is a quarterly burst the bot's own caches miss.

**🔑 CONFIRM/VETO GATE — now `mistral-large` (switched 2026-06-24).** History: Gemini → Groq `llama-4-scout` (06-23) → **mistral-large (06-24)** because Groq is DEPRECATING llama-4-scout. From the n=345 bake-off (`overnight_confirm_veto.py`, real news_call option P&L): scout was #1 (+5.2%, SIG) but is gone; **mistral-large was the validated #2 — edge +4.7%, boot CI [+0.5,+10.7], SIG** — and it's on MISTRAL (not Groq), so immune to Groq model churn. ~2.7s latency (fine at ~10-30 gate calls/day). Reasoning models (gpt-oss-120b, qwen3.6) were NOT viable as the gate (collapsed/parse-failed at scale; the parse part is fixable — see A/B below — but rate-limits remain). **Config (config.py): the `GEMINI_*` names now WIRE TO MISTRAL** (`GEMINI_BASE_URL/KEY`=`MISTRAL_BASE_URL`/`MISTRAL_API_KEY`, `GEMINI_MODEL`=`mistral-large-latest`, `GATE_DAILY_CAP=1000`, `GATE_LATCH_ON_RATELIMIT=0`). `GATE_LABEL="mistral-large"` in logs. Internal scorer TAGS stay `"gemini"`/`"gemini_veto"` for analytics continuity. Override any of these with `CONFIRM_GATE_*` env vars. **Gate had not fired live as of 06-23** (no `gemini_decisions.csv` yet — regime blocked longs until the bypass); watch for first firings.

**🔑 REGIME BYPASS (2026-06-23):** `REGIME_BYPASS_MIN_MAGNITUDE=0.85` — bullish catalysts at/above this magnitude trade EVEN IN A DOWNTREND (then still face the gate). Backtest (`regime_bypass_option_pnl.py`, real option P&L): down-regime ≥0.85 calls made money — recent pullbacks +$274/trade, even the 2022 BEAR stayed positive (+$29/trade, never a net loss). The blunt regime filter was blocking strong idiosyncratic catalysts (e.g. SCWX relisting). Set ≥1.01 to disable.

**🔑 GROQ PRIMARY-SCORER investigation (for a CLOUD move where local Ollama isn't practical → run Groq as primary):**
- **LIVE A/B** (shadow, in bot.py): scores every article with `GROQ_AB_MODELS` + Ollama → `scorer_ab.csv` (logs `groq_model` + tickers, for forward-P&L via `groq_vs_ollama_pnl.py`). ⚠️ `GROQ_AB_MODELS` still = [llama-3.3-70b, **llama-4-scout**] — scout is deprecating, so this should be updated (drop scout / add gpt-oss-120b) on the next bot restart. Old ticker-less log archived `scorer_ab_pre_ticker_20260623.csv` (prior: 79% sentiment agree, Groq 5.6× faster but more conservative; can't be PRIMARY on free tier — 53% rate-limited).
- **BACKTEST (faster, no waiting):** `groq_vs_ollama_backtest.py` scores the full ~1,906-article universe (incl. Ollama-neutral, so "Groq-only" trades are measurable) with each Groq model, vs Ollama AS PRIMARY. `resume_after_reset.py` is a daemon that resumes it after each Groq daily-cap reset (00:10 UTC) until the gated models finish; circuit-breaker stops a model cleanly at its rate wall (no None-flooding). Section A = fixed mag≥0.75 (shows Groq's lower mag scale), Section B = MATCHED selectivity (the fair pick-quality test). `REPORT_ONLY=1` reads partials anytime.
- **FINDINGS so far:** scout (now deprecated) slightly BEAT Ollama (matched +3.0% vs +2.7%) — moot. **llama-3.1-8b is WORSE than Ollama** (+1.9% vs +2.7% — weaker picker, ruled out). 70b inconclusive (can't finish on free tier). **gpt-oss-120b ADDED 2026-06-24** as the remaining candidate.
- **Reasoning-model fix (validated 2026-06-24):** gpt-oss-120b & qwen3.6's earlier "failures" were mostly RATE-LIMITS, not parse. With `reasoning_effort=low`, **gpt-oss-120b = 100% clean parse @ ~565ms** (viable; now in the backtest via the new `extra` param on `confirm_veto_sweep.score_one`). qwen3.6 parses with `max_tokens=4000` but is ~13s/call → too slow, left out. `/no_think` is ignored by qwen.
- **CONSTRAINT:** no paid Groq plan available (couldn't open one), so all backtest scoring is free-tier rate-limited (multi-day grind via the scheduler). scout/8b/gpt-oss daily-capped ~1k/day; only the 8b-class (14.4k/day) finishes in ~1 day but it's TPM=6000-throttled to ~7/min.

**Sizing:** FLAT (`MAGNITUDE_SIZING_ENABLED=False`, `NONMAG_SIZE_FRAC=0.65`×$1000) — magnitude uncalibrated (below), no magnitude-scaling.

**Live legs:** REAL = news_call (mag≥0.75), lotto (mag≥0.70&conf≥0.85), pead (earnings, `PEAD_MAX_POSITIONS=20`). SHADOW = stock (`STOCK_ENABLED=False`), news_call sub-threshold, pead overflow, soft-catalyst-blocked, gate-vetoed. ACTIVE in all regimes = pairs (market-neutral L/S — carried the down days). OFF=qqq_macro. RETIRED=bear_short.

**Dashboard fixes (2026-06-23):** Open Positions has an explicit **SHORT/LONG** column (was a cramped `↓` reading as a dash); short legs now show their strategy + correct short-side P&L (matcher was skipping `stock_short`). Recent Trades now merges **closes** (from `closed_trades.csv`) showing realized P&L + close mechanism (Max hold / Trailing stop / Take-profit cap).

**Guardrails:** daily-loss breaker 2% equity/day (`DAILY_LOSS_LIMIT_PCT`, $2000 floor), resets 00:00 UTC. `MAX_OPEN_POSITIONS=45`. Account CANNOT place exchange-held stops → in-process monitor is the ONLY protection; downed bot = unprotected positions.

**KEY FINDINGS (still current):**
- **Magnitude is UNCALIBRATED** (rank-IC≈0; loser avg mag≈winner). Only the coarse 0.75+ band has weak value. Don't size/rank on it.
- **No model calibrates magnitude better than local llama** (`model_calibration.py`, incl. frontier) — task-hardness limit. BUT a different-family model as a CONFIRM/VETO gate DOES add value (the scout/mistral-large gate finding above). The two are different roles: primary scoring (hard, llama fine) vs catching losers via disagreement (gate works).
- **Local confirm/veto ruled OUT (8GB RAM):** tested qwen2.5:3b/gemma2:2b/phi3:mini — all confirm everything / no separation; the models that work (≥14B) don't fit (4.7-4.9GB models kernel-panicked the box). Gate must stay cloud. Only `llama3.2` (2GB) remains in Ollama.
- **Losers cluster by CATALYST TYPE** (`loser_analysis.py`) → the pre-score soft-catalyst filter (validated −$4,038 removed, CI<0).

**Analysis tools (cached, re-runnable):** `missed_movers.py` (calibration rank-IC — use after ANY prompt change), `loser_analysis.py`, `model_calibration.py`, `prefilter_pnl.py` (filter P&L), `confirm_veto_sweep.py` + `overnight_confirm_veto.py` (gate model bake-off), `regime_bypass_backtest.py` + `regime_bypass_option_pnl.py` (bypass), `groq_vs_ollama_pnl.py` (primary-scorer A/B). Caches: `yahoo_*_cache.json`, `*_cv_cache.json`, `regime_bypass_option_trades.json`.

**WHEN YOU RETURN, check:** (1) `gemini_decisions.csv` (gate decisions, now mistral-large) joined to outcomes → first live confirm/veto firings + bypass (`⚡` log) trades. (2) `groq_vs_ollama_pnl.py` → is paid-Groq-as-primary ≥ Ollama? (3) `loser_analysis.py` → soft-catalyst filter cutting loss buckets. (4) watch the option-loser tails (ACN −91%, ASTS −45% — calls bled to ~0 before the premium trailing stop fired).

Note: `~/.claude/projects/-Users-jeff-Claude-Trader/memory/project_trader_bot.md` is 276KB — a `/consolidate-memory` pass is overdue. Focused memories now exist for the gate, regime bypass, local rule-out, and Groq A/B.

---

## What this project is

A news-driven **options paper-trading bot**. It scores financial headlines with a
**local Ollama LLM** (`llama3.2` at `localhost:11434`) and trades **long call
options** on equities + ETFs via **Alpaca paper trading**. Crypto was removed
(see below) — it is options-only now.

Pipeline: Alpaca/Benzinga news WebSocket + SEC EDGAR 8-K RSS + NewsAPI → Ollama
sentiment/magnitude/confidence score → conviction gate → pick ~37-DTE ~0.40-delta
call → trailing-stop monitor.

---

## Environment & how to run (IMPORTANT)

- **Always use `.venv/bin/python`** — system Python 3.14 is managed, `pip` is
  blocked system-wide (PEP 668). The venv at `.venv/` has all deps
  (numpy ✓, pandas ✓, Flask ✓; scipy ✗ — not needed).
- **Backtests run cache-only (NO Ollama needed):** all 8,633 article scores are
  cached in `backtest_score_cache.json`. Scripts look up by `cache_key(headline,
  body)` and skip misses. Only the **live bot** needs Ollama running.
- **Alpaca is a cloud API** (keys in `.env`) — price/option/news data works from
  anywhere with the keys. The local dependency is Ollama + the running bot
  process; everything else is portable.
- Run the bot:        `.venv/bin/python bot.py >> bot.log 2>&1 &`
- Run the dashboard:  `.venv/bin/python dashboard.py >> dashboard.log 2>&1 &`
  (serves `0.0.0.0:5001`; Mac LAN IP was `192.168.68.100`)
- **Currently running (as of 2026-06-01):** bot.py (PID was 30865), dashboard.py
  (PID was 29479) — live on paper.

---

## File inventory

| File | Role |
|---|---|
| `bot.py` | Live trading bot (options-only). News→Ollama→option trades + in-process trailing-stop monitor. |
| `backtest.py` | 6-month historical replay. Now prices options with Black-Scholes + IV-crush (see below). |
| `config.py` | Single source of truth for all tunable params (imported by bot + backtest). |
| `pricing.py` | Black-Scholes pricer + vol utils: `bs_call_price`, `bs_put_price`, `strike_for_delta`, `put_strike_for_delta`, `implied_vol_call`, `realized_vol`, `iv_crush_path`. Uses stdlib `statistics.NormalDist`. |
| `calibrate_iv.py` | Estimates `NEWS_IV_MULTIPLIER` by inverting BS on real option prints. |
| `benchmark.py` | Beta benchmark: re-prices strategy trades as QQQ/SPY/random to isolate selection skill vs market beta. |
| `qqq_routing.py` | Tests routing macro-bullish (no-ticker) signals into QQQ calls. |
| `bear_backtest.py` | Tests long-put strategy on bearish-flagged names + benchmarks. |
| `dashboard.py` | Local Flask monitoring UI (status, P&L, positions w/ stop column, activity feed, errors, raw log tail). |
| `tune.py` | Parameter sweep (grid search). Pre-caches bars to avoid per-combo HTTP. |
| Data: `backtest_score_cache.json` (Ollama scores), `backtest_trades.csv`, `backtest_signals.csv`, `trades.csv` (live), `bot_state.json` (live stop prices for dashboard) | |

There is also a richer running memory at
`~/.claude/projects/-Users-jeff-Claude-Trader/memory/` (`MEMORY.md` +
`project_trader_bot.md` + `feedback_trading_bot.md`) — read it for full detail if
accessible; this HANDOFF.md is the portable summary.

---

## Current tuned parameters (config.py)

| Param | Value |
|---|---|
| MIN_MAGNITUDE | 0.35 |
| BASE_CONFIDENCE | 0.55 |
| CONFIDENCE_SLOPE | 0.25 |
| TRAILING_STOP_PCT | 0.15 |
| MIN/MAX_DAYS_TO_EXPIRY | 28 / 45 (target ~37) |
| TARGET_DELTA | 0.40 |
| MAX_POSITION_USD | 1,000 (scaled by mag×conf) |
| MAX_HOLD_DAYS | 30 (backtest only; live has NO time cap — a known divergence) |
| Backtest IV model | RISK_FREE_RATE 0.04, REALIZED_VOL_WINDOW 20, **NEWS_IV_MULTIPLIER 1.10** (calibrated), IV_CRUSH_HALFLIFE_DAYS 3, IV_FLOOR 0.15, IV_CAP 2.5 |

---

## What was done this session (2026-06-01)

1. **Validation backtest** (old model): +$62,534, Sharpe 0.99, 50.4% win, 940 trades.
2. **Built `dashboard.py`** — mobile-friendly local web UI: bot status, today's P&L,
   counters (articles/signals/trades from `bot.log`), activity feed, errors panel,
   open positions **with a Stop column** (fed by `bot_state.json`), recent trades,
   and a color-coded raw log tail. Refreshes 15s (log 5s). LAN-only for now;
   cloudflared/ngrok for internet later.
3. **Fixed event-loop hang bug** — `get_stock_price` on a cache-miss ticker called
   `stock_stream.subscribe_trades` from inside the running loop → deadlock that
   froze news AND the stop monitor. Fix: equity trade path runs in
   `run_in_executor` under a 30s `wait_for` timeout (`execute_equity_trade`).
4. **Fixed market-halt close handling + orphan bug** — monitor used to `del` a
   position even when the close FAILED (e.g. during a halt) → unprotected
   position. Fix: closes return bool, only de-register on confirmed success,
   `market_is_open()` guard defers closes during halts, retries next cycle.
   (Note: this account is "not eligible to trade uncovered option contracts" →
   exchange-held stop orders all fail; **stops are enforced ONLY by the
   in-process monitor**. If the bot is down, positions are unprotected.)
5. **Removed crypto entirely** — backtest never simulated it (no historical option
   analog), so it was unproven. Stripped from bot.py + config.py.
6. **Replaced the synthetic option-pricing model with Black-Scholes + IV crush**
   (`pricing.py` + rewritten `simulate_option_pnl`). Old model had no implied vol
   → blind to IV crush (the dominant risk for buying calls on news). New model
   prices off dense stock bars: base IV = trailing realized vol; entry IV =
   base × NEWS_IV_MULTIPLIER; IV decays back to base over the hold; theta from
   shrinking T. **Calibrated NEWS_IV_MULTIPLIER = 1.10** against 109 real option
   prints (median real_iv/base_iv).
7. **Found + fixed a data-quality bug** — the free **IEX feed has data holes**;
   fetching too-wide a window returned a later, discontinuous price segment
   (e.g. post-split regime), producing fake 25-60× option "gains" (AZN). Fixes in
   `backtest.py`: `adjustment=Adjustment.ALL`, fetch only the ~hold window, skip
   trades with >7-day entry gap or >50% single-day jump.

---

## Key findings (the important conclusions)

- **The corrected BS backtest** = +$171,938, Sharpe 2.72, win 32.8%, PF 4.31, but
  **top 20 trades = 82% of P&L** (lottery-ticket profile).
- **The entire 6-month window is a one-directional BULL MELT-UP**: SPY +10.4%
  with zero drawdown, QQQ +18.2%. A long-calls-only strategy in a melt-up always
  prints — so headline P&L is mostly **market beta, not demonstrated alpha**.
- **Beta benchmark (`benchmark.py`):** strategy beats RANDOM name selection
  ~2.5× P&L / 2.4× Sharpe → the LLM/news signal has **real name-selection skill**.
  But it LOSES to just buying QQQ calls ($54K vs $210K on the sample) → beta
  dominates single-name selection in this regime. Strategy has much lower
  drawdown than the index arms.
- **QQQ routing (`qqq_routing.py`):** routing 157 discarded macro-bullish signals
  into QQQ calls adds +$97.6K and +0.18 Sharpe (combined $269,527 / 2.90) — but
  it's more regime-dependent beta and raises max drawdown ($15K→$20K).
- **Bear test (`bear_backtest.py`):** only 90 bearish signals had a tradeable
  ticker (the scorer prompt says "Only flag bullish" → model tags bearish tickers
  in just 0.4% of cases; tickers here came from NEWS METADATA, not the model).
  All put arms lose in the bull regime, but bear LOSES LEAST (−$10.1K vs random
  −$14.1K vs QQQ-puts −$20.6K) with best PF (2.53) → **bearish detection also has
  relative selection skill.**
- **SYNTHESIS:** the model shows **two-sided name-selection skill**, but all P&L
  is **regime-bound to one bull window** and we can't separate durable edge from
  beta on all-bull data. The signal has information; cross-regime profitability is
  unproven.

---

## What was done — session 2026-06-01 (Cowork)

### New files
| File | Role |
|---|---|
| `rescore_all.py` | **Modified** — added `--output-file` arg (parallel runs write to different files); fixed env var names (`ALPACA_API_KEY` / `ALPACA_SECRET_KEY`). |
| `tune_v2.py` | **New** — full replacement for `tune.py`. Reads `dual_score_cache.json` directly (no Ollama needed after rescore). Supports `--mode bull\|bear`, `--end-date`, `--days`, `--cache-file`, `--holdout-pct`. Uses Black-Scholes pricing (`backtest.simulate_option_pnl`) via monkey-patching module globals per combo. 243-combo grid (DTE pairs × delta × mag × conf × trail). 80/20 holdout split; reports top-5 on holdout. |
| `run_overnight.sh` | **New** — orchestrates the full overnight pipeline (see below). |

### Overnight pipeline — READY TO RUN

**Pre-flight status:** ✅ all checks passed. Ollama is running (`llama3.2:latest`, PID 37516 from `ollama serve &`). Smoke test confirmed 10 articles score correctly at ~4-5s each.

**One cleanup step first** — the smoke test left 10 `SCORE_FAILED` entries in `dual_score_cache.json` (Ollama was down for the first run). Delete it so the overnight run starts clean:

```bash
cd ~/Claude/Trader
rm dual_score_cache.json
./run_overnight.sh
```

**What `run_overnight.sh` does:**
1. Smoke-tests `rescore_all.py --limit 10` — aborts if broken
2. Launches `rescore_all.py` (current 180d bull window) → `dual_score_cache.json` in background
3. Launches `rescore_all.py --end-date 2022-06-30 --days 90 --output-file bear_dual_cache.json` → `bear_dual_cache.json` in background
4. Starts a watcher subprocess for each rescore; when it finishes, auto-launches `tune_v2.py` for that mode
5. Prints all PIDs + a `tail -f` command

**Expected timeline:** bull rescore ~9-11h (7,962 articles × ~4-5s), bear rescore ~4-5h (2022-H1, 90d). Tunes run in ~10-20 min each after their respective rescore finishes.

**Monitor:**
```bash
tail -f rescore_all.log rescore_bear.log tune_bull.log tune_bear.log
```

**tune_v2.py grid** (243 combos):
- `DTE_PAIR`: (14,21), (21,30), (28,45)
- `TARGET_DELTA`: 0.30, 0.40, 0.50
- `MIN_MAGNITUDE`: 0.25, 0.35, 0.45
- `BASE_CONFIDENCE`: 0.45, 0.55, 0.65
- `TRAILING_STOP_PCT`: 0.10, 0.15, 0.20

**Outputs after overnight run:**
- `dual_score_cache.json` — ~7,962 dual-scored articles (bull+bear), current window
- `bear_dual_cache.json` — ~4,000 dual-scored articles, 2022-H1 window
- `tune_v2_bull_results.csv` + `tune_v2_bull_report.html` — full grid on training set
- `tune_v2_bull_holdout.csv` — top-5 combos validated on holdout
- `tune_v2_bear_results.csv` + `tune_v2_bear_report.html` + `tune_v2_bear_holdout.csv`

---

## Open decisions / next steps (pick up here)

1. **Run the overnight pipeline** (immediate) — see above. `rm dual_score_cache.json && ./run_overnight.sh`
2. **Review tune results** — open `tune_v2_bull_report.html` and `tune_v2_bear_report.html`; compare top holdout combos between regimes. Update `config.py` with the best cross-regime params.
3. **Multi-regime validation** — the bear window (2022-H1) tests whether selection skill holds when the market isn't melting up. This is the core open question from prior sessions.
4. **Multi-strategy harness** — DECISION ALREADY MADE: tag trades by strategy in ONE paper account (not separate accounts) for apples-to-apples fills.
5. **Wire QQQ-routing into the live bot** — route macro-bullish (no-ticker) signals to QQQ calls (currently discarded).
6. **Data feed decision** — pay for SIP/consolidated data vs accept IEX limitations + lean on forward paper.

---

## Gotchas / non-obvious facts

- `.venv/bin/python` for everything (pip blocked system-wide).
- **`rescore_all.py` env vars**: uses `ALPACA_API_KEY` / `ALPACA_SECRET_KEY` (with `_API_` / `_SECRET_` — matches `.env`). Fixed 2026-06-01.
- **`dual_score_cache.json`** is the new score cache (replaces `backtest_score_cache.json` for the dual-signal workflow). `tune_v2.py` reads from it directly — no Ollama needed during tuning.
- `tune_v2.py` monkey-patches `backtest` module globals per combo — this is intentional and correct. The patched vars are: `MIN_DAYS_TO_EXPIRY`, `MAX_DAYS_TO_EXPIRY`, `DTE_TARGET`, `TARGET_DELTA`, `TRAILING_STOP_PCT`, `MIN_MAGNITUDE`, `BASE_CONFIDENCE`.
- **Do not run two `rescore_all.py` processes writing to the same output file** — use `--output-file` to separate them (already done in `run_overnight.sh`).
- Backtests are **cache-only** for `backtest_score_cache.json` — do NOT let scripts call Ollama on cache misses. `tune_v2.py` reads `dual_score_cache.json` directly (no Ollama path).
- `tune_v2.py` MUST pre-fetch all bars into `_bar_cache` before the grid loop (it does — see `prefetch_bars()`).
- Account can't place exchange-held stops → in-process monitor is the ONLY protection; a downed bot = unprotected positions.
- IEX feed has data holes/discontinuities — keep the data-quality guards in `simulate_option_pnl`; don't widen the bar-fetch window.
- Backtest force-exits at MAX_HOLD_DAYS=30; live bot has no time cap (divergence to reconcile if precision matters).
