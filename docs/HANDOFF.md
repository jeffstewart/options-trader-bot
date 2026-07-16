# Trader Bot — Session Handoff

**Purpose of this file:** a self-contained context dump so another Claude
session (Cowork or otherwise) with file access to this Mac can pick up the work
without the prior conversation. Last updated **2026-07-16** (live-state block at top; sections below are 2026-06-01 history).

Working directory: `/Users/jeff/Claude/Trader`

---

## ⚠️ CURRENT LIVE STATE (2026-07-16) — read this first; everything below is from 2026-06-01 and is STALE

**📉 LIVE P&L REALITY CHECK (trade-level review 2026-07-16, of the unattended 2026-06-09→07-16 run):**
realized **−$16,801 over 165 closed trades, 25% win, negative EVERY week** (paper equity $83.2k).
news_call −$11.4k (17% win) + lotto −$4.1k = 92% of the loss; stock/pairs/pead ≈ flat. **Root cause =
the OPTIONS WRAPPER, not the signal**: same signals as stock ran median −1.2%/trade vs options median
−35.6%. Decomposition (all from live data): (1) ~10.6% median round-trip spread tax (buy ask, exit
bid; entry=peak=ask while the monitor marks mid); (2) overnight IV crush — ~80% of option trades die
<24h ("buy midday on news, stopped next morning"; NKE stock +4.1% yet its call −67%, MSFT −1.9% →
−73%); (3) genuine post-news reversals (ACN −91%). **Exit engineering is a DEAD END on this data**
(contradicts the June TP@+25% backtest at the P&L level): from real position_paths.csv, TP@+25% saves
only ~$350 of a $6.5k hole (24/38 losers never reached +10%) and even PERFECT sell-at-peak ≈ +$5.2k
ceiling. `news_call_options_test.py`: NO delta×DTE geometry flips the news population median-positive
(unlike PEAD, where Δ0.70/45d works — that forward shadow continues). Shadow book agrees:
news_call_shadow's "+$13.4k" is a mirage (top-5 = one Jun-12-15 semi cluster incl. a pre-budget-guard
$14k-notional SNDK artifact = 133% of it; ex-top-5 **−$4.4k/106**), stock_shadow ≈ flat (−$5/trade).
Jeff's decision: market conditions (Iran-war chop: SPY +1.8% but 14/26 sessions down) — do NOT
overhaul the strategy; fix the GATE instead (below) and keep collecting.

**🔑 REGIME GATE (reworked 2026-07-16, deployed + bot restarted):** `market_in_uptrend()` now requires
**200d SMA AND 3d momentum AND a CHOP BRAKE — down-day density <60% over the last 10 SPY sessions**
(`REGIME_DOWNDAY_WINDOW=10`/`REGIME_DOWNDAY_MAX_DENSITY=0.60`, env-overridable, 0=off). Evidence:
`regime_gate_chop_sweep.py` on 3 tapes (bull-meltup / 2022-bear / live-chop from regime_decisions.csv)
— keeps 98% of bull P&L at best-tested Sharpe (2.65→2.91), full bear block, flips live-chop −$10.9k →
+$3.9k (skip 73%). ⚠ dd threshold is KNIFE-EDGED on the one 4-week chop sample (65% never fires, 50%
blocks all) — `regime_block_shadow` tracks blocks; check its forward P&L in a few weeks. ER (Kaufman)
is the graceful-dial fallback if 60% proves overfit. A self-referential "heat" gate FAILED (lags chop;
standalone in bear it's worse than no gate). **REGIME BYPASS DISABLED** (mag≥0.85-in-downtrend,
`REGIME_BYPASS_MIN_MAGNITUDE` 0.85→1.01): live it went 1/13 (8% win, −$2.0k) while regime_block_shadow
showed the blocked trades would ALSO have lost (−$2.2k) — the gate was right, the bypass overrode it.

**🔑 SONNET5 / ANTHROPIC PRIMARY-SCORER (2026-07-13/16, jeff bought API credits) — swap is an open TODO:**
`anthropic_scorer.py` scored the backtest universe with haiku + sonnet5 (~1,683 each; sonnet-4-6/opus
only sampled); `anthropic_backtest.py` = P&L + sweeps + bootstrap/jackknife (reports in data/*.txt).
**sonnet5 is the best primary scorer tested**: matched-selectivity +4.07%/trade vs Ollama +3.09%
(n=176); at bot thresholds (mag≥0.45, conf≥0.60) +4.72%/trade, boot CI [+2.88,+7.30] — but smallest-n
and jackknife shows temporal drift (+6.22% → +3.67%, both halves positive). Cost ≈ **$1.57/month** at
41 articles/day with prompt caching (intro pricing through Aug 2026). **BUT sonnet5 is an INVERTED
confirm/veto gate** (`anthropic_gate_bakeoff.py`: edge −1.1% at the live bar — good picker ≠ good
gate), **and the live mistral gate hurts sonnet5's book** (`sonnet5_mistral_pipeline.py`, full 136-pick
coverage after `score_sonnet5_gap_mistral.py`): mistral confirms 82%, but its vetoes had an 80% WIN
rate → gate costs ~14% of total P&L by cutting winners. If sonnet5 goes primary: run UNGATED (also
removes the ~20s gate-latency drag) or re-tune the gate for sonnet5's candidate mix. Deferred pending
jeff's go-ahead (real per-call cost).

**Scoring flow (live, unchanged engine):** Ollama (`llama3.2`, unified_v1) scores every article →
pre-score noise+soft-catalyst filters → regime gate (3 conditions above; NO bypass) → mistral-large
confirm/veto gate → legs. Gate live since 06-24: 472 decisions in `gemini_decisions.csv` (~84%
confirm). Gate latency stopgap `GATE_WAIT_TIMEOUT_SECS=20` still in place; "move gate to fast Groq
production model" plan is now competing with "drop the gate entirely if sonnet5 goes primary".

**⚠️ KNOWN LIVE ISSUE:** `GROQ_AB_MODELS` still defaults to deprecated `llama-3.3-70b-versatile` —
the A/B shadow is 429-ing on EVERY article (visible in bot.log). Harmless to trading (off-trade-path)
but it's dead quota noise: swap to a live production model (e.g. gpt-oss-20b won't work — no
reasoning_effort param on that path — so likely just empty the list) on the next config touch+restart.

**GROQ PRIMARY-SCORER BACKTEST — DONE (100% coverage, 2026-07-15):** llama-3.3-70b DROPPED (Groq
deprecated it 07-11, quota died incomplete). Final matched-selectivity vs Ollama +3.09%: gpt-oss-20b
+3.47%, gpt-oss-120b +3.60% (n=249-276) — both edge Ollama but BOTH BEATEN by sonnet5 (+4.07%). The
cloud-primary decision is now sonnet5-vs-gpt-oss, not Groq-vs-Ollama. `resume_after_reset.py` got a
cache-read retry (write-race fix); the backtest daemon is done/down — no need to restart it.

**🗂 RESTRUCTURED 2026-06-24 — the project is now a git repo with an organized layout (was flat):**
- `core/` runtime+engine (bot.py, config.py, dashboard.py, backtest.py, pricing.py, yahoo_data.py, router_eval.py, spawn_daemon.py) · `research/` ~112 backtest/analysis scripts · `tests/` pytest suite · `docs/` (this file) · `data/` caches+CSVs+`bot_state.json` (gitignored) · `logs/` (gitignored).
- **Imports** work via `_trader_paths.pth` in the venv site-packages (puts `core/`+`research/` on sys.path) — so flat `import bot`/`import config` still work from anywhere. **Daemons run with CWD=`data/`** (set by `spawn_daemon.py` WORKDIR) so bare data-file refs resolve into `data/`. `config.DATA_DIR`/`LOG_DIR` are the explicit anchors.
- **Manage daemons with `./manage.sh {start|sched|stop|restart-bot|status|test}`** (wraps the now-verbose absolute-path launch commands). pids in `data/*.pid`, logs in `logs/`.
- **git:** `main` branch; code+docs tracked, `.env`/data/logs/caches gitignored. Commit going forward.

**Running:** bot, dashboard, and `resume_after_reset.py` (backtest scheduler) are daemons (ppid 1) via `core/spawn_daemon.py`; the groq backtest daemon is DONE/down (100% coverage) — no need to restart it. Watchdog cron + unattended hardening: `docs/UNATTENDED.md`. Restart the bot: `./manage.sh restart-bot`. Process match is still `pgrep -f 'core/bot.py'` (or `bot.py`). Log timestamps are LOCAL PDT, not UTC. Run tests: `./manage.sh test` (or `.venv/bin/python -m pytest`).

**Pre-score filters (detail):** index-inclusion filter added 2026-06-24 (`PRESCORE_SKIP_PATTERNS`): backtested on past rebalance windows via Alpaca historical news (`index_event_backtest.py`) — added stocks show NO directional edge (3-day fwd avg +0.1%, ~50% positive) yet over-score to 0.85 → tripped the (now-removed) regime bypass (GOOG/DJIA −36% live). Russell/S&P/Dow reconstitution is a quarterly burst the bot's own caches miss. Also drops analyst/PT, technical, acquirer-M&A soft catalysts and Benzinga returns-calculator recaps pre-score.

**🔑 CONFIRM/VETO GATE — now `mistral-large` (switched 2026-06-24).** History: Gemini → Groq `llama-4-scout` (06-23) → **mistral-large (06-24)** because Groq is DEPRECATING llama-4-scout. From the n=345 bake-off (`overnight_confirm_veto.py`, real news_call option P&L): scout was #1 (+5.2%, SIG) but is gone; **mistral-large was the validated #2 — edge +4.7%, boot CI [+0.5,+10.7], SIG** — and it's on MISTRAL (not Groq), so immune to Groq model churn. ~2.7s latency (fine at ~10-30 gate calls/day). Reasoning models (gpt-oss-120b, qwen3.6) were NOT viable as the gate (collapsed/parse-failed at scale; the parse part is fixable — see A/B below — but rate-limits remain). **Config (config.py): the `GEMINI_*` names now WIRE TO MISTRAL** (`GEMINI_BASE_URL/KEY`=`MISTRAL_BASE_URL`/`MISTRAL_API_KEY`, `GEMINI_MODEL`=`mistral-large-latest`, `GATE_DAILY_CAP=1000`, `GATE_LATCH_ON_RATELIMIT=0`). `GATE_LABEL="mistral-large"` in logs. Internal scorer TAGS stay `"gemini"`/`"gemini_veto"` for analytics continuity. Override any of these with `CONFIRM_GATE_*` env vars. Gate has been live since 06-24 — 472 decisions in `gemini_decisions.csv` (~84% confirm rate). ⚠️ For sonnet5-as-primary the gate is a NET NEGATIVE at current tuning (see the sonnet5 block at top).

**🔑 REGIME BYPASS — DISABLED 2026-07-16** (see the regime-gate block at top for the live evidence that killed it). Historical context: deployed 2026-06-23 on `regime_bypass_option_pnl.py` backtest evidence (down-regime mag≥0.85 calls +$274/trade in pullbacks, positive even in the 2022 bear). The live forward test falsified it — a reminder that magnitude is uncalibrated (loser mag ≈ winner mag), so a magnitude bar can't identify "market-independent" alpha.

**🔑 GROQ PRIMARY-SCORER investigation — CONCLUDED (see DONE block at top for final numbers). Kept for context:**
- **LIVE A/B** (shadow, in bot.py): scores every article with `GROQ_AB_MODELS` + Ollama → `scorer_ab.csv` (logs `groq_model` + tickers, for forward-P&L via `groq_vs_ollama_pnl.py`). ⚠️ now 429-ing on every article — see KNOWN LIVE ISSUE at top. Old ticker-less log archived `scorer_ab_pre_ticker_20260623.csv` (prior: 79% sentiment agree, Groq 5.6× faster but more conservative; can't be PRIMARY on free tier — 53% rate-limited).
- **BACKTEST:** `groq_vs_ollama_backtest.py` scored the full universe (incl. Ollama-neutral, so "Groq-only" trades are measurable) with each Groq model, vs Ollama AS PRIMARY. Section A = fixed mag≥0.75 (shows Groq's ~3-5× lower mag scale), Section B = MATCHED selectivity (the fair pick-quality test). `REPORT_ONLY=1` re-reads anytime. Model churn absorbed along the way: llama-4-scout (preview, deprecated 06-24), llama-3.1-8b (→gpt-oss-20b, 06-25), llama-3.3-70b (07-11). **Rule: pin live+backtest scoring to Groq PRODUCTION-tier models only.**
- **Reasoning-model fix (validated 2026-06-24):** with `reasoning_effort=low`, gpt-oss-120b = 100% clean parse @ ~565ms (via the `extra` param on `confirm_veto_sweep.score_one`). qwen3.6 parses with `max_tokens=4000` but ~13s/call → too slow.

**Sizing:** FLAT (`MAGNITUDE_SIZING_ENABLED=False`, `NONMAG_SIZE_FRAC=0.65`×$1000) — magnitude uncalibrated (below), no magnitude-scaling.

**Live legs:** REAL = news_call (mag≥0.75, Δ0.40/DTE7-13, 3d cap, 40/25/15 ratchet trail), lotto (mag≥0.70&conf≥0.85), pead (earnings, stock, `PEAD_MAX_POSITIONS=20`). SHADOW = stock (`STOCK_ENABLED=False`), news_call sub-threshold, pead overflow, **pead_option (Δ0.70/~45d ITM — the geometry that backtests broad-positive for PEAD)**, soft-catalyst-blocked, gate-vetoed, regime-blocked. ACTIVE in all regimes = pairs (market-neutral L/S — carried the down days; 18 of 25 currently-open positions). OFF=qqq_macro. RETIRED=bear_short.

**Dashboard fixes (2026-06-23):** Open Positions has an explicit **SHORT/LONG** column (was a cramped `↓` reading as a dash); short legs now show their strategy + correct short-side P&L (matcher was skipping `stock_short`). Recent Trades now merges **closes** (from `closed_trades.csv`) showing realized P&L + close mechanism (Max hold / Trailing stop / Take-profit cap).

**Guardrails:** daily-loss breaker 2% equity/day (`DAILY_LOSS_LIMIT_PCT`, $2000 floor), resets 00:00 UTC. `MAX_OPEN_POSITIONS=45`. Account CANNOT place exchange-held stops → in-process monitor is the ONLY protection; downed bot = unprotected positions.

**KEY FINDINGS (still current):**
- **Magnitude is UNCALIBRATED** (rank-IC≈0; loser avg mag≈winner). Only the coarse 0.75+ band has weak value. Don't size/rank on it.
- **No model calibrates magnitude better than local llama** (`model_calibration.py`, incl. frontier) — task-hardness limit. BUT a different-family model as a CONFIRM/VETO gate DOES add value (the scout/mistral-large gate finding above). The two are different roles: primary scoring (hard, llama fine) vs catching losers via disagreement (gate works).
- **Local confirm/veto ruled OUT (8GB RAM):** tested qwen2.5:3b/gemma2:2b/phi3:mini — all confirm everything / no separation; the models that work (≥14B) don't fit (4.7-4.9GB models kernel-panicked the box). Gate must stay cloud. Only `llama3.2` (2GB) remains in Ollama.
- **Losers cluster by CATALYST TYPE** (`loser_analysis.py`) → the pre-score soft-catalyst filter (validated −$4,038 removed, CI<0).

**Analysis tools (cached, re-runnable):** `missed_movers.py` (calibration rank-IC — use after ANY prompt change), `loser_analysis.py`, `model_calibration.py`, `prefilter_pnl.py` (filter P&L), `confirm_veto_sweep.py` + `overnight_confirm_veto.py` (gate model bake-off), `groq_vs_ollama_pnl.py` (primary-scorer A/B). NEW 2026-07: `anthropic_scorer.py` + `anthropic_backtest.py` (Anthropic-as-primary), `anthropic_gate_bakeoff.py` + `sonnet5_mistral_pipeline.py` (sonnet5×gate), `news_call_options_test.py` (expression geometry), `regime_gate_chop_sweep.py` (gate variants on 3 tapes incl. the live chop window). Caches: `yahoo_*_cache.json`, `*_cv_cache.json`, `anthropic_backtest_cache.json`, `overnight_cv_cache.json` (now also holds mistral scores of sonnet5's picks).

**WHEN YOU RETURN, check:** (1) **chop brake forward evidence** — `regime_block_shadow` closes in `shadow_trades.csv` since 07-16: is the dd10<60% condition blocking losers (validating) or winners (the knife-edge caveat biting)? (2) **sonnet5 swap decision** — the open TODO; if pursued, run ungated or re-tune the gate for sonnet5's candidate mix. (3) **pead_option shadow** (Δ0.70/45d) accumulating real-quote closes — the confirmation gate for flipping PEAD to options. (4) **fix `GROQ_AB_MODELS`** (deprecated 70b, 429-ing constantly — see KNOWN LIVE ISSUE). (5) weekly P&L trend post-gate-rework — did the chop brake stop the bleed?

Note: `~/.claude/projects/-Users-jeff-Claude-Trader/memory/project_trader_bot.md` is 276KB — a `/consolidate-memory` pass is overdue. Focused memories exist for the gate, regime bypass, local rule-out, Groq A/B, **sonnet5 primary scorer (`project_sonnet5_primary_scorer.md`), and the live loss review (`project_live_loss_review_202607.md`)**.

### 🚀 GO-LIVE / DEPLOYMENT PLAN (decided 2026-06-25 — for when this moves to real money + cloud)

**Move stops BROKER-SIDE on real money — this is the key resilience change.** Alpaca supports
exchange-held stop / trailing-stop orders on REAL accounts (not paper — which is why the live bot
currently relies on the in-process trailing-stop monitor, the *sole* protection today → a downed bot
= unprotected positions). On the real account, place the stop at the broker when opening a position
so the exchange enforces it even if the bot is down.
- **VERIFY FIRST (the open question):** does this apply to the OPTION legs (long calls)? The original
  blocker was *options* eligibility ("not eligible to trade uncovered option contracts"), a separate
  axis from paper-vs-real. Confirm `stop`/`trailing_stop` order support on options at the account's
  options approval tier. The EQUITY/pairs legs almost certainly support broker stops regardless.
- **Why it matters:** broker-side stops convert uptime from **safety-critical → opportunity-only**.
  Downed bot then = *missed new trades* (opportunity cost), NOT unprotected positions. Broker stops
  become the resilience layer.

**Deployment architecture (given the above):** keep it SIMPLE — the bot is a stateful **singleton**
(two instances would double-trade the same account), so:
- **NO Kubernetes/minikube** (orchestration/replicas buy nothing for a singleton) and **NO serverless**
  (it's a long-running process holding WebSocket streams + a position monitor).
- Target a **single always-on container on a managed platform (Fly.io is the standout fit — singleton
  machines + volumes + auto-restart + health) OR a small VM + `systemd Restart=always`.** Container
  value here = reproducibility + managed auto-restart/health, not scaling.
- **State is light:** positions reconstruct from Alpaca on startup; only the trailing-stop high-water
  marks (`bot_state.json`) must persist → a small volume or a few Redis keys. (Config is already
  container-ready: `TRADER_DATA_DIR`/`TRADER_LOG_DIR` env overrides, pinned `requirements.txt`,
  no host paths in `core/`. Add a SIGTERM handler for clean restarts when you write the Dockerfile.)
- **Instance sizing: the cloud-primary question is now answered in principle** — sonnet5 (Anthropic
  API, ≈$1.57/mo at current volume) beat both Ollama and the Groq gpt-oss models on pick quality, so
  the cloud box can be TINY (no local model). The remaining open call is jeff's go-ahead on paid
  per-call scoring + the gate decision that comes with it (run sonnet5 ungated vs re-tune the gate).
  NOTE: the production deploy won't carry shadow-trade infrastructure (jeff's call, 2026-07-16).

**Caveats when flipping stops broker-side:** (1) broker stops don't cover the bot's *non-stop* exits
(max-hold time exit, lotto take-profit cap) — those still need the bot, but they're optimization, not
protection. (2) Alpaca's native `trailing_stop` is a FIXED trail %/$ — dumber than the in-process
tiered/profit-scaled logic; **backtest "fixed broker trail vs. current tiered" before flipping** so
you know the behavior you're trading away for resilience.

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
