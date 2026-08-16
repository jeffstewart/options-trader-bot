# Trader Bot — Session Handoff

**Purpose of this file:** a self-contained context dump so another Claude session (Cowork or
otherwise) with file access to this Mac can pick up the work without the prior conversation.
Fully rewritten **2026-08-15** to reflect current live state — no stale sections carried forward
from before. For narrower/detailed history, see the memory files under
`~/.claude/projects/-Users-jeff-Claude-Trader/memory/` (index: `MEMORY.md`), which this doc
periodically summarizes but does not replace.

Working directory: `/Users/jeff/Claude/Trader`

---

## What this project is

Two independent, event-driven options **paper-trading** bots, both running against Alpaca:

- **v1 (`core/`)** — the original bot. Parses every article that hits the news feed, scores it with
  a local Ollama LLM, gates the strongest signals through a cloud confirm/veto model, and trades
  `news_call` / `lotto` / `pead` / `pairs`. Runs natively on the host (not containerized).
- **v2 (`v2/`)** — a lean rewrite built for an eventual small real-money account. Runs on its own
  separate Alpaca paper account, independent of v1. No confirm/veto gate, no pairs, equity-relative
  sizing, currently **lotto-only**. Runs in a Docker container (via Colima) for the bot, native
  dashboard.

Both bots are started/stopped together with `./everything.sh {start|stop|restart|status}` from the
repo root. See `README.md` and `v2/README.md` for the full architecture and setup instructions —
this file is the state/history/decisions log, not the reference doc.

---

## Current live configuration (as of 2026-08-15)

### v1 (`core/config.py`)

| Param | Value |
|---|---|
| Primary scorer | `llama3.2`, local Ollama |
| Confirm/veto gate | `mistral-large` (env `GATE_LABEL="mistral-large"`), `GEMINI_CONFIRM_MIN_MAGNITUDE=0.50`, `GATE_WAIT_TIMEOUT_SECS=20` |
| MIN_MAGNITUDE / BASE_CONFIDENCE / CONFIDENCE_SLOPE | 0.35 / 0.70 / 0.25 |
| TARGET_DELTA | 0.50 |
| MIN/MAX_DAYS_TO_EXPIRY | 14 / 21 |
| TRAILING_STOP_PCT | 0.10 |
| MAX_POSITION_USD | $1,000 (flat — `MAGNITUDE_SIZING_ENABLED=False`, `NONMAG_SIZE_FRAC=0.65`) |
| MAX_HOLD_DAYS | 30 (live has no hard time cap outside this; historical divergence note from backtest-only origins may no longer apply — not reverified this pass) |
| Regime gate | 200d SMA + 3d momentum + chop brake (`REGIME_DOWNDAY_WINDOW=10`, `REGIME_DOWNDAY_MAX_DENSITY=0.60`) |
| Regime bypass | **DISABLED** (`REGIME_BYPASS_MIN_MAGNITUDE=1.01`, effectively unreachable) |
| DAILY_LOSS_LIMIT | $2,000 floor (~2%/day), resets 00:00 UTC |
| MAX_OPEN_POSITIONS | 45 |
| Strategies live | `news_call`, `lotto` (mag≥0.70, conf≥0.85, late-day entry cutoff), `pead` (`PEAD_MAX_POSITIONS=20`), `pairs` (all regimes) |
| Strategies off/shadow | `stock` (`STOCK_ENABLED=False`), `bear_short`, `qqq_macro` |
| Grid collector | ON — `GRID_COLLECTOR_ENABLED=True`, threshold `mag×conf≥0.13`, see below |

### v2 (`v2/config.py`)

| Param | Value |
|---|---|
| Primary scorer | `kimi-k2.6` via Moonshot API (`SCORER_MODEL`, `SCORER_EFFORT=low`) |
| Ticker corrector | local Ollama `llama3.2`, `TICKER_CORRECTOR_ENABLED=True` |
| Confirm/veto gate | none (removed by design) |
| MIN/MAX_DAYS_TO_EXPIRY | 14 / 21 |
| TARGET_DELTA | 0.50 (news_call geometry — currently unused, news_call is off) |
| Lotto geometry | `LOTTO_TARGET_DELTA=0.25`, DTE 8-21, strike hi mult 1.25 |
| Lotto sizing | `LOTTO_POSITION_FRAC_OF_EQUITY=0.10` (equity-relative, not a fixed dollar amount) |
| Lotto exits | `LOTTO_STOP_LOSS_PCT=0.20`, `LOTTO_TRAIL_PCT=0.15`, `LOTTO_SAME_DAY_EXIT=True`, `LOTTO_MAX_HOLD_DAYS=3` (fallback net) |
| Lotto entry cutoff | 30 min before close (`LOTTO_ENTRY_CUTOFF_MIN_BEFORE_CLOSE`), checked at scoring AND at order placement |
| Contract-mispricing switch | **LIVE** (`MISPRICING_SWITCH_ENABLED=True`, `MISPRICING_MIN_RESIDUAL_GAP=0.02`) — redirects to a cheaper same-expiry neighbor when the IV-smile fit says the target contract is overpriced |
| Regime gate | same 200d SMA + 3d momentum + chop brake as v1, runs BEFORE scoring (nothing downstream needs it during a downtrend) |
| MAX_OPEN_POSITIONS | 3 (working value, not final-tuned) |
| Strategies live | `lotto` only |
| Strategies off | `news_call` (`NEWS_CALL_ENABLED=False`), `pead` (`PEAD_ENABLED=False`) — no positive-Sharpe cell found for either in the small-account sweep |
| Grid collector | not present in v2 (research collector lives only in v1 — see below) |
| Deployment | bot.py in Docker (Colima backend), dashboard.py native on host, both reading the same bind-mounted `v2/data/` |

---

## Contract-grid research collector (v1 only, added 2026-08)

**What it is:** a fire-and-forget, pure-market-data background task in `core/grid_collector.py`,
hooked into `core/bot.py`'s `process_signal()` right after sentiment filtering and BEFORE
`signal_checks()` — so it sees ~88% of historically-bullish signals (Ollama's own confidence
distribution is clustered, not smooth; 0.13 is the closest achievable approximation of a 90th
percentile) rather than only what clears today's trade threshold. For every qualifying signal it
fetches a wide DTE(3-30)×strike(0.85x-1.25x) grid of real contracts, snapshots bid/ask/greeks/OI,
and re-polls that same grid on a decaying schedule (30s → 3min → 15min → 30min, market-hours-gated
only) through each contract's own expiry (capped at `GRID_MAX_TRACK_DAYS=32`). Output:
`data/contract_grid_snapshots.csv` (36 columns).

**Why:** to answer "which contract would have performed best, and how far past today's exit point
did contracts keep moving" independent of what the live contract-picker actually chose — the
picker's own proximity-first sort + 5-candidate shortlist means a lot of real information about
untested candidates is currently invisible to backtesting. Jeff's plan is a later 80/20 train/test
split over this dataset to fit a contract-selection model against every factor available at
signal time, so the schema was reviewed (2026-08-15) against that goal and widened beyond just
market-data:

- **Signal-level** (repeated on every row for a signal_id): `signal_id`, `signal_ts`, `ticker`,
  `headline`, `source` (Benzinga/SEC EDGAR/NewsAPI), `scorer_model` (so results can be
  recalibrated across future model swaps), `magnitude`, `confidence`, `catalyst` (the LLM's third
  score), `stock_price_at_signal`, `passes_news_call_gate` / `passes_lotto_gate` (pure-logic
  threshold-eligibility flags, duplicated from bot.py's formulas rather than replaying the real
  gate — see below).
- **Market state at signal time**: `spy_price`, `spy_sma`, `regime_uptrend`, `regime_mom_ok`,
  `regime_chop_ok`, `regime_dd_density` (same 200d-SMA/3d-momentum/chop-brake regime the live gate
  uses, cached once/day) and a VIX proxy — see below.
- **Per-snapshot**: `snapshot_ts`, `minutes_since_signal`, `stock_price` (re-fetched every poll,
  not just at signal time — tracks moneyness drift over the life of the hold), `symbol`, `strike`,
  `expiry`, `dte_at_signal`, `oi_at_signal`, `bid`, `ask`, `mid`, `spread_pct`, `iv`, `delta`,
  `gamma`, `theta`, `vega`, `rho`.

Day-of-week / time-of-day were considered but not added as separate columns — both timestamp
columns are full datetimes, so they're trivially derivable during analysis.

**VIX proxy (`vol_proxy_*` columns):** true CBOE VIX is not reachable on this Alpaca plan —
confirmed live 2026-08-15: `/v1beta1/indices/*` 404s, and a bare `"VIX"` stock-symbol lookup
returns no trade data. `VIXY` (ProShares VIX Short-Term Futures ETF, a normal tradable equity) is
used as the closest available proxy. Logged as `vol_proxy_pctile` — the latest close's
**percentile rank within its own trailing `GRID_VOL_PROXY_LOOKBACK_DAYS` (20) window** — rather
than a raw price or an SMA comparison, because VIXY bleeds value over time from contango roll cost
and gets reverse-split periodically, so its own long-run price trend is structural decay, not a
volatility signal; a long-window SMA would misread that decay as "always calm." The percentile
cancels the drift out and answers the question that matters: is volatility elevated relative to
its own recent past, right now. Config: `GRID_VOL_PROXY_SYMBOL`, `GRID_VOL_PROXY_LOOKBACK_DAYS`.

**Design decisions worth remembering:**
- Deliberately lives in **v1, not v2** — v1 parses every article regardless of trade eligibility,
  v2 is selective about calling its paid scorer, so v1 is the cheaper place to piggyback broad
  collection.
- Long-end poll interval is 30 min, not the original 2-hour proposal — jeff's call, more
  granularity is worth the extra API calls.
- Skips entirely while the market is closed rather than trying to track elapsed "market time"
  separately from wall-clock time — the schedule just doesn't advance and resumes on reopen.
- No new paper account / no capital needed — this replaced an earlier proposal to actually trade a
  wide grid of contracts in a funded account; the batched Alpaca snapshot endpoint (`get_option_snapshot`)
  does the same job for free.
- `passes_news_call_gate`/`passes_lotto_gate` are a coarse eligibility label only — they do NOT
  account for the confirm/veto gate, cooldown, liquidity, or position caps. Replaying the real gate
  for every logged signal would cost a real paid API call each time, defeating the point of a
  free/broad collector. The confirm/veto score itself is reconstructable later by rescoring the
  article, so this was judged an acceptable gap on its own — but a signal can also fail to trade
  purely on live bot state (cooldown, `MAX_OPEN_POSITIONS`) that rescoring can never recover, which
  is why the join key below was added rather than relying on rescoring alone.
- **`grid_signal_id` join key (added 2026-08-15):** `grid_collector.make_signal_id(ticker,
  signal_ts)` is a pure function; `bot.py`'s `process_signal()` mints the id itself at the moment
  it decides to collect (not inside `grid_collector.start_collection`, which would mint one a few
  hundred ms later) and stashes it on `signal["_grid_signal_id"]`. `place_option_trade` /
  `place_stock_trade` / `place_short_trade` carry it into `_monitored_positions`, and
  `log_trade`/`log_trade_stock`/`log_closed_trade` write it as a trailing column in
  `trades.csv`/`closed_trades.csv`. This is what lets analysis compare "what the grid says was the
  best contract" against what the bot actually bought, its real fill, and its real exit — none of
  which rescoring the article can reconstruct after the fact. The option-position restart-reload
  path recovers it from `trades.csv` (same `entry_log` pattern already used for `strategy`), so a
  position surviving a bot restart still closes with its join key intact. `data/trades.csv` and
  `data/closed_trades.csv` (gitignored, pre-existing) had their header LINE patched in place to
  add the column; old rows are simply shorter than the header and read back as `grid_signal_id=""`
  via `DictReader`, no data was rewritten.
- **Known Alpaca limitation this collector cannot work around**: there is no historical *bid/ask
  quote* endpoint, only a live one — so this collector is only useful going forward from
  2026-08-XX; it cannot backfill true historical spreads for past signals. Same plan gap rules out
  historical VIX, incidentally — the indices endpoint 404s regardless of date.

Still accumulating data as of this writing — not yet used to inform a real contract-picker or
exit-logic change. Revisit once there's enough signal-days to look at.

---

## Contract-picker investigation (v1, 2026-08)

A deep-dive into why signals sometimes die with "no liquid contracts found" despite clearing every
upstream gate. Findings, from mining `logs/bot.log`'s full candidate-search sequence
(`research/spread_cap_log_mining.py`):

- Of signals that exhausted the full 5-candidate shortlist and still failed, rejections split
  roughly evenly between **spread-too-wide (46.6%)** and **open-interest-below-minimum (52.5%)** —
  not primarily spread, as first assumed. Root-symbol mismatches and bid/ask-sanity failures are
  rare (<1% combined).
- A **liquidity-first candidate sort** (OI-filtered-then-proximity, instead of pure
  proximity-first) was prototyped and live-dry-run tested (`research/liquidity_first_pick_dryrun.py`)
  against 21 real tickers: rescued 1 of 21 that the current sort missed entirely (DIS, 4th
  candidate). Modest, not dramatic — not yet deployed live.
- A cap-sweep on realized historical trades found a **12% spread cap would have outperformed the
  current 10%** for lotto — tightening from 12%→10% specifically excludes a real winner (RIVN,
  +$138) for only marginal loss reduction. Not naively monotonic; not yet changed live pending a
  real backtest (blocked — see below).
- **A real backtest of a spread-cap change is not currently feasible**: Alpaca has no historical
  options bid/ask endpoint (confirmed via live API test), the existing synthetic spread model in
  `core/backtest.py` is demonstrably wrong on at least one real counter-example (BA), and the
  contract-picker's own fallback mechanism means a rejected candidate might just redirect to an
  untested one rather than disappear — none of which the matched-trade-count method can account
  for. The grid collector above is the intended eventual fix for this data gap, going forward.
- Secondary bug surfaced but not yet fixed: a contract with `oi=None` (missing data) currently
  passes the OI guardrail silently (`if oi is not None and int(oi) < MIN_OPEN_INTEREST` never
  triggers on `None`) — seen live on a GE pick.

---

## P&L reality check (v1, trade-level review, most recent restriction 2026-08 for trades opened
since 2026-07-16)

Full-history review found **−$16,801 over 165 closed trades (25% win rate)** in the
2026-06-09→07-16 unattended run, with the loss concentrated in `news_call` (−$11.4k, 17% win) and
`lotto` (−$4.1k); root cause diagnosed as the **options wrapper itself** (spread tax + overnight IV
crush), not the underlying signal — same signals traded as stock ran roughly flat. Perfect-exit
analysis put the realistic ceiling at only ≈+$5k even with perfect sell-at-peak timing; a
backtest-favored TP@+25% rule was **contradicted by real intraday option-price paths** (most losers
never even touched +10%).

Restricting the same review to trades opened **after 2026-07-16** (post chop-brake + gate rework)
found `news_call`'s win rate improved 26.1%→35.1%, and the spread-width finding replicated cleanly;
a previously-noted day-of-week pattern did NOT replicate in the narrower window; zero `stock`
strategy trades exist in this window (flagged, not yet investigated).

Jeff's decision at the time: don't overhaul the strategy on this data alone — the regime (chop) was
a real confound — fix the regime gate first (done, see chop brake above) and keep collecting. This
review is what motivated the current contract-picker investigation, on the theory that if the
signal is directionally OK, the remaining edge is in the wrapper (contract choice, sizing, exits),
which the grid collector is now built to test directly.

---

## Scorer history (both bots)

- **v1**: stayed on local Ollama (`llama3.2`) as primary throughout. Confirm/veto gate went
  Gemini → Groq `llama-4-scout` → **mistral-large** (current, since 2026-06-24) as each prior
  option hit a deprecation or reliability wall. A parallel investigation found **sonnet5 beats
  Ollama as v1's primary scorer** (+4.07%/trade matched-selectivity, plus far better ticker
  attribution) but is an *inverted* confirm/veto gate — good picker ≠ good gate — and the live
  Mistral gate actively hurts sonnet5's book (~14% of P&L, vetoing high-win-rate trades). This
  swap was never made live in v1; treat it as an open, deferred option, not current state.
- **v2**: launched on local Ollama, then briefly ran **sonnet5** (Anthropic API) as primary once
  jeff bought credits, then moved to **kimi-k2.6** (Moonshot API) when those credits ran out.
  Matched-trade-count testing across 5 candidate models found no P&L separation at matched n
  between sonnet5 and kimi-k2.6 on the same prompt (`[[project_kimi_scorer]]` memory,
  closed 2026-08-01) — so the swap needed no threshold retuning, and swapping back is a one-line
  `SCORER_MODEL` change in `v2/config.py` if credits allow. Ollama is still used in v2, but only
  for the cheap ticker-selection corrector, not primary scoring.

---

## Infrastructure

- **Restructured 2026-06-24** into `core/` (v1) + `v2/` + `research/` + `tests/` + `docs/` + `data/`
  (gitignored) + `logs/` (gitignored), tracked as a git repo (`main` branch).
- **v1** runs natively via `core/spawn_daemon.py`-launched daemons (ppid 1, CWD=`data/`), managed
  by `./manage.sh {start|sched|stop|restart-bot|restart-dash|watchdog|backup|report|status|test}`.
  `manage.sh start` is idempotent (guards against double-spawning a second untracked bot process —
  caught via a real incident, two live `bot.py` processes 4 minutes apart).
- **v2** runs its bot in a Docker container (Colima backend, not Docker Desktop — switched
  2026-08-03 to save ~1-2GB idle RAM) via `docker compose`, with `dashboard.py` running natively on
  the host reading the same bind-mounted `v2/data/`/`v2/logs/`.
- **`./everything.sh {start|stop|restart|status}`** (added 2026-08-03, after both bots + Ollama got
  shut off over a weekend and had to be restarted piecemeal) is the single entry point for the
  whole stack: starts/stops Colima, Ollama, v1 (via `manage.sh`), the v2 Docker container, and the
  v2 dashboard, in a dependency-safe order (trading engines stop first, Colima stops last).
- **Container DNS**: the v2 container sees transient DNS-resolution bursts against all external
  hosts (SEC, Alpaca, Anthropic) — mitigated with retries + a startup preflight check, not fully
  eliminated. See `[[project_container_dns]]` memory.
- **Backup**: `./manage.sh backup` mirrors code (git bundle) + `data/`/`eod_reports/` to iCloud
  Drive, ~67MB total, no remote git host or Time Machine dependency.

---

## Key structural findings (still current, validated repeatedly across sessions)

- **Magnitude is uncalibrated** — rank-IC≈0, loser avg magnitude ≈ winner avg magnitude, across
  every model tested (incl. frontier). Don't size or rank on it directly; only a coarse high-band
  cutoff has weak value. This is why sizing is flat/equity-relative rather than magnitude-scaled in
  both bots.
- **No model calibrates magnitude better than local llama** at the *primary-scoring* task — but a
  **different-family model used as a confirm/veto disagreement gate** does add real value in v1.
  These are different roles (primary picking vs. catching over-hyped losers via disagreement), and
  results from one don't transfer to the other — this is the reason v1 keeps a gate (mistral-large)
  while v2 deliberately dropped its gate (it tested net-negative for v2's candidate mix).
- **Local confirm/veto is ruled out on this 8GB Mac** — small models (qwen2.5:3b, gemma2:2b,
  phi3:mini) show no separation; models large enough to plausibly work (≥14B) don't fit in RAM
  without risking a kernel panic. Any gate role stays cloud-hosted by necessity.
- **The pre-score filters materially matter** — headline-pattern filters for soft catalysts
  (analyst/PT notes, technical/chart commentary, acquirer-side M&A talk, index-inclusion
  reconstitution, intraday listicles, reactive "why is X surging" recaps, pre-event speculation,
  pundit segments) drop a meaningful fraction of signals before they ever reach an LLM call, and
  were validated to remove real negative-EV trades. The LLM's own `catalyst` score cannot do this
  job — winners and losers score it identically.
- **Overnight 8-K backtest (2026-08, `research/overnight_8k_backtest.py`, n≈11k, 6 months)**: no
  drift edge found in either the overnight or intraday 8-K lane; an earlier-looking intraday t=2.2
  signal from July collapsed at larger n — a mirage. Also confirmed llama3.2 struggles to read raw
  8-K filing bodies. Not pursued further; see `[[project_overnight_8k_backtest]]` memory.
- **SEC EDGAR 8-K feed is non-functional as originally wired** — ~99.7% of items arrive with no
  resolvable ticker. Fixed going forward (see `research/sec_prompt_compare.py` /
  `sec_backtest.py` / `sec_backtest_sameday.py` for the follow-up investigation), but historical
  trade-level comparison against it was blocked by log rotation before the fix landed.

---

## Open items / where to look next

1. **Grid collector data maturity** — check `data/contract_grid_snapshots.csv` row/signal-day
   count; once there's enough coverage, revisit the liquidity-first contract sort and the OI=None
   guardrail bug (both above) with real backtest-quality data instead of a single dry run.
2. **v1 `stock` strategy has zero trades since 2026-07-16** — flagged in the post-cutoff trade
   review, not yet investigated. Check whether it's actually still wired to fire.
3. **v1 `pairs` closed 0-for-10 in the post-07/16 window** (per the trade review) despite running
   in all regimes — a disable candidate that hasn't been acted on.
4. **sonnet5-as-v1-primary remains an open, deferred TODO** — real edge shown in backtests, but
   needs either an ungated run or a re-tuned gate before it's worth making live; blocked on jeff's
   call re: per-call API spend.
5. **v2 sizing/position-cap numbers are not final-tuned** (`MAX_OPEN_POSITIONS=3`,
   `LOTTO_POSITION_FRAC_OF_EQUITY=0.10`) — flagged in v2/README.md as needing a dedicated
   small-account backtest pass before any real-money move.
6. **Go-live / real-money deployment plan** (from the original 2026-06-25 decision, still valid in
   outline): move stops broker-side on a real account (Alpaca supports exchange-held
   stop/trailing-stop orders on real, non-paper accounts — verify this actually extends to long
   option legs, not just equities, before relying on it), keep the deploy architecture a **single
   always-on singleton** (no Kubernetes, no serverless — this is a stateful process holding
   WebSocket streams + a position monitor), target a small always-on host (Fly.io or a VM with
   `systemd Restart=always`). v2's Docker container is a step toward this, not the finished
   version — the current container is for local crash/reboot survival, not a cloud deploy.

---

## Gotchas / non-obvious facts

- `.venv/bin/python` for v1 + `research/` (system pip is blocked, PEP 668); v2 has its own
  `requirements.txt` and is normally run via Docker, not the root venv.
- Log timestamps are **local PDT, not UTC**.
- v1's account **cannot place exchange-held stop orders** — the in-process trailing-stop monitor is
  the *only* protection; a downed v1 bot means unprotected open positions. This has not changed.
- `git checkout -- <file>` reverts to the **last commit**, not "your last edit" — if you're
  discarding one set of uncommitted changes to a file, check `git diff` first for anything else
  uncommitted in that same file that you don't want to lose (bit a session directly — see git log
  around 2026-08 for the recovery).
- `get_option_contracts` (Alpaca) silently truncates to ~100 results without an explicit `limit` —
  always pass `limit=500` and follow `next_page_token`.
- Alpaca has **no historical options bid/ask quote endpoint** — only a live one
  (`OptionLatestQuoteRequest`). This is a hard, confirmed constraint on any historical
  spread-reconstruction work.
- `closed_trades.csv`'s `symbol` column is a **composite key** (`"TICKER__strategy"`) for
  stock-shaped strategies (`stock`, `pead`, `pairs_long`, `pairs_short`, `qqq_macro`) — use the
  `underlying` column instead when joining against `trades.csv`'s bare ticker. Aggregate open/close
  counts can match exactly even when every individual join is wrong — don't trust totals alone.
