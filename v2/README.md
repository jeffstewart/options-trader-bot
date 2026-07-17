# Trader Bot v2

A lean rebuild for the eventual $500-1000 real-money deploy. Runs independently of `core/`
(v1) — v1 keeps running unmodified as the ongoing data source / safety net while v2 is free to
diverge. See `~/.claude/projects/-Users-jeff-Claude-Trader/memory/project_v2_small_account_architecture.md`
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
  two-stage compromise.
- **No shadow-trade infrastructure by default.** Most shadow branches were superseded this
  session by backtesting the historical cache directly (faster, no weeks-long wait). Add a
  narrow one only for a question backtesting genuinely can't answer.
- **Equity-relative position sizing**, not a fixed dollar constant. v1's `MAX_POSITION_USD=$1,000`
  was ~the entire target account in one trade. Also fixes a real bug: v1's daily-loss-breaker
  fail-safe floor ($2,000) exceeded the whole target account, so if the equity fetch itself
  failed at exactly that moment the breaker gave zero protection.
- **Ticker-selection corrector** (new): a second, cheap (~0.7s) Ollama call that picks the primary
  beneficiary from the feed's own `symbols` metadata instead of trusting the primary scorer's
  free-generated ticker. Fixes the invented-ticker and wrong-beneficiary failure modes directly
  implicated in real live losses. Never used as a trade/no-trade gate.
- **Only news_call, lotto, and pead (stock leg) are implemented.** "Stock" as its own strategy
  isn't — the small-account sweep found zero positive-Sharpe cell for it on Ollama's own scoring
  at any threshold; the edge only showed up with sonnet5. Revisit alongside the scorer-swap
  decision, not before.
- **Scorer stays local Ollama for now.** The sonnet5 swap is real per-call spend — that's your
  call to make, not a default to flip. `scoring.py` isolates it to one function so swapping is a
  one-function change.

## Not yet finalized (see TODO(size) comments in config.py)

Position-size fraction, max open positions, news_call contract geometry (today's ITM/longer-DTE
finding may or may not survive a small budget's affordability filter), lotto sizing. These need a
dedicated small-account backtest pass before real money — the structural fixes are in place, the
exact numbers aren't tuned yet.

## Running it

```bash
pip install -r requirements.txt
cp .env .env.bak   # if you already put real keys in .env, back it up first
# edit .env: fill in a FRESH Alpaca PAPER account's ALPACA_API_KEY / ALPACA_SECRET_KEY
#            (not v1's account — this is meant to run side-by-side on its own book)
ollama serve &
ollama pull llama3.2
python bot.py
```

Tests: `pytest` (self-contained — own `pytest.ini`, own `tests/conftest.py`).
