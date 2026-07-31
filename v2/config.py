"""
config.py (v2) — single source of truth for tunable parameters, pruned for a small ($500-1000)
real-money account. See docs/HANDOFF.md and memory project_v2_small_account_architecture for the
research behind each change. Sections marked TODO(size) need a dedicated backtest pass before going
live with real capital — placeholders here are structurally correct but not yet tuned.

Dropped vs v1 (core/config.py), with why:
  - confirm/veto gate (mistral-large)   -- net-negative even on Ollama's own picks on rerun, and
    a clear net-negative once sonnet5's picks were tested (cuts P&L ~14% by vetoing winners)
  - pairs (market-neutral L/S)          -- structurally incompatible with a small account:
    ~66 concurrent legs in steady state (Little's law, no live time cap) vs. any realistic 3-5
    position budget, and 61-76% of its own picks are priced above what a small leg budget can
    afford for even 1 share (small_account_pairs_test.py)
  - bear_short, qqq_macro, materiality 2nd-pass, shadow router -- already retired/disabled in v1,
    no reason to carry dead config into a lean rebuild
  - all shadow-trade infrastructure     -- most shadow branches were superseded this session by
    backtesting the historical cache directly (faster, no weeks-long wait). Exceptions
    (regime_block_shadow, pead_option_shadow) stay on v1 -- no need to duplicate them here.
"""

import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

# ── Paths ──────────────────────────────────────────────────────────────────────
# v2 is self-contained: own .env, own data/logs dirs. Same TRADER_DATA_DIR/LOG_DIR override
# pattern as v1 so a container can mount a volume without code changes.
BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = Path(os.environ.get("TRADER_DATA_DIR", BASE_DIR / "data"))
LOG_DIR  = Path(os.environ.get("TRADER_LOG_DIR",  BASE_DIR / "logs"))

# ═══════════════════════════════════════════════════════════════════════════════
# CREDENTIALS  (own .env -- a FRESH paper account, not v1's)
# ═══════════════════════════════════════════════════════════════════════════════

ALPACA_KEY    = os.environ["ALPACA_API_KEY"]
ALPACA_SECRET = os.environ["ALPACA_SECRET_KEY"]

# Dry run: when set, place_option_trade/place_stock_trade log what they WOULD do (ticker,
# strategy, contract/shares, budget) and return without calling trading_client.submit_order.
# Everything upstream (Alpaca connectivity, Ollama scoring, regime gate, ticker corrector, the
# monitor/pollers starting cleanly) still runs for real -- only the actual order submission is
# short-circuited. Use for a startup smoke test before the small-account sizing is finalized.
DRY_RUN = os.environ.get("DRY_RUN", "0") == "1"

# ═══════════════════════════════════════════════════════════════════════════════
# MODEL
# ═══════════════════════════════════════════════════════════════════════════════

# ── PRIMARY SCORER: Sonnet 5 (swapped from local llama3.2 2026-07-29, jeff's call) ──────────
# Why: llama3.2 does not discriminate on the fields the strategy gates on. Across 3,145 bullish
# scores its confidence was 0.82 for 97% of signals and magnitude sat in a 0.6-0.75 band -- the
# selectivity grid's llama rows are literally IDENTICAL for conf 0.60 through 0.80 (n=368 every
# row), i.e. the confidence gate carried zero information. sonnet5 on the same lotto geometry:
# win 39.5% vs llama's best comparable cell at 30.7% (research/lotto_sonnet5_selectivity_grid.py).
# The point of the swap is a working dial, not a better average.
SCORER_MODEL   = os.environ.get("SCORER_MODEL", "claude-sonnet-5")
SCORER_EFFORT  = os.environ.get("SCORER_EFFORT", "low")   # low|medium|high|xhigh|max
# Sonnet 5 runs adaptive thinking by DEFAULT (unlike Sonnet 4.6, where omitting the field meant no
# thinking). For this workload -- a bounded classification against a fixed schema, on a hot path
# where news edge decays in minutes -- adaptive+low is the documented starting point, but thinking
# means variable latency. Flip to "disabled" for the tightest, most predictable latency once the
# logged numbers below tell us what we're actually paying. One-line change either way.
SCORER_THINKING = os.environ.get("SCORER_THINKING", "adaptive")   # adaptive|disabled
SCORER_MAX_TOKENS = 1024        # schema-bounded output; nowhere near needing streaming
SCORER_TIMEOUT_S  = float(os.environ.get("SCORER_TIMEOUT_S", "20"))
SCORER_MAX_RETRIES = int(os.environ.get("SCORER_MAX_RETRIES", "4"))  # see DNS note in scoring.py
ANTHROPIC_API_KEY = os.environ.get("ANTHROPIC_API_KEY", "")

# Ollama is still used for the TICKER CORRECTOR only (below) -- deliberately left local/free.
OLLAMA_BASE_URL = os.environ.get("OLLAMA_BASE_URL", "http://localhost:11434/v1")
OLLAMA_MODEL    = os.environ.get("OLLAMA_MODEL", "llama3.2")

# Ticker-selection corrector (validated 2026-07-16, prompt_v2_test.py): a second, cheap (~0.7s)
# Ollama call that picks the primary beneficiary from the feed's own `symbols` metadata, instead
# of trusting the primary scorer's free-generated ticker. 85-96% agreement with sonnet5 on the
# hard cases (where the primary scorer's ticker was wrong) across 850 articles tested; fixes the
# invented-ticker (VIKT, PELO) and wrong-beneficiary (BEAM/Prime, Dell/Pentagon->MSFT) failure
# modes directly implicated in real live losses. NEVER used as a trade/no-trade gate -- it only
# corrects the symbol on signals that already passed every other gate (it force-picks a candidate
# ~96% of the time even when there's no real beneficiary, so it must not be trusted for that).
TICKER_CORRECTOR_ENABLED = True

# ═══════════════════════════════════════════════════════════════════════════════
# PRE-SCORE FILTERS -- patterns + matcher functions live in filters.py (unchanged from v1,
# validated, ported as-is). These two flags just toggle them on/off.
# ═══════════════════════════════════════════════════════════════════════════════

PRESCORE_FILTER_ENABLED    = True
SOFT_CATALYST_GATE_ENABLED = True

# Stale-news drop -- skip anything older than this BEFORE scoring (bounds feed lag under a burst).
MAX_FEED_LAG_SECS = int(os.environ.get("MAX_FEED_LAG_SECS", "300"))

# ═══════════════════════════════════════════════════════════════════════════════
# SIGNAL THRESHOLDS
# ═══════════════════════════════════════════════════════════════════════════════

# ONE threshold pair, checked in ONE place (2026-07-29, jeff's call). v1's sliding floor
# (`required_conf = BASE_CONFIDENCE + (1 - magnitude) * CONFIDENCE_SLOPE`) is GONE from v2:
#   - It was a second, redundant gate. With lotto the only enabled leg, a signal that clears the
#     lotto thresholds and then fails a separate global floor is just a confusing way to say no.
#   - Its constants were silently SCORER-SPECIFIC. The identical formula passed ~78% of llama3.2's
#     bullish signals and ~3% of sonnet5's, because llama pins confidence at 0.82 while sonnet5
#     runs a median of 0.55 and never exceeded 0.85 across 1,682 scored articles. A "risk setting"
#     that inverts meaning when the scorer changes is a calibration constant wearing a disguise.
#   - v1's own comment table documented the pre-2026-06-02 numbers for over a month (fixed
#     2026-07-29), understating the strictest requirement 0.71 vs 0.862 -- which is exactly how
#     the sonnet5 incompatibility stayed hidden.
#
# ⚠️ THESE NUMBERS ARE ON SONNET 5's SCALE, not llama3.2's. Validated in
# research/lotto_sonnet5_selectivity_grid.py (n=38, win 39.5%, Sharpe +5.49, best total-$ cell;
# llama's best comparable cell was win 30.7%). On llama3.2 a 0.70 confidence floor is a near-no-op
# (97% of its bullish signals score exactly 0.82) -- do NOT run this bot on llama3.2 with these
# values expecting them to filter anything. Re-derive on any scorer change; see
# memory project_v2_small_account_architecture.
MIN_MAGNITUDE  = 0.35
MIN_CONFIDENCE = 0.70
COOLDOWN_SECS  = 300

# ═══════════════════════════════════════════════════════════════════════════════
# POSITION SIZING -- equity-relative, not a fixed dollar figure (v1's MAX_POSITION_USD=$1,000
# was ~the entire target account in one trade; a $500-1000 account must size off its OWN current
# equity, fetched at runtime, not a constant tuned for an $83k book).
# ═══════════════════════════════════════════════════════════════════════════════

# TODO(size): placeholder fraction, not yet backtested at this exact value. Effective size =
# equity * MAX_POSITION_FRAC * magnitude * confidence (floor 10% of the fraction, same shape as
# v1's scale_position_usd, just anchored to equity instead of a constant).
MAX_POSITION_FRAC_OF_EQUITY = 0.15

# Real fix for a live bug found 2026-07-16: v1's DAILY_LOSS_LIMIT_PCT carried a $2,000 FLOOR --
# at $500-1000 equity that floor exceeds the entire account, so the breaker could never trip
# (zero protection). No floor here; a pure equity percentage with a tiny minimum just to avoid
# halting on dust losses.
# ⚠️ DAILY_LOSS_LIMIT_PCT is NOT set here any more -- it is DERIVED further down, after the lotto
# sizing constants it depends on. See "DAILY LOSS BREAKER" near LOTTO_POSITION_FRAC_OF_EQUITY.
DAILY_LOSS_MIN_FLOOR_USD = 25   # small, deliberately << target equity (was $2,000 in v1)

# Decided 2026-07-18: with news_call/PEAD disabled, this is now effectively the lotto cap. The
# same-day EOD exit rule (LOTTO_SAME_DAY_EXIT) collapsed typical hold time to a few hours, so
# lotto positions rarely overlap at all (most days: 0-1 signal; Little's-Law concurrency is near
# zero, not the multi-day-hold math that drove v1's old 45-position cap). 3 is a safety ceiling
# for a rare cluster day, not a scarce resource being rationed -- at LOTTO_POSITION_FRAC_OF_EQUITY
# below, 3 slots caps worst-case simultaneous exposure at 30% of equity.
MAX_OPEN_POSITIONS = 3

# Skip a contract if ONE costs more than this x the position budget -- prevents the qty=max(1,..)
# rule from forcing a wildly oversized fill (this is EXACTLY the mechanism that let a live
# $12,201 SNDK contract through v1 at the old MAX_CONTRACT_BUDGET_MULT=3.0). Tightened hard for a
# small account: a position budget of ~$75-150 can't absorb a 3x overage.
MAX_CONTRACT_BUDGET_MULT = 1.15

# ═══════════════════════════════════════════════════════════════════════════════
# STRATEGY ON/OFF -- only news_call, lotto, pead (stock leg). No pairs, no bear_short,
# no qqq_macro (all retired/incompatible; see module docstring). Stock-as-a-strategy is NOT
# implemented in v2 yet: the small-account sweep found ZERO positive-Sharpe cell for stock on
# Ollama's own scoring at ANY threshold (20-cell grid, best was -0.29) -- the edge only showed up
# with sonnet5 scoring. Revisit alongside the scorer-swap decision, not before.
# ═══════════════════════════════════════════════════════════════════════════════

NEWS_CALL_ENABLED       = False  # disabled 2026-07-18: no non-lottery Sharpe shape found at any
                                  # threshold/scorer tested (small_account_news_call_sweep.py) --
                                  # lotto alone is simpler and covers the same convex-bet niche.
NEWS_CALL_MIN_MAGNITUDE = 0.75
NEWS_CALL_DTE_MIN       = 7
NEWS_CALL_DTE_MAX       = 13
NEWS_CALL_TARGET_DELTA  = 0.40   # TODO(size): today's ITM/longer-DTE finding (news_call_options_test.py)
                                 # improves median/win% but costs more per contract -- may not survive
                                 # a small budget's affordability filter. Needs the small-account
                                 # geometry x budget cross before changing from the current live value.
NEWS_CALL_MAX_PER_TICKER_DAY = 1
NEWS_CALL_MAX_HOLD_DAYS  = 3
NEWS_CALL_TRAIL_PCT      = 0.40
NEWS_CALL_EXIT_TIERS     = [(0.15, 0.40), (0.35, 0.25), (float("inf"), 0.15)]

LOTTO_ENABLED        = True
# Aliases, not independent knobs: lotto IS the only enabled leg, so its entry bar and the
# pipeline's single threshold pair are the same thing by definition. Bound by name (not copied
# values) so they cannot drift apart. If news_call/PEAD are ever re-enabled, give THOSE legs their
# own constants (NEWS_CALL_MIN_MAGNITUDE already exists) rather than un-aliasing these.
# Previous values were 0.70/0.85 on llama3.2's scale; 0.85 was above what llama emits for 97% of
# its signals, which is why v2 opened exactly one position attempt in its entire lifetime.
LOTTO_MIN_MAGNITUDE  = MIN_MAGNITUDE
LOTTO_MIN_CONFIDENCE = MIN_CONFIDENCE
LOTTO_TARGET_DELTA   = 0.25
LOTTO_MIN_DTE        = 8
LOTTO_MAX_DTE        = 21
LOTTO_STRIKE_HI_MULT = 1.25
# ── 3x take-profit: DISABLED 2026-07-31 (jeff) ───────────────────────────────────────────────
# Measured across cap levels on research/lotto_intraday_exit_test.py (trail_15pct rule, n=14):
#     3x   Sigma$ +5,949  sharpe +0.85   |  5x  Sigma$ +5,919  sharpe +0.84
#     8x   Sigma$ +6,268  sharpe +0.88   |  none Sigma$ +7,991  sharpe +0.71
# Two things decided it:
#   1. NON-MONOTONIC in cap level -- 5x is worse than 3x, 8x better than both. A real risk/return
#      dial would move smoothly, so the sweep cannot resolve WHICH cap is right at this n. Only the
#      binary capped-vs-uncapped distinction survives.
#   2. avgLoss (-$1) and worst (-$2) are IDENTICAL at every cap level including none. The cap only
#      ever exits winners, so it contributes ZERO downside protection -- the trail and the same-day
#      EOD exit do all of that. Dropping it costs nothing in risk, it only trades Sharpe for dollars
#      (+34% total$, -0.14 sharpe).
# Also relevant: LOTTO_SAME_DAY_EXIT already forces a close at the entry day's close, so a winner
# can only run for hours, not days -- the cap was truncating a single-session spike, not a
# multi-day compounding tail. And the trail means a position at +250% carries a stop at +212%,
# so it cannot round-trip anyway; the cap and the trail were partly redundant.
# Re-enable with one line if forward data shows uncapped variance is unacceptable.
LOTTO_HARD_CAP_ENABLED  = False
LOTTO_HARD_CAP_MULT     = 3.0
# Exit rule replaced 2026-07-17 (research/lotto_intraday_exit_test.py, 15-min-bar Black-Scholes sim
# on the sonnet5 mag>=0.3/conf>=0.70 gate): the old peak-trailing tiers let a green-at-the-open trade
# drift back to a big loss by the close. Same-day EOD exit + a stop-loss anchored to ENTRY (not peak)
# beat the old tiers on both total $ and win% while cutting avg loss roughly in half (-$22 -> -$12 at
# $100 budget) and capping worst-case loss (-$40 vs uncapped). All of 15-35% stop thresholds tested
# performed similarly; 20% chosen as tight enough to matter without clipping every recoverable dip.
LOTTO_STOP_LOSS_PCT     = 0.20   # hard stop anchored to entry premium, checked continuously

# ── TRAILING stop (2026-07-31, jeff) ──────────────────────────────────────────────────────────
# The exit level must RISE as the contract gains. An entry-anchored stop alone cannot do that:
# nothing ever lifts it above its opening level, so a winner short of the 3x cap can only be
# banked by the same-day EOD exit -- which makes "profit" contingent on holding to the close
# rather than on the trade working. AMZN 2026-07-31 peaked +30.4% and stopped out at -22.6%.
# (A first attempt here used two discrete tiers -- breakeven at +15%, +25% at +50% -- but that
# leaves the stop parked at breakeven across the whole +15%..+50% range, which is the same
# complaint in a smaller form. A continuous trail is what the risk preference actually asks for.)
#
# Stop = max(entry-anchored floor, peak * (1 - LOTTO_TRAIL_PCT)):
#   - the floor caps worst case at -LOTTO_STOP_LOSS_PCT of entry
# On research/lotto_intraday_exit_test.py (re-read 2026-07-31, do NOT trust second-hand summaries
# of it): that test does NOT show entry-anchored beating peak-trailing -- it never tested a peak
# trail at all. Its stop fired ONCE in 14 trades ({'eod': 13, 'stop-loss': 1}) and every threshold
# from 15% to 35% returned the same total ($8,279-8,296), so its real finding is "the same-day EOD
# exit is what wins; the stop level only shapes avgLoss/worst". Its two LOCK variants had the best
# risk shape in the table (breakeven_lock sharpe +0.84, eod_with_lock avgLoss -$1 vs baseline -$16),
# which mildly supports lifting the stop as the position gains. n=14, simulated BS premiums, and a
# 71-79% win rate vs ~35-40% live -- a winner-skewed sample where stops rarely bind.
#   - the trail lifts the exit continuously with every new peak, and never loosens because
#     peak_price is monotonic
# On the real AMZN 30s path a 20% trail returned -0.6% vs the live rule's -22.6%.
# Set to 0.15 (jeff, 2026-07-31). The hourly sim could not separate 15% from 20% -- both
# returned byte-identical results with the cap on -- so this is a preference call, not a
# measured one. position_paths.csv now records real 30s quotes; exit_path_replay.py can
# settle it properly once a handful of lotto positions have logged full paths.
LOTTO_TRAIL_PCT = 0.15
LOTTO_SAME_DAY_EXIT     = True   # force close at EOD of the entry day regardless of P&L
LOTTO_MAX_HOLD_DAYS     = 3      # fallback safety net only -- same-day exit should always fire first
# Entry cutoff (2026-07-18, research/lotto_entry_cutoff_test.py): since every lotto position gets
# force-closed the SAME day regardless of P&L, a signal firing late in the session leaves almost
# no runway for the stop-loss or any real move to matter before the forced exit -- pure spread
# cost for no developed edge. Backtest (Ollama live-gate-adjacent signals, n=106 across buckets)
# didn't show a clean monotonic "more runway = better" gradient -- lotto's convex few-big-winners
# structure means most bucket averages (n=18-38) are noise-dominated. The one bucket that WAS
# directionally consistent with the mechanical concern: 0.0-0.5h before close had both the worst
# avg $ (-$18) and a below-average win rate (29%) of any populated bucket, while 0.5-1.0h (n=38)
# was actually the strongest performer -- so the cutoff is deliberately small (doesn't touch the
# well-performing 0.5-1.0h window) rather than an aggressive number the data doesn't support.
LOTTO_ENTRY_CUTOFF_MIN_BEFORE_CLOSE = 30   # stop opening NEW lotto positions inside this window
# Decided 2026-07-18 (equity-relative, not v1's fixed $250 -- scales automatically with account
# growth, per jeff's ask: a well-performing bot should size up its bets as equity grows, rather
# than needing a manual re-tune). 10% survives a realistic losing streak reasonably well: at the
# live gate's ~35-40% win rate, 5 losses in a row happens ~12% of the time, but since the exit
# rule caps an average loss to ~12-22% of the BUDGET (not the whole budget), that's only ~1-2%
# of equity per average loss -- a bad 5-loss stretch costs roughly 7-10% equity, not 50%. Also
# matches the $100-budget tier (at today's $1,000 paper equity) that all the exit-rule testing
# this session was actually validated at, rather than an untested extrapolation.
LOTTO_POSITION_FRAC_OF_EQUITY = 0.10

# ═══════════════════════════════════════════════════════════════════════════════
# DAILY LOSS BREAKER -- derived, NOT a free-standing number
# ═══════════════════════════════════════════════════════════════════════════════
# The 2026-07-31 AMZN trade exposed an algebraic identity, not bad luck:
#     daily limit      = equity * DAILY_LOSS_LIMIT_PCT              = equity * 0.02
#     max loss / trade = equity * LOTTO_POSITION_FRAC * STOP_LOSS   = equity * 0.10 * 0.20
#                                                                   = equity * 0.02
# Both were EXACTLY $19.50 at $974.96 equity. A breaker meant to stop a bad *day* was calibrated
# to fire on the first normal stop-out, so it could never do its job -- it was equivalent to
# "halt after any losing trade". (The AMZN stop actually realised -29.76%, not the intended -20%,
# so it tripped with room to spare: $25 loss vs a $19.50 limit.)
#
# Expressing the limit as a MULTIPLE of the per-trade worst case makes that collision impossible
# to reintroduce: retune position size or the stop and the breaker moves with them.
# 3 stop-outs is chosen to match MAX_OPEN_POSITIONS -- the breaker fires when the entire
# concurrent book has gone against us, which for a strategy that trades ~1x/week is a genuinely
# anomalous day rather than an ordinary loss.
# NOTE: realised stop-outs overshoot the intended level (the 30s monitor interval lets a wide
# deep-OTM spread gap through), so 3 INTENDED stop-outs is ~2.3 REALISED ones at the one data
# point we have. Revisit once more stops have actually fired.
DAILY_LOSS_MAX_STOPOUTS = 3
DAILY_LOSS_LIMIT_PCT = round(
    LOTTO_POSITION_FRAC_OF_EQUITY * LOTTO_STOP_LOSS_PCT * DAILY_LOSS_MAX_STOPOUTS, 4)   # = 0.06

# PEAD -- stock leg only (matches what's currently live in v1; the ITM-options variant is still
# being validated on v1's pead_option_shadow, not duplicated here).
PEAD_ENABLED        = False  # disabled 2026-07-18: best win-rate shape found all session (53.9%,
                              # small_account_lotto_pead_sweep.py) but too slow/capital-intensive
                              # for this account -- ~11.7 concurrent positions needed (median hold
                              # rides the full 10d cap), rejected on that basis, not signal quality.
PEAD_EXIT_TIERS     = [(float("inf"), 0.20)]   # flat 20% trail
PEAD_BREAKEVEN_LOCK = 0.0
PEAD_MAX_HOLD_DAYS  = 10
PEAD_MAX_POSITIONS  = 4   # TODO(size): reserved sub-cap pattern kept from v1, shrunk for MAX_OPEN_POSITIONS=5

# ═══════════════════════════════════════════════════════════════════════════════
# OPTIONS PARAMETERS (generic contract-selection defaults; per-strategy overrides above)
# ═══════════════════════════════════════════════════════════════════════════════

MIN_DAYS_TO_EXPIRY = 14
MAX_DAYS_TO_EXPIRY = 21
TARGET_DELTA       = 0.50

# ── Contract-mispricing switch (2026-07-18) ────────────────────────────────────────────────────
# jeff's idea: instead of always taking the contract nearest the target delta/DTE, check whether a
# same-expiry neighbor is priced BELOW the local IV smile (implied vol backed out from its own
# real quote, fit a local quadratic across ~7 nearby strikes) -- i.e. the market hasn't repriced
# that specific strike as fast as the rest of the chain after the news. research/
# lotto_mispricing_test.py found a small, promising-but-unvalidated signal on a historical n=15
# (switching improved win%/avg $/total $, but individual residual gaps were often within
# plausible fit-noise range); v1 (core/bot.py) runs the SAME check as a SHADOW ONLY (logs the
# comparison, never changes what gets bought) so the two can be compared going forward. jeff
# opted to wire this in live for v2 despite the small validated sample ("I have a lot of faith in
# this") -- MISPRICING_MIN_RESIDUAL_GAP exists specifically to guard against acting on pure fit
# noise: only switch when the gap is meaningfully larger than what spot-checking the historical
# cases showed noise alone could produce (most noise-range gaps were <=0.02 IV points).
MISPRICING_SWITCH_ENABLED   = True
MISPRICING_MIN_RESIDUAL_GAP = 0.02   # IV points; below this, don't switch off the target contract

# ── Regime gate -- runs FIRST, before any scoring call (local or paid) ─────────────────────────
# With pairs/bear_short/qqq_macro all gone, NOTHING in v2 needs scoring during a downtrend (no
# all-regime strategy remains) -- so the gate can sit strictly before scoring, no two-stage
# compromise needed. This directly saves the Ollama call (and, if the scorer swap ever happens,
# real API spend) on every article during a blocked regime.
REGIME_FILTER_ENABLED = True
REGIME_INDEX          = "SPY"
REGIME_MA_DAYS         = 200
REGIME_MOMENTUM_DAYS   = 3
# Chop brake (ported from v1, deployed there 2026-07-16): block when >= this fraction of the last
# REGIME_DOWNDAY_WINDOW SPY sessions closed down. Threshold is knife-edged on a single 4-week
# sample in v1's forward data -- watch v1's regime_block_shadow for validation before trusting it
# further here.
REGIME_DOWNDAY_WINDOW      = int(os.environ.get("REGIME_DOWNDAY_WINDOW", "10"))
REGIME_DOWNDAY_MAX_DENSITY = float(os.environ.get("REGIME_DOWNDAY_MAX_DENSITY", "0.60"))
# No bypass in v2 (v1's REGIME_BYPASS_MIN_MAGNITUDE mechanism is gone entirely) -- it was tightly
# coupled to the confirm/veto gate's fail-closed safety logic, which no longer exists, and the
# live evidence that killed it (8% win, -$2.0k) applies regardless of gate architecture.

# ── Data-sanity / sizing guards ─────────────────────────────────────────────────
MAX_SANE_STOCK_PRICE = 10_000

# ── Live price cross-check ──────────────────────────────────────────────────────
PRICE_CROSSCHECK_ENABLED = True
PRICE_CROSSCHECK_TOL     = 0.25

# ═══════════════════════════════════════════════════════════════════════════════
# GUARDRAILS
# ═══════════════════════════════════════════════════════════════════════════════

MIN_STOCK_PRICE        = 5.00
MIN_MARKET_CAP_B       = 2.0
MAX_SPREAD_PCT         = 0.20
MAX_MONITOR_SPREAD_PCT = float(os.environ.get("MAX_MONITOR_SPREAD_PCT", "0.60"))
MIN_OPEN_INTEREST      = 100

# ═══════════════════════════════════════════════════════════════════════════════
# TRAILING STOP (generic; per-strategy overrides above)
# ═══════════════════════════════════════════════════════════════════════════════

TRAILING_STOP_PCT = 0.10
EXIT_TIERS        = [(1.00, 0.30), (3.00, 0.20), (float("inf"), 0.13)]
MONITOR_INTERVAL  = 30
TIME_STOP_EOD_WINDOW_MIN = 15

# ═══════════════════════════════════════════════════════════════════════════════
# TRADEABLE UNIVERSES / NEWS SOURCES
# ═══════════════════════════════════════════════════════════════════════════════

TRADEABLE_ETFS = {"SPY", "QQQ", "IWM", "GLD", "TLT"}

# SEC EDGAR 8-K feed: DISABLED 2026-07-29 (jeff's call). The feed has never demonstrated
# profitability in v1, and the 6-month/11k-filing backtest found no exploitable drift in either
# the overnight or the intraday RTH lane with llama3.2 selection (research/overnight_8k_backtest.py)
# — so v2 shouldn't spend request budget or (once the scorer swap lands) paid API tokens on it.
# It was also the noisiest thing in v2's log: ~1,000 poll failures with an empty error string plus
# real DNS failures resolving www.sec.gov from inside the container. Kept as a one-line re-enable
# rather than deleted code, in case a better scorer revives the lane.
SEC_FEED_ENABLED = os.environ.get("SEC_FEED_ENABLED", "false").lower() in ("1", "true", "yes")
SEC_RSS_URL = (
    "https://www.sec.gov/cgi-bin/browse-edgar"
    "?action=getcurrent&type=8-K&dateb=&owner=include"
    "&count=10&search_text=&output=atom"
)
SEC_POLL_SECS  = int(os.environ.get("SEC_POLL_SECS", "3"))
SEC_USER_AGENT = os.environ.get("SEC_USER_AGENT", "Trader Bot v2 admin@example.com")

# Alpaca's free data plan allows 1 concurrent websocket connection PER LOGIN, not per paper
# account -- v1 and v2 use different API keys but the same underlying Alpaca login, so v1's own
# stock+news streams permanently occupy that slot and v2's streams can never authenticate while
# v1 is running (root-caused 2026-07-19). REST calls aren't subject to this cap. Default OFF
# (REST polling) so the two bots can run concurrently at no cost during v2's testing phase; flip
# to True once v1 retires or the account moves to a paid plan with a higher connection limit.
USE_ALPACA_STREAMS = os.environ.get("USE_ALPACA_STREAMS", "false").lower() == "true"
NEWS_POLL_SECS = int(os.environ.get("NEWS_POLL_SECS", "5"))
