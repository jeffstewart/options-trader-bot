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

# ═══════════════════════════════════════════════════════════════════════════════
# MODEL
# ═══════════════════════════════════════════════════════════════════════════════

# Scorer stays local Ollama for now -- the sonnet5 swap is real per-call spend and is jeff's call
# to make, not a default to flip silently. Kept as ONE isolated function (score_article in
# scoring.py) so swapping providers later is a single-function change, not a pipeline rewrite.
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

# TODO(size): the small-account selectivity sweep (small_account_news_call_sweep.py,
# stock_selectivity_sweep.py) found NO threshold on Ollama's OWN magnitude/confidence produces a
# stable "few winning trades" profile -- tightening just re-slices the same uncalibrated signal
# into a noisier sample. These are today's live-equivalent defaults, kept until the sizing
# backtest picks final numbers (likely alongside the sonnet5-swap decision, which DOES show a
# real edge on the same threshold shape).
MIN_MAGNITUDE    = 0.35
BASE_CONFIDENCE  = 0.70
CONFIDENCE_SLOPE = 0.25
COOLDOWN_SECS    = 300

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
# halting on dust losses. TODO(size): confirm 2% is still the right day-loss bar at this scale.
DAILY_LOSS_LIMIT_PCT     = 0.02
DAILY_LOSS_MIN_FLOOR_USD = 25   # small, deliberately << target equity (was $2,000 in v1)

# TODO(size): v1 ran up to 45 concurrent positions; at $500-1000 equity even single-digit
# concurrency risks over-concentration. Placeholder until the sizing backtest picks a number.
MAX_OPEN_POSITIONS = 5

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

NEWS_CALL_ENABLED       = True
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
LOTTO_MIN_MAGNITUDE  = 0.70
LOTTO_MIN_CONFIDENCE = 0.85
LOTTO_TARGET_DELTA   = 0.25
LOTTO_MIN_DTE        = 8
LOTTO_MAX_DTE        = 21
LOTTO_STRIKE_HI_MULT = 1.25
LOTTO_HARD_CAP_ENABLED  = True
LOTTO_HARD_CAP_MULT     = 3.0
LOTTO_EXIT_TIERS        = [(1.0, 0.40), (3.0, 0.30), (float("inf"), 0.20)]
LOTTO_MAX_HOLD_DAYS     = 3
# TODO(size): v1 sized this as a fixed $250 ("small, it's a lottery ticket") -- for a $500-1000
# account that's still 25-50% of equity in one convex bet. Should probably also become equity-
# relative; today's sonnet5 sweep showed lotto stays fundamentally tail-driven (top3-of-42 = 71%
# of P&L) regardless of scorer, so this is a bet-sizing question, not a selectivity one.
LOTTO_POSITION_FRAC_OF_EQUITY = 0.10

# PEAD -- stock leg only (matches what's currently live in v1; the ITM-options variant is still
# being validated on v1's pead_option_shadow, not duplicated here).
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

SEC_RSS_URL = (
    "https://www.sec.gov/cgi-bin/browse-edgar"
    "?action=getcurrent&type=8-K&dateb=&owner=include"
    "&count=10&search_text=&output=atom"
)
SEC_POLL_SECS  = int(os.environ.get("SEC_POLL_SECS", "3"))
SEC_USER_AGENT = os.environ.get("SEC_USER_AGENT", "Trader Bot v2 admin@example.com")
