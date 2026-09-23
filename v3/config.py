"""
config.py (v3) — single source of truth for tunable parameters in v3.
Features:
  - Pluggable LLM provider (Ollama, Gemini, Anthropic, Moonshot)
  - Parameterized news context lookback window (default 7 days)
  - SQLite news archive configuration
  - Empirical execution geometry (~0.50 Delta, 6.5% spread cap, 10 OI floor, 30m timed exit)
  - Equity-relative sizing for small or medium accounts
"""

import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

# ── Paths ──────────────────────────────────────────────────────────────────────
BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = Path(os.environ.get("TRADER_V3_DATA_DIR", BASE_DIR / "data"))
LOG_DIR  = Path(os.environ.get("TRADER_V3_LOG_DIR",  BASE_DIR / "logs"))

DATA_DIR.mkdir(parents=True, exist_ok=True)
LOG_DIR.mkdir(parents=True, exist_ok=True)

# ── Credentials ────────────────────────────────────────────────────────────────
ALPACA_KEY    = os.environ.get("ALPACA_API_KEY", "")
ALPACA_SECRET = os.environ.get("ALPACA_SECRET_KEY", "")

# Dry run mode: logs simulated orders without submitting to Alpaca
DRY_RUN = os.environ.get("DRY_RUN", "0") == "1"

# ── Model & Scorer Configuration ───────────────────────────────────────────────
# Providers: 'ollama' | 'gemini' | 'anthropic' | 'moonshot'
SCORER_PROVIDER = os.environ.get("SCORER_PROVIDER", "ollama").lower()
SCORER_MODEL    = os.environ.get("SCORER_MODEL", "llama3.2")
SCORER_TIMEOUT_S = float(os.environ.get("SCORER_TIMEOUT_S", "25.0"))
SCORER_MAX_RETRIES = int(os.environ.get("SCORER_MAX_RETRIES", "3"))

# Ollama local endpoint
OLLAMA_BASE_URL = os.environ.get("OLLAMA_BASE_URL", "http://localhost:11434/v1")

# Cloud provider credentials
GEMINI_API_KEY    = os.environ.get("GEMINI_API_KEY", "")
GEMINI_PACING_SECS = float(os.environ.get("GEMINI_PACING_SECS", "4.5"))  # <= 13 RPM (stay under 15 RPM cap)
GEMINI_MAX_DAILY_CALLS = int(os.environ.get("GEMINI_MAX_DAILY_CALLS", "120"))  # safety cap under 1,500 daily quota
ANTHROPIC_API_KEY = os.environ.get("ANTHROPIC_API_KEY", "")
MOONSHOT_API_KEY  = os.environ.get("MOONSHOT_API_KEY", "")
MOONSHOT_BASE_URL = os.environ.get("MOONSHOT_BASE_URL", "https://api.moonshot.ai/v1")

# Cheap ticker corrector using local Ollama (picks primary beneficiary from news symbols)
TICKER_CORRECTOR_ENABLED = True
TICKER_CORRECTOR_MODEL   = "llama3.2"

# ── News Archive & Context Enrichment ──────────────────────────────────────────
NEWS_DB_PATH = DATA_DIR / "news.db"

# Trailing window for ticker-specific news history sent to LLM
NEWS_LOOKBACK_DAYS = int(os.environ.get("NEWS_LOOKBACK_DAYS", "7"))
NEWS_MAX_TICKER_ARTICLES = int(os.environ.get("NEWS_MAX_TICKER_ARTICLES", "10"))

# Trailing window for macro/market news sentiment flow
NEWS_MARKET_SENTIMENT_HOURS = int(os.environ.get("NEWS_MARKET_SENTIMENT_HOURS", "24"))

# Polling interval for live news stream
NEWS_POLL_SECS = int(os.environ.get("NEWS_POLL_SECS", "15"))

# Startup backfill: fetch past N days of news to ensure bot doesn't start blind
BACKFILL_ON_STARTUP = os.environ.get("BACKFILL_ON_STARTUP", "1") == "1"
BACKFILL_DAYS = int(os.environ.get("BACKFILL_DAYS", "7"))

# Stale news threshold: ignore real-time articles older than this before scoring
MAX_FEED_LAG_SECS = int(os.environ.get("MAX_FEED_LAG_SECS", "300"))

# ── Signal & Entry Thresholds ──────────────────────────────────────────────────
MIN_MAGNITUDE  = float(os.environ.get("MIN_MAGNITUDE", "0.35"))
MIN_CONFIDENCE = float(os.environ.get("MIN_CONFIDENCE", "0.70"))
COOLDOWN_SECS  = int(os.environ.get("COOLDOWN_SECS", "300"))

# Pre-score filters
PRESCORE_FILTER_ENABLED    = True
SOFT_CATALYST_GATE_ENABLED = True

# ── Empirical Contract Selection ───────────────────────────────────────────────
# Based on 5-week contract grid evaluation (7.58M rows / 652 signals):
# Delta ~0.50 cleanly outperforms 0.70, 0.30, and far-OTM lotto
TARGET_DELTA   = 0.50
DTE_MIN        = 14
DTE_MAX        = 28
STRIKE_LO_MULT = 0.95
STRIKE_HI_MULT = 1.10

# Hard empirical guardrails (discovered in research/contract_strategy_tuner.py):
# Spreads > 6.5% and low OI account for the vast majority of catastrophic losses
MAX_SPREAD_PCT    = 0.065   # 6.5% hard cap
MIN_OPEN_INTEREST = 10      # minimum OI floor
MIN_STOCK_PRICE   = 10.0    # avoid penny stock options distortion

# ── Exit Strategy ──────────────────────────────────────────────────────────────
# Empirical ranking showed timed 30-minute horizon outperforms trailing stops
# (which trigger prematurely on bid/ask noise)
EXIT_STRATEGY          = "fixed_horizon"
EXIT_HORIZON_MINUTES   = 30
TRAILING_STOP_ENABLED  = False   # optional fallback if fixed horizon is toggled off
TRAILING_STOP_PCT      = 0.15
TRAILING_STOP_FLOOR_M  = 10      # minimum hold floor before trailing stop is armed

# ── Sizing & Risk Management ───────────────────────────────────────────────────
# Equity-relative position sizing
POSITION_FRAC_OF_EQUITY = float(os.environ.get("POSITION_FRAC_OF_EQUITY", "0.10"))
MAX_CONTRACT_BUDGET_MULT = 1.5  # allow 1 contract up to 1.5x target budget
MAX_OPEN_POSITIONS       = int(os.environ.get("MAX_OPEN_POSITIONS", "5"))
DAILY_LOSS_LIMIT_PCT     = 0.02  # halt new trades if daily loss exceeds 2% of equity

# ── Market Regime Filters ──────────────────────────────────────────────────────
REGIME_GATE_ENABLED = True
REGIME_SPY_SMA_DAYS = 200
REGIME_MOM_DAYS     = 3
REGIME_DOWNDAY_WINDOW = 10
REGIME_DOWNDAY_MAX_DENSITY = 0.60

