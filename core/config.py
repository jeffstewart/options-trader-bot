"""
config.py — single source of truth for all tunable parameters.

Both bot.py and backtest.py import from here, so any change you make
is immediately reflected in both live trading and backtesting.

Credentials are loaded from .env — never hardcode keys here.
"""

import os
import re
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

# ── Project paths (2026-06-24 restructure: code in core/+research/, data in data/, logs in logs/) ──
# config.py lives in core/, so the project root is two levels up. Daemons run with CWD=data/ (set in
# spawn_daemon.py) so the codebase's bare data-file refs (open("foo.json")) resolve into data/.
# DATA_DIR/LOG_DIR are env-overridable (TRADER_DATA_DIR / TRADER_LOG_DIR) so a container can mount a
# persistent volume anywhere for state — defaults are today's in-repo dirs (behavior-neutral).
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = Path(os.environ.get("TRADER_DATA_DIR", BASE_DIR / "data"))   # caches, CSVs, bot_state.json
LOG_DIR  = Path(os.environ.get("TRADER_LOG_DIR",  BASE_DIR / "logs"))

# ═══════════════════════════════════════════════════════════════════════════════
# CREDENTIALS  (loaded from .env)
# ═══════════════════════════════════════════════════════════════════════════════

ALPACA_KEY    = os.environ["ALPACA_API_KEY"]
ALPACA_SECRET = os.environ["ALPACA_SECRET_KEY"]
NEWSAPI_KEY   = os.environ.get("NEWSAPI_KEY", "")   # optional

# ═══════════════════════════════════════════════════════════════════════════════
# MODEL
# ═══════════════════════════════════════════════════════════════════════════════

OLLAMA_BASE_URL = os.environ.get("OLLAMA_BASE_URL", "http://localhost:11434/v1")
OLLAMA_MODEL    = os.environ.get("OLLAMA_MODEL", "llama3.2")

# ── Groq scorer A/B (forward latency + free-tier rate-limit test) ─────────────
# SHADOW only: on every live signal we ALSO score on Groq's hosted model, off the
# trade path, and log latency + agreement + rate-limit status → scorer_ab.csv.
# Live trading stays on local Ollama. Reuses the LAB_HOSTED_* keys already in .env.
GROQ_SCORER_SHADOW_ENABLED = os.environ.get("GROQ_SCORER_SHADOW_ENABLED", "1") == "1"
GROQ_BASE_URL = os.environ.get("LAB_HOSTED_BASE_URL", "")
GROQ_KEY      = os.environ.get("LAB_HOSTED_KEY", "")
GROQ_MODEL    = os.environ.get("LAB_HOSTED_MODEL", "")
# A/B PRIMARY-scorer bake-off: score every article with EACH of these Groq models (one scorer_ab.csv
# row per model, tagged `groq_model`) → groq_vs_ollama_pnl.py compares each vs Ollama on forward P&L.
# Validating a move to paid Groq scoring in the cloud. NOTE: N models = N Groq calls/article — heavier
# on the free-tier rate limit; trim to one (or add a paid key) once a winner emerges.
GROQ_AB_MODELS = [m.strip() for m in os.environ.get(
    "GROQ_AB_MODELS",
    # llama-4-scout dropped 2026-06-24 (Groq deprecating it). gpt-oss-120b NOT added here — it would
    # compete with the backtest for gpt-oss's tight daily quota (8K TPM / 200K TPD) and need the
    # reasoning_effort=low param the live path doesn't pass. The backtest covers gpt-oss instead.
    "llama-3.3-70b-versatile").split(",") if m.strip()]

# ── Pre-score noise filter (2026-06-22) ────────────────────────────────────────────────
# Drop high-confidence-NOISE articles BEFORE the ~3-8s scoring (saves Ollama compute + the Groq
# A/B quota, and reduces backlog pressure). Conservative, validated on the corpus (drops ~1.7%,
# 0 genuine fresh catalysts): macro/index wraps (no tradeable single name), explicit listicles,
# and Benzinga "What's Going On With X" REACTIVE recaps (the move already happened → trading is
# late). NOT the analyst/technical/M&A categories — those stay in the post-score soft-catalyst gate
# (shadowed) until that gate is validated, then can be promoted here for bigger savings.
PRESCORE_FILTER_ENABLED = True
PRESCORE_SKIP_PATTERNS = (
    # macro / index wraps — no tradeable single name
    "stock futures", "dow futures", "nasdaq futures", "s&p futures", "futures rise", "futures fall",
    "futures point", "futures slip", "premarket", "pre-market", "market wrap", "this week in markets",
    "stock market today", "closing bell", "opening bell", "stocks close higher", "stocks close lower",
    "treasury yield", "10-year yield",
    # listicles / opinion roundups
    "stocks to watch", "stocks to buy", "stocks to avoid", "stocks to consider",
    "best stocks to", "top stocks to",
    # Benzinga reactive recaps — the move already happened
    "what's going on with", "whats going on with",
    # index reconstitution / inclusion (index_event_backtest 2026-06-24): added stocks show NO
    # directional edge (3-day fwd avg +0.1%, ~50% positive) yet the model over-scores them to 0.85,
    # tripping the regime bypass (GOOG/DJIA −36% live). Pure index noise → drop pre-score.
    "russell 3000", "russell 2000", "russell 1000", "russell microcap",
    "s&p 600", "s&p 400", "s&p smallcap", "s&p midcap",
    "dow jones industrial average", "nasdaq biotechnology index",
    "to join the s&p", "joins the s&p", "joins s&p", "added to the s&p",
    "to join the russell", "joins the russell", "added to the russell", "added to russell",
    "to join the nasdaq", "join the nasdaq-100", "join the dow jones", "joins the dow jones",
)

# Regex skip patterns — for noise that needs more than a fixed substring. Benzinga's algorithmic
# "returns-calculator" / performance-recap articles ("$100 Invested In X 10 Years Ago Would Be Worth…",
# "If You Invested $1000 In Y…", "Z Has Outperformed The Market Over The Past 10 Years…") are pure
# retrospective filler with NO forward catalyst, yet the model over-scores them (264 of 511 hit bullish
# mag≥0.75 → would trade, e.g. the QCOM pairs entry). They're ~12% of all articles. Anchored on the
# retrospective phrasing so REAL capex ("$100M invested in a plant") is not caught — validated 2026-06-26:
# 511/4420 matched, ZERO without a retrospective marker. (gate_threshold_sweep showed the gate's mag bar
# is fine — the weak-catalyst leak is CONTENT, so filter it pre-score.)
PRESCORE_SKIP_REGEXES = (
    re.compile(r"\bif you (?:had )?invested\b", re.I),                     # "If You Invested $1000 In X…"
    re.compile(r"\$[\d,]+\s+invested\s+in\b", re.I),                        # "$100 Invested In X …" (bare $ → not $100M)
    re.compile(r"\d+\s+years?\s+ago\s+would\s+be\s+worth", re.I),           # "…10 Years Ago Would Be Worth…"
    re.compile(r"\boutperformed\b[^.]{0,40}\bover the (?:past|last)\s+\d+\s+(?:year|month)", re.I),
)

# ── Stale-news drop (2026-06-22) ───────────────────────────────────────────────────────
# Ollama scores each article SYNCHRONOUSLY (~3-8s) in the async news handler, blocking the event
# loop; with no backpressure a news burst builds a backlog the bot can never clear → observed
# ~75-min feed lag (it was scoring hour-old news, defeating the fast-reaction premise). FIX: skip
# any article older than MAX_FEED_LAG_SECS BEFORE the expensive scoring — a skip costs ~0, so the
# bot blows through the backlog instantly, stays on live news, and the lag is bounded to this.
MAX_FEED_LAG_SECS = int(os.environ.get("MAX_FEED_LAG_SECS", "300"))   # 5 min; tighten for fresher-only

# ── Soft-catalyst gate (loser_analysis 2026-06-21) ─────────────────────────────────────
# The bot's REAL losers cluster by CATALYST TYPE, all scored ~0.85 by the model despite the
# prompt's rules: analyst/price-target notes (-$4,154, 10% win), acquirer-side M&A (-$977 —
# acquirers drop), technical/chart signals (0% win). llama3.2 over-scores these "soft" catalysts
# regardless of prompt guidance, so we GATE them deterministically: a bullish signal whose
# headline/body matches these patterns is NOT traded for real — it's shadowed instead, so we can
# verify the gate was right (join shadow P&L to soft_catalyst_blocked tags). Tune patterns freely.
SOFT_CATALYST_GATE_ENABLED = True
SOFT_CATALYST_PATTERNS = (
    # analyst ratings / price targets — opinion, already priced, not a surprise
    "price target", " pt ", "raises pt", "lowers pt", "analyst", "analysts", "upgraded",
    "downgraded", "rating to", "reiterat", "initiates coverage", "initiated coverage",
    "buy rating", "sell rating", "outperform rating", "underperform rating", "fab five",
    "their forecast", "their price", "their estimates",
    # technical / chart signals — not a fundamental catalyst
    "trading signal", "flashed", "circuit breaker", "breakout", "death cross", "golden cross",
    "oversold", "overbought", "moving average", "chart pattern", "technical indicator",
    # acquirer-side M&A — the BUYER usually drops (target pops; "to be acquired" is NOT blocked)
    "to acquire", "agreed to acquire", "in discussions to buy", "in talks to buy",
    "is acquiring", "to purchase", "deal to buy", "bid for",
)

# ── Position sizing: magnitude is UNCALIBRATED (missed_movers rank-IC ≈ 0 across 1d/3d/event-day;
# loser avg mag 0.77 vs winner 0.78). So magnitude-scaled sizing is false precision — size FLAT.
# NONMAG_SIZE_FRAC ≈ the old average mag×conf (~0.75×0.85) so average exposure is ~unchanged.
MAGNITUDE_SIZING_ENABLED = False
NONMAG_SIZE_FRAC         = 0.65

# ── Confirm/veto gate: Groq llama-4-scout CONFIRMATION gate over the local Ollama scorer ──
# Ollama scores every article (fast/local/free). When its signal would open a REAL leg, the
# gate re-scores the SAME article (same unified_v1 prompt) as an independent second opinion.
#   confirm → trade for real (scorer=gemini)        ┐ internal scorer TAGS kept as "gemini"/
#   veto    → DON'T trade; shadow the pick instead  ┤ "gemini_veto" for analytics continuity
#   cap/err → fall back to Ollama's call (scorer=ollama_fallback) and count it
# 2026-06-23: switched Gemini → Groq llama-4-scout (n=345 bake-off: edge +5.2%, SIG).
# 2026-06-24: switched llama-4-scout → MISTRAL-LARGE — Groq deprecated llama-4-scout. mistral-large
#   was the validated #2 in the same bake-off (edge +4.7%, boot CI [+0.5,+10.7], SIG) and lives on
#   MISTRAL (not Groq), so it's immune to Groq model churn. ~2.7s latency — fine at ~10-30 gate calls/day.
#   The GEMINI_* names are kept (imported across bot.py) but now drive the Mistral endpoint;
#   override any of them with the CONFIRM_GATE_* env vars to repoint elsewhere.
HYBRID_SCORER_ENABLED        = os.environ.get("HYBRID_SCORER_ENABLED", "1") == "1"
GEMINI_BASE_URL              = os.environ.get("CONFIRM_GATE_BASE_URL", os.environ.get("MISTRAL_BASE_URL", ""))
GEMINI_KEY                   = os.environ.get("CONFIRM_GATE_KEY", os.environ.get("MISTRAL_API_KEY", ""))
GEMINI_MODEL                 = os.environ.get("CONFIRM_GATE_MODEL", "mistral-large-latest")
# How long process_signal waits for the gate before falling back/skipping. Bumped 12→20s 2026-06-26:
# mistral-large free-tier latency is server-side and variable (idle-machine test: median 24s, p90 grew
# 7.7→10.4s over 2 days), so a 12s cap was timing out the slow tail and — now that the downtrend bypass
# fails closed — needlessly skipping trades. 20s catches most of the tail; the entry delay is negligible
# for a multi-day hold. Real fix is moving the gate to a fast Groq production model (pending backtest).
GATE_WAIT_TIMEOUT_SECS       = int(os.environ.get("GATE_WAIT_TIMEOUT_SECS", "20"))
GEMINI_DAILY_CAP             = int(os.environ.get("CONFIRM_GATE_DAILY_CAP", "1000"))  # gate volume ~10-30/day — well under any free-tier cap
GEMINI_CONFIRM_MIN_MAGNITUDE = 0.50   # gate must independently see ≥ this BULLISH magnitude to confirm. Raised
                                      # 0.40→0.50 (2026-06-24) for mistral-large: its mags run higher (median 0.65);
                                      # threshold sweep on n=344 showed 0.50 lifts edge +2.5%→+2.8%, C-hit 28%→30%.
GEMINI_VETO_SHADOW_ENABLED   = True   # paper-track the picks the gate vetoes → forward P&L on its vetoes
GATE_LABEL                   = os.environ.get("CONFIRM_GATE_LABEL", "mistral-large")  # honest name in live logs
GATE_LATCH_ON_RATELIMIT      = os.environ.get("GATE_LATCH_ON_RATELIMIT", "0") == "1"  # Mistral 429 = transient RPM, not a daily wall → don't latch off

# ═══════════════════════════════════════════════════════════════════════════════
# SIGNAL THRESHOLDS
# ═══════════════════════════════════════════════════════════════════════════════

# Don't trade on any event scoring below this magnitude regardless of confidence.
# 0.0 = noise/irrelevant  0.3 = routine  0.5 = meaningful catalyst  0.7+ = high-impact
MIN_MAGNITUDE    = 0.35

# Confidence floor scales with magnitude: weaker catalyst -> higher conviction required.
# Formula: required_confidence = BASE_CONFIDENCE + (1 - magnitude) * CONFIDENCE_SLOPE
# At defaults:
#   magnitude 1.0 -> need 0.55 confidence
#   magnitude 0.7 -> need 0.63 confidence
#   magnitude 0.5 -> need 0.68 confidence
#   magnitude 0.35 (gate floor) -> need 0.71 confidence
BASE_CONFIDENCE  = 0.70   # cross-regime tuned 2026-06-02 (was 0.55) — stricter gate
CONFIDENCE_SLOPE = 0.25

# Minimum seconds between trades on the same underlying symbol.
COOLDOWN_SECS    = 300

# ── Stock-leg selectivity gate (added 2026-06-13, RECALIBRATED 2026-06-14) ────
# History: on the OLD two-pass scorer the always-on `stock` leg flooded with junk
# (~$4/tr, Sharpe 0.85) and conf≥0.85 was a real CLIFF that ~tripled per-trade
# quality. We shipped conf≥0.85 on that basis. BUT the recalibration sweep on the
# LIVE unified_v1 score distribution (stock_selectivity_sweep.py UNI_PROMPT=unified_v1,
# cap-aware, sized like live, bootstrap + regime) found NO cliff: quality is flat at
# ~$10/tr Sharpe ~1.8 from the dynamic floor (n=712) all the way to conf≥0.85 (n=165)
# — tightening to 0.85 just discards ~75% of EQUALLY-clean trades (sweep verdict:
# "marginal / not robust"). unified_v1's magnitude already self-rejects analyst noise
# upstream, so the loose set is already clean. → RELAXED to 0.0 so the existing
# dynamic confidence floor (BASE_CONFIDENCE + (1-mag)*CONFIDENCE_SLOPE, ~0.70-0.71)
# governs the stock leg and feeds the 30-slot cap with the full clean signal flow.
# Still strategy=="stock" ONLY (NOT pead's earnings gate, NOT shared MIN_MAGNITUDE).
# Re-tighten only if forward data shows the loose flood underperforms (conf≥0.90 was
# the only gate with a per-trade bump, ~$15-17/tr, but on ~65 trades/180d it starves
# the cap and the gain wasn't robust).
STOCK_MIN_CONFIDENCE = 0.0
STOCK_MIN_MAGNITUDE  = 0.0

# ═══════════════════════════════════════════════════════════════════════════════
# POSITION SIZING
# ═══════════════════════════════════════════════════════════════════════════════

# Effective size = MAX_*_USD * magnitude * confidence
# Both must be high to deploy full capital.
MAX_POSITION_USD  = 1_000   # maximum options premium per trade

# Halt all new trades for the session if cumulative realised losses hit this.
DAILY_LOSS_LIMIT  = 2_000

# ── Position exposure cap ──────────────────────────────────────────────────────
# Hard cap on monitored positions. Without this the bot stacked 98 positions in
# one day (stock+pead+news_call all fire per signal = 2-3 entries per ticker),
# deploying $93k of a $100k account and running to $1.5k cash. Existing positions
# are NOT closed — cap only blocks NEW entries.
# RAISED 30→45 on 2026-06-15: the 30-cap was jammed by the legacy stock book and
# blocked news_call from EVER trading live (571 stock + ~equal news_call skips in one
# day; news_call never fired). With the stock leg now SHADOW-ONLY (no real refills) the
# real consumers are just news_call/lotto/pead, so 45 gives news_call immediate room for
# its first live test without waiting for the Jun-12 batch to time-stop Wed. Capital-safe:
# ~45 × ~$500 ≈ $22.5k vs $81k cash. Revisit if deployed capital or concentration grows.
MAX_OPEN_POSITIONS  = 45
# PEAD gets its OWN reserved sub-cap (does NOT count against MAX_OPEN_POSITIONS), so the
# news_call flood can never starve it again — the root cause of pead firing once ever:
# the real `stock` leg hogged 33 slots in a day, dropping CASY/CBRL/ORCL pead signals at
# the cap. Sized to guarantee ≥2 new pead entries/day for every day one can stay open
# (PEAD_MAX_HOLD_DAYS), so a full holding period of entries always fits. Defined below the
# pead config (needs PEAD_MAX_HOLD_DAYS).

# ── Strategy on/off switches ───────────────────────────────────────────────────
# news_call: standard ATM (Δ0.50) calls on bullish signals. Disabled 2026-06-05 after
# 38 live trades showed 13% win / -$6,601 — realistic option costs ate the edge at the
# loose gate on the OLD two-pass scorer.
# RE-ENABLED 2026-06-14 at the unified_v1 SWEET-SPOT gate. news_call_sweep_unified.py
# (Δ0.50, DTE17, tiered_trail, NET of realistic spreads) showed the unified_v1 prompt's
# better name-selection makes news_call cost-survivable: at mag≥0.75 & conf≥dynamic-floor,
# ~333 tr, +$35-39K net, $105/tr, Sharpe 1.76, bootstrap CI>0 (P>0=98%), NOT one-trade-
# driven (drop biggest → still +$24K). Chose mag≥0.75 over the looser gate to (a) take only
# strong-catalyst signals and (b) limit pressure on the shared 30-slot cap (the stock leg
# already fires on every bullish signal). CAVEATS (why gate stays conservative): bull-meltup
# window only (long calls = leveraged beta; gated to uptrends by the regime filter), 29% win
# (tail-driven), 2022-bear test inconclusive (n=3). Forward-watch realized P&L vs the +$105/tr
# backtest; re-disable or tighten if live underperforms.
# ── stock leg: SHADOW-ONLY 2026-06-15 ─────────────────────────────────────────
# The always-on stock leg was demoted to PAPER/shadow on 2026-06-15. Live evidence
# (stock_vs_index_live.py): over June 9-15 the real stock picks returned -1.95% on
# ~$31k deployed while the SAME money in SPY made +0.88% / QQQ +2.02%, beating the
# index on only ~1/3 of trades — consistent with the long-running backtest finding
# that the news name-selection beats RANDOM but not the INDEX in a bull regime. So we
# STOP committing real capital + cap slots to stock (also frees slots for news_call,
# which the 30-cap was starving), but keep SHADOWING every stock signal (no orders, no
# capital) to gather forward unified_v1 data. Existing real stock positions ride out
# via their stops/holds. Re-enable (STOCK_ENABLED=True) only if the shadow stock leg
# proves it beats the index over a fuller unified_v1 window.
STOCK_ENABLED        = False   # real stock entries OFF (shadow instead)
STOCK_SHADOW_ENABLED = True    # paper-track every stock signal for forward data

NEWS_CALL_ENABLED   = True
# news_call-specific MAGNITUDE floor (the sweet-spot gate). Applied ON TOP of the shared
# dynamic confidence floor (BASE_CONFIDENCE + (1-mag)*CONFIDENCE_SLOPE), which at mag≥0.75
# already requires conf≥~0.76 — so this magnitude floor IS the binding selectivity lever.
# Weaker (mag<0.75) bullish signals fall through to the news_call shadow instead of trading.
NEWS_CALL_MIN_MAGNITUDE = 0.75
# Shadow news_call: log the trade news_call WOULD make on signals it does NOT trade (paper,
# NO orders, NO capital) — forward data + contract-selection samples without risk.
# Still useful now that news_call is enabled: it shadows the sub-0.75-magnitude signals.
NEWS_CALL_SHADOW_ENABLED = True
# news_call option geometry (added 2026-06-14) — news_call-SPECIFIC, threaded through
# execute_equity_trade for strategy=="news_call" only (generic defaults DTE14-21/Δ0.50 unchanged
# for other equity-option callers). news_call_geom_sweep.py swept Δ×DTE at the sweet-spot gate,
# NET of realistic spreads: a clean MONOTONIC gradient — shorter DTE wins at every delta, lower
# delta wins at short DTE. Corner Δ0.40/DTE10 was best AND robust ($76K net, $188/tr, Sharpe 2.13,
# bootstrap CI [$23K,$135K] P>0=100%, drop-biggest still +$60K, broad 35-quadrupler tail). It's a
# cheap short-dated slightly-OTM convex call. CAVEAT: this is the MAX-convexity / most-bull-beta /
# 26%-win profile (flattered by the melt-up window; bear guard n=11 ~break-even, inconclusive) and
# it OVERLAPS the lotto leg on signals that are also conf≥0.85 → such names get 2 short-dated convex
# legs. Watch concentration + realized P&L forward; revert toward Δ0.50/DTE17 if it underperforms.
NEWS_CALL_DTE_MIN      = 7      # ~DTE10 window (brackets the sweep optimum so weeklies are pickable)
NEWS_CALL_DTE_MAX      = 13
NEWS_CALL_TARGET_DELTA = 0.40   # slightly OTM, convex
# news_call-specific budget guard (added 2026-06-16). The generic MAX_CONTRACT_BUDGET_MULT (3.0)
# let news_call deploy up to ~3× its budget on a single high-premium near-money contract (a $235
# stock's Δ0.40 call is ~$8 = ~$800/contract; qty=1 can't go smaller). Tightened to 1.5× so a
# single contract slightly over budget is OK (whole-contract rounding) but egregious oversize is
# skipped — same intent as lotto's hard cap, just less strict (news_call must still trade pricey
# AI names). Threaded through execute_equity_trade → place_option_trade(budget_mult=) for news_call.
NEWS_CALL_BUDGET_MULT  = 1.5
# Per-ticker daily cap (added 2026-06-16): max news_call entries per ticker per UTC day. On busy
# news days one catalyst gets republished by multiple outlets hours apart (QCOM bought 3× on
# 2026-06-16 — 2 were the SAME Tenstorrent rumor 2h apart; the 5-min cooldown can't catch that).
# 1 = one news_call position per ticker per day (accepts missing a rare genuinely-distinct 2nd
# same-day catalyst, in exchange for never over-betting one name on echoed news).
NEWS_CALL_MAX_PER_TICKER_DAY = 1

# ═══════════════════════════════════════════════════════════════════════════════
# OPTIONS PARAMETERS
# ═══════════════════════════════════════════════════════════════════════════════

# Expiry window — cross-regime tuned 2026-06-02 (was 28/45). Short-dated 14-21
# DTE won on real Yahoo data across both regimes (holdout-validated).
MIN_DAYS_TO_EXPIRY = 14
MAX_DAYS_TO_EXPIRY = 21

# Target delta for contract selection — tuned to 0.50 (was 0.40); 0.60 was the
# bull-only optimum but fragile in bears, so 0.50 is the cross-regime-robust pick.
TARGET_DELTA       = 0.50

# ── qqq_macro option geometry (added 2026-06-13) ──────────────────────────────
# The qqq_macro strategy (macro-bullish signal, no specific ticker → QQQ call)
# previously inherited the generic news_call defaults (DTE 14-21, Δ0.50). Give it
# its own shorter-DTE / higher-delta geometry (~DTE10 / Δ0.60): QQQ is a deep,
# liquid index with tight spreads, so a higher-delta, shorter-dated call tracks the
# macro move more directly with less theta drag than a 14-21 DTE Δ0.50 call. Threaded
# through execute_equity_trade for strategy=="qqq_macro" ONLY (news_call/stock
# defaults above are unchanged). DTE10/Δ0.60 was the sweep optimum; 7-13 window
# brackets it so contract-selection still has expiries to choose from.
QQQ_MACRO_DTE_MIN      = 7
QQQ_MACRO_DTE_MAX      = 13
QQQ_MACRO_TARGET_DELTA = 0.60

# qqq_macro on/off. DISABLED 2026-06-14: on unified_v1 the macro-bullish→QQQ-call leg has no
# tradeable edge — net −$2,336 (frictionless ≈ $0 = pure spread cost), and a 25-cell Δ×DTE
# geometry sweep found NO robustly-positive window (positives were melt-up beta-lottery, CI
# spans 0). The QQQ-stock alternative was only +$184 (trivial). Geometry params above are kept
# dormant in case we revisit. When False, process_signal logs+skips no-ticker macro signals.
QQQ_MACRO_ENABLED      = False

# ── Regime filter ─────────────────────────────────────────────────────────────
# Only open new long-call positions when the market is in an uptrend (index above
# its N-day SMA). Backtest (regime_filter.py): the 200-day gate keeps ~91% of
# bull-market profit while cutting the 2022 bear-market loss ~53%. Long calls are
# leveraged long-beta, so this sits out sustained downturns.
REGIME_FILTER_ENABLED = True
REGIME_INDEX          = "SPY"
REGIME_MA_DAYS        = 200
# Short-term momentum brake (added 2026-06-17). The 200d gate is BLIND inside a bull — SPY can be
# +7.7% above its 200d yet falling for days (as it was 2026-06-16/17 while the bot bought calls
# into the dip). regime_gate_sweep.py found the 200d does NOTHING in the bull window (it's always
# above), so ALL added value comes from a short-term condition. The winner: also require SPY today
# ≥ SPY this many TRADING days ago. On the bull window this kept 84% of news_call P&L while LIFTING
# Sharpe 2.20→2.90 and cutting maxDD -$12.2k→-$9.4k (sheds pullback losers, keeps winners). Trade-off:
# ~16% less upside, ~half the trades, some whipsaw. 0 = momentum brake off.
REGIME_MOMENTUM_DAYS  = 3

# Chop brake (added 2026-07-16). The Jun–Jul 2026 live window exposed the gate's blind spot: SPY
# finished +1.8% (never near its 200d, momentum brake skipped only 7% of signals) yet 14/26 sessions
# closed down and news_call lost every week — chop passes a LEVEL+SIGN gate but kills a 3-day call
# trade that needs FOLLOW-THROUGH. Block new longs when ≥ this fraction of the last
# REGIME_DOWNDAY_WINDOW SPY sessions closed down. regime_gate_chop_sweep.py (3 tapes): adding
# dd10<60% to the live gate kept 98% of bull P&L at the best Sharpe tested (2.65→2.91), preserved
# the full 2022-bear block, and flipped the live chop tape −$10.9k→+$3.9k (skip 73%). ⚠ The
# threshold is knife-edged on the one 4-week chop sample (65% never fires, 50% blocks all) —
# regime_block_shadow tracks what it blocks, so watch the forward evidence. 0 = chop brake off.
REGIME_DOWNDAY_WINDOW      = int(os.environ.get("REGIME_DOWNDAY_WINDOW", "10"))
REGIME_DOWNDAY_MAX_DENSITY = float(os.environ.get("REGIME_DOWNDAY_MAX_DENSITY", "0.60"))

# High-conviction bypass (added 2026-06-23). The regime filter blocks ALL long-beta on down days,
# but that also blocks strong IDIOSYNCRATIC catalysts (relisting/M&A/guidance) whose alpha is largely
# market-independent. regime_bypass_option_pnl.py (real news_call option P&L, BS+IV+spreads+exits):
# in down-regime, mag≥0.85 catalysts MADE money — recent pullbacks +$274/trade, and even the full
# 2022 bear stayed positive (+$29/trade, never a net loss) while the 0.75–0.85 tier was flat. So
# bullish signals at/above this magnitude trade even in a downtrend; the llama-4-scout confirm/veto
# gate (downstream) still vets them. Set ≥ 1.01 to disable the bypass entirely.
# DISABLED 2026-07-16: live forward data contradicted the backtest — bypass trades went 1/13 (8% win,
# −$2.0k) in the Jun–Jul chop while regime_block_shadow showed the blocked trades would have LOST
# −$2.2k (the gate was right; the bypass was overriding it). Magnitude is uncalibrated (loser mag ≈
# winner mag), so mag≥0.85 isn't the "market-independent alpha" marker the backtest made it look like.
# Re-enable (env or lower default) only with fresh forward evidence.
REGIME_BYPASS_MIN_MAGNITUDE = float(os.environ.get("REGIME_BYPASS_MIN_MAGNITUDE", "1.01"))

# ── Backtest option-pricing model (Black-Scholes + IV crush) ──────────────────
# The historical option tape is too sparse to price options directly, so the
# backtest prices long calls with Black-Scholes off the dense stock bars.
RISK_FREE_RATE          = 0.04   # annualized; used in BS discounting
REALIZED_VOL_WINDOW     = 20     # trading days of stock history → base IV
# News events inflate IV at entry; it then decays back to base ("IV crush").
# Calibrated 2026-06-01 against 109 real option prints (calibrate_iv.py):
# median real_iv/base_iv = 1.10 (mean 1.13, p25–p75 0.88–1.36).
NEWS_IV_MULTIPLIER      = 1.10   # entry IV = base_iv × this
IV_CRUSH_HALFLIFE_DAYS  = 3.0    # half-life of the elevated-IV decay (trading days)
IV_FLOOR                = 0.15   # clamp base IV so thin/quiet names aren't underpriced
IV_CAP                  = 2.50   # clamp to reject absurd realized-vol spikes

# ── Data-sanity / sizing guards (reject backtest artifacts) ───────────────────
# The free IEX feed has bad historical bars (e.g. EDBL priced at $417,000 in
# 2022). These guards stop a single garbage bar from producing a fake huge trade.
MAX_SANE_STOCK_PRICE     = 10_000  # skip entries above this (no optionable single
                                   # name trades here; > this = bad data)
MAX_CONTRACT_BUDGET_MULT = 3.0     # skip if ONE contract costs more than this ×
                                   # the position budget (prevents the qty=max(1,…)
                                   # rule from forcing a wildly oversized position)

# ── Live price cross-check (data-quality guard) ───────────────────────────────
# Before entering a trade, validate Alpaca's stock price against a SECOND source
# (Yahoo). Skip the trade if they disagree by more than the tolerance — catches
# gross bad quotes (e.g. SNDK showing $1,568 vs reality). Tolerance is deliberately
# loose so genuine high-magnitude NEWS moves (the bot's bread and butter) aren't
# blocked by a slightly-delayed second feed. Fail-OPEN if Yahoo is unavailable — a
# Yahoo outage must never halt trading.
PRICE_CROSSCHECK_ENABLED = True
PRICE_CROSSCHECK_TOL      = 0.25   # max |alpaca/yahoo − 1| before skipping the trade

# ═══════════════════════════════════════════════════════════════════════════════
# GUARDRAILS
# ═══════════════════════════════════════════════════════════════════════════════

# Skip any stock below this price -- avoids penny stocks with wide spreads.
MIN_STOCK_PRICE    = 5.00

# Minimum market cap in billions. ETFs bypass this check.
MIN_MARKET_CAP_B   = 2.0

# Skip option contracts where bid/ask spread exceeds this % of mid-price.
# Prevents entering a position that is immediately deep underwater.
MAX_SPREAD_PCT     = 0.20   # 20%

# Monitoring quote sanity: in the trailing-stop monitor (and post-close capture), reject an option
# quote whose bid/ask spread exceeds this — a crossed/garbage two-sided quote yields a bad mid that
# corrupts the peak (→ ratchet stop fires falsely) and the position_paths data. 2026-06-26: a $1.96
# print on a ~$0.56 BA call (spread ~195%) set a false +250% peak and force-closed the trade. Well
# above the 20% entry cap and normal intraday widening, so only garbage is dropped.
MAX_MONITOR_SPREAD_PCT = float(os.environ.get("MAX_MONITOR_SPREAD_PCT", "0.60"))   # 60%

# Skip contracts with fewer than this many open contracts -- proxy for liquidity.
MIN_OPEN_INTEREST  = 100

# ═══════════════════════════════════════════════════════════════════════════════
# TRAILING STOP
# ═══════════════════════════════════════════════════════════════════════════════

# Baseline / fallback trailing stop (also the tightest tier). Tuned to 0.10.
TRAILING_STOP_PCT  = 0.10

# Profit-tiered trailing stop ("tier_runner_wide", chosen 2026-06-02). The trail
# is WIDE early (give a developing move room → don't exit prematurely) and
# TIGHTENS once it's a big winner (lock in the runner). Format: (gain_threshold,
# trail_pct) — first tier whose threshold the current gain is BELOW applies.
#   gain < +100%  → 30% trail   (let it run)
#   gain < +300%  → 20% trail
#   gain ≥ +300%  → 13% trail   (protect the big winner)
EXIT_TIERS = [(1.00, 0.30), (3.00, 0.20), (float("inf"), 0.13)]

# news_call exit. Flat 40% (NEWS_CALL_TRAIL_PCT) was the 2026-06-14 daily-sweep choice and is kept as
# the fallback. RATCHET deployed 2026-06-26 (NEWS_CALL_EXIT_TIERS): a peak-gain tiered trail — 40%
# until +15%, 25% until +35%, 15% above. exit_intraday_sweep.py (hourly bars, de-biased) showed it
# beats flat-40% on win rate (37% vs 30%), P&L and the +15%→red round-trip: start wide so early noise
# doesn't stop you out (preserves win rate), tighten after a gain so faded winners exit green. Applied
# via long_trail_for(), which receives the PEAK gain → monotonic ratchet (stop only ever rises).
# RE-EVALUATE on the real position_paths.csv 30s quotes (incl. post-close) once accumulated. Set
# NEWS_CALL_EXIT_TIERS = [] to revert to the flat NEWS_CALL_TRAIL_PCT.
NEWS_CALL_TRAIL_PCT  = 0.40
NEWS_CALL_EXIT_TIERS = [(0.15, 0.40), (0.35, 0.25), (float("inf"), 0.15)]

# How often the monitor loop checks open positions (seconds).
MONITOR_INTERVAL   = 30

# ═══════════════════════════════════════════════════════════════════════════════
# TRADEABLE UNIVERSES
# ═══════════════════════════════════════════════════════════════════════════════

# ETFs that are eligible for options trading (bypass market cap check).
TRADEABLE_ETFS   = {"SPY", "QQQ", "IWM", "GLD", "TLT"}

# ═══════════════════════════════════════════════════════════════════════════════
# NEWS SOURCES
# ═══════════════════════════════════════════════════════════════════════════════

SEC_RSS_URL       = (
    "https://www.sec.gov/cgi-bin/browse-edgar"
    "?action=getcurrent&type=8-K&dateb=&owner=include"
    "&count=10&search_text=&output=atom"
)
# Near-real-time EDGAR. SEC offers no FREE push (the real-time PDS feed is paid/
# institutional), so the free equivalent is fast polling of the "getcurrent" latest-
# filings feed. 3s = ~1.5s avg detection latency (was ~60s avg at 120s), and at
# 0.33 req/s is far under SEC's 10 req/s fair-access limit. SEC requires a declared
# User-Agent with contact info — SET SEC_USER_AGENT in .env to your real name+email
# (the default is a placeholder; a real contact avoids throttling/blocks).
SEC_POLL_SECS     = int(os.environ.get("SEC_POLL_SECS", "3"))
SEC_USER_AGENT    = os.environ.get("SEC_USER_AGENT", "Trader Bot research admin@example.com")

NEWSAPI_URL       = "https://newsapi.org/v2/top-headlines"
NEWSAPI_POLL_SECS = 180   # poll NewsAPI every 3 minutes
NEWSAPI_PARAMS    = {"category": "business", "language": "en", "pageSize": 20}

# ── PEAD trailing stop (stock, tiered to lock in profit) ─────────────────────
# Wide early (give earnings drift room), then tightens as gains grow.
# Breakeven lock: once gain ≥ PEAD_BREAKEVEN_LOCK, stop can never fall below
# entry (prevents giving back all profit after a big move).
#
#   gain < 15%  → 20% trail   (wide, let the drift develop)
#   gain < 30%  → 12% trail   (tighten once solidly up)
#   gain ≥ 30%  →  7% trail   (protect the big winner)
# Tuned 2026-06-03 via pead_tier_tune.py (378-combo grid, 80/20 holdout, n=475 stock trades).
# 2026-06-14: exit_sweep_unified (on unified_v1) found a FLAT 20% trail + 10d hold beats the
# tiered config on Sharpe (2.5→3.95) — collapsed the tiers to a single flat 20% trail. The
# old tiered values are preserved in the comment above if we want to revert. (pead_trail_pct
# walks these tiers; a single (inf, 0.20) tier = flat 20%.)
PEAD_EXIT_TIERS      = [(float("inf"), 0.20)]   # flat 20% trail (was [(0.20,0.25),(0.35,0.12),(inf,0.10)])
PEAD_BREAKEVEN_LOCK  = 0.0    # disabled — hurts by cutting positions before recovery

# ── Bear-short trailing stop ───────────────────────────────────────────────────
# Tuned 2026-06-03 via pead_bear.py (hold=45, trail=0.15 won; 69% win, Sharpe 6.36).
# Stop fires when price RISES trail% above the trough (lowest seen since entry).
BEAR_SHORT_TRAIL_PCT = 0.15
# RETIRED 2026-06-18: legs_unified_backtest on the live unified_v1 prompt showed bear_short's
# model picks barely beat RANDOM shorts (+$607 vs +$268 bull, +$2,049 vs +$1,746 bear) — ~85% of
# the bear-window P&L is just beta from shorting a falling tape, not selection skill. Plus the
# bullish-only prompt produces almost no clean bearish signals (5% pass the gate w/ a ticker).
# Net-negligible edge + short-side risk/borrow complexity → disabled. (pairs left dormant; it's
# the only long-derived leg positive in the 2022-bear window, but needs a dedicated bear scorer.)
BEAR_SHORT_ENABLED   = False

# ── Per-strategy max hold days (live bot) ─────────────────────────────────────
# Multi-horizon signal_lab analysis (2026-06-05) showed the news signal is a
# 1-DAY phenomenon: IC/lift are 2-3× higher at fwd=1d than at 3d/5d. Holding
# news-driven stock positions for weeks adds ~29 days of pure random-walk risk
# on top of a 1-day genuine edge.
# Rules: whichever fires first — the trailing stop OR the time limit — exits.
# pead is deliberately longer: earnings drift is a multi-week phenomenon.
# Pairs/bear_short not capped (they're regime-independent and less time-sensitive).
# 0 = no time cap for that strategy.
NEWS_STOCK_MAX_HOLD_DAYS = 5   # 2026-06-14: exit_sweep_unified re-swept on unified_v1 — 5d ~DOUBLES
                              # stock P&L vs 2d ($3k→$6k) and lifts Sharpe 2.0→3.1 (bootstrap CI
                              # excludes 0). Stock has no theta, so it needs room for the drift; the
                              # old 2d (from the 1-day-IC analysis) cut winners too early. Was 2.
PEAD_MAX_HOLD_DAYS       = 10  # 2026-06-14: exit_sweep_unified — 45d was too long; 10d + 20% flat
                              # trail lifts Sharpe 2.5→3.95 at similar $ (drift plays out faster than
                              # 45d). Was 45.

# PEAD-as-OPTIONS forward shadow (2026-06-30). pead_options_test: PEAD's slow ~10d drift LOSES on ATM/
# short-dated calls (theta), but is POSITIVE on ITM + longer-DTE (Δ0.70/45d → median +12%, 57% win — a
# broad-based edge on the robust median/win view, not a tail mirage). Paper-track it on REAL quotes to
# confirm before ever trading it live (the live PEAD leg stays STOCK). Held PEAD_MAX_HOLD_DAYS (10d) with
# a 40% premium trail; DTE well clear of the hold so it can't expire mid-drift.
PEAD_OPTION_SHADOW_ENABLED = True
PEAD_OPTION_DELTA          = 0.70    # ITM — stock-like, low theta (the geometry the sweep favored)
PEAD_OPTION_DTE_MIN        = 35      # ~45-DTE target
PEAD_OPTION_DTE_MAX        = 55
PEAD_OPTION_TRAIL          = 0.40    # premium trail (options ~5× more volatile than the 20% stock trail)

# ── PEAD reserved sub-cap + shadow overflow (see MAX_OPEN_POSITIONS note) ───────
# pead is the highest-Sharpe leg (4.2 on unified_v1) but fired once ever — starved by the
# stock-leg flood at the shared cap. Give it dedicated slots: ≥2 entries/day × the full hold
# window, so a steady earnings cadence never overflows. Reserved (not counted vs the main cap).
PEAD_MAX_POSITIONS       = 2 * PEAD_MAX_HOLD_DAYS   # = 20 reserved pead slots
# If the reserved sub-cap IS hit, don't drop the signal — paper-track it (base_strategy="pead",
# MTM'd by the shadow monitor's stock branch with pead's own tiered-trail + 10d exit) so we keep
# forward data and can see whether 20 slots was ever the binding constraint.
PEAD_SHADOW_ENABLED      = True
NEWS_CALL_MAX_HOLD_DAYS  = 3   # 2026-06-14: exit_sweep_unified — the Δ0.40/DTE10 calls capture the
                              # move in ~3 days then bleed theta/IV-crush on non-movers. 3d cap (+ 40%
                              # flat trail, see NEWS_CALL_TRAIL_PCT) = +$41k / Sharpe 2.12→2.53 vs the
                              # old no-cap tiered exit. Was 0 (no time cap).
LOTTO_MAX_HOLD_DAYS      = 3   # 2026-06-10: re-swept for the short-DTE window (8-21) — 3d beats
                              # 7d (+10% P&L/+0.17 Sharpe; jackknife: edge is tail-INDEPENDENT,
                              # all from cutting non-movers' theta sooner since winners exit at
                              # the 3× cap before day 3 anyway). Was 7 (tuned for the old ~21 DTE).

# Time-stop exits fire NEAR THE CLOSE on the final hold day (not intraday, and not
# carried to the next morning's open). overnight_gap_test.py: the close→next-open gap
# is ~zero-mean (slightly negative) with ~1.8% nightly stdev — uncompensated risk. So
# we hold through the final day (trailing-stop still active) and exit in this window.
TIME_STOP_EOD_WINDOW_MIN = 15  # minutes before the regular close to run the time-stop exit

# ── Market-neutral L/S pairs ───────────────────────────────────────────────────
# On any day with BOTH a bullish and a bearish signal, go long the top-scored
# bull name and short the top-scored bear name in equal dollar size. The market
# beta largely cancels → near-market-neutral (SPY corr +0.16 in backtest), so
# this is the ONE strategy that works in both regimes WITHOUT a regime gate.
# Tuned 2026-06-03 (pairs_tune.py): long_trail 0.25, short_trail 0.15, hold 45d.
# Cross-regime: bull Sharpe ~2.7, 2022-bear Sharpe ~4.2 (vs random L/S 1.5).
PAIRS_ENABLED      = True
PAIRS_LONG_TRAIL   = 0.25   # flat trailing stop on the long leg (peak − 25%)
PAIRS_SHORT_TRAIL  = 0.15   # trailing stop on the short leg (trough + 15%)
PAIRS_MAX_HOLD     = 45     # backtest hold cap (days). NOTE: live bot has no time
                            # cap yet — documented divergence (same as other legs).

# ── Lotto: cheap OTM calls on the strongest bullish signals ────────────────────
# A high-conviction subcategory of the call strategy: when the LLM is very bullish
# AND a big swing looks likely, buy a small, cheap out-of-the-money call for a
# convex payoff. Backtested 2026-06-03 (lotto_backtest.py): bull Sharpe ~2.75,
# +$15K, 9 trades 4x+, only ~25% eaten by spreads (much less than ATM calls).
# Loses modestly in 2022 (regime gate + small size contain it). Long-beta →
# regime-gated like the other long strategies.
LOTTO_ENABLED        = True
LOTTO_MIN_MAGNITUDE  = 0.70   # only the strongest catalysts
LOTTO_MIN_CONFIDENCE = 0.85   # and high directional certainty
LOTTO_TARGET_DELTA   = 0.25   # OTM (sweet spot in backtest); cheap + convex
LOTTO_POSITION_USD   = 250    # small — it's a lottery ticket
# Short-DTE window (2026-06-09): backtest sweep at the live 7-day hold showed lotto
# edge is MONOTONICALLY better at shorter DTE (Δ0.25 bull-meltup: 9-DTE Sharpe 2.84 /
# $13,193 vs old 21-DTE 2.22 / $8,184 — cheaper near-dated call = more convexity per
# $250). This ALSO fixes the "no contracts found" gap: non-weekly mid/large-caps
# (IFF/WST/CP/SJM/INCY…) only list monthlies (~9 & ~38 DTE) and the old 18-28 window
# fell in the dead zone between them. 8-21 catches the ~9-DTE near-monthly. Floor=8 is
# just above the 7-day hold so the option survives to exit; sort target ≈14. Liquidity
# guards (MAX_SPREAD_PCT, MIN_OPEN_INTEREST) still reject illiquid near-expiry picks.
# CAVEAT: backtest prices via BS w/ base spreads — real near-expiry friction is worse;
# watch realized fills via contract_selection.csv / closed_trades.csv.
LOTTO_MIN_DTE        = 8
LOTTO_MAX_DTE        = 21     # ~14 DTE target — short-dated convex bet
LOTTO_STRIKE_HI_MULT = 1.25   # allow strikes up to 25% OTM in the contract search
# Out-of-window SHADOW (2026-06-09): when a high-conviction lotto signal has NO contracts
# inside the 8-21 DTE window (truly optionless, OR nearest expiry is outside it — e.g. only
# a 30/38-DTE monthly), open a PAPER shadow on the closest-to-window contract instead of
# silently forfeiting the name. Captures forward data on what we'd otherwise miss → tells
# us whether to widen further. LOTTO_OOW_MAX_DTE bounds how far out we'll look.
LOTTO_OOW_SHADOW_ENABLED = True
LOTTO_OOW_MAX_DTE        = 120
# Hard profit cap (Tier-1-validated 2026-06-09): take FULL profit the first time a lotto
# option's premium reaches LOTTO_HARD_CAP_MULT × entry. Short-DTE convex winners round-trip
# violently, so the wide "let it run" trail gives back too much — a 3× cap locks the gain.
# Robust in backtest (jackknife wins at every level; the edge lives in the broad body of
# mid-winners, not the tail) and ASYMMETRIC-SAFE (can only exit a winner early, never adds a
# loss). 3× = the sweep sweet spot (beats 4×). lotto_cap_log.csv records each live cap exit's
# trigger-vs-fill to confirm the cap fills near 3× intraday (the daily-bar backtest couldn't).
LOTTO_HARD_CAP_ENABLED   = True
LOTTO_HARD_CAP_MULT      = 3.0
# Wide "let it run" trail (convex bets need room); tightens only on a big winner.
LOTTO_EXIT_TIERS     = [(1.0, 0.40), (3.0, 0.30), (float("inf"), 0.20)]

# ── Materiality shadow filter for LOTTO (forward A/B, 2026-06-08) ───────────────
# Stage-3 backtest finding: gating lotto entries on a llama3.2 "materiality" score
# (materiality_fewshot prompt; score = magnitude×confidence) lifted lotto P&L +11%,
# Sharpe 2.39→3.03, halved drawdown, kept 8/9 multibaggers — the ONE leg where the
# model's news-selection converts to dollars (convex payoff). To validate forward
# WITHOUT cutting trades, we run a SHADOW A/B: every lotto signal still trades, but
# we additionally score its materiality and log it (lotto_materiality.csv). Forward
# analysis splits realized lotto P&L by materiality ≥ MATERIALITY_GATE to confirm
# the backtested edge before making it a hard live gate.
MATERIALITY_SHADOW_ENABLED = True
MATERIALITY_GATE           = 0.15   # backtest sweet spot (≥0.15 → best Sharpe, kept the winners)
# PROMOTED 2026-06-10 from shadow A/B → LIVE lotto SELECTOR. lotto_pnl_by_scorer showed the
# materiality_fewshot score is the dominant lotto-selection lever (+$8,934 vs money-losing
# alternatives on the same candidates); the materiality A/B backtest had +11% P&L / Sharpe
# 2.39→3.03. When True, lotto fires ONLY if materiality ≥ MATERIALITY_GATE; signals that pass
# the old mag/conf gate but FAIL materiality are DEMOTED to shadow logging (paper, for the
# ongoing forward A/B). Cost: materiality is scored BEFORE the order (+~2.5s lotto latency).
# DISABLED 2026-06-13: the unified-prompt eval (unified_v1) showed the materiality 2nd pass is
# now VESTIGIAL — unified_v1's own magnitude self-rejects analyst noise, so the separate gate
# adds no independent filtering (the catalyst-gate sweep was flat/collinear across the full
# 6.2k pool). Dropping it removes the redundant ~2.5s 2nd LLM call per lotto. When False, bot.py
# skips the score_materiality call entirely. Re-enable only if forward data shows the single
# unified pass leaks false positives the 2nd opinion would have caught.
MATERIALITY_GATE_ENABLED   = False

# ── Shadow strategy router (logging-only) ──────────────────────────────────────
# The rule-based router maps each signal to ONE strategy (router_backtest.py):
#   bull+bear same day → pairs · bull+earnings → pead · bull → stock ·
#   bear → bear_short · macro-bull (no ticker) → qqq_macro.
# When enabled, the bot LOGS the router's pick per signal to router_decisions.csv
# but opens NO extra orders — the bot already trades every strategy, so we later
# attribute their realized P&L to the router's choices (router_eval.py) to test
# whether routing beats any single fixed strategy. Purely observational.
ROUTER_SHADOW_ENABLED = True

# ═══════════════════════════════════════════════════════════════════════════════
# BACKTEST-ONLY PARAMETERS
# ═══════════════════════════════════════════════════════════════════════════════

# Maximum days to hold a simulated position before forcing an exit.
MAX_HOLD_DAYS  = 30

# Stock price move windows (in trading days) used for signal quality evaluation.
MOVE_WINDOWS   = [1, 3, 5]