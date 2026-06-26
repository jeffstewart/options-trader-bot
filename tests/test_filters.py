"""Pre-score filters — the deterministic gates that drop weak/noise catalysts BEFORE scoring.
These are pure string matchers (no network), so they're cheap to lock down. Regressions here would
silently let losers (index-inclusion, analyst notes) back into the live trade path."""
import bot
import pytest


# ── _prescore_noise: macro/index noise dropped before any scoring call ────────────────────
INDEX_INCLUSION = [
    "Alphabet Selected To Join The Dow Jones Industrial Average",
    "Alphabet To Replace Verizon In Dow Jones Industrial Average",
    "Security National Financial To Join Russell 3000 Index",
    "Toast To Replace TopBuild Corp. In The S&P Midcap 400",
    "Ralliant To Replace Wolfspeed In S&P 600 On July 1",
    "Protagonist Therapeutics To Join S&P SmallCap 600",
    "Diginex Joins S&P Global BMI Index",
    "Lifecore Biomedical To Join Nasdaq Biotechnology Index",
]
MACRO_NOISE = [
    "Stock Futures Rise Ahead Of Jobs Report",
    "S&P Futures Slip As Yields Climb",
    "Closing Bell: Stocks Close Higher",
    "5 Stocks To Buy Now Before Earnings",
    "What's Going On With Tesla Stock Today",
]
REAL_CATALYSTS = [
    "Nvidia Beats Earnings, Raises Guidance",
    "Upstart Announces $600 Million Investment",
    "Broadcom Unveils Jalapeño AI Inference Chip",
    "Qualcomm To Exit FY26 At $6B In Automotive Revenue",
    "Pfizer's Drug Gets FDA Approval For New Indication",
]


@pytest.mark.parametrize("headline", INDEX_INCLUSION)
def test_prescore_drops_index_inclusion(headline):
    assert bot._prescore_noise(headline) is not None, f"should drop index headline: {headline!r}"


@pytest.mark.parametrize("headline", MACRO_NOISE)
def test_prescore_drops_macro_noise(headline):
    assert bot._prescore_noise(headline) is not None, f"should drop macro/noise: {headline!r}"


@pytest.mark.parametrize("headline", REAL_CATALYSTS)
def test_prescore_keeps_real_catalysts(headline):
    assert bot._prescore_noise(headline) is None, f"should NOT drop real catalyst: {headline!r}"


# ── _soft_catalyst_hit: over-scored loser catalyst types (gated, shadowed) ─────────────────
SOFT = [
    ("Analyst Raises Price Target On Apple To $300", ""),
    ("Goldman Upgraded Tesla To Buy", ""),
    ("Acme Stock Flashes A Golden Cross Breakout", ""),
    ("Microsoft Agreed To Acquire A Gaming Studio", ""),    # acquirer-side M&A
]
NOT_SOFT = [
    ("Acme Beats Q3 Earnings, Raises Full-Year Outlook", ""),
    ("FDA Approves New Drug From Biotech", ""),
]


@pytest.mark.parametrize("headline,body", SOFT)
def test_soft_catalyst_blocks_weak_types(headline, body):
    assert bot._soft_catalyst_hit(headline, body) is not None, f"should gate: {headline!r}"


@pytest.mark.parametrize("headline,body", NOT_SOFT)
def test_soft_catalyst_allows_real(headline, body):
    assert bot._soft_catalyst_hit(headline, body) is None, f"should allow: {headline!r}"


# ── Benzinga "returns-calculator" / performance-recap noise (regex, 2026-06-26) ───────────
RECAP_NOISE = [
    "Here's How Much $100 Invested In General Dynamics 5 Years Ago Would Be Worth Today",
    "$100 Invested In First Solar 5 Years Ago Would Be Worth This Much Today",
    "If You Invested $1000 In W.W. Grainger Stock 5 Years Ago, You Would Have This Much",
    "Qualcomm Has Outperformed The Market Over The Past 10 Years By 1.09% Annually",
]
# Adversarial: real capex / catalysts that look NEAR the recap patterns but must NOT be filtered.
RECAP_ADVERSARIAL_KEEP = [
    "Tesla Invested $1 Billion In Lithium Refining Capacity",   # real capex (invested-before-$, no "if you")
    "$100M Invested In New Fab By TSMC This Quarter",           # bare-$ guard: \d+\s+invested excludes "$100M"
    "Analysts Say Nvidia Could Be Worth $5 Trillion",           # "worth" but no "N years ago"
]


@pytest.mark.parametrize("headline", RECAP_NOISE)
def test_prescore_drops_recap_calculators(headline):
    assert bot._prescore_noise(headline) is not None, f"should drop recap: {headline!r}"


@pytest.mark.parametrize("headline", RECAP_ADVERSARIAL_KEEP)
def test_prescore_keeps_recap_near_misses(headline):
    assert bot._prescore_noise(headline) is None, f"should NOT drop: {headline!r}"
