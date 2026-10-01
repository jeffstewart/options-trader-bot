"""
test_filters.py — unit tests for v3 deterministic pre-score noise & soft-catalyst filters.
"""

import pytest
from v3 import filters


def test_normalize_headline():
    """Verify HTML entities and typographic quotes are normalized."""
    raw = "Apple&rsquo;s &ldquo;Next Big Thing&rdquo; &amp; Market &quot;Gains&quot;"
    norm = filters.normalize_headline(raw)
    assert norm == "Apple's \"Next Big Thing\" & Market \"Gains\""


@pytest.mark.parametrize("headline", [
    "Stock futures rise ahead of key inflation data",
    "Dow futures slip 100 points in pre-market trade",
    "S&P 500 futures point to lower open",
    "Closing bell: stocks close higher on tech rally",
    "10-year Treasury yield jumps to 4.25%",
])
def test_prescore_macro_wraps(headline):
    """Verify macro, index, and market wrap headlines are filtered out."""
    hit = filters.prescore_noise(headline)
    assert hit is not None


@pytest.mark.parametrize("headline", [
    "3 high-growth stocks to watch this week",
    "Top stocks to buy for long-term investors",
    "5 best stocks to consider in tech",
    "Stocks to avoid during market volatility",
])
def test_prescore_listicles(headline):
    """Verify listicles and stock roundups are dropped before scoring."""
    hit = filters.prescore_noise(headline)
    assert hit is not None


@pytest.mark.parametrize("headline", [
    "Why is Tesla stock surging today?",
    "Nvidia shares are soaring: what's driving the rally?",
    "Why Apple stock is jumping higher after hours",
    "Palantir shares climbing: here's why",
    "Why Intel shares are tumbling today",
    "Boeing stock is plunging: here's why",
    "Why Pfizer shares are falling after hours",
])
def test_prescore_reactive_recaps(headline):
    """Verify reactive post-move explanations are dropped."""
    hit = filters.prescore_noise(headline)
    assert hit is not None


@pytest.mark.parametrize("headline", [
    "If you had invested $1,000 in Amazon 10 years ago, here is how much you would have",
    "$10,000 invested in Microsoft in 2015 would be worth this today",
    "Apple has outperformed the S&P 500 over the past 5 years",
])
def test_prescore_historical_calculators(headline):
    """Verify hypothetical historical returns calculators are dropped."""
    hit = filters.prescore_noise(headline)
    assert hit is not None


@pytest.mark.parametrize("headline", [
    "Palantir to join the S&P 500 index this month",
    "Dell joins the S&P 500 in quarterly reconstitution",
    "Super Micro added to the Russell 2000 index",
    "CrowdStrike added to the S&P 500 index",
])
def test_prescore_index_inclusion(headline):
    """Verify index inclusion and reconstitution headlines are dropped."""
    hit = filters.prescore_noise(headline)
    assert hit is not None


@pytest.mark.parametrize("headline,body", [
    ("Morgan Stanley raises PT on AMD to $210", "Analyst reiterates Overweight rating"),
    ("Goldman Sachs downgrades Intel to Sell rating", "Lowers price target to $18"),
    ("JPMorgan initiates coverage on Arm with Outperform rating", "Sets $150 price target"),
    ("Bank of America reiterates Buy rating on Nvidia", "Their forecast remains bullish"),
])
def test_soft_catalyst_analyst_ratings(headline, body):
    """Verify analyst ratings and price target updates are flagged as soft catalysts."""
    hit = filters.soft_catalyst_hit(headline, body)
    assert hit is not None


@pytest.mark.parametrize("headline,body", [
    ("Tesla chart pattern flashes golden cross signal", "Moving average breakout indicated"),
    ("Bitcoin enters oversold territory on technical indicators", "Relative strength index dips"),
])
def test_soft_catalyst_technical_chatter(headline, body):
    """Verify technical chart indicators are flagged as soft catalysts."""
    hit = filters.soft_catalyst_hit(headline, body)
    assert hit is not None


@pytest.mark.parametrize("headline,body", [
    ("Broadcom in discussions to buy cyber startup for $2B", "Acquirer seeks to expand software unit"),
    ("Salesforce agreed to acquire data management firm", "Purchase price undisclosed"),
])
def test_soft_catalyst_acquirer_ma(headline, body):
    """Verify acquirer-side M&A deals are flagged as soft catalysts."""
    hit = filters.soft_catalyst_hit(headline, body)
    assert hit is not None


@pytest.mark.parametrize("headline,body", [
    ("Pfizer receives FDA approval for novel oncology therapy", "Phase 3 clinical trial met all primary endpoints with statistical significance."),
    ("Lockheed Martin awarded $4.2B Navy missile contract", "Defense department confirms multi-year procurement agreement."),
    ("Microsoft announces $10B infrastructure partnership with OpenAI", "Deal includes custom cloud hardware deployment starting next quarter."),
    ("Eli Lilly reports Q3 EPS beats estimates by 35% on record Mounjaro sales", "Raises full-year revenue guidance above Wall Street consensus."),
])
def test_genuine_catalysts_pass_cleanly(headline, body):
    """Verify genuine fundamental catalysts pass all pre-score noise and soft-catalyst filters."""
    noise_hit = filters.prescore_noise(headline)
    assert noise_hit is None, f"Expected no noise hit, but matched: {noise_hit}"

    soft_hit = filters.soft_catalyst_hit(headline, body)
    assert soft_hit is None, f"Expected no soft catalyst hit, but matched: {soft_hit}"

