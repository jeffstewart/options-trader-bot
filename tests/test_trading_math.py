"""Trading math — realized P&L, position sizing, hold caps. Pure functions; a regression here is a
real-money bug (wrong P&L sign on shorts, wrong sizing), so these guard the core arithmetic."""
import bot
import config
import pytest


# ── _realized_pnl: $ P&L by asset type ────────────────────────────────────────────────────
def test_realized_pnl_stock_long():
    # bought 5 @ 100, sold @ 110 → +$50
    assert bot._realized_pnl("stock", 100.0, 110.0, 5) == pytest.approx(50.0)


def test_realized_pnl_stock_short_profits_when_lower():
    # shorted 5 @ 100, covered @ 90 → +$50 (the sign that matters)
    assert bot._realized_pnl("stock_short", 100.0, 90.0, 5) == pytest.approx(50.0)
    # covered HIGHER → a loss
    assert bot._realized_pnl("stock_short", 100.0, 110.0, 5) == pytest.approx(-50.0)


def test_realized_pnl_option_uses_100_multiplier():
    # 1 call @ $2.00 → $3.00 = +$100 (100 shares/contract)
    assert bot._realized_pnl("option", 2.0, 3.0, 1) == pytest.approx(100.0)
    assert bot._realized_pnl("option", 2.0, 1.0, 2) == pytest.approx(-200.0)


# ── scale_position_usd: flat sizing while magnitude is uncalibrated ────────────────────────
def test_scale_position_flat_when_magnitude_sizing_off():
    if config.MAGNITUDE_SIZING_ENABLED:
        pytest.skip("magnitude sizing is enabled — flat-sizing test N/A")
    expected = 1000.0 * config.NONMAG_SIZE_FRAC
    assert bot.scale_position_usd(1000.0, 0.85, 0.82) == pytest.approx(expected)


def test_scale_position_independent_of_mag_conf_when_flat():
    if config.MAGNITUDE_SIZING_ENABLED:
        pytest.skip("magnitude sizing is enabled")
    # flat sizing → same $ regardless of mag/conf
    assert bot.scale_position_usd(1000.0, 0.10, 0.10) == bot.scale_position_usd(1000.0, 0.99, 0.99)


# ── max_hold_days_for: per-strategy time caps ─────────────────────────────────────────────
def test_max_hold_news_call_is_capped():
    assert bot.max_hold_days_for("news_call", "option") == config.NEWS_CALL_MAX_HOLD_DAYS
    assert bot.max_hold_days_for("news_call", "option") > 0


def test_max_hold_pairs_has_no_time_cap():
    # pairs / bear_short / qqq_macro: 0 = no time cap (trailing stop still fires)
    assert bot.max_hold_days_for("pairs", "stock") == 0


def test_max_hold_unknown_strategy_defaults_to_no_cap():
    assert bot.max_hold_days_for("nonexistent_strategy", "stock") == 0


# ── news_call RATCHET trail: tightens by peak-gain (deployed 2026-06-26) ──────────────────
def test_news_call_ratchet_tightens_with_peak_gain():
    # fed the PEAK gain by long_trail_for → monotonic ratchet. Default tiers 40/25/15 @ +15/+35%.
    assert bot.news_call_trail_pct(0.00) == 0.40
    assert bot.news_call_trail_pct(0.149) == 0.40
    assert bot.news_call_trail_pct(0.15) == 0.25
    assert bot.news_call_trail_pct(0.349) == 0.25
    assert bot.news_call_trail_pct(0.35) == 0.15
    assert bot.news_call_trail_pct(2.0) == 0.15


def test_news_call_ratchet_is_monotonic_non_increasing():
    gains = [i / 100 for i in range(0, 200, 5)]
    trails = [bot.news_call_trail_pct(g) for g in gains]
    assert all(b <= a for a, b in zip(trails, trails[1:])), "ratchet trail must only tighten"


def test_long_trail_for_news_call_uses_ratchet():
    # routes through the ratchet, not the flat NEWS_CALL_TRAIL_PCT, once past the first tier
    assert bot.long_trail_for("news_call", "option", 0.20) == 0.25
    assert bot.long_trail_for("news_call", "option", 0.0) == config.NEWS_CALL_TRAIL_PCT  # first tier = flat 40%
