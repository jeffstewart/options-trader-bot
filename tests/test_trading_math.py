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
