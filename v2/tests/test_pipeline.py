"""Tests for the v2-specific pipeline changes: regime-gate-first ordering, the ticker-selection
corrector, and equity-relative sizing (the structural fixes from the small-account rework)."""
import asyncio

import config as cfg
import execution as ex
import market as mkt
import scoring
import bot


def test_regime_gate_blocks_before_any_scoring_call(monkeypatch):
    """The key structural change: process_signal must never call the (local or, eventually,
    paid) scorer when the regime gate is closed -- that's the entire point of moving the gate
    before scoring."""
    monkeypatch.setattr(mkt, "market_is_open", lambda: True)
    monkeypatch.setattr(mkt, "market_in_uptrend", lambda: False)

    called = []
    monkeypatch.setattr(scoring, "score_article", lambda h, b, s: called.append(1) or None)

    asyncio.run(bot.process_signal("Some bullish headline", "body", "test"))
    assert called == [], "scorer must not be called when the regime gate is closed"


def test_regime_gate_open_allows_scoring(monkeypatch):
    monkeypatch.setattr(mkt, "market_is_open", lambda: True)
    monkeypatch.setattr(mkt, "market_in_uptrend", lambda: True)
    monkeypatch.setattr(ex, "log_regime_decision", lambda *a, **k: None)

    called = []
    monkeypatch.setattr(scoring, "score_article",
                        lambda h, b, s: called.append(1) or {"sentiment": "neutral"})

    asyncio.run(bot.process_signal("Some headline", "body", "test"))
    assert called == [1], "scorer SHOULD be called once the regime gate is open"


def test_ticker_corrector_overrides_primary_scorer_ticker(monkeypatch):
    """Validated 2026-07-16 (research/prompt_v2_test.py): the corrector fixes wrong-beneficiary
    and invented-ticker errors from the primary scorer. Here it should change which ticker the
    legs are dispatched to."""
    monkeypatch.setattr(mkt, "market_is_open", lambda: True)
    monkeypatch.setattr(mkt, "market_in_uptrend", lambda: True)
    monkeypatch.setattr(mkt, "on_cooldown", lambda tk: False)
    monkeypatch.setattr(ex, "log_regime_decision", lambda *a, **k: None)
    monkeypatch.setattr(cfg, "NEWS_CALL_ENABLED", True)  # exercise the news_call dispatch path directly
    monkeypatch.setattr(scoring, "score_article", lambda h, b, s: {
        "tickers": ["MSFT"], "sentiment": "bullish", "confidence": 0.90,
        "magnitude": 0.85, "reasoning": "test",
    })
    monkeypatch.setattr(scoring, "correct_ticker", lambda h, b, cands: "DELL")

    dispatched = []
    monkeypatch.setattr(ex, "execute_news_call", lambda tk, sig: dispatched.append(tk))
    monkeypatch.setattr(ex, "execute_pead", lambda tk, sig: None)
    monkeypatch.setattr(ex, "execute_lotto", lambda tk, sig: None)

    asyncio.run(bot.process_signal("Dell Wins Pentagon Contract", "body", "test", symbols=["DELL", "MSFT"]))
    assert dispatched == ["DELL"], "should trade the corrector's pick, not the primary scorer's raw ticker"


def test_ticker_corrector_no_change_keeps_primary_ticker(monkeypatch):
    """correct_ticker returning None means 'keep the primary scorer's ticker' -- never a gate."""
    monkeypatch.setattr(mkt, "market_is_open", lambda: True)
    monkeypatch.setattr(mkt, "market_in_uptrend", lambda: True)
    monkeypatch.setattr(mkt, "on_cooldown", lambda tk: False)
    monkeypatch.setattr(ex, "log_regime_decision", lambda *a, **k: None)
    monkeypatch.setattr(cfg, "NEWS_CALL_ENABLED", True)  # exercise the news_call dispatch path directly
    monkeypatch.setattr(scoring, "score_article", lambda h, b, s: {
        "tickers": ["AAPL"], "sentiment": "bullish", "confidence": 0.90,
        "magnitude": 0.85, "reasoning": "test",
    })
    monkeypatch.setattr(scoring, "correct_ticker", lambda h, b, cands: None)

    dispatched = []
    monkeypatch.setattr(ex, "execute_news_call", lambda tk, sig: dispatched.append(tk))
    monkeypatch.setattr(ex, "execute_pead", lambda tk, sig: None)
    monkeypatch.setattr(ex, "execute_lotto", lambda tk, sig: None)

    asyncio.run(bot.process_signal("Apple headline", "body", "test", symbols=["AAPL"]))
    assert dispatched == ["AAPL"]


def test_position_budget_is_equity_relative_not_fixed(monkeypatch):
    """v1's MAX_POSITION_USD=$1,000 was ~the entire target ($500-1000) account in one trade.
    v2 must size off CURRENT equity, so the same fraction produces different dollar budgets at
    different account sizes."""
    monkeypatch.setattr(mkt, "account_equity", lambda: 600.0)
    assert mkt.position_budget_usd() == 600.0 * cfg.MAX_POSITION_FRAC_OF_EQUITY

    monkeypatch.setattr(mkt, "account_equity", lambda: 60_000.0)
    assert mkt.position_budget_usd() == 60_000.0 * cfg.MAX_POSITION_FRAC_OF_EQUITY


def test_daily_loss_fail_safe_floor_is_small_not_2000(monkeypatch):
    """The live bug found 2026-07-16: v1's DAILY_LOSS_LIMIT fail-safe floor ($2,000) exceeds an
    entire $500-1000 account, so if the equity fetch fails at exactly that moment the breaker
    provides ZERO protection. v2's floor must be small relative to a small account."""
    monkeypatch.setattr(mkt, "account_equity", lambda: None)   # simulate an equity-fetch failure
    limit = mkt._daily_loss_limit()
    assert limit == cfg.DAILY_LOSS_MIN_FLOOR_USD
    assert limit < 100, "fail-safe floor must stay well under a $500-1000 account, not $2,000"


def test_no_pairs_or_bear_short_config_exists():
    """Structural guard: pairs and bear_short were dropped for the small-account rebuild
    (small_account_pairs_test.py: ~66 concurrent legs needed vs. any realistic 3-5 position
    budget). If either config knob reappears, something regressed the v1 config into v2."""
    assert not hasattr(cfg, "PAIRS_ENABLED")
    assert not hasattr(cfg, "BEAR_SHORT_ENABLED")
    assert not hasattr(cfg, "HYBRID_SCORER_ENABLED")   # confirm/veto gate must not reappear either


def test_news_call_and_pead_disabled_by_default():
    """Disabled 2026-07-18: news_call never found a non-lottery Sharpe shape at any threshold
    (small_account_news_call_sweep.py), and PEAD -- despite the best win-rate shape found all
    session, 53.9% -- needs ~11.7 concurrent positions in steady state
    (small_account_lotto_pead_sweep.py), too capital-intensive for this account. Lotto alone is
    the live leg for now; both flags stay in config as a one-line re-enable, not deleted."""
    assert cfg.NEWS_CALL_ENABLED is False
    assert cfg.PEAD_ENABLED is False
    assert cfg.LOTTO_ENABLED is True


def test_lotto_position_sizing_decided_2026_07_18():
    """jeff's call: 10% of equity per lotto bet, 3-position cap. 10% survives a realistic losing
    streak (per-trade avg loss is only ~12-22% of the BUDGET thanks to the exit rule, so a bad
    5-loss stretch at a ~35-40% win rate costs ~7-10% equity, not 50%) and matches the $100-budget
    tier the exit-rule testing was actually validated at. The 3-position cap is a safety ceiling,
    not a scarce resource -- the same-day exit rule means lotto positions rarely overlap at all."""
    assert cfg.LOTTO_POSITION_FRAC_OF_EQUITY == 0.10
    assert cfg.MAX_OPEN_POSITIONS == 3


def test_lotto_peak_trailing_tiers_removed():
    """Structural guard: the old peak-trailing LOTTO_EXIT_TIERS were replaced 2026-07-17
    (research/lotto_intraday_exit_test.py) by an entry-anchored stop-loss + same-day EOD exit.
    If the tiered config reappears, something regressed the old exit rule back in."""
    assert not hasattr(cfg, "LOTTO_EXIT_TIERS")
    assert not hasattr(mkt, "lotto_trail_pct")
    assert cfg.LOTTO_STOP_LOSS_PCT == 0.20
    assert cfg.LOTTO_SAME_DAY_EXIT is True


def test_lotto_stop_loss_is_entry_anchored_not_peak_anchored():
    """The whole point of the new rule vs. the old trail: the stop must not move up with the
    peak. A trade that spiked to +50% then fell back to -10% must still trip the entry-anchored
    20% stop's threshold check the same as a trade that never rallied at all."""
    pos = {"entry_price": 1.00}
    assert not mkt._lotto_stop_loss_hit(pos, 0.85)   # -15%, above the 20% stop
    assert mkt._lotto_stop_loss_hit(pos, 0.80)        # exactly -20% -> triggers
    assert mkt._lotto_stop_loss_hit(pos, 0.50)        # deep loss -> triggers
    # peak_price is irrelevant to this check -- only entry_price and current mid matter
    pos_after_spike = {"entry_price": 1.00, "peak_price": 1.50}
    assert mkt._lotto_stop_loss_hit(pos_after_spike, 0.79)


def test_lotto_same_day_exit_requires_same_calendar_date(monkeypatch):
    from datetime import datetime, timedelta, timezone
    now = datetime.now(timezone.utc)
    pos_today = {"entry_dt": now - timedelta(hours=2)}
    pos_yesterday = {"entry_dt": now - timedelta(days=1)}
    assert mkt._lotto_same_day_exit_hit(pos_today)
    assert not mkt._lotto_same_day_exit_hit(pos_yesterday)

    monkeypatch.setattr(cfg, "LOTTO_SAME_DAY_EXIT", False)
    assert not mkt._lotto_same_day_exit_hit(pos_today)
