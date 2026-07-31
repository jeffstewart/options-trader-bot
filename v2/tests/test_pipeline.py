"""Tests for the v2-specific pipeline changes: regime-gate-first ordering, the ticker-selection
corrector, and equity-relative sizing (the structural fixes from the small-account rework)."""
import asyncio

import pytest

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


def _open_gates(monkeypatch):
    monkeypatch.setattr(mkt, "market_is_open", lambda: True)
    monkeypatch.setattr(mkt, "market_in_uptrend", lambda: True)
    monkeypatch.setattr(mkt, "near_market_close", lambda **k: False)
    monkeypatch.setattr(mkt, "on_cooldown", lambda t: False)
    monkeypatch.setattr(ex, "log_regime_decision", lambda *a, **k: None)


def test_threshold_check_precedes_the_ticker_corrector(monkeypatch):
    """The single threshold gate must sit BEFORE the corrector: once the scorer is a paid model
    the corrector is a second billable call, and spending it on a signal that cannot open a
    position is pure waste. Guards the ordering, which is easy to break by moving the check
    down to leg dispatch."""
    _open_gates(monkeypatch)
    monkeypatch.setattr(cfg, "MIN_MAGNITUDE", 0.35)
    monkeypatch.setattr(cfg, "MIN_CONFIDENCE", 0.70)
    monkeypatch.setattr(scoring, "score_article", lambda h, b, s: {
        "sentiment": "bullish", "magnitude": 0.40, "confidence": 0.55, "tickers": ["AAPL"]})

    corrector_calls, lotto_calls = [], []
    monkeypatch.setattr(scoring, "correct_ticker",
                        lambda h, b, c: corrector_calls.append(1) or "AAPL")
    monkeypatch.setattr(ex, "execute_lotto", lambda t, s: lotto_calls.append(1))

    asyncio.run(bot.process_signal("headline", "body", "test", symbols=["AAPL"]))
    assert corrector_calls == [], "corrector must not run on a below-threshold signal"
    assert lotto_calls == []


def test_single_threshold_pair_admits_the_validated_cell(monkeypatch):
    """mag>=0.35 AND conf>=0.70 (the sonnet5 grid's best total-$ cell) is the ONLY bar. A signal
    at exactly the floor must reach execute_lotto -- under v1's old sliding formula the same
    signal needed conf>=0.862 and was rejected."""
    _open_gates(monkeypatch)
    monkeypatch.setattr(cfg, "MIN_MAGNITUDE", 0.35)
    monkeypatch.setattr(cfg, "MIN_CONFIDENCE", 0.70)
    monkeypatch.setattr(cfg, "LOTTO_MIN_MAGNITUDE", 0.35)
    monkeypatch.setattr(cfg, "LOTTO_MIN_CONFIDENCE", 0.70)
    monkeypatch.setattr(cfg, "TICKER_CORRECTOR_ENABLED", False)
    monkeypatch.setattr(scoring, "score_article", lambda h, b, s: {
        "sentiment": "bullish", "magnitude": 0.35, "confidence": 0.70, "tickers": ["AAPL"]})

    fired = []
    monkeypatch.setattr(ex, "execute_lotto", lambda t, s: fired.append(t))
    asyncio.run(bot.process_signal("headline", "body", "test"))
    assert fired == ["AAPL"]


def test_lotto_thresholds_are_bound_to_the_global_pair():
    """Aliases, not copies -- if these ever diverge, a signal could clear the pipeline gate and
    then be silently dropped at dispatch, which is the exact two-gate confusion v2 removed."""
    assert cfg.LOTTO_MIN_MAGNITUDE is cfg.MIN_MAGNITUDE
    assert cfg.LOTTO_MIN_CONFIDENCE is cfg.MIN_CONFIDENCE
    assert not hasattr(cfg, "CONFIDENCE_SLOPE"), "v1's sliding gate must stay gone from v2"


def test_gate_drops_are_logged_but_throttled(monkeypatch, caplog):
    """Regression guard for the 2026-07-29 diagnosis: both pre-scoring gates used to `return`
    silently, making a fully gated bot look identical to a dead one in the log. They must now
    leave a trace -- but throttled, since every incoming article hits them."""
    monkeypatch.setattr(mkt, "market_is_open", lambda: True)
    monkeypatch.setattr(mkt, "market_in_uptrend", lambda: False)
    bot._gate_drop_counts.clear()

    with caplog.at_level("INFO"):
        for _ in range(bot.GATE_LOG_EVERY + 1):
            asyncio.run(bot.process_signal("headline", "body", "test"))

    gate_lines = [r.getMessage() for r in caplog.records if "regime gate" in r.getMessage()]
    assert len(gate_lines) == 2, f"expected first + one throttled summary, got {gate_lines}"
    assert bot._gate_drop_counts["regime gate: not in uptrend (long-only legs paused)"] == \
        bot.GATE_LOG_EVERY + 1


def test_market_closed_and_regime_drops_count_separately(monkeypatch, caplog):
    """The two reasons must not share a counter -- otherwise a closed market masks the regime
    state (the exact ambiguity this logging exists to remove)."""
    bot._gate_drop_counts.clear()
    monkeypatch.setattr(mkt, "market_is_open", lambda: False)
    with caplog.at_level("INFO"):
        asyncio.run(bot.process_signal("headline", "body", "test"))
    monkeypatch.setattr(mkt, "market_is_open", lambda: True)
    monkeypatch.setattr(mkt, "market_in_uptrend", lambda: False)
    with caplog.at_level("INFO"):
        asyncio.run(bot.process_signal("headline", "body", "test"))

    assert bot._gate_drop_counts == {"market closed": 1,
                                     "regime gate: not in uptrend (long-only legs paused)": 1}


def test_lotto_entry_cutoff_blocks_scoring_when_lotto_is_the_only_leg(monkeypatch):
    """jeff's ask (2026-07-18): once the scorer swap to a paid model (Sonnet) happens, every
    scored article costs API credits, so the late-day entry cutoff must be checked BEFORE
    scoring, not just inside execute_lotto after the (paid) call already happened."""
    monkeypatch.setattr(mkt, "market_is_open", lambda: True)
    monkeypatch.setattr(mkt, "market_in_uptrend", lambda: True)
    monkeypatch.setattr(mkt, "near_market_close", lambda within_min=0: True)
    monkeypatch.setattr(cfg, "NEWS_CALL_ENABLED", False)
    monkeypatch.setattr(cfg, "PEAD_ENABLED", False)

    called = []
    monkeypatch.setattr(scoring, "score_article", lambda h, b, s: called.append(1) or None)

    asyncio.run(bot.process_signal("Some bullish headline", "body", "test"))
    assert called == [], "scorer must not be called inside the lotto entry-cutoff window"


def test_lotto_entry_cutoff_does_not_block_scoring_when_other_legs_are_live(monkeypatch):
    """Safety property: the pre-score skip is only valid because lotto (same-day exit) is the
    ONLY enabled strategy today. If news_call/PEAD (no same-day-exit constraint) are ever
    re-enabled, this must stop applying automatically rather than silently dropping their
    signals near the close too."""
    monkeypatch.setattr(mkt, "market_is_open", lambda: True)
    monkeypatch.setattr(mkt, "market_in_uptrend", lambda: True)
    monkeypatch.setattr(mkt, "near_market_close", lambda within_min=0: True)
    monkeypatch.setattr(cfg, "NEWS_CALL_ENABLED", True)
    monkeypatch.setattr(cfg, "PEAD_ENABLED", False)

    called = []
    monkeypatch.setattr(scoring, "score_article",
                        lambda h, b, s: called.append(1) or {"sentiment": "neutral"})

    asyncio.run(bot.process_signal("Some bullish headline", "body", "test"))
    assert called == [1], "must still score when a non-lotto leg (without the same-day constraint) is live"


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


def test_lotto_stop_is_peak_anchored_reversing_the_old_entry_only_rule():
    """DELIBERATE REVERSAL (2026-07-31). This test previously asserted the opposite -- that the
    stop must NOT move with the peak. jeff's objection: with an entry-anchored stop nothing can
    ever lift the exit above its opening level, so a winner can only be banked by holding to the
    EOD close, and any rally round-trips (AMZN peaked +30.4%, stopped out -22.6%). The stop is now
    max(entry floor, peak * (1 - LOTTO_TRAIL_PCT)) and MUST track the peak."""
    flat = {"entry_price": 1.00, "peak_price": 1.00}
    spiked = {"entry_price": 1.00, "peak_price": 1.50}
    # identical current price, different history -> different stop. That IS the change.
    assert mkt._lotto_stop_loss_hit(flat, 0.84)
    assert not mkt._lotto_stop_loss_hit(spiked, 0.84) or mkt.lotto_stop_price(spiked) > \
        mkt.lotto_stop_price(flat)
    assert mkt.lotto_stop_price(spiked) > mkt.lotto_stop_price(flat)


def test_lotto_same_day_exit_requires_same_calendar_date(monkeypatch):
    from datetime import datetime, timedelta, timezone
    now = datetime.now(timezone.utc)
    pos_today = {"entry_dt": now - timedelta(hours=2)}
    pos_yesterday = {"entry_dt": now - timedelta(days=1)}
    assert mkt._lotto_same_day_exit_hit(pos_today)
    assert not mkt._lotto_same_day_exit_hit(pos_yesterday)

    monkeypatch.setattr(cfg, "LOTTO_SAME_DAY_EXIT", False)
    assert not mkt._lotto_same_day_exit_hit(pos_today)


def test_lotto_entry_cutoff_blocks_new_positions_near_close(monkeypatch):
    """research/lotto_entry_cutoff_test.py (2026-07-18): since every lotto position force-closes
    the SAME day, a signal firing inside the cutoff window has almost no runway before the forced
    exit -- pure spread cost for no developed edge. execute_lotto must skip opening a NEW position
    (but this must never touch position management for already-open ones -- that's a separate
    check in the monitor)."""
    monkeypatch.setattr(mkt, "near_market_close", lambda within_min=0: True)
    called = []
    monkeypatch.setattr(ex, "place_option_trade", lambda *a, **k: called.append(1))
    ex.execute_lotto("AAPL", {"magnitude": 0.9, "confidence": 0.9})
    assert called == [], "must not open a new position inside the entry cutoff window"


def test_lotto_entry_cutoff_allows_positions_with_enough_runway(monkeypatch):
    monkeypatch.setattr(mkt, "near_market_close", lambda within_min=0: False)
    monkeypatch.setattr(mkt, "_at_position_cap", lambda strategy: False)
    monkeypatch.setattr(mkt, "get_stock_price", lambda tk: 100.0)
    monkeypatch.setattr(mkt, "passes_stock_price_guardrail", lambda tk, px: True)
    monkeypatch.setattr(mkt, "lotto_budget_usd", lambda: 100.0)
    called = []
    monkeypatch.setattr(ex, "place_option_trade", lambda *a, **k: called.append(1))
    ex.execute_lotto("AAPL", {"magnitude": 0.9, "confidence": 0.9})
    assert called == [1], "must still open a position when there's enough runway before close"


def test_daily_loss_limit_exceeds_a_single_stopout():
    """Regression guard for 2026-07-31: DAILY_LOSS_LIMIT_PCT (0.02) was exactly equal to
    LOTTO_POSITION_FRAC_OF_EQUITY * LOTTO_STOP_LOSS_PCT (0.10 * 0.20), so the day breaker fired on
    the FIRST normal stop-out — it could never limit a bad *sequence*, only announce a bad trade.
    The limit must stay a genuine multiple of the per-trade worst case."""
    per_trade = cfg.LOTTO_POSITION_FRAC_OF_EQUITY * cfg.LOTTO_STOP_LOSS_PCT
    assert cfg.DAILY_LOSS_LIMIT_PCT >= per_trade * 2, (
        f"daily limit {cfg.DAILY_LOSS_LIMIT_PCT} is only "
        f"{cfg.DAILY_LOSS_LIMIT_PCT / per_trade:.1f}x a single stop-out ({per_trade}) — "
        f"one losing trade would halt the day")
    assert cfg.DAILY_LOSS_MAX_STOPOUTS >= 2


def test_lotto_stop_trails_the_peak_and_never_loosens():
    """jeff's risk requirement (2026-07-31): the exit level must RISE as the contract gains. An
    entry-anchored stop can never lift above its opening level, so the only route to profit is
    holding to the EOD close. Stop = max(entry floor, peak * (1 - trail))."""
    t = cfg.LOTTO_TRAIL_PCT
    floor = 1.0 - cfg.LOTTO_STOP_LOSS_PCT
    pos = {"entry_price": 1.00, "peak_price": 1.00}
    # At entry the stop is whichever is TIGHTER of the trail and the entry floor. At the live
    # trail of 0.15 the trail wins (0.85 > 0.80), so the floor is currently inert -- it only binds
    # if the trail is ever set looser than LOTTO_STOP_LOSS_PCT.
    assert mkt.lotto_stop_price(pos) == pytest.approx(max(floor, 1.0 - t))
    for peak in (1.10, 1.25, 2.00, 4.00):
        pos["peak_price"] = peak
        assert mkt.lotto_stop_price(pos) == pytest.approx(max(floor, peak * (1 - t)))
    # and it must rise monotonically with the peak
    levels = []
    for peak in (1.00, 1.10, 1.25, 2.00, 4.00):
        levels.append(mkt.lotto_stop_price({"entry_price": 1.00, "peak_price": peak}))
    assert levels == sorted(levels)


def test_lotto_trailing_stop_beats_entry_anchored_on_the_real_amzn_path():
    """Replay of the real 2026-07-31 quotes: entry 0.84, peak 1.095 (+30.4%), close 0.59. The live
    entry-anchored rule exited at -22.6%; a trailing stop must exit materially higher."""
    pos = {"entry_price": 0.84, "peak_price": 0.84}
    exit_px = None
    for mid in (0.815, 0.785, 0.84, 0.94, 1.01, 1.095, 1.00, 0.92, 0.87, 0.80, 0.70, 0.59):
        pos["peak_price"] = max(pos["peak_price"], mid)
        if mkt._lotto_stop_loss_hit(pos, mid):
            exit_px = mid
            break
    assert exit_px is not None, "trailing stop never fired on a round-tripping path"
    assert exit_px > 0.84 * 0.80, f"exited at {exit_px}, no better than the entry-anchored stop"


def test_position_path_logging_writes_a_replayable_row(tmp_path, monkeypatch):
    """Ported from v1 2026-07-31 so exit rules can be judged on REAL quotes. Before this, the only
    record of a position's path was the 👁 log lines — the AMZN post-mortem had to regex 203 of
    them out of bot.log, and they vanish on rotation."""
    monkeypatch.chdir(tmp_path)
    pos = {"strategy": "lotto", "asset_type": "option", "underlying": "AMZN",
           "entry_price": 0.84, "peak_price": 1.095}
    ex._log_position_path("AMZN260814C00295000", pos, 0.90, 7.14, 0.876)
    ex._log_position_path("AMZN260814C00295000", pos, 0.88, 4.76, 0.876)
    rows = (tmp_path / "position_paths.csv").read_text().strip().splitlines()
    assert len(rows) == 3                                  # header + 2 samples
    assert rows[0].startswith("ts,symbol,strategy")
    cols = rows[1].split(",")
    assert cols[1] == "AMZN260814C00295000" and cols[2] == "lotto"
    assert float(cols[6]) == 0.9 and float(cols[7]) == 1.095   # mid, peak
    assert float(cols[9]) == 0.876                             # stop travels with the row


def test_position_path_logging_never_breaks_the_monitor(monkeypatch):
    """Telemetry must fail silently — a full disk or a bad path can't be allowed to stop the
    stop-loss checks that run in the same loop."""
    def boom(*a, **k):
        raise OSError("disk full")
    monkeypatch.setattr("builtins.open", boom)
    ex._log_position_path("X", {"entry_price": 1.0}, 1.0, 0.0, 0.8)   # must not raise
