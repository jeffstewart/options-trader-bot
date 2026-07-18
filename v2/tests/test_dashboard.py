"""dashboard.py (v2) -- pure-logic pieces only (log parsing, close-reason labels). Account/
positions/chart functions hit the live Alpaca API and were smoke-tested manually against the
real paper account when the dashboard was built; no network calls in this test file."""
import dashboard


SAMPLE_LOG = """\
14:09:23  INFO      \U0001F4F0 [Alpaca] Beyond Meat Announces Debt Restructuring
14:09:24  INFO        \U0001F916 llama3.2: bullish conf=0.82 mag=0.75 ticker=BYND
14:09:24  INFO             beats estimates and raises outlook
14:09:25  INFO        \U0001F50D mispricing switch: BYND260731C00010000 (resid +0.0120) -> BYND260731C00011000 (resid -0.0080, gap 0.0200) [BYND]
14:09:26  INFO      ✅ OPTION BUY  BYND260731C00011000 x2  limit=$0.5500  max_loss=$110
15:55:10  INFO        → BYND: too close to the same-day forced exit (<30min left) — skipping new lotto entry
16:00:01  INFO      ✅ SELL TO CLOSE  BYND260731C00011000 x2 (lotto_eod) limit=0.60
16:00:02  WARNING   Option quote failed for XYZ123: timeout
"""


def _write_log(tmp_path, text=SAMPLE_LOG):
    p = tmp_path / "bot.log"
    p.write_text(text)
    return p


def test_parse_bot_log_classifies_every_event_kind(tmp_path, monkeypatch):
    monkeypatch.setattr(dashboard, "BOT_LOG", _write_log(tmp_path))
    result = dashboard.parse_bot_log(50)
    kinds = [a["kind"] for a in result["activity"]]
    assert set(kinds) == {"article", "signal", "switch", "trade", "cutoff", "closed"}
    assert len(result["errors"]) == 1
    assert result["errors"][0]["level"] == "WARNING"


def test_parse_bot_log_counters_are_accurate(tmp_path, monkeypatch):
    monkeypatch.setattr(dashboard, "BOT_LOG", _write_log(tmp_path))
    counters = dashboard.parse_bot_log(50)["counters"]
    assert counters["articles"] == 1
    assert counters["scored"] == 1
    assert counters["bullish"] == 1
    assert counters["mispricing_switches"] == 1
    assert counters["trades"] == 1
    assert counters["cutoff_skips"] == 1


def test_parse_bot_log_cutoff_text_strips_arrow_prefix_not_characters(tmp_path, monkeypatch):
    """Regression: an earlier version used msg.strip("  →") which strips a CHARACTER SET
    from both ends (fragile -- would also eat trailing spaces/arrows from real content), not a
    literal prefix. Must produce clean text starting with the ticker, not a mangled string."""
    monkeypatch.setattr(dashboard, "BOT_LOG", _write_log(tmp_path))
    cutoff = next(a for a in dashboard.parse_bot_log(50)["activity"] if a["kind"] == "cutoff")
    assert cutoff["text"].startswith("BYND:")
    assert "→" not in cutoff["text"]


def test_parse_bot_log_activity_is_newest_first(tmp_path, monkeypatch):
    monkeypatch.setattr(dashboard, "BOT_LOG", _write_log(tmp_path))
    activity = dashboard.parse_bot_log(50)["activity"]
    timestamps = [a["ts"] for a in activity]
    assert timestamps == sorted(timestamps, reverse=True)


def test_parse_bot_log_missing_file_returns_empty_shape(tmp_path, monkeypatch):
    monkeypatch.setattr(dashboard, "BOT_LOG", tmp_path / "does_not_exist.log")
    result = dashboard.parse_bot_log(50)
    assert result == {"activity": [], "errors": [], "counters": {}}


def test_close_reason_label_known_and_fallback_cases():
    assert dashboard._close_reason_label("lotto_eod") == "Same-day close"
    assert dashboard._close_reason_label("lotto_stop_loss") == "Stop-loss"
    assert dashboard._close_reason_label("lotto_cap") == "Take-profit cap (3x)"
    assert dashboard._close_reason_label("something_with_profit") == "Take-profit cap"
    assert dashboard._close_reason_label("") == "—"
    assert dashboard._close_reason_label("weird_unmapped_reason") == "weird_unmapped_reason"
