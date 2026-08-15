"""Scorer swap to Sonnet 5 (2026-07-29). No network in tests -- the Anthropic client is stubbed.
Covers the failure modes that are NEW with a remote paid scorer: refusals, empty/partial content,
connection failures after retries, and the latency telemetry that decides the thinking setting."""
from types import SimpleNamespace

import anthropic
import pytest

import config as cfg
import scoring


class _Resp(SimpleNamespace):
    pass


def _resp(text=None, stop_reason="end_turn", category=None):
    content = [SimpleNamespace(type="text", text=text)] if text is not None else []
    return _Resp(content=content, stop_reason=stop_reason,
                 stop_details=SimpleNamespace(category=category),
                 usage=SimpleNamespace(input_tokens=1500, output_tokens=90,
                                       cache_read_input_tokens=0))


def _stub(monkeypatch, tmp_path, result):
    """result: a _Resp to return, or an Exception instance to raise. Pins SCORER_MODEL to the
    Anthropic path being tested -- SCORER_MODEL's default became kimi-k2.6 on 2026-08-04 (jeff
    out of Anthropic credits), so these tests must not rely on the ambient default routing here."""
    monkeypatch.chdir(tmp_path)          # keep the latency CSV out of the repo
    monkeypatch.setattr(cfg, "SCORER_MODEL", "claude-sonnet-5")
    def create(**kwargs):
        create.kwargs = kwargs
        if isinstance(result, Exception):
            raise result
        return result
    monkeypatch.setattr(scoring.anthropic_client.messages, "create", create)
    return create


def test_scores_a_valid_response(monkeypatch, tmp_path):
    create = _stub(monkeypatch, tmp_path, _resp(
        '{"tickers":["ADTN"],"sentiment":"bullish","confidence":0.85,'
        '"magnitude":0.6,"catalyst":0.8,"reasoning":"beat and raise"}'))
    s = scoring.score_article("ADTN beats", "body", "Alpaca/Benzinga")
    assert s["tickers"] == ["ADTN"] and s["confidence"] == 0.85
    # Schema must be enforced server-side, not by post-hoc parsing.
    assert create.kwargs["output_config"]["format"]["type"] == "json_schema"
    assert create.kwargs["model"] == cfg.SCORER_MODEL


def test_refusal_returns_none_without_indexing_content(monkeypatch, tmp_path):
    """Safety classifiers decline with HTTP 200 + stop_reason 'refusal' and an EMPTY content list.
    Indexing content[0] here would raise instead of degrading to 'unscorable'."""
    _stub(monkeypatch, tmp_path, _resp(text=None, stop_reason="refusal", category="cyber"))
    assert scoring.score_article("something", "body") is None


def test_empty_content_returns_none(monkeypatch, tmp_path):
    _stub(monkeypatch, tmp_path, _resp(text=None, stop_reason="max_tokens"))
    assert scoring.score_article("h", "b") is None


def test_connection_error_after_retries_returns_none(monkeypatch, tmp_path):
    """The container's DNS drops in bursts; the SDK retries, but a long burst still surfaces here.
    A dead scorer must degrade to 'no signal', never propagate and kill the poller task."""
    _stub(monkeypatch, tmp_path,
          anthropic.APIConnectionError(request=SimpleNamespace()))
    assert scoring.score_article("h", "b") is None


def test_latency_is_recorded_on_success_and_on_failure(monkeypatch, tmp_path):
    _stub(monkeypatch, tmp_path, _resp('{"tickers":[],"sentiment":"neutral","confidence":0.1,'
                                       '"magnitude":0.1,"catalyst":0.0,"reasoning":"x"}'))
    scoring.score_article("h", "b")
    _stub(monkeypatch, tmp_path, anthropic.APIConnectionError(request=SimpleNamespace()))
    scoring.score_article("h", "b")
    rows = (tmp_path / scoring.SCORER_LATENCY_CSV).read_text().strip().splitlines()
    assert len(rows) == 3                       # header + 2 calls
    assert rows[1].split(",")[2] == "1"         # ok=1 on the success
    assert rows[2].split(",")[2] == "0"         # ok=0 on the failure
    assert float(rows[1].split(",")[3]) >= 0    # latency_ms present


def test_preflight_fails_closed_without_a_key(monkeypatch):
    monkeypatch.setattr(cfg, "SCORER_MODEL", "claude-sonnet-5")
    monkeypatch.setattr(cfg, "ANTHROPIC_API_KEY", "")
    monkeypatch.delenv("ANTHROPIC_AUTH_TOKEN", raising=False)
    assert scoring.preflight() is False


# ── Kimi (Moonshot) path -- temporary default since 2026-08-04, jeff out of Anthropic credits ──

def _stub_kimi(monkeypatch, tmp_path, content=None, exc=None):
    """content: the raw string kimi's message.content would hold. exc: an Exception to raise
    instead. Always pins SCORER_MODEL to a kimi id so _is_kimi() routes here regardless of
    whatever the ambient default happens to be."""
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(cfg, "SCORER_MODEL", "kimi-k2.6")
    def create(**kwargs):
        create.kwargs = kwargs
        if exc is not None:
            raise exc
        usage = SimpleNamespace(prompt_tokens=340, completion_tokens=80)
        msg = SimpleNamespace(content=content)
        return SimpleNamespace(choices=[SimpleNamespace(message=msg)], usage=usage)
    monkeypatch.setattr(scoring.kimi_client.chat.completions, "create", create)
    return create


def test_kimi_scores_a_valid_response_and_disables_thinking_by_default(monkeypatch, tmp_path):
    create = _stub_kimi(monkeypatch, tmp_path,
                         '{"tickers":["ADTN"],"sentiment":"bullish","confidence":0.85,'
                         '"magnitude":0.6,"reasoning":"beat and raise"}')
    s = scoring.score_article("ADTN beats", "body", "Alpaca/Benzinga")
    assert s["tickers"] == ["ADTN"] and s["confidence"] == 0.85
    assert create.kwargs["model"] == "kimi-k2.6"
    assert "temperature" not in create.kwargs           # kimi-k2.6 rejects anything but 1
    assert create.kwargs["extra_body"] == {"thinking": {"type": "disabled"}}


def test_kimi_thinking_enabled_omits_the_disable_flag(monkeypatch, tmp_path):
    create = _stub_kimi(monkeypatch, tmp_path,
                         '{"tickers":[],"sentiment":"neutral","confidence":0.1,'
                         '"magnitude":0.1,"reasoning":"x"}')
    monkeypatch.setattr(cfg, "SCORER_THINKING", "enabled")
    scoring.score_article("h", "b")
    assert "extra_body" not in create.kwargs


def test_kimi_strips_markdown_fences_before_parsing(monkeypatch, tmp_path):
    _stub_kimi(monkeypatch, tmp_path,
               '```json\n{"tickers":["BA"],"sentiment":"bullish","confidence":0.7,'
               '"magnitude":0.5,"reasoning":"x"}\n```')
    s = scoring.score_article("h", "b")
    assert s["tickers"] == ["BA"]


def test_kimi_unparseable_text_returns_none(monkeypatch, tmp_path):
    _stub_kimi(monkeypatch, tmp_path, "not json at all")
    assert scoring.score_article("h", "b") is None


def test_kimi_connection_error_returns_none(monkeypatch, tmp_path):
    import openai
    _stub_kimi(monkeypatch, tmp_path, exc=openai.APIConnectionError(request=SimpleNamespace()))
    assert scoring.score_article("h", "b") is None


def test_kimi_latency_uses_prompt_completion_token_names(monkeypatch, tmp_path):
    """OpenAI's usage object has prompt_tokens/completion_tokens, not Anthropic's
    input_tokens/output_tokens -- the normalization in scoring._Usage must translate, not just
    forward whichever attribute happens to exist (which would silently write blanks)."""
    _stub_kimi(monkeypatch, tmp_path,
               '{"tickers":[],"sentiment":"neutral","confidence":0.1,'
               '"magnitude":0.1,"reasoning":"x"}')
    scoring.score_article("h", "b")
    rows = (tmp_path / scoring.SCORER_LATENCY_CSV).read_text().strip().splitlines()
    header, row = rows[0].split(","), rows[1].split(",")
    assert row[header.index("input_tokens")] == "340"
    assert row[header.index("output_tokens")] == "80"


def test_kimi_preflight_fails_closed_without_a_key(monkeypatch):
    monkeypatch.setattr(cfg, "SCORER_MODEL", "kimi-k2.6")
    monkeypatch.setattr(cfg, "MOONSHOT_API_KEY", "")
    assert scoring.preflight() is False


def test_kimi_preflight_succeeds_when_model_listed(monkeypatch):
    monkeypatch.setattr(cfg, "SCORER_MODEL", "kimi-k2.6")
    monkeypatch.setattr(cfg, "MOONSHOT_API_KEY", "fake-key-for-test")
    monkeypatch.setattr(scoring.kimi_client.models, "list",
                         lambda: SimpleNamespace(data=[SimpleNamespace(id="kimi-k2.6")]))
    assert scoring.preflight() is True


def test_kimi_preflight_fails_when_model_not_entitled(monkeypatch):
    monkeypatch.setattr(cfg, "SCORER_MODEL", "kimi-k3")
    monkeypatch.setattr(cfg, "MOONSHOT_API_KEY", "fake-key-for-test")
    monkeypatch.setattr(scoring.kimi_client.models, "list",
                         lambda: SimpleNamespace(data=[SimpleNamespace(id="kimi-k2.6")]))
    assert scoring.preflight() is False
