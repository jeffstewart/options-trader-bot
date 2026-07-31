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
    """result: a _Resp to return, or an Exception instance to raise."""
    monkeypatch.chdir(tmp_path)          # keep the latency CSV out of the repo
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
    monkeypatch.setattr(cfg, "ANTHROPIC_API_KEY", "")
    monkeypatch.delenv("ANTHROPIC_AUTH_TOKEN", raising=False)
    assert scoring.preflight() is False
