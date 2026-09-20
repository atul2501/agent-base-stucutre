"""Covers reasoning/ollama_advisor.py's consult_batch - resolves a whole
cycle's worth of ambiguous signals in one request instead of one call per
agent (see engine/orchestrator.py::_process_entries). Mocks the Ollama
client's .chat() so these run with no network access."""
import json

import pytest

from reasoning import ollama_advisor
from strategy.genome import Genome
from strategy.signals import Features, Signal


@pytest.fixture(autouse=True)
def _ready_status():
    ollama_advisor._set_status("ready")
    yield


def _genome() -> Genome:
    return Genome.from_dict({"coin": "SOL", "timeframe": "1m"})


def _features() -> Features:
    return Features(
        mid_price=100.0, ema_fast=100.0, ema_slow=99.0, trend_up=True,
        rsi_value=28.0, ob_imbalance=1.1, spread_pct=0.01, oi_change_pct=1.0,
        funding=0.0001, premium=0.0001, atr_pct=0.3, adx_value=22.0, volume_ratio=1.1,
        vwap_deviation_pct=-0.2, macd_hist=0.01, daily_change_pct=1.0,
        bb_percent_b=0.2, stoch_rsi_k=20.0, htf_trend_up=True,
    )


def _candidate(score=1) -> Signal:
    return Signal("long", 0.4, score, ["uptrend pullback"], ambiguous=True)


class _FakeResponse(dict):
    """Mimics the ollama.Client().chat() return shape: response["message"]["content"]."""
    def __init__(self, content: str):
        super().__init__(message={"content": content})


class TestConsultBatch:
    def test_empty_input_returns_empty_list(self):
        assert ollama_advisor.consult_batch([]) == []

    def test_resolves_each_item_in_order(self, monkeypatch):
        items = [(_genome(), _features(), _candidate(score=i)) for i in range(3)]

        def fake_chat(**kwargs):
            return _FakeResponse(json.dumps({"decisions": [
                {"id": 0, "action": "long", "confidence": 0.8, "rationale": "ok"},
                {"id": 1, "action": "hold", "confidence": 0.0, "rationale": "conflicting"},
                {"id": 2, "action": "short", "confidence": 0.6, "rationale": "ok"},
            ]}))

        monkeypatch.setattr(ollama_advisor._client, "chat", fake_chat)
        results = ollama_advisor.consult_batch(items)

        assert [r.action for r in results] == ["long", "hold", "short"]
        assert results[0].confidence == pytest.approx(0.8)
        assert all(r.ambiguous is False for r in results)

    def test_missing_decision_for_one_id_only_holds_that_one(self, monkeypatch):
        items = [(_genome(), _features(), _candidate(score=i)) for i in range(2)]

        def fake_chat(**kwargs):
            # Only returns a decision for id 0 - id 1 is missing entirely.
            return _FakeResponse(json.dumps({"decisions": [
                {"id": 0, "action": "long", "confidence": 0.9, "rationale": "ok"},
            ]}))

        monkeypatch.setattr(ollama_advisor._client, "chat", fake_chat)
        results = ollama_advisor.consult_batch(items)

        assert results[0].action == "long"
        assert results[1].action == "hold"

    def test_client_exception_holds_the_entire_batch(self, monkeypatch):
        items = [(_genome(), _features(), _candidate(score=i)) for i in range(3)]

        def fake_chat(**kwargs):
            raise RuntimeError("network error")

        monkeypatch.setattr(ollama_advisor._client, "chat", fake_chat)
        monkeypatch.setattr(ollama_advisor, "check_health_async", lambda: None)
        results = ollama_advisor.consult_batch(items)

        assert len(results) == 3
        assert all(r.action == "hold" for r in results)

    def test_malformed_json_holds_the_entire_batch(self, monkeypatch):
        items = [(_genome(), _features(), _candidate())]

        def fake_chat(**kwargs):
            return _FakeResponse("not json")

        monkeypatch.setattr(ollama_advisor._client, "chat", fake_chat)
        monkeypatch.setattr(ollama_advisor, "check_health_async", lambda: None)
        results = ollama_advisor.consult_batch(items)

        assert results[0].action == "hold"

    def test_already_known_broken_status_skips_the_request_entirely(self, monkeypatch):
        ollama_advisor._set_status("not_configured")
        called = {"count": 0}

        def fake_chat(**kwargs):
            called["count"] += 1
            return _FakeResponse("{}")

        monkeypatch.setattr(ollama_advisor._client, "chat", fake_chat)
        items = [(_genome(), _features(), _candidate())]
        results = ollama_advisor.consult_batch(items)

        assert called["count"] == 0
        assert results[0].action == "hold"

    def test_out_of_range_confidence_is_clamped(self, monkeypatch):
        items = [(_genome(), _features(), _candidate())]

        def fake_chat(**kwargs):
            return _FakeResponse(json.dumps({"decisions": [
                {"id": 0, "action": "long", "confidence": 5.0, "rationale": "ok"},
            ]}))

        monkeypatch.setattr(ollama_advisor._client, "chat", fake_chat)
        results = ollama_advisor.consult_batch(items)

        assert results[0].confidence == 1.0
