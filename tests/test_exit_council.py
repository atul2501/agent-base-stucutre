"""Covers the advisory, tighten-only exit-council layer: strategy/signals.py's
council_oppose_position and _council_tighten_stop, reasoning/ollama_advisor.py's
consult_exit_batch, and engine/orchestrator.py::_process_exits wiring - see
config.py's EXIT_COUNCIL_ENABLED comment for the full design rationale.

Monkeypatches strategy.signals.evaluate_entry so vote counts are fully
deterministic instead of depending on real indicator math (there is no
existing test that constructs real candle data satisfying arbitrary evolved
genome thresholds - see test_ollama_advisor.py for the batching-call mocking
style this file also follows).
"""
import json
import random

import pytest

from agents.population import Population
from config import Config
from engine.orchestrator import Orchestrator
from market.hyperliquid_client import MarketSnapshot
from reasoning import ollama_advisor
from strategy.genome import Genome
from strategy.signals import ExitAction, Features, Signal, _council_tighten_stop, council_oppose_position


@pytest.fixture(autouse=True)
def _ready_status():
    ollama_advisor._set_status("ready")
    yield


def _snapshot() -> MarketSnapshot:
    return MarketSnapshot(
        coin="SOL", mid_price=100.0, mark_price=100.0, oracle_price=100.0,
        premium=0.0, funding=0.0001, open_interest=1000.0, prev_day_price=100.0,
        day_notional_volume=1_000_000.0, candles=[], bid_levels=[], ask_levels=[], sz_decimals=2,
    )


def _features() -> Features:
    return Features(
        mid_price=100.0, ema_fast=100.0, ema_slow=99.0, trend_up=True,
        rsi_value=28.0, ob_imbalance=1.1, spread_pct=0.01, oi_change_pct=1.0,
        funding=0.0001, premium=0.0001, atr_pct=0.3, adx_value=22.0, volume_ratio=1.1,
        vwap_deviation_pct=-0.2, macd_hist=0.01, daily_change_pct=1.0,
        bb_percent_b=0.2, stoch_rsi_k=20.0, htf_trend_up=True,
    )


def _genomes(n, rng):
    return [Genome.random("SOL", "1m", rng) for _ in range(n)]


def _scripted_votes(monkeypatch, actions):
    """Makes strategy.signals.evaluate_entry return actions[i] (a bare
    action string) on the i-th call, in council iteration order."""
    calls = {"i": 0}

    def fake_evaluate_entry(genome, features):
        action = actions[calls["i"]]
        calls["i"] += 1
        return Signal(action, 0.5, 3, [], ambiguous=False)

    monkeypatch.setattr("strategy.signals.evaluate_entry", fake_evaluate_entry)


class TestCouncilOpposePosition:
    def test_opposed_when_quorum_favors_opposite_side(self, monkeypatch, rng):
        genomes = _genomes(5, rng)
        _scripted_votes(monkeypatch, ["short", "short", "short", "short", "long"])

        opposed, inconclusive, reason = council_oppose_position(
            "long", _snapshot(), None, None, genomes, quorum_pct=0.6, min_active_voters=3,
        )

        assert opposed is True
        assert inconclusive is False
        assert "short" in reason

    def test_not_opposed_when_votes_split_without_quorum(self, monkeypatch, rng):
        genomes = _genomes(4, rng)
        _scripted_votes(monkeypatch, ["short", "short", "long", "long"])

        opposed, inconclusive, reason = council_oppose_position(
            "long", _snapshot(), None, None, genomes, quorum_pct=0.6, min_active_voters=3,
        )

        assert opposed is False
        assert inconclusive is False
        assert reason == ""

    def test_inconclusive_when_too_few_active_voters(self, monkeypatch, rng):
        genomes = _genomes(5, rng)
        _scripted_votes(monkeypatch, ["hold", "hold", "hold", "short", "hold"])

        opposed, inconclusive, reason = council_oppose_position(
            "long", _snapshot(), None, None, genomes, quorum_pct=0.6, min_active_voters=3,
        )

        assert opposed is False
        assert inconclusive is True

    def test_not_opposed_when_council_agrees_with_held_side(self, monkeypatch, rng):
        genomes = _genomes(4, rng)
        _scripted_votes(monkeypatch, ["long", "long", "long", "long"])

        opposed, inconclusive, _reason = council_oppose_position(
            "long", _snapshot(), None, None, genomes, quorum_pct=0.6, min_active_voters=3,
        )

        assert opposed is False
        assert inconclusive is False

    def test_short_side_opposed_by_long_quorum(self, monkeypatch, rng):
        genomes = _genomes(4, rng)
        _scripted_votes(monkeypatch, ["long", "long", "long", "short"])

        opposed, inconclusive, reason = council_oppose_position(
            "short", _snapshot(), None, None, genomes, quorum_pct=0.6, min_active_voters=3,
        )

        assert opposed is True
        assert inconclusive is False
        assert "long" in reason


class TestCouncilTightenStop:
    def test_long_tightens_toward_price(self):
        assert _council_tighten_stop("long", current_stop=90.0, price=100.0, tighten_frac=0.5) == pytest.approx(95.0)

    def test_short_tightens_toward_price(self):
        assert _council_tighten_stop("short", current_stop=110.0, price=100.0, tighten_frac=0.5) == pytest.approx(105.0)

    def test_zero_tighten_frac_is_a_no_op(self):
        assert _council_tighten_stop("long", current_stop=90.0, price=100.0, tighten_frac=0.0) is None

    def test_never_loosens_long(self):
        # price has moved further AGAINST the position than current_stop -
        # naive math must not produce a stop looser than current_stop.
        assert _council_tighten_stop("long", current_stop=90.0, price=80.0, tighten_frac=0.5) is None

    def test_never_loosens_short(self):
        assert _council_tighten_stop("short", current_stop=90.0, price=100.0, tighten_frac=0.5) is None


class _FakeResponse(dict):
    """Mimics the ollama.Client().chat() return shape: response["message"]["content"]."""
    def __init__(self, content: str):
        super().__init__(message={"content": content})


class TestConsultExitBatch:
    def test_empty_input_returns_empty_list(self):
        assert ollama_advisor.consult_exit_batch([]) == []

    def test_resolves_each_item_in_order(self, monkeypatch):
        items = [(Genome.from_dict({"coin": "SOL", "timeframe": "1m"}), _features(), "long") for _ in range(3)]

        def fake_chat(**kwargs):
            return _FakeResponse(json.dumps({"decisions": [
                {"id": 0, "tighten": True, "rationale": "reversal"},
                {"id": 1, "tighten": False, "rationale": "still fine"},
                {"id": 2, "tighten": True, "rationale": "reversal"},
            ]}))

        monkeypatch.setattr(ollama_advisor._client, "chat", fake_chat)
        results = ollama_advisor.consult_exit_batch(items)

        assert results == [True, False, True]

    def test_missing_decision_for_one_id_only_no_ops_that_one(self, monkeypatch):
        items = [(Genome.from_dict({"coin": "SOL", "timeframe": "1m"}), _features(), "long") for _ in range(2)]

        def fake_chat(**kwargs):
            return _FakeResponse(json.dumps({"decisions": [
                {"id": 0, "tighten": True, "rationale": "reversal"},
            ]}))

        monkeypatch.setattr(ollama_advisor._client, "chat", fake_chat)
        results = ollama_advisor.consult_exit_batch(items)

        assert results == [True, False]

    def test_client_exception_no_ops_the_entire_batch(self, monkeypatch):
        items = [(Genome.from_dict({"coin": "SOL", "timeframe": "1m"}), _features(), "long") for _ in range(3)]

        def fake_chat(**kwargs):
            raise RuntimeError("network error")

        monkeypatch.setattr(ollama_advisor._client, "chat", fake_chat)
        monkeypatch.setattr(ollama_advisor, "check_health_async", lambda: None)
        results = ollama_advisor.consult_exit_batch(items)

        assert results == [False, False, False]

    def test_malformed_json_no_ops_the_batch(self, monkeypatch):
        items = [(Genome.from_dict({"coin": "SOL", "timeframe": "1m"}), _features(), "long")]

        def fake_chat(**kwargs):
            return _FakeResponse("not json")

        monkeypatch.setattr(ollama_advisor._client, "chat", fake_chat)
        monkeypatch.setattr(ollama_advisor, "check_health_async", lambda: None)
        results = ollama_advisor.consult_exit_batch(items)

        assert results == [False]

    def test_already_known_broken_status_skips_the_request_entirely(self, monkeypatch):
        ollama_advisor._set_status("not_configured")
        called = {"count": 0}

        def fake_chat(**kwargs):
            called["count"] += 1
            return _FakeResponse("{}")

        monkeypatch.setattr(ollama_advisor._client, "chat", fake_chat)
        items = [(Genome.from_dict({"coin": "SOL", "timeframe": "1m"}), _features(), "long")]
        results = ollama_advisor.consult_exit_batch(items)

        assert called["count"] == 0
        assert results == [False]


def _orchestrator(db, **config_overrides):
    cfg = Config(backtest_enabled=False, **config_overrides)
    pop = Population(db, cfg, rng=random.Random(0))
    return Orchestrator(db=db, hl=None, population=pop, config=cfg, live=None)


class TestProcessExitsSkipsAdvisoryLayerWhenSomethingElseFired:
    """The exit-council check must only run on the evaluate_position() `none`
    branch - never stack with a trail/partial/close decided the same cycle."""

    def test_disabled_by_default_leaves_stop_untouched(self, db, monkeypatch):
        orch = _orchestrator(db, exit_council_enabled=False)
        agent_id = db.create_agent({"coin": "SOL", "timeframe": "1m"}, balance=1000.0)
        db.open_trade(agent_id, "SOL", "long", 100.0, 1.0, 100.0, 95.0, 110.0, "test")

        called = {"count": 0}

        def fake_council_oppose(*a, **k):
            called["count"] += 1
            return False, False, ""

        monkeypatch.setattr("engine.orchestrator.council_oppose_position", fake_council_oppose)
        monkeypatch.setattr("engine.orchestrator.evaluate_position",
                             lambda genome, trade_row, features: ExitAction(kind="none", pnl_pct=0.0))
        orch._process_exits(_snapshot())

        assert called["count"] == 0
        trade = db.get_open_trade(agent_id)
        assert trade["stop_loss"] == 95.0

    def test_skipped_when_trail_already_fired_this_cycle(self, db, monkeypatch):
        orch = _orchestrator(db, exit_council_enabled=True)
        agent_id = db.create_agent({"coin": "SOL", "timeframe": "1m"}, balance=1000.0)
        db.open_trade(agent_id, "SOL", "long", 100.0, 1.0, 100.0, 95.0, 110.0, "test")

        called = {"count": 0}

        def fake_council_oppose(*a, **k):
            called["count"] += 1
            return True, False, "should never be reached"

        monkeypatch.setattr("engine.orchestrator.council_oppose_position", fake_council_oppose)
        monkeypatch.setattr("engine.orchestrator.evaluate_position",
                             lambda genome, trade_row, features: ExitAction(kind="trail", new_stop_loss=97.0, pnl_pct=1.0))
        orch._process_exits(_snapshot())

        assert called["count"] == 0

    def test_tightens_stop_when_council_opposes(self, db, monkeypatch):
        orch = _orchestrator(db, exit_council_enabled=True, exit_council_tighten_frac=0.5)
        agent_id = db.create_agent({"coin": "SOL", "timeframe": "1m"}, balance=1000.0)
        db.open_trade(agent_id, "SOL", "long", 100.0, 1.0, 100.0, 95.0, 110.0, "test")

        monkeypatch.setattr("engine.orchestrator.council_oppose_position",
                             lambda *a, **k: (True, False, "exit council: opposed"))
        monkeypatch.setattr("engine.orchestrator.evaluate_position",
                             lambda genome, trade_row, features: ExitAction(kind="none", pnl_pct=0.0))
        orch._process_exits(_snapshot())  # mid_price=100.0

        trade = db.get_open_trade(agent_id)
        assert trade["stop_loss"] == pytest.approx(97.5)  # halfway from 95 toward 100


def test_backtest_engine_never_imports_the_exit_council_advisory_layer():
    """Guards backtest/live parity: a live council/LLM opinion can't be
    replayed historically, so the backtester must stay on the purely
    deterministic evaluate_position() path forever - see
    backtest/engine.py's comment next to its evaluate_position() call."""
    import backtest.engine as bt_engine

    assert not hasattr(bt_engine, "council_oppose_position")
    assert "council_oppose_position" not in bt_engine.__dict__
    assert "consult_exit_batch" not in bt_engine.__dict__
