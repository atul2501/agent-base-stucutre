"""Covers engine/orchestrator.py's paper-population circuit breaker -
unlike the live one (which only protects real money), this guards the
paper population that drives every evolutionary decision, since it
previously had no systemic drawdown protection at all."""
import random

from agents.population import Population
from config import Config
from db.database import Database
from engine.orchestrator import Orchestrator


def _orchestrator(db, **config_overrides):
    cfg = Config(backtest_enabled=False, **config_overrides)
    pop = Population(db, cfg, rng=random.Random(0))
    return Orchestrator(db=db, hl=None, population=pop, config=cfg, live=None)


def _record_realized_pnl(db: Database, pnl: float) -> None:
    agent_id = db.create_agent({"coin": "SOL"}, balance=1000.0)
    trade_id = db.open_trade(agent_id, "SOL", "long", 100.0, 1.0, 100.0, 95.0, 110.0, "test")
    db.close_trade(trade_id, 100.0, pnl, "win" if pnl >= 0 else "loss", "test")


class TestPaperDrawdownGuard:
    def test_disabled_returns_false(self, db):
        orch = _orchestrator(db, paper_circuit_breaker_enabled=False)
        _record_realized_pnl(db, -10_000.0)
        assert orch._check_paper_drawdown() is False

    def test_does_not_trip_below_threshold(self, db):
        orch = _orchestrator(db, paper_max_drawdown_usd=1000.0)
        _record_realized_pnl(db, -500.0)
        assert orch._check_paper_drawdown() is False

    def test_trips_past_threshold(self, db):
        orch = _orchestrator(db, paper_max_drawdown_usd=1000.0)
        _record_realized_pnl(db, -1500.0)
        assert orch._check_paper_drawdown() is True
        assert db.get_meta("paper_breaker_paused") == "1"

    def test_stays_paused_while_still_drawn_down(self, db):
        orch = _orchestrator(db, paper_max_drawdown_usd=1000.0)
        _record_realized_pnl(db, -1500.0)
        assert orch._check_paper_drawdown() is True
        _record_realized_pnl(db, 100.0)  # recovers a little, but not past half-threshold
        assert orch._check_paper_drawdown() is True

    def test_auto_resumes_once_recovered_past_half_threshold(self, db):
        orch = _orchestrator(db, paper_max_drawdown_usd=1000.0)
        _record_realized_pnl(db, -1500.0)
        assert orch._check_paper_drawdown() is True
        _record_realized_pnl(db, 1200.0)  # drawdown now well under 500 (half of 1000)
        assert orch._check_paper_drawdown() is False
        assert db.get_meta("paper_breaker_paused") == "0"

    def test_peak_tracks_new_realized_pnl_highs(self, db):
        orch = _orchestrator(db, paper_max_drawdown_usd=1000.0)
        _record_realized_pnl(db, 2000.0)  # new all-time high
        assert orch._check_paper_drawdown() is False
        _record_realized_pnl(db, -1500.0)  # drawdown measured from the NEW peak (2000), not 0
        assert orch._check_paper_drawdown() is True
