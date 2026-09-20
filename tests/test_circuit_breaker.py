"""Covers engine/orchestrator.py's circuit breaker - the one safeguard that
directly protects real money. A regression here (failing to trip, or not
staying tripped across a restart) is the highest-stakes bug class this
codebase can have."""
import random


from agents.population import Population
from config import Config
from engine.orchestrator import Orchestrator
from market.hyperliquid_client import MarketSnapshot


class FakeLive:
    def __init__(self, equity):
        self.equity = equity
        self.adjust_to_calls = []

    def get_account_equity(self):
        return self.equity

    def adjust_to(self, coin, target_side, target_size, sz_decimals):
        self.adjust_to_calls.append((coin, target_side, target_size))
        return []


def _snapshot(mid_price=100.0):
    return MarketSnapshot(
        coin="SOL", mid_price=mid_price, mark_price=mid_price, oracle_price=mid_price,
        premium=0.0, funding=0.0001, open_interest=1000.0, prev_day_price=mid_price,
        day_notional_volume=1_000_000.0, candles=[], bid_levels=[], ask_levels=[], sz_decimals=2,
    )


def _orchestrator(db, live, **config_overrides):
    cfg = Config(backtest_enabled=False, **config_overrides)
    pop = Population(db, cfg, rng=random.Random(0))
    return Orchestrator(db=db, hl=None, population=pop, config=cfg, live=live)


class TestNoLiveExecutor:
    def test_returns_false_when_not_live(self, db):
        orch = _orchestrator(db, live=None)
        assert orch._check_circuit_breaker(_snapshot()) is False

    def test_returns_false_when_breaker_disabled(self, db):
        orch = _orchestrator(db, live=FakeLive(1000.0), live_circuit_breaker_enabled=False)
        assert orch._check_circuit_breaker(_snapshot()) is False


class TestDrawdownTrip:
    def test_trips_on_drawdown_past_threshold(self, db):
        live = FakeLive(equity=1000.0)
        orch = _orchestrator(db, live=live, live_max_drawdown_pct=20.0)
        assert orch._check_circuit_breaker(_snapshot()) is False  # establishes peak=1000

        live.equity = 750.0  # 25% down from peak - past the 20% threshold
        tripped = orch._check_circuit_breaker(_snapshot(mid_price=101.0))

        assert tripped is True
        assert db.get_meta("live_breaker_tripped") == "1"
        assert live.adjust_to_calls  # the real position was flattened

    def test_does_not_trip_below_threshold(self, db):
        # live_max_daily_loss_pct disabled (99%) so only the drawdown check
        # (the thing this test targets) is in play.
        live = FakeLive(equity=1000.0)
        orch = _orchestrator(db, live=live, live_max_drawdown_pct=20.0, live_max_daily_loss_pct=99.0)
        orch._check_circuit_breaker(_snapshot())

        live.equity = 900.0  # only 10% down - under the 20% threshold
        assert orch._check_circuit_breaker(_snapshot(mid_price=101.0)) is False
        assert db.get_meta("live_breaker_tripped") is None


class TestStalePriceFeedTrip:
    def test_trips_after_n_consecutive_identical_prices(self, db):
        live = FakeLive(equity=1000.0)
        orch = _orchestrator(db, live=live, live_stale_price_cycles=3, live_max_drawdown_pct=99.0)

        # First sighting only records _last_seen_price - the stale count
        # starts at 0 and increments only on each SUBSEQUENT matching cycle.
        assert orch._check_circuit_breaker(_snapshot(mid_price=50.0)) is False  # count=0
        assert orch._check_circuit_breaker(_snapshot(mid_price=50.0)) is False  # count=1
        assert orch._check_circuit_breaker(_snapshot(mid_price=50.0)) is False  # count=2
        tripped = orch._check_circuit_breaker(_snapshot(mid_price=50.0))        # count=3 -> trips

        assert tripped is True
        assert db.get_meta("live_breaker_tripped") == "1"

    def test_price_movement_resets_the_stale_counter(self, db):
        live = FakeLive(equity=1000.0)
        orch = _orchestrator(db, live=live, live_stale_price_cycles=2, live_max_drawdown_pct=99.0)

        orch._check_circuit_breaker(_snapshot(mid_price=50.0))
        orch._check_circuit_breaker(_snapshot(mid_price=51.0))  # price moved - resets stale count
        tripped = orch._check_circuit_breaker(_snapshot(mid_price=51.0))

        assert tripped is False
        assert db.get_meta("live_breaker_tripped") is None


class TestPersistsAcrossRestart:
    def test_stays_tripped_for_a_new_orchestrator_instance(self, db):
        live = FakeLive(equity=1000.0)
        orch = _orchestrator(db, live=live, live_max_drawdown_pct=20.0)
        orch._check_circuit_breaker(_snapshot())
        live.equity = 500.0
        assert orch._check_circuit_breaker(_snapshot(mid_price=101.0)) is True

        # Simulates a process restart: brand-new Orchestrator, in-memory
        # staleness counters reset, but the trip itself must survive since
        # it's persisted to the DB, not just held in memory.
        fresh_orch = _orchestrator(db, live=FakeLive(equity=500.0), live_max_drawdown_pct=20.0)
        assert fresh_orch._check_circuit_breaker(_snapshot()) is True

    def test_stale_price_count_survives_a_restart(self, db):
        # The staleness counter used to be in-memory only, so a restart
        # right after a feed freeze handed back a few free cycles before
        # re-tripping - it must now be persisted the same way the trip
        # flag already was.
        live = FakeLive(equity=1000.0)
        orch = _orchestrator(db, live=live, live_stale_price_cycles=3, live_max_drawdown_pct=99.0)
        orch._check_circuit_breaker(_snapshot(mid_price=50.0))  # count=0
        orch._check_circuit_breaker(_snapshot(mid_price=50.0))  # count=1

        # Simulates a process restart: brand-new Orchestrator, same DB.
        fresh_orch = _orchestrator(db, live=FakeLive(equity=1000.0),
                                    live_stale_price_cycles=3, live_max_drawdown_pct=99.0)
        assert fresh_orch._stale_price_count == 1
        tripped = fresh_orch._check_circuit_breaker(_snapshot(mid_price=50.0))  # count=2
        assert tripped is False
        tripped = fresh_orch._check_circuit_breaker(_snapshot(mid_price=50.0))  # count=3 -> trips
        assert tripped is True

    def test_clearing_meta_un_trips_it(self, db):
        live = FakeLive(equity=1000.0)
        orch = _orchestrator(db, live=live, live_max_drawdown_pct=20.0)
        orch._check_circuit_breaker(_snapshot())
        live.equity = 500.0
        orch._check_circuit_breaker(_snapshot(mid_price=101.0))
        assert db.get_meta("live_breaker_tripped") == "1"

        # Mirrors `python3 main.py --clear-live-breaker`.
        db.set_meta("live_breaker_tripped", "0")
        live.equity = 1000.0
        assert orch._check_circuit_breaker(_snapshot(mid_price=102.0)) is False
