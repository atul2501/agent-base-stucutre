import random

import pytest

from config import Config
from db.database import Database


@pytest.fixture
def db(tmp_path):
    database = Database(str(tmp_path / "test.db"))
    yield database
    database.close()


@pytest.fixture
def config():
    # Small numbers so population tests run fast and are easy to reason
    # about, not the real 500/50 production defaults.
    return Config(
        population_cap=10,
        active_trader_count=4,
        guaranteed_newcomer_slots=1,
        guaranteed_longest_benched_slots=1,
        min_population_floor=2,
        backtest_enabled=False,
        live_circuit_breaker_enabled=True,
    )


@pytest.fixture
def rng():
    return random.Random(1234)
