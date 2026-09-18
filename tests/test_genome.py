"""Covers strategy/genome.py's invariants: every genome-producing path
(random/mutate/crossover/from_dict) must stay within BOUNDS and keep the
ordered-pair / TP:SL-ratio / hold-window invariants intact, since these feed
directly into stop_loss_pct/take_profit_pct/max_hold_hours used by real
order sizing and exits."""

import pytest

from strategy.genome import BOUNDS, _MIN_TP_SL_RATIO, Genome, genome_distance


def _assert_within_bounds(genome: Genome) -> None:
    d = genome.to_dict()
    for field_name, (lo, hi) in BOUNDS.items():
        value = d[field_name]
        assert lo <= value <= hi, f"{field_name}={value} outside bounds ({lo}, {hi})"


def _assert_invariants(genome: Genome) -> None:
    _assert_within_bounds(genome)
    assert genome.ema_slow > genome.ema_fast
    assert genome.max_atr_pct > genome.min_atr_pct
    assert genome.take_profit_pct >= genome.stop_loss_pct * _MIN_TP_SL_RATIO - 1e-9
    assert genome.max_hold_hours <= BOUNDS["max_hold_hours"][1]


class TestRandom:
    def test_many_random_genomes_all_satisfy_invariants(self, rng):
        for _ in range(200):
            genome = Genome.random("SOL", "1m", rng)
            _assert_invariants(genome)


class TestMutate:
    def test_mutated_child_satisfies_invariants(self, rng):
        parent = Genome.random("SOL", "1m", rng)
        for _ in range(200):
            child = parent.mutate(rng, mutation_rate=1.0)  # mutate every field
            _assert_invariants(child)

    def test_mutate_never_changes_coin(self, rng):
        parent = Genome.random("SOL", "1m", rng)
        child = parent.mutate(rng, mutation_rate=1.0)
        assert child.coin == "SOL"

    def test_zero_mutation_rate_returns_identical_copy(self, rng):
        parent = Genome.random("SOL", "1m", rng)
        child = parent.mutate(rng, mutation_rate=0.0)
        assert child.to_dict() == parent.to_dict()
        assert child is not parent  # deep copy, not the same object


class TestCrossover:
    def test_crossed_child_satisfies_invariants(self, rng):
        parent_a = Genome.random("SOL", "1m", rng)
        parent_b = Genome.random("SOL", "1m", rng)
        for _ in range(200):
            child = parent_a.crossover(parent_b, rng)
            _assert_invariants(child)

    def test_crossed_field_comes_from_one_parent_or_the_other(self, rng):
        parent_a = Genome.random("SOL", "1m", rng)
        parent_b = Genome.random("SOL", "1m", rng)
        child = parent_a.crossover(parent_b, rng)
        for field_name in BOUNDS.keys():
            if field_name in ("ema_slow", "max_atr_pct", "stop_loss_pct", "take_profit_pct", "max_hold_hours"):
                continue  # these can be adjusted afterward to restore invariants
            value = getattr(child, field_name)
            assert value in (getattr(parent_a, field_name), getattr(parent_b, field_name))


class TestFromDict:
    def test_round_trips_a_valid_genome(self, rng):
        # Field-by-field approx comparison rather than dataclass equality:
        # _enforce_tp_sl_ratio's ceil-to-2-decimals floor can differ from
        # _resample_tp_sl_ratio's unrounded floor by float epsilon, which
        # strict equality would flag as a mismatch even though nothing
        # meaningful changed.
        original = Genome.random("SOL", "1m", rng)
        restored = Genome.from_dict(original.to_dict())
        for field_name, value in original.to_dict().items():
            restored_value = getattr(restored, field_name)
            if isinstance(value, float):
                assert restored_value == pytest.approx(value, abs=1e-6)
            else:
                assert restored_value == value

    def test_fills_missing_fields_with_bounds_midpoint(self):
        minimal = {"coin": "SOL", "timeframe": "1m"}
        genome = Genome.from_dict(minimal)
        _assert_invariants(genome)

    def test_is_idempotent(self):
        genome = Genome.from_dict({"coin": "SOL", "timeframe": "1m"})
        again = Genome.from_dict(genome.to_dict())
        assert again == genome

    def test_repairs_inverted_ema_pair(self):
        d = Genome.from_dict({"coin": "SOL", "timeframe": "1m"}).to_dict()
        d["ema_fast"], d["ema_slow"] = 40, 10  # deliberately inverted
        genome = Genome.from_dict(d)
        assert genome.ema_slow > genome.ema_fast

    def test_repairs_bad_tp_sl_ratio(self):
        d = Genome.from_dict({"coin": "SOL", "timeframe": "1m"}).to_dict()
        d["stop_loss_pct"], d["take_profit_pct"] = 5.0, 2.0  # way below the 2x floor
        genome = Genome.from_dict(d)
        assert genome.take_profit_pct >= genome.stop_loss_pct * _MIN_TP_SL_RATIO - 1e-9


class TestGenomeDistance:
    def test_identical_genomes_have_zero_distance(self, rng):
        genome = Genome.random("SOL", "1m", rng)
        assert genome_distance(genome, genome) == pytest.approx(0.0)

    def test_distance_is_symmetric(self, rng):
        a = Genome.random("SOL", "1m", rng)
        b = Genome.random("SOL", "1m", rng)
        assert genome_distance(a, b) == pytest.approx(genome_distance(b, a))

    def test_distance_is_bounded_between_zero_and_one(self, rng):
        for _ in range(50):
            a = Genome.random("SOL", "1m", rng)
            b = Genome.random("SOL", "1m", rng)
            d = genome_distance(a, b)
            assert 0.0 <= d <= 1.0 + 1e-9
