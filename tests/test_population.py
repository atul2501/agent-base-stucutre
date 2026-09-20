"""Covers agents/population.py's do-or-die lifecycle: culling, active-trader
ranking, breeding, and promotion. These are the mechanics the review council
flagged as having zero test coverage despite being the intellectual core of
the system."""

import pytest

from agents.population import Population
from config import Config
from strategy.genome import Genome


def _make_agent(db, rng, balance=1000.0):
    genome = Genome.random("SOL", "1m", rng)
    return db.create_agent(genome.to_dict(), balance=balance)


class TestCulling:
    def test_fresh_agents_are_protected_from_cap_based_culling(self, db, rng):
        cfg = Config(
            population_cap=3, active_trader_count=2, guaranteed_newcomer_slots=0,
            guaranteed_longest_benched_slots=0, diversity_floor_enabled=False,
            forced_rotation_every_n_cycles=0, backtest_enabled=False,
        )
        pop = Population(db, cfg, rng=rng)
        for _ in range(5):
            _make_agent(db, rng)

        pop.rank_and_enforce()

        # Overflow (5 alive vs cap 3) exists, but none of them have completed
        # a trade or sat idle past the cull timeout, so none are cullable.
        assert db.count_alive() == 5

    def test_cull_removes_lowest_fitness_traded_agents_first(self, db, rng):
        cfg = Config(
            population_cap=2, active_trader_count=2, guaranteed_newcomer_slots=0,
            guaranteed_longest_benched_slots=0, diversity_floor_enabled=False,
            forced_rotation_every_n_cycles=0, backtest_enabled=False,
        )
        pop = Population(db, cfg, rng=rng)
        ids = [_make_agent(db, rng) for _ in range(4)]
        # Give each a completed trade (trades_count > 0, so cullable) with
        # distinct pnl so fitness ranking is unambiguous.
        for agent_id, pnl in zip(ids, [10.0, 50.0, 5.0, 100.0]):
            db.record_win(agent_id, pnl)

        pop.rank_and_enforce()

        alive_ids = {a.id for a in db.list_alive_agents()}
        assert alive_ids == {ids[1], ids[3]}  # the two highest-pnl agents
        assert db.count_alive() == 2

    def test_idle_agent_with_zero_trades_can_be_culled_past_timeout(self, db, rng, monkeypatch):
        cfg = Config(
            population_cap=1, active_trader_count=1, guaranteed_newcomer_slots=0,
            guaranteed_longest_benched_slots=0, diversity_floor_enabled=False,
            forced_rotation_every_n_cycles=0, backtest_enabled=False,
            max_idle_hours_before_cull=0,  # anything is "idle enough" immediately
        )
        pop = Population(db, cfg, rng=rng)
        _make_agent(db, rng)
        _make_agent(db, rng)

        pop.rank_and_enforce()

        assert db.count_alive() == 1


class TestActiveTraderSelection:
    def test_top_fitness_agents_become_active_traders(self, db, rng):
        cfg = Config(
            population_cap=10, active_trader_count=2, guaranteed_newcomer_slots=0,
            guaranteed_longest_benched_slots=0, diversity_floor_enabled=False,
            forced_rotation_every_n_cycles=0, backtest_enabled=False,
        )
        pop = Population(db, cfg, rng=rng)
        ids = [_make_agent(db, rng) for _ in range(4)]
        for agent_id, pnl in zip(ids, [10.0, 50.0, 5.0, 100.0]):
            db.record_win(agent_id, pnl)

        pop.rank_and_enforce()

        active_ids = {a.id for a in db.list_active_traders()}
        assert active_ids == {ids[1], ids[3]}


class TestWinLossLifecycle:
    def test_handle_win_spawns_configured_number_of_children(self, db, rng):
        cfg = Config(children_per_win=3, backtest_enabled=False, crossover_probability=0.0)
        pop = Population(db, cfg, rng=rng)
        agent_id = _make_agent(db, rng)

        pop.handle_win(agent_id, pnl=25.0)

        children = db.children_of(agent_id)
        assert len(children) == 3
        agent = db.get_agent(agent_id)
        assert agent.wins == 1
        assert agent.total_pnl == pytest.approx(25.0)
        assert agent.status == "alive"

    def test_handle_loss_kills_the_agent(self, db, rng):
        cfg = Config(backtest_enabled=False)
        pop = Population(db, cfg, rng=rng)
        agent_id = _make_agent(db, rng)

        pop.handle_loss(agent_id, pnl=-10.0)

        agent = db.get_agent(agent_id)
        assert agent.status == "dead"
        assert agent.death_reason == "losing trade"
        assert agent.total_pnl == pytest.approx(-10.0)

    def test_strategy_share_only_recorded_on_win_streak_threshold(self, db, rng):
        """Regression test: a batching refactor of handle_win once
        accidentally de-indented record_strategy_share out of its
        win_streak-threshold `if`, making it fire unconditionally on every
        win instead of only every win_streak_share_threshold wins."""
        cfg = Config(win_streak_share_threshold=3, backtest_enabled=False,
                      crossover_probability=0.0, children_per_win=0)
        pop = Population(db, cfg, rng=rng)
        agent_id = _make_agent(db, rng)

        pop.handle_win(agent_id, pnl=1.0)  # streak=1 - below threshold
        assert db.sample_shared_genome() is None
        pop.handle_win(agent_id, pnl=1.0)  # streak=2 - below threshold
        assert db.sample_shared_genome() is None
        pop.handle_win(agent_id, pnl=1.0)  # streak=3 - hits the threshold
        assert db.sample_shared_genome() is not None

    def test_parent_promoted_once_both_children_have_won(self, db, rng):
        cfg = Config(children_per_win=2, backtest_enabled=False, crossover_probability=0.0)
        pop = Population(db, cfg, rng=rng)
        parent_id = _make_agent(db, rng)

        pop.handle_win(parent_id, pnl=10.0)  # spawns 2 children
        children = db.children_of(parent_id)
        assert len(children) == 2

        # Neither child has won yet - parent should still be standard tier.
        assert db.get_agent(parent_id).tier == "standard"

        pop.handle_win(children[0].id, pnl=5.0)
        assert db.get_agent(parent_id).tier == "standard"  # only 1 of 2 so far

        pop.handle_win(children[1].id, pnl=5.0)
        assert db.get_agent(parent_id).tier == "professional"
        assert db.get_agent(children[0].id).tier == "professional"
        assert db.get_agent(children[1].id).tier == "professional"
