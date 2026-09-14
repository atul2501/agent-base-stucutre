"""The evolutionary lifecycle: win -> survive + spawn children, lose -> die.

- Win: agent survives, balance grows by the pnl, and it spawns
  `children_per_win` mutated copies of itself into the population.
- Loss: agent dies immediately (do-or-die).
- Promotion: once BOTH children of a parent have won at least one trade of
  their own, that parent + those two children are promoted to the
  "professional" tier (a proven, self-sustaining lineage).
- Sharing: an agent that reaches an 8-win streak (configurable) has its
  genome recorded in `strategy_shares`; new agents spawned to backfill the
  population are preferentially cloned from shared genomes instead of pure
  random, so proven strategies propagate through the population.
- Population cap: at most `population_cap` agents stay alive; when a win's
  new children would exceed it, the lowest-fitness agents (with at least one
  completed trade, so newborns get a grace period) are culled.
- Active traders: only the top `active_trader_count` alive agents by fitness
  are allowed to open new positions each cycle; the rest keep tracking but
  wait their turn.
- Backtest pre-screening: every newly-born agent (seeded, floor refill, or
  a winner's children) is chosen from several candidate genomes by real
  historical performance (backtest/engine.py), not committed blind. This
  doesn't touch the live do-or-die mechanic at all - it only changes what
  genome an agent starts its live life with, so agents entering the real
  gauntlet start from a better-than-random point.
"""
from __future__ import annotations

import logging
import random
from datetime import datetime, timezone

from backtest.engine import backtest_genome, fitness_score
from config import Config
from db.database import AgentRow, Database
from strategy.genome import Genome

log = logging.getLogger(__name__)


def _idle_hours(agent: AgentRow) -> float:
    created = datetime.fromisoformat(agent.created_at)
    return (datetime.now(timezone.utc) - created).total_seconds() / 3600.0


class Population:
    def __init__(self, db: Database, config: Config, rng: random.Random | None = None,
                 backtest_candles: list[dict] | None = None,
                 backtest_funding: list[tuple[int, float, float]] | None = None):
        self.db = db
        self.config = config
        self.rng = rng or random.Random()
        self.backtest_candles = backtest_candles
        self.backtest_funding = backtest_funding or []

    def _pick_best(self, candidates: list[Genome]) -> Genome:
        """Backtest each candidate genome against real recent history and
        keep the best-scoring one. Falls back to the first candidate
        untouched if backtesting is disabled or no historical data was
        supplied (e.g. in tests) - never blocks agent creation."""
        if not self.config.backtest_enabled or not self.backtest_candles or len(candidates) == 1:
            return candidates[0]
        best_genome, best_score = candidates[0], None
        for genome in candidates:
            result = backtest_genome(genome, self.backtest_candles, self.backtest_funding,
                                      starting_balance=self.config.starting_paper_balance)
            score = fitness_score(result)
            if best_score is None or score > best_score:
                best_genome, best_score = genome, score
        return best_genome

    # ---- seeding ----

    def seed_if_empty(self) -> None:
        if self.db.count_alive() > 0:
            return
        log.info("No agents found - seeding initial population of %d for %s",
                  self.config.initial_population, self.config.token)
        for _ in range(self.config.initial_population):
            candidates = [Genome.random(self.config.token, self.config.timeframe, self.rng)
                          for _ in range(self.config.backtest_candidates)]
            genome = self._pick_best(candidates)
            self.db.create_agent(genome.to_dict(), balance=self.config.starting_paper_balance)

    def refill_if_below_floor(self) -> None:
        alive = self.db.count_alive()
        if alive >= self.config.min_population_floor:
            return
        needed = self.config.min_population_floor - alive
        log.info("Population (%d) below floor (%d) - spawning %d replacements",
                  alive, self.config.min_population_floor, needed)
        for _ in range(needed):
            candidates = []
            for _ in range(self.config.backtest_candidates):
                shared = self.db.sample_shared_genome() if self.rng.random() < 0.5 else None
                if shared:
                    candidates.append(Genome.from_dict(shared).mutate(self.rng, mutation_rate=0.3))
                else:
                    candidates.append(Genome.random(self.config.token, self.config.timeframe, self.rng))
            genome = self._pick_best(candidates)
            self.db.create_agent(genome.to_dict(), balance=self.config.starting_paper_balance)

    # ---- trade outcome -> lifecycle ----

    def handle_win(self, agent_id: int, pnl: float) -> None:
        self.db.record_win(agent_id, pnl)
        agent = self.db.get_agent(agent_id)
        log.info("Agent %d WON trade (pnl=%.2f, streak=%d) - spawning %d children",
                  agent_id, pnl, agent.win_streak, self.config.children_per_win)

        parent_genome = Genome.from_dict(agent.genome)
        for _ in range(self.config.children_per_win):
            candidates = [parent_genome.mutate(self.rng) for _ in range(self.config.backtest_candidates)]
            child_genome = self._pick_best(candidates)
            self.db.create_agent(
                child_genome.to_dict(),
                balance=self.config.starting_paper_balance,
                parent_id=agent.id,
                generation=agent.generation + 1,
                tier=agent.tier,
            )

        # This win might be the second of two children needed to promote *its
        # parent's* lineage - not this agent's own (brand-new) children.
        self._check_promotion(agent.parent_id)

        if agent.win_streak > 0 and agent.win_streak % self.config.win_streak_share_threshold == 0:
            log.info("Agent %d hit a %d-win streak - sharing strategy with the population",
                      agent_id, agent.win_streak)
            self.db.record_strategy_share(agent.id, agent.genome, agent.win_streak, agent.total_pnl)

    def handle_loss(self, agent_id: int, pnl: float) -> None:
        self.db.record_loss_and_kill(agent_id, pnl)
        log.info("Agent %d LOST trade (pnl=%.2f) - do or die: eliminated", agent_id, pnl)

    def _check_promotion(self, parent_id: int) -> None:
        if parent_id is None:
            return
        children = self.db.children_of(parent_id)
        winners = [c for c in children if c.wins >= 1]
        if len(children) >= 2 and len(winners) >= 2:
            parent = self.db.get_agent(parent_id)
            if parent.tier != "professional":
                log.info("Both children of agent %d are winning - promoting lineage to professional tier", parent_id)
                self.db.set_tier(parent_id, "professional")
                for c in winners[:2]:
                    self.db.set_tier(c.id, "professional")

    # ---- ranking / population control ----

    def rank_and_enforce(self) -> dict:
        alive = self.db.list_alive_agents()

        # Cull down to the population cap, protecting fresh (0-trade) agents -
        # unless they've sat idle so long (regime filters never admitting a
        # trade) that they're just squatting a slot; those become culuable too.
        overflow = len(alive) - self.config.population_cap
        if overflow > 0:
            culuable = sorted(
                [a for a in alive if a.trades_count > 0
                 or _idle_hours(a) > self.config.max_idle_hours_before_cull],
                key=lambda a: a.fitness,
            )
            for agent in culuable[:overflow]:
                reason = "culled: population cap exceeded" if agent.trades_count > 0 else \
                    "culled: idle too long with no trades (population cap exceeded)"
                self.db.kill_agent(agent.id, reason)
            killed_ids = {a.id for a in culuable[:overflow]}
            alive = [a for a in alive if a.id not in killed_ids]

        ranked = sorted(alive, key=lambda a: a.fitness, reverse=True)
        top_traders = ranked[: self.config.active_trader_count]

        # Guarantee a few slots for the newest untested agents. Without
        # this, once alive count exceeds active_trader_count, a brand-new
        # agent (fitness=0) ties with every other never-traded agent and
        # ranks behind anyone who's ever completed even a marginal trade -
        # it could go permanently benched, never getting a shot to prove
        # itself, purely because older agents haven't died yet.
        if len(top_traders) < len(ranked) and self.config.guaranteed_newcomer_slots > 0:
            top_ids = {a.id for a in top_traders}
            newcomers = sorted(
                (a for a in ranked if a.trades_count == 0 and a.id not in top_ids),
                key=lambda a: a.created_at, reverse=True,
            )[: self.config.guaranteed_newcomer_slots]
            if newcomers:
                keep = sorted(top_traders, key=lambda a: a.fitness, reverse=True)[
                    : len(top_traders) - len(newcomers)
                ]
                top_traders = keep + newcomers

        self.db.set_active_traders({a.id for a in top_traders})

        # Live capital only ever goes to the most proven subset of the
        # already-proven top traders - see trading/live_executor.py.
        if self.config.is_live():
            top_live = top_traders[: self.config.live_active_trader_count]
            self.db.set_live_traders({a.id for a in top_live})
        else:
            self.db.set_live_traders(set())

        best = ranked[0] if ranked else None
        professional_count = self.db.count_professional()
        return {
            "alive_count": len(ranked),
            "active_trader_count": len(top_traders),
            "professional_count": professional_count,
            "best_agent_id": best.id if best else None,
            "best_total_pnl": best.total_pnl if best else None,
        }
