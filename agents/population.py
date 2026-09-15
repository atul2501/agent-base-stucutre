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

import json
import logging
import random
from datetime import datetime, timezone
from pathlib import Path

from backtest.engine import backtest_genome, fitness_score
from config import Config
from db.database import AgentRow, Database
from strategy.genome import Genome

log = logging.getLogger(__name__)


def _idle_hours(agent: AgentRow) -> float:
    created = datetime.fromisoformat(agent.created_at)
    return (datetime.now(timezone.utc) - created).total_seconds() / 3600.0


def _family(genome: Genome) -> str:
    """Coarse strategy-family split for the diversity floor - a simple
    heuristic on how deep a pullback this agent waits for, not a rigorous
    clustering. Good enough to stop one lineage's style from monopolizing
    every active-trader slot before a genuinely different style gets a shot."""
    return "deep-value" if genome.rsi_oversold <= 25 else "momentum-moderate"


class Population:
    def __init__(self, db: Database, config: Config, rng: random.Random | None = None,
                 backtest_candles: list[dict] | None = None,
                 backtest_funding: list[tuple[int, float, float]] | None = None,
                 regime_windows: list[tuple[list[dict], list[tuple[int, float, float]]]] | None = None):
        self.db = db
        self.config = config
        self.rng = rng or random.Random()
        self.backtest_candles = backtest_candles
        self.backtest_funding = backtest_funding or []
        # Several separate, non-overlapping older historical windows (see
        # main.py) each covering a different market regime - used alongside
        # backtest_candles (the recent-window score) so a candidate genome
        # that only works in whatever regime just happened doesn't win by
        # default. Empty/omitted falls back to today's single-window-only
        # behavior (e.g. the extra fetch failed, or in tests).
        self.regime_windows = regime_windows or []

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
            score = fitness_score(result) + self._regime_consistency_score(genome)
            if best_score is None or score > best_score:
                best_genome, best_score = genome, score
        return best_genome

    def _regime_consistency_score(self, genome: Genome) -> float:
        """Backtests one genome across every historical regime segment and
        combines the scores to reward consistency, not just a high average -
        `mean - 0.5*(mean - worst)` explicitly penalizes a genome that only
        excels in one segment and craters in another, which a plain average
        would hide. Returns 0.0 (neutral - no effect on ranking) if no
        regime windows were supplied."""
        if not self.regime_windows:
            return 0.0
        scores = [
            fitness_score(backtest_genome(genome, candles, funding,
                                           starting_balance=self.config.starting_paper_balance))
            for candles, funding in self.regime_windows
        ]
        mean_score = sum(scores) / len(scores)
        worst_score = min(scores)
        return mean_score - 0.5 * (mean_score - worst_score)

    # ---- seeding ----

    def _load_snapshot_genomes(self) -> list[Genome]:
        """Hand-picked genomes saved with `python3 main.py --snapshot-best`
        (see main.py) - a head start for a FRESH population instead of pure
        random genomes. Only used by seed_if_empty (below), so an
        already-running population is never touched by dropping files into
        this folder. Silently skips a file that fails to parse or was
        snapshotted for a different token (this population only ever trades
        one - see README "one token at a time") rather than blocking startup."""
        snap_dir = Path(self.config.snapshot_dir)
        if not snap_dir.is_dir():
            return []
        genomes = []
        for path in sorted(snap_dir.glob("*.json")):
            try:
                data = json.loads(path.read_text())
                genome_dict = data["genome"]
                if genome_dict.get("coin") != self.config.token:
                    log.warning("Skipping snapshot %s - built for %s, current token is %s",
                                path.name, genome_dict.get("coin"), self.config.token)
                    continue
                genomes.append(Genome.from_dict(genome_dict))
            except Exception:
                log.exception("Skipping unreadable snapshot file %s", path)
        return genomes

    def seed_if_empty(self) -> None:
        if self.db.count_alive() > 0:
            return
        snapshot_genomes = self._load_snapshot_genomes()
        for genome in snapshot_genomes:
            self.db.create_agent(genome.to_dict(), balance=self.config.starting_paper_balance)
        if snapshot_genomes:
            log.info("Seeded %d agent(s) from snapshot library (%s)",
                      len(snapshot_genomes), self.config.snapshot_dir)

        remaining = max(0, self.config.initial_population - len(snapshot_genomes))
        log.info("Seeding %d more random-candidate agent(s) for %s (initial population %d)",
                  remaining, self.config.token, self.config.initial_population)
        for _ in range(remaining):
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

    def _pick_breeding_partner(self, exclude_id: int) -> Genome | None:
        """A second parent for crossover - sampled from the current
        top-fitness agents, excluding the one that just won."""
        alive = [a for a in self.db.list_alive_agents() if a.id != exclude_id]
        if not alive:
            return None
        top = sorted(alive, key=lambda a: a.fitness, reverse=True)[:10]
        partner = self.rng.choice(top)
        return Genome.from_dict(partner.genome)

    def revalidate_top_agents(self) -> int:
        """Re-backtests the top-fitness alive agents' OWN genomes against
        `self.backtest_candles`/`self.backtest_funding` (the same
        recent-window data used to screen new agents - kept fresh by
        orchestrator's periodic refresh) and records the result via
        `db.set_revalidation`. Purely informational - flags a drifted
        veteran on the dashboard instead of silently trusting a genome that
        proved itself once, potentially a long time ago under different
        market conditions. Never touches status, fitness, or which agents
        are active/live traders. Returns how many agents were checked."""
        if not self.config.backtest_enabled or not self.backtest_candles:
            return 0
        alive = self.db.list_alive_agents()
        top = sorted(alive, key=lambda a: a.fitness, reverse=True)[: self.config.revalidation_agent_limit]
        for agent in top:
            genome = Genome.from_dict(agent.genome)
            result = backtest_genome(genome, self.backtest_candles, self.backtest_funding,
                                      starting_balance=self.config.starting_paper_balance)
            score = fitness_score(result)
            self.db.set_revalidation(agent.id, score)
        return len(top)

    def council_genomes(self, exclude_id: int) -> list[Genome]:
        """The current top-fitness alive agents' own genomes (excluding the
        one asking) - an ensemble "second opinion" panel for an ambiguous
        signal. See strategy/signals.py::council_consult."""
        alive = [a for a in self.db.list_alive_agents() if a.id != exclude_id]
        top = sorted(alive, key=lambda a: a.fitness, reverse=True)[: self.config.council_size]
        return [Genome.from_dict(a.genome) for a in top]

    def handle_win(self, agent_id: int, pnl: float) -> None:
        self.db.record_win(agent_id, pnl)
        agent = self.db.get_agent(agent_id)
        log.info("Agent %d WON trade (pnl=%.2f, streak=%d) - spawning %d children",
                  agent_id, pnl, agent.win_streak, self.config.children_per_win)

        parent_genome = Genome.from_dict(agent.genome)
        for _ in range(self.config.children_per_win):
            candidates = []
            for _ in range(self.config.backtest_candidates):
                partner_genome = (
                    self._pick_breeding_partner(agent.id)
                    if self.rng.random() < self.config.crossover_probability else None
                )
                if partner_genome is not None:
                    candidates.append(parent_genome.crossover(partner_genome, self.rng).mutate(self.rng, mutation_rate=0.15))
                else:
                    candidates.append(parent_genome.mutate(self.rng))
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

        # Diversity floor: make sure at least one agent from each coarse
        # strategy family is active, if any exist at all in the alive
        # population - evicting the current weakest active trader to make
        # room. Without this, one early lucky lineage could occupy every
        # slot and a genuinely different approach might never get tried.
        if self.config.diversity_floor_enabled and len(top_traders) < len(ranked):
            top_ids = {a.id for a in top_traders}
            present_families = {_family(Genome.from_dict(a.genome)) for a in top_traders}
            all_families = {_family(Genome.from_dict(a.genome)) for a in ranked}
            for fam in all_families - present_families:
                candidate = next(
                    (a for a in ranked if a.id not in top_ids and _family(Genome.from_dict(a.genome)) == fam),
                    None,
                )
                if candidate is None:
                    continue
                lowest = min(top_traders, key=lambda a: a.fitness)
                top_traders = [a for a in top_traders if a.id != lowest.id] + [candidate]
                top_ids = {a.id for a in top_traders}

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
