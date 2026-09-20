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
from strategy.genome import Genome, genome_distance

log = logging.getLogger(__name__)

# Small nudge added to a candidate's backtest score in _pick_best, weighted
# by how far it sits (0.0=identical, ~1.0=opposite on every field) from its
# nearest already-alive neighbor. fitness_score commonly spans tens of
# points, so this is meant to break ties/near-ties toward genuine diversity,
# not to override a real fitness edge - a genome-quality review found
# repeated near-duplicate clone clusters in the population and traced part
# of the cause to _pick_best having zero awareness of what's already alive.
_DIVERSITY_BONUS_WEIGHT = 2.0


def _idle_hours(agent: AgentRow) -> float:
    created = datetime.fromisoformat(agent.created_at)
    return (datetime.now(timezone.utc) - created).total_seconds() / 3600.0


def _split_train_validation(
    candles: list[dict], funding: list[tuple[int, float, float]], train_frac: float = 0.7,
) -> tuple[list[dict], list[tuple[int, float, float]], list[dict], list[tuple[int, float, float]]]:
    """Chronological (not random - avoids any lookahead) split of the
    recent-window backtest data into an earlier 'train' slice and a later
    'validation' slice. Nothing is actually FIT to the train slice (no
    genome parameter comes from optimizing against this data - candidates
    are mutated/crossed-over/random), so the risk here isn't classic
    overfitting; it's SELECTION bias: among several independently-generated
    candidates, picking whichever scored best on one full window can just
    reward a lucky fluke on that specific window. Scoring candidates on a
    held-out LATER slice they weren't picked to fit approximates a genuine
    walk-forward check instead. Returns ([], [], [], []) unchanged-shape
    empty validation lists if there isn't enough data to split meaningfully
    - callers should fall back to the full window in that case."""
    n = len(candles)
    if n < 100:
        return candles, funding, [], []
    cut = int(n * train_frac)
    train_candles, validation_candles = candles[:cut], candles[cut:]
    if not validation_candles:
        return candles, funding, [], []
    split_ms = validation_candles[0]["t"]
    train_funding = [f for f in funding if f[0] < split_ms]
    validation_funding = [f for f in funding if f[0] >= split_ms]
    return train_candles, train_funding, validation_candles, validation_funding


def _family(genome: Genome) -> str:
    """Strategy-family bucket for the diversity floor - a simple heuristic
    combining pullback depth, risk:reward shape, trend-strictness, and
    volume-conviction sensitivity into a composite key, not a rigorous
    clustering. A single-axis (pullback-depth only) version let a clone
    family dominate entirely within one of only two buckets undetected - a
    genome-quality review flagged this as a real cause of population
    diversity collapse; a 3-axis follow-up review flagged that even that
    version could still let genomes with wildly different oscillator/volume
    logic collapse into the same family. Good enough to stop one lineage's
    style from monopolizing every active-trader slot before a genuinely
    different style gets a shot."""
    pullback = "deep-pullback" if genome.rsi_oversold <= 25 else "shallow-pullback"
    ratio = genome.take_profit_pct / genome.stop_loss_pct if genome.stop_loss_pct else 0.0
    if ratio < 3.0:
        rr = "tight-rr"
    elif ratio < 5.0:
        rr = "mid-rr"
    else:
        rr = "wide-rr"
    if genome.min_adx <= 17.0:
        trend = "loose-trend-filter"
    elif genome.min_adx <= 24.0:
        trend = "mid-trend-filter"
    else:
        trend = "strict-trend-filter"
    conviction = "low-conviction" if genome.volume_spike_threshold <= 2.0 else "high-conviction"
    return f"{pullback}/{rr}/{trend}/{conviction}"


class Population:
    def __init__(self, db: Database, config: Config, rng: random.Random | None = None,
                 backtest_candles: list[dict] | None = None,
                 backtest_funding: list[tuple[int, float, float]] | None = None,
                 regime_windows: list[tuple[list[dict], list[tuple[int, float, float]]]] | None = None,
                 htf_candles: list[dict] | None = None):
        self.db = db
        self.config = config
        self.rng = rng or random.Random()
        # Several separate, non-overlapping older historical windows (see
        # main.py) each covering a different market regime - used alongside
        # the recent-window score so a candidate genome that only works in
        # whatever regime just happened doesn't win by default. Empty/
        # omitted falls back to today's single-window-only behavior (e.g.
        # the extra fetch failed, or in tests).
        self.regime_windows = regime_windows or []
        self.set_backtest_window(backtest_candles, backtest_funding, htf_candles)
        # Counts calls to rank_and_enforce (~1 per trading cycle) - used to
        # pace forced active-trader rotation without coupling Population to
        # the orchestrator's own cycle counter.
        self._rank_enforce_calls = 0

    def set_backtest_window(self, candles: list[dict] | None,
                             funding: list[tuple[int, float, float]] | None,
                             htf_candles: list[dict] | None = None) -> None:
        """Sets the recent-window backtest data AND recomputes the train/
        validation split used by _pick_best - the single place this should
        happen, so orchestrator's periodic refresh (see
        engine/orchestrator.py::_refresh_backtest_window) can't update one
        without the other going stale. `htf_candles` (optional -
        higher-timeframe candles covering the same window) lets
        backtest_genome backtest the HTF trend filter for real instead of
        treating it as neutral - see backtest/engine.py's module docstring."""
        self.backtest_candles = candles
        self.backtest_funding = funding or []
        self.htf_candles = htf_candles
        (self.train_candles, self.train_funding,
         self.validation_candles, self.validation_funding) = _split_train_validation(
            self.backtest_candles or [], self.backtest_funding,
        )

    def _pick_best(self, candidates: list[Genome]) -> Genome:
        """Backtest each candidate genome and keep the best-scoring one.
        Falls back to the first candidate untouched if backtesting is
        disabled or no historical data was supplied (e.g. in tests) - never
        blocks agent creation.

        Scores against the held-out VALIDATION slice (the later ~30% of the
        recent window) rather than the full window when there's enough data
        to split - see _split_train_validation for why. Falls back to the
        full window when there isn't (e.g. a short BACKTEST_LOOKBACK_HOURS).

        Also adds a small diversity bonus (_DIVERSITY_BONUS_WEIGHT) based on
        each candidate's distance from its nearest already-alive genome, so
        a candidate that's a near-duplicate of an existing agent doesn't win
        by default over an equally-fit but genuinely distinct alternative."""
        if not self.config.backtest_enabled or not self.backtest_candles or len(candidates) == 1:
            return candidates[0]
        score_candles = self.validation_candles or self.backtest_candles
        score_funding = self.validation_funding if self.validation_candles else self.backtest_funding
        alive_genomes = [Genome.from_dict(a.genome) for a in self.db.list_alive_agents()]
        best_genome, best_score = candidates[0], None
        for genome in candidates:
            result = backtest_genome(genome, score_candles, score_funding,
                                      starting_balance=self.config.starting_paper_balance,
                                      htf_candles=self.htf_candles)
            score = fitness_score(result) + self._regime_consistency_score(genome)
            if alive_genomes:
                nearest = min(genome_distance(genome, other) for other in alive_genomes)
                score += nearest * _DIVERSITY_BONUS_WEIGHT
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
                # A real (not just probable) shot at reinstating a genuinely
                # all-time-best genome's lineage - strategy_shares below is
                # always mutated before reuse, so a proven peak genome
                # otherwise only ever gets tested as a derivative of itself,
                # never itself again. Lightly mutated (not used bit-for-bit)
                # so repeated draws can't stamp out a second exact copy of an
                # already-alive agent - see config.py's
                # hall_of_fame_exact_clone_rate comment. Still screened by
                # _pick_best against held-out data below, not committed blind.
                hof = (self.db.sample_hall_of_fame_genome()
                       if self.rng.random() < self.config.hall_of_fame_exact_clone_rate else None)
                if hof:
                    candidates.append(Genome.from_dict(hof).mutate(self.rng, mutation_rate=0.1))
                    continue
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
        top-fitness agents, excluding the one that just won. Professional-
        tier agents (a lineage proven across 2 winning generations, not just
        one lucky trade) sort ahead of standard-tier ones regardless of raw
        fitness - previously `tier` was purely cosmetic (set on promotion but
        never read anywhere outside the dashboard), so a lucky one-trade
        standard agent could crowd out a genuinely proven lineage here."""
        alive = [a for a in self.db.list_alive_agents() if a.id != exclude_id]
        if not alive:
            return None
        top = sorted(alive, key=lambda a: (a.tier == "professional", a.fitness), reverse=True)[:10]
        partner = self.rng.choice(top)
        return Genome.from_dict(partner.genome)

    def revalidate_top_agents(self) -> int:
        """Re-backtests the top-fitness alive agents' OWN genomes against
        `self.backtest_candles`/`self.backtest_funding` (the same
        recent-window data used to screen new agents - kept fresh by
        orchestrator's periodic refresh) and records the result via
        `db.set_revalidation`. Flags a drifted veteran on the dashboard
        instead of silently trusting a genome that proved itself once,
        potentially a long time ago under different market conditions.
        Never touches the agent's stored status or fitness column - only
        applies a ranking-only penalty when rank_and_enforce next picks
        active/live traders (see `_effective_rank_score`). Returns how many
        agents were checked."""
        if not self.config.backtest_enabled or not self.backtest_candles:
            return 0
        alive = self.db.list_alive_agents()
        top = sorted(alive, key=lambda a: a.fitness, reverse=True)[: self.config.revalidation_agent_limit]
        for agent in top:
            genome = Genome.from_dict(agent.genome)
            result = backtest_genome(genome, self.backtest_candles, self.backtest_funding,
                                      starting_balance=self.config.starting_paper_balance,
                                      htf_candles=self.htf_candles)
            score = fitness_score(result)
            self.db.set_revalidation(agent.id, score)
        return len(top)

    def council_genomes(self, exclude_id: int) -> list[Genome]:
        """The current top-fitness alive agents' own genomes (excluding the
        one asking) - an ensemble "second opinion" panel for an ambiguous
        signal. See strategy/signals.py::council_consult. Professional-tier
        agents sort ahead of standard-tier ones regardless of raw fitness -
        see _pick_breeding_partner for why."""
        alive = [a for a in self.db.list_alive_agents() if a.id != exclude_id]
        top = sorted(alive, key=lambda a: (a.tier == "professional", a.fitness), reverse=True)[: self.config.council_size]
        return [Genome.from_dict(a.genome) for a in top]

    def _maybe_record_hall_of_fame(self, agent: AgentRow) -> None:
        """Snapshots agent's EXACT current genome into hall_of_fame if its
        fitness is a new all-time high - unlike strategy_shares (always
        mutated before reuse), this is the only place a genuinely
        best-ever genome survives verbatim. Called both on win (so a
        long-lived agent's peak is captured while it's still alive) and
        right before death (so do-or-die killing it doesn't erase a peak
        that hasn't been captured yet - this is the critical hook, since a
        professional-tier agent dies on its next loss exactly like anyone
        else, and its exact genome would otherwise be gone forever)."""
        if agent.fitness > self.db.get_max_hall_of_fame_fitness():
            self.db.record_hall_of_fame(
                agent.id, agent.genome, agent.win_streak, agent.total_pnl,
                agent.fitness, reason="new_all_time_high_fitness",
            )
            log.info("Agent %d fitness %.2f is a new all-time high - preserved in hall_of_fame",
                      agent.id, agent.fitness)

    def handle_win(self, agent_id: int, pnl: float) -> None:
        self.db.record_win(agent_id, pnl)
        agent = self.db.get_agent(agent_id)
        log.info("Agent %d WON trade (pnl=%.2f, streak=%d) - spawning %d children",
                  agent_id, pnl, agent.win_streak, self.config.children_per_win)
        self._maybe_record_hall_of_fame(agent)

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
        # Captured BEFORE the kill so this reflects the agent's peak
        # performance from its winning streak, not the losing trade that's
        # about to drag its fitness down - that peak is exactly what's
        # worth preserving.
        agent = self.db.get_agent(agent_id)
        self._maybe_record_hall_of_fame(agent)
        self.db.record_loss_and_kill(agent_id, pnl)
        log.info("Agent %d LOST trade (pnl=%.2f) - do or die: eliminated", agent_id, pnl)

    def _check_promotion(self, parent_id: int) -> None:
        if parent_id is None:
            return
        # Only alive children count - a lineage that already died back down
        # (won once, then lost its next do-or-die trade) isn't the "proven,
        # self-sustaining lineage" this promotion is meant to reward.
        children = [c for c in self.db.children_of(parent_id) if c.status == "alive"]
        winners = [c for c in children if c.wins >= 1]
        if len(children) >= 2 and len(winners) >= 2:
            parent = self.db.get_agent(parent_id)
            if parent.tier != "professional":
                log.info("Both children of agent %d are winning - promoting lineage to professional tier", parent_id)
                self.db.set_tier(parent_id, "professional")
                for c in winners[:2]:
                    self.db.set_tier(c.id, "professional")

    # ---- ranking / population control ----

    def _effective_rank_score(self, agent: AgentRow) -> float:
        """Ranking-only score used for active/live-trader slot selection -
        identical to agent.fitness except a drifted agent (revalidation
        score below revalidation_drift_threshold) is penalized proportional
        to how far past the threshold it drifted. Never written back to the
        DB and never used for the population-cap cull or promotion checks -
        the stored `fitness`/dashboard history are untouched by this; only
        which agents get a trading slot this cycle is affected."""
        score = agent.revalidation_score
        if score is not None and score < self.config.revalidation_drift_threshold:
            drift_severity = self.config.revalidation_drift_threshold - score
            return agent.fitness - drift_severity * self.config.revalidation_drift_penalty_factor
        return agent.fitness

    def rank_and_enforce(self) -> dict:
        alive = self.db.list_alive_agents()

        # Cull down to the population cap, protecting fresh (0-trade) agents -
        # unless they've sat idle so long (regime filters never admitting a
        # trade) that they're just squatting a slot; those become cullable too.
        overflow = len(alive) - self.config.population_cap
        if overflow > 0:
            cullable = sorted(
                [a for a in alive if a.trades_count > 0
                 or _idle_hours(a) > self.config.max_idle_hours_before_cull],
                key=lambda a: a.fitness,
            )
            for agent in cullable[:overflow]:
                reason = "culled: population cap exceeded" if agent.trades_count > 0 else \
                    "culled: idle too long with no trades (population cap exceeded)"
                self.db.kill_agent(agent.id, reason)
            killed_ids = {a.id for a in cullable[:overflow]}
            alive = [a for a in alive if a.id not in killed_ids]

        ranked = sorted(alive, key=self._effective_rank_score, reverse=True)
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

        # Guarantee a few slots for the LONGEST-benched untested agents -
        # distinct from the newcomer slots above (which favor the NEWEST
        # 0-trade agents). Without this, an agent that's been sitting at
        # 0 trades for a long time (not new, just never fitness-ranked
        # into a slot) has no path back in except outliving everyone via
        # the idle-cull timeout - this gives it an actual shot first.
        if len(top_traders) < len(ranked) and self.config.guaranteed_longest_benched_slots > 0:
            top_ids = {a.id for a in top_traders}
            longest_benched = sorted(
                (a for a in ranked if a.trades_count == 0 and a.id not in top_ids),
                key=lambda a: a.created_at,
            )[: self.config.guaranteed_longest_benched_slots]
            if longest_benched:
                keep = sorted(top_traders, key=lambda a: a.fitness, reverse=True)[
                    : len(top_traders) - len(longest_benched)
                ]
                top_traders = keep + longest_benched

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

        # Periodic forced rotation: every forced_rotation_every_n_cycles
        # calls, swap the single lowest-fitness active trader for one
        # random benched 0-trade agent. Guarantees no slot is held forever
        # purely because nothing ever forces a re-test of the bench -
        # touches exactly one slot, so a genuinely weak agent just gets
        # crowded back out by the normal fitness sort next cycle anyway.
        self._rank_enforce_calls += 1
        if (self.config.forced_rotation_every_n_cycles > 0
                and self._rank_enforce_calls % self.config.forced_rotation_every_n_cycles == 0
                and len(top_traders) < len(ranked)):
            top_ids = {a.id for a in top_traders}
            benched = [a for a in ranked if a.trades_count == 0 and a.id not in top_ids]
            if benched and top_traders:
                pick = self.rng.choice(benched)
                weakest = min(top_traders, key=lambda a: a.fitness)
                top_traders = [a for a in top_traders if a.id != weakest.id] + [pick]
                log.info("Forced rotation: benched agent %d swapped in for active trader %d",
                          pick.id, weakest.id)

        self.db.set_active_traders({a.id for a in top_traders})

        # Live capital only ever goes to the most proven subset of the
        # already-proven top traders - see trading/live_executor.py.
        # Professional-tier agents (survived past a single-trade do-or-die
        # noise floor into a 2-winning-generation-proven lineage) fill live
        # slots first, regardless of raw fitness; standard-tier agents only
        # backfill remaining slots if there aren't enough professional-tier
        # agents yet, so live trading isn't left completely empty this early
        # in the population's life (count_professional() is currently 0 - no
        # lineage has reached professional tier yet). Each tier group is
        # itself sorted by effective (drift-penalized) score rather than
        # top_traders' existing order, since newcomer/longest-benched/
        # diversity-floor slots can append agents out of fitness order - this
        # ensures live capital specifically avoids a drifted agent even if
        # one ended up elsewhere in top_traders.
        if self.config.is_live():
            professional = sorted(
                (a for a in top_traders if a.tier == "professional"),
                key=self._effective_rank_score, reverse=True,
            )
            standard = sorted(
                (a for a in top_traders if a.tier != "professional"),
                key=self._effective_rank_score, reverse=True,
            )
            top_live = (professional + standard)[: self.config.live_active_trader_count]
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
