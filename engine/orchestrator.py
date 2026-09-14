"""Ties market data, signals, paper execution, and the population lifecycle
together into one repeating cycle. Swing-trading timeframe, so cycles run
every `cycle_seconds` (default 15 min) rather than tick-by-tick.
"""
from __future__ import annotations

import logging

from config import Config
from db.database import Database
from market.hyperliquid_client import HyperliquidClient, MarketSnapshot
from agents.population import Population
from reasoning import ollama_advisor
from strategy.genome import Genome
from strategy.signals import build_features, evaluate_entry, evaluate_exit
from trading.paper_executor import close_paper_position, open_paper_position

log = logging.getLogger(__name__)


class Orchestrator:
    def __init__(self, db: Database, hl: HyperliquidClient, population: Population, config: Config):
        self.db = db
        self.hl = hl
        self.population = population
        self.config = config
        self.prev_open_interest: dict[str, float] = {}
        self.cycle = 0

    def _fetch_snapshots(self) -> dict[str, MarketSnapshot]:
        snapshots: dict[str, MarketSnapshot] = {}
        for coin in self.config.symbols:
            try:
                snapshots[coin] = self.hl.get_snapshot(coin, self.config.timeframe)
            except Exception as e:
                log.warning("Failed to fetch market data for %s: %s", coin, e)
        return snapshots

    def _process_exits(self, snapshots: dict[str, MarketSnapshot]) -> None:
        for agent in self.db.list_alive_agents():
            genome = Genome.from_dict(agent.genome)
            snap = snapshots.get(genome.coin)
            if snap is None:
                continue
            trade_row = self.db.get_open_trade(agent.id)
            if trade_row is None:
                continue

            features = build_features(snap, genome, self.prev_open_interest.get(genome.coin))
            outcome = evaluate_exit(genome, trade_row, features)
            if outcome is None:
                continue

            result, reason = outcome
            exit_price, pnl = close_paper_position(
                trade_row["entry_price"], features.mid_price, trade_row["size"], trade_row["side"]
            )
            self.db.close_trade(trade_row["id"], exit_price, pnl, result, reason)

            if result == "win":
                self.population.handle_win(agent.id, pnl)
            else:
                self.population.handle_loss(agent.id, pnl)

    def _process_entries(self, snapshots: dict[str, MarketSnapshot]) -> None:
        for agent in self.db.list_active_traders():
            if self.db.get_open_trade(agent.id) is not None:
                continue
            genome = Genome.from_dict(agent.genome)
            snap = snapshots.get(genome.coin)
            if snap is None:
                continue

            features = build_features(snap, genome, self.prev_open_interest.get(genome.coin))
            signal = evaluate_entry(genome, features)

            if signal.ambiguous and self.config.ollama_enabled:
                signal = ollama_advisor.consult(genome, features, signal)

            if signal.action == "hold":
                continue

            fill_price, size, notional = open_paper_position(
                agent.balance, features.mid_price, signal.action, genome.position_size_pct
            )
            if signal.action == "long":
                stop_loss = fill_price * (1 - genome.stop_loss_pct / 100)
                take_profit = fill_price * (1 + genome.take_profit_pct / 100)
            else:
                stop_loss = fill_price * (1 + genome.stop_loss_pct / 100)
                take_profit = fill_price * (1 - genome.take_profit_pct / 100)

            self.db.open_trade(
                agent.id, genome.coin, signal.action, fill_price, size, notional,
                stop_loss, take_profit, entry_reason="; ".join(signal.reasons),
            )
            log.info("Agent %d opened %s %s @ %.4f (confidence=%.2f) - %s",
                      agent.id, signal.action.upper(), genome.coin, fill_price,
                      signal.confidence, signal.reasons[0] if signal.reasons else "")

    def run_cycle(self) -> None:
        self.cycle += 1
        snapshots = self._fetch_snapshots()
        if not snapshots:
            log.warning("No market data available this cycle - skipping")
            return

        self._process_exits(snapshots)
        self.population.refill_if_below_floor()
        stats = self.population.rank_and_enforce()
        self._process_entries(snapshots)

        self.db.record_population_cycle(self.cycle, **stats)
        self.prev_open_interest = {coin: snap.open_interest for coin, snap in snapshots.items()}

        log.info(
            "Cycle %d done | alive=%d active_traders=%d professional=%d best_agent=%s best_pnl=%s",
            self.cycle, stats["alive_count"], stats["active_trader_count"],
            stats["professional_count"], stats["best_agent_id"], stats["best_total_pnl"],
        )
