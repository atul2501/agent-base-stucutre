"""Ties market data, signals, paper execution, and the population lifecycle
together into one repeating cycle. Swing-trading timeframe, so cycles run
every `cycle_seconds` (default 15 min) rather than tick-by-tick.

The population always trades exactly one token (config.token) - see
strategy/genome.py and README.md "one token at a time" design.
"""
from __future__ import annotations

import logging

from config import Config
from db.database import Database
from market.hyperliquid_client import HyperliquidClient, MarketSnapshot
from agents.population import Population
from reasoning import ollama_advisor
from strategy.genome import Genome
from strategy.signals import build_features, compute_htf_trend, evaluate_entry, evaluate_exit
from trading.live_executor import LiveExecutor
from trading.paper_executor import close_paper_position, open_paper_position

log = logging.getLogger(__name__)


class Orchestrator:
    def __init__(self, db: Database, hl: HyperliquidClient, population: Population, config: Config,
                 live: LiveExecutor | None = None):
        self.db = db
        self.hl = hl
        self.population = population
        self.config = config
        self.live = live
        self.prev_open_interest: float | None = None
        self.htf_trend_up: bool | None = None
        self.cycle = 0

    def _fetch_snapshot(self) -> MarketSnapshot | None:
        try:
            return self.hl.get_snapshot(self.config.token, self.config.timeframe)
        except Exception as e:
            log.warning("Failed to fetch market data for %s: %s", self.config.token, e)
            return None

    def _fetch_htf_trend(self) -> bool | None:
        try:
            htf_snap = self.hl.get_snapshot(self.config.token, self.config.higher_timeframe)
            return compute_htf_trend(htf_snap.candles)
        except Exception as e:
            log.warning("Failed to fetch higher-timeframe (%s) data: %s - treating as neutral this cycle",
                        self.config.higher_timeframe, e)
            return None

    def _process_exits(self, snap: MarketSnapshot) -> None:
        for agent in self.db.list_alive_agents():
            trade_row = self.db.get_open_trade(agent.id)
            if trade_row is None:
                continue
            genome = Genome.from_dict(agent.genome)

            features = build_features(snap, genome, self.prev_open_interest, self.htf_trend_up)
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

    def _process_entries(self, snap: MarketSnapshot) -> None:
        for agent in self.db.list_active_traders():
            if self.db.get_open_trade(agent.id) is not None:
                continue
            genome = Genome.from_dict(agent.genome)

            features = build_features(snap, genome, self.prev_open_interest, self.htf_trend_up)
            signal = evaluate_entry(genome, features)

            if signal.ambiguous and self.config.ollama_enabled:
                signal = ollama_advisor.consult(genome, features, signal)

            if signal.action == "hold":
                continue

            # Scale position size by how strongly the signal was confirmed -
            # a barely-passing setup risks less than a strongly-confirmed
            # one, instead of both risking the same genome-fixed %. Floored
            # at 30% of the genome's intended size so a weak-but-approved
            # signal isn't shrunk to near nothing.
            effective_size_pct = genome.position_size_pct * max(0.3, min(1.0, signal.confidence))
            fill_price, size, notional = open_paper_position(
                agent.balance, features.mid_price, signal.action, effective_size_pct
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

    def _sync_live_exposure(self, snap: MarketSnapshot) -> None:
        """Real money mirror of the paper population's consensus. Only the
        current top `live_active_trader_count` agents (db.list_live_traders)
        count; each contributes one fixed-size 'slot' toward the net
        long/short direction. See trading/live_executor.py for why this is
        one aggregate position rather than N independent ones."""
        if self.live is None:
            return

        live_traders = self.db.list_live_traders()
        n_long = 0
        n_short = 0
        for agent in live_traders:
            trade_row = self.db.get_open_trade(agent.id)
            if trade_row is None:
                continue
            if trade_row["coin"] != self.config.token:
                continue
            if trade_row["side"] == "long":
                n_long += 1
            else:
                n_short += 1

        net = n_long - n_short
        per_slot = self.config.live_max_total_notional_usd / max(1, self.config.live_active_trader_count)
        desired_notional = min(abs(net) * per_slot, self.config.live_max_total_notional_usd)
        desired_side = "long" if net > 0 else ("short" if net < 0 else None)
        desired_size = (desired_notional / snap.mid_price) if desired_side else 0.0

        try:
            results = self.live.adjust_to(self.config.token, desired_side, desired_size, snap.sz_decimals)
        except Exception as e:
            log.exception("Live exposure sync failed - leaving real position unchanged")
            self.db.record_live_order(self.config.token, "sync_error", desired_side or "flat",
                                       desired_notional, None, "error", str(e)[:300])
            return

        for r in results:
            self.db.record_live_order(self.config.token, r["action"], r["side"],
                                       desired_notional, r["fill_price"], r["status"], r["detail"])
            if r["status"] == "filled":
                log.info("LIVE order: %s %s %s (fill=%s)", r["action"], r["side"], self.config.token, r["fill_price"])
            else:
                log.error("LIVE order FAILED: %s %s %s - %s", r["action"], r["side"], self.config.token, r["detail"])

        self.db.set_live_position(self.config.token, desired_side, desired_size, desired_notional)

    def run_cycle(self) -> None:
        self.cycle += 1
        snap = self._fetch_snapshot()
        if snap is None:
            log.warning("No market data available this cycle - skipping")
            return

        self.htf_trend_up = self._fetch_htf_trend()
        self._process_exits(snap)
        self.population.refill_if_below_floor()
        stats = self.population.rank_and_enforce()
        self._process_entries(snap)
        self._sync_live_exposure(snap)

        self.db.record_population_cycle(self.cycle, **stats)
        self.prev_open_interest = snap.open_interest
        # Lets the dashboard show unrealized PnL on open trades instead of a
        # blank "-" for however long a position stays open (max_hold_hours
        # can be up to 96h) - see dashboard/server.py's /api/trades.
        self.db.set_meta("last_price", str(snap.mid_price))

        log.info(
            "Cycle %d done | %s | alive=%d active_traders=%d professional=%d best_agent=%s best_pnl=%s",
            self.cycle, "LIVE" if self.config.is_live() else "paper",
            stats["alive_count"], stats["active_trader_count"],
            stats["professional_count"], stats["best_agent_id"], stats["best_total_pnl"],
        )
