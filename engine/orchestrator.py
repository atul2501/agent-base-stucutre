"""Ties market data, signals, paper execution, and the population lifecycle
together into one repeating cycle, run every `cycle_seconds` (default 60s,
paired with the default 1m `timeframe`) rather than tick-by-tick.

The population always trades exactly one token (config.token) - see
strategy/genome.py and README.md "one token at a time" design.
"""
from __future__ import annotations

import logging
from datetime import datetime, timezone

from config import Config
from db.database import Database, now_iso
from market.hyperliquid_client import HyperliquidClient, MarketSnapshot
from agents.population import Population
from reasoning import ollama_advisor
from strategy.genome import Genome
from strategy.signals import (
    build_features, classify_regime, compute_htf_trend, council_consult, evaluate_entry, evaluate_exit,
)
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
        # Circuit breaker staleness tracking - deliberately in-memory only
        # (not persisted), so a restart gives the price feed a fresh chance
        # rather than carrying over a stale count from before a deploy/
        # restart. The TRIP itself (once it happens) IS persisted - see
        # _check_circuit_breaker.
        self._last_seen_price: float | None = None
        self._stale_price_count = 0

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
            # Approximates total funding paid/received over the hold as the
            # average of the entry-time and exit-time funding rate * notional
            # * hours held - live trading only ever observes the CURRENT rate
            # each cycle, unlike the backtest engine, which has a real
            # historical funding series to sum exactly (see
            # backtest/engine.py). Funding was previously used only as an
            # entry SIGNAL and never actually charged against simulated PnL,
            # which systematically overstated returns for any hold spanning
            # a funding interval.
            opened_at = datetime.fromisoformat(trade_row["opened_at"])
            hours_held = (datetime.now(timezone.utc) - opened_at).total_seconds() / 3600.0
            avg_funding_rate = (trade_row["entry_funding"] + features.funding) / 2.0
            funding_cost = trade_row["notional"] * avg_funding_rate * hours_held
            if trade_row["side"] == "short":
                funding_cost = -funding_cost
            exit_price, pnl = close_paper_position(
                trade_row["entry_price"], features.mid_price, trade_row["size"], trade_row["side"],
                spread_pct=features.spread_pct, funding_cost=funding_cost,
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

            if signal.ambiguous and self.config.council_enabled:
                council = self.population.council_genomes(agent.id)
                signal = council_consult(
                    signal, snap, self.prev_open_interest, self.htf_trend_up, council,
                    self.config.council_quorum_pct, self.config.council_min_active_voters,
                )
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
                agent.balance, features.mid_price, signal.action, effective_size_pct,
                spread_pct=features.spread_pct,
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
                regime=classify_regime(features.trend_up, features.adx_value),
                entry_funding=features.funding,
            )
            log.info("Agent %d opened %s %s @ %.4f (confidence=%.2f) - %s",
                      agent.id, signal.action.upper(), genome.coin, fill_price,
                      signal.confidence, signal.reasons[0] if signal.reasons else "")

    def _is_breaker_tripped(self) -> bool:
        return self.db.get_meta("live_breaker_tripped") == "1"

    def _notify_breaker_webhook(self, reason: str) -> None:
        """Best-effort ping to an external webhook (Slack/Discord-compatible
        or any JSON endpoint) so a trip isn't silent if nobody's watching
        the dashboard. Never raises - a failed notification must never be
        confused with a failed (or worse, un-flattened) trip."""
        url = self.config.live_breaker_webhook_url
        if not url:
            return
        import json
        import urllib.request
        payload = {
            "text": f":rotating_light: LIVE CIRCUIT BREAKER TRIPPED ({self.config.token}): {reason} "
                    f"- real position flattened, live trading paused until "
                    f"`python3 main.py --clear-live-breaker` is run.",
            "token": self.config.token,
            "reason": reason,
            "tripped_at": now_iso(),
        }
        try:
            req = urllib.request.Request(
                url, data=json.dumps(payload).encode("utf-8"),
                headers={"Content-Type": "application/json"}, method="POST",
            )
            urllib.request.urlopen(req, timeout=5)
        except Exception as e:
            log.warning("Circuit breaker webhook notification failed (trip itself still stands): %s", e)

    def _trip_circuit_breaker(self, reason: str, snap: MarketSnapshot) -> None:
        """The one safeguard that protects real money rather than just
        improving statistical confidence. Flattens the real position
        immediately and persists the trip so it survives a restart - live
        trading stays paused until a human runs
        `python3 main.py --clear-live-breaker`, not just until conditions
        look better again."""
        log.critical(
            "LIVE CIRCUIT BREAKER TRIPPED: %s - flattening real position and pausing live "
            "trading until explicitly cleared with `python3 main.py --clear-live-breaker`",
            reason,
        )
        self.db.set_meta("live_breaker_tripped", "1")
        self.db.set_meta("live_breaker_reason", reason)
        self.db.set_meta("live_breaker_tripped_at", now_iso())
        if self.live is not None:
            try:
                results = self.live.adjust_to(self.config.token, None, 0.0, snap.sz_decimals)
                for r in results:
                    self.db.record_live_order(self.config.token, r["action"], r["side"],
                                               0.0, r["fill_price"], r["status"], r["detail"])
                self.db.set_live_position(self.config.token, None, 0.0, 0.0)
            except Exception:
                log.exception("Circuit breaker flatten order failed - live position may "
                               "still be open, check the exchange directly")
        self._notify_breaker_webhook(reason)

    def _check_circuit_breaker(self, snap: MarketSnapshot) -> bool:
        """Returns True if live trading should stay paused this cycle -
        either already tripped, or trips right now on drawdown, a fast
        single-day loss, or a frozen price feed. Paper trading and the
        evolutionary population are never affected by this; only the real
        aggregate live position is."""
        if self.live is None or not self.config.live_circuit_breaker_enabled:
            return False

        if self._is_breaker_tripped():
            reason = self.db.get_meta("live_breaker_reason") or "unknown"
            log.warning("Live circuit breaker still tripped (%s) - live trading paused. "
                        "Clear it explicitly with `python3 main.py --clear-live-breaker` once resolved.", reason)
            return True

        # Stale/frozen price feed: more dangerous than a merely quiet market -
        # trading decisions this cycle would be based on data that isn't real.
        if self._last_seen_price is not None and snap.mid_price == self._last_seen_price:
            self._stale_price_count += 1
        else:
            self._stale_price_count = 0
        self._last_seen_price = snap.mid_price
        if self._stale_price_count >= self.config.live_stale_price_cycles:
            self._trip_circuit_breaker(
                f"price feed unchanged ({snap.mid_price}) for {self._stale_price_count} "
                "consecutive cycles - possible broken/frozen feed",
                snap,
            )
            return True

        # Both equity-based checks below share one API call. A None read
        # (transient API failure) skips both - never trip on our own
        # inability to ask, only on a real, confirmed number.
        equity = self.live.get_account_equity()
        if equity is None:
            return False

        # All-time peak drawdown.
        peak_raw = self.db.get_meta("live_peak_equity")
        peak = max(float(peak_raw), equity) if peak_raw else equity
        self.db.set_meta("live_peak_equity", str(peak))
        if peak > 0:
            drawdown_pct = (peak - equity) / peak * 100.0
            if drawdown_pct >= self.config.live_max_drawdown_pct:
                self._trip_circuit_breaker(
                    f"drawdown {drawdown_pct:.1f}% from peak equity ${peak:.2f} (current ${equity:.2f})",
                    snap,
                )
                return True

        # Fast single-day loss - catches a bleed that hasn't yet pulled the
        # full live_max_drawdown_pct off the ALL-TIME peak (e.g. already
        # down 15% from a much older peak, then loses another 10% today).
        # Resets at each new UTC calendar day.
        today = now_iso()[:10]
        day_start_date = self.db.get_meta("live_day_start_date")
        if day_start_date != today:
            self.db.set_meta("live_day_start_date", today)
            self.db.set_meta("live_day_start_equity", str(equity))
        else:
            day_start_raw = self.db.get_meta("live_day_start_equity")
            day_start_equity = float(day_start_raw) if day_start_raw else equity
            if day_start_equity > 0:
                daily_loss_pct = (day_start_equity - equity) / day_start_equity * 100.0
                if daily_loss_pct >= self.config.live_max_daily_loss_pct:
                    self._trip_circuit_breaker(
                        f"daily loss {daily_loss_pct:.1f}% (today's start equity ${day_start_equity:.2f}, "
                        f"current ${equity:.2f})",
                        snap,
                    )
                    return True

        return False

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

    def _refresh_backtest_window(self) -> None:
        """The recent-window backtest data (used for new-agent screening AND
        revalidation) is otherwise only ever fetched once at startup - stale
        after the first few hours of a long-running deployment. Refetches it
        periodically so "backtested against the newest data" stays true.
        Also recomputes the train/validation split via
        Population.set_backtest_window - failure just keeps the previous
        (older but still usable) window and split."""
        try:
            snap = self.hl.get_snapshot(self.config.token, self.config.timeframe,
                                         candle_lookback_hours=self.config.backtest_lookback_hours)
            funding = self.hl.get_funding_history(self.config.token, self.config.backtest_lookback_hours)
            self.population.set_backtest_window(snap.candles, funding)
            log.info("Refreshed recent-window backtest data: %d candles, %d funding points",
                      len(snap.candles), len(funding))
        except Exception:
            log.exception("Failed to refresh recent-window backtest data - keeping the previous window")

    def _maybe_revalidate(self) -> None:
        """Periodically re-backtests already-proven top agents against the
        (freshly-refreshed) recent-window data and flags drift on the
        dashboard - see agents/population.py::revalidate_top_agents. Purely
        informational: never kills, culls, or re-ranks anything."""
        try:
            checked = self.population.revalidate_top_agents()
            if checked:
                log.info("Revalidated %d top agents against the current recent-window data", checked)
        except Exception:
            log.exception("Revalidation pass failed - will retry next scheduled cycle")

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
        if not self._check_circuit_breaker(snap):
            self._sync_live_exposure(snap)

        self.db.record_population_cycle(self.cycle, **stats)
        self.prev_open_interest = snap.open_interest
        # Lets the dashboard show unrealized PnL on open trades instead of a
        # blank "-" for however long a position stays open (max_hold_hours
        # can be up to 96h) - see dashboard/server.py's /api/trades.
        self.db.set_meta("last_price", str(snap.mid_price))

        if self.config.backtest_enabled and self.cycle % self.config.backtest_refresh_interval_cycles == 0:
            self._refresh_backtest_window()
        if self.config.revalidation_enabled and self.cycle % self.config.revalidation_interval_cycles == 0:
            self._maybe_revalidate()

        log.info(
            "Cycle %d done | %s | alive=%d active_traders=%d professional=%d best_agent=%s best_pnl=%s",
            self.cycle, "LIVE" if self.config.is_live() else "paper",
            stats["alive_count"], stats["active_trader_count"],
            stats["professional_count"], stats["best_agent_id"], stats["best_total_pnl"],
        )
