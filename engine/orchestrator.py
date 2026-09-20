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
    _council_tighten_stop, build_features, classify_regime, compute_exit_levels, compute_htf_trend,
    council_consult, council_oppose_position, evaluate_entry, evaluate_position,
)
from trading.live_executor import LiveExecutor
from trading.paper_executor import (
    close_paper_position, open_paper_position, risk_normalized_size_pct, simulate_fill_price,
)

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
        # Circuit breaker staleness tracking - persisted to the DB (see
        # _check_circuit_breaker) rather than in-memory-only, so a restart
        # right after a feed freeze doesn't hand back a few fresh cycles of
        # runway exactly when a crash/deploy might coincide with real
        # market stress. Loaded here from whatever a previous instance
        # last recorded; a fresh DB (no prior run) falls back to None/0.
        last_price_raw = self.db.get_meta("live_stale_last_price")
        self._last_seen_price: float | None = float(last_price_raw) if last_price_raw else None
        count_raw = self.db.get_meta("live_stale_price_count")
        self._stale_price_count = int(count_raw) if count_raw else 0

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
        # Advisory, tighten-only exit-council layer: for positions where
        # nothing else fired this cycle, collect any whose genome-ensemble
        # check was inconclusive (too few active voters) so they can be
        # resolved with ONE batched Ollama call after the main loop, same
        # two-pass pattern as _process_entries below - see
        # strategy/signals.py::council_oppose_position.
        # (trade_id, side, current_stop, mid_price, genome, features) - the
        # current stop/price are captured now so the batch resolution below
        # doesn't need to re-fetch the trade row.
        pending_exit_ollama: list[tuple] = []

        for agent in self.db.list_alive_agents():
            trade_row = self.db.get_open_trade(agent.id)
            if trade_row is None:
                continue
            genome = Genome.from_dict(agent.genome)

            features = build_features(snap, genome, self.prev_open_interest, self.htf_trend_up)
            action = evaluate_position(genome, trade_row, features)

            if action.kind == "none":
                current_stop = trade_row["stop_loss"]
                if self.config.exit_council_enabled and current_stop is not None:
                    council = self.population.council_genomes(agent.id)
                    opposed, inconclusive, reason = council_oppose_position(
                        trade_row["side"], snap, self.prev_open_interest, self.htf_trend_up, council,
                        self.config.council_quorum_pct, self.config.council_min_active_voters,
                    )
                    if opposed:
                        new_stop = _council_tighten_stop(
                            trade_row["side"], current_stop, features.mid_price,
                            self.config.exit_council_tighten_frac,
                        )
                        if new_stop is not None:
                            self.db.update_trade_stop(trade_row["id"], new_stop)
                            log.info("Agent %d exit-council tightened stop to %.4f - %s",
                                      agent.id, new_stop, reason)
                    elif inconclusive and self.config.ollama_enabled:
                        pending_exit_ollama.append(
                            (trade_row["id"], trade_row["side"], current_stop, features.mid_price, genome, features)
                        )
                continue

            if action.kind == "trail":
                self.db.update_trade_stop(trade_row["id"], action.new_stop_loss)
                continue

            remaining_size = trade_row["remaining_size"] or trade_row["size"]

            if action.kind == "partial":
                close_size = remaining_size * action.close_fraction
                new_remaining = remaining_size - close_size
                _exit_price, partial_pnl = close_paper_position(
                    trade_row["entry_price"], features.mid_price, close_size, trade_row["side"],
                    spread_pct=features.spread_pct,
                )
                partial_frac_of_original = close_size / trade_row["size"] if trade_row["size"] else 0.0
                self.db.record_partial_close(trade_row["id"], new_remaining, partial_frac_of_original,
                                              action.pnl_pct, partial_pnl, action.new_stop_loss)
                self.db.apply_partial_pnl(agent.id, partial_pnl)
                log.info("Agent %d took partial profit (%.0f%% of position, pnl=%.2f) - %s",
                          agent.id, action.close_fraction * 100, partial_pnl, action.reason)
                continue

            # action.kind == "close"
            result, reason = action.result, action.reason
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
            exit_price, leg_pnl = close_paper_position(
                trade_row["entry_price"], features.mid_price, remaining_size, trade_row["side"],
                spread_pct=features.spread_pct, funding_cost=funding_cost,
            )
            # trades.pnl records the TOTAL trade economics (any earlier
            # partial + this final leg) for accurate history, but the
            # partial's dollar pnl was already credited to the agent's
            # balance/total_pnl at partial-close time (apply_partial_pnl) -
            # only leg_pnl (the incremental amount) goes to
            # population.handle_win/handle_loss below, so it isn't double-
            # counted.
            partial_realized_pnl = trade_row["partial_realized_pnl"] or 0.0
            total_pnl = partial_realized_pnl + leg_pnl
            self.db.close_trade(trade_row["id"], exit_price, total_pnl, result, reason)

            if result == "win":
                self.population.handle_win(agent.id, leg_pnl)
            else:
                self.population.handle_loss(agent.id, leg_pnl)

        if pending_exit_ollama:
            tighten_decisions = ollama_advisor.consult_exit_batch(
                [(genome, features, side) for _trade_id, side, _stop, _price, genome, features in pending_exit_ollama]
            )
            for (trade_id, side, current_stop, mid_price, _genome, _features), tighten in zip(
                pending_exit_ollama, tighten_decisions
            ):
                if not tighten:
                    continue
                new_stop = _council_tighten_stop(side, current_stop, mid_price, self.config.exit_council_tighten_frac)
                if new_stop is not None:
                    self.db.update_trade_stop(trade_id, new_stop)
                    log.info("Trade %d exit-council (ollama) tightened stop to %.4f", trade_id, new_stop)

    def _process_entries(self, snap: MarketSnapshot) -> None:
        # Two passes: collect every still-ambiguous signal (after the free
        # rule-based/council checks) across ALL active traders first, then
        # resolve them with ONE batched Ollama call instead of one call per
        # agent - see reasoning/ollama_advisor.py::consult_batch.
        resolved: list[tuple] = []  # (agent, genome, features, signal)
        pending_ollama: list[tuple] = []  # (agent, genome, features, signal)

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
                pending_ollama.append((agent, genome, features, signal))
                continue

            resolved.append((agent, genome, features, signal))

        if pending_ollama:
            batch_results = ollama_advisor.consult_batch(
                [(genome, features, signal) for _agent, genome, features, signal in pending_ollama]
            )
            for (agent, genome, features, _signal), resolved_signal in zip(pending_ollama, batch_results):
                resolved.append((agent, genome, features, resolved_signal))

        for agent, genome, features, signal in resolved:
            if signal.action == "hold":
                continue

            # Scale position size by how strongly the signal was confirmed -
            # a barely-passing setup risks less than a strongly-confirmed
            # one, instead of both risking the same genome-fixed %. Floored
            # at 30% of the genome's intended size so a weak-but-approved
            # signal isn't shrunk to near nothing.
            effective_size_pct = genome.position_size_pct * max(0.3, min(1.0, signal.confidence))
            # simulate_fill_price is a pure function of (mid_price, side,
            # is_entry, spread_pct) - computing it here to derive the exit
            # levels/risk normalization BEFORE sizing, then calling
            # open_paper_position normally below, reproduces the exact same
            # fill_price deterministically rather than duplicating its
            # notional/size math.
            preview_fill = simulate_fill_price(features.mid_price, signal.action, is_entry=True,
                                                spread_pct=features.spread_pct)
            levels = compute_exit_levels(genome, signal.action, preview_fill, features.atr_pct)
            actual_stop_pct = abs(preview_fill - levels.stop_loss) / preview_fill * 100.0 if preview_fill else genome.stop_loss_pct
            effective_size_pct = risk_normalized_size_pct(effective_size_pct, genome.stop_loss_pct, actual_stop_pct)

            fill_price, size, notional = open_paper_position(
                agent.balance, features.mid_price, signal.action, effective_size_pct,
                spread_pct=features.spread_pct,
            )

            self.db.open_trade(
                agent.id, genome.coin, signal.action, fill_price, size, notional,
                levels.stop_loss, levels.take_profit, entry_reason="; ".join(signal.reasons),
                regime=classify_regime(features.trend_up, features.adx_value),
                entry_funding=features.funding,
                partial_target=levels.partial_target,
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
        self.db.set_meta("live_stale_last_price", str(self._last_seen_price))
        self.db.set_meta("live_stale_price_count", str(self._stale_price_count))
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

    def _check_paper_drawdown(self) -> bool:
        """Returns True if new paper entries should be paused this cycle -
        see config.py's paper_circuit_breaker_enabled comment for why this
        exists alongside the live-only circuit breaker above. Tracks the
        swarm-wide realized-pnl peak and pauses once the drawdown from it
        passes paper_max_drawdown_usd, auto-resuming at half that
        threshold (hysteresis, so it doesn't flap open/closed every
        cycle right at the boundary)."""
        if not self.config.paper_circuit_breaker_enabled:
            return False

        realized = self.db.realized_pnl_stats()["total_realized_pnl"]
        peak_raw = self.db.get_meta("paper_peak_realized_pnl")
        peak = max(float(peak_raw), realized) if peak_raw else max(0.0, realized)
        self.db.set_meta("paper_peak_realized_pnl", str(peak))
        drawdown_usd = peak - realized

        paused = self.db.get_meta("paper_breaker_paused") == "1"
        if not paused and drawdown_usd >= self.config.paper_max_drawdown_usd:
            self.db.set_meta("paper_breaker_paused", "1")
            log.warning("Paper-population drawdown $%.2f past threshold ($%.2f) - pausing new "
                        "paper entries until it recovers", drawdown_usd, self.config.paper_max_drawdown_usd)
            return True
        if paused:
            if drawdown_usd <= self.config.paper_max_drawdown_usd / 2.0:
                self.db.set_meta("paper_breaker_paused", "0")
                log.info("Paper-population drawdown recovered to $%.2f - resuming new paper entries", drawdown_usd)
                return False
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

        if any(r["status"] != "filled" for r in results):
            # At least one order failed - the real position may not match
            # desired_side/desired_size any more (e.g. a flip's close leg
            # filled but its open leg didn't). Reconcile against the
            # exchange's own ground truth instead of recording the target
            # we merely attempted, so the dashboard/DB never claims a
            # position was reached that a real order just failed to reach.
            # get_actual_position (unlike get_account_equity) doesn't catch
            # its own network errors, so this is wrapped here - a fetch
            # failure right after an order failure must not turn into an
            # uncaught exception that skips the rest of this cycle
            # (record_population_cycle, prev_open_interest, etc below).
            try:
                actual_side, actual_size = self.live.get_actual_position(self.config.token)
                actual_notional = actual_size * snap.mid_price if actual_side else 0.0
                self.db.set_live_position(self.config.token, actual_side, actual_size, actual_notional)
            except Exception:
                log.exception("Failed to reconcile live position after an order failure - "
                               "live_position table may be stale until the next successful sync")
        else:
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
            # Best-effort: keeps the previous HTF series (rather than
            # dropping to neutral) if this particular fetch fails, since
            # it's a nice-to-have refresh, not the primary window.
            htf_candles = self.population.htf_candles
            try:
                htf_candles = self.hl.get_candles(self.config.token, self.config.higher_timeframe,
                                                   self.config.backtest_lookback_hours)
            except Exception:
                log.warning("Failed to refresh higher-timeframe backtest data - keeping the previous HTF series")
            self.population.set_backtest_window(snap.candles, funding, htf_candles)
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
        if not self._check_paper_drawdown():
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
