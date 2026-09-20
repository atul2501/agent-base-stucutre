"""Entry point: an evolving population of swing-trading agents on Hyperliquid,
trading exactly one token at a time (config.TOKEN).

Do-or-die lifecycle: an agent that wins a trade survives and spawns mutated
children; an agent that loses dies immediately. Up to `population_cap` agents
stay alive at once, but only the top `active_trader_count` by performance are
allowed to open new positions. Strategies that reach a long win streak are
shared across the population. See README.md for the full design.

TRADING_MODE is the one switch that matters: "paper" (default) always
simulates, no matter what network you're pointed at. "live" places real
orders on whichever HL_NETWORK is set - testnet (fake funds, safe to test
the real order-placement code) or mainnet (real money, which additionally
requires LIVE_TRADING_CONFIRMED set to the exact phrase in .env.example).
See config.py.

Run with --reset to wipe all agents/trades and start fresh - required
whenever you change TOKEN, since a population's strategies are only ever
tuned to one token (see agents/population.py and README "one token at a
time").
"""
from __future__ import annotations

import argparse
import faulthandler
import json
import logging
import signal
import sys
import threading
import time
from logging.handlers import RotatingFileHandler
from pathlib import Path

faulthandler.enable()  # print a native stack trace instead of a silent segfault

from config import CONFIG
from db.database import Database, now_iso
from market.hyperliquid_client import HyperliquidClient
from agents.population import Population
from engine.orchestrator import Orchestrator
from reasoning import ollama_advisor
from trading.live_executor import LiveExecutor

# Everything (cycles, trades, agent deaths, etc.) goes to the log file.
# The terminal only shows warnings/errors plus Flask's own startup banner
# ("Serving Flask app" / "Debug mode" / "Running on") - that one prints
# directly to stdout regardless of this config, so it always shows.
LOG_DIR = Path(__file__).parent / "logs"
LOG_DIR.mkdir(exist_ok=True)


def cleanup_old_logs(log_dir: Path, max_age_hours: float) -> None:
    """Deletes any file in log_dir last modified more than max_age_hours ago
    - covers agent_swarm.log's rotated .1/.2/.3 backups (RotatingFileHandler
    only caps them by COUNT, not age) and, if this runs under systemd with
    output redirected to logs/systemd*.log, those too, since nothing else
    in this codebase ever rotates or cleans those. Never touches the
    currently-active agent_swarm.log itself even if it's old, since deleting
    a file a live logging.FileHandler still has open would just create an
    orphaned inode - RotatingFileHandler's own size-based rotation already
    handles that file's lifecycle."""
    cutoff = time.time() - max_age_hours * 3600
    active_log = (log_dir / "agent_swarm.log").resolve()
    for f in log_dir.glob("*.log*"):
        try:
            if f.resolve() == active_log:
                continue
            if f.is_file() and f.stat().st_mtime < cutoff:
                f.unlink()
        except OSError:
            pass  # best-effort - a log file mid-write/deleted concurrently isn't worth crashing over


_file_handler = RotatingFileHandler(LOG_DIR / "agent_swarm.log", maxBytes=5_000_000, backupCount=3)
_file_handler.setLevel(logging.INFO)
_file_handler.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(name)s: %(message)s"))

_console_handler = logging.StreamHandler()
_console_handler.setLevel(logging.WARNING)
_console_handler.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(message)s"))

logging.basicConfig(level=logging.INFO, handlers=[_file_handler, _console_handler])
log = logging.getLogger("main")

cleanup_old_logs(LOG_DIR, CONFIG.log_max_age_hours)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--reset", action="store_true",
        help="Wipe all agents/trades/history and start learning from scratch "
             "(step 0). Required when switching TOKEN to a different asset.",
    )
    parser.add_argument("--no-dashboard", action="store_true", help="Don't start the web dashboard.")
    parser.add_argument(
        "--clear-live-breaker", action="store_true",
        help="Clear a tripped live-trading circuit breaker (drawdown or stale-feed trip - "
             "see engine/orchestrator.py) and resume live trading. Investigate the reason "
             "logged at trip time BEFORE clearing this - it does not re-arm automatically "
             "by design. Exits immediately after clearing; does not start the trading loop.",
    )
    parser.add_argument(
        "--stress-test", action="store_true",
        help="Run the exit-logic stress test suite (flash crash, flash spike, extreme "
             "favorable gap, max-hold timeout, near-zero price - see backtest/stress_test.py) "
             "against a representative genome and exit. No DB/network needed; doesn't start "
             "the trading loop.",
    )
    parser.add_argument(
        "--snapshot-best", type=int, nargs="?", const=10, default=None, metavar="N",
        help="Save the current top N alive agents' genomes (by fitness, default N=10) to "
             "SNAPSHOT_DIR (default 'snapshots/') as JSON files, then exit. These are "
             "reloaded automatically the next time a FRESH population is seeded (an empty "
             "DB, or after --reset) - a head start instead of pure random genomes. Never "
             "touches the currently-running population.",
    )
    return parser.parse_args()


def enforce_single_token_guard(db: Database, reset: bool) -> None:
    if reset:
        log.warning("--reset passed: wiping ALL agents, trades, and history (step 0)")
        db.reset_all()
        db.set_meta("current_token", CONFIG.token)
        return

    current = db.get_meta("current_token")
    if current is None:
        db.set_meta("current_token", CONFIG.token)
        return

    if current != CONFIG.token:
        log.error(
            "This database was built for %s but TOKEN=%s now. A population's "
            "strategies only make sense for the token they were trained on. "
            "Re-run with --reset to wipe it and start learning %s from scratch, "
            "or point DB_PATH at a fresh file, or set TOKEN back to %s.",
            current, CONFIG.token, CONFIG.token, current,
        )
        sys.exit(1)


def _handle_sigterm(signum, frame) -> None:
    """SIGTERM is how `kill <pid>`, `systemctl stop`, and `docker stop` all
    ask a process to exit by default - unlike Ctrl+C's SIGINT, Python does
    NOT turn it into a catchable exception on its own. Without this, an
    unattended deployment (EC2, systemd, a container) would die abruptly
    on every stop instead of hitting the same graceful shutdown path."""
    raise KeyboardInterrupt()


def main() -> None:
    signal.signal(signal.SIGTERM, _handle_sigterm)
    args = parse_args()

    if args.stress_test:
        from backtest.stress_test import main as run_stress_test
        sys.exit(run_stress_test())

    db: Database | None = None

    # The whole body - including startup (historical data fetch, ~20-30s of
    # backtest-screened seeding) - is inside this try, not just the main
    # loop. A signal arriving mid-startup used to produce an uncaught
    # traceback instead of a graceful shutdown; verified by reproducing it
    # with a real SIGTERM sent during startup before this fix.
    try:
        log.info("=" * 70)
        log.info("%s MODE | network=%s | token=%s",
                  "LIVE" if CONFIG.is_live() else "PAPER TRADING",
                  CONFIG.hl_network, CONFIG.token)
        if CONFIG.is_live():
            if CONFIG.hl_network == "mainnet":
                log.warning("LIVE TRADING IS ARMED ON MAINNET. Real orders, real money.")
            else:
                log.warning("LIVE TRADING IS ARMED ON TESTNET. Real orders, fake testnet funds.")
            log.warning("Hard caps: $%.2f total notional, top %d agents, %dx leverage",
                         CONFIG.live_max_total_notional_usd, CONFIG.live_active_trader_count,
                         CONFIG.live_max_leverage)
        log.info("timeframe=%s cycle=%ds population_cap=%d active_traders=%d",
                  CONFIG.timeframe, CONFIG.cycle_seconds, CONFIG.population_cap, CONFIG.active_trader_count)
        log.info("=" * 70)

        db = Database(CONFIG.db_path)

        if args.clear_live_breaker:
            if db.get_meta("live_breaker_tripped") != "1":
                log.info("Live circuit breaker is not currently tripped - nothing to clear.")
            else:
                reason = db.get_meta("live_breaker_reason") or "unknown"
                log.warning("Clearing live circuit breaker (was tripped: %s)", reason)
                db.set_meta("live_breaker_tripped", "0")
                db.set_meta("live_breaker_reason", "")
                db.set_meta("live_peak_equity", "")
                # A genuinely fresh chance for the price feed too, in case
                # the trip was staleness-caused and the operator has since
                # fixed the feed - otherwise the persisted count (see
                # engine/orchestrator.py) would immediately re-trip on the
                # very next cycle.
                db.set_meta("live_stale_last_price", "")
                db.set_meta("live_stale_price_count", "0")
                log.info("Cleared. Live trading will resume from the next run of `python3 main.py` "
                         "(without --clear-live-breaker) if TRADING_MODE=live.")
            return

        if args.snapshot_best is not None:
            top = sorted(db.list_alive_agents(), key=lambda a: a.fitness, reverse=True)[: args.snapshot_best]
            if not top:
                log.info("No alive agents to snapshot.")
                return
            snap_dir = Path(CONFIG.snapshot_dir)
            snap_dir.mkdir(parents=True, exist_ok=True)
            stamp = now_iso().replace(":", "").replace("+00:00", "Z")
            for agent in top:
                path = snap_dir / f"agent{agent.id}_{CONFIG.token}_{stamp}.json"
                path.write_text(json.dumps({
                    "source_agent_id": agent.id,
                    "token": CONFIG.token,
                    "snapshotted_at": now_iso(),
                    "fitness": agent.fitness,
                    "tier": agent.tier,
                    "generation": agent.generation,
                    "wins": agent.wins,
                    "losses": agent.losses,
                    "total_pnl": agent.total_pnl,
                    "genome": agent.genome,
                }, indent=2))
            log.info("Snapshotted %d agent genome(s) to %s - they'll be used to seed any "
                      "future fresh %s population automatically.", len(top), snap_dir, CONFIG.token)
            return

        enforce_single_token_guard(db, args.reset)

        hl = HyperliquidClient(network=CONFIG.hl_network)
        if not hl.is_valid_coin(CONFIG.token):
            log.error("TOKEN=%s is not a valid Hyperliquid perp symbol.", CONFIG.token)
            sys.exit(1)

        ollama_advisor.check_health_async()

        live_executor = None
        if CONFIG.is_live():
            try:
                live_executor = LiveExecutor(CONFIG)
                log.info("Live executor ready for wallet %s", live_executor.address)
                db.set_meta("live_executor_ready", "1")
                db.set_meta("live_executor_error", "")
            except Exception as e:
                log.exception("Failed to initialize live executor - falling back to paper for this run")
                # TRADING_MODE=live is still configured (config.is_live()
                # stays True everywhere else - DB is_live_trader flags, the
                # dashboard's mode badge) even though no real orders will
                # ever be placed this run. Without this flag, the dashboard
                # would silently show "LIVE - REAL MONEY" while the system
                # is actually paper-only - see dashboard/server.py's
                # /api/overview and dashboard/static/index.html's header
                # badge.
                db.set_meta("live_executor_ready", "0")
                db.set_meta("live_executor_error", str(e)[:300])

        backtest_candles, backtest_funding = [], []
        htf_candles: list[dict] = []
        regime_windows: list[tuple[list[dict], list[tuple[int, float, float]]]] = []
        if CONFIG.backtest_enabled:
            try:
                backtest_snap = hl.get_snapshot(CONFIG.token, CONFIG.timeframe,
                                                 candle_lookback_hours=CONFIG.backtest_lookback_hours)
                backtest_candles = backtest_snap.candles
                backtest_funding = hl.get_funding_history(CONFIG.token, CONFIG.backtest_lookback_hours)
                log.info("Backtest pre-screening ready: %d historical candles, %d funding points",
                          len(backtest_candles), len(backtest_funding))
            except Exception:
                log.exception("Failed to fetch historical data for backtest pre-screening - "
                               "new agents will be born from unscreened random/mutated genomes this run")

            try:
                # Same recent window, at the higher timeframe - lets the
                # backtest evaluate the higher-timeframe trend filter for
                # real instead of treating it as neutral (see
                # backtest/engine.py). Best-effort: failure just keeps the
                # previous (neutral) treatment, same as before this existed.
                htf_candles = hl.get_candles(CONFIG.token, CONFIG.higher_timeframe, CONFIG.backtest_lookback_hours)
                log.info("Higher-timeframe backtest data ready: %d %s candles",
                          len(htf_candles), CONFIG.higher_timeframe)
            except Exception:
                log.exception("Failed to fetch higher-timeframe backtest data - the HTF trend filter "
                               "will stay neutral in backtest scoring this run")

            try:
                regime_candles = hl.get_candles(CONFIG.token, CONFIG.backtest_regime_timeframe,
                                                 CONFIG.backtest_regime_lookback_hours)
                regime_funding_all = hl.get_funding_history(CONFIG.token, CONFIG.backtest_regime_lookback_hours)
                segments = CONFIG.backtest_regime_segments
                seg_len = max(1, len(regime_candles) // segments)
                for i in range(segments):
                    seg_candles = regime_candles[i * seg_len: (i + 1) * seg_len if i < segments - 1 else len(regime_candles)]
                    if len(seg_candles) < 50:
                        continue
                    lo_ms, hi_ms = seg_candles[0]["t"], seg_candles[-1]["t"]
                    seg_funding = [f for f in regime_funding_all if lo_ms <= f[0] <= hi_ms]
                    regime_windows.append((seg_candles, seg_funding))
                log.info("Multi-regime backtest screening ready: %d segments from %d %s candles",
                          len(regime_windows), len(regime_candles), CONFIG.backtest_regime_timeframe)
            except Exception:
                log.exception("Failed to fetch multi-regime backtest data - new agents will be "
                               "screened against the recent window only this run")

        population = Population(db, CONFIG, backtest_candles=backtest_candles, backtest_funding=backtest_funding,
                                 regime_windows=regime_windows, htf_candles=htf_candles)
        population.seed_if_empty()
        orchestrator = Orchestrator(db, hl, population, CONFIG, live=live_executor)

        if not args.no_dashboard:
            from dashboard.server import run_dashboard
            threading.Thread(target=run_dashboard, args=(CONFIG,), daemon=True).start()
            log.info("Dashboard: http://%s:%d", CONFIG.dashboard_host, CONFIG.dashboard_port)

        last_log_cleanup = time.monotonic()
        while True:
            try:
                orchestrator.run_cycle()
            except Exception:
                log.exception("Cycle %d failed - will retry next interval", orchestrator.cycle)
            # Startup alone only catches restarts - a long-lived process that
            # never restarts still needs this so logs older than
            # log_max_age_hours get cleaned up while running, not just
            # accumulate forever until the next restart.
            if time.monotonic() - last_log_cleanup >= 3600:
                cleanup_old_logs(LOG_DIR, CONFIG.log_max_age_hours)
                last_log_cleanup = time.monotonic()
            time.sleep(CONFIG.cycle_seconds)
    except KeyboardInterrupt:
        # Fires on Ctrl+C (SIGINT) or a stop signal (SIGTERM, e.g. `kill`,
        # `systemctl stop`, `docker stop`) - never deletes anything, agent/
        # trade history stays in CONFIG.db_path exactly as it is. Only
        # --reset wipes it.
        log.info("Shutting down. Agent data preserved in %s - "
                  "restart with `python3 main.py` (no --reset) to keep training from here.",
                  CONFIG.db_path)
        print("exit")
    finally:
        if db is not None:
            db.close()


if __name__ == "__main__":
    main()
