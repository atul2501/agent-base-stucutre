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
import logging
import sys
import threading
import time
from logging.handlers import RotatingFileHandler
from pathlib import Path

faulthandler.enable()  # print a native stack trace instead of a silent segfault

from config import CONFIG
from db.database import Database
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

_file_handler = RotatingFileHandler(LOG_DIR / "agent_swarm.log", maxBytes=5_000_000, backupCount=3)
_file_handler.setLevel(logging.INFO)
_file_handler.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(name)s: %(message)s"))

_console_handler = logging.StreamHandler()
_console_handler.setLevel(logging.WARNING)
_console_handler.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(message)s"))

logging.basicConfig(level=logging.INFO, handlers=[_file_handler, _console_handler])
log = logging.getLogger("main")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--reset", action="store_true",
        help="Wipe all agents/trades/history and start learning from scratch "
             "(step 0). Required when switching TOKEN to a different asset.",
    )
    parser.add_argument("--no-dashboard", action="store_true", help="Don't start the web dashboard.")
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


def main() -> None:
    args = parse_args()

    log.info("=" * 70)
    log.info("%s MODE | network=%s | token=%s",
              "LIVE" if CONFIG.is_live() else "PAPER TRADING",
              CONFIG.hl_network, CONFIG.token)
    if CONFIG.is_live():
        if CONFIG.hl_network.lower() == "mainnet":
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
    enforce_single_token_guard(db, args.reset)

    hl = HyperliquidClient(network="mainnet" if CONFIG.hl_network == "mainnet" else "testnet")
    if not hl.is_valid_coin(CONFIG.token):
        log.error("TOKEN=%s is not a valid Hyperliquid perp symbol.", CONFIG.token)
        sys.exit(1)

    ollama_advisor.check_health_async()

    live_executor = None
    if CONFIG.is_live():
        try:
            live_executor = LiveExecutor(CONFIG)
            log.info("Live executor ready for wallet %s", live_executor.address)
        except Exception:
            log.exception("Failed to initialize live executor - falling back to paper for this run")

    backtest_candles, backtest_funding = [], []
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

    population = Population(db, CONFIG, backtest_candles=backtest_candles, backtest_funding=backtest_funding)
    population.seed_if_empty()
    orchestrator = Orchestrator(db, hl, population, CONFIG, live=live_executor)

    if not args.no_dashboard:
        from dashboard.server import run_dashboard
        threading.Thread(target=run_dashboard, args=(CONFIG,), daemon=True).start()
        log.info("Dashboard: http://%s:%d", CONFIG.dashboard_host, CONFIG.dashboard_port)

    try:
        while True:
            try:
                orchestrator.run_cycle()
            except Exception:
                log.exception("Cycle %d failed - will retry next interval", orchestrator.cycle)
            time.sleep(CONFIG.cycle_seconds)
    except KeyboardInterrupt:
        # Ctrl+C never deletes anything - agent/trade history stays in
        # CONFIG.db_path exactly as it is. Only --reset wipes it.
        log.info("Shutting down (Ctrl+C). Agent data preserved in %s - "
                  "restart with `python3 main.py` (no --reset) to keep training from here.",
                  CONFIG.db_path)
        print("exit")
    finally:
        db.close()


if __name__ == "__main__":
    main()
