"""Entry point: an evolving population of swing-trading agents on Hyperliquid.

Do-or-die lifecycle: an agent that wins a trade survives and spawns mutated
children; an agent that loses dies immediately. Up to `population_cap` agents
stay alive at once, but only the top `active_trader_count` by performance are
allowed to open new positions. Strategies that reach a long win streak are
shared across the population. See README.md for the full design.

Runs in PAPER TRADING MODE ONLY - it simulates fills against real Hyperliquid
market data. No real orders are ever placed by this code path.
"""
from __future__ import annotations

import logging
import time

from config import CONFIG
from db.database import Database
from market.hyperliquid_client import HyperliquidClient
from agents.population import Population
from engine.orchestrator import Orchestrator

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
log = logging.getLogger("main")


def main() -> None:
    if not CONFIG.is_paper():
        raise SystemExit(
            "TRADING_MODE=live is not implemented in this codebase - live order "
            "placement was intentionally left out. Paper trading only for now."
        )

    log.info("=" * 70)
    log.info("PAPER TRADING MODE - simulated fills only, no real funds at risk")
    log.info("Symbols=%s timeframe=%s cycle=%ds population_cap=%d active_traders=%d",
              CONFIG.symbols, CONFIG.timeframe, CONFIG.cycle_seconds,
              CONFIG.population_cap, CONFIG.active_trader_count)
    log.info("=" * 70)

    db = Database(CONFIG.db_path)
    hl = HyperliquidClient(network=CONFIG.hl_network)
    population = Population(db, CONFIG)
    population.seed_if_empty()
    orchestrator = Orchestrator(db, hl, population, CONFIG)

    try:
        while True:
            try:
                orchestrator.run_cycle()
            except Exception:
                log.exception("Cycle %d failed - will retry next interval", orchestrator.cycle)
            time.sleep(CONFIG.cycle_seconds)
    except KeyboardInterrupt:
        log.info("Shutting down.")
    finally:
        db.close()


if __name__ == "__main__":
    main()
