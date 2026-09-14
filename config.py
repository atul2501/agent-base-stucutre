"""Central configuration, loaded from environment variables / .env."""
from __future__ import annotations

import os
from dataclasses import dataclass

from dotenv import load_dotenv

load_dotenv()

LIVE_CONFIRMATION_PHRASE = "I_UNDERSTAND_THE_RISK"


def _bool(name: str, default: bool) -> bool:
    val = os.getenv(name)
    if val is None:
        return default
    return val.strip().lower() in ("1", "true", "yes", "on")


@dataclass(frozen=True)
class Config:
    # --- market: single token at a time, by design (see README "step 0" reset) ---
    token: str = os.getenv("TOKEN", "SOL")
    timeframe: str = os.getenv("TIMEFRAME", "1h")
    cycle_seconds: int = int(os.getenv("CYCLE_SECONDS", "900"))
    # A slower timeframe checked alongside the primary one - trading with
    # the bigger trend instead of against short-term noise. Live-only (see
    # strategy/signals.py) - not genome-tunable, deliberately a fixed macro
    # context rather than another per-agent parameter to hyper-tune.
    higher_timeframe: str = os.getenv("HIGHER_TIMEFRAME", "1h")

    # --- trading mode: the one switch that matters ---
    # TRADING_MODE=paper (default) - always simulated, never touches the real
    # exchange, no matter what HL_NETWORK is set to.
    # TRADING_MODE=live - places real orders on whichever HL_NETWORK you're
    # pointed at. On testnet that's real testnet orders with fake funds (a
    # safe way to test the live order-placement code itself). On mainnet
    # it's real money, so it additionally requires LIVE_TRADING_CONFIRMED to
    # be set to the exact phrase below - this extra gate only applies to
    # mainnet, since testnet has nothing real to lose.
    trading_mode: str = os.getenv("TRADING_MODE", "paper")
    hl_network: str = os.getenv("HL_NETWORK", "testnet")
    hl_wallet_address: str = os.getenv("HYPERLIQUID_WALLET_ADDRESS", "")
    hl_private_key: str = os.getenv("HYPERLIQUID_PRIVATE_KEY", "")
    live_trading_confirmed: str = os.getenv("LIVE_TRADING_CONFIRMED", "")

    # --- live trading hard risk limits (only relevant once is_live() is True) ---
    live_max_total_notional_usd: float = float(os.getenv("LIVE_MAX_TOTAL_NOTIONAL_USD", "100"))
    live_active_trader_count: int = int(os.getenv("LIVE_ACTIVE_TRADER_COUNT", "20"))
    live_max_leverage: int = int(os.getenv("LIVE_MAX_LEVERAGE", "5"))

    # --- population / evolution ---
    population_cap: int = int(os.getenv("POPULATION_CAP", "500"))
    active_trader_count: int = int(os.getenv("ACTIVE_TRADER_COUNT", "50"))
    min_population_floor: int = int(os.getenv("MIN_POPULATION_FLOOR", "50"))
    # Of active_trader_count slots, this many are reserved for the newest
    # untested (0-trade) agents regardless of fitness, so a brand-new agent
    # is guaranteed at least one real trade instead of potentially never
    # being picked once older/positive-fitness agents occupy every slot.
    guaranteed_newcomer_slots: int = int(os.getenv("GUARANTEED_NEWCOMER_SLOTS", "5"))
    # New agents are pre-screened against real historical data before they
    # enter the live do-or-die population (see backtest/engine.py) - this
    # doesn't change live trading at all, it just means an agent is BORN
    # with a better-than-random genome. Hyperliquid caps a single candle
    # request at ~5000 bars (~17 days of 5m data); default sits safely
    # under that.
    backtest_enabled: bool = _bool("BACKTEST_ENABLED", True)
    backtest_lookback_hours: int = int(os.getenv("BACKTEST_LOOKBACK_HOURS", "360"))
    backtest_candidates: int = int(os.getenv("BACKTEST_CANDIDATES", "5"))
    # An agent whose regime filters never admit a trade would otherwise
    # squat a population slot forever (the normal cull only considers agents
    # that have completed >=1 trade). Past this many idle hours, it becomes
    # culuable too, but only when the population actually needs the room.
    max_idle_hours_before_cull: int = int(os.getenv("MAX_IDLE_HOURS_BEFORE_CULL", "6"))
    children_per_win: int = int(os.getenv("CHILDREN_PER_WIN", "2"))
    # Chance a child comes from crossing two independently-successful
    # agents instead of pure self-mutation - lets a win also draw on a
    # second proven lineage's traits, not just perturb its own genome.
    crossover_probability: float = float(os.getenv("CROSSOVER_PROBABILITY", "0.3"))
    # Guarantees at least one agent from each coarse strategy "family"
    # (see agents/population.py::_family) stays among the active traders,
    # so one early lineage can't crowd out a genuinely different approach
    # before it's had a chance to prove out.
    diversity_floor_enabled: bool = _bool("DIVERSITY_FLOOR_ENABLED", True)
    win_streak_share_threshold: int = int(os.getenv("WIN_STREAK_SHARE_THRESHOLD", "8"))
    initial_population: int = int(os.getenv("INITIAL_POPULATION", "40"))
    starting_paper_balance: float = float(os.getenv("STARTING_PAPER_BALANCE", "1000"))

    # --- Ollama reasoning fallback (cloud API by default - no local model needed) ---
    ollama_enabled: bool = _bool("OLLAMA_ENABLED", True)
    ollama_host: str = os.getenv("OLLAMA_HOST", "https://ollama.com")
    ollama_api_key: str = os.getenv("OLLAMA_API_KEY", "")
    ollama_model: str = os.getenv("OLLAMA_MODEL", "gpt-oss:20b")

    # --- storage ---
    db_path: str = os.getenv("DB_PATH", "db/trading_agents.db")

    # --- dashboard ---
    dashboard_host: str = os.getenv("DASHBOARD_HOST", "127.0.0.1")
    dashboard_port: int = int(os.getenv("DASHBOARD_PORT", "8000"))

    fee_rate: float = 0.00035  # approx Hyperliquid taker fee
    slippage_bps: float = 2.0  # assumed slippage in basis points on paper fills

    def is_live(self) -> bool:
        if self.trading_mode.lower() != "live":
            return False
        if self.hl_network.lower() == "mainnet":
            return self.live_trading_confirmed == LIVE_CONFIRMATION_PHRASE
        return True  # live on testnet is fake money - no extra confirmation needed

    def is_paper(self) -> bool:
        return not self.is_live()

    def is_using_cloud_ollama(self) -> bool:
        return "ollama.com" in self.ollama_host


CONFIG = Config()
