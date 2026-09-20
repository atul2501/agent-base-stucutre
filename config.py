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
    timeframe: str = os.getenv("TIMEFRAME", "1m")
    cycle_seconds: int = int(os.getenv("CYCLE_SECONDS", "60"))
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
    # Normalized once here (stripped + lowercased) so every other place in
    # the codebase that checks "mainnet" vs "testnet" compares against this
    # single clean value instead of re-deriving it - three call sites used
    # to each do their own (inconsistent) .lower()/comparison, so a stray
    # space or a value like "Mainnet" could make the live-executor's network
    # pick disagree with is_live()'s mainnet-confirmation gate. __post_init__
    # below rejects anything that isn't exactly "testnet" or "mainnet".
    hl_network: str = os.getenv("HL_NETWORK", "testnet").strip().lower()
    hl_wallet_address: str = os.getenv("HYPERLIQUID_WALLET_ADDRESS", "")
    hl_private_key: str = os.getenv("HYPERLIQUID_PRIVATE_KEY", "")
    live_trading_confirmed: str = os.getenv("LIVE_TRADING_CONFIRMED", "")

    # --- live trading hard risk limits (only relevant once is_live() is True) ---
    live_max_total_notional_usd: float = float(os.getenv("LIVE_MAX_TOTAL_NOTIONAL_USD", "100"))
    live_active_trader_count: int = int(os.getenv("LIVE_ACTIVE_TRADER_COUNT", "20"))
    live_max_leverage: int = int(os.getenv("LIVE_MAX_LEVERAGE", "5"))
    # Circuit breaker: the one safeguard that actually protects real money
    # rather than just improving long-run statistical confidence. Trips on
    # EITHER a drawdown past this % from the peak account equity seen since
    # the breaker was last cleared, OR the price feed reporting the exact
    # same mid-price for this many consecutive cycles (a frozen/broken feed
    # is more dangerous than a merely-quiet market). On trip: the real
    # position is flattened immediately and live trading stays paused - not
    # just until conditions improve - until a human explicitly clears it
    # with `python3 main.py --clear-live-breaker`. See engine/orchestrator.py.
    live_circuit_breaker_enabled: bool = _bool("LIVE_CIRCUIT_BREAKER_ENABLED", True)
    live_max_drawdown_pct: float = float(os.getenv("LIVE_MAX_DRAWDOWN_PCT", "20.0"))
    live_stale_price_cycles: int = int(os.getenv("LIVE_STALE_PRICE_CYCLES", "5"))
    # Separate from the peak-drawdown trip above: catches a fast single-day
    # bleed that hasn't yet pulled 20% off the ALL-TIME peak (e.g. equity
    # was already down 15% from a prior peak, then loses another 10% today -
    # peak-drawdown alone wouldn't trip until 20% total). Resets at each new
    # UTC calendar day.
    live_max_daily_loss_pct: float = float(os.getenv("LIVE_MAX_DAILY_LOSS_PCT", "10.0"))
    # Optional webhook (Slack/Discord-compatible {"text": ...} payload, or
    # any endpoint that accepts JSON) pinged when the breaker trips - the
    # dashboard banner alone is silent if nobody's looking at it. Empty
    # disables; failure to reach it is logged, never blocks the trip itself.
    live_breaker_webhook_url: str = os.getenv("LIVE_BREAKER_WEBHOOK_URL", "")

    # --- paper-trading circuit breaker: the live one above only ever
    # protects the real aggregate position - the paper population (which
    # drives EVERY evolutionary decision: what genome an agent is born
    # with, who wins do-or-die, who gets promoted) previously had no
    # systemic guard at all. Do-or-die already bounds a single agent's own
    # risk, but nothing caught a correlated, swarm-wide bleed (e.g. a
    # broken feature or a regime nothing in the population handles well).
    # Tracks the swarm-wide REALIZED pnl curve (same number the dashboard
    # already shows - see db.realized_pnl_stats/population_cycles.
    # total_realized_pnl) rather than summed alive-agent balances, since
    # that sum is dominated by population-size churn (spawns/deaths) and
    # isn't a stable equity curve. Pauses NEW paper entries only - existing
    # open positions still exit normally - and auto-resumes once the
    # drawdown recovers to half the threshold (no manual clear needed,
    # unlike the live breaker: there's no real money at stake, so the cost
    # of being overly cautious here is a few paused training cycles, not
    # lost capital). See engine/orchestrator.py::_check_paper_drawdown.
    paper_circuit_breaker_enabled: bool = _bool("PAPER_CIRCUIT_BREAKER_ENABLED", True)
    paper_max_drawdown_usd: float = float(os.getenv("PAPER_MAX_DRAWDOWN_USD", "5000.0"))

    # --- population / evolution ---
    population_cap: int = int(os.getenv("POPULATION_CAP", "500"))
    active_trader_count: int = int(os.getenv("ACTIVE_TRADER_COUNT", "50"))
    min_population_floor: int = int(os.getenv("MIN_POPULATION_FLOOR", "50"))
    # Of active_trader_count slots, this many are reserved for the newest
    # untested (0-trade) agents regardless of fitness, so a brand-new agent
    # is guaranteed at least one real trade instead of potentially never
    # being picked once older/positive-fitness agents occupy every slot.
    guaranteed_newcomer_slots: int = int(os.getenv("GUARANTEED_NEWCOMER_SLOTS", "5"))
    # A second, distinct reservation from guaranteed_newcomer_slots above:
    # that one favors the NEWEST 0-trade agents (created_at DESC), so an
    # agent that's been benched with zero trades for a long time (not new
    # anymore, just never won a fitness-ranked slot) could otherwise wait
    # forever. This reserves slots for the OLDEST 0-trade agents instead
    # (created_at ASC), so nobody's shut out purely by how long they've
    # already been waiting.
    guaranteed_longest_benched_slots: int = int(os.getenv("GUARANTEED_LONGEST_BENCHED_SLOTS", "3"))
    # Every N calls to rank_and_enforce (~N cycles), force-swap the single
    # lowest-fitness active trader for one random benched 0-trade agent.
    # Without this, a slot held by fitness ranking alone can go to whoever
    # got a marginal early win and simply never lose it, since nothing else
    # ever re-tests whether a currently-benched agent might do better -
    # only ever touches one slot per window, so it can't meaningfully
    # undermine the fitness-based meritocracy (a genuinely weak agent gets
    # crowded back out next cycle by the normal fitness sort anyway).
    forced_rotation_every_n_cycles: int = int(os.getenv("FORCED_ROTATION_EVERY_N_CYCLES", "20"))
    # New agents are pre-screened against real historical data before they
    # enter the live do-or-die population (see backtest/engine.py) - this
    # doesn't change live trading at all, it just means an agent is BORN
    # with a better-than-random genome. Hyperliquid caps a single candle
    # request at ~5000 bars (~17 days of 5m data); default sits safely
    # under that.
    backtest_enabled: bool = _bool("BACKTEST_ENABLED", True)
    # At the default 1m timeframe, Hyperliquid's ~5000-bar cap covers only
    # ~83h - 80 stays safely under that. Raise this back up if TIMEFRAME is
    # set to something coarser (e.g. 360 at 1h, matching the old default).
    backtest_lookback_hours: int = int(os.getenv("BACKTEST_LOOKBACK_HOURS", "80"))
    backtest_candidates: int = int(os.getenv("BACKTEST_CANDIDATES", "5"))
    # A second, coarser-timeframe backtest window used ONLY to check that a
    # new candidate genome isn't just tuned to whatever regime happened in
    # the last few days (backtest_lookback_hours above) - split into several
    # segments and screened separately so a genome that only works in one
    # regime (e.g. a strong uptrend) doesn't win by default. Coarser
    # timeframe (default 1h) so the same ~5000-bar API cap covers months
    # instead of days. See agents/population.py::_pick_best.
    backtest_regime_timeframe: str = os.getenv("BACKTEST_REGIME_TIMEFRAME", "1h")
    backtest_regime_lookback_hours: int = int(os.getenv("BACKTEST_REGIME_LOOKBACK_HOURS", "4000"))
    backtest_regime_segments: int = int(os.getenv("BACKTEST_REGIME_SEGMENTS", "3"))
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
    # Floor-refill candidates drawn from hall_of_fame at this rate get a real
    # (not just probable) shot at reinstating a genuinely all-time-best
    # genome's lineage, since do-or-die means even that genome's original
    # agent eventually dies on one loss like anyone else. Lightly mutated
    # (not used bit-for-bit unmutated) before competing in _pick_best - a
    # genome-quality review found repeated unmutated HOF draws were the
    # direct cause of exact-duplicate clone clusters in the population;
    # a small mutation keeps the near-peak "shot" without ever stamping out
    # a second bit-for-bit copy of an already-alive agent. Also lowered from
    # a prior 0.25 default for the same reason - still screened by the same
    # backtest candidate selection as every other new agent, not committed
    # blind.
    hall_of_fame_exact_clone_rate: float = float(os.getenv("HALL_OF_FAME_EXACT_CLONE_RATE", "0.10"))
    initial_population: int = int(os.getenv("INITIAL_POPULATION", "40"))
    starting_paper_balance: float = float(os.getenv("STARTING_PAPER_BALANCE", "1000"))
    # Folder of hand-picked agent genomes (see `python3 main.py --snapshot-best`)
    # that get reloaded to seed any future FRESH population (empty DB, or after
    # --reset) - a head start instead of starting every fresh run from pure
    # random genomes. Never touches an already-running population - see
    # agents/population.py::seed_if_empty.
    snapshot_dir: str = os.getenv("SNAPSHOT_DIR", "snapshots")

    # --- Revalidation: periodic re-backtest of already-proven agents ---
    # A veteran agent proves itself once (a real win) and then keeps trading
    # indefinitely until it randomly loses - nothing re-checks whether it
    # still fits the CURRENT market. This periodically re-backtests the
    # top-fitness alive agents against the freshest recent-window data and
    # flags (dashboard) any whose genome has drifted out of sync. It never
    # touches an agent's stored status or fitness, and never kills or culls
    # anyone - the live do-or-die mechanic is completely unchanged by this.
    # It DOES apply a ranking-only penalty (see revalidation_drift_penalty_factor
    # below and Population._effective_rank_score) when choosing active/live
    # traders each cycle, so a drifted agent can lose its slot without its
    # underlying fitness number or dashboard history being altered. See
    # agents/population.py::revalidate_top_agents.
    revalidation_enabled: bool = _bool("REVALIDATION_ENABLED", True)
    revalidation_interval_cycles: int = int(os.getenv("REVALIDATION_INTERVAL_CYCLES", "20"))
    revalidation_agent_limit: int = int(os.getenv("REVALIDATION_AGENT_LIMIT", "50"))
    # A re-backtest score below this is flagged as "drifted" on the dashboard.
    revalidation_drift_threshold: float = float(os.getenv("REVALIDATION_DRIFT_THRESHOLD", "-5.0"))
    # Ranking-only penalty applied per point a drifted agent's revalidation
    # score falls below the threshold above - e.g. scoring -15 against a
    # -5.0 threshold (10 points past it) loses 10*this many points off its
    # EFFECTIVE rank score used for active/live-trader slot selection only
    # (see Population._effective_rank_score). Proportional rather than a
    # hard exclude, since revalidation runs on a sample and a hard cutoff
    # right at the threshold would be noisy.
    revalidation_drift_penalty_factor: float = float(os.getenv("REVALIDATION_DRIFT_PENALTY_FACTOR", "2.0"))
    # The recent-window backtest data (used both for new-agent screening and
    # for revalidation above) is only ever fetched once at startup otherwise -
    # stale after the first few hours of a long-running deployment. Refetched
    # on this cadence so "against the newest data" stays true over time.
    backtest_refresh_interval_cycles: int = int(os.getenv("BACKTEST_REFRESH_INTERVAL_CYCLES", "20"))

    # --- Council vote: a cheap ensemble second opinion for ambiguous signals ---
    # When an agent's own rule-based signal is ambiguous (score 0-2), poll the
    # current top-fitness alive agents' OWN genomes against the same market
    # snapshot before ever calling Ollama - if enough of them independently
    # reach the same directional call, that's a stronger, free, and
    # rate-limit-proof confirmation than one LLM guess. Ollama (below) is
    # only consulted afterward, as a tie-breaker for whatever the council
    # itself couldn't resolve. See strategy/signals.py::council_consult.
    council_enabled: bool = _bool("COUNCIL_ENABLED", True)
    council_size: int = int(os.getenv("COUNCIL_SIZE", "10"))
    # Fraction of ACTIVE (non-hold) council votes needed to confirm or veto
    # the candidate direction - a hold vote just means that agent's own
    # unrelated genome didn't trigger, not that it disagrees, so holds are
    # excluded from the quorum math entirely (see council_consult).
    council_quorum_pct: float = float(os.getenv("COUNCIL_QUORUM_PCT", "0.6"))
    # Below this many active (long/short) votes, the poll is too thin to mean
    # anything - falls through to Ollama/hold instead of a shaky "majority of 2".
    council_min_active_voters: int = int(os.getenv("COUNCIL_MIN_ACTIVE_VOTERS", "3"))

    # --- Exit council: optional, tighten-only advisory layer on top of the
    # deterministic ATR-adaptive stop/trailing system above. When enabled, an
    # OPEN position with nothing else to do this cycle (no partial/trail/close
    # fired) gets an extra check: does the same council ensemble (and, if
    # inconclusive, Ollama) now favor the OPPOSITE side? If so, pull the stop
    # tighter - never loosen it, never force a close. Off by default; never
    # invoked from the backtester, since a live council/LLM opinion can't be
    # replayed historically (see engine/orchestrator.py::_process_exits and
    # strategy/signals.py::council_oppose_position).
    exit_council_enabled: bool = _bool("EXIT_COUNCIL_ENABLED", False)
    # How far from the current stop toward the current price to pull it when
    # the council opposes the held position - 0 is a no-op, 1 pulls the stop
    # all the way to current price.
    exit_council_tighten_frac: float = float(os.getenv("EXIT_COUNCIL_TIGHTEN_FRAC", "0.5"))

    # --- Ollama reasoning fallback (cloud API by default - no local model needed) ---
    ollama_enabled: bool = _bool("OLLAMA_ENABLED", True)
    ollama_host: str = os.getenv("OLLAMA_HOST", "https://ollama.com")
    ollama_api_key: str = os.getenv("OLLAMA_API_KEY", "")
    ollama_model: str = os.getenv("OLLAMA_MODEL", "gpt-oss:20b")
    # consult() runs synchronously inside the trading cycle - a hung request
    # must fail fast into the "hold" fallback rather than stall the cycle.
    ollama_timeout_seconds: float = float(os.getenv("OLLAMA_TIMEOUT_SECONDS", "15"))

    # --- storage ---
    db_path: str = os.getenv("DB_PATH", "db/trading_agents.db")

    # --- logging ---
    # Log files (logs/agent_swarm.log + its rotated .1/.2/.3 backups, plus
    # systemd.log/systemd-error.log if run under systemd) older than this
    # are deleted automatically - see main.py::_cleanup_old_logs. Runs once
    # at startup and once per hour while running, so a long-lived deployment
    # doesn't need an external logrotate/cron job to avoid unbounded disk
    # growth.
    log_max_age_hours: float = float(os.getenv("LOG_MAX_AGE_HOURS", "48"))

    # --- dashboard ---
    dashboard_host: str = os.getenv("DASHBOARD_HOST", "127.0.0.1")
    dashboard_port: int = int(os.getenv("DASHBOARD_PORT", "8000"))

    fee_rate: float = 0.00035  # approx Hyperliquid taker fee
    slippage_bps: float = 2.0  # assumed slippage in basis points on paper fills

    def __post_init__(self) -> None:
        if self.hl_network not in ("testnet", "mainnet"):
            raise ValueError(
                f"HL_NETWORK={self.hl_network!r} is invalid - must be exactly "
                f"'testnet' or 'mainnet'. Refusing to guess which network "
                f"real orders (if TRADING_MODE=live) would go to."
            )

    def is_live(self) -> bool:
        if self.trading_mode.lower() != "live":
            return False
        if self.hl_network == "mainnet":
            return self.live_trading_confirmed == LIVE_CONFIRMATION_PHRASE
        return True  # live on testnet is fake money - no extra confirmation needed

    def is_using_cloud_ollama(self) -> bool:
        return "ollama.com" in self.ollama_host


CONFIG = Config()
