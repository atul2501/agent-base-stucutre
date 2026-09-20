CREATE TABLE IF NOT EXISTS agents (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    parent_id INTEGER,
    generation INTEGER NOT NULL DEFAULT 0,
    genome_json TEXT NOT NULL,
    status TEXT NOT NULL DEFAULT 'alive' CHECK (status IN ('alive', 'dead')),
    tier TEXT NOT NULL DEFAULT 'standard' CHECK (tier IN ('standard', 'professional')),
    is_active_trader INTEGER NOT NULL DEFAULT 0,
    is_live_trader INTEGER NOT NULL DEFAULT 0,
    balance REAL NOT NULL,
    wins INTEGER NOT NULL DEFAULT 0,
    losses INTEGER NOT NULL DEFAULT 0,
    win_streak INTEGER NOT NULL DEFAULT 0,
    total_pnl REAL NOT NULL DEFAULT 0,
    trades_count INTEGER NOT NULL DEFAULT 0,
    created_at TEXT NOT NULL,
    died_at TEXT,
    death_reason TEXT,
    -- Periodic re-backtest of an already-proven agent's genome against the
    -- NEWEST market data (see agents/population.py::revalidate_top_agents) -
    -- purely informational, does not affect the live do-or-die mechanic or
    -- active-trader ranking. Lets a drifted-out-of-sync veteran be flagged
    -- on the dashboard instead of silently coasting until it randomly loses.
    revalidation_score REAL,
    revalidated_at TEXT,
    FOREIGN KEY (parent_id) REFERENCES agents(id)
);

CREATE INDEX IF NOT EXISTS idx_agents_status ON agents(status);
CREATE INDEX IF NOT EXISTS idx_agents_parent ON agents(parent_id);

CREATE TABLE IF NOT EXISTS trades (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    agent_id INTEGER NOT NULL,
    coin TEXT NOT NULL,
    side TEXT NOT NULL CHECK (side IN ('long', 'short')),
    entry_price REAL NOT NULL,
    exit_price REAL,
    size REAL NOT NULL,
    notional REAL NOT NULL,
    stop_loss REAL,
    take_profit REAL,
    pnl REAL,
    result TEXT NOT NULL DEFAULT 'open' CHECK (result IN ('open', 'win', 'loss')),
    entry_reason TEXT,
    exit_reason TEXT,
    regime TEXT,                                 -- trending-up | trending-down | ranging, at entry
    entry_funding REAL DEFAULT 0.0,              -- funding rate in effect at entry, for approximating hold cost at close
    -- ATR-adaptive/trailing/partial exit bookkeeping (see
    -- strategy/signals.py::evaluate_position). remaining_size shrinks on a
    -- partial close; stop_loss (above) is mutated in place as the trailing
    -- stop tightens and moved to breakeven after a partial fires.
    remaining_size REAL,
    partial_target REAL,                         -- price level for the (optional) partial take-profit; NULL if unused
    partial_taken INTEGER NOT NULL DEFAULT 0,
    partial_frac_taken REAL NOT NULL DEFAULT 0.0, -- fraction of the ORIGINAL size closed by the partial, for _blended_result
    partial_pnl_pct REAL NOT NULL DEFAULT 0.0,    -- pnl_pct at which the partial closed, for _blended_result
    partial_realized_pnl REAL NOT NULL DEFAULT 0.0, -- $ pnl already credited to the agent at partial-close time, folded into trades.pnl at final close
    opened_at TEXT NOT NULL,
    closed_at TEXT,
    FOREIGN KEY (agent_id) REFERENCES agents(id)
);

CREATE INDEX IF NOT EXISTS idx_trades_agent ON trades(agent_id);
CREATE INDEX IF NOT EXISTS idx_trades_result ON trades(result);

CREATE TABLE IF NOT EXISTS strategy_shares (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    source_agent_id INTEGER NOT NULL,
    genome_json TEXT NOT NULL,
    win_streak_at_share INTEGER NOT NULL,
    total_pnl_at_share REAL NOT NULL,
    shared_at TEXT NOT NULL
);

-- Preserves an exceptional genome EXACTLY, unlike strategy_shares (which
-- is always mutated before reuse). Even a top-tier agent dies on its next
-- loss like anyone else under do-or-die - this is the only place a
-- genuinely all-time-best genome survives that death byte-for-byte,
-- available to be re-tested unmutated later (see
-- Population.refill_if_below_floor / hall_of_fame_exact_clone_rate).
CREATE TABLE IF NOT EXISTS hall_of_fame (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    source_agent_id INTEGER NOT NULL,
    genome_json TEXT NOT NULL,
    win_streak INTEGER NOT NULL,
    total_pnl REAL NOT NULL,
    fitness REAL NOT NULL,
    recorded_at TEXT NOT NULL,
    reason TEXT NOT NULL  -- 'new_all_time_high_fitness'
);

CREATE INDEX IF NOT EXISTS idx_hof_fitness ON hall_of_fame(fitness);

CREATE TABLE IF NOT EXISTS population_cycles (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    cycle INTEGER NOT NULL,
    alive_count INTEGER NOT NULL,
    active_trader_count INTEGER NOT NULL,
    professional_count INTEGER NOT NULL,
    best_agent_id INTEGER,
    best_total_pnl REAL,
    -- Swarm-wide realized PnL to date at this cycle (sum of all closed
    -- trades' pnl so far) - a real equity curve, unlike best_total_pnl
    -- which jumps around as WHICH agent is currently ranked #1 changes
    -- under the do-or-die lifecycle. See dashboard's "Best Agent PnL by
    -- Cycle" chart.
    total_realized_pnl REAL,
    ran_at TEXT NOT NULL
);

-- Simple key/value store: tracks which token the population was built for
-- (see main.py --reset), so switching tokens without an explicit reset is
-- refused rather than silently mixing strategies learned on a different asset.
CREATE TABLE IF NOT EXISTS meta (
    key TEXT PRIMARY KEY,
    value TEXT
);

-- Real Hyperliquid perp accounts hold ONE net position per (account, coin),
-- so "top N agents trading live" is represented as one aggregate real
-- position that tracks the net long/short consensus of those agents' paper
-- positions (see README - Live trading design). This row is that ground
-- truth mirror, reconciled against the real exchange each cycle.
CREATE TABLE IF NOT EXISTS live_position (
    id INTEGER PRIMARY KEY CHECK (id = 1),
    coin TEXT NOT NULL,
    side TEXT CHECK (side IN ('long', 'short') OR side IS NULL),
    size REAL NOT NULL DEFAULT 0,
    notional REAL NOT NULL DEFAULT 0,
    updated_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS live_orders (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    coin TEXT NOT NULL,
    action TEXT NOT NULL CHECK (action IN ('open', 'increase', 'decrease', 'close', 'flip_close', 'flip_open')),
    side TEXT NOT NULL CHECK (side IN ('long', 'short')),
    notional REAL NOT NULL,
    fill_price REAL,
    status TEXT NOT NULL CHECK (status IN ('filled', 'error')),
    detail TEXT,
    placed_at TEXT NOT NULL
);
