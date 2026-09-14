CREATE TABLE IF NOT EXISTS agents (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    parent_id INTEGER,
    generation INTEGER NOT NULL DEFAULT 0,
    genome_json TEXT NOT NULL,
    status TEXT NOT NULL DEFAULT 'alive',        -- alive | dead
    tier TEXT NOT NULL DEFAULT 'standard',       -- standard | professional
    is_active_trader INTEGER NOT NULL DEFAULT 0,
    balance REAL NOT NULL,
    wins INTEGER NOT NULL DEFAULT 0,
    losses INTEGER NOT NULL DEFAULT 0,
    win_streak INTEGER NOT NULL DEFAULT 0,
    total_pnl REAL NOT NULL DEFAULT 0,
    trades_count INTEGER NOT NULL DEFAULT 0,
    created_at TEXT NOT NULL,
    died_at TEXT,
    death_reason TEXT,
    FOREIGN KEY (parent_id) REFERENCES agents(id)
);

CREATE INDEX IF NOT EXISTS idx_agents_status ON agents(status);
CREATE INDEX IF NOT EXISTS idx_agents_parent ON agents(parent_id);

CREATE TABLE IF NOT EXISTS trades (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    agent_id INTEGER NOT NULL,
    coin TEXT NOT NULL,
    side TEXT NOT NULL,                          -- long | short
    entry_price REAL NOT NULL,
    exit_price REAL,
    size REAL NOT NULL,
    notional REAL NOT NULL,
    stop_loss REAL,
    take_profit REAL,
    pnl REAL,
    result TEXT NOT NULL DEFAULT 'open',         -- open | win | loss
    entry_reason TEXT,
    exit_reason TEXT,
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

CREATE TABLE IF NOT EXISTS population_cycles (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    cycle INTEGER NOT NULL,
    alive_count INTEGER NOT NULL,
    active_trader_count INTEGER NOT NULL,
    professional_count INTEGER NOT NULL,
    best_agent_id INTEGER,
    best_total_pnl REAL,
    ran_at TEXT NOT NULL
);
