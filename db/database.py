"""SQLite persistence layer for agents, trades, and shared strategies."""
from __future__ import annotations

import json
import math
import sqlite3
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

SCHEMA_PATH = Path(__file__).parent / "schema.sql"


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def fitness_score(total_pnl: float, balance: float, wins: int) -> float:
    """Shared fitness formula for AgentRow.fitness and the dashboard
    leaderboard - factored out after the two drifted out of sync once
    already when the dashboard carried its own inline copy."""
    starting_balance = balance - total_pnl
    pct_return = (total_pnl / starting_balance * 100.0) if starting_balance > 0 else 0.0
    return pct_return + math.log1p(wins) * 10.0


@dataclass
class AgentRow:
    id: int
    parent_id: Optional[int]
    generation: int
    genome_json: str
    status: str
    tier: str
    is_active_trader: bool
    balance: float
    wins: int
    losses: int
    win_streak: int
    total_pnl: float
    trades_count: int
    created_at: str
    died_at: Optional[str]
    death_reason: Optional[str]
    revalidation_score: Optional[float]
    revalidated_at: Optional[str]

    @property
    def genome(self) -> dict:
        return json.loads(self.genome_json)

    @property
    def win_rate(self) -> float:
        total = self.wins + self.losses
        return self.wins / total if total else 0.0

    @property
    def fitness(self) -> float:
        # Ranking score used for population-cap culling and top-N trader
        # selection. Normalized as % return on starting balance (not raw
        # PnL dollars) so agents are compared on risk-adjusted performance
        # rather than whoever happened to size a bigger position - a $5
        # profit on a $50 starting balance ranks above a $5 profit on a
        # $5000 one. `balance - total_pnl` recovers the starting balance
        # without needing to store it separately (balance is only ever
        # mutated by +=pnl in record_win/record_loss_and_kill).
        #
        # win_rate is not used here: under the do-or-die rule a single loss
        # kills the agent, so every alive agent has losses == 0 and win_rate
        # is always exactly 0.0 or 1.0 - it can't distinguish a 1-win agent
        # from a 20-win one. `wins` (== win_streak while alive) is used
        # instead, with log1p so the bonus grows with a proven track record
        # but with diminishing returns rather than unbounded linear growth.
        return fitness_score(self.total_pnl, self.balance, self.wins)


class Database:
    def __init__(self, db_path: str):
        parent = Path(db_path).parent
        if str(parent) not in ("", "."):
            parent.mkdir(parents=True, exist_ok=True)
        self.conn = sqlite3.connect(db_path)
        self.conn.row_factory = sqlite3.Row
        self.conn.execute("PRAGMA foreign_keys = ON")
        self.conn.execute("PRAGMA journal_mode = WAL")  # lets the dashboard read concurrently
        self._init_schema()

    def _init_schema(self) -> None:
        with open(SCHEMA_PATH) as f:
            self.conn.executescript(f.read())
        self.conn.commit()
        self._migrate()

    def _migrate(self) -> None:
        """Additive schema changes for existing databases - CREATE TABLE IF
        NOT EXISTS above is a no-op on a table that already exists, so any
        new column needs an explicit ALTER here. Keeps old agent/trade
        history intact instead of requiring --reset."""
        existing_cols = {row["name"] for row in self.conn.execute("PRAGMA table_info(trades)")}
        if "regime" not in existing_cols:
            self.conn.execute("ALTER TABLE trades ADD COLUMN regime TEXT")
            self.conn.commit()
        if "entry_funding" not in existing_cols:
            # The funding rate in effect at entry - used at close time to
            # approximate total funding paid/received over the hold (average
            # of entry and exit rate * notional * hours held). NULL/0.0 for
            # existing open trades predating this column just means their
            # funding cost is approximated as 0 for whatever portion of the
            # hold already elapsed - not retroactively knowable.
            self.conn.execute("ALTER TABLE trades ADD COLUMN entry_funding REAL DEFAULT 0.0")
            self.conn.commit()
        if "remaining_size" not in existing_cols:
            # ATR-adaptive/trailing/partial exit bookkeeping - see
            # strategy/signals.py::evaluate_position and schema.sql. An
            # existing OPEN trade predating this column has remaining_size
            # backfilled to its full original size (no partial could have
            # happened before this feature existed); a closed trade's
            # remaining_size is irrelevant and left NULL.
            self.conn.execute("ALTER TABLE trades ADD COLUMN remaining_size REAL")
            self.conn.execute("ALTER TABLE trades ADD COLUMN partial_target REAL")
            self.conn.execute("ALTER TABLE trades ADD COLUMN partial_taken INTEGER NOT NULL DEFAULT 0")
            self.conn.execute("ALTER TABLE trades ADD COLUMN partial_frac_taken REAL NOT NULL DEFAULT 0.0")
            self.conn.execute("ALTER TABLE trades ADD COLUMN partial_pnl_pct REAL NOT NULL DEFAULT 0.0")
            self.conn.execute("ALTER TABLE trades ADD COLUMN partial_realized_pnl REAL NOT NULL DEFAULT 0.0")
            self.conn.execute(
                "UPDATE trades SET remaining_size = size WHERE result = 'open' AND remaining_size IS NULL"
            )
            self.conn.commit()

        agent_cols = {row["name"] for row in self.conn.execute("PRAGMA table_info(agents)")}
        if "revalidation_score" not in agent_cols:
            self.conn.execute("ALTER TABLE agents ADD COLUMN revalidation_score REAL")
            self.conn.execute("ALTER TABLE agents ADD COLUMN revalidated_at TEXT")
            self.conn.commit()

        cycle_cols = {row["name"] for row in self.conn.execute("PRAGMA table_info(population_cycles)")}
        if "total_realized_pnl" not in cycle_cols:
            self.conn.execute("ALTER TABLE population_cycles ADD COLUMN total_realized_pnl REAL")
            self.conn.commit()

        # live_orders.action CHECK originally omitted 'flip_close'/'flip_open',
        # which is exactly what a real position-side flip inserts - every
        # flip raised an uncaught IntegrityError right after a real order was
        # already placed on the exchange. SQLite can't ALTER a CHECK
        # constraint, so rebuild the table on any DB still carrying the old
        # constraint text; a fresh DB gets the fixed constraint straight from
        # schema.sql and never hits this branch.
        lo_row = self.conn.execute(
            "SELECT sql FROM sqlite_master WHERE type='table' AND name='live_orders'"
        ).fetchone()
        if lo_row and "flip_close" not in lo_row["sql"]:
            self.conn.executescript(
                """
                ALTER TABLE live_orders RENAME TO live_orders_old;
                CREATE TABLE live_orders (
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
                INSERT INTO live_orders (id, coin, action, side, notional, fill_price, status, detail, placed_at)
                    SELECT id, coin, action, side, notional, fill_price, status, detail, placed_at FROM live_orders_old;
                DROP TABLE live_orders_old;
                """
            )
            self.conn.commit()

    def close(self) -> None:
        self.conn.close()

    # ---- agents ----

    def create_agent(
        self,
        genome: dict,
        balance: float,
        parent_id: Optional[int] = None,
        generation: int = 0,
        tier: str = "standard",
    ) -> int:
        cur = self.conn.execute(
            """INSERT INTO agents (parent_id, generation, genome_json, balance, tier, created_at)
               VALUES (?, ?, ?, ?, ?, ?)""",
            (parent_id, generation, json.dumps(genome), balance, tier, now_iso()),
        )
        self.conn.commit()
        return cur.lastrowid

    def get_agent(self, agent_id: int) -> Optional[AgentRow]:
        row = self.conn.execute("SELECT * FROM agents WHERE id = ?", (agent_id,)).fetchone()
        return self._row_to_agent(row) if row else None

    def list_alive_agents(self) -> list[AgentRow]:
        rows = self.conn.execute("SELECT * FROM agents WHERE status = 'alive'").fetchall()
        return [self._row_to_agent(r) for r in rows]

    def list_active_traders(self) -> list[AgentRow]:
        rows = self.conn.execute(
            "SELECT * FROM agents WHERE status = 'alive' AND is_active_trader = 1"
        ).fetchall()
        return [self._row_to_agent(r) for r in rows]

    def count_alive(self) -> int:
        return self.conn.execute(
            "SELECT COUNT(*) FROM agents WHERE status = 'alive'"
        ).fetchone()[0]

    def count_professional(self) -> int:
        return self.conn.execute(
            "SELECT COUNT(*) FROM agents WHERE status = 'alive' AND tier = 'professional'"
        ).fetchone()[0]

    def children_of(self, parent_id: int) -> list[AgentRow]:
        # Ordered by id (creation order) so callers that take winners[:N] get
        # a deterministic, reproducible pick rather than whatever order
        # SQLite happens to return.
        rows = self.conn.execute(
            "SELECT * FROM agents WHERE parent_id = ? ORDER BY id", (parent_id,)
        ).fetchall()
        return [self._row_to_agent(r) for r in rows]

    def set_active_traders(self, agent_ids: set[int]) -> None:
        self.conn.execute("UPDATE agents SET is_active_trader = 0 WHERE status = 'alive'")
        if agent_ids:
            qmarks = ",".join("?" for _ in agent_ids)
            self.conn.execute(
                f"UPDATE agents SET is_active_trader = 1 WHERE id IN ({qmarks})",
                tuple(agent_ids),
            )
        self.conn.commit()

    def list_live_traders(self) -> list[AgentRow]:
        rows = self.conn.execute(
            "SELECT * FROM agents WHERE status = 'alive' AND is_live_trader = 1"
        ).fetchall()
        return [self._row_to_agent(r) for r in rows]

    def set_live_traders(self, agent_ids: set[int]) -> None:
        self.conn.execute("UPDATE agents SET is_live_trader = 0 WHERE status = 'alive'")
        if agent_ids:
            qmarks = ",".join("?" for _ in agent_ids)
            self.conn.execute(
                f"UPDATE agents SET is_live_trader = 1 WHERE id IN ({qmarks})",
                tuple(agent_ids),
            )
        self.conn.commit()

    def set_tier(self, agent_id: int, tier: str) -> None:
        self.conn.execute("UPDATE agents SET tier = ? WHERE id = ?", (tier, agent_id))
        self.conn.commit()

    def set_revalidation(self, agent_id: int, score: float) -> None:
        """Records the result of re-backtesting an already-proven agent's
        genome against the newest data - informational only (see
        agents/population.py::revalidate_top_agents), never touches status,
        fitness, or the do-or-die kill mechanic."""
        self.conn.execute(
            "UPDATE agents SET revalidation_score = ?, revalidated_at = ? WHERE id = ?",
            (score, now_iso(), agent_id),
        )
        self.conn.commit()

    def record_win(self, agent_id: int, pnl: float) -> None:
        self.conn.execute(
            """UPDATE agents
               SET wins = wins + 1, win_streak = win_streak + 1,
                   total_pnl = total_pnl + ?, balance = balance + ?,
                   trades_count = trades_count + 1
               WHERE id = ?""",
            (pnl, pnl, agent_id),
        )
        self.conn.commit()

    def record_loss_and_kill(self, agent_id: int, pnl: float) -> None:
        self.conn.execute(
            """UPDATE agents
               SET losses = losses + 1, win_streak = 0,
                   total_pnl = total_pnl + ?, balance = balance + ?,
                   trades_count = trades_count + 1,
                   status = 'dead', died_at = ?, death_reason = 'losing trade'
               WHERE id = ?""",
            (pnl, pnl, now_iso(), agent_id),
        )
        self.conn.commit()

    def kill_agent(self, agent_id: int, reason: str) -> None:
        self.conn.execute(
            "UPDATE agents SET status = 'dead', died_at = ?, death_reason = ? WHERE id = ?",
            (now_iso(), reason, agent_id),
        )
        self.conn.commit()

    def _row_to_agent(self, row: sqlite3.Row) -> AgentRow:
        return AgentRow(
            id=row["id"],
            parent_id=row["parent_id"],
            generation=row["generation"],
            genome_json=row["genome_json"],
            status=row["status"],
            tier=row["tier"],
            is_active_trader=bool(row["is_active_trader"]),
            balance=row["balance"],
            wins=row["wins"],
            losses=row["losses"],
            win_streak=row["win_streak"],
            total_pnl=row["total_pnl"],
            trades_count=row["trades_count"],
            created_at=row["created_at"],
            died_at=row["died_at"],
            death_reason=row["death_reason"],
            revalidation_score=row["revalidation_score"],
            revalidated_at=row["revalidated_at"],
        )

    # ---- trades ----

    def open_trade(
        self,
        agent_id: int,
        coin: str,
        side: str,
        entry_price: float,
        size: float,
        notional: float,
        stop_loss: float,
        take_profit: float,
        entry_reason: str,
        regime: str | None = None,
        entry_funding: float = 0.0,
        partial_target: float | None = None,
    ) -> int:
        cur = self.conn.execute(
            """INSERT INTO trades
               (agent_id, coin, side, entry_price, size, notional, stop_loss, take_profit,
                entry_reason, regime, entry_funding, remaining_size, partial_target, opened_at, result)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 'open')""",
            (agent_id, coin, side, entry_price, size, notional, stop_loss, take_profit,
             entry_reason, regime, entry_funding, size, partial_target, now_iso()),
        )
        self.conn.commit()
        return cur.lastrowid

    def get_open_trade(self, agent_id: int) -> Optional[sqlite3.Row]:
        return self.conn.execute(
            "SELECT * FROM trades WHERE agent_id = ? AND result = 'open'", (agent_id,)
        ).fetchone()

    def close_trade(self, trade_id: int, exit_price: float, pnl: float, result: str, exit_reason: str) -> None:
        """`pnl` is the TOTAL trade pnl (any partial-close pnl already
        applied via apply_partial_pnl, plus this final leg) - see
        engine/orchestrator.py::_process_exits and
        backtest/engine.py::backtest_genome for why the two must be summed
        here rather than double-applying the partial portion."""
        self.conn.execute(
            """UPDATE trades SET exit_price = ?, pnl = ?, result = ?, exit_reason = ?, closed_at = ?
               WHERE id = ?""",
            (exit_price, pnl, result, exit_reason, now_iso(), trade_id),
        )
        self.conn.commit()

    def update_trade_stop(self, trade_id: int, new_stop_loss: float) -> None:
        """Trailing-stop tightening - see strategy/signals.py::evaluate_position."""
        self.conn.execute("UPDATE trades SET stop_loss = ? WHERE id = ?", (new_stop_loss, trade_id))
        self.conn.commit()

    def record_partial_close(self, trade_id: int, remaining_size: float, partial_frac_taken: float,
                              partial_pnl_pct: float, partial_realized_pnl: float, new_stop_loss: float) -> None:
        """Persists a partial take-profit: shrinks remaining_size, records
        the fraction/pnl_pct needed to blend the final result label (see
        strategy/signals.py::_blended_result), moves the stop to breakeven,
        and marks partial_taken so it can't fire twice. partial_realized_pnl
        (the $ amount, already credited to the agent via apply_partial_pnl)
        is stored here purely so the final close can fold it into
        trades.pnl for accurate trade-history reporting - it is NOT
        re-applied to the agent's balance at that point."""
        self.conn.execute(
            """UPDATE trades
               SET remaining_size = ?, partial_taken = 1, partial_frac_taken = ?,
                   partial_pnl_pct = ?, partial_realized_pnl = partial_realized_pnl + ?, stop_loss = ?
               WHERE id = ?""",
            (remaining_size, partial_frac_taken, partial_pnl_pct, partial_realized_pnl, new_stop_loss, trade_id),
        )
        self.conn.commit()

    def apply_partial_pnl(self, agent_id: int, pnl: float) -> None:
        """Immediately credits a partial take-profit's realized $ pnl to
        the agent's balance AND total_pnl (a real economic event - it
        should count in both right away, the same as a full close would).
        Does NOT touch wins/losses/trades_count/win_streak - those are
        still driven solely by the trade's eventual final-close result,
        same do-or-die mechanic as before this existed."""
        self.conn.execute(
            "UPDATE agents SET balance = balance + ?, total_pnl = total_pnl + ? WHERE id = ?",
            (pnl, pnl, agent_id),
        )
        self.conn.commit()

    # ---- strategy shares ----

    def record_strategy_share(self, source_agent_id: int, genome: dict, win_streak: int, total_pnl: float) -> None:
        self.conn.execute(
            """INSERT INTO strategy_shares
               (source_agent_id, genome_json, win_streak_at_share, total_pnl_at_share, shared_at)
               VALUES (?, ?, ?, ?, ?)""",
            (source_agent_id, json.dumps(genome), win_streak, total_pnl, now_iso()),
        )
        self.conn.commit()

    def sample_shared_genome(self) -> Optional[dict]:
        row = self.conn.execute(
            "SELECT genome_json FROM strategy_shares ORDER BY RANDOM() LIMIT 1"
        ).fetchone()
        return json.loads(row["genome_json"]) if row else None

    # ---- hall of fame (exact-genome preservation, unlike strategy_shares) ----

    def get_max_hall_of_fame_fitness(self) -> float:
        row = self.conn.execute("SELECT MAX(fitness) AS m FROM hall_of_fame").fetchone()
        return row["m"] if row and row["m"] is not None else 0.0

    def record_hall_of_fame(self, source_agent_id: int, genome: dict, win_streak: int,
                             total_pnl: float, fitness: float, reason: str) -> None:
        self.conn.execute(
            """INSERT INTO hall_of_fame
               (source_agent_id, genome_json, win_streak, total_pnl, fitness, recorded_at, reason)
               VALUES (?, ?, ?, ?, ?, ?, ?)""",
            (source_agent_id, json.dumps(genome), win_streak, total_pnl, fitness, now_iso(), reason),
        )
        self.conn.commit()

    def sample_hall_of_fame_genome(self) -> Optional[dict]:
        row = self.conn.execute(
            "SELECT genome_json FROM hall_of_fame ORDER BY RANDOM() LIMIT 1"
        ).fetchone()
        return json.loads(row["genome_json"]) if row else None

    # ---- population cycle log ----

    def record_population_cycle(
        self,
        cycle: int,
        alive_count: int,
        active_trader_count: int,
        professional_count: int,
        best_agent_id: Optional[int],
        best_total_pnl: Optional[float],
    ) -> None:
        # Swarm-wide realized PnL to date, computed fresh each cycle rather
        # than threaded through as a running total - cheap (indexed by
        # result) and can never drift from the trades table's ground truth.
        total_realized_pnl = self.conn.execute(
            "SELECT COALESCE(SUM(pnl), 0) FROM trades WHERE result IN ('win', 'loss')"
        ).fetchone()[0]
        self.conn.execute(
            """INSERT INTO population_cycles
               (cycle, alive_count, active_trader_count, professional_count,
                best_agent_id, best_total_pnl, total_realized_pnl, ran_at)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
            (cycle, alive_count, active_trader_count, professional_count,
             best_agent_id, best_total_pnl, total_realized_pnl, now_iso()),
        )
        self.conn.commit()

    def recent_population_cycles(self, limit: int = 200) -> list[sqlite3.Row]:
        return self.conn.execute(
            "SELECT * FROM population_cycles ORDER BY id DESC LIMIT ?", (limit,)
        ).fetchall()

    # ---- meta (single-token guard / step-0 reset) ----

    def get_meta(self, key: str) -> Optional[str]:
        row = self.conn.execute("SELECT value FROM meta WHERE key = ?", (key,)).fetchone()
        return row["value"] if row else None

    def set_meta(self, key: str, value: str) -> None:
        self.conn.execute(
            "INSERT INTO meta (key, value) VALUES (?, ?) "
            "ON CONFLICT(key) DO UPDATE SET value = excluded.value",
            (key, value),
        )
        self.conn.commit()

    def reset_all(self) -> None:
        """Wipe every agent/trade/strategy/live record - 'step 0', forget everything."""
        for table in ("trades", "strategy_shares", "population_cycles", "live_orders",
                      "live_position", "agents", "meta"):
            self.conn.execute(f"DELETE FROM {table}")
        self.conn.execute("DELETE FROM sqlite_sequence")
        self.conn.commit()

    # ---- live position (single aggregate real position per coin) ----

    def get_live_position(self) -> Optional[sqlite3.Row]:
        return self.conn.execute("SELECT * FROM live_position WHERE id = 1").fetchone()

    def set_live_position(self, coin: str, side: Optional[str], size: float, notional: float) -> None:
        self.conn.execute(
            """INSERT INTO live_position (id, coin, side, size, notional, updated_at)
               VALUES (1, ?, ?, ?, ?, ?)
               ON CONFLICT(id) DO UPDATE SET
                 coin = excluded.coin, side = excluded.side, size = excluded.size,
                 notional = excluded.notional, updated_at = excluded.updated_at""",
            (coin, side, size, notional, now_iso()),
        )
        self.conn.commit()

    def record_live_order(self, coin: str, action: str, side: str, notional: float,
                           fill_price: Optional[float], status: str, detail: str = "") -> None:
        self.conn.execute(
            """INSERT INTO live_orders (coin, action, side, notional, fill_price, status, detail, placed_at)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
            (coin, action, side, notional, fill_price, status, detail, now_iso()),
        )
        self.conn.commit()

    def recent_live_orders(self, limit: int = 50) -> list[sqlite3.Row]:
        return self.conn.execute(
            "SELECT * FROM live_orders ORDER BY id DESC LIMIT ?", (limit,)
        ).fetchall()

    # ---- dashboard aggregates ----

    def realized_pnl_stats(self) -> dict:
        row = self.conn.execute(
            """SELECT
                 COALESCE(SUM(pnl), 0) AS total_pnl,
                 COUNT(*) AS closed_trades,
                 SUM(CASE WHEN result = 'win' THEN 1 ELSE 0 END) AS wins,
                 SUM(CASE WHEN result = 'loss' THEN 1 ELSE 0 END) AS losses
               FROM trades WHERE result IN ('win', 'loss')"""
        ).fetchone()
        closed = row["closed_trades"] or 0
        wins = row["wins"] or 0
        return {
            "total_realized_pnl": row["total_pnl"],
            "closed_trades": closed,
            "wins": wins,
            "losses": row["losses"] or 0,
            "win_rate": (wins / closed) if closed else 0.0,
        }

    def leaderboard(self, limit: int = 25) -> list[sqlite3.Row]:
        return self.conn.execute(
            """SELECT * FROM agents WHERE status = 'alive'
               ORDER BY (total_pnl + CASE WHEN (wins + losses) > 0
                         THEN CAST(wins AS REAL) / (wins + losses) ELSE 0 END) DESC
               LIMIT ?""",
            (limit,),
        ).fetchall()

    def recent_trades(self, limit: int = 50) -> list[sqlite3.Row]:
        return self.conn.execute(
            "SELECT * FROM trades ORDER BY id DESC LIMIT ?", (limit,)
        ).fetchall()

    def total_agents_ever(self) -> int:
        return self.conn.execute("SELECT COUNT(*) FROM agents").fetchone()[0]

    def max_generation(self) -> int:
        row = self.conn.execute("SELECT MAX(generation) FROM agents").fetchone()
        return row[0] or 0
