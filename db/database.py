"""SQLite persistence layer for agents, trades, and shared strategies."""
from __future__ import annotations

import json
import sqlite3
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

SCHEMA_PATH = Path(__file__).parent / "schema.sql"


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


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
        starting_balance = self.balance - self.total_pnl
        pct_return = (self.total_pnl / starting_balance * 100.0) if starting_balance > 0 else 0.0
        return pct_return + self.win_rate * 10.0


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

        agent_cols = {row["name"] for row in self.conn.execute("PRAGMA table_info(agents)")}
        if "revalidation_score" not in agent_cols:
            self.conn.execute("ALTER TABLE agents ADD COLUMN revalidation_score REAL")
            self.conn.execute("ALTER TABLE agents ADD COLUMN revalidated_at TEXT")
            self.conn.commit()

        cycle_cols = {row["name"] for row in self.conn.execute("PRAGMA table_info(population_cycles)")}
        if "total_realized_pnl" not in cycle_cols:
            self.conn.execute("ALTER TABLE population_cycles ADD COLUMN total_realized_pnl REAL")
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
        rows = self.conn.execute("SELECT * FROM agents WHERE parent_id = ?", (parent_id,)).fetchall()
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
    ) -> int:
        cur = self.conn.execute(
            """INSERT INTO trades
               (agent_id, coin, side, entry_price, size, notional, stop_loss, take_profit,
                entry_reason, regime, opened_at, result)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 'open')""",
            (agent_id, coin, side, entry_price, size, notional, stop_loss, take_profit,
             entry_reason, regime, now_iso()),
        )
        self.conn.commit()
        return cur.lastrowid

    def get_open_trade(self, agent_id: int) -> Optional[sqlite3.Row]:
        return self.conn.execute(
            "SELECT * FROM trades WHERE agent_id = ? AND result = 'open'", (agent_id,)
        ).fetchone()

    def close_trade(self, trade_id: int, exit_price: float, pnl: float, result: str, exit_reason: str) -> None:
        self.conn.execute(
            """UPDATE trades SET exit_price = ?, pnl = ?, result = ?, exit_reason = ?, closed_at = ?
               WHERE id = ?""",
            (exit_price, pnl, result, exit_reason, now_iso(), trade_id),
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
