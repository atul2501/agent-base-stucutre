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
        # selection: realized pnl first, win rate as a tie-break so brand
        # new 0-trade agents (fitness 0) aren't favored over proven losers.
        return self.total_pnl + self.win_rate


class Database:
    def __init__(self, db_path: str):
        self.conn = sqlite3.connect(db_path)
        self.conn.row_factory = sqlite3.Row
        self.conn.execute("PRAGMA foreign_keys = ON")
        self._init_schema()

    def _init_schema(self) -> None:
        with open(SCHEMA_PATH) as f:
            self.conn.executescript(f.read())
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

    def set_tier(self, agent_id: int, tier: str) -> None:
        self.conn.execute("UPDATE agents SET tier = ? WHERE id = ?", (tier, agent_id))
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
    ) -> int:
        cur = self.conn.execute(
            """INSERT INTO trades
               (agent_id, coin, side, entry_price, size, notional, stop_loss, take_profit,
                entry_reason, opened_at, result)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 'open')""",
            (agent_id, coin, side, entry_price, size, notional, stop_loss, take_profit,
             entry_reason, now_iso()),
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
        self.conn.execute(
            """INSERT INTO population_cycles
               (cycle, alive_count, active_trader_count, professional_count,
                best_agent_id, best_total_pnl, ran_at)
               VALUES (?, ?, ?, ?, ?, ?, ?)""",
            (cycle, alive_count, active_trader_count, professional_count,
             best_agent_id, best_total_pnl, now_iso()),
        )
        self.conn.commit()
