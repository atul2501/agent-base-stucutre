"""Covers db/database.py's schema contract and the shared fitness formula.

The flip-order tests are a direct regression test for a real bug the review
council found: trading/live_executor.py emits action="flip_close"/"flip_open"
on a position-side reversal, but the live_orders.action CHECK constraint used
to only allow ('open','increase','decrease','close','flip') - every flip
raised an uncaught IntegrityError right after a real order had already been
placed on the exchange.
"""
import math

import pytest

from db.database import fitness_score


class TestFlipOrderSchema:
    def test_flip_close_action_is_accepted(self, db):
        db.record_live_order("SOL", "flip_close", "long", 100.0, 1.23, "filled", "test")
        rows = db.recent_live_orders(limit=1)
        assert rows[0]["action"] == "flip_close"

    def test_flip_open_action_is_accepted(self, db):
        db.record_live_order("SOL", "flip_open", "short", 100.0, 1.23, "filled", "test")
        rows = db.recent_live_orders(limit=1)
        assert rows[0]["action"] == "flip_open"

    def test_all_actions_emitted_by_live_executor_are_accepted(self, db):
        # trading/live_executor.py::adjust_to's full set of action strings -
        # if any of these regress to an unlisted value, this fails loudly in
        # CI instead of silently on a live position flip.
        for action in ("open", "increase", "decrease", "close", "flip_close", "flip_open"):
            db.record_live_order("SOL", action, "long", 10.0, 1.0, "filled", "")

    def test_unknown_action_is_still_rejected(self, db):
        # The CHECK constraint should still be a real constraint, not have
        # been loosened into a no-op while fixing the flip bug.
        import sqlite3
        with pytest.raises(sqlite3.IntegrityError):
            db.record_live_order("SOL", "not_a_real_action", "long", 10.0, 1.0, "filled", "")


class TestMigrationRebuildsExistingLiveOrdersTable:
    def test_old_schema_db_is_migrated_to_accept_flip_actions(self, tmp_path):
        """Simulates a DB created before the fix (old CHECK constraint) to
        verify _migrate() rebuilds live_orders in place - not just that a
        freshly created DB has the right constraint from schema.sql."""
        import sqlite3

        from db.database import Database

        db_path = tmp_path / "legacy.db"
        conn = sqlite3.connect(str(db_path))
        conn.execute(
            """CREATE TABLE live_orders (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                coin TEXT NOT NULL,
                action TEXT NOT NULL CHECK (action IN ('open', 'increase', 'decrease', 'close', 'flip')),
                side TEXT NOT NULL CHECK (side IN ('long', 'short')),
                notional REAL NOT NULL,
                fill_price REAL,
                status TEXT NOT NULL CHECK (status IN ('filled', 'error')),
                detail TEXT,
                placed_at TEXT NOT NULL
            )"""
        )
        conn.execute(
            "INSERT INTO live_orders (coin, action, side, notional, status, placed_at) "
            "VALUES ('SOL', 'open', 'long', 50.0, 'filled', '2026-01-01T00:00:00+00:00')"
        )
        conn.commit()
        conn.close()

        migrated = Database(str(db_path))
        try:
            migrated.record_live_order("SOL", "flip_close", "long", 100.0, 1.0, "filled", "")
            rows = migrated.recent_live_orders(limit=10)
            assert len(rows) == 2  # pre-existing row survived the rebuild
            assert any(r["action"] == "flip_close" for r in rows)
            assert any(r["action"] == "open" for r in rows)
        finally:
            migrated.close()


class TestFitnessScore:
    def test_zero_pnl_zero_wins_is_zero(self):
        assert fitness_score(total_pnl=0.0, balance=1000.0, wins=0) == 0.0

    def test_positive_pnl_gives_positive_pct_return_component(self):
        # +100 pnl on a 1000 starting balance (balance=1100, pnl=100) = +10%
        score = fitness_score(total_pnl=100.0, balance=1100.0, wins=0)
        assert score == pytest.approx(10.0)

    def test_wins_add_diminishing_log_bonus(self):
        one_win = fitness_score(total_pnl=0.0, balance=1000.0, wins=1)
        two_wins = fitness_score(total_pnl=0.0, balance=1000.0, wins=2)
        ten_wins = fitness_score(total_pnl=0.0, balance=1000.0, wins=10)
        assert one_win == pytest.approx(math.log1p(1) * 10.0)
        # Diminishing returns: the 1->2 jump is bigger than the 9->10 jump.
        assert (two_wins - one_win) > (ten_wins - fitness_score(total_pnl=0.0, balance=1000.0, wins=9))

    def test_non_positive_starting_balance_does_not_divide_by_zero(self):
        # balance - total_pnl <= 0 (e.g. a fully wiped-out agent) must not
        # raise ZeroDivisionError - falls back to 0% return component.
        score = fitness_score(total_pnl=1000.0, balance=0.0, wins=0)
        assert score == 0.0


class TestAgentRowFitnessMatchesSharedFormula:
    def test_agent_row_fitness_uses_fitness_score(self, db):
        agent_id = db.create_agent({"coin": "SOL"}, balance=1000.0)
        db.record_win(agent_id, pnl=50.0)
        agent = db.get_agent(agent_id)
        assert agent.fitness == pytest.approx(fitness_score(agent.total_pnl, agent.balance, agent.wins))
