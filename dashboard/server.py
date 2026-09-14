"""Local read-only dashboard: population health, profitability, leaderboard,
recent trades, and live/paper + mainnet/testnet + Ollama status at a glance.

Runs in a background thread inside main.py; talks to its own SQLite
connection (the main Database enables WAL mode so this can read safely
while the orchestrator thread keeps writing).
"""
from __future__ import annotations

import logging
import sqlite3
import time
from pathlib import Path

from flask import Flask, jsonify, send_from_directory

from config import Config
from reasoning import ollama_advisor

log = logging.getLogger(__name__)
STATIC_DIR = Path(__file__).parent / "static"


def _row_to_dict(row: sqlite3.Row) -> dict:
    return {k: row[k] for k in row.keys()}


def create_app(config: Config) -> Flask:
    app = Flask(__name__, static_folder=None)
    conn = sqlite3.connect(config.db_path, check_same_thread=False)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA query_only = ON")

    @app.get("/")
    def index():
        return send_from_directory(STATIC_DIR, "index.html")

    @app.get("/api/overview")
    def overview():
        last_cycle = conn.execute(
            "SELECT cycle, ran_at FROM population_cycles ORDER BY id DESC LIMIT 1"
        ).fetchone()
        return jsonify({
            "token": config.token,
            "timeframe": config.timeframe,
            "cycle_seconds": config.cycle_seconds,
            "network": config.hl_network,
            "mode": "live" if config.is_live() else "paper",
            "is_live": config.is_live(),
            "population_cap": config.population_cap,
            "active_trader_count": config.active_trader_count,
            "live_active_trader_count": config.live_active_trader_count,
            "live_max_total_notional_usd": config.live_max_total_notional_usd,
            "live_max_leverage": config.live_max_leverage,
            "ollama": ollama_advisor.get_status(),
            "last_cycle": _row_to_dict(last_cycle) if last_cycle else None,
            "server_time": time.time(),
        })

    @app.get("/api/population")
    def population():
        row = conn.execute(
            """SELECT
                 SUM(CASE WHEN status='alive' THEN 1 ELSE 0 END) AS alive_count,
                 SUM(CASE WHEN status='dead' THEN 1 ELSE 0 END) AS dead_count,
                 SUM(CASE WHEN status='alive' AND is_active_trader=1 THEN 1 ELSE 0 END) AS active_trader_count,
                 SUM(CASE WHEN status='alive' AND is_live_trader=1 THEN 1 ELSE 0 END) AS live_trader_count,
                 SUM(CASE WHEN status='alive' AND tier='professional' THEN 1 ELSE 0 END) AS professional_count,
                 COUNT(*) AS total_ever,
                 MAX(generation) AS max_generation,
                 AVG(CASE WHEN status='alive' THEN balance END) AS avg_balance
               FROM agents"""
        ).fetchone()
        pnl_stats = conn.execute(
            """SELECT COALESCE(SUM(pnl),0) AS total_pnl, COUNT(*) AS closed,
                      SUM(CASE WHEN result='win' THEN 1 ELSE 0 END) AS wins,
                      SUM(CASE WHEN result='loss' THEN 1 ELSE 0 END) AS losses
               FROM trades WHERE result IN ('win','loss')"""
        ).fetchone()
        closed = pnl_stats["closed"] or 0
        wins = pnl_stats["wins"] or 0
        d = _row_to_dict(row)
        d["total_realized_pnl"] = pnl_stats["total_pnl"]
        d["closed_trades"] = closed
        d["overall_win_rate"] = (wins / closed) if closed else 0.0
        return jsonify(d)

    @app.get("/api/leaderboard")
    def leaderboard():
        rows = conn.execute(
            """SELECT id, parent_id, generation, tier, status, is_active_trader, is_live_trader,
                      balance, wins, losses, win_streak, total_pnl, trades_count, genome_json
               FROM agents WHERE status='alive'
               ORDER BY (total_pnl + CASE WHEN (wins+losses)>0
                         THEN CAST(wins AS REAL)/(wins+losses) ELSE 0 END) DESC
               LIMIT 30"""
        ).fetchall()
        return jsonify([_row_to_dict(r) for r in rows])

    @app.get("/api/trades")
    def trades():
        rows = conn.execute(
            "SELECT * FROM trades ORDER BY id DESC LIMIT 50"
        ).fetchall()
        return jsonify([_row_to_dict(r) for r in rows])

    @app.get("/api/pnl_history")
    def pnl_history():
        rows = conn.execute(
            "SELECT cycle, alive_count, active_trader_count, professional_count, best_total_pnl, ran_at "
            "FROM population_cycles ORDER BY id DESC LIMIT 200"
        ).fetchall()
        return jsonify(list(reversed([_row_to_dict(r) for r in rows])))

    @app.get("/api/live")
    def live():
        pos = conn.execute("SELECT * FROM live_position WHERE id = 1").fetchone()
        orders = conn.execute("SELECT * FROM live_orders ORDER BY id DESC LIMIT 30").fetchall()
        return jsonify({
            "position": _row_to_dict(pos) if pos else None,
            "recent_orders": [_row_to_dict(r) for r in orders],
        })

    return app


def run_dashboard(config: Config) -> None:
    app = create_app(config)
    log.info("Dashboard listening on http://%s:%d", config.dashboard_host, config.dashboard_port)
    app.run(host=config.dashboard_host, port=config.dashboard_port, debug=False, use_reloader=False)
