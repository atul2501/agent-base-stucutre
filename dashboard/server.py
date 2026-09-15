"""Local read-only dashboard: population health, profitability, leaderboard,
recent trades, and live/paper + mainnet/testnet + Ollama status at a glance.

Runs in a background thread inside main.py. Opens a fresh SQLite connection
per request (closed at teardown) rather than sharing one long-lived
connection with the orchestrator's writer thread - a long-lived connection
shared across threads against a WAL-mode database is a known source of
native crashes on macOS, and per-request connections are cheap enough for a
dashboard polled every few seconds.
"""
from __future__ import annotations

import logging
import sqlite3
import time
from pathlib import Path

from flask import Flask, g, jsonify, request, send_from_directory

from config import Config
from reasoning import ollama_advisor

log = logging.getLogger(__name__)
STATIC_DIR = Path(__file__).parent / "static"


def _row_to_dict(row: sqlite3.Row) -> dict:
    return {k: row[k] for k in row.keys()}


def create_app(config: Config) -> Flask:
    app = Flask(__name__, static_folder=None)

    def get_conn() -> sqlite3.Connection:
        if "db" not in g:
            g.db = sqlite3.connect(config.db_path)
            g.db.row_factory = sqlite3.Row
            g.db.execute("PRAGMA query_only = ON")
        return g.db

    @app.teardown_appcontext
    def close_conn(exception=None):
        db = g.pop("db", None)
        if db is not None:
            db.close()

    @app.get("/")
    def index():
        return send_from_directory(STATIC_DIR, "index.html")

    @app.get("/api/overview")
    def overview():
        conn = get_conn()
        last_cycle = conn.execute(
            "SELECT cycle, ran_at FROM population_cycles ORDER BY id DESC LIMIT 1"
        ).fetchone()
        breaker_row = conn.execute(
            "SELECT key, value FROM meta WHERE key IN "
            "('live_breaker_tripped', 'live_breaker_reason', 'live_breaker_tripped_at')"
        ).fetchall()
        breaker_meta = {r["key"]: r["value"] for r in breaker_row}
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
            "revalidation_enabled": config.revalidation_enabled,
            "revalidation_drift_threshold": config.revalidation_drift_threshold,
            "live_breaker_tripped": breaker_meta.get("live_breaker_tripped") == "1",
            "live_breaker_reason": breaker_meta.get("live_breaker_reason") or None,
            "live_breaker_tripped_at": breaker_meta.get("live_breaker_tripped_at") or None,
            "ollama": ollama_advisor.get_status(),
            "last_cycle": _row_to_dict(last_cycle) if last_cycle else None,
            "server_time": time.time(),
        })

    @app.get("/api/population")
    def population():
        conn = get_conn()
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
        conn = get_conn()
        # Top 50 by fitness out of up to population_cap (500) alive agents -
        # matches ACTIVE_TRADER_COUNT, the set actually allowed to trade.
        # Mirrors AgentRow.fitness in db/database.py exactly (% return on
        # starting balance + win_rate*10, not raw PnL dollars) so the
        # dashboard's ranking matches what actually decides active-trader
        # selection - these had drifted apart before.
        rows = conn.execute(
            """SELECT id, parent_id, generation, tier, status, is_active_trader, is_live_trader,
                      balance, wins, losses, win_streak, total_pnl, trades_count, genome_json,
                      revalidation_score, revalidated_at
               FROM agents WHERE status='alive'
               ORDER BY (
                 CASE WHEN (balance - total_pnl) > 0
                      THEN total_pnl / (balance - total_pnl) * 100.0 ELSE 0 END
                 + CASE WHEN (wins+losses) > 0
                        THEN CAST(wins AS REAL) / (wins+losses) * 10.0 ELSE 0 END
               ) DESC
               LIMIT 50"""
        ).fetchall()
        return jsonify([_row_to_dict(r) for r in rows])

    @app.get("/api/trades")
    def trades():
        # `limit` grows via the dashboard's "Load More" button (15 at a
        # time) instead of true offset pagination - simpler to keep in sync
        # with the 5s auto-refresh (which just re-fetches the top `limit`
        # every tick) without the loaded set shifting or duplicating as new
        # trades come in. Capped so a runaway query string can't force a
        # huge full-table scan.
        limit = min(max(int(request.args.get("limit", 15)), 1), 2000)
        conn = get_conn()
        rows = conn.execute(
            "SELECT * FROM trades ORDER BY id DESC LIMIT ?", (limit,)
        ).fetchall()
        total = conn.execute("SELECT COUNT(*) AS c FROM trades").fetchone()["c"]
        last_price_row = conn.execute("SELECT value FROM meta WHERE key = 'last_price'").fetchone()
        last_price = float(last_price_row["value"]) if last_price_row else None

        results = []
        for r in rows:
            d = _row_to_dict(r)
            # Open trades have pnl=None until they actually close (win/loss)
            # - without this, a position can sit open for hours (max_hold_hours
            # up to 96h) with zero visibility into whether it's winning or
            # losing right now. Mark-to-market estimate only, not the fill
            # price a real close would get (see trading/paper_executor.py).
            if d["result"] == "open" and last_price is not None:
                if d["side"] == "long":
                    d["unrealized_pnl"] = (last_price - d["entry_price"]) * d["size"]
                else:
                    d["unrealized_pnl"] = (d["entry_price"] - last_price) * d["size"]
            else:
                d["unrealized_pnl"] = None
            results.append(d)
        return jsonify({"trades": results, "total": total})

    @app.get("/api/regime_stats")
    def regime_stats():
        # Swarm-wide aggregate, not per-agent - a per-agent x per-regime
        # breakdown for up to 500 agents would be an impractically large
        # table; this answers the actionable question directly: does the
        # population hold up in every regime, or only one?
        conn = get_conn()
        rows = conn.execute(
            """SELECT COALESCE(regime, 'unknown') AS regime,
                      COUNT(*) AS trades,
                      SUM(CASE WHEN result='win' THEN 1 ELSE 0 END) AS wins,
                      SUM(CASE WHEN result='loss' THEN 1 ELSE 0 END) AS losses,
                      COALESCE(SUM(pnl), 0) AS total_pnl
               FROM trades WHERE result IN ('win','loss')
               GROUP BY regime"""
        ).fetchall()
        results = []
        for r in rows:
            d = _row_to_dict(r)
            d["win_rate"] = (d["wins"] / d["trades"]) if d["trades"] else 0.0
            results.append(d)
        return jsonify(results)

    @app.get("/api/pnl_history")
    def pnl_history():
        conn = get_conn()
        rows = conn.execute(
            "SELECT cycle, alive_count, active_trader_count, professional_count, best_total_pnl, ran_at "
            "FROM population_cycles ORDER BY id DESC LIMIT 200"
        ).fetchall()
        return jsonify(list(reversed([_row_to_dict(r) for r in rows])))

    @app.get("/api/live")
    def live():
        conn = get_conn()
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
    app.run(host=config.dashboard_host, port=config.dashboard_port, debug=False, use_reloader=False, threaded=True)
