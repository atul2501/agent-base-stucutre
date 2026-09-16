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
            "('live_breaker_tripped', 'live_breaker_reason', 'live_breaker_tripped_at', 'last_price')"
        ).fetchall()
        breaker_meta = {r["key"]: r["value"] for r in breaker_row}
        last_price = breaker_meta.get("last_price")
        return jsonify({
            "token": config.token,
            "last_price": float(last_price) if last_price is not None else None,
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
        # starting balance + log1p(wins)*10, not raw PnL dollars) so the
        # dashboard's ranking matches what actually decides active-trader
        # selection - these had drifted apart before.
        #
        # win_rate isn't used: under do-or-die a single loss kills the
        # agent, so every alive agent has losses == 0 and win_rate is
        # always exactly 0.0 or 1.0, unable to distinguish a 1-win agent
        # from a 20-win one. LN(wins+1) grows with a proven track record
        # but with diminishing returns rather than a flat +10 for any win.
        rows = conn.execute(
            """SELECT id, parent_id, generation, tier, status, is_active_trader, is_live_trader,
                      balance, wins, losses, win_streak, total_pnl, trades_count, genome_json,
                      revalidation_score, revalidated_at
               FROM agents WHERE status='alive'
               ORDER BY (
                 CASE WHEN (balance - total_pnl) > 0
                      THEN total_pnl / (balance - total_pnl) * 100.0 ELSE 0 END
                 + LN(wins + 1) * 10.0
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

    @app.get("/api/sentiment")
    def sentiment():
        """A synthetic 0-100 'Fear & Greed' read on the SWARM'S OWN current
        behavior - NOT the real crypto-market Fear & Greed Index (that's an
        external, market-wide metric this project has no relationship to).
        This one is built entirely from what your agents are actually doing
        right now, as four equally-weighted components:

        - Long/short bias of currently open positions (more long = greedier,
          more short = more fearful - the standard risk-on/risk-off convention).
        - Overall win rate across all closed trades (higher = more confident).
        - Recent realized-PnL momentum (comparing the latest cycle's swarm-wide
          total to ~10 cycles ago).
        - Participation rate: what fraction of active traders currently have
          an open position (more agents deployed = greedier/more confident,
          more sitting out = more cautious/fearful).
        """
        conn = get_conn()

        side_rows = conn.execute(
            "SELECT side, COUNT(*) AS n FROM trades WHERE result = 'open' GROUP BY side"
        ).fetchall()
        side_counts = {r["side"]: r["n"] for r in side_rows}
        long_n, short_n = side_counts.get("long", 0), side_counts.get("short", 0)
        open_total = long_n + short_n
        bias_score = (long_n / open_total * 100.0) if open_total else 50.0

        closed = conn.execute(
            "SELECT COUNT(*) AS n, SUM(CASE WHEN result='win' THEN 1 ELSE 0 END) AS wins "
            "FROM trades WHERE result IN ('win','loss')"
        ).fetchone()
        win_rate = (closed["wins"] / closed["n"]) if closed["n"] else 0.5
        win_rate_score = win_rate * 100.0

        active_trader_count = conn.execute(
            "SELECT COUNT(*) AS n FROM agents WHERE status='alive' AND is_active_trader=1"
        ).fetchone()["n"]
        participation_score = (open_total / active_trader_count * 100.0) if active_trader_count else 50.0
        participation_score = min(100.0, participation_score)

        pnl_rows = conn.execute(
            "SELECT total_realized_pnl FROM population_cycles ORDER BY id DESC LIMIT 10"
        ).fetchall()
        pnl_vals = [r["total_realized_pnl"] for r in pnl_rows if r["total_realized_pnl"] is not None]
        if len(pnl_vals) >= 2:
            latest, earliest = pnl_vals[0], pnl_vals[-1]
            diff = latest - earliest
            scale = max(10.0, abs(earliest))
            momentum_score = 50.0 + max(-25.0, min(25.0, (diff / scale) * 25.0))
        else:
            momentum_score = 50.0

        index = (bias_score + win_rate_score + participation_score + momentum_score) / 4.0
        index = max(0.0, min(100.0, index))

        if index < 25:
            label = "Extreme Fear"
        elif index < 45:
            label = "Fear"
        elif index <= 55:
            label = "Neutral"
        elif index <= 75:
            label = "Greed"
        else:
            label = "Extreme Greed"

        return jsonify({
            "index": round(index, 1),
            "label": label,
            "components": {
                "long_short_bias": round(bias_score, 1),
                "win_rate": round(win_rate_score, 1),
                "participation": round(participation_score, 1),
                "momentum": round(momentum_score, 1),
            },
            "open_long": long_n,
            "open_short": short_n,
        })

    @app.get("/api/win_rate_trend")
    def win_rate_trend():
        # Daily buckets (by close date, UTC) rather than by population cycle -
        # trades don't map cleanly to a single cycle number, but every trade
        # has a real closed_at timestamp. Shows whether the swarm's edge is
        # improving, flat, or decaying over time, not just one all-time number.
        conn = get_conn()
        rows = conn.execute(
            """SELECT date(closed_at) AS day,
                      COUNT(*) AS trades,
                      SUM(CASE WHEN result='win' THEN 1 ELSE 0 END) AS wins
               FROM trades WHERE result IN ('win','loss') AND closed_at IS NOT NULL
               GROUP BY day ORDER BY day"""
        ).fetchall()
        results = []
        for r in rows:
            d = _row_to_dict(r)
            d["win_rate"] = (d["wins"] / d["trades"]) if d["trades"] else 0.0
            results.append(d)
        return jsonify(results)

    @app.get("/api/trade_duration_histogram")
    def trade_duration_histogram():
        # Fixed hour buckets - shows whether exits are mostly fast TP/SL hits
        # or mostly riding out to max_hold_hours.
        conn = get_conn()
        rows = conn.execute(
            """SELECT (julianday(closed_at) - julianday(opened_at)) * 24.0 AS hours
               FROM trades WHERE result IN ('win','loss') AND closed_at IS NOT NULL"""
        ).fetchall()
        bins = [(0, 0.5, "0-30m"), (0.5, 1, "30m-1h"), (1, 4, "1-4h"),
                (4, 12, "4-12h"), (12, 24, "12-24h"), (24, 48, "24-48h"),
                (48, float("inf"), "48h+")]
        counts = {label: 0 for _, _, label in bins}
        for r in rows:
            h = r["hours"]
            if h is None:
                continue
            for lo, hi, label in bins:
                if lo <= h < hi:
                    counts[label] += 1
                    break
        return jsonify([{"bucket": label, "count": counts[label]} for _, _, label in bins])

    @app.get("/api/lifespan_histogram")
    def lifespan_histogram():
        # How many trades an agent completed before its do-or-die loss -
        # only DEAD agents (an alive agent's lifespan isn't over yet). A
        # dead agent's trades_count is always its win streak + 1 final loss.
        conn = get_conn()
        rows = conn.execute(
            "SELECT trades_count FROM agents WHERE status = 'dead'"
        ).fetchall()
        bins = [(0, 1, "1st trade"), (1, 2, "1 win"),
                (2, 3, "2 wins"), (3, 5, "3-4 wins"),
                (5, 10, "5-9 wins"), (10, float("inf"), "10+ wins")]
        counts = {label: 0 for _, _, label in bins}
        for r in rows:
            n = r["trades_count"]
            for lo, hi, label in bins:
                if lo <= n < hi:
                    counts[label] += 1
                    break
        return jsonify([{"bucket": label, "count": counts[label]} for _, _, label in bins])

    @app.get("/api/generation_depth")
    def generation_depth():
        # Currently-alive agents by generation number - shows whether
        # evolution is actually producing deep lineages or mostly staying
        # shallow (e.g. constant floor-refills at generation 0 outpacing
        # winners' children surviving to reproduce further).
        conn = get_conn()
        rows = conn.execute(
            "SELECT generation, COUNT(*) AS count FROM agents WHERE status='alive' "
            "GROUP BY generation ORDER BY generation"
        ).fetchall()
        return jsonify([_row_to_dict(r) for r in rows])

    @app.get("/api/lineage")
    def lineage():
        # Every agent ever created (alive or dead), just enough fields to
        # build a parent -> children family tree client-side. Small enough
        # to send in full even near the population cap (500 alive + however
        # many have died), and the tree shape itself - who bred from whom,
        # how deep lineages go, which branches died out - is exactly what a
        # flat leaderboard (top 50 by fitness, alive only) can't show.
        conn = get_conn()
        rows = conn.execute(
            """SELECT id, parent_id, generation, status, tier,
                      wins, losses, trades_count, total_pnl, created_at, died_at
               FROM agents ORDER BY id"""
        ).fetchall()
        return jsonify([_row_to_dict(r) for r in rows])

    @app.get("/api/pnl_history")
    def pnl_history():
        conn = get_conn()
        rows = conn.execute(
            "SELECT cycle, alive_count, active_trader_count, professional_count, "
            "best_total_pnl, total_realized_pnl, ran_at "
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
