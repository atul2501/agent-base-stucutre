"""Historical backtest: gives a newly-spawned genome an instant fitness
estimate from real historical data, instead of making it wait on rare live
setups (this was measured directly: over 72h of real SOL data, a genuinely
selective RSI+trend trigger fired 0-7 times depending on threshold, so a
single agent can go a long time between opportunities to prove itself
live). This is used only to choose which candidate genome a NEW agent is
BORN with - see agents/population.py. The live do-or-die mechanic itself
is completely unchanged: a backtested-good agent still has to win its
first real trade to survive, same as before.

Reuses strategy/signals.py's evaluate_entry/evaluate_exit UNCHANGED (not a
reimplementation) so live and backtested scoring can never drift into two
different definitions of "confirms" - see build_backtest_features below,
which is the backtest's equivalent of strategy/signals.py::build_features.

Honesty note: Hyperliquid's public API has no historical series for order
book depth or open interest (point-in-time snapshots only), so those two
confirmations are neutral during backtesting (ob_imbalance=1.0,
oi_change_pct=None - evaluate_entry already treats both as "no signal").
Funding AND mark/oracle premium DO have real historical series
(Info.funding_history includes both), so those use real historical data.
The higher-timeframe trend filter is also neutral here (htf_trend_up=None)
- it's live-only for now; backtesting it properly would need a second
historical candle series time-aligned per bar without introducing
lookahead bias, which wasn't worth rushing.
"""
from __future__ import annotations

import bisect
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone

from strategy.genome import Genome
from strategy.indicators import adx, atr, bollinger_percent_b, ema, macd_histogram, rsi, stochastic_rsi, vwap
from strategy.signals import Features, evaluate_entry, evaluate_exit
from trading.paper_executor import close_paper_position, open_paper_position

# ~24h of lookback for the "24h momentum" feature, expressed in candles -
# computed from the timeframe at call time rather than hardcoded to 5m.
_DAY_MS = 24 * 60 * 60 * 1000


@dataclass
class BacktestResult:
    trades: int
    wins: int
    losses: int
    win_rate: float
    return_pct: float  # total simulated PnL / starting balance * 100
    max_drawdown_pct: float = 0.0  # largest peak-to-trough drop in simulated balance


def _min_required_candles(genome: Genome) -> int:
    return max(
        genome.ema_slow, genome.rsi_period, genome.atr_period, genome.adx_period,
        genome.vwap_period, genome.volume_lookback, genome.bb_period,
        genome.stoch_rsi_period + genome.stoch_k_smooth,
        genome.ema_slow + genome.macd_signal_period,
    ) + 2


@dataclass
class _Series:
    closes: list
    volumes: list
    ema_fast: object
    ema_slow: object
    rsi: object
    atr: object
    adx: object
    macd: object
    bb: object
    stoch: object
    day_candles: int
    funding_points: list
    funding_times: list


def _precompute(genome: Genome, candles: list[dict], funding_points: list[tuple[int, float, float]]) -> _Series:
    closes = [c["c"] for c in candles]
    highs = [c["h"] for c in candles]
    lows = [c["l"] for c in candles]
    volumes = [c["v"] for c in candles]

    if len(candles) > 1 and candles[1]["t"] > candles[0]["t"]:
        candle_ms = candles[1]["t"] - candles[0]["t"]
    else:
        candle_ms = 5 * 60 * 1000

    return _Series(
        closes=closes, volumes=volumes,
        ema_fast=ema(closes, genome.ema_fast), ema_slow=ema(closes, genome.ema_slow),
        rsi=rsi(closes, genome.rsi_period), atr=atr(highs, lows, closes, genome.atr_period),
        adx=adx(highs, lows, closes, genome.adx_period),
        macd=macd_histogram(closes, genome.ema_fast, genome.ema_slow, genome.macd_signal_period),
        bb=bollinger_percent_b(closes, genome.bb_period, genome.bb_std_dev),
        stoch=stochastic_rsi(closes, genome.stoch_rsi_period, genome.stoch_rsi_period, genome.stoch_k_smooth),
        day_candles=max(1, _DAY_MS // candle_ms),
        funding_points=funding_points, funding_times=[p[0] for p in funding_points],
    )


def build_backtest_features(genome: Genome, series: _Series, i: int, ts_ms: int) -> Features:
    """The backtest's equivalent of strategy/signals.py::build_features -
    same indicator math, computed from a precomputed series instead of a
    live snapshot. Exposed (not just inlined in the loop) so it can be
    directly compared against the live path for correctness."""
    price = series.closes[i]

    if series.funding_points:
        idx = max(0, bisect.bisect_right(series.funding_times, ts_ms) - 1)
        funding, premium = series.funding_points[idx][1], series.funding_points[idx][2]
    else:
        funding, premium = 0.0, 0.0

    recent_vol = series.volumes[max(0, i - genome.volume_lookback + 1): i + 1]
    avg_vol = sum(recent_vol) / len(recent_vol) if recent_vol else 0.0
    volume_ratio = (series.volumes[i] / avg_vol) if avg_vol > 0 else 1.0

    vwap_window_start = max(0, i - genome.vwap_period + 1)
    vwap_val = vwap(series.closes[vwap_window_start: i + 1], series.volumes[vwap_window_start: i + 1], genome.vwap_period)
    vwap_deviation_pct = (price - vwap_val) / vwap_val * 100.0 if vwap_val else 0.0

    prev_day_price = series.closes[max(0, i - series.day_candles)]
    daily_change_pct = (price - prev_day_price) / prev_day_price * 100.0 if prev_day_price else 0.0

    return Features(
        mid_price=price,
        ema_fast=float(series.ema_fast[i]), ema_slow=float(series.ema_slow[i]),
        trend_up=bool(series.ema_fast[i] > series.ema_slow[i]), rsi_value=float(series.rsi[i]),
        ob_imbalance=1.0, spread_pct=0.0, oi_change_pct=None,  # not available historically
        funding=funding, premium=premium,
        atr_pct=float(series.atr[i]) / price * 100.0 if price else 0.0,
        adx_value=float(series.adx[i]),
        volume_ratio=volume_ratio, vwap_deviation_pct=vwap_deviation_pct,
        macd_hist=float(series.macd[i]), daily_change_pct=daily_change_pct,
        bb_percent_b=float(series.bb[i]), stoch_rsi_k=float(series.stoch[i]),
        # Higher-timeframe trend isn't backtested (would need a second
        # historical series properly time-aligned per bar without
        # lookahead bias) - neutral here, same treatment as order book/OI.
        htf_trend_up=None,
    )


def backtest_genome(
    genome: Genome,
    candles: list[dict],
    funding_points: list[tuple[int, float, float]],
    starting_balance: float = 1000.0,
) -> BacktestResult:
    """`candles`: oldest->newest dicts with t,o,h,l,c,v (same shape as
    MarketSnapshot.candles). `funding_points`: (timestamp_ms, funding,
    premium) tuples, oldest->newest, e.g. from Info.funding_history."""
    n = len(candles)
    min_required = _min_required_candles(genome)
    if n < min_required + 10:
        return BacktestResult(0, 0, 0, 0.0, 0.0)

    series = _precompute(genome, candles, funding_points)
    timestamps = [c["t"] for c in candles]

    balance = starting_balance
    peak_balance = starting_balance
    max_drawdown_pct = 0.0
    wins = losses = 0
    position: dict | None = None

    for i in range(min_required, n):
        price = series.closes[i]
        ts_ms = timestamps[i]
        now = datetime.fromtimestamp(ts_ms / 1000.0, tz=timezone.utc)
        f = build_backtest_features(genome, series, i, ts_ms)

        if position is None:
            sig = evaluate_entry(genome, f)
            if sig.action in ("long", "short"):
                fill_price, size, _notional = open_paper_position(balance, price, sig.action, genome.position_size_pct)
                # Index into funding_points as of entry - only funding events
                # AFTER this one accrue against the position (see below).
                last_funding_idx = (max(0, bisect.bisect_right(series.funding_times, ts_ms) - 1)
                                     if series.funding_points else -1)
                position = {"side": sig.action, "entry_price": fill_price, "size": size,
                            "opened_at": now.isoformat(), "last_funding_idx": last_funding_idx,
                            "funding_accrued": 0.0}
        else:
            # Funding accrual: a real historical funding series exists here
            # (unlike live, which only ever sees the current rate), so this
            # walks every funding event actually crossed during the hold and
            # charges/credits it against the position's notional at the time -
            # exact, not an approximation. Standard perpetual convention: a
            # positive rate is paid BY longs TO shorts.
            if series.funding_points:
                current_idx = max(0, bisect.bisect_right(series.funding_times, ts_ms) - 1)
                while position["last_funding_idx"] < current_idx:
                    position["last_funding_idx"] += 1
                    rate = series.funding_points[position["last_funding_idx"]][1]
                    cost = position["size"] * price * rate
                    position["funding_accrued"] += cost if position["side"] == "long" else -cost
            outcome = evaluate_exit(genome, position, f, now=now)
            if outcome is not None:
                result, _reason = outcome
                _exit_price, pnl = close_paper_position(
                    position["entry_price"], price, position["size"], position["side"],
                    funding_cost=position["funding_accrued"],
                )
                balance += pnl
                peak_balance = max(peak_balance, balance)
                if peak_balance > 0:
                    max_drawdown_pct = max(max_drawdown_pct, (peak_balance - balance) / peak_balance * 100.0)
                if result == "win":
                    wins += 1
                else:
                    losses += 1
                position = None

    trades = wins + losses
    win_rate = wins / trades if trades else 0.0
    return_pct = (balance - starting_balance) / starting_balance * 100.0
    return BacktestResult(trades=trades, wins=wins, losses=losses, win_rate=win_rate,
                           return_pct=return_pct, max_drawdown_pct=max_drawdown_pct)


# How much a single point of max intra-backtest drawdown % costs in fitness -
# calibrated so a severe (~20%) drawdown costs about as much as the win-rate
# term's entire possible contribution (10.0 at a 100% win rate), making
# drawdown a real, comparably-weighted consideration rather than a token one.
_DRAWDOWN_PENALTY_WEIGHT = 0.5

# Shrinks a small-sample win_rate toward a neutral 50% baseline before it
# feeds fitness_score - equivalent to adding ~5 "virtual" trades split evenly
# between win/loss as a prior. Without this, a lucky 2-trade genome (e.g.
# 2/2 wins) scores identically to a proven 20-trade one on the win_rate term,
# despite the former being statistically almost uninformative.
_WIN_RATE_PRIOR_TRADES = 5.0
_WIN_RATE_PRIOR_WINS = 2.5


def fitness_score(result: BacktestResult) -> float:
    """Single comparable number for picking the best of several candidate
    genomes. Needs at least a couple of trades to mean anything - an
    untested genome (0 trades) scores strictly below any tested one so a
    genome that never once found a setup in ~17 days of history doesn't
    win by default against one that traded and lost narrowly.

    Risk-adjusted: penalizes max_drawdown_pct so a volatile/streaky genome
    doesn't outrank a steadier one purely on total return, and shrinks
    win_rate toward 50% for small trade counts so a couple of lucky wins
    can't masquerade as a proven high win rate."""
    if result.trades == 0:
        return -1.0
    shrunk_win_rate = (result.wins + _WIN_RATE_PRIOR_WINS) / (result.trades + _WIN_RATE_PRIOR_TRADES)
    return (result.return_pct
            + shrunk_win_rate * 10.0
            - result.max_drawdown_pct * _DRAWDOWN_PENALTY_WEIGHT)
