"""Turns a market snapshot + a genome into a trade decision.

Swing-entry logic: trend from EMA cross, trigger from RSI pullback/rally.
Two hard pre-filters gate everything: spread (liquidity) and ATR regime
(not too dead, not too chaotic). The trigger is then scored against every
other signal Hyperliquid's data supports - order book imbalance, open
interest change, funding-rate crowding, mark/oracle premium, volume
conviction, VWAP deviation, MACD momentum, Bollinger %B, Stochastic RSI,
24h macro momentum, and a higher-timeframe trend check (trading with the
bigger trend, not against short-term noise). A clear score fires a trade
or a clear HOLD; a genuinely mixed read is flagged `ambiguous` so the
orchestrator can consult the Ollama advisor instead of guessing.

The higher-timeframe trend is live-only: `htf_trend_up=None` means
"unknown" (used during backtesting, which doesn't fetch a second
historical series for this in the current implementation - see
backtest/engine.py) and is treated as neutral, contributing neither a
confirmation nor a contradiction.
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone

from market.hyperliquid_client import MarketSnapshot
from strategy.genome import Genome
from strategy.indicators import (
    adx, atr, bollinger_percent_b, ema, macd_histogram, rsi, stochastic_rsi, vwap,
)


@dataclass
class Features:
    mid_price: float
    ema_fast: float
    ema_slow: float
    trend_up: bool
    rsi_value: float
    ob_imbalance: float           # bid_vol / ask_vol over top 10 levels
    spread_pct: float
    oi_change_pct: float | None   # None if no prior snapshot to diff against
    funding: float
    premium: float
    atr_pct: float                # ATR as a % of price - volatility regime
    adx_value: float              # trend STRENGTH (not direction) - low = ranging/choppy
    volume_ratio: float           # latest candle volume / its own recent average
    vwap_deviation_pct: float     # (price - vwap) / vwap * 100
    macd_hist: float
    daily_change_pct: float
    bb_percent_b: float           # 0 = at lower Bollinger band, 1 = at upper
    stoch_rsi_k: float            # 0-100, Stochastic RSI %K
    htf_trend_up: bool | None = None  # higher-timeframe EMA trend; None = unknown/neutral


@dataclass
class Signal:
    action: str          # "long" | "short" | "hold"
    confidence: float
    score: int
    reasons: list[str]
    ambiguous: bool


def _neutral_features(snap: MarketSnapshot) -> Features:
    return Features(
        mid_price=snap.mid_price, ema_fast=snap.mid_price, ema_slow=snap.mid_price,
        trend_up=True, rsi_value=50.0, ob_imbalance=1.0, spread_pct=0.0, oi_change_pct=None,
        funding=snap.funding, premium=snap.premium, atr_pct=0.0, adx_value=0.0, volume_ratio=1.0,
        vwap_deviation_pct=0.0, macd_hist=0.0, daily_change_pct=0.0,
        bb_percent_b=0.5, stoch_rsi_k=50.0, htf_trend_up=None,
    )


_HTF_EMA_FAST = 20
_HTF_EMA_SLOW = 50


def compute_htf_trend(htf_candles: list[dict]) -> bool | None:
    """Trend read from a slower timeframe's candles, fixed EMA periods
    (not genome-tunable - a shared macro context, not a per-agent knob).
    Returns None if there isn't enough history yet."""
    closes = [c["c"] for c in htf_candles]
    if len(closes) < _HTF_EMA_SLOW + 2:
        return None
    fast = ema(closes, _HTF_EMA_FAST)
    slow = ema(closes, _HTF_EMA_SLOW)
    return bool(fast[-1] > slow[-1])


# Fixed, non-evolved threshold - unlike a genome's own tunable `min_adx`,
# this is used purely to LABEL a trade's market regime for reporting, so
# every agent's trades are classified on the same consistent scale
# regardless of what ADX threshold their own genome happens to trade on.
_REGIME_ADX_THRESHOLD = 20.0


def classify_regime(trend_up: bool, adx_value: float) -> str:
    """Labels the market condition a trade was opened in, for dashboard
    reporting only - does not affect entry/exit decisions."""
    if adx_value < _REGIME_ADX_THRESHOLD:
        return "ranging"
    return "trending-up" if trend_up else "trending-down"


def build_features(snap: MarketSnapshot, genome: Genome, prev_open_interest: float | None,
                    htf_trend_up: bool | None = None) -> Features:
    closes = [c["c"] for c in snap.candles]
    highs = [c["h"] for c in snap.candles]
    lows = [c["l"] for c in snap.candles]
    volumes = [c["v"] for c in snap.candles]

    min_required = max(
        genome.ema_slow, genome.rsi_period, genome.atr_period, genome.adx_period,
        genome.vwap_period, genome.volume_lookback, genome.bb_period,
        genome.stoch_rsi_period + genome.stoch_k_smooth,
        genome.ema_slow + genome.macd_signal_period,
    ) + 2
    if len(closes) < min_required:
        return _neutral_features(snap)

    ema_fast_series = ema(closes, genome.ema_fast)
    ema_slow_series = ema(closes, genome.ema_slow)
    rsi_series = rsi(closes, genome.rsi_period)
    atr_series = atr(highs, lows, closes, genome.atr_period)
    adx_series = adx(highs, lows, closes, genome.adx_period)
    macd_series = macd_histogram(closes, genome.ema_fast, genome.ema_slow, genome.macd_signal_period)
    bb_series = bollinger_percent_b(closes, genome.bb_period, genome.bb_std_dev)
    stoch_series = stochastic_rsi(closes, genome.stoch_rsi_period, genome.stoch_rsi_period, genome.stoch_k_smooth)

    bid_vol = sum(l["sz"] for l in snap.bid_levels[:10])
    ask_vol = sum(l["sz"] for l in snap.ask_levels[:10])
    ob_imbalance = bid_vol / ask_vol if ask_vol > 0 else 2.0

    best_bid = snap.bid_levels[0]["px"] if snap.bid_levels else snap.mid_price
    best_ask = snap.ask_levels[0]["px"] if snap.ask_levels else snap.mid_price
    spread_pct = (best_ask - best_bid) / snap.mid_price * 100.0 if snap.mid_price else 0.0

    oi_change_pct = None
    if prev_open_interest and prev_open_interest > 0:
        oi_change_pct = (snap.open_interest - prev_open_interest) / prev_open_interest * 100.0

    recent_volumes = volumes[-genome.volume_lookback:]
    avg_volume = sum(recent_volumes) / len(recent_volumes) if recent_volumes else 0.0
    volume_ratio = (volumes[-1] / avg_volume) if avg_volume > 0 else 1.0

    vwap_value = vwap(closes, volumes, genome.vwap_period)
    vwap_deviation_pct = (snap.mid_price - vwap_value) / vwap_value * 100.0 if vwap_value else 0.0

    daily_change_pct = (
        (snap.mid_price - snap.prev_day_price) / snap.prev_day_price * 100.0
        if snap.prev_day_price else 0.0
    )

    return Features(
        mid_price=snap.mid_price,
        ema_fast=float(ema_fast_series[-1]),
        ema_slow=float(ema_slow_series[-1]),
        trend_up=bool(ema_fast_series[-1] > ema_slow_series[-1]),
        rsi_value=float(rsi_series[-1]),
        ob_imbalance=ob_imbalance,
        spread_pct=spread_pct,
        oi_change_pct=oi_change_pct,
        funding=snap.funding,
        premium=snap.premium,
        atr_pct=float(atr_series[-1]) / snap.mid_price * 100.0 if snap.mid_price else 0.0,
        adx_value=float(adx_series[-1]),
        volume_ratio=volume_ratio,
        vwap_deviation_pct=vwap_deviation_pct,
        macd_hist=float(macd_series[-1]),
        daily_change_pct=daily_change_pct,
        bb_percent_b=float(bb_series[-1]),
        stoch_rsi_k=float(stoch_series[-1]),
        htf_trend_up=htf_trend_up,
    )


def evaluate_entry(genome: Genome, f: Features) -> Signal:
    # Hard regime filters - fail either and there's no point even looking for
    # a setup, regardless of how good it might otherwise look.
    if f.spread_pct > genome.max_spread_pct:
        return Signal("hold", 0.0, 0, [f"spread {f.spread_pct:.3f}% too wide (max {genome.max_spread_pct}%)"], ambiguous=False)
    if f.atr_pct < genome.min_atr_pct:
        return Signal("hold", 0.0, 0, [f"volatility too low (ATR {f.atr_pct:.3f}% < {genome.min_atr_pct}%)"], ambiguous=False)
    if f.atr_pct > genome.max_atr_pct:
        return Signal("hold", 0.0, 0, [f"volatility too chaotic (ATR {f.atr_pct:.3f}% > {genome.max_atr_pct}%)"], ambiguous=False)
    if f.adx_value < genome.min_adx:
        return Signal("hold", 0.0, 0, [f"market ranging/choppy (ADX {f.adx_value:.1f} < {genome.min_adx})"], ambiguous=False)

    reasons: list[str] = []

    if f.trend_up and f.rsi_value <= genome.rsi_oversold:
        candidate = "long"
        reasons.append(f"uptrend pullback: rsi {f.rsi_value:.1f} <= {genome.rsi_oversold}")
    elif not f.trend_up and f.rsi_value >= genome.rsi_overbought:
        candidate = "short"
        reasons.append(f"downtrend rally: rsi {f.rsi_value:.1f} >= {genome.rsi_overbought}")
    else:
        return Signal("hold", 0.0, 0, ["no trend+rsi trigger"], ambiguous=False)

    score = 1  # base trigger
    sign = 1 if candidate == "long" else -1

    # Order book imbalance: for a long, bid-heavy confirms; ask-heavy contradicts (mirrored for short).
    if sign * (f.ob_imbalance - 1) >= (genome.ob_imbalance_threshold - 1):
        score += 1
        reasons.append(f"order book favors {candidate} ({f.ob_imbalance:.2f}x)")
    elif sign * (f.ob_imbalance - 1) <= -(genome.ob_imbalance_threshold - 1):
        score -= 1
        reasons.append(f"order book contradicts {candidate} ({f.ob_imbalance:.2f}x)")

    # Open interest change: NOT direction-mirrored - rising OI means more
    # participants are piling into whatever the current move is (long or
    # short), so it confirms either candidate the same way; falling OI means
    # the move is losing participation, which contradicts either candidate.
    if f.oi_change_pct is not None:
        if f.oi_change_pct >= genome.oi_change_threshold:
            score += 1
            reasons.append(f"open interest rising {f.oi_change_pct:.2f}% with {candidate}")
        elif f.oi_change_pct <= -genome.oi_change_threshold:
            score -= 1
            reasons.append("open interest dropping - trend losing conviction")

    # Funding rate crowding: contrarian - a squeeze risk against you, a tailwind for you.
    if sign * f.funding >= genome.funding_extreme:
        score -= 1
        reasons.append(f"funding {f.funding:.5f} crowded {candidate} - squeeze risk")
    elif sign * f.funding <= -genome.funding_extreme:
        score += 1
        reasons.append(f"funding {f.funding:.5f} crowded opposite - squeeze tailwind")

    # Mark/oracle premium: same contrarian treatment as funding.
    if sign * f.premium >= genome.premium_extreme:
        score -= 1
        reasons.append(f"premium {f.premium:.5f} rich in {candidate} direction - chasing")
    elif sign * f.premium <= -genome.premium_extreme:
        score += 1
        reasons.append(f"premium {f.premium:.5f} cheap in {candidate} direction - room to run")

    # Volume conviction: a spike in participation backing the move.
    if f.volume_ratio >= genome.volume_spike_threshold:
        score += 1
        reasons.append(f"volume spike ({f.volume_ratio:.2f}x average) backs the move")

    # VWAP: buying below fair value / selling above it confirms; chasing far
    # past it on the wrong side of that logic contradicts.
    if sign * f.vwap_deviation_pct <= -genome.vwap_deviation_threshold:
        score += 1
        reasons.append(f"price {f.vwap_deviation_pct:.2f}% from VWAP - good {candidate} value")
    elif sign * f.vwap_deviation_pct >= genome.vwap_deviation_threshold:
        score -= 1
        reasons.append(f"price {f.vwap_deviation_pct:.2f}% from VWAP - chasing, extended")

    # MACD momentum in the trade's direction.
    if sign * f.macd_hist > 0:
        score += 1
        reasons.append(f"MACD histogram {f.macd_hist:.4f} backs {candidate} momentum")
    elif sign * f.macd_hist < 0:
        score -= 1
        reasons.append(f"MACD histogram {f.macd_hist:.4f} contradicts {candidate} momentum")

    # 24h macro momentum aligned with the trade direction.
    if sign * f.daily_change_pct >= genome.daily_momentum_threshold:
        score += 1
        reasons.append(f"24h change {f.daily_change_pct:.2f}% aligned with {candidate}")
    elif sign * f.daily_change_pct <= -genome.daily_momentum_threshold:
        score -= 1
        reasons.append(f"24h change {f.daily_change_pct:.2f}% fights the daily trend")

    # Higher-timeframe trend: trading against the bigger trend is a
    # stronger red flag than most single confirmations, so disagreement
    # costs 2 instead of the usual 1. None (unknown/backtest) is neutral.
    if f.htf_trend_up is not None:
        htf_agrees = f.htf_trend_up if candidate == "long" else not f.htf_trend_up
        if htf_agrees:
            score += 1
            reasons.append(f"higher-timeframe trend agrees with {candidate}")
        else:
            score -= 2
            reasons.append(f"higher-timeframe trend fights {candidate} - trading against the bigger trend")

    # Bollinger %B: near the band on your side of the trade confirms (bands
    # aren't symmetric around zero like the signals above, so this is
    # written explicitly per direction rather than via the sign trick).
    if candidate == "long":
        if f.bb_percent_b <= genome.bb_entry_threshold:
            score += 1
            reasons.append(f"price near lower Bollinger band (%B={f.bb_percent_b:.2f})")
        elif f.bb_percent_b >= 1 - genome.bb_entry_threshold:
            score -= 1
            reasons.append(f"price near upper Bollinger band (%B={f.bb_percent_b:.2f}) - extended")
    else:
        if f.bb_percent_b >= 1 - genome.bb_entry_threshold:
            score += 1
            reasons.append(f"price near upper Bollinger band (%B={f.bb_percent_b:.2f})")
        elif f.bb_percent_b <= genome.bb_entry_threshold:
            score -= 1
            reasons.append(f"price near lower Bollinger band (%B={f.bb_percent_b:.2f}) - extended")

    # Stochastic RSI: a faster, more sensitive oversold/overbought read than RSI itself.
    if candidate == "long":
        if f.stoch_rsi_k <= genome.stoch_rsi_oversold:
            score += 1
            reasons.append(f"StochRSI oversold ({f.stoch_rsi_k:.1f})")
        elif f.stoch_rsi_k >= genome.stoch_rsi_overbought:
            score -= 1
            reasons.append(f"StochRSI overbought ({f.stoch_rsi_k:.1f}) - contradicts")
    else:
        if f.stoch_rsi_k >= genome.stoch_rsi_overbought:
            score += 1
            reasons.append(f"StochRSI overbought ({f.stoch_rsi_k:.1f})")
        elif f.stoch_rsi_k <= genome.stoch_rsi_oversold:
            score -= 1
            reasons.append(f"StochRSI oversold ({f.stoch_rsi_k:.1f}) - contradicts")

    # Up to 11 confirmation dimensions now (order book, OI, funding, premium,
    # volume, VWAP, MACD, daily momentum, Bollinger, StochRSI, higher-
    # timeframe trend) on top of the base trigger, so the score range is
    # wider than a simple +-1 vote.
    if score < 0:
        return Signal("hold", 0.0, score, reasons + ["net contradicted, skipping"], ambiguous=False)
    if score <= 2:
        return Signal(candidate, 0.4, score, reasons, ambiguous=True)

    confidence = min(1.0, score / 9)
    return Signal(candidate, confidence, score, reasons, ambiguous=False)


def council_consult(
    candidate: Signal,
    snap: MarketSnapshot,
    prev_open_interest: float | None,
    htf_trend_up: bool | None,
    council_genomes: list[Genome],
    quorum_pct: float,
    min_active_voters: int,
) -> Signal:
    """Ensemble second opinion for an ambiguous signal: independently runs
    every council agent's OWN genome against the same market snapshot and
    checks whether enough of them agree. A council member's `hold` just
    means ITS unrelated genome/thresholds didn't trigger on this snapshot -
    not that it disagrees with the candidate - so holds are excluded from
    the quorum math entirely; only long/short votes count as "active."

    Returns a decisive Signal (ambiguous=False) if the candidate direction
    is confirmed or vetoed by quorum; otherwise returns `candidate`
    unchanged (still ambiguous) so the caller can fall back to Ollama, same
    as before this existed.
    """
    long_votes = short_votes = 0
    for genome in council_genomes:
        features = build_features(snap, genome, prev_open_interest, htf_trend_up)
        vote = evaluate_entry(genome, features)
        if vote.action == "long":
            long_votes += 1
        elif vote.action == "short":
            short_votes += 1

    active = long_votes + short_votes
    if active < min_active_voters:
        return candidate

    agree = long_votes if candidate.action == "long" else short_votes
    oppose = short_votes if candidate.action == "long" else long_votes

    if agree / active >= quorum_pct:
        return Signal(
            candidate.action, round(agree / active, 2), candidate.score,
            candidate.reasons + [f"council: {agree}/{active} active voters agree {candidate.action}"],
            ambiguous=False,
        )
    if oppose / active >= quorum_pct:
        opposite = "short" if candidate.action == "long" else "long"
        return Signal(
            "hold", 0.0, candidate.score,
            candidate.reasons + [f"council: {oppose}/{active} active voters favor {opposite} instead - vetoed"],
            ambiguous=False,
        )
    return candidate


def evaluate_exit(genome: Genome, trade_row, f: Features, now: datetime | None = None) -> tuple[str, str] | None:
    """Returns (result, reason) if the open position should close, else None.

    `now` defaults to wall-clock time for live trading; the backtest engine
    (backtest/engine.py) passes the simulated candle timestamp instead so
    the exact same function drives both live and backtested max-hold logic.
    """
    side = trade_row["side"]
    entry = trade_row["entry_price"]
    price = f.mid_price

    if side == "long":
        pnl_pct = (price - entry) / entry * 100.0
    else:
        pnl_pct = (entry - price) / entry * 100.0

    if pnl_pct >= genome.take_profit_pct:
        return "win", f"take profit hit ({pnl_pct:.2f}%)"
    if pnl_pct <= -genome.stop_loss_pct:
        return "loss", f"stop loss hit ({pnl_pct:.2f}%)"

    opened_at = datetime.fromisoformat(trade_row["opened_at"])
    current_time = now if now is not None else datetime.now(timezone.utc)
    hours_open = (current_time - opened_at).total_seconds() / 3600.0
    if hours_open >= genome.max_hold_hours:
        result = "win" if pnl_pct > 0 else "loss"
        return result, f"max hold {genome.max_hold_hours}h reached ({pnl_pct:.2f}%)"

    return None
