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
from strategy.genome import _MIN_TP_SL_RATIO, Genome
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

    bid_vol = sum(lvl["sz"] for lvl in snap.bid_levels[:10])
    ask_vol = sum(lvl["sz"] for lvl in snap.ask_levels[:10])
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

    # Higher-timeframe trend hard veto: unlike every other confirmation
    # above, disagreement here isn't a score penalty - it's a full veto,
    # since trading against the bigger trend is qualitatively different
    # from one indicator disagreeing. None (unknown/backtest - see
    # compute_htf_trend and backtest/engine.py) stays neutral, same as
    # before. Runs last so `reasons`/`score` still document the full
    # setup that got vetoed.
    if f.htf_trend_up is not None:
        htf_agrees = f.htf_trend_up if candidate == "long" else not f.htf_trend_up
        if htf_agrees:
            score += 1
            reasons.append(f"higher-timeframe trend agrees with {candidate}")
        else:
            reasons.append(f"higher-timeframe trend fights {candidate} - vetoed, trading against the bigger trend")
            return Signal("hold", 0.0, score, reasons, ambiguous=False)

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


def _poll_council_votes(
    council_genomes: list[Genome],
    snap: MarketSnapshot,
    prev_open_interest: float | None,
    htf_trend_up: bool | None,
) -> tuple[int, int]:
    """Runs every council agent's OWN genome against the same market
    snapshot and tallies (long_votes, short_votes). Shared by
    council_consult (entries) and council_oppose_position (open positions)."""
    long_votes = short_votes = 0
    for genome in council_genomes:
        features = build_features(snap, genome, prev_open_interest, htf_trend_up)
        vote = evaluate_entry(genome, features)
        if vote.action == "long":
            long_votes += 1
        elif vote.action == "short":
            short_votes += 1
    return long_votes, short_votes


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
    long_votes, short_votes = _poll_council_votes(council_genomes, snap, prev_open_interest, htf_trend_up)

    active = long_votes + short_votes
    if active < max(min_active_voters, 1):
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


def council_oppose_position(
    side: str,
    snap: MarketSnapshot,
    prev_open_interest: float | None,
    htf_trend_up: bool | None,
    council_genomes: list[Genome],
    quorum_pct: float,
    min_active_voters: int,
) -> tuple[bool, bool, str]:
    """Ensemble check for an OPEN position: does the council now favor the
    OPPOSITE side? This never closes or overrides anything itself - callers
    only use the result to decide whether to pull the position's stop
    tighter (see _council_tighten_stop), never to loosen a stop or force a
    close. Live-only advisory layer; never called from the backtester, since
    a live council/LLM opinion can't be replayed historically (see
    engine/orchestrator.py::_process_exits).

    Returns (opposed, inconclusive, reason):
      - opposed=True: quorum of active voters now favor the opposite side -
        caller should tighten the stop.
      - inconclusive=True (opposed always False here): too few council
        members took a directional stance this cycle to mean anything -
        caller may escalate to Ollama, same as the entry-side ladder.
      - both False: enough votes were cast but they didn't reach quorum
        against the held side - resolved, no action needed, no escalation.
    """
    long_votes, short_votes = _poll_council_votes(council_genomes, snap, prev_open_interest, htf_trend_up)

    active = long_votes + short_votes
    if active < max(min_active_voters, 1):
        return False, True, ""

    oppose = short_votes if side == "long" else long_votes
    if oppose / active >= quorum_pct:
        opposite = "short" if side == "long" else "long"
        return True, False, f"exit council: {oppose}/{active} active voters now favor {opposite} - tightening stop"
    return False, False, ""


def _row_get(row, key: str, default=None):
    """Uniform accessor for both sqlite3.Row (raises IndexError on a
    missing/None-absent key) and a plain dict (raises KeyError) - lets
    evaluate_position() work against a full DB-backed trade row AND a
    minimal hand-built dict (e.g. backtest/stress_test.py's trade rows,
    which predate the ATR-adaptive/trailing/partial columns) without two
    code paths."""
    try:
        value = row[key]
    except (KeyError, IndexError):
        return default
    return default if value is None else value


@dataclass
class ExitLevels:
    stop_loss: float
    take_profit: float
    partial_target: float | None
    partial_frac: float


def compute_exit_levels(genome: Genome, side: str, entry_price: float, atr_pct: float) -> ExitLevels:
    """ATR-adaptive stop/target computed once at entry (and re-used
    unchanged by the trailing-stop logic in evaluate_position, which only
    ever moves stop_loss - never take_profit or the partial target).

    Blends the genome's own evolved stop_loss_pct/take_profit_pct (the
    sanity envelope, still meaningful and still mutated/evolved) with a
    volatility-scaled distance driven by the CURRENT ATR% at entry, so the
    same genome's exits widen in a chaotic market and tighten in a calm
    one instead of always using the same fixed %. The ATR-based distance
    is clamped to [0.5x, 2x] of the pct-based distance so a freak ATR
    reading at entry can't produce a nonsensical stop/target, and the
    final target:stop ratio is re-floored at _MIN_TP_SL_RATIO in case
    clamping pulled the two distances close together.
    """
    atr_pct = max(atr_pct, 1e-6)
    raw_stop_pct = genome.atr_stop_mult * atr_pct
    raw_target_pct = genome.atr_target_mult * atr_pct
    stop_pct = min(max(raw_stop_pct, genome.stop_loss_pct * 0.5), genome.stop_loss_pct * 2.0)
    target_pct = min(max(raw_target_pct, genome.take_profit_pct * 0.5), genome.take_profit_pct * 2.0)
    if target_pct < stop_pct * _MIN_TP_SL_RATIO:
        target_pct = stop_pct * _MIN_TP_SL_RATIO

    if side == "long":
        stop_loss = entry_price * (1 - stop_pct / 100.0)
        take_profit = entry_price * (1 + target_pct / 100.0)
    else:
        stop_loss = entry_price * (1 + stop_pct / 100.0)
        take_profit = entry_price * (1 - target_pct / 100.0)

    partial_target = None
    if genome.partial_tp_frac > 0:
        partial_dist = (take_profit - entry_price) * genome.partial_tp_r_mult
        partial_target = entry_price + partial_dist

    return ExitLevels(stop_loss=stop_loss, take_profit=take_profit,
                       partial_target=partial_target, partial_frac=genome.partial_tp_frac)


def _blended_result(trade_row, leg_pnl_pct: float) -> str:
    """Win/loss label for a closing leg, accounting for any partial
    take-profit already realized on this trade. Size-weighted blend of the
    partial leg's pnl_pct and this (remaining-size) leg's pnl_pct, so a
    position that banked a real partial profit and then closes its
    remainder at/near breakeven is correctly labeled a win overall for
    do-or-die purposes, instead of the remaining leg's own small loss
    killing an agent that was actually net profitable on the trade. With
    no partial taken (the default), partial_frac_taken is 0 and this
    reduces to exactly leg_pnl_pct > 0 - unchanged prior behavior."""
    partial_frac_taken = _row_get(trade_row, "partial_frac_taken", 0.0) or 0.0
    partial_pnl_pct = _row_get(trade_row, "partial_pnl_pct", 0.0) or 0.0
    blended = partial_frac_taken * partial_pnl_pct + (1 - partial_frac_taken) * leg_pnl_pct
    return "win" if blended > 0 else "loss"


def _trailing_stop_candidate(genome: Genome, side: str, entry: float, take_profit: float,
                              current_stop: float, price: float, atr_pct: float) -> float | None:
    """Returns a new (tighter) stop_loss price once the position has moved
    favorably past `trail_activation_frac` of the distance to target, or
    None if it isn't armed yet or hasn't improved - the stop only ever
    moves in the favorable direction, never loosens."""
    if genome.trail_distance_atr_mult <= 0:
        return None
    activation_dist = abs(take_profit - entry) * genome.trail_activation_frac
    trail_dist = max(atr_pct, 0.0) / 100.0 * price * genome.trail_distance_atr_mult
    if trail_dist <= 0:
        return None
    if side == "long":
        if price < entry + activation_dist:
            return None
        candidate = price - trail_dist
        return candidate if candidate > current_stop else None
    else:
        if price > entry - activation_dist:
            return None
        candidate = price + trail_dist
        return candidate if candidate < current_stop else None


def _council_tighten_stop(side: str, current_stop: float, price: float, tighten_frac: float) -> float | None:
    """Pulls the stop `tighten_frac` of the way from current_stop toward the
    current price, for the advisory exit-council layer (see
    council_oppose_position). Mirrors _trailing_stop_candidate's invariant -
    only ever returns a value STRICTER than current_stop, never looser;
    returns None if the computed candidate wouldn't actually tighten
    anything (e.g. tighten_frac <= 0)."""
    if side == "long":
        candidate = current_stop + (price - current_stop) * tighten_frac
        return candidate if candidate > current_stop else None
    else:
        candidate = current_stop - (current_stop - price) * tighten_frac
        return candidate if candidate < current_stop else None


@dataclass
class ExitAction:
    """One cycle's worth of exit decision for an open position.

    kind:
      "none"    - nothing to do this cycle.
      "trail"   - tighten trade_row's stored stop_loss to new_stop_loss;
                  position stays open.
      "partial" - close close_fraction of the remaining size at market
                  now, move the stop to new_stop_loss (breakeven), and
                  record partial_frac_taken/partial_pnl_pct on the trade
                  row; position stays open for the rest.
      "close"   - close the entire remaining position; `result` is the
                  do-or-die win/loss label.
    """
    kind: str
    result: str | None = None
    reason: str = ""
    new_stop_loss: float | None = None
    close_fraction: float = 1.0
    pnl_pct: float = 0.0


def evaluate_position(genome: Genome, trade_row, f: Features, now: datetime | None = None) -> ExitAction:
    """Single source of truth for what should happen to an open position
    this cycle - partial take-profit, trailing-stop tightening, or a full
    close (take-profit / stop-loss / max-hold). Used identically by live
    trading (engine/orchestrator.py) and the backtester
    (backtest/engine.py) so the two can never drift into different exit
    behavior - see evaluate_exit() below for the narrower backward-
    compatible wrapper still used by backtest/stress_test.py.

    `now` defaults to wall-clock time for live trading; the backtest engine
    passes the simulated candle timestamp instead so the exact same
    function drives both live and backtested max-hold logic.
    """
    side = trade_row["side"]
    entry = trade_row["entry_price"]
    price = f.mid_price

    stop_loss = _row_get(trade_row, "stop_loss")
    take_profit = _row_get(trade_row, "take_profit")
    if stop_loss is None or take_profit is None:
        # Backward-compat fallback for a trade row that predates stored
        # price levels (an old open trade from before this feature, or a
        # caller like backtest/stress_test.py that hand-builds a minimal
        # row) - recompute fresh off entry using the genome's plain pct
        # genes, exactly matching the pre-ATR-adaptive behavior.
        if side == "long":
            stop_loss = entry * (1 - genome.stop_loss_pct / 100)
            take_profit = entry * (1 + genome.take_profit_pct / 100)
        else:
            stop_loss = entry * (1 + genome.stop_loss_pct / 100)
            take_profit = entry * (1 - genome.take_profit_pct / 100)

    if side == "long":
        pnl_pct = (price - entry) / entry * 100.0
    else:
        pnl_pct = (entry - price) / entry * 100.0

    partial_taken = bool(_row_get(trade_row, "partial_taken", 0))
    partial_target = _row_get(trade_row, "partial_target")

    # 1. Partial take-profit - fires at most once per trade.
    if not partial_taken and partial_target is not None and genome.partial_tp_frac > 0:
        hit = price >= partial_target if side == "long" else price <= partial_target
        if hit:
            return ExitAction(kind="partial", reason=f"partial target hit ({pnl_pct:.2f}%)",
                               close_fraction=genome.partial_tp_frac, new_stop_loss=entry, pnl_pct=pnl_pct)

    # 2. Full take-profit / stop-loss against the (possibly trailed) stored levels.
    if side == "long":
        if price >= take_profit:
            return ExitAction(kind="close", result="win", reason=f"take profit hit ({pnl_pct:.2f}%)", pnl_pct=pnl_pct)
        if price <= stop_loss:
            return ExitAction(kind="close", result=_blended_result(trade_row, pnl_pct),
                               reason=f"stop loss hit ({pnl_pct:.2f}%)", pnl_pct=pnl_pct)
    else:
        if price <= take_profit:
            return ExitAction(kind="close", result="win", reason=f"take profit hit ({pnl_pct:.2f}%)", pnl_pct=pnl_pct)
        if price >= stop_loss:
            return ExitAction(kind="close", result=_blended_result(trade_row, pnl_pct),
                               reason=f"stop loss hit ({pnl_pct:.2f}%)", pnl_pct=pnl_pct)

    # 3. Max-hold time fallback.
    opened_at = datetime.fromisoformat(trade_row["opened_at"])
    current_time = now if now is not None else datetime.now(timezone.utc)
    hours_open = (current_time - opened_at).total_seconds() / 3600.0
    if hours_open >= genome.max_hold_hours:
        return ExitAction(kind="close", result=_blended_result(trade_row, pnl_pct),
                           reason=f"max hold {genome.max_hold_hours}h reached ({pnl_pct:.2f}%)", pnl_pct=pnl_pct)

    # 4. Trailing stop tightening - only reached if nothing above closed/partialed this cycle.
    new_stop = _trailing_stop_candidate(genome, side, entry, take_profit, stop_loss, price, f.atr_pct)
    if new_stop is not None:
        return ExitAction(kind="trail", new_stop_loss=new_stop, pnl_pct=pnl_pct)

    return ExitAction(kind="none", pnl_pct=pnl_pct)


def evaluate_exit(genome: Genome, trade_row, f: Features, now: datetime | None = None) -> tuple[str, str] | None:
    """Narrower backward-compatible view of evaluate_position(): returns
    (result, reason) on a full close, else None - trailing/partial-take-
    profit activity (which don't close the position) are invisible to this
    interface. Still used by backtest/stress_test.py, whose extreme-move
    scenarios only ever care about a full close or none."""
    action = evaluate_position(genome, trade_row, f, now=now)
    if action.kind == "close":
        return action.result, action.reason
    return None
