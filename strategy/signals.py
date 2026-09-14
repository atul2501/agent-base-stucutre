"""Turns a market snapshot + a genome into a trade decision.

Swing-entry logic: trend from EMA cross, trigger from RSI pullback/rally,
then confirmed or contradicted by order book imbalance, open-interest
change, and funding-rate crowding. A clear score fires a trade or a clear
HOLD; a genuinely mixed read is flagged `ambiguous` so the orchestrator can
consult the Ollama advisor instead of guessing.
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone

from market.hyperliquid_client import MarketSnapshot
from strategy.genome import Genome
from strategy.indicators import ema, rsi


@dataclass
class Features:
    mid_price: float
    ema_fast: float
    ema_slow: float
    trend_up: bool
    rsi_value: float
    ob_imbalance: float          # bid_vol / ask_vol over top 10 levels
    oi_change_pct: float | None  # None if no prior snapshot to diff against
    funding: float


@dataclass
class Signal:
    action: str          # "long" | "short" | "hold"
    confidence: float
    score: int
    reasons: list[str]
    ambiguous: bool


def build_features(snap: MarketSnapshot, genome: Genome, prev_open_interest: float | None) -> Features:
    closes = [c["c"] for c in snap.candles]
    if len(closes) < max(genome.ema_slow, genome.rsi_period) + 2:
        # Not enough history yet - treat as neutral/flat.
        return Features(
            mid_price=snap.mid_price, ema_fast=snap.mid_price, ema_slow=snap.mid_price,
            trend_up=True, rsi_value=50.0, ob_imbalance=1.0, oi_change_pct=None,
            funding=snap.funding,
        )

    ema_fast_series = ema(closes, genome.ema_fast)
    ema_slow_series = ema(closes, genome.ema_slow)
    rsi_series = rsi(closes, genome.rsi_period)

    bid_vol = sum(l["sz"] for l in snap.bid_levels[:10])
    ask_vol = sum(l["sz"] for l in snap.ask_levels[:10])
    ob_imbalance = bid_vol / ask_vol if ask_vol > 0 else 2.0

    oi_change_pct = None
    if prev_open_interest and prev_open_interest > 0:
        oi_change_pct = (snap.open_interest - prev_open_interest) / prev_open_interest * 100.0

    return Features(
        mid_price=snap.mid_price,
        ema_fast=float(ema_fast_series[-1]),
        ema_slow=float(ema_slow_series[-1]),
        trend_up=bool(ema_fast_series[-1] > ema_slow_series[-1]),
        rsi_value=float(rsi_series[-1]),
        ob_imbalance=ob_imbalance,
        oi_change_pct=oi_change_pct,
        funding=snap.funding,
    )


def evaluate_entry(genome: Genome, f: Features) -> Signal:
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

    if candidate == "long":
        if f.ob_imbalance >= genome.ob_imbalance_threshold:
            score += 1
            reasons.append(f"order book bid-heavy ({f.ob_imbalance:.2f}x)")
        elif f.ob_imbalance <= 1 / genome.ob_imbalance_threshold:
            score -= 1
            reasons.append(f"order book ask-heavy ({f.ob_imbalance:.2f}x) - contradicts")

        if f.oi_change_pct is not None:
            if f.oi_change_pct >= genome.oi_change_threshold:
                score += 1
                reasons.append(f"open interest rising {f.oi_change_pct:.2f}% with uptrend")
            elif f.oi_change_pct <= -genome.oi_change_threshold:
                score -= 1
                reasons.append("open interest dropping - trend losing conviction")

        if f.funding >= genome.funding_extreme:
            score -= 1
            reasons.append(f"funding {f.funding:.5f} crowded long - squeeze risk")
        elif f.funding <= -genome.funding_extreme:
            score += 1
            reasons.append(f"funding {f.funding:.5f} crowded short - squeeze tailwind")

    else:  # short
        if f.ob_imbalance <= 1 / genome.ob_imbalance_threshold:
            score += 1
            reasons.append(f"order book ask-heavy ({f.ob_imbalance:.2f}x)")
        elif f.ob_imbalance >= genome.ob_imbalance_threshold:
            score -= 1
            reasons.append(f"order book bid-heavy ({f.ob_imbalance:.2f}x) - contradicts")

        if f.oi_change_pct is not None:
            if f.oi_change_pct >= genome.oi_change_threshold:
                score += 1
                reasons.append(f"open interest rising {f.oi_change_pct:.2f}% with downtrend")
            elif f.oi_change_pct <= -genome.oi_change_threshold:
                score -= 1
                reasons.append("open interest dropping - trend losing conviction")

        if f.funding <= -genome.funding_extreme:
            score -= 1
            reasons.append(f"funding {f.funding:.5f} crowded short - squeeze risk")
        elif f.funding >= genome.funding_extreme:
            score += 1
            reasons.append(f"funding {f.funding:.5f} crowded long - squeeze tailwind")

    if score < 0:
        return Signal("hold", 0.0, score, reasons + ["net contradicted, skipping"], ambiguous=False)
    if score <= 1:
        return Signal(candidate, 0.4, score, reasons, ambiguous=True)

    confidence = min(1.0, score / 4)
    return Signal(candidate, confidence, score, reasons, ambiguous=False)


def evaluate_exit(genome: Genome, trade_row, f: Features) -> tuple[str, str] | None:
    """Returns (result, reason) if the open position should close, else None."""
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
    hours_open = (datetime.now(timezone.utc) - opened_at).total_seconds() / 3600.0
    if hours_open >= genome.max_hold_hours:
        result = "win" if pnl_pct > 0 else "loss"
        return result, f"max hold {genome.max_hold_hours}h reached ({pnl_pct:.2f}%)"

    return None
