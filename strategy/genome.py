"""Genome = the swing-trading strategy parameters an agent carries and passes to its children.

Draws on every signal dimension Hyperliquid's public data exposes: price
trend + momentum (EMA/RSI/MACD), order book imbalance and spread, open
interest, funding rate, mark/oracle premium, volume conviction, volatility
regime (ATR), VWAP mean-reversion, and 24h momentum. Each agent gets its own
thresholds for all of it, which is what makes it a distinct strategy - and
what mutation perturbs when a winner spawns children.
"""
from __future__ import annotations

import copy
import math
import random
from dataclasses import asdict, dataclass

# (min, max) bounds used both for random gen-0 genomes and to clamp mutations.
# ATR/spread/premium bounds are calibrated against real SOL 5m data (~0.28%
# ATR, ~0.01% spread, ~0.0002 premium) so a random agent's filters usually
# admit normal market conditions rather than never firing.
BOUNDS = {
    "ema_fast": (5, 20),
    "ema_slow": (21, 60),
    "rsi_period": (7, 21),
    "rsi_oversold": (15.0, 35.0),
    "rsi_overbought": (65.0, 85.0),
    "ob_imbalance_threshold": (1.1, 2.0),    # bid_vol/ask_vol ratio to count as one-sided
    "oi_change_threshold": (0.5, 5.0),       # % change in open interest considered significant
    "funding_extreme": (0.0003, 0.002),      # |funding| above this = crowded positioning
    "premium_extreme": (0.0002, 0.0015),     # |mark-vs-oracle premium| above this = rich/cheap perp
    "volume_spike_threshold": (1.2, 3.0),    # candle volume vs its own average, to count as conviction
    "volume_lookback": (10, 30),             # periods for the volume average
    "atr_period": (7, 21),
    "min_atr_pct": (0.02, 0.15),             # below this ATR%, market's too dead to bother
    "max_atr_pct": (0.3, 1.2),               # above this ATR%, market's too chaotic to trust
    "adx_period": (10, 20),
    "min_adx": (10.0, 30.0),                 # below this ADX, market's ranging/choppy - skip it
    "max_spread_pct": (0.02, 0.15),          # wider than this bid/ask spread = too illiquid
    "vwap_period": (10, 50),
    "vwap_deviation_threshold": (0.1, 1.5),  # % price needs to sit away from VWAP to matter
    "macd_signal_period": (5, 12),
    "daily_momentum_threshold": (0.5, 4.0),  # % 24h move to count as a macro trend confirmation
    "bb_period": (10, 30),
    "bb_std_dev": (1.5, 3.0),
    "bb_entry_threshold": (0.1, 0.3),        # %B distance from a band edge that counts as a signal
    "stoch_rsi_period": (7, 21),
    "stoch_k_smooth": (2, 5),
    "stoch_rsi_oversold": (10.0, 30.0),
    "stoch_rsi_overbought": (70.0, 90.0),
    "stop_loss_pct": (1.0, 5.0),
    "take_profit_pct": (2.0, 22.0),
    "max_hold_hours": (12, 96),
    "position_size_pct": (2.0, 25.0),        # % of agent balance risked as notional
}

# Fields where two values must stay ordered (lo < hi) - handled specially in
# random()/mutate() rather than mutated independently like the rest.
_ORDERED_PAIRS = [("ema_fast", "ema_slow"), ("min_atr_pct", "max_atr_pct")]

# Minimum take_profit_pct:stop_loss_pct ratio - without this, SL/TP are drawn
# fully independently and can land on genomes needing an improbable win rate
# just to break even (e.g. SL=7%, TP=2.5%). 2.0x keeps the breakeven win rate
# around ~33% (comfortable margin over the ~0.11% round-trip fee/slippage
# cost) and stays feasible across the whole BOUNDS grid: at the worst case
# stop_loss_pct=5.0 (its max), the floor is take_profit_pct>=10.0, still
# comfortably under take_profit_pct's own 22.0 ceiling.
_MIN_TP_SL_RATIO = 2.0

# How far above the bare ratio floor _resample_tp_sl_ratio is willing to
# redraw take_profit_pct to (as a multiple of the floor itself) - keeps
# resampled values from clustering right back at the floor's edge.
_TP_RESAMPLE_CEILING_MULT = 1.4

# Heuristic cushion for _enforce_hold_window: how many hours a genome should
# be given per 1% of its own take_profit_pct target, so a big target never
# gets paired with a hold window too short to plausibly reach it. Not derived
# from real SOL move-rate data - a deliberately loose, order-of-magnitude
# guard rather than a precise timing model.
_MIN_HOLD_HOURS_PER_TP_PCT = 4.0


def _enforce_tp_sl_ratio(stop_loss_pct: float, take_profit_pct: float) -> tuple[float, float]:
    """Returns (stop_loss_pct, take_profit_pct): stop_loss_pct clamped into
    its own BOUNDS first (so the ratio floor below is always achievable even
    from a corrupted/out-of-bounds input, e.g. a hand-edited genome dict),
    then take_profit_pct bumped up if needed so it's >=
    _MIN_TP_SL_RATIO * stop_loss_pct. Same compute-then-clamp-the-dependent-
    field pattern as _ORDERED_PAIRS.

    Uses ceil (not round) for the 2-decimal floor: round() can land BELOW
    the true product for values like 4.85 * 1.5 = 7.275, which floats
    represent as 7.2749999999999995 and round() then rounds down to 7.27 -
    silently violating the invariant it's meant to enforce.

    This is the DETERMINISTIC form - used by from_dict(), which must stay
    idempotent for genomes reloaded from storage every cycle. Genome-creation
    call sites (random/mutate/crossover) use _resample_tp_sl_ratio instead,
    which has its own rng and avoids piling deficient draws up at exactly
    this floor.
    """
    lo, hi = BOUNDS["stop_loss_pct"]
    stop_loss_pct = max(lo, min(hi, stop_loss_pct))
    min_tp = math.ceil(stop_loss_pct * _MIN_TP_SL_RATIO * 100) / 100
    take_profit_pct = max(take_profit_pct, min(min_tp, BOUNDS["take_profit_pct"][1]))
    return stop_loss_pct, take_profit_pct


def _resample_tp_sl_ratio(rng: random.Random, stop_loss_pct: float, take_profit_pct: float) -> tuple[float, float]:
    """Like _enforce_tp_sl_ratio, but for genome-creation call sites that
    have their own rng. Clamping a deficient take_profit_pct to exactly the
    floor makes many genomes pile up bare-minimum on risk:reward - a real
    pattern a genome-quality review flagged (a thin edge once ~0.11%
    round-trip fees/slippage are subtracted). Instead, redraw it from a band
    comfortably above the floor. An already-compliant take_profit_pct is
    left untouched, preserving whatever value evolution actually produced."""
    lo, hi = BOUNDS["stop_loss_pct"]
    stop_loss_pct = max(lo, min(hi, stop_loss_pct))
    floor = stop_loss_pct * _MIN_TP_SL_RATIO
    if take_profit_pct >= floor:
        return stop_loss_pct, take_profit_pct
    cap = BOUNDS["take_profit_pct"][1]
    upper = min(floor * _TP_RESAMPLE_CEILING_MULT, cap)
    take_profit_pct = min(floor, cap) if upper <= floor else round(rng.uniform(floor, upper), 2)
    return stop_loss_pct, take_profit_pct


def _enforce_hold_window(take_profit_pct: float, max_hold_hours: float) -> float:
    """Returns max_hold_hours, bumped up if needed so it isn't too short for
    the genome's own take_profit_pct to plausibly be reached - the "big
    target, short hold window" incoherence a genome-quality review flagged.
    Never lowers max_hold_hours, so a genome that already gives itself a
    long window keeps it even for a tiny target."""
    min_hold = take_profit_pct * _MIN_HOLD_HOURS_PER_TP_PCT
    return max(max_hold_hours, min(min_hold, BOUNDS["max_hold_hours"][1]))


def genome_distance(g1: "Genome", g2: "Genome") -> float:
    """Mean absolute per-field distance between two genomes, each field
    normalized to its own BOUNDS span so a wide-range field (e.g.
    take_profit_pct's 2-22) can't dominate a narrow one (e.g. bb_std_dev's
    1.5-3.0). 0.0 = identical on every tunable field, up to ~1.0 = opposite
    extremes on every field. Used to steer new genomes away from near-clones
    of already-alive agents - see Population._pick_best."""
    d1, d2 = g1.to_dict(), g2.to_dict()
    diffs = []
    for field_name, (lo, hi) in BOUNDS.items():
        span = hi - lo
        if span <= 0:
            continue
        diffs.append(abs(d1[field_name] - d2[field_name]) / span)
    return sum(diffs) / len(diffs) if diffs else 0.0


@dataclass
class Genome:
    coin: str
    timeframe: str
    ema_fast: int
    ema_slow: int
    rsi_period: int
    rsi_oversold: float
    rsi_overbought: float
    ob_imbalance_threshold: float
    oi_change_threshold: float
    funding_extreme: float
    premium_extreme: float
    volume_spike_threshold: float
    volume_lookback: int
    atr_period: int
    min_atr_pct: float
    max_atr_pct: float
    adx_period: int
    min_adx: float
    max_spread_pct: float
    vwap_period: int
    vwap_deviation_threshold: float
    macd_signal_period: int
    daily_momentum_threshold: float
    bb_period: int
    bb_std_dev: float
    bb_entry_threshold: float
    stoch_rsi_period: int
    stoch_k_smooth: int
    stoch_rsi_oversold: float
    stoch_rsi_overbought: float
    stop_loss_pct: float
    take_profit_pct: float
    max_hold_hours: float
    position_size_pct: float

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "Genome":
        """Loads a genome, filling in any fields missing from an older schema
        (bounds midpoint) so existing trained agents survive an indicator-set
        expansion without needing a --reset. Idempotent and deterministic."""
        d = dict(d)
        for field_name, (lo, hi) in BOUNDS.items():
            if field_name not in d:
                mid = (lo + hi) / 2
                d[field_name] = int(round(mid)) if isinstance(lo, int) else round(mid, 5)
        if d["ema_slow"] <= d["ema_fast"]:
            d["ema_slow"] = d["ema_fast"] + 5
        if d["max_atr_pct"] <= d["min_atr_pct"]:
            d["max_atr_pct"] = round(d["min_atr_pct"] + 0.1, 4)
        d["stop_loss_pct"], d["take_profit_pct"] = _enforce_tp_sl_ratio(d["stop_loss_pct"], d["take_profit_pct"])
        d["max_hold_hours"] = _enforce_hold_window(d["take_profit_pct"], d["max_hold_hours"])
        return cls(**d)

    @classmethod
    def random(cls, coin: str, timeframe: str, rng: random.Random) -> "Genome":
        def u(name: str) -> float:
            lo, hi = BOUNDS[name]
            return rng.uniform(lo, hi)

        ema_fast = int(u("ema_fast"))
        ema_slow = max(int(u("ema_slow")), ema_fast + 5)
        min_atr_pct = round(u("min_atr_pct"), 4)
        max_atr_pct = max(round(u("max_atr_pct"), 4), min_atr_pct + 0.1)
        stop_loss_pct, take_profit_pct = _resample_tp_sl_ratio(
            rng, round(u("stop_loss_pct"), 2), round(u("take_profit_pct"), 2)
        )
        max_hold_hours = _enforce_hold_window(take_profit_pct, round(u("max_hold_hours"), 1))

        return cls(
            coin=coin,
            timeframe=timeframe,
            ema_fast=ema_fast,
            ema_slow=ema_slow,
            rsi_period=int(u("rsi_period")),
            rsi_oversold=round(u("rsi_oversold"), 1),
            rsi_overbought=round(u("rsi_overbought"), 1),
            ob_imbalance_threshold=round(u("ob_imbalance_threshold"), 2),
            oi_change_threshold=round(u("oi_change_threshold"), 2),
            funding_extreme=round(u("funding_extreme"), 5),
            premium_extreme=round(u("premium_extreme"), 5),
            volume_spike_threshold=round(u("volume_spike_threshold"), 2),
            volume_lookback=int(u("volume_lookback")),
            atr_period=int(u("atr_period")),
            min_atr_pct=min_atr_pct,
            max_atr_pct=max_atr_pct,
            adx_period=int(u("adx_period")),
            min_adx=round(u("min_adx"), 1),
            max_spread_pct=round(u("max_spread_pct"), 3),
            vwap_period=int(u("vwap_period")),
            vwap_deviation_threshold=round(u("vwap_deviation_threshold"), 2),
            macd_signal_period=int(u("macd_signal_period")),
            daily_momentum_threshold=round(u("daily_momentum_threshold"), 2),
            bb_period=int(u("bb_period")),
            bb_std_dev=round(u("bb_std_dev"), 2),
            bb_entry_threshold=round(u("bb_entry_threshold"), 2),
            stoch_rsi_period=int(u("stoch_rsi_period")),
            stoch_k_smooth=int(u("stoch_k_smooth")),
            stoch_rsi_oversold=round(u("stoch_rsi_oversold"), 1),
            stoch_rsi_overbought=round(u("stoch_rsi_overbought"), 1),
            stop_loss_pct=stop_loss_pct,
            take_profit_pct=take_profit_pct,
            max_hold_hours=max_hold_hours,
            position_size_pct=round(u("position_size_pct"), 2),
        )

    def mutate(self, rng: random.Random, mutation_rate: float = 0.25) -> "Genome":
        """Return a mutated copy - this is how a winning agent's children differ from it.

        Coin is intentionally never mutated: the population is scoped to one
        token at a time by design (see Population "step 0" reset) so every
        agent stays specialized to it.
        """
        child = copy.deepcopy(self)
        for field_name in BOUNDS.keys():
            if rng.random() > mutation_rate:
                continue
            lo, hi = BOUNDS[field_name]
            current = getattr(child, field_name)
            perturbation = current * rng.uniform(-0.2, 0.2)
            new_val = current + perturbation
            new_val = max(lo, min(hi, new_val))
            if isinstance(current, int):
                new_val = int(round(new_val))
            else:
                new_val = round(new_val, 5)
            setattr(child, field_name, new_val)

        if child.ema_slow <= child.ema_fast:
            child.ema_slow = child.ema_fast + 5
        if child.max_atr_pct <= child.min_atr_pct:
            child.max_atr_pct = round(child.min_atr_pct + 0.1, 4)
        child.stop_loss_pct, child.take_profit_pct = _resample_tp_sl_ratio(
            rng, child.stop_loss_pct, child.take_profit_pct
        )
        child.max_hold_hours = _enforce_hold_window(child.take_profit_pct, child.max_hold_hours)

        return child

    def crossover(self, other: "Genome", rng: random.Random) -> "Genome":
        """Combine roughly half of each parent's genes - used alongside
        (not instead of) mutation so a win can also draw on a second
        independently-successful lineage's traits, not just perturb its
        own genome. E.g. one parent's well-tuned volatility filter can end
        up paired with another parent's well-tuned VWAP logic."""
        child = copy.deepcopy(self)
        for field_name in BOUNDS.keys():
            if rng.random() < 0.5:
                setattr(child, field_name, getattr(other, field_name))

        if child.ema_slow <= child.ema_fast:
            child.ema_slow = child.ema_fast + 5
        if child.max_atr_pct <= child.min_atr_pct:
            child.max_atr_pct = round(child.min_atr_pct + 0.1, 4)
        child.stop_loss_pct, child.take_profit_pct = _resample_tp_sl_ratio(
            rng, child.stop_loss_pct, child.take_profit_pct
        )
        child.max_hold_hours = _enforce_hold_window(child.take_profit_pct, child.max_hold_hours)

        return child
