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
    "take_profit_pct": (2.0, 12.0),
    "max_hold_hours": (12, 96),
    "position_size_pct": (2.0, 15.0),        # % of agent balance risked as notional
}

# Fields where two values must stay ordered (lo < hi) - handled specially in
# random()/mutate() rather than mutated independently like the rest.
_ORDERED_PAIRS = [("ema_fast", "ema_slow"), ("min_atr_pct", "max_atr_pct")]


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
            stop_loss_pct=round(u("stop_loss_pct"), 2),
            take_profit_pct=round(u("take_profit_pct"), 2),
            max_hold_hours=round(u("max_hold_hours"), 1),
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

        return child
