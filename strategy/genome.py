"""Genome = the swing-trading strategy parameters an agent carries and passes to its children."""
from __future__ import annotations

import copy
import random
from dataclasses import asdict, dataclass

# (min, max) bounds used both for random gen-0 genomes and to clamp mutations.
BOUNDS = {
    "ema_fast": (5, 20),
    "ema_slow": (21, 60),
    "rsi_period": (7, 21),
    "rsi_oversold": (15.0, 35.0),
    "rsi_overbought": (65.0, 85.0),
    "ob_imbalance_threshold": (1.1, 2.0),   # bid_vol/ask_vol ratio to count as one-sided
    "oi_change_threshold": (0.5, 5.0),       # % change in open interest considered significant
    "funding_extreme": (0.0003, 0.002),      # |funding| above this = crowded positioning
    "stop_loss_pct": (1.0, 5.0),
    "take_profit_pct": (2.0, 12.0),
    "max_hold_hours": (12, 96),
    "position_size_pct": (2.0, 15.0),        # % of agent balance risked as notional
}


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
    stop_loss_pct: float
    take_profit_pct: float
    max_hold_hours: float
    position_size_pct: float

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "Genome":
        return cls(**d)

    @classmethod
    def random(cls, coin: str, timeframe: str, rng: random.Random) -> "Genome":
        def u(name: str) -> float:
            lo, hi = BOUNDS[name]
            return rng.uniform(lo, hi)

        ema_fast = int(u("ema_fast"))
        ema_slow = max(int(u("ema_slow")), ema_fast + 5)
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
        numeric_fields = [f for f in BOUNDS.keys()]
        for field_name in numeric_fields:
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

        return child
