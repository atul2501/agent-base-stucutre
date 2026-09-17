"""Simulated order fills against real market prices. No real funds are moved.

Applies a small slippage + taker fee model so paper PnL is a realistic
estimate rather than a frictionless fantasy number.
"""
from __future__ import annotations

from config import CONFIG


def simulate_fill_price(mid_price: float, side: str, is_entry: bool, spread_pct: float = 0.0) -> float:
    """`spread_pct` is the CURRENT real bid/ask spread (see
    strategy/signals.py Features.spread_pct), if known. A fixed slippage
    assumption (CONFIG.slippage_bps) understates real cost during a
    volatility spike, when the spread itself widens well past 2bps - so the
    fill uses whichever is WORSE: the fixed floor, or half the live spread.
    Defaults to 0.0 (unknown - e.g. backtesting, which has no historical
    order-book series - see backtest/engine.py), which falls back to the
    fixed floor exactly as before this existed."""
    fixed_slip_pct = CONFIG.slippage_bps / 10_000.0
    half_spread_pct = (spread_pct / 100.0) / 2.0
    slip = mid_price * max(fixed_slip_pct, half_spread_pct)
    if (side == "long" and is_entry) or (side == "short" and not is_entry):
        return mid_price + slip
    return mid_price - slip


def open_paper_position(balance: float, mid_price: float, side: str, position_size_pct: float,
                         spread_pct: float = 0.0) -> tuple[float, float, float]:
    """Returns (fill_price, size, notional)."""
    notional = balance * (position_size_pct / 100.0)
    fill_price = simulate_fill_price(mid_price, side, is_entry=True, spread_pct=spread_pct)
    size = notional / fill_price
    return fill_price, size, notional


def close_paper_position(entry_price: float, mid_price: float, size: float, side: str,
                          spread_pct: float = 0.0, funding_cost: float = 0.0) -> tuple[float, float]:
    """Returns (exit_fill_price, net_pnl_after_fees_and_funding).

    `funding_cost` is the caller-computed total funding charged against this
    trade over its hold (positive = a cost to this position, e.g. a long
    paying positive funding; negative = a credit, e.g. a short receiving it -
    see engine/orchestrator.py and backtest/engine.py for how each computes
    it). Defaults to 0.0, so a caller that hasn't been updated to pass it
    keeps its exact previous behavior - funding was previously used only as
    an entry SIGNAL and never actually charged against simulated PnL, which
    systematically overstated returns for any position held across a funding
    interval."""
    exit_price = simulate_fill_price(mid_price, side, is_entry=False, spread_pct=spread_pct)
    gross = (exit_price - entry_price) * size if side == "long" else (entry_price - exit_price) * size
    fees = (entry_price * size + exit_price * size) * CONFIG.fee_rate
    return exit_price, gross - fees - funding_cost
