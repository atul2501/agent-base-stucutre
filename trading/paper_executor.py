"""Simulated order fills against real market prices. No real funds are moved.

Applies a small slippage + taker fee model so paper PnL is a realistic
estimate rather than a frictionless fantasy number.
"""
from __future__ import annotations

from config import CONFIG


def simulate_fill_price(mid_price: float, side: str, is_entry: bool) -> float:
    slip = mid_price * (CONFIG.slippage_bps / 10_000.0)
    if (side == "long" and is_entry) or (side == "short" and not is_entry):
        return mid_price + slip
    return mid_price - slip


def open_paper_position(balance: float, mid_price: float, side: str, position_size_pct: float) -> tuple[float, float, float]:
    """Returns (fill_price, size, notional)."""
    notional = balance * (position_size_pct / 100.0)
    fill_price = simulate_fill_price(mid_price, side, is_entry=True)
    size = notional / fill_price
    return fill_price, size, notional


def close_paper_position(entry_price: float, mid_price: float, size: float, side: str) -> tuple[float, float]:
    """Returns (exit_fill_price, net_pnl_after_fees)."""
    exit_price = simulate_fill_price(mid_price, side, is_entry=False)
    gross = (exit_price - entry_price) * size if side == "long" else (entry_price - exit_price) * size
    fees = (entry_price * size + exit_price * size) * CONFIG.fee_rate
    return exit_price, gross - fees
