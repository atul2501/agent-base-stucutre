"""Minimal technical indicators (no TA-lib dependency)."""
from __future__ import annotations

import numpy as np


def ema(values: list[float], period: int) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    if len(values) == 0:
        return values
    alpha = 2.0 / (period + 1)
    out = np.empty_like(values)
    out[0] = values[0]
    for i in range(1, len(values)):
        out[i] = alpha * values[i] + (1 - alpha) * out[i - 1]
    return out


def rsi(values: list[float], period: int) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    n = len(values)
    out = np.full(n, 50.0)
    if n <= period:
        return out
    deltas = np.diff(values)
    gains = np.where(deltas > 0, deltas, 0.0)
    losses = np.where(deltas < 0, -deltas, 0.0)

    avg_gain = gains[:period].mean()
    avg_loss = losses[:period].mean()

    def rsi_from(avg_gain: float, avg_loss: float) -> float:
        if avg_loss == 0:
            return 100.0
        rs = avg_gain / avg_loss
        return 100.0 - (100.0 / (1.0 + rs))

    out[period] = rsi_from(avg_gain, avg_loss)
    for i in range(period + 1, n):
        gain = gains[i - 1]
        loss = losses[i - 1]
        avg_gain = (avg_gain * (period - 1) + gain) / period
        avg_loss = (avg_loss * (period - 1) + loss) / period
        out[i] = rsi_from(avg_gain, avg_loss)
    out[:period] = out[period]
    return out


def atr(highs: list[float], lows: list[float], closes: list[float], period: int) -> np.ndarray:
    """Average True Range - a volatility measure, used as a regime filter."""
    highs = np.asarray(highs, dtype=float)
    lows = np.asarray(lows, dtype=float)
    closes = np.asarray(closes, dtype=float)
    n = len(closes)
    if n < 2:
        return np.zeros(n)

    prev_close = np.concatenate([[closes[0]], closes[:-1]])
    true_range = np.maximum.reduce([
        highs - lows,
        np.abs(highs - prev_close),
        np.abs(lows - prev_close),
    ])

    out = np.zeros(n)
    if n <= period:
        out[:] = true_range.mean()
        return out
    out[period] = true_range[1:period + 1].mean()
    for i in range(period + 1, n):
        out[i] = (out[i - 1] * (period - 1) + true_range[i]) / period
    out[:period] = out[period]
    return out


def adx(highs: list[float], lows: list[float], closes: list[float], period: int) -> np.ndarray:
    """Average Directional Index (Wilder's) - trend STRENGTH, not direction.
    Low ADX (<~20) means the market is ranging/choppy regardless of which
    way price is drifting; high ADX means a real trend is underway. Used as
    a regime gate alongside ATR's volatility gate."""
    highs = np.asarray(highs, dtype=float)
    lows = np.asarray(lows, dtype=float)
    closes = np.asarray(closes, dtype=float)
    n = len(closes)
    if n < 2:
        return np.full(n, 0.0)

    up_move = highs[1:] - highs[:-1]
    down_move = lows[:-1] - lows[1:]
    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0.0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0.0)

    prev_close = closes[:-1]
    true_range = np.maximum.reduce([
        highs[1:] - lows[1:],
        np.abs(highs[1:] - prev_close),
        np.abs(lows[1:] - prev_close),
    ])

    m = len(true_range)  # = n - 1
    if m <= period:
        return np.full(n, 0.0)

    def _wilder_smooth(series: np.ndarray) -> np.ndarray:
        out = np.zeros(m)
        out[period - 1] = series[:period].sum()
        for i in range(period, m):
            out[i] = out[i - 1] - (out[i - 1] / period) + series[i]
        return out

    smoothed_tr = _wilder_smooth(true_range)
    smoothed_plus_dm = _wilder_smooth(plus_dm)
    smoothed_minus_dm = _wilder_smooth(minus_dm)

    with np.errstate(divide="ignore", invalid="ignore"):
        plus_di = np.where(smoothed_tr > 0, 100.0 * smoothed_plus_dm / smoothed_tr, 0.0)
        minus_di = np.where(smoothed_tr > 0, 100.0 * smoothed_minus_dm / smoothed_tr, 0.0)
        di_sum = plus_di + minus_di
        dx = np.where(di_sum > 0, 100.0 * np.abs(plus_di - minus_di) / di_sum, 0.0)

    adx_out = np.zeros(m)
    # First valid DX is at index period-1, so a window of `period` DX values
    # ends at period-1 + (period-1) = 2*period-2, not 2*period-1 - the old
    # formula pulled in one extra DX value for the seed average.
    start = period - 1 + (period - 1)
    if start >= m:
        adx_out[period - 1:] = dx[period - 1:].mean() if m > period - 1 else 0.0
        out = np.zeros(n)
        out[1:] = adx_out
        out[:2] = out[2] if n > 2 else 0.0
        return out

    adx_out[start] = dx[period - 1:start + 1].mean()
    for i in range(start + 1, m):
        adx_out[i] = (adx_out[i - 1] * (period - 1) + dx[i]) / period
    adx_out[:start] = adx_out[start]

    out = np.zeros(n)
    out[1:] = adx_out
    out[0] = out[1] if n > 1 else 0.0
    return out


def macd_histogram(closes: list[float], fast_period: int, slow_period: int, signal_period: int) -> np.ndarray:
    """MACD histogram (MACD line minus its signal line) - positive = bullish momentum."""
    macd_line = ema(closes, fast_period) - ema(closes, slow_period)
    signal_line = ema(list(macd_line), signal_period)
    return macd_line - signal_line


def vwap(closes: list[float], volumes: list[float], period: int) -> float:
    """Volume-weighted average price over the trailing `period` candles."""
    closes = np.asarray(closes[-period:], dtype=float)
    volumes = np.asarray(volumes[-period:], dtype=float)
    total_volume = volumes.sum()
    if total_volume <= 0:
        return float(closes[-1]) if len(closes) else 0.0
    return float((closes * volumes).sum() / total_volume)


def bollinger_percent_b(closes: list[float], period: int, num_std: float) -> np.ndarray:
    """%B: where price sits within the Bollinger Bands. 0 = at lower band,
    1 = at upper band, 0.5 = at the middle (SMA)."""
    closes = np.asarray(closes, dtype=float)
    n = len(closes)
    out = np.full(n, 0.5)
    if n < period:
        return out
    for i in range(period - 1, n):
        window = closes[i - period + 1: i + 1]
        mean = window.mean()
        std = window.std()
        upper = mean + num_std * std
        lower = mean - num_std * std
        out[i] = 0.5 if upper == lower else (closes[i] - lower) / (upper - lower)
    out[:period - 1] = out[period - 1]
    return out


def stochastic_rsi(closes: list[float], rsi_period: int, stoch_period: int, k_smooth: int) -> np.ndarray:
    """Stochastic RSI %K (smoothed): how extreme the RSI itself is relative
    to its own recent range, scaled 0-100. More sensitive than raw RSI."""
    rsi_series = rsi(closes, rsi_period)
    n = len(rsi_series)
    raw_k = np.full(n, 50.0)
    for i in range(n):
        window = rsi_series[max(0, i - stoch_period + 1): i + 1]
        lo, hi = window.min(), window.max()
        raw_k[i] = 50.0 if hi == lo else (rsi_series[i] - lo) / (hi - lo) * 100.0

    k_smooth = max(1, k_smooth)
    out = np.empty(n)
    for i in range(n):
        window = raw_k[max(0, i - k_smooth + 1): i + 1]
        out[i] = window.mean()
    return out
