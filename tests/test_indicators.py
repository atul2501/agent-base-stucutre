"""Covers strategy/indicators.py's numeric correctness and edge-case
behavior on short/degenerate input - this system runs unattended, so a
too-short candle history or a flat market (both of which will happen in
production) must not crash or silently return nonsense."""
import numpy as np
import pytest

from strategy.indicators import adx, atr, bollinger_percent_b, ema, macd_histogram, rsi, stochastic_rsi, vwap


class TestEma:
    def test_constant_series_stays_constant(self):
        out = ema([100.0] * 20, period=10)
        assert np.allclose(out, 100.0)

    def test_empty_input_returns_empty(self):
        assert len(ema([], period=10)) == 0

    def test_reacts_toward_a_step_change(self):
        values = [100.0] * 10 + [200.0] * 10
        out = ema(values, period=5)
        assert out[9] == pytest.approx(100.0)
        assert out[-1] > out[10]  # trending up after the step
        assert out[-1] < 200.0     # hasn't fully caught up yet


class TestRsi:
    def test_short_series_returns_neutral_default(self):
        # n <= period: not enough data for a real reading yet.
        out = rsi([1.0, 2.0, 3.0], period=14)
        assert np.allclose(out, 50.0)

    def test_all_gains_saturates_high(self):
        values = list(range(1, 40))  # strictly increasing
        out = rsi([float(v) for v in values], period=14)
        assert out[-1] > 90.0

    def test_all_losses_saturates_low(self):
        values = list(range(40, 1, -1))  # strictly decreasing
        out = rsi([float(v) for v in values], period=14)
        assert out[-1] < 10.0

    def test_output_is_bounded_0_100(self):
        rng = np.random.default_rng(0)
        values = list(np.cumsum(rng.normal(size=100)) + 100)
        out = rsi(values, period=14)
        assert (out >= 0.0).all() and (out <= 100.0).all()


class TestAtr:
    def test_flat_ohlc_gives_zero_volatility(self):
        n = 20
        out = atr([100.0] * n, [100.0] * n, [100.0] * n, period=14)
        assert np.allclose(out, 0.0)

    def test_single_bar_does_not_crash(self):
        out = atr([100.0], [99.0], [99.5], period=14)
        assert len(out) == 1

    def test_is_never_negative(self):
        rng = np.random.default_rng(1)
        closes = list(np.cumsum(rng.normal(size=50)) + 100)
        highs = [c + abs(rng.normal()) for c in closes]
        lows = [c - abs(rng.normal()) for c in closes]
        out = atr(highs, lows, closes, period=14)
        assert (out >= 0.0).all()


class TestAdx:
    def test_short_series_does_not_crash(self):
        out = adx([1.0], [1.0], [1.0], period=14)
        assert len(out) == 1

    def test_output_is_bounded_0_100(self):
        rng = np.random.default_rng(2)
        closes = list(np.cumsum(rng.normal(size=100)) + 100)
        highs = [c + abs(rng.normal()) for c in closes]
        lows = [c - abs(rng.normal()) for c in closes]
        out = adx(highs, lows, closes, period=14)
        assert (out >= 0.0).all() and (out <= 100.0).all()


class TestMacdHistogram:
    def test_constant_series_is_flat_zero(self):
        out = macd_histogram([100.0] * 40, fast_period=12, slow_period=26, signal_period=9)
        assert np.allclose(out, 0.0, atol=1e-9)


class TestVwap:
    def test_matches_manual_weighted_average(self):
        closes = [10.0, 20.0, 30.0]
        volumes = [1.0, 1.0, 2.0]
        expected = (10 * 1 + 20 * 1 + 30 * 2) / 4
        assert vwap(closes, volumes, period=3) == pytest.approx(expected)

    def test_zero_total_volume_falls_back_to_last_close(self):
        assert vwap([10.0, 20.0], [0.0, 0.0], period=2) == pytest.approx(20.0)

    def test_empty_input_does_not_crash(self):
        assert vwap([], [], period=5) == 0.0


class TestBollingerPercentB:
    def test_short_series_returns_neutral_default(self):
        out = bollinger_percent_b([100.0, 101.0], period=20, num_std=2.0)
        assert np.allclose(out, 0.5)

    def test_flat_series_is_neutral_not_divide_by_zero(self):
        out = bollinger_percent_b([100.0] * 25, period=20, num_std=2.0)
        assert np.allclose(out, 0.5)


class TestStochasticRsi:
    def test_output_is_bounded_0_100(self):
        rng = np.random.default_rng(3)
        closes = list(np.cumsum(rng.normal(size=60)) + 100)
        out = stochastic_rsi(closes, rsi_period=14, stoch_period=14, k_smooth=3)
        assert (out >= 0.0).all() and (out <= 100.0).all()
