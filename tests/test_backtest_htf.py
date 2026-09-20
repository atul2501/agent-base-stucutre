"""Covers backtest/engine.py's higher-timeframe trend backtesting
(_precompute_htf / _htf_trend_at) - closes part of the backtest/live
feature-parity gap (see Priority 4) by letting the HTF filter be backtested
for real using historical price data, instead of always neutral."""
from backtest.engine import _htf_trend_at, _precompute_htf


def _htf_candles(n: int, start_ms: int = 0, step_ms: int = 3_600_000, uptrend: bool = True) -> list[dict]:
    candles = []
    price = 100.0
    for i in range(n):
        price += (0.5 if uptrend else -0.5)
        candles.append({"t": start_ms + i * step_ms, "o": price, "h": price, "l": price, "c": price, "v": 1.0})
    return candles


class TestPrecomputeHtf:
    def test_returns_none_with_too_little_history(self):
        assert _precompute_htf(_htf_candles(10)) is None

    def test_returns_none_when_no_candles_given(self):
        assert _precompute_htf(None) is None
        assert _precompute_htf([]) is None

    def test_returns_a_series_with_enough_history(self):
        htf = _precompute_htf(_htf_candles(60))
        assert htf is not None
        assert len(htf.timestamps) == 60


class TestHtfTrendAt:
    def test_none_series_is_neutral(self):
        assert _htf_trend_at(None, ts_ms=1_000_000) is None

    def test_none_before_enough_history_has_closed(self):
        htf = _precompute_htf(_htf_candles(60))
        # Right at the start of the series - not enough fully-closed history yet.
        assert _htf_trend_at(htf, ts_ms=htf.timestamps[5]) is None

    def test_reads_uptrend_from_a_rising_price_series(self):
        candles = _htf_candles(80, uptrend=True)
        htf = _precompute_htf(candles)
        late_ts = candles[-1]["t"] + 3_600_000  # well after the series closed
        assert _htf_trend_at(htf, ts_ms=late_ts) is True

    def test_reads_downtrend_from_a_falling_price_series(self):
        candles = _htf_candles(80, uptrend=False)
        htf = _precompute_htf(candles)
        late_ts = candles[-1]["t"] + 3_600_000
        assert _htf_trend_at(htf, ts_ms=late_ts) is False

    def test_never_uses_a_candle_that_has_not_fully_closed_yet(self):
        step_ms = 3_600_000
        candles = _htf_candles(80, step_ms=step_ms, uptrend=False)  # steady downtrend...
        candles[-1] = {**candles[-1], "c": candles[-1]["c"] + 5000.0}  # ...until one huge final spike
        htf = _precompute_htf(candles)

        # Right as the last (spike) candle OPENS - it hasn't closed yet, so
        # this must read the same as if that candle didn't exist: downtrend.
        at_open = _htf_trend_at(htf, ts_ms=candles[-1]["t"])
        # Once it has fully closed (one candle_ms later) - now visible.
        at_close = _htf_trend_at(htf, ts_ms=candles[-1]["t"] + step_ms)

        assert at_open is False
        assert at_close is True
