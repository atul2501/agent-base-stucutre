"""Covers trading/paper_executor.py's risk-normalized position sizing -
shrinks size when the actual stop distance is wider than the genome's own
baseline, and must never size UP beyond the flat position_size_pct ceiling."""
from trading.paper_executor import risk_normalized_size_pct


class TestRiskNormalizedSizePct:
    def test_unchanged_when_actual_stop_matches_baseline(self):
        assert risk_normalized_size_pct(10.0, baseline_stop_pct=3.0, actual_stop_pct=3.0) == 10.0

    def test_unchanged_when_actual_stop_is_tighter_than_baseline(self):
        # Never sizes UP for a tighter-than-baseline stop.
        assert risk_normalized_size_pct(10.0, baseline_stop_pct=3.0, actual_stop_pct=1.5) == 10.0

    def test_shrinks_when_actual_stop_is_wider_than_baseline(self):
        result = risk_normalized_size_pct(10.0, baseline_stop_pct=3.0, actual_stop_pct=6.0)
        assert result == 5.0  # half the stop-baseline ratio -> half the size

    def test_never_exceeds_the_flat_ceiling(self):
        for actual_stop_pct in (0.5, 1.0, 3.0, 6.0, 12.0):
            result = risk_normalized_size_pct(10.0, baseline_stop_pct=3.0, actual_stop_pct=actual_stop_pct)
            assert result <= 10.0

    def test_handles_zero_baseline_without_dividing_by_zero(self):
        assert risk_normalized_size_pct(10.0, baseline_stop_pct=0.0, actual_stop_pct=5.0) == 10.0
