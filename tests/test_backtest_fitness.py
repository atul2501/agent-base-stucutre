"""Covers backtest/engine.py's fitness_score - specifically the return_pct
shrinkage added so a low-trade-count genome's lucky return can't dominate
genome selection the way an unshrunk return_pct term used to allow."""
from backtest.engine import BacktestResult, fitness_score


def _result(trades, wins, return_pct, max_drawdown_pct=0.0):
    losses = trades - wins
    win_rate = wins / trades if trades else 0.0
    return BacktestResult(trades=trades, wins=wins, losses=losses, win_rate=win_rate,
                           return_pct=return_pct, max_drawdown_pct=max_drawdown_pct)


class TestFitnessScore:
    def test_untested_genome_scores_below_any_tested_one(self):
        untested = fitness_score(_result(trades=0, wins=0, return_pct=0.0))
        barely_tested = fitness_score(_result(trades=1, wins=0, return_pct=-1.0))
        assert untested < barely_tested

    def test_lucky_low_trade_count_return_no_longer_dominates_a_proven_track_record(self):
        # A single-trade genome that got lucky on one huge move...
        lucky = fitness_score(_result(trades=1, wins=1, return_pct=100.0))
        # ...must no longer beat a 30-trade genome with a much smaller but
        # consistent, proven return - before shrinking return_pct, the raw
        # +100.0 term alone would have made this impossible to lose.
        proven = fitness_score(_result(trades=30, wins=20, return_pct=20.0))
        assert proven > lucky

    def test_return_shrinkage_approaches_the_raw_return_as_trades_grow(self):
        small_sample = fitness_score(_result(trades=3, wins=2, return_pct=10.0))
        large_sample = fitness_score(_result(trades=300, wins=200, return_pct=10.0))
        # Same win rate and return_pct, but the large-sample genome's
        # return contributes much closer to the raw 10.0 than the
        # small-sample one's does (which is pulled toward 0).
        assert large_sample > small_sample

    def test_zero_trades_returns_the_sentinel_score(self):
        assert fitness_score(_result(trades=0, wins=0, return_pct=999.0)) == -1.0

    def test_drawdown_still_penalizes_score(self):
        no_drawdown = fitness_score(_result(trades=20, wins=10, return_pct=10.0, max_drawdown_pct=0.0))
        with_drawdown = fitness_score(_result(trades=20, wins=10, return_pct=10.0, max_drawdown_pct=20.0))
        assert with_drawdown < no_drawdown
