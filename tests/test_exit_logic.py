"""Covers strategy/signals.py's ATR-adaptive/trailing/partial exit machinery
(compute_exit_levels, evaluate_position, _blended_result,
_trailing_stop_candidate) - this determines when real (simulated) money
closes a position, so its invariants matter as much as genome.py's."""
from datetime import datetime, timedelta, timezone

import pytest

from strategy.genome import Genome
from strategy.signals import (
    Features, _blended_result, _trailing_stop_candidate, compute_exit_levels, evaluate_position,
)


def _features(mid_price: float, atr_pct: float = 1.0) -> Features:
    return Features(
        mid_price=mid_price, ema_fast=mid_price, ema_slow=mid_price, trend_up=True,
        rsi_value=50.0, ob_imbalance=1.0, spread_pct=0.0, oi_change_pct=None,
        funding=0.0, premium=0.0, atr_pct=atr_pct, adx_value=25.0, volume_ratio=1.0,
        vwap_deviation_pct=0.0, macd_hist=0.0, daily_change_pct=0.0,
        bb_percent_b=0.5, stoch_rsi_k=50.0, htf_trend_up=None,
    )


def _genome(**overrides) -> Genome:
    base = {
        "coin": "SOL", "timeframe": "1m",
        "stop_loss_pct": 3.0, "take_profit_pct": 10.0,
        "atr_stop_mult": 2.0, "atr_target_mult": 6.0,
        "trail_activation_frac": 0.5, "trail_distance_atr_mult": 1.0,
        "partial_tp_frac": 0.3, "partial_tp_r_mult": 0.5,
    }
    base.update(overrides)
    return Genome.from_dict(base)


def _trade_row(genome: Genome, side: str, entry: float, opened_at: datetime, **overrides) -> dict:
    levels = compute_exit_levels(genome, side, entry, atr_pct=1.0)
    row = {
        "side": side, "entry_price": entry, "opened_at": opened_at.isoformat(),
        "stop_loss": levels.stop_loss, "take_profit": levels.take_profit,
        "partial_target": levels.partial_target, "remaining_size": 10.0, "size": 10.0,
        "partial_taken": 0, "partial_frac_taken": 0.0, "partial_pnl_pct": 0.0,
    }
    row.update(overrides)
    return row


class TestComputeExitLevels:
    def test_long_levels_bracket_entry(self):
        genome = _genome()
        levels = compute_exit_levels(genome, "long", entry_price=100.0, atr_pct=1.0)
        assert levels.stop_loss < 100.0 < levels.take_profit

    def test_short_levels_bracket_entry(self):
        genome = _genome()
        levels = compute_exit_levels(genome, "short", entry_price=100.0, atr_pct=1.0)
        assert levels.take_profit < 100.0 < levels.stop_loss

    def test_target_stop_ratio_never_drops_below_floor(self):
        # Deliberately extreme ATR + mult combination that clamping alone
        # might not fix without the final re-floor step.
        genome = _genome(atr_stop_mult=4.0, atr_target_mult=8.0, stop_loss_pct=1.0, take_profit_pct=22.0)
        for atr_pct in (0.02, 0.5, 1.2, 5.0):
            levels = compute_exit_levels(genome, "long", entry_price=100.0, atr_pct=atr_pct)
            stop_dist = 100.0 - levels.stop_loss
            target_dist = levels.take_profit - 100.0
            assert target_dist >= stop_dist * 2.0 - 1e-9

    def test_partial_target_disabled_when_frac_is_zero(self):
        genome = _genome(partial_tp_frac=0.0)
        levels = compute_exit_levels(genome, "long", entry_price=100.0, atr_pct=1.0)
        assert levels.partial_target is None

    def test_partial_target_sits_between_entry_and_full_target(self):
        genome = _genome()
        levels = compute_exit_levels(genome, "long", entry_price=100.0, atr_pct=1.0)
        assert 100.0 < levels.partial_target < levels.take_profit


class TestEvaluatePositionFullClose:
    def test_take_profit_closes_as_win(self):
        genome = _genome(partial_tp_frac=0.0)  # isolate TP/SL behavior from partial-exit
        opened_at = datetime.now(timezone.utc) - timedelta(hours=1)
        row = _trade_row(genome, "long", 100.0, opened_at)
        action = evaluate_position(genome, row, _features(row["take_profit"] + 1))
        assert action.kind == "close"
        assert action.result == "win"

    def test_stop_loss_closes_as_loss(self):
        genome = _genome(partial_tp_frac=0.0)
        opened_at = datetime.now(timezone.utc) - timedelta(hours=1)
        row = _trade_row(genome, "long", 100.0, opened_at)
        action = evaluate_position(genome, row, _features(row["stop_loss"] - 1))
        assert action.kind == "close"
        assert action.result == "loss"

    def test_no_action_between_stop_and_target(self):
        genome = _genome(partial_tp_frac=0.0)
        opened_at = datetime.now(timezone.utc) - timedelta(hours=1)
        row = _trade_row(genome, "long", 100.0, opened_at)
        action = evaluate_position(genome, row, _features(100.1))
        assert action.kind in ("none", "trail")  # never a close this close to entry


class TestEvaluatePositionPartial:
    def test_partial_fires_once_at_partial_target(self):
        genome = _genome()
        opened_at = datetime.now(timezone.utc) - timedelta(hours=1)
        row = _trade_row(genome, "long", 100.0, opened_at)
        action = evaluate_position(genome, row, _features(row["partial_target"] + 0.01))
        assert action.kind == "partial"
        assert action.close_fraction == pytest.approx(genome.partial_tp_frac)
        assert action.new_stop_loss == pytest.approx(100.0)  # moved to breakeven

    def test_partial_does_not_refire_once_taken(self):
        genome = _genome()
        opened_at = datetime.now(timezone.utc) - timedelta(hours=1)
        row = _trade_row(genome, "long", 100.0, opened_at, partial_taken=1)
        action = evaluate_position(genome, row, _features(row["partial_target"] + 0.01))
        assert action.kind != "partial"


class TestTrailingStop:
    def test_does_not_arm_before_activation_distance(self):
        genome = _genome()
        candidate = _trailing_stop_candidate(
            genome, "long", entry=100.0, take_profit=106.0, current_stop=98.0,
            price=101.0, atr_pct=1.0,
        )
        assert candidate is None

    def test_arms_and_tightens_after_activation(self):
        genome = _genome()
        candidate = _trailing_stop_candidate(
            genome, "long", entry=100.0, take_profit=106.0, current_stop=98.0,
            price=105.0, atr_pct=1.0,
        )
        assert candidate is not None
        assert candidate > 98.0

    def test_never_loosens_an_existing_tighter_stop(self):
        genome = _genome()
        # current_stop already tighter than what this price/ATR would produce.
        candidate = _trailing_stop_candidate(
            genome, "long", entry=100.0, take_profit=106.0, current_stop=104.5,
            price=105.0, atr_pct=1.0,
        )
        assert candidate is None


class TestBlendedResult:
    def test_no_partial_taken_matches_plain_sign_check(self):
        row = {"partial_frac_taken": 0.0, "partial_pnl_pct": 0.0}
        assert _blended_result(row, leg_pnl_pct=1.0) == "win"
        assert _blended_result(row, leg_pnl_pct=-1.0) == "loss"

    def test_net_profitable_trade_wins_even_if_remaining_leg_is_flat(self):
        # A real partial profit was banked; the remainder closes at
        # breakeven (0%) - the trade was still net profitable overall and
        # must not be labeled a do-or-die loss.
        row = {"partial_frac_taken": 0.3, "partial_pnl_pct": 5.0}
        assert _blended_result(row, leg_pnl_pct=0.0) == "win"

    def test_net_losing_trade_still_loses_despite_a_tiny_partial(self):
        row = {"partial_frac_taken": 0.3, "partial_pnl_pct": 0.5}
        assert _blended_result(row, leg_pnl_pct=-3.0) == "loss"
