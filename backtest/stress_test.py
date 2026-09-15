"""Extreme-move stress tests for the exit logic (evaluate_exit +
paper_executor), independent of whether any genome's own entry trigger would
ever fire on these synthetic scenarios.

Backtesting genomes against real history (backtest/engine.py) proves
"performs reasonably on data that actually happened" - it says nothing about
whether the exit math stays SANE (no crash, no NaN/inf, no unbounded runaway
loss beyond what the position size and side make possible) when price moves
far outside anything the recent history contained: a flash crash, a violent
gap, a near-zero price. This is a repeatable regression check for exactly
that, runnable on demand via `python3 main.py --stress-test` - not something
that needs to run every cycle, only whenever the exit/paper-fill logic
changes.

Honesty note: a SHORT position's theoretical max loss is unbounded (price
can rise arbitrarily) - that's correct, real behavior, not a bug this
catches. A LONG position's max loss is bounded (price floors at 0), which
IS asserted below.
"""
from __future__ import annotations

import math
from datetime import datetime, timedelta, timezone

from strategy.genome import Genome
from strategy.signals import Features, evaluate_exit
from trading.paper_executor import close_paper_position


def _features(mid_price: float) -> Features:
    """Minimal valid Features for evaluate_exit, which only reads
    `f.mid_price` - every other field is neutral filler."""
    return Features(
        mid_price=mid_price, ema_fast=mid_price, ema_slow=mid_price, trend_up=True,
        rsi_value=50.0, ob_imbalance=1.0, spread_pct=0.0, oi_change_pct=None,
        funding=0.0, premium=0.0, atr_pct=0.5, adx_value=25.0, volume_ratio=1.0,
        vwap_deviation_pct=0.0, macd_hist=0.0, daily_change_pct=0.0,
        bb_percent_b=0.5, stoch_rsi_k=50.0, htf_trend_up=None,
    )


def _trade_row(side: str, entry_price: float, opened_at: datetime) -> dict:
    return {"side": side, "entry_price": entry_price, "opened_at": opened_at.isoformat()}


def default_genome() -> Genome:
    """A deterministic 'typical' genome (every field at its bounds
    midpoint) - reuses Genome.from_dict's existing backward-compat fill
    logic rather than hand-picking values, so this stays in sync
    automatically if BOUNDS ever changes."""
    return Genome.from_dict({"coin": "SOL", "timeframe": "3m"})


def _check(name: str, fn) -> dict:
    try:
        fn()
        return {"name": name, "passed": True, "detail": "ok"}
    except AssertionError as e:
        return {"name": name, "passed": False, "detail": str(e)}
    except Exception as e:
        return {"name": name, "passed": False, "detail": f"CRASHED: {type(e).__name__}: {e}"}


def run_stress_tests(genome: Genome, size: float = 10.0) -> list[dict]:
    """Runs every scenario against `genome`'s own stop_loss_pct/
    take_profit_pct/max_hold_hours and returns one {name, passed, detail}
    result per scenario. Never raises - a scenario that crashes is reported
    as a failed result, not an uncaught exception."""
    opened_at = datetime.now(timezone.utc) - timedelta(hours=1)
    results = []

    def flash_crash_long():
        entry = 100.0
        crashed = entry * 0.65  # -35% instant move
        trade_row = _trade_row("long", entry, opened_at)
        outcome = evaluate_exit(genome, trade_row, _features(crashed))
        assert outcome is not None, "expected the crash to close the position"
        result, _reason = outcome
        assert result == "loss", f"expected loss, got {result}"
        exit_price, pnl = close_paper_position(entry, crashed, size, "long")
        assert math.isfinite(exit_price) and math.isfinite(pnl), "non-finite fill/pnl"
        assert pnl < 0, f"expected a loss, got pnl={pnl}"
        assert pnl >= -(entry * size), f"long loss ${-pnl:.2f} exceeded the max possible (${entry*size:.2f})"

    def flash_spike_short():
        entry = 100.0
        spiked = entry * 1.40  # +40% instant move against the short
        trade_row = _trade_row("short", entry, opened_at)
        outcome = evaluate_exit(genome, trade_row, _features(spiked))
        assert outcome is not None, "expected the spike to close the position"
        result, _reason = outcome
        assert result == "loss", f"expected loss, got {result}"
        exit_price, pnl = close_paper_position(entry, spiked, size, "short")
        assert math.isfinite(exit_price) and math.isfinite(pnl), "non-finite fill/pnl"
        assert pnl < 0, f"expected a loss, got pnl={pnl}"
        # Short losses are correctly unbounded (price can rise arbitrarily) -
        # no upper-bound assertion here, unlike the long case above.

    def extreme_favorable_gap_long():
        entry = 100.0
        spiked = entry * 2.5  # +150% instant move - far past any take_profit_pct bound
        trade_row = _trade_row("long", entry, opened_at)
        outcome = evaluate_exit(genome, trade_row, _features(spiked))
        assert outcome is not None, "expected the spike to close the position"
        result, _reason = outcome
        assert result == "win", f"expected win, got {result}"
        exit_price, pnl = close_paper_position(entry, spiked, size, "long")
        assert math.isfinite(exit_price) and math.isfinite(pnl), "non-finite fill/pnl on a huge favorable move"
        assert pnl > 0, f"expected a win, got pnl={pnl}"

    def max_hold_timeout_tiny_move():
        entry = 100.0
        barely_moved = entry * 1.005  # +0.5%, below virtually any genome's stop/TP thresholds
        trade_row = _trade_row("long", entry, opened_at)
        far_future = opened_at + timedelta(hours=genome.max_hold_hours + 1)
        outcome = evaluate_exit(genome, trade_row, _features(barely_moved), now=far_future)
        assert outcome is not None, "expected max_hold_hours to force a close"
        result, _reason = outcome
        assert result == "win", f"tiny positive move past max_hold should count as a win, got {result}"

    def near_zero_price_no_crash():
        entry = 0.0001  # smallest realistic perp price scale
        crashed = entry * 0.30  # -70%
        trade_row = _trade_row("long", entry, opened_at)
        outcome = evaluate_exit(genome, trade_row, _features(crashed))
        assert outcome is not None
        result, _reason = outcome
        assert result == "loss"
        exit_price, pnl = close_paper_position(entry, crashed, size, "long")
        assert math.isfinite(exit_price) and math.isfinite(pnl), "division/overflow issue at tiny price scale"

    for name, fn in [
        ("flash_crash_long", flash_crash_long),
        ("flash_spike_short", flash_spike_short),
        ("extreme_favorable_gap_long", extreme_favorable_gap_long),
        ("max_hold_timeout_tiny_move", max_hold_timeout_tiny_move),
        ("near_zero_price_no_crash", near_zero_price_no_crash),
    ]:
        results.append(_check(name, fn))

    return results


def main() -> int:
    genome = default_genome()
    results = run_stress_tests(genome)
    print(f"Exit-logic stress test ({len(results)} scenarios, genome: SL={genome.stop_loss_pct}% "
          f"TP={genome.take_profit_pct}% max_hold={genome.max_hold_hours}h)")
    print("-" * 70)
    all_passed = True
    for r in results:
        status = "PASS" if r["passed"] else "FAIL"
        if not r["passed"]:
            all_passed = False
        print(f"[{status}] {r['name']}: {r['detail']}")
    print("-" * 70)
    print("ALL SCENARIOS PASSED" if all_passed else "SOME SCENARIOS FAILED - investigate before trusting live/paper fills under extreme moves")
    return 0 if all_passed else 1


if __name__ == "__main__":
    import sys
    sys.exit(main())
