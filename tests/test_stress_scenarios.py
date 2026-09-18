"""Wires the existing backtest/stress_test.py exit-logic scenarios (flash
crash, flash spike, extreme favorable gap, max-hold timeout, near-zero price)
into the automated suite so they run on every push instead of only when
someone remembers to run `python3 main.py --stress-test` by hand."""
from backtest.stress_test import default_genome, run_stress_tests


def test_all_exit_logic_stress_scenarios_pass():
    results = run_stress_tests(default_genome())
    failures = [r for r in results if not r["passed"]]
    assert not failures, f"Stress scenario(s) failed: {failures}"
