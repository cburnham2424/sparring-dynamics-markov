"""Tests for sparring_dynamics.simulation (Fighter + SparringMatch)."""
import numpy as np

from sparring_dynamics.simulation.fighter import Fighter, compute_lambda
from sparring_dynamics.simulation.match import SparringMatch
from sparring_dynamics.config import (
    F1_BASE_DEFAULT, F2_BASE_DEFAULT,
    F1_ADAPTATION_DEFAULT, F2_ADAPTATION_DEFAULT,
    F1_PAYOFF_DEFAULT, F2_PAYOFF_DEFAULT,
    ATTACK, FEINT, DISENGAGE, F1_COLOR, F2_COLOR,
)


def _make_match(selection_strength=1.0):
    cj = Fighter.from_matrices("CJ", F1_BASE_DEFAULT, F1_ADAPTATION_DEFAULT,
                                F1_PAYOFF_DEFAULT, color=F1_COLOR)
    cp = Fighter.from_matrices("Counter-Fighter", F2_BASE_DEFAULT, F2_ADAPTATION_DEFAULT,
                                F2_PAYOFF_DEFAULT, color=F2_COLOR)
    return SparringMatch(cj, cp, f1_tracked_state=ATTACK, f2_tracked_state=FEINT,
                          selection_strength=selection_strength)


def test_fighter_from_matrices_defaults():
    cj = Fighter.from_matrices("CJ", F1_BASE_DEFAULT, F1_ADAPTATION_DEFAULT, F1_PAYOFF_DEFAULT)
    assert cj.name == "CJ"
    assert np.array_equal(cj.base_matrix, F1_BASE_DEFAULT)
    assert cj.max_exposure == 10.0
    print("  test_fighter_from_matrices_defaults: PASS")


def test_fighter_reset_zeroes_state():
    cj = Fighter.from_matrices("CJ", F1_BASE_DEFAULT, F1_ADAPTATION_DEFAULT, F1_PAYOFF_DEFAULT)
    cj.exposure = 7.5
    cj.reset(DISENGAGE)
    assert cj.exposure == 0.0
    assert cj.current_state == DISENGAGE
    assert cj.state_history == []
    print("  test_fighter_reset_zeroes_state: PASS")


def test_compute_lambda_boundaries():
    assert compute_lambda(0.0) == 0.0
    assert abs(compute_lambda(10.0) - 1.0) < 1e-9
    print("  test_compute_lambda_boundaries: PASS")


def test_match_simulate_reproducible():
    match = _make_match()
    r1 = match.simulate(200, start_state=DISENGAGE, seed=1)
    r2 = match.simulate(200, start_state=DISENGAGE, seed=1)
    assert np.array_equal(r1['f1_seq'], r2['f1_seq'])
    assert np.array_equal(r1['f2_seq'], r2['f2_seq'])
    print("  test_match_simulate_reproducible: PASS")


def test_match_simulate_result_keys_and_shapes():
    match = _make_match()
    n_steps = 150
    result = match.simulate(n_steps, start_state=DISENGAGE, seed=5)
    expected_keys = {
        "f1_seq", "f2_seq", "f1_history", "f2_history",
        "f1_exposure_history", "f2_exposure_history",
        "f1_lambda_history", "f2_lambda_history",
        "f1_defend_prob_history", "f2_defend_prob_history",
        "f1_fitness_history", "f2_fitness_history",
        "f1_cumulative_fitness", "f2_cumulative_fitness",
    }
    assert expected_keys.issubset(set(result.keys()))
    assert len(result['f1_seq']) == n_steps
    assert len(result['f1_defend_prob_history']) == n_steps - 1
    print("  test_match_simulate_result_keys_and_shapes: PASS")


def test_match_cumulative_fitness_monotonically_increasing():
    match = _make_match()
    result = match.simulate(200, start_state=DISENGAGE, seed=3)
    diffs = np.diff(result['f1_cumulative_fitness'])
    assert np.all(diffs >= 0)  # payoffs are non-negative
    print("  test_match_cumulative_fitness_monotonically_increasing: PASS")


def test_match_repr_and_print_summary_no_errors():
    match = _make_match()
    match.simulate(50, start_state=DISENGAGE, seed=9)
    repr(match)
    match.print_result_summary()  # should not raise
    print("  test_match_repr_and_print_summary_no_errors: PASS")


def run_all():
    print("test_simulation.py")
    test_fighter_from_matrices_defaults()
    test_fighter_reset_zeroes_state()
    test_compute_lambda_boundaries()
    test_match_simulate_reproducible()
    test_match_simulate_result_keys_and_shapes()
    test_match_cumulative_fitness_monotonically_increasing()
    test_match_repr_and_print_summary_no_errors()


if __name__ == "__main__":
    run_all()
    print("ALL test_simulation.py TESTS PASSED")
