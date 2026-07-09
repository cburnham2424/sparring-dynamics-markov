"""Tests for sparring_dynamics.estimation (transitions + payoffs)."""
import os
import tempfile

import numpy as np

from sparring_dynamics.data.loader import generate_placeholder_csv, load_exchange_csv
from sparring_dynamics.estimation.transitions import (
    count_transitions, estimate_transition_matrix, estimate_both_transition_matrices,
)
from sparring_dynamics.estimation.payoffs import (
    accumulate_rewards, smooth_and_normalize, estimate_payoff_matrices,
)
from sparring_dynamics.config import (
    F1_BASE_DEFAULT, F2_BASE_DEFAULT, F1_PAYOFF_DEFAULT, F2_PAYOFF_DEFAULT, N_STATES,
)


def test_count_transitions():
    sequences = [["Attack", "Feint", "Attack", "Disengage"]]
    counts = count_transitions(sequences)
    assert counts.shape == (N_STATES, N_STATES)
    assert counts.sum() == 3  # 4 states -> 3 transitions
    print("  test_count_transitions: PASS")


def test_estimate_transition_matrix_empty_uses_correct_default():
    f1_matrix, counts, estimated = estimate_transition_matrix([], default_matrix=F1_BASE_DEFAULT)
    assert estimated is False
    assert np.array_equal(f1_matrix, F1_BASE_DEFAULT)

    f2_matrix, counts, estimated = estimate_transition_matrix([], default_matrix=F2_BASE_DEFAULT)
    assert estimated is False
    assert np.array_equal(f2_matrix, F2_BASE_DEFAULT)
    assert not np.array_equal(f2_matrix, F1_BASE_DEFAULT)
    print("  test_estimate_transition_matrix_empty_uses_correct_default: PASS")


def test_estimate_transition_matrix_from_data():
    sequences = [["Attack", "Feint"] * 20]
    matrix, counts, estimated = estimate_transition_matrix(sequences, alpha=1.0)
    assert estimated is True
    row_sums = matrix.sum(axis=1)
    assert np.allclose(row_sums, 1.0)
    print("  test_estimate_transition_matrix_from_data: PASS")


def test_estimate_both_transition_matrices_mixed():
    f1_sequences = [["Attack", "Feint", "Attack", "Feint", "Attack"]]
    result = estimate_both_transition_matrices(f1_sequences, [])
    assert result['f1_estimated'] is True
    assert result['f2_estimated'] is False
    assert np.array_equal(result['f2_matrix'], F2_BASE_DEFAULT)
    print("  test_estimate_both_transition_matrices_mixed: PASS")


def test_accumulate_rewards_and_smooth():
    with tempfile.TemporaryDirectory() as tmp:
        path = os.path.join(tmp, "placeholder.csv")
        generate_placeholder_csv(path, n_exchanges=60)
        exchanges = load_exchange_csv(path)

    f1_rewards, f2_rewards, totals = accumulate_rewards(exchanges)
    assert totals.sum() == 60
    assert f1_rewards.shape == (N_STATES, N_STATES)

    f1_matrix = smooth_and_normalize(f1_rewards, totals)
    assert f1_matrix.min() >= 0 - 1e-9
    assert f1_matrix.max() <= 1 + 1e-9
    print("  test_accumulate_rewards_and_smooth: PASS")


def test_estimate_payoff_matrices_empty_uses_defaults():
    result = estimate_payoff_matrices([])
    assert result['estimated'] is False
    assert np.array_equal(result['f1_matrix'], F1_PAYOFF_DEFAULT)
    assert np.array_equal(result['f2_matrix'], F2_PAYOFF_DEFAULT)
    print("  test_estimate_payoff_matrices_empty_uses_defaults: PASS")


def test_estimate_payoff_matrices_from_placeholder_data():
    with tempfile.TemporaryDirectory() as tmp:
        path = os.path.join(tmp, "placeholder.csv")
        generate_placeholder_csv(path, n_exchanges=60)
        exchanges = load_exchange_csv(path)

    result = estimate_payoff_matrices(exchanges)
    assert result['estimated'] is True
    assert result['f1_matrix'].shape == (N_STATES, N_STATES)
    assert np.all((result['f1_matrix'] >= 0) & (result['f1_matrix'] <= 1))
    assert np.all((result['f2_matrix'] >= 0) & (result['f2_matrix'] <= 1))
    print("  test_estimate_payoff_matrices_from_placeholder_data: PASS")


def run_all():
    print("test_estimation.py")
    test_count_transitions()
    test_estimate_transition_matrix_empty_uses_correct_default()
    test_estimate_transition_matrix_from_data()
    test_estimate_both_transition_matrices_mixed()
    test_accumulate_rewards_and_smooth()
    test_estimate_payoff_matrices_empty_uses_defaults()
    test_estimate_payoff_matrices_from_placeholder_data()


if __name__ == "__main__":
    run_all()
    print("ALL test_estimation.py TESTS PASSED")
