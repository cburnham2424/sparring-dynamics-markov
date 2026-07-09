"""
Validation utilities for matrices and data structures.
Used by estimation and simulation modules to catch
errors early with clear messages.
"""
import numpy as np

from sparring_dynamics.config import STATES, STATE_INDEX, N_STATES


def validate_stochastic_matrix(matrix, name, tol=1e-8):
    """
    Verify matrix is a valid row-stochastic matrix:
    - Shape is (N_STATES, N_STATES)
    - All values >= 0
    - All rows sum to 1.0 within tolerance

    Returns True if valid. Raises ValueError with clear message if not.
    """
    if matrix.shape != (N_STATES, N_STATES):
        raise ValueError(
            f"{name}: Expected shape ({N_STATES},{N_STATES}), "
            f"got {matrix.shape}"
        )

    if np.any(matrix < -tol):
        neg_idx = np.where(matrix < -tol)
        raise ValueError(
            f"{name}: Negative probabilities at indices {neg_idx}: "
            f"{matrix[neg_idx]}"
        )

    row_sums = matrix.sum(axis=1)
    bad_rows = np.where(np.abs(row_sums - 1.0) > tol)[0]
    if len(bad_rows) > 0:
        for r in bad_rows:
            raise ValueError(
                f"{name}: Row {r} ({STATES[r]}) sums to "
                f"{row_sums[r]:.8f}, expected 1.0"
            )

    return True


def validate_payoff_matrix(matrix, name, tol=1e-8):
    """
    Verify matrix is a valid payoff matrix:
    - Shape is (N_STATES, N_STATES)
    - All values in [0, 1]
    - No NaN or Inf
    """
    if matrix.shape != (N_STATES, N_STATES):
        raise ValueError(
            f"{name}: Expected shape ({N_STATES},{N_STATES}), "
            f"got {matrix.shape}"
        )

    if np.any(np.isnan(matrix)):
        raise ValueError(f"{name}: Contains NaN values")

    if np.any(np.isinf(matrix)):
        raise ValueError(f"{name}: Contains Inf values")

    if np.any(matrix < -tol) or np.any(matrix > 1.0 + tol):
        raise ValueError(
            f"{name}: Values must be in [0,1]. "
            f"Min={matrix.min():.4f}, Max={matrix.max():.4f}"
        )

    return True


def validate_exchanges(exchanges, min_exchanges=10):
    """
    Validate a list of exchange dicts has sufficient coverage.
    Warns (does not raise) for sparse state pairs.
    Returns coverage report dict.
    """
    coverage = np.zeros((N_STATES, N_STATES), dtype=int)

    for ex in exchanges:
        i = STATE_INDEX[ex['f1_state']]
        j = STATE_INDEX[ex['f2_state']]
        coverage[i, j] += 1

    total = len(exchanges)
    sparse_pairs = []

    for i, s1 in enumerate(STATES):
        for j, s2 in enumerate(STATES):
            if coverage[i, j] < 3:
                sparse_pairs.append((s1, s2, coverage[i, j]))

    if total < min_exchanges:
        print(f"WARNING: Only {total} exchanges loaded. "
              f"Recommend at least {min_exchanges} for reliable estimation.")

    if sparse_pairs:
        print(f"WARNING: {len(sparse_pairs)} state pairs have < 3 observations "
              f"— confidence weighting will pull these toward 0.5:")
        for s1, s2, count in sparse_pairs[:5]:
            print(f"  {s1} vs {s2}: {count} observations")
        if len(sparse_pairs) > 5:
            print(f"  ... and {len(sparse_pairs)-5} more")

    return {
        'total': total,
        'coverage': coverage,
        'sparse_pairs': sparse_pairs,
        'min_count': coverage.min(),
        'max_count': coverage.max()
    }
