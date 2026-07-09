"""
Transition matrix estimation from observed state sequences.
Uses maximum likelihood estimation with Laplace smoothing.
"""
import numpy as np

from sparring_dynamics.config import (
    STATE_INDEX, N_STATES,
    DEFAULT_TRANSITION_ALPHA,
    F1_BASE_DEFAULT, F2_BASE_DEFAULT
)
from sparring_dynamics.data.validator import validate_stochastic_matrix


def count_transitions(sequences):
    """
    Count observed transitions across all sequences.
    Returns (N_STATES, N_STATES) count matrix.
    """
    counts = np.zeros((N_STATES, N_STATES), dtype=float)
    for seq in sequences:
        for t in range(len(seq) - 1):
            i = STATE_INDEX[seq[t]]
            j = STATE_INDEX[seq[t + 1]]
            counts[i, j] += 1
    return counts


def estimate_transition_matrix(sequences, alpha=DEFAULT_TRANSITION_ALPHA, default_matrix=None):
    """
    Estimate transition matrix from sequences with Laplace smoothing.

    If sequences is empty or None, returns default_matrix (falling back
    to F1_BASE_DEFAULT if none was given — pass the correct per-fighter
    default explicitly when calling this for Fighter 2, since a caller
    that forgets to would otherwise silently get Fighter 1's matrix).
    Otherwise estimates from data.

    Returns (matrix, counts, was_estimated) tuple:
    - matrix: (N_STATES, N_STATES) stochastic matrix
    - counts: raw transition counts before smoothing
    - was_estimated: True if estimated from data, False if default used
    """
    if not sequences:
        fallback = default_matrix if default_matrix is not None else F1_BASE_DEFAULT
        return fallback.copy(), None, False

    counts   = count_transitions(sequences)
    smoothed = counts + alpha
    row_sums = smoothed.sum(axis=1, keepdims=True)
    row_sums = np.where(row_sums < 1e-10, 1.0, row_sums)
    matrix   = smoothed / row_sums

    validate_stochastic_matrix(matrix, "Estimated transition matrix")
    return matrix, counts, True


def estimate_both_transition_matrices(f1_sequences, f2_sequences,
                                      alpha=DEFAULT_TRANSITION_ALPHA):
    """
    Estimate F1 and F2 transition matrices.
    Falls back to the correct per-fighter default for whichever has no data.

    Returns dict with full diagnostics.
    """
    f1_matrix, f1_counts, f1_estimated = estimate_transition_matrix(
        f1_sequences, alpha, default_matrix=F1_BASE_DEFAULT
    )
    f2_matrix, f2_counts, f2_estimated = estimate_transition_matrix(
        f2_sequences, alpha, default_matrix=F2_BASE_DEFAULT
    )

    return {
        'f1_matrix':    f1_matrix,
        'f2_matrix':    f2_matrix,
        'f1_counts':    f1_counts,
        'f2_counts':    f2_counts,
        'f1_estimated': f1_estimated,
        'f2_estimated': f2_estimated,
        'alpha':        alpha
    }
