"""
Transition matrix estimation from observed state sequences.
Uses maximum likelihood estimation with Laplace smoothing.
"""
import numpy as np

from sparring_dynamics.config import (
    STATE_INDEX, STATES, N_STATES,
    DEFAULT_TRANSITION_ALPHA, DEFAULT_MIN_OBS,
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


def create_hybrid_transition_matrix(learned_matrix, counts, default_matrix,
                                     min_obs=DEFAULT_MIN_OBS):
    """
    Row-level hybrid between a learned transition matrix and a
    hand-crafted default: for each FROM-state (row), use the learned
    row if that state was left at least min_obs times in the observed
    data, otherwise fall back to the default row.

    Rows, not individual cells, are the right unit here. A transition
    matrix row is one probability distribution — its cells are jointly
    normalized, not independently estimated — so the only meaningful
    "enough data" question is "how many times did we ever leave this
    state," i.e. counts[row].sum(). Splicing individual cells within a
    row from two different sources would still happen to sum to 1 (each
    source row already does), but the result would silently blend two
    unrelated processes cell-by-cell inside what is supposed to be a
    single coherent conditional distribution — not statistically
    meaningful. Substituting whole rows keeps every row a genuine,
    single-source probability distribution.

    Parameters:
    learned_matrix: row-stochastic matrix from estimate_transition_matrix
    counts:         raw (pre-smoothing) count matrix from the same call
                    (None if no sequences were observed at all)
    default_matrix: hand-crafted fallback matrix
    min_obs:        minimum number of observed departures from a state
                    required to trust its learned row

    Returns (hybrid_matrix, row_sources) where row_sources is a list of
    'learned'/'default' per row, indicating which source was used.
    """
    hybrid = default_matrix.copy()
    row_sources = ['default'] * N_STATES

    if counts is not None:
        row_totals = counts.sum(axis=1)
        for i in range(N_STATES):
            if row_totals[i] >= min_obs:
                hybrid[i] = learned_matrix[i]
                row_sources[i] = 'learned'

    validate_stochastic_matrix(hybrid, "Hybrid transition matrix")
    return hybrid, row_sources
