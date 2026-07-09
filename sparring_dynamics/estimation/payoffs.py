"""
Payoff matrix estimation from annotated exchange outcomes.
Uses points-weighted rewards with Laplace smoothing and
confidence weighting for sparse state pairs.
"""
import numpy as np

from sparring_dynamics.config import (
    STATE_INDEX, N_STATES,
    DEFAULT_PAYOFF_ALPHA, DEFAULT_MIN_OBS, DEFAULT_MAX_POINTS,
    F1_PAYOFF_DEFAULT, F2_PAYOFF_DEFAULT
)
from sparring_dynamics.data.validator import validate_payoff_matrix


def accumulate_rewards(exchanges, max_points=DEFAULT_MAX_POINTS):
    """
    Accumulate points-weighted rewards from exchange list.
    Returns f1_rewards, f2_rewards, totals — all (N_STATES, N_STATES).
    """
    f1_rewards = np.zeros((N_STATES, N_STATES))
    f2_rewards = np.zeros((N_STATES, N_STATES))
    totals     = np.zeros((N_STATES, N_STATES))

    for ex in exchanges:
        i = STATE_INDEX[ex['f1_state']]
        j = STATE_INDEX[ex['f2_state']]
        totals[i, j] += 1

        winner   = ex['winner']
        f1_pts   = ex['f1_points']
        f2_pts   = ex['f2_points']

        if winner == 'F1':
            f1_rewards[i, j] += f1_pts / max_points
        elif winner == 'F2':
            f2_rewards[i, j] += f2_pts / max_points
        elif winner == 'Double':
            f1_rewards[i, j] += (f1_pts / max_points) * 0.5
            f2_rewards[i, j] += (f2_pts / max_points) * 0.5
        # None: no reward

    return f1_rewards, f2_rewards, totals


def smooth_and_normalize(rewards, totals,
                          alpha=DEFAULT_PAYOFF_ALPHA,
                          min_obs=DEFAULT_MIN_OBS):
    """
    Apply Laplace smoothing, confidence weighting, and min-max normalization.
    Returns final payoff matrix in [0, 1].
    """
    # Laplace smoothing toward 0.5 prior
    smoothed    = (rewards + alpha * 0.5) / (totals + alpha)

    # Confidence weighting toward 0.5 for sparse pairs
    confidence  = np.minimum(totals / min_obs, 1.0)
    weighted    = confidence * smoothed + (1 - confidence) * 0.5

    # Min-max normalize
    mn, mx = weighted.min(), weighted.max()
    if mx - mn < 1e-8:
        return np.full_like(weighted, 0.5)
    return (weighted - mn) / (mx - mn)


def estimate_payoff_matrices(exchanges,
                              alpha=DEFAULT_PAYOFF_ALPHA,
                              min_obs=DEFAULT_MIN_OBS,
                              max_points=DEFAULT_MAX_POINTS):
    """
    Full payoff estimation pipeline from exchange list.
    Falls back to defaults if exchanges is empty or None.

    Returns dict with matrices and diagnostics.
    """
    if not exchanges:
        return {
            'f1_matrix':    F1_PAYOFF_DEFAULT.copy(),
            'f2_matrix':    F2_PAYOFF_DEFAULT.copy(),
            'totals':       None,
            'estimated':    False
        }

    f1_raw, f2_raw, totals = accumulate_rewards(exchanges, max_points)

    f1_matrix = smooth_and_normalize(f1_raw, totals, alpha, min_obs)
    f2_matrix = smooth_and_normalize(f2_raw, totals, alpha, min_obs)

    validate_payoff_matrix(f1_matrix, "F1 payoff matrix")
    validate_payoff_matrix(f2_matrix, "F2 payoff matrix")

    return {
        'f1_matrix':    f1_matrix,
        'f2_matrix':    f2_matrix,
        'totals':       totals,
        'estimated':    True,
        'alpha':        alpha,
        'min_obs':      min_obs
    }
