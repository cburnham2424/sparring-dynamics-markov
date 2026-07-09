"""
Per-time-step and per-state statistics across Monte Carlo simulation runs:
mean, variance, std, t-distribution confidence intervals, quantiles,
and state occupancy.
"""
import numpy as np
from scipy import stats


def compute_statistics(data_array, confidence_level=0.95):
    """
    Compute statistics across N simulations at each time step.

    data_array: shape (N, T) — N simulations, T time steps

    Returns dict with arrays of shape (T,): mean, variance, std,
    ci_lower, ci_upper, median, q25, q75.

    Uses a t-distribution CI (not normal) since N may not be huge:
    se = std / sqrt(N); t_crit = scipy.stats.t.ppf((1+confidence_level)/2, df=N-1)
    """
    N, T = data_array.shape
    mean     = data_array.mean(axis=0)
    variance = data_array.var(axis=0, ddof=1)
    std      = data_array.std(axis=0, ddof=1)
    median   = np.median(data_array, axis=0)
    q25      = np.percentile(data_array, 25, axis=0)
    q75      = np.percentile(data_array, 75, axis=0)

    se = std / np.sqrt(N)
    t_crit = stats.t.ppf((1 + confidence_level) / 2, df=N - 1)
    ci_lower = mean - t_crit * se
    ci_upper = mean + t_crit * se

    return {
        'mean': mean, 'variance': variance, 'std': std,
        'ci_lower': ci_lower, 'ci_upper': ci_upper,
        'median': median, 'q25': q25, 'q75': q75,
    }


def compute_state_occupancy(state_array, n_states):
    """
    Compute average fraction of time spent in each state across all
    simulations.

    state_array: shape (N, T) — integer state indices

    Returns dict with:
    occupancy: shape (N, n_states) — fraction of time in each state per sim
    mean, std, ci_lower, ci_upper: shape (n_states,)
    """
    N, T = state_array.shape
    occupancy = np.zeros((N, n_states))

    for i in range(N):
        for s in range(n_states):
            occupancy[i, s] = np.sum(state_array[i] == s) / T

    mean_occ = occupancy.mean(axis=0)
    std_occ  = occupancy.std(axis=0, ddof=1)
    se       = std_occ / np.sqrt(N)
    t_crit   = stats.t.ppf(0.975, df=N - 1)

    return {
        'occupancy': occupancy,
        'mean':      mean_occ,
        'std':       std_occ,
        'ci_lower':  mean_occ - t_crit * se,
        'ci_upper':  mean_occ + t_crit * se,
    }
