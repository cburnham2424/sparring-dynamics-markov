"""
Monte Carlo experiment wrapper around sparring_markov_two_agent.simulate().

This module adds a statistical layer on top of the existing single-run
simulator without modifying its logic: run N independent realizations,
aggregate mean/variance/CI statistics at each time step, and plot
confidence bands instead of single trajectories.
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
import pandas as pd

from sparring_markov_two_agent import (
    simulate,
    F1_BASE, F2_BASE,
    F1_ADAPTATION_MATRIX, F2_ADAPTATION_MATRIX,
    F1_PAYOFF as F1_PAYOFF_MATRIX,
    F2_PAYOFF as F2_PAYOFF_MATRIX,
    STATES as states,
    N_STATES as n,
    N_STEPS as SIM_N_STEPS,
)

# ── Dark mode theme ──────────────────────────────────────────
BG_COLOR = '#1a1a19'
GRID_COLOR = '#2c2c2a'
TICK_COLOR = '#c3c2b7'
TITLE_COLOR = '#ffffff'

plt.style.use('dark_background')
plt.rcParams.update({
    'figure.facecolor': BG_COLOR,
    'axes.facecolor': BG_COLOR,
    'savefig.facecolor': BG_COLOR,
    'axes.edgecolor': GRID_COLOR,
    'axes.labelcolor': TICK_COLOR,
    'axes.titlecolor': TITLE_COLOR,
    'xtick.color': TICK_COLOR,
    'ytick.color': TICK_COLOR,
    'text.color': TITLE_COLOR,
    'axes.grid': True,
    'grid.color': GRID_COLOR,
    'grid.alpha': 0.5,
})


def _darken_figure(fig, axes):
    fig.patch.set_facecolor(BG_COLOR)
    for ax in np.atleast_1d(axes).flat:
        ax.set_facecolor(BG_COLOR)


# MONTE CARLO CONFIGURATION
N_SIMULATIONS = 500      # number of independent runs
N_STEPS = 500             # steps per simulation — must match the simulator's
                          # own N_STEPS (simulate() does not take n_steps as
                          # an argument; it always runs the module's N_STEPS)
START_STATE = 2           # Disengage — informational only: simulate() always
                          # starts from the module's own START_STATE
                          # ("Disengage"), which happens to be this state, but
                          # this constant is not wired into the simulator call
SELECTION_STRENGTH = 1.0  # moderate EGT (matches existing reference run)
CONFIDENCE_LEVEL = 0.95   # for confidence intervals
ALPHA = 1 - CONFIDENCE_LEVEL  # 0.05

ROLLING_WINDOW = 20

F1_COLOR = "#7F77DD"
F2_COLOR = "#E8593C"

assert N_STEPS == SIM_N_STEPS, (
    f"N_STEPS ({N_STEPS}) must match sparring_markov_two_agent.N_STEPS ({SIM_N_STEPS})")


# ---------------------------------------------------------------------------
# STEP 1 — Run N independent simulations
# ---------------------------------------------------------------------------

def run_monte_carlo(n_simulations, n_steps, start_state,
                     selection_strength, random_seed=None):
    """
    Run N independent simulations using the existing simulate() function.
    Each run uses a different random seed derived from random_seed, so the
    N runs are genuinely independent realizations rather than N copies of
    the same trajectory.

    simulate() itself takes no n_steps/start_state arguments — the number
    of steps and starting state are fixed by the simulator module's own
    N_STEPS and START_STATE constants. They are accepted here for interface
    clarity and validated against the simulator, not passed through.

    Returns a results dict containing:
    {
        'f1_fitness':     np.array shape (N, n_steps) — per-step payoff F1
        'f2_fitness':     np.array shape (N, n_steps) — per-step payoff F2
        'f1_cumulative':  np.array shape (N, n_steps) — cumulative payoff F1
        'f2_cumulative':  np.array shape (N, n_steps) — cumulative payoff F2
        'f1_states':      np.array shape (N, n_steps) — state index F1
        'f2_states':      np.array shape (N, n_steps) — state index F2
        'f1_lambda':      np.array shape (N, n_steps) — adaptation weight F1
        'f2_lambda':      np.array shape (N, n_steps) — adaptation weight F2
        'n_simulations':  int
        'n_steps':        int
    }
    """
    seed_rng = np.random.default_rng(random_seed)
    run_seeds = seed_rng.integers(0, 2**31 - 1, size=n_simulations)

    f1_fitness    = np.zeros((n_simulations, n_steps))
    f2_fitness    = np.zeros((n_simulations, n_steps))
    f1_cumulative = np.zeros((n_simulations, n_steps))
    f2_cumulative = np.zeros((n_simulations, n_steps))
    f1_states     = np.zeros((n_simulations, n_steps), dtype=int)
    f2_states     = np.zeros((n_simulations, n_steps), dtype=int)
    f1_lambda     = np.zeros((n_simulations, n_steps))
    f2_lambda     = np.zeros((n_simulations, n_steps))

    for i in range(n_simulations):
        result = simulate(selection_strength, seed=int(run_seeds[i]))

        f1_fitness[i]    = result['f1_fitness_history']
        f2_fitness[i]    = result['f2_fitness_history']
        f1_cumulative[i] = result['f1_cumulative_fitness']
        f2_cumulative[i] = result['f2_cumulative_fitness']
        f1_states[i]     = result['f1_seq']
        f2_states[i]     = result['f2_seq']
        f1_lambda[i]     = result['f1_lambda_history']
        f2_lambda[i]     = result['f2_lambda_history']

        if (i + 1) % 100 == 0:
            print(f"  Completed {i+1}/{n_simulations} simulations...")

    return {
        'f1_fitness': f1_fitness,
        'f2_fitness': f2_fitness,
        'f1_cumulative': f1_cumulative,
        'f2_cumulative': f2_cumulative,
        'f1_states': f1_states,
        'f2_states': f2_states,
        'f1_lambda': f1_lambda,
        'f2_lambda': f2_lambda,
        'n_simulations': n_simulations,
        'n_steps': n_steps,
    }


# ---------------------------------------------------------------------------
# STEP 2 — Compute statistics
# ---------------------------------------------------------------------------

def compute_statistics(data_array, confidence_level=0.95):
    """
    Compute statistics across N simulations at each time step.

    data_array: shape (N, T) — N simulations, T time steps

    Returns dict with arrays of shape (T,): mean, variance, std, ci_lower,
    ci_upper, median, q25, q75.

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


def rolling_average_2d(data_array, window=ROLLING_WINDOW):
    """Apply a rolling mean of the given window to every row of a (N, T) array."""
    N, T = data_array.shape
    kernel = np.ones(window) / window
    T_out = T - window + 1
    out = np.zeros((N, T_out))
    for i in range(N):
        out[i] = np.convolve(data_array[i], kernel, mode="valid")
    return out


# ---------------------------------------------------------------------------
# STEP 3 — Compute state occupancy
# ---------------------------------------------------------------------------

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


# ---------------------------------------------------------------------------
# STEP 4 — Full analysis pipeline
# ---------------------------------------------------------------------------

def analyze_monte_carlo(results, confidence_level=0.95):
    """Run all statistical analyses on Monte Carlo results."""
    analysis = {}

    for key in ['f1_fitness', 'f2_fitness',
                'f1_cumulative', 'f2_cumulative',
                'f1_lambda', 'f2_lambda']:
        analysis[key] = compute_statistics(results[key], confidence_level)

    analysis['f1_occupancy'] = compute_state_occupancy(results['f1_states'], n)
    analysis['f2_occupancy'] = compute_state_occupancy(results['f2_states'], n)

    return analysis


# ---------------------------------------------------------------------------
# STEP 5 — Printed summary
# ---------------------------------------------------------------------------

def print_summary(results, analysis):
    N = results['n_simulations']
    T = results['n_steps']

    print(f"\n{'='*60}")
    print(f"MONTE CARLO SUMMARY — {N} simulations × {T} steps")
    print(f"{'='*60}")

    for fighter, key in [("Fighter 1 (CJ)", "f1_cumulative"),
                          ("Fighter 2 (Counter-Fighter)", "f2_cumulative")]:
        stats_dict = analysis[key]
        final_mean = stats_dict['mean'][-1]
        final_std  = stats_dict['std'][-1]
        final_var  = stats_dict['variance'][-1]
        final_ci_l = stats_dict['ci_lower'][-1]
        final_ci_u = stats_dict['ci_upper'][-1]

        print(f"\n{fighter} — Final Cumulative Fitness:")
        print(f"  Mean:              {final_mean:.4f}")
        print(f"  Variance:          {final_var:.4f}")
        print(f"  Std Dev:           {final_std:.4f}")
        print(f"  95% CI:            [{final_ci_l:.4f}, {final_ci_u:.4f}]")
        print(f"  Median:            {stats_dict['median'][-1]:.4f}")
        print(f"  IQR:               [{stats_dict['q25'][-1]:.4f}, "
              f"{stats_dict['q75'][-1]:.4f}]")

    print(f"\n{'─'*60}")
    print("FIGHTER 1 (CJ) — Average State Occupancy:")
    f1_occ = analysis['f1_occupancy']
    for i, state in enumerate(states):
        print(f"  {state:12s}: {f1_occ['mean'][i]:.4f} ± "
              f"{f1_occ['std'][i]:.4f}  "
              f"95% CI [{f1_occ['ci_lower'][i]:.4f}, "
              f"{f1_occ['ci_upper'][i]:.4f}]")

    print("\nFIGHTER 2 (Counter-Fighter) — Average State Occupancy:")
    f2_occ = analysis['f2_occupancy']
    for i, state in enumerate(states):
        print(f"  {state:12s}: {f2_occ['mean'][i]:.4f} ± "
              f"{f2_occ['std'][i]:.4f}  "
              f"95% CI [{f2_occ['ci_lower'][i]:.4f}, "
              f"{f2_occ['ci_upper'][i]:.4f}]")

    f1_wins = np.sum(
        results['f1_cumulative'][:, -1] > results['f2_cumulative'][:, -1]
    )
    f2_wins = np.sum(
        results['f2_cumulative'][:, -1] > results['f1_cumulative'][:, -1]
    )
    ties = N - f1_wins - f2_wins

    print(f"\n{'─'*60}")
    print(f"OUTCOME DISTRIBUTION across {N} simulations:")
    print(f"  Fighter 1 wins:  {f1_wins:4d} ({100*f1_wins/N:.1f}%)")
    print(f"  Fighter 2 wins:  {f2_wins:4d} ({100*f2_wins/N:.1f}%)")
    print(f"  Dead heats:      {ties:4d} ({100*ties/N:.1f}%)")

    f1_ci_l = analysis['f1_cumulative']['ci_lower'][-1]
    f1_ci_u = analysis['f1_cumulative']['ci_upper'][-1]
    f2_ci_l = analysis['f2_cumulative']['ci_lower'][-1]
    f2_ci_u = analysis['f2_cumulative']['ci_upper'][-1]

    overlap = not (f1_ci_u < f2_ci_l or f2_ci_u < f1_ci_l)
    print(f"\n  95% CI overlap: {overlap}")
    if overlap:
        print("  Interpretation: No statistically significant fitness "
              "difference between fighters at 95% confidence.")
        print("  Tumor parallel: This matchup is in evolutionary "
              "stable co-existence — neither strategy dominates.")
    else:
        winner = "Fighter 1" if f1_ci_l > f2_ci_u else "Fighter 2"
        print(f"  Interpretation: {winner} has significantly higher "
              f"fitness at 95% confidence.")
        print("  Tumor parallel: One strategy dominates — equivalent "
              "to immune clearance or tumor escape.")


# ---------------------------------------------------------------------------
# STEP 6 — Visualizations
# ---------------------------------------------------------------------------

def _plot_band(ax, x, stats_dict, color, label_prefix):
    ax.plot(x, stats_dict['mean'], color=color, linewidth=2, label=f"{label_prefix} mean")
    ax.fill_between(x, stats_dict['ci_lower'], stats_dict['ci_upper'],
                     color=color, alpha=0.15, label="95% CI")
    ax.fill_between(x, stats_dict['q25'], stats_dict['q75'],
                     color=color, alpha=0.15, label="IQR")


def plot_monte_carlo_summary(results, analysis, filename="monte_carlo_sparring.png",
                              n_traces=5):
    N, T = results['n_simulations'], results['n_steps']
    x = np.arange(T)
    fig, axes = plt.subplots(3, 2, figsize=(15, 16))
    _darken_figure(fig, axes)

    # Top row: cumulative fitness with confidence bands
    for col, (fighter, key, color) in enumerate([
        ("Fighter 1 (CJ)", "f1_cumulative", F1_COLOR),
        ("Fighter 2 (Counter-Fighter)", "f2_cumulative", F2_COLOR),
    ]):
        ax = axes[0, col]
        for i in range(min(n_traces, N)):
            ax.plot(x, results[key][i], color=color, alpha=0.05, linewidth=0.5)
        _plot_band(ax, x, analysis[key], color, fighter.split(" ")[0])
        ax.set_xlabel("Exchange step")
        ax.set_ylabel("Cumulative fitness")
        ax.set_title(f"{fighter} — Cumulative Fitness\n(N={N}, 95% CI shaded)")
        ax.legend(fontsize=8)
        ax.grid(axis="y", linestyle="--", alpha=0.3)

    # Middle row: per-step payoff, rolling 20-step mean, with CI bands
    for col, (fighter, key, color) in enumerate([
        ("Fighter 1", "f1_fitness", F1_COLOR),
        ("Fighter 2", "f2_fitness", F2_COLOR),
    ]):
        ax = axes[1, col]
        rolled = rolling_average_2d(results[key], ROLLING_WINDOW)
        rolled_stats = compute_statistics(rolled, CONFIDENCE_LEVEL)
        x_roll = np.arange(ROLLING_WINDOW - 1, T)
        ax.plot(x_roll, rolled_stats['mean'], color=color, linewidth=2, label="Mean")
        ax.fill_between(x_roll, rolled_stats['ci_lower'], rolled_stats['ci_upper'],
                         color=color, alpha=0.15, label="95% CI")
        ax.set_xlabel("Exchange step")
        ax.set_ylabel("Payoff")
        ax.set_title(f"{fighter} — Per-Step Payoff (Rolling {ROLLING_WINDOW}-step avg)")
        ax.legend(fontsize=8)
        ax.grid(axis="y", linestyle="--", alpha=0.3)

    # Bottom left: state occupancy bar chart with error bars
    ax = axes[2, 0]
    f1_occ, f2_occ = analysis['f1_occupancy'], analysis['f2_occupancy']
    x_pos = np.arange(n)
    width = 0.35
    f1_err = [f1_occ['mean'] - f1_occ['ci_lower'], f1_occ['ci_upper'] - f1_occ['mean']]
    f2_err = [f2_occ['mean'] - f2_occ['ci_lower'], f2_occ['ci_upper'] - f2_occ['mean']]
    ax.bar(x_pos - width / 2, f1_occ['mean'], width, yerr=f1_err, capsize=4,
           color=F1_COLOR, label="Fighter 1")
    ax.bar(x_pos + width / 2, f2_occ['mean'], width, yerr=f2_err, capsize=4,
           color=F2_COLOR, label="Fighter 2")
    ax.set_xticks(x_pos)
    ax.set_xticklabels(states)
    ax.set_xlabel("States")
    ax.set_ylabel("Mean fraction of time")
    ax.set_title(f"Average State Occupancy ± 95% CI\n(N={N} simulations)")
    ax.legend(fontsize=8)
    ax.grid(axis="y", linestyle="--", alpha=0.3)

    # Bottom right: lambda over time with CI bands
    ax = axes[2, 1]
    for frac in (0.25, 0.5, 0.75):
        ax.axhline(frac, color="gray", linestyle="--", alpha=0.5)
    ax.plot(x, analysis['f1_lambda']['mean'], color=F1_COLOR, linewidth=2, label="F1 mean λ")
    ax.fill_between(x, analysis['f1_lambda']['ci_lower'], analysis['f1_lambda']['ci_upper'],
                     color=F1_COLOR, alpha=0.15)
    ax.plot(x, analysis['f2_lambda']['mean'], color=F2_COLOR, linewidth=2, label="F2 mean λ")
    ax.fill_between(x, analysis['f2_lambda']['ci_lower'], analysis['f2_lambda']['ci_upper'],
                     color=F2_COLOR, alpha=0.15)
    ax.set_ylim(0, 1.05)
    ax.set_xlabel("Exchange step")
    ax.set_ylabel("λ")
    ax.set_title(f"Adaptation Weight λ Over Time\n(Mean ± 95% CI, N={N})")
    ax.legend(fontsize=8)
    ax.grid(axis="y", linestyle=":", alpha=0.2)

    fig.tight_layout()
    fig.savefig(filename, dpi=150)
    plt.close(fig)


def plot_monte_carlo_distributions(results, analysis, filename="monte_carlo_distributions.png"):
    fig, axes = plt.subplots(2, 2, figsize=(13, 11))
    _darken_figure(fig, axes)

    f1_final = results['f1_cumulative'][:, -1]
    f2_final = results['f2_cumulative'][:, -1]

    # Top left: F1 final fitness histogram
    ax = axes[0, 0]
    ax.hist(f1_final, bins=30, color=F1_COLOR, alpha=0.8, edgecolor="white")
    mean1 = analysis['f1_cumulative']['mean'][-1]
    ci_l1 = analysis['f1_cumulative']['ci_lower'][-1]
    ci_u1 = analysis['f1_cumulative']['ci_upper'][-1]
    ax.axvline(mean1, color=TITLE_COLOR, linestyle="--", linewidth=1.5, label="Mean")
    ax.axvspan(ci_l1, ci_u1, color=F1_COLOR, alpha=0.15, label="95% CI")
    ax.set_xlabel("Final cumulative fitness")
    ax.set_ylabel("Count")
    ax.set_title(f"Distribution of F1 Final Fitness\n(N={results['n_simulations']})")
    ax.legend(fontsize=8)

    # Top right: F2 final fitness histogram
    ax = axes[0, 1]
    ax.hist(f2_final, bins=30, color=F2_COLOR, alpha=0.8, edgecolor="white")
    mean2 = analysis['f2_cumulative']['mean'][-1]
    ci_l2 = analysis['f2_cumulative']['ci_lower'][-1]
    ci_u2 = analysis['f2_cumulative']['ci_upper'][-1]
    ax.axvline(mean2, color=TITLE_COLOR, linestyle="--", linewidth=1.5, label="Mean")
    ax.axvspan(ci_l2, ci_u2, color=F2_COLOR, alpha=0.15, label="95% CI")
    ax.set_xlabel("Final cumulative fitness")
    ax.set_ylabel("Count")
    ax.set_title(f"Distribution of F2 Final Fitness\n(N={results['n_simulations']})")
    ax.legend(fontsize=8)

    # Bottom left: scatter F1 vs F2 final fitness, colored by winner
    ax = axes[1, 0]
    f1_win_mask = f1_final > f2_final
    f2_win_mask = f2_final > f1_final
    tie_mask = ~f1_win_mask & ~f2_win_mask
    ax.scatter(f1_final[f1_win_mask], f2_final[f1_win_mask], color=F1_COLOR,
               alpha=0.6, s=15, label="F1 wins")
    ax.scatter(f1_final[f2_win_mask], f2_final[f2_win_mask], color=F2_COLOR,
               alpha=0.6, s=15, label="F2 wins")
    ax.scatter(f1_final[tie_mask], f2_final[tie_mask], color="gray",
               alpha=0.6, s=15, label="Tie")
    lims = [min(f1_final.min(), f2_final.min()), max(f1_final.max(), f2_final.max())]
    ax.plot(lims, lims, color=TICK_COLOR, linestyle="--", linewidth=1, alpha=0.5)
    ax.set_xlabel("F1 final fitness")
    ax.set_ylabel("F2 final fitness")
    ax.set_title(f"F1 vs F2 Final Fitness\n(N={results['n_simulations']} simulations)")
    ax.legend(fontsize=8)

    # Bottom right: distribution of final lambda values
    ax = axes[1, 1]
    f1_lambda_final = results['f1_lambda'][:, -1]
    f2_lambda_final = results['f2_lambda'][:, -1]
    ax.hist(f1_lambda_final, bins=30, color=F1_COLOR, alpha=0.6, label="Fighter 1")
    ax.hist(f2_lambda_final, bins=30, color=F2_COLOR, alpha=0.6, label="Fighter 2")
    ax.set_xlabel("Final λ")
    ax.set_ylabel("Count")
    ax.set_title("Distribution of Final Adaptation Weight λ")
    ax.legend(fontsize=8)

    fig.tight_layout()
    fig.savefig(filename, dpi=150)
    plt.close(fig)


# ---------------------------------------------------------------------------
# MAIN EXECUTION
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print(f"Running Monte Carlo experiment: "
          f"{N_SIMULATIONS} simulations × {N_STEPS} steps")
    print(f"Selection strength: {SELECTION_STRENGTH}")
    print(f"Confidence level: {CONFIDENCE_LEVEL*100:.0f}%\n")

    results = run_monte_carlo(
        n_simulations=N_SIMULATIONS,
        n_steps=N_STEPS,
        start_state=START_STATE,
        selection_strength=SELECTION_STRENGTH,
        random_seed=42,
    )

    analysis = analyze_monte_carlo(results, CONFIDENCE_LEVEL)
    print_summary(results, analysis)

    plot_monte_carlo_summary(results, analysis)
    plot_monte_carlo_distributions(results, analysis)
    print("\nSaved plot to monte_carlo_sparring.png")
    print("Saved plot to monte_carlo_distributions.png")
