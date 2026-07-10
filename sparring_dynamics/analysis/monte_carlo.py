"""
Monte Carlo experiment orchestration: run N independent realizations
of a SparringMatch and aggregate statistics across them.
"""
import numpy as np

from sparring_dynamics.config import STATES, N_STATES
from sparring_dynamics.simulation.match import SparringMatch
from sparring_dynamics.analysis.statistics import compute_statistics, compute_state_occupancy


def run_monte_carlo(match, n_simulations, n_steps, start_state, random_seed=None):
    """
    Run n_simulations independent realizations of the given SparringMatch.

    match is reused across runs — SparringMatch.simulate() fully resets
    both fighters at the start of each call, so this is safe and avoids
    re-constructing Fighter objects per run. Each run draws a distinct
    seed from a NumPy Generator seeded by random_seed, so the runs are
    genuinely independent rather than N copies of the same trajectory.

    Returns a results dict:
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
    assert isinstance(match, SparringMatch), "run_monte_carlo requires a SparringMatch instance"

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
        result = match.simulate(n_steps, start_state, seed=int(run_seeds[i]))

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


def analyze_monte_carlo(results, confidence_level=0.95):
    """Run all statistical analyses on Monte Carlo results."""
    analysis = {}

    for key in ['f1_fitness', 'f2_fitness',
                'f1_cumulative', 'f2_cumulative',
                'f1_lambda', 'f2_lambda']:
        analysis[key] = compute_statistics(results[key], confidence_level)

    analysis['f1_occupancy'] = compute_state_occupancy(results['f1_states'], N_STATES)
    analysis['f2_occupancy'] = compute_state_occupancy(results['f2_states'], N_STATES)

    return analysis


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
    for i, state in enumerate(STATES):
        print(f"  {state:12s}: {f1_occ['mean'][i]:.4f} ± "
              f"{f1_occ['std'][i]:.4f}  "
              f"95% CI [{f1_occ['ci_lower'][i]:.4f}, "
              f"{f1_occ['ci_upper'][i]:.4f}]")

    print("\nFIGHTER 2 (Counter-Fighter) — Average State Occupancy:")
    f2_occ = analysis['f2_occupancy']
    for i, state in enumerate(STATES):
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
