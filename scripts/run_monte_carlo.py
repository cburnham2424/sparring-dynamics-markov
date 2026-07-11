"""
Thin CLI entry point: run a standalone Monte Carlo simulation using
hand-crafted default fighters (no CSV loading/estimation step — for
that, use scripts/run_pipeline.py instead).
"""
import argparse
import os
import sys

# Running this file directly (`python scripts/run_monte_carlo.py`)
# only puts scripts/ on sys.path, not the repo root, so
# sparring_dynamics wouldn't otherwise be importable.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from sparring_dynamics.simulation.fighter import Fighter
from sparring_dynamics.simulation.match import SparringMatch
from sparring_dynamics.analysis.monte_carlo import (
    run_monte_carlo, analyze_monte_carlo, print_summary
)
from sparring_dynamics.visualization.plots import (
    plot_monte_carlo_summary, plot_distributions
)
from sparring_dynamics.config import (
    DEFAULT_START_STATE, DEFAULT_N_SIMULATIONS, DEFAULT_N_STEPS,
    DEFAULT_SELECTION, DEFAULT_RANDOM_SEED,
    F1_BASE_DEFAULT, F2_BASE_DEFAULT,
    F1_ADAPTATION_DEFAULT, F2_ADAPTATION_DEFAULT,
    F1_PAYOFF_DEFAULT, F2_PAYOFF_DEFAULT,
    F1_COLOR, F2_COLOR,
)


def main(argv=None):
    parser = argparse.ArgumentParser(
        description="Run a standalone Monte Carlo simulation with "
                     "hand-crafted default fighters."
    )
    parser.add_argument('--n-sims', type=int, default=DEFAULT_N_SIMULATIONS)
    parser.add_argument('--n-steps', type=int, default=DEFAULT_N_STEPS)
    parser.add_argument('--selection', type=float, default=DEFAULT_SELECTION)
    parser.add_argument('--seed', type=int, default=DEFAULT_RANDOM_SEED)
    args = parser.parse_args(argv)

    f1 = Fighter.from_matrices("CJ", F1_BASE_DEFAULT, F1_ADAPTATION_DEFAULT,
                               F1_PAYOFF_DEFAULT, color=F1_COLOR)
    f2 = Fighter.from_matrices("Counter-Fighter", F2_BASE_DEFAULT, F2_ADAPTATION_DEFAULT,
                               F2_PAYOFF_DEFAULT, color=F2_COLOR)
    match = SparringMatch(f1, f2, selection_strength=args.selection)

    results = run_monte_carlo(match, n_simulations=args.n_sims, n_steps=args.n_steps,
                               start_state=DEFAULT_START_STATE, random_seed=args.seed)
    analysis = analyze_monte_carlo(results)
    print_summary(results, analysis)
    plot_monte_carlo_summary(results, analysis)
    plot_distributions(results, analysis)


if __name__ == "__main__":
    main()
