"""Thin CLI entry point: run the full parameter sensitivity analysis suite."""
import argparse
import os
import sys

# Running this file directly (`python scripts/run_sensitivity.py`)
# only puts scripts/ on sys.path, not the repo root, so
# sparring_dynamics wouldn't otherwise be importable.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from sparring_dynamics.analysis.sensitivity import (
    run_full_sensitivity_analysis, print_sensitivity_summary
)


def main(argv=None):
    parser = argparse.ArgumentParser(
        description="Run the full sensitivity analysis (1D + 2D sweeps, "
                     "plots, and summary tables)."
    )
    parser.add_argument('--n-sims-1d', type=int, default=100)
    parser.add_argument('--n-sims-2d', type=int, default=50)
    args = parser.parse_args(argv)

    all_results = run_full_sensitivity_analysis(
        n_simulations_1d=args.n_sims_1d,
        n_simulations_2d=args.n_sims_2d,
    )
    print_sensitivity_summary(all_results)


if __name__ == "__main__":
    main()
