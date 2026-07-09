"""
End-to-end pipeline:

    annotation.csv
         ↓
    Load & validate exchanges (+ sequences, if a separate sequence CSV exists)
         ↓
    Estimate transition matrices (or use defaults)
         ↓
    Estimate payoff matrices (or use defaults)
         ↓
    Initialize Fighter objects
         ↓
    Run Monte Carlo experiment
         ↓
    Compute statistics
         ↓
    Produce and save all plots
         ↓
    Print summary

Usage:
    python pipeline.py                          # uses annotation.csv
    python pipeline.py --csv my_data.csv        # custom CSV path
    python pipeline.py --template               # generate blank template
    python pipeline.py --no-estimate             # skip estimation, use defaults
    python pipeline.py --n-sims 100              # faster run for testing
    python pipeline.py --seed 123                # set random seed

Note on data formats: --csv points at an exchange-format annotation file
(f1_state, f2_state, winner, f1_points, f2_points), which feeds payoff
estimation. Transition-matrix estimation needs a *separate*
sequence-format file (fighter, sequence — see
sparring_dynamics.data.loader.load_sequence_csv); this pipeline doesn't
derive one from the exchange CSV automatically, so F1/F2 base
transition matrices report "default" unless such a file is wired in.
"""
import argparse
import os
import time

from sparring_dynamics.config import (
    ATTACK, FEINT,
    DEFAULT_N_STEPS, DEFAULT_START_STATE, DEFAULT_SELECTION,
    DEFAULT_N_SIMULATIONS, DEFAULT_CONFIDENCE, DEFAULT_RANDOM_SEED,
    DEFAULT_TRANSITION_ALPHA, DEFAULT_PAYOFF_ALPHA,
    DEFAULT_MIN_OBS, DEFAULT_MAX_POINTS,
    F1_ADAPTATION_DEFAULT, F2_ADAPTATION_DEFAULT,
    F1_COLOR, F2_COLOR, OUTPUT_DIR
)
from sparring_dynamics.data.loader import (
    load_exchange_csv, load_sequence_csv, create_annotation_template
)
from sparring_dynamics.data.validator import validate_exchanges
from sparring_dynamics.estimation.transitions import (
    estimate_both_transition_matrices
)
from sparring_dynamics.estimation.payoffs import estimate_payoff_matrices
from sparring_dynamics.simulation.fighter import Fighter
from sparring_dynamics.simulation.match import SparringMatch
from sparring_dynamics.analysis.monte_carlo import (
    run_monte_carlo, analyze_monte_carlo, print_summary
)
from sparring_dynamics.visualization.plots import (
    plot_monte_carlo_summary, plot_distributions
)


def build_pipeline_report(args, transition_result, payoff_result,
                           analysis, results):
    """
    Print a structured pipeline report showing:
    - Which data sources were used
    - Whether matrices were estimated or defaulted
    - Key simulation parameters
    - Final statistical findings
    """
    print("\n" + "="*65)
    print("SPARRING DYNAMICS PIPELINE REPORT")
    print("="*65)

    print(f"\n── Data ──────────────────────────────────────────────────")
    print(f"  CSV:               {args.csv}")
    print(f"  Estimation:        {'Enabled' if not args.no_estimate else 'Disabled (defaults used)'}")

    if transition_result:
        print(f"  F1 transitions:    "
              f"{'Estimated from data' if transition_result['f1_estimated'] else 'Default (no sequence data)'}")
        print(f"  F2 transitions:    "
              f"{'Estimated from data' if transition_result['f2_estimated'] else 'Default (no sequence data)'}")

    if payoff_result:
        print(f"  Payoff matrices:   "
              f"{'Estimated from data' if payoff_result['estimated'] else 'Default (no exchange data)'}")

    print(f"\n── Simulation ────────────────────────────────────────────")
    print(f"  N simulations:     {results['n_simulations']}")
    print(f"  Steps per sim:     {results['n_steps']}")
    print(f"  Selection strength:{args.selection}")
    print(f"  Random seed:       {args.seed}")

    print(f"\n── Key Findings ──────────────────────────────────────────")
    f1_mean = analysis['f1_cumulative']['mean'][-1]
    f2_mean = analysis['f2_cumulative']['mean'][-1]
    f1_ci_l = analysis['f1_cumulative']['ci_lower'][-1]
    f1_ci_u = analysis['f1_cumulative']['ci_upper'][-1]
    f2_ci_l = analysis['f2_cumulative']['ci_lower'][-1]
    f2_ci_u = analysis['f2_cumulative']['ci_upper'][-1]

    print(f"  F1 mean fitness:   {f1_mean:.4f}  95% CI [{f1_ci_l:.4f}, {f1_ci_u:.4f}]")
    print(f"  F2 mean fitness:   {f2_mean:.4f}  95% CI [{f2_ci_l:.4f}, {f2_ci_u:.4f}]")

    ci_overlap = not (f1_ci_u < f2_ci_l or f2_ci_u < f1_ci_l)
    print(f"  95% CI overlap:    {ci_overlap}")

    if ci_overlap:
        print(f"  Verdict:           Evolutionary stable co-existence")
        print(f"  Tumor parallel:    Tumor-immune dynamic equilibrium")
    else:
        winner = "Fighter 1 (CJ)" if f1_ci_l > f2_ci_u else "Fighter 2"
        print(f"  Verdict:           {winner} dominates")
        print(f"  Tumor parallel:    One strategy drives the other to extinction")

    print(f"\n── Outputs ───────────────────────────────────────────────")
    print(f"  Directory: {OUTPUT_DIR}/")
    print(f"  Files:     monte_carlo_summary.png")
    print(f"             monte_carlo_distributions.png")
    print("="*65)


def run_pipeline(args):
    """
    Execute the full pipeline for a parsed argparse.Namespace (or any
    object with the same attributes: csv, no_estimate, n_sims, n_steps,
    selection, seed). Returns a dict with every intermediate result
    (transition_result, payoff_result, match, results, analysis) so
    callers — including tests — can inspect pipeline state directly
    instead of scraping stdout.
    """
    start_time = time.time()
    print("SPARRING DYNAMICS PIPELINE")
    print(f"{'─'*40}")

    # ── 1. Load data ──────────────────────────────────────────
    if not args.no_estimate and os.path.exists(args.csv):
        print(f"\n[1/5] Loading annotations from {args.csv}...")
        try:
            exchanges = load_exchange_csv(args.csv)
            print(f"      Loaded {len(exchanges)} exchanges.")
            validate_exchanges(exchanges)

            # Sequence data (for transition estimation) lives in a
            # separate file/schema — see module docstring. Attempting
            # to read it from the exchange CSV will fail validation and
            # fall back to empty, which is expected, not an error.
            try:
                f1_seqs, f2_seqs = load_sequence_csv(args.csv)
            except Exception:
                f1_seqs, f2_seqs = [], []
        except Exception as e:
            print(f"      WARNING: Could not load CSV: {e}")
            print(f"      Falling back to hand-crafted defaults.")
            exchanges  = []
            f1_seqs    = []
            f2_seqs    = []
    else:
        if args.no_estimate:
            print(f"\n[1/5] Skipping estimation (--no-estimate flag).")
        else:
            print(f"\n[1/5] {args.csv} not found — using defaults.")
        exchanges  = []
        f1_seqs    = []
        f2_seqs    = []

    # ── 2. Estimate transition matrices ───────────────────────
    print(f"\n[2/5] Estimating transition matrices...")
    transition_result = estimate_both_transition_matrices(
        f1_seqs, f2_seqs, alpha=DEFAULT_TRANSITION_ALPHA
    )
    f1_base = transition_result['f1_matrix']
    f2_base = transition_result['f2_matrix']

    status_f1 = "estimated" if transition_result['f1_estimated'] else "default"
    status_f2 = "estimated" if transition_result['f2_estimated'] else "default"
    print(f"      F1 base matrix: {status_f1}")
    print(f"      F2 base matrix: {status_f2}")

    # ── 3. Estimate payoff matrices ───────────────────────────
    print(f"\n[3/5] Estimating payoff matrices...")
    payoff_result = estimate_payoff_matrices(
        exchanges,
        alpha=DEFAULT_PAYOFF_ALPHA,
        min_obs=DEFAULT_MIN_OBS,
        max_points=DEFAULT_MAX_POINTS
    )
    f1_payoff = payoff_result['f1_matrix']
    f2_payoff = payoff_result['f2_matrix']
    status_p  = "estimated" if payoff_result['estimated'] else "default"
    print(f"      Payoff matrices: {status_p}")

    # ── 4. Initialize fighters ────────────────────────────────
    print(f"\n[4/5] Initializing fighters...")
    cj = Fighter.from_matrices(
        name              = "CJ",
        base_matrix       = f1_base,
        adaptation_matrix = F1_ADAPTATION_DEFAULT,
        payoff_matrix     = f1_payoff,
        color             = F1_COLOR
    )
    cp = Fighter.from_matrices(
        name              = "Counter-Puncher",
        base_matrix       = f2_base,
        adaptation_matrix = F2_ADAPTATION_DEFAULT,
        payoff_matrix     = f2_payoff,
        color             = F2_COLOR
    )
    match = SparringMatch(
        fighter1            = cj,
        fighter2            = cp,
        f1_tracked_state    = ATTACK,
        f2_tracked_state    = FEINT,
        selection_strength  = args.selection
    )
    print(f"      {match}")

    # ── 5. Run Monte Carlo ────────────────────────────────────
    print(f"\n[5/5] Running Monte Carlo "
          f"({args.n_sims} simulations × {args.n_steps} steps)...")

    results  = run_monte_carlo(
        match          = match,
        n_simulations  = args.n_sims,
        n_steps        = args.n_steps,
        start_state    = DEFAULT_START_STATE,
        random_seed    = args.seed
    )
    analysis = analyze_monte_carlo(results, DEFAULT_CONFIDENCE)

    # ── Print summary ─────────────────────────────────────────
    print_summary(results, analysis)

    # ── Produce plots ─────────────────────────────────────────
    print(f"\nGenerating plots...")
    summary_path = plot_monte_carlo_summary(results, analysis)
    distributions_path = plot_distributions(results, analysis)

    elapsed = time.time() - start_time

    return {
        'transition_result': transition_result,
        'payoff_result': payoff_result,
        'match': match,
        'results': results,
        'analysis': analysis,
        'summary_plot_path': summary_path,
        'distributions_plot_path': distributions_path,
        'elapsed_seconds': elapsed,
    }


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description='Sparring Dynamics Pipeline'
    )
    parser.add_argument('--csv', default='annotation.csv',
                         help='Path to annotation CSV')
    parser.add_argument('--template', action='store_true',
                         help='Generate blank annotation template and exit')
    parser.add_argument('--no-estimate', action='store_true',
                         help='Skip estimation, use hand-crafted defaults')
    parser.add_argument('--n-sims', type=int,
                         default=DEFAULT_N_SIMULATIONS)
    parser.add_argument('--n-steps', type=int,
                         default=DEFAULT_N_STEPS)
    parser.add_argument('--selection', type=float,
                         default=DEFAULT_SELECTION)
    parser.add_argument('--seed', type=int,
                         default=DEFAULT_RANDOM_SEED)
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)

    # Generate template and exit
    if args.template:
        create_annotation_template('annotation.csv')
        print("Edit annotation.csv with your match data, "
              "then run: python pipeline.py")
        return

    pipeline_result = run_pipeline(args)

    build_pipeline_report(
        args,
        pipeline_result['transition_result'],
        pipeline_result['payoff_result'],
        pipeline_result['analysis'],
        pipeline_result['results'],
    )
    print(f"\nTotal runtime: {pipeline_result['elapsed_seconds']:.1f}s")


if __name__ == "__main__":
    main()
