"""
Automated sensitivity analysis: sweep model parameters (EGT selection
strength, memory decay/growth, adaptation sigmoid steepness) across
ranges, run Monte Carlo at each point, and characterize how strongly
each parameter drives fitness outcomes, win rates, and the
co-existence/dominance boundary between the two fighters.
"""
import itertools
import json
import os
import time

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.colors import TwoSlopeNorm

from sparring_dynamics.config import (
    ATTACK, FEINT,
    DEFAULT_N_STEPS, DEFAULT_START_STATE,
    DEFAULT_SELECTION, DEFAULT_CONFIDENCE, DEFAULT_RANDOM_SEED,
    DEFAULT_MEMORY_GROWTH, DEFAULT_MEMORY_DECAY, DEFAULT_STEEPNESS,
    F1_BASE_DEFAULT, F2_BASE_DEFAULT,
    F1_ADAPTATION_DEFAULT, F2_ADAPTATION_DEFAULT,
    F1_PAYOFF_DEFAULT, F2_PAYOFF_DEFAULT,
    F1_COLOR, F2_COLOR, FIGURE_DPI, OUTPUT_DIR
)
from sparring_dynamics.simulation.fighter import Fighter
from sparring_dynamics.simulation.match import SparringMatch
from sparring_dynamics.analysis.monte_carlo import (
    run_monte_carlo, analyze_monte_carlo
)


# ---------------------------------------------------------------------------
# SECTION 1 — Parameter grid definition
# ---------------------------------------------------------------------------

BASELINE_PARAMS = {
    'selection_strength': DEFAULT_SELECTION,      # 1.0
    'memory_decay':       DEFAULT_MEMORY_DECAY,   # 0.95
    'memory_growth':      DEFAULT_MEMORY_GROWTH,  # 1.5
    'steepness':          DEFAULT_STEEPNESS,      # 0.6
}

SWEEP_RANGES = {
    'selection_strength': [0.0, 0.25, 0.5, 0.75, 1.0, 1.5, 2.0],
    'memory_decay':       [0.80, 0.85, 0.90, 0.92, 0.95, 0.97, 0.99],
    'memory_growth':      [0.5, 0.8, 1.0, 1.2, 1.5, 2.0, 2.5],
    'steepness':          [0.1, 0.2, 0.4, 0.6, 0.8, 1.0, 1.5],
}

PARAM_LABELS = {
    'selection_strength': 'Selection Strength (s)',
    'memory_decay':       'Memory Decay (δ)',
    'memory_growth':      'Memory Growth Rate (γ)',
    'steepness':          'Adaptation Rate / Sigmoid Steepness (k)',
}

PARAM_TUMOR_PARALLEL = {
    'selection_strength': (
        "Analogous to the intensity of immune surveillance — "
        "how strongly fitness differences drive state transitions. "
        "s=0 is neutral drift; s=2 is strong selection pressure, "
        "mirroring aggressive immune attack on tumor cells."
    ),
    'memory_decay':       (
        "Analogous to immunological memory half-life — "
        "how quickly the immune system 'forgets' a tumor's pattern "
        "after the tumor changes strategy. "
        "Low δ = fast forgetting; high δ = persistent memory."
    ),
    'memory_growth':      (
        "Analogous to the rate of adaptive resistance accumulation — "
        "how quickly the tumor develops resistance to a repeated "
        "treatment strategy. High γ = rapid resistance."
    ),
    'steepness':          (
        "Analogous to the nonlinearity of the resistance curve — "
        "how sharply the transition from sensitive to resistant occurs. "
        "Low k = gradual adaptation; high k = sudden resistance switch."
    ),
}


def _build_match(params):
    """Construct a fresh CJ-vs-Counter-Fighter SparringMatch from a params dict."""
    cj = Fighter.from_matrices(
        name              = "CJ",
        base_matrix       = F1_BASE_DEFAULT,
        adaptation_matrix = F1_ADAPTATION_DEFAULT,
        payoff_matrix     = F1_PAYOFF_DEFAULT,
        memory_growth     = params['memory_growth'],
        memory_decay      = params['memory_decay'],
        steepness         = params['steepness'],
        color             = F1_COLOR
    )
    cp = Fighter.from_matrices(
        name              = "Counter-Fighter",
        base_matrix       = F2_BASE_DEFAULT,
        adaptation_matrix = F2_ADAPTATION_DEFAULT,
        payoff_matrix     = F2_PAYOFF_DEFAULT,
        memory_growth     = params['memory_growth'],
        memory_decay      = params['memory_decay'],
        steepness         = params['steepness'],
        color             = F2_COLOR
    )
    return SparringMatch(
        fighter1            = cj,
        fighter2            = cp,
        f1_tracked_state    = ATTACK,
        f2_tracked_state    = FEINT,
        selection_strength  = params['selection_strength']
    )


# ---------------------------------------------------------------------------
# SECTION 2 — Single parameter sweep
# ---------------------------------------------------------------------------

def sweep_single_parameter(param_name,
                            param_values,
                            n_simulations=100,
                            n_steps=DEFAULT_N_STEPS,
                            random_seed=DEFAULT_RANDOM_SEED,
                            verbose=True):
    """
    Sweep one parameter across its value range while holding all others
    at baseline. Runs Monte Carlo for each value. Returns a list of
    result dicts, one per parameter value.
    """
    if verbose:
        print(f"\nSweeping {PARAM_LABELS[param_name]}:")
        print(f"  Values: {param_values}")
        print(f"  MC runs per value: {n_simulations}")

    sweep_results = []

    for val in param_values:
        t0 = time.time()

        params = BASELINE_PARAMS.copy()
        params[param_name] = val

        match = _build_match(params)

        mc_results = run_monte_carlo(
            match          = match,
            n_simulations  = n_simulations,
            n_steps        = n_steps,
            start_state    = DEFAULT_START_STATE,
            random_seed    = random_seed
        )
        analysis = analyze_monte_carlo(mc_results, DEFAULT_CONFIDENCE)

        f1_finals = mc_results['f1_cumulative'][:, -1]
        f2_finals = mc_results['f2_cumulative'][:, -1]

        f1_wins = int(np.sum(f1_finals > f2_finals))
        f2_wins = int(np.sum(f2_finals > f1_finals))
        ties    = n_simulations - f1_wins - f2_wins

        f1_ci_l = float(analysis['f1_cumulative']['ci_lower'][-1])
        f1_ci_u = float(analysis['f1_cumulative']['ci_upper'][-1])
        f2_ci_l = float(analysis['f2_cumulative']['ci_lower'][-1])
        f2_ci_u = float(analysis['f2_cumulative']['ci_upper'][-1])
        ci_overlap = not (f1_ci_u < f2_ci_l or f2_ci_u < f1_ci_l)

        runtime = time.time() - t0

        row = {
            'param_name':         param_name,
            'param_value':        float(val),
            'params':             params.copy(),
            'f1_mean_fitness':    float(analysis['f1_cumulative']['mean'][-1]),
            'f2_mean_fitness':    float(analysis['f2_cumulative']['mean'][-1]),
            'f1_std_fitness':     float(analysis['f1_cumulative']['std'][-1]),
            'f2_std_fitness':     float(analysis['f2_cumulative']['std'][-1]),
            'f1_ci_lower':        f1_ci_l,
            'f1_ci_upper':        f1_ci_u,
            'f2_ci_lower':        f2_ci_l,
            'f2_ci_upper':        f2_ci_u,
            'f1_win_rate':        f1_wins / n_simulations,
            'f2_win_rate':        f2_wins / n_simulations,
            'tie_rate':           ties / n_simulations,
            'f1_mean_occupancy':  analysis['f1_occupancy']['mean'].tolist(),
            'f2_mean_occupancy':  analysis['f2_occupancy']['mean'].tolist(),
            'f1_mean_lambda':     float(analysis['f1_lambda']['mean'][-1]),
            'f2_mean_lambda':     float(analysis['f2_lambda']['mean'][-1]),
            'ci_overlap':         ci_overlap,
            'runtime_s':          runtime
        }
        sweep_results.append(row)

        if verbose:
            overlap_str = "overlap" if ci_overlap else "NO overlap"
            print(f"  {param_name}={val:.3f} | "
                  f"F1={row['f1_mean_fitness']:.3f} "
                  f"F2={row['f2_mean_fitness']:.3f} | "
                  f"CI {overlap_str} | "
                  f"{runtime:.1f}s")

    return sweep_results


# ---------------------------------------------------------------------------
# SECTION 3 — Two-parameter sweep (heatmap)
# ---------------------------------------------------------------------------

def sweep_two_parameters(param1_name, param1_values,
                          param2_name, param2_values,
                          n_simulations=50,
                          n_steps=DEFAULT_N_STEPS,
                          random_seed=DEFAULT_RANDOM_SEED,
                          verbose=True):
    """
    Sweep two parameters simultaneously, running MC for every
    combination. Returns results as a 2D grid suitable for heatmaps.
    """
    n1 = len(param1_values)
    n2 = len(param2_values)

    total_combos = n1 * n2
    if verbose:
        total_runs = total_combos * n_simulations
        print(f"\nTwo-parameter sweep: "
              f"{PARAM_LABELS[param1_name]} × "
              f"{PARAM_LABELS[param2_name]}")
        print(f"  Grid: {n1} × {n2} = {total_combos} combinations")
        print(f"  MC runs per combo: {n_simulations}")
        print(f"  Total simulations: {total_runs:,}")

    grid_results     = []
    f1_grid          = np.zeros((n1, n2))
    f2_grid          = np.zeros((n1, n2))
    diff_grid        = np.zeros((n1, n2))
    ci_overlap_grid  = np.zeros((n1, n2), dtype=bool)
    f1_win_grid      = np.zeros((n1, n2))

    combo_count = 0
    for i, v1 in enumerate(param1_values):
        for j, v2 in enumerate(param2_values):
            combo_count += 1

            params = BASELINE_PARAMS.copy()
            params[param1_name] = v1
            params[param2_name] = v2

            match = _build_match(params)

            mc = run_monte_carlo(
                match         = match,
                n_simulations = n_simulations,
                n_steps       = n_steps,
                start_state   = DEFAULT_START_STATE,
                random_seed   = random_seed
            )
            an = analyze_monte_carlo(mc, DEFAULT_CONFIDENCE)

            f1_mean  = float(an['f1_cumulative']['mean'][-1])
            f2_mean  = float(an['f2_cumulative']['mean'][-1])
            f1_ci_l  = float(an['f1_cumulative']['ci_lower'][-1])
            f1_ci_u  = float(an['f1_cumulative']['ci_upper'][-1])
            f2_ci_l  = float(an['f2_cumulative']['ci_lower'][-1])
            f2_ci_u  = float(an['f2_cumulative']['ci_upper'][-1])
            overlap  = not (f1_ci_u < f2_ci_l or f2_ci_u < f1_ci_l)

            f1_finals = mc['f1_cumulative'][:, -1]
            f2_finals = mc['f2_cumulative'][:, -1]
            f1_wins   = float(np.sum(f1_finals > f2_finals)) / n_simulations

            f1_grid[i, j]         = f1_mean
            f2_grid[i, j]         = f2_mean
            diff_grid[i, j]       = f1_mean - f2_mean
            ci_overlap_grid[i, j] = overlap
            f1_win_grid[i, j]     = f1_wins

            row = {
                'param1_name':     param1_name,
                'param1_value':    float(v1),
                'param2_name':     param2_name,
                'param2_value':    float(v2),
                'params':          params.copy(),
                'f1_mean_fitness': f1_mean,
                'f2_mean_fitness': f2_mean,
                'fitness_diff':    f1_mean - f2_mean,
                'f1_ci_lower':     f1_ci_l,
                'f1_ci_upper':     f1_ci_u,
                'f2_ci_lower':     f2_ci_l,
                'f2_ci_upper':     f2_ci_u,
                'f1_win_rate':     f1_wins,
                'ci_overlap':      overlap,
            }
            grid_results.append(row)

            if verbose and combo_count % 5 == 0:
                print(f"  [{combo_count}/{total_combos}] "
                      f"{param1_name}={v1:.2f}, "
                      f"{param2_name}={v2:.2f} → "
                      f"diff={f1_mean-f2_mean:+.3f}")

    return {
        'grid_results':      grid_results,
        'f1_fitness_grid':   f1_grid,
        'f2_fitness_grid':   f2_grid,
        'fitness_diff_grid': diff_grid,
        'ci_overlap_grid':   ci_overlap_grid,
        'f1_win_rate_grid':  f1_win_grid,
        'param1_name':       param1_name,
        'param2_name':       param2_name,
        'param1_values':     param1_values,
        'param2_values':     param2_values,
    }


# ---------------------------------------------------------------------------
# SECTION 4 — Full sensitivity analysis
# ---------------------------------------------------------------------------

def run_full_sensitivity_analysis(n_simulations_1d=100,
                                   n_simulations_2d=50,
                                   n_steps=DEFAULT_N_STEPS,
                                   random_seed=DEFAULT_RANDOM_SEED,
                                   output_dir=None,
                                   verbose=True):
    """
    Run the complete sensitivity analysis: 1D sweeps for all 4
    parameters, 2D sweeps for all 6 parameter pairs, save results to
    JSON/CSV, generate all plots, print summary tables.
    """
    if output_dir is None:
        output_dir = os.path.join(OUTPUT_DIR, 'sensitivity')
    os.makedirs(output_dir, exist_ok=True)

    start_time = time.time()
    all_results = {
        'metadata': {
            'n_simulations_1d': n_simulations_1d,
            'n_simulations_2d': n_simulations_2d,
            'n_steps':          n_steps,
            'random_seed':      random_seed,
            'baseline_params':  BASELINE_PARAMS,
            'timestamp':        time.strftime('%Y-%m-%d %H:%M:%S')
        },
        'single_sweeps': {},
        'pairwise_sweeps': {}
    }

    params = list(SWEEP_RANGES.keys())

    # ── 1D sweeps ────────────────────────────────────────────
    print(f"\n{'='*60}")
    print("PHASE 1: Single-parameter sweeps")
    print(f"{'='*60}")

    for param in params:
        results = sweep_single_parameter(
            param_name    = param,
            param_values  = SWEEP_RANGES[param],
            n_simulations = n_simulations_1d,
            n_steps       = n_steps,
            random_seed   = random_seed,
            verbose       = verbose
        )
        all_results['single_sweeps'][param] = results

    # ── 2D sweeps ────────────────────────────────────────────
    print(f"\n{'='*60}")
    print("PHASE 2: Two-parameter sweeps (heatmaps)")
    print(f"{'='*60}")

    param_pairs = list(itertools.combinations(params, 2))

    for p1, p2 in param_pairs:
        key = f"{p1}_x_{p2}"
        # Use 5-value subset for 2D sweeps to keep runtime manageable
        v1 = SWEEP_RANGES[p1][::2][:5]
        v2 = SWEEP_RANGES[p2][::2][:5]

        grid = sweep_two_parameters(
            param1_name   = p1,
            param1_values = v1,
            param2_name   = p2,
            param2_values = v2,
            n_simulations = n_simulations_2d,
            n_steps       = n_steps,
            random_seed   = random_seed,
            verbose       = verbose
        )
        all_results['pairwise_sweeps'][key] = grid

    # ── Save results ─────────────────────────────────────────
    print(f"\nSaving results...")
    _save_results(all_results, output_dir)

    # ── Generate all plots ────────────────────────────────────
    print(f"\nGenerating plots...")
    plot_sensitivity_results(all_results, output_dir)

    # ── Print summary tables ──────────────────────────────────
    print_sensitivity_summary(all_results)

    elapsed = time.time() - start_time
    print(f"\nFull sensitivity analysis complete in {elapsed:.1f}s")
    print(f"Results saved to: {output_dir}/")

    return all_results


# ---------------------------------------------------------------------------
# SECTION 5 — Save and load results
# ---------------------------------------------------------------------------

def _save_results(all_results, output_dir):
    """
    Save all results in two formats: JSON (complete, for programmatic
    access) and CSV (one row per parameter combination, for R/Excel/
    pandas).
    """
    json_path = os.path.join(output_dir, 'sensitivity_results.json')

    def make_serializable(obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, (np.float32, np.float64)):
            return float(obj)
        if isinstance(obj, (np.int32, np.int64)):
            return int(obj)
        if isinstance(obj, np.bool_):
            return bool(obj)
        return obj

    def serialize_dict(d):
        if isinstance(d, dict):
            return {k: serialize_dict(v) for k, v in d.items()}
        if isinstance(d, list):
            return [serialize_dict(i) for i in d]
        return make_serializable(d)

    with open(json_path, 'w') as f:
        json.dump(serialize_dict(all_results), f, indent=2)
    print(f"  JSON saved: {json_path}")

    # CSV — flat table of all 1D sweep results
    rows = []
    for param, results in all_results['single_sweeps'].items():
        for r in results:
            row = {
                'param_name':      r['param_name'],
                'param_value':     r['param_value'],
                'f1_mean_fitness': r['f1_mean_fitness'],
                'f2_mean_fitness': r['f2_mean_fitness'],
                'f1_std_fitness':  r['f1_std_fitness'],
                'f2_std_fitness':  r['f2_std_fitness'],
                'f1_ci_lower':     r['f1_ci_lower'],
                'f1_ci_upper':     r['f1_ci_upper'],
                'f2_ci_lower':     r['f2_ci_lower'],
                'f2_ci_upper':     r['f2_ci_upper'],
                'f1_win_rate':     r['f1_win_rate'],
                'f2_win_rate':     r['f2_win_rate'],
                'tie_rate':        r['tie_rate'],
                'f1_mean_lambda':  r['f1_mean_lambda'],
                'f2_mean_lambda':  r['f2_mean_lambda'],
                'ci_overlap':      r['ci_overlap'],
                'runtime_s':       r['runtime_s'],
            }
            row.update(r['params'])
            rows.append(row)

    csv_path = os.path.join(output_dir, 'sensitivity_1d.csv')
    pd.DataFrame(rows).to_csv(csv_path, index=False)
    print(f"  CSV saved: {csv_path}")

    # CSV — 2D grid results
    grid_rows = []
    for key, grid in all_results['pairwise_sweeps'].items():
        for r in grid['grid_results']:
            grid_rows.append(r)

    if grid_rows:
        grid_csv_path = os.path.join(output_dir, 'sensitivity_2d.csv')
        pd.DataFrame(grid_rows).to_csv(grid_csv_path, index=False)
        print(f"  CSV saved: {grid_csv_path}")


def load_sensitivity_results(output_dir):
    """Load previously saved sensitivity results from JSON."""
    json_path = os.path.join(output_dir, 'sensitivity_results.json')
    if not os.path.exists(json_path):
        raise FileNotFoundError(
            f"No saved results found at {json_path}. "
            f"Run run_full_sensitivity_analysis() first."
        )
    with open(json_path) as f:
        return json.load(f)


# ---------------------------------------------------------------------------
# SECTION 6 — Visualization
# ---------------------------------------------------------------------------

def plot_sensitivity_results(all_results, output_dir):
    """Generate all sensitivity plots."""

    _plot_1d_sweeps_overview(
        all_results['single_sweeps'],
        os.path.join(output_dir, 'sensitivity_1d_overview.png')
    )

    for param, results in all_results['single_sweeps'].items():
        _plot_1d_sweep_detail(
            results, param,
            os.path.join(output_dir, f'sensitivity_1d_{param}.png')
        )

    _plot_win_rate_curves(
        all_results['single_sweeps'],
        os.path.join(output_dir, 'sensitivity_win_rates.png')
    )

    _plot_ci_overlap_summary(
        all_results['single_sweeps'],
        os.path.join(output_dir, 'sensitivity_ci_overlap.png')
    )

    for key, grid in all_results['pairwise_sweeps'].items():
        _plot_2d_heatmaps(
            grid,
            os.path.join(output_dir, f'sensitivity_2d_{key}.png')
        )

    _plot_robustness_summary(
        all_results['single_sweeps'],
        os.path.join(output_dir, 'sensitivity_robustness.png')
    )

    print(f"  All sensitivity plots saved to {output_dir}/")


def _plot_1d_sweeps_overview(single_sweeps, filepath):
    """2x2 grid: mean fitness vs parameter value for all 4 params, with CI bands."""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10),
                               constrained_layout=True)
    fig.suptitle('Sensitivity Analysis — Mean Final Fitness vs Parameter Value',
                  fontsize=13, fontweight='bold')

    params = list(single_sweeps.keys())

    for ax, param in zip(axes.flat, params):
        results = single_sweeps[param]
        xs      = [r['param_value'] for r in results]
        f1_mean = [r['f1_mean_fitness'] for r in results]
        f2_mean = [r['f2_mean_fitness'] for r in results]
        f1_ci_l = [r['f1_ci_lower'] for r in results]
        f1_ci_u = [r['f1_ci_upper'] for r in results]
        f2_ci_l = [r['f2_ci_lower'] for r in results]
        f2_ci_u = [r['f2_ci_upper'] for r in results]

        ax.fill_between(xs, f1_ci_l, f1_ci_u,
                         alpha=0.2, color=F1_COLOR)
        ax.fill_between(xs, f2_ci_l, f2_ci_u,
                         alpha=0.2, color=F2_COLOR)
        ax.plot(xs, f1_mean, 'o-', color=F1_COLOR,
                 linewidth=2, markersize=5, label='CJ')
        ax.plot(xs, f2_mean, 's-', color=F2_COLOR,
                 linewidth=2, markersize=5, label='Counter-Fighter')

        baseline = BASELINE_PARAMS[param]
        ax.axvline(baseline, color='gray', linestyle='--',
                    linewidth=1, alpha=0.7, label=f'Baseline={baseline}')

        ax.set_xlabel(PARAM_LABELS[param], fontsize=9)
        ax.set_ylabel('Mean Final Cumulative Fitness', fontsize=9)
        ax.set_title(PARAM_LABELS[param], fontsize=10, fontweight='bold')
        ax.legend(fontsize=8)

    plt.savefig(filepath, dpi=FIGURE_DPI, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {filepath}")


def _plot_1d_sweep_detail(results, param, filepath):
    """3-panel detail figure: mean fitness+CI, fitness difference+CI, win-rate stack."""
    fig, axes = plt.subplots(3, 1, figsize=(10, 12),
                               constrained_layout=True)
    fig.suptitle(
        f"Detailed Sensitivity — {PARAM_LABELS[param]}\n"
        f"{PARAM_TUMOR_PARALLEL[param]}",
        fontsize=10, fontweight='bold'
    )

    xs      = [r['param_value'] for r in results]
    f1_mean = np.array([r['f1_mean_fitness'] for r in results])
    f2_mean = np.array([r['f2_mean_fitness'] for r in results])
    f1_ci_l = np.array([r['f1_ci_lower'] for r in results])
    f1_ci_u = np.array([r['f1_ci_upper'] for r in results])
    f2_ci_l = np.array([r['f2_ci_lower'] for r in results])
    f2_ci_u = np.array([r['f2_ci_upper'] for r in results])
    f1_win  = np.array([r['f1_win_rate'] for r in results])
    f2_win  = np.array([r['f2_win_rate'] for r in results])
    ties    = np.array([r['tie_rate'] for r in results])

    ax = axes[0]
    ax.fill_between(xs, f1_ci_l, f1_ci_u, alpha=0.2, color=F1_COLOR)
    ax.fill_between(xs, f2_ci_l, f2_ci_u, alpha=0.2, color=F2_COLOR)
    ax.plot(xs, f1_mean, 'o-', color=F1_COLOR, linewidth=2,
             markersize=5, label='CJ (Fighter 1)')
    ax.plot(xs, f2_mean, 's-', color=F2_COLOR, linewidth=2,
             markersize=5, label='Counter-Fighter (Fighter 2)')
    ax.axvline(BASELINE_PARAMS[param], color='gray',
                linestyle='--', linewidth=1, label='Baseline')
    ax.set_ylabel('Mean Final Fitness', fontsize=9)
    ax.set_title('Mean Final Cumulative Fitness ± 95% CI', fontsize=9)
    ax.legend(fontsize=8)

    ax = axes[1]
    diff     = f1_mean - f2_mean
    diff_ci_l = f1_ci_l - f2_ci_u
    diff_ci_u = f1_ci_u - f2_ci_l

    ax.fill_between(xs, diff_ci_l, diff_ci_u, alpha=0.2, color='purple')
    ax.plot(xs, diff, 'o-', color='purple', linewidth=2, markersize=5)
    ax.axhline(0, color='black', linewidth=0.8, linestyle='--')
    ax.axvline(BASELINE_PARAMS[param], color='gray',
                linestyle='--', linewidth=1)
    ax.fill_between(xs, diff, 0,
                     where=(diff > 0), alpha=0.15, color=F1_COLOR,
                     label='F1 advantage')
    ax.fill_between(xs, diff, 0,
                     where=(diff < 0), alpha=0.15, color=F2_COLOR,
                     label='F2 advantage')
    ax.set_ylabel('Fitness Difference (F1 − F2)', fontsize=9)
    ax.set_title('Fitness Advantage (F1 − F2) ± 95% CI', fontsize=9)
    ax.legend(fontsize=8)

    ax = axes[2]
    ax.stackplot(xs, f1_win, ties, f2_win,
                  labels=['F1 wins', 'Ties', 'F2 wins'],
                  colors=[F1_COLOR, 'lightgray', F2_COLOR],
                  alpha=0.8)
    ax.axvline(BASELINE_PARAMS[param], color='gray',
                linestyle='--', linewidth=1)
    ax.set_ylabel('Fraction of simulations', fontsize=9)
    ax.set_xlabel(PARAM_LABELS[param], fontsize=9)
    ax.set_title('Win Rate Distribution', fontsize=9)
    ax.set_ylim(0, 1)
    ax.legend(fontsize=8, loc='upper right')

    plt.savefig(filepath, dpi=FIGURE_DPI, bbox_inches='tight')
    plt.close()


def _plot_win_rate_curves(single_sweeps, filepath):
    """F1 win rate vs normalized parameter value, all 4 params on one axes."""
    fig, ax = plt.subplots(figsize=(10, 6), constrained_layout=True)

    colors_params = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']

    for color, (param, results) in zip(
        colors_params, single_sweeps.items()
    ):
        xs_raw   = np.array([r['param_value'] for r in results])
        f1_win   = np.array([r['f1_win_rate'] for r in results])

        xs_norm  = (xs_raw - xs_raw.min()) / (xs_raw.max() - xs_raw.min())

        ax.plot(xs_norm, f1_win, 'o-', color=color,
                 linewidth=2, markersize=5, label=PARAM_LABELS[param])

    ax.axhline(0.5, color='black', linestyle='--',
                linewidth=0.8, alpha=0.5, label='50% (equal odds)')
    ax.set_xlabel('Normalized parameter value (0=min, 1=max)', fontsize=10)
    ax.set_ylabel('F1 (CJ) Win Rate', fontsize=10)
    ax.set_title('F1 Win Rate vs Each Parameter (Normalized)\n'
                  'Steeper curves = higher parameter sensitivity',
                  fontsize=11, fontweight='bold')
    ax.set_ylim(0, 1)
    ax.legend(fontsize=9)

    plt.savefig(filepath, dpi=FIGURE_DPI, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {filepath}")


def _plot_ci_overlap_summary(single_sweeps, filepath):
    """
    2x2 subplot grid: for each parameter, points colored green (CIs
    overlap = co-existence) or red (no overlap = dominance), answering
    at what parameter values the system transitions between the two.
    """
    fig, axes = plt.subplots(2, 2, figsize=(12, 8),
                               constrained_layout=True)
    fig.suptitle(
        'CI Overlap Map — Green: Co-existence | Red: Dominance\n'
        'Evolutionary stable co-existence vs dominant strategy regions',
        fontsize=11, fontweight='bold'
    )

    for ax, (param, results) in zip(axes.flat, single_sweeps.items()):
        xs      = [r['param_value'] for r in results]
        overlap = [r['ci_overlap'] for r in results]
        f1_mean = [r['f1_mean_fitness'] for r in results]
        f2_mean = [r['f2_mean_fitness'] for r in results]

        colors = ['#2ecc71' if o else '#e74c3c' for o in overlap]

        ax.scatter(xs, f1_mean, c=colors, s=80,
                    zorder=3, label='F1 mean fitness', marker='o')
        ax.scatter(xs, f2_mean, c=colors, s=80,
                    zorder=3, label='F2 mean fitness', marker='s',
                    edgecolors='black', linewidth=0.5)

        ax.plot(xs, f1_mean, '-', color=F1_COLOR,
                 linewidth=1, alpha=0.4)
        ax.plot(xs, f2_mean, '-', color=F2_COLOR,
                 linewidth=1, alpha=0.4)

        ax.axvline(BASELINE_PARAMS[param], color='gray',
                    linestyle='--', linewidth=1)

        green_patch = mpatches.Patch(
            color='#2ecc71', label='CIs overlap (co-exist)'
        )
        red_patch = mpatches.Patch(
            color='#e74c3c', label='CIs separate (dominance)'
        )
        ax.legend(handles=[green_patch, red_patch], fontsize=7)
        ax.set_xlabel(PARAM_LABELS[param], fontsize=8)
        ax.set_ylabel('Mean Fitness', fontsize=8)
        ax.set_title(PARAM_LABELS[param], fontsize=9, fontweight='bold')

    plt.savefig(filepath, dpi=FIGURE_DPI, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {filepath}")


def _plot_2d_heatmaps(grid, filepath):
    """
    2x2 heatmap grid for a two-parameter sweep: F1 fitness, F2 fitness,
    fitness difference (diverging colormap), F1 win rate with CI
    overlap boundary contour.
    """
    p1_name = grid['param1_name']
    p2_name = grid['param2_name']
    p1_vals = grid['param1_values']
    p2_vals = grid['param2_values']

    fig, axes = plt.subplots(2, 2, figsize=(14, 11),
                               constrained_layout=True)
    fig.suptitle(
        f"Two-Parameter Sensitivity Heatmap\n"
        f"{PARAM_LABELS[p1_name]} × {PARAM_LABELS[p2_name]}",
        fontsize=12, fontweight='bold'
    )

    xtick_labels = [f"{v:.2f}" for v in p2_vals]
    ytick_labels = [f"{v:.2f}" for v in p1_vals]

    heatmap_configs = [
        (grid['f1_fitness_grid'],   'F1 (CJ) Mean Fitness',        'Blues',  False, axes[0,0]),
        (grid['f2_fitness_grid'],   'F2 Mean Fitness',              'Oranges',False, axes[0,1]),
        (grid['fitness_diff_grid'], 'Fitness Difference (F1 − F2)','RdBu_r', True,  axes[1,0]),
        (grid['f1_win_rate_grid'],  'F1 Win Rate',                  'RdYlGn', False, axes[1,1]),
    ]

    for data, title, cmap, diverging, ax in heatmap_configs:
        if diverging:
            vmax = np.abs(data).max()
            norm = TwoSlopeNorm(vmin=-vmax, vcenter=0, vmax=vmax)
            im = ax.imshow(data, cmap=cmap, norm=norm, aspect='auto')
        else:
            im = ax.imshow(data, cmap=cmap, aspect='auto',
                            vmin=data.min(), vmax=data.max())

        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

        for i in range(len(p1_vals)):
            for j in range(len(p2_vals)):
                val = data[i, j]
                ax.text(j, i, f"{val:.3f}",
                         ha='center', va='center', fontsize=7,
                         color='white' if abs(val) > 0.6*data.max()
                               else 'black')

        if title == 'F1 Win Rate':
            overlap = grid['ci_overlap_grid'].astype(float)
            if overlap.min() != overlap.max():
                cs = ax.contour(overlap, levels=[0.5],
                                 colors=['black'], linewidths=1.5)
                ax.clabel(cs, fmt={0.5: 'CI boundary'}, fontsize=7)

        ax.set_xticks(range(len(p2_vals)))
        ax.set_yticks(range(len(p1_vals)))
        ax.set_xticklabels(xtick_labels, rotation=45,
                             ha='right', fontsize=8)
        ax.set_yticklabels(ytick_labels, fontsize=8)
        ax.set_xlabel(PARAM_LABELS[p2_name], fontsize=9)
        ax.set_ylabel(PARAM_LABELS[p1_name], fontsize=9)
        ax.set_title(title, fontsize=10, fontweight='bold')

    plt.savefig(filepath, dpi=FIGURE_DPI, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {filepath}")


def _plot_robustness_summary(single_sweeps, filepath):
    """
    Robustness plot: box plots of fitness range per parameter, plus
    coefficient-of-variation sensitivity index bars.
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 6),
                               constrained_layout=True)
    fig.suptitle('Parameter Robustness Analysis',
                  fontsize=12, fontweight='bold')

    params      = list(single_sweeps.keys())
    labels      = [PARAM_LABELS[p].split('(')[0].strip()
                   for p in params]

    ax = axes[0]
    f1_data = [
        [r['f1_mean_fitness'] for r in single_sweeps[p]]
        for p in params
    ]
    bp = ax.boxplot(f1_data, vert=False, patch_artist=True,
                     notch=True, tick_labels=labels)
    for patch in bp['boxes']:
        patch.set_facecolor(F1_COLOR)
        patch.set_alpha(0.7)
    ax.axvline(
        np.mean([r['f1_mean_fitness']
                 for results in single_sweeps.values()
                 for r in results]),
        color='navy', linestyle='--', linewidth=1,
        label='Grand mean F1 fitness'
    )
    ax.set_xlabel('F1 Mean Final Fitness', fontsize=10)
    ax.set_title('F1 Fitness Distribution Across\nParameter Sweep Range',
                  fontsize=10, fontweight='bold')
    ax.legend(fontsize=8)

    ax = axes[1]
    cv_f1 = []
    cv_f2 = []

    for param in params:
        results  = single_sweeps[param]
        f1_vals  = np.array([r['f1_mean_fitness'] for r in results])
        f2_vals  = np.array([r['f2_mean_fitness'] for r in results])
        cv_f1.append(f1_vals.std() / f1_vals.mean()
                      if f1_vals.mean() > 0 else 0)
        cv_f2.append(f2_vals.std() / f2_vals.mean()
                      if f2_vals.mean() > 0 else 0)

    x   = np.arange(len(params))
    w   = 0.35
    ax.bar(x - w/2, cv_f1, w, color=F1_COLOR,
            alpha=0.8, label='CJ (F1)', edgecolor='white')
    ax.bar(x + w/2, cv_f2, w, color=F2_COLOR,
            alpha=0.8, label='Counter-Fighter (F2)', edgecolor='white')

    for i, (v1, v2) in enumerate(zip(cv_f1, cv_f2)):
        ax.text(i - w/2, v1 + 0.001, f"{v1:.4f}",
                 ha='center', va='bottom', fontsize=7)
        ax.text(i + w/2, v2 + 0.001, f"{v2:.4f}",
                 ha='center', va='bottom', fontsize=7)

    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=20, ha='right', fontsize=9)
    ax.set_ylabel('Coefficient of Variation (std/mean)', fontsize=10)
    ax.set_title('Parameter Sensitivity Index\n'
                  'Higher = more sensitive to this parameter',
                  fontsize=10, fontweight='bold')
    ax.legend(fontsize=9)

    plt.savefig(filepath, dpi=FIGURE_DPI, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {filepath}")


# ---------------------------------------------------------------------------
# SECTION 7 — Summary tables
# ---------------------------------------------------------------------------

def print_sensitivity_summary(all_results):
    """Print formatted summary tables: sensitivity ranking, critical transition
    points, robustness assessment, tumor-immune interpretations."""
    print(f"\n{'='*70}")
    print("SENSITIVITY ANALYSIS SUMMARY")
    print(f"{'='*70}")

    print(f"\n── Table 1: Parameter Sensitivity Ranking ──────────────────")
    print(f"{'Parameter':<35} {'F1 CV':>8} {'F2 CV':>8} {'F1 Range':>10}")
    print("─" * 65)

    rankings = []
    for param, results in all_results['single_sweeps'].items():
        f1_vals = np.array([r['f1_mean_fitness'] for r in results])
        f2_vals = np.array([r['f2_mean_fitness'] for r in results])
        cv_f1   = float(f1_vals.std() / f1_vals.mean()) if f1_vals.mean() > 0 else 0
        cv_f2   = float(f2_vals.std() / f2_vals.mean()) if f2_vals.mean() > 0 else 0
        f1_range = float(f1_vals.max() - f1_vals.min())
        rankings.append((param, cv_f1, cv_f2, f1_range))

    rankings.sort(key=lambda x: x[1], reverse=True)
    for param, cv_f1, cv_f2, rng in rankings:
        label = PARAM_LABELS[param].split('(')[0].strip()
        print(f"  {label:<33} {cv_f1:>8.4f} {cv_f2:>8.4f} {rng:>10.4f}")

    most_sensitive = rankings[0][0]
    print(f"\n  Most sensitive parameter: {PARAM_LABELS[most_sensitive]}")

    print(f"\n── Table 2: Co-existence Breakdown Points ───────────────────")
    print(f"{'Parameter':<35} {'Co-exist range':>20} {'Dominance range':>20}")
    print("─" * 78)

    for param, results in all_results['single_sweeps'].items():
        label   = PARAM_LABELS[param].split('(')[0].strip()
        vals    = [r['param_value'] for r in results]
        overlap = [r['ci_overlap'] for r in results]

        coexist  = [v for v, o in zip(vals, overlap) if o]
        dominant = [v for v, o in zip(vals, overlap) if not o]

        co_str  = (f"[{min(coexist):.2f}, {max(coexist):.2f}]"
                    if coexist else "None")
        dom_str = (f"[{min(dominant):.2f}, {max(dominant):.2f}]"
                    if dominant else "None")

        print(f"  {label:<33} {co_str:>20} {dom_str:>20}")

    print(f"\n── Table 3: Robustness Assessment ──────────────────────────")
    for param, results in all_results['single_sweeps'].items():
        label   = PARAM_LABELS[param]
        f1_vals = np.array([r['f1_mean_fitness'] for r in results])
        cv      = float(f1_vals.std() / f1_vals.mean()) if f1_vals.mean() > 0 else 0

        if cv < 0.01:
            robustness = "Very Robust — parameter has minimal effect"
        elif cv < 0.03:
            robustness = "Robust — small but noticeable effect"
        elif cv < 0.06:
            robustness = "Moderate — parameter meaningfully affects outcomes"
        else:
            robustness = "Sensitive — parameter strongly drives outcomes"

        print(f"\n  {label}")
        print(f"    CV={cv:.4f} → {robustness}")

    print(f"\n── Table 4: Tumor-Immune Parallels ──────────────────────────")
    for param in all_results['single_sweeps']:
        print(f"\n  {PARAM_LABELS[param]}:")
        print(f"    {PARAM_TUMOR_PARALLEL[param]}")

    print(f"\n{'='*70}")
