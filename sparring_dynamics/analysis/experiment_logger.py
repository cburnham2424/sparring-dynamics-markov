"""
Experiment logging system: persist one complete Monte Carlo experiment
(parameters, matrices, aggregate results, and optionally full raw
arrays) to disk in a structured, reproducible directory layout, and
maintain a flat CSV registry across many experiments for cross-run
comparison.
"""
import hashlib
import json
import os
import sys
import subprocess
import time
from datetime import datetime

import numpy as np
import pandas as pd

from sparring_dynamics.config import STATES, N_STATES, OUTPUT_DIR


# ---------------------------------------------------------------------------
# SECTION 1 — Experiment ID and metadata
# ---------------------------------------------------------------------------

def generate_experiment_id(params, random_seed):
    """
    Generate a deterministic experiment ID from parameters and seed so
    identical experiments always get the same ID *hash*.

    Format: EXP_{8-char hash}_{timestamp}

    The hash is computed from a canonical JSON serialization of
    params + seed, so it is reproducible across runs with the same
    configuration; the timestamp suffix keeps re-runs of the same
    configuration from colliding on the same directory on disk.
    """
    canonical = json.dumps(
        {'params': params, 'seed': random_seed},
        sort_keys=True
    )
    hash_str = hashlib.sha256(
        canonical.encode()
    ).hexdigest()[:8].upper()
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    return f"EXP_{hash_str}_{timestamp}"


def build_experiment_metadata(experiment_id, params,
                               random_seed, n_simulations,
                               n_steps, notes=""):
    """
    Build a complete metadata dict for one experiment: experiment_id,
    timestamp, random_seed, n_simulations, n_steps, notes, git_commit,
    python/numpy/pandas versions, plus all simulation parameters
    flattened in.
    """
    try:
        git_commit = subprocess.check_output(
            ['git', 'rev-parse', '--short', 'HEAD'],
            stderr=subprocess.DEVNULL
        ).decode().strip()
    except Exception:
        git_commit = 'unknown'

    meta = {
        'experiment_id':  experiment_id,
        'timestamp':      datetime.now().isoformat(),
        'random_seed':    random_seed,
        'n_simulations':  n_simulations,
        'n_steps':        n_steps,
        'notes':          notes,
        'git_commit':     git_commit,
        'python_version': sys.version.split()[0],
        'numpy_version':  np.__version__,
        'pandas_version': pd.__version__,
    }
    meta.update(params)
    return meta


# ---------------------------------------------------------------------------
# SECTION 2 — Experiment logger class
# ---------------------------------------------------------------------------

class ExperimentLogger:
    """
    Logs one complete experiment (one Monte Carlo run or one
    sensitivity sweep combination) to disk in multiple formats.

    Directory structure created per experiment:

    outputs/experiments/{experiment_id}/
    |-- metadata.json
    |-- parameters.csv
    |-- matrices/{f1,f2}_{base,adaptation,payoff}.csv
    |-- results/cumulative_fitness.csv, per_step_fitness.csv,
    |           state_frequencies.csv, transition_frequencies_{f1,f2}.csv,
    |           lambda_history.csv, outcome_summary.csv
    `-- raw/{f1,f2}_{cumulative,states}_all_runs.csv   (if save_raw)
    """

    def __init__(self, experiment_id,
                 base_dir=None,
                 save_raw=True):

        self.experiment_id = experiment_id
        self.base_dir = base_dir or os.path.join(
            OUTPUT_DIR, 'experiments'
        )
        self.save_raw = save_raw

        self.exp_dir     = os.path.join(
            self.base_dir, experiment_id
        )
        self.matrices_dir = os.path.join(self.exp_dir, 'matrices')
        self.results_dir  = os.path.join(self.exp_dir, 'results')
        self.raw_dir      = os.path.join(self.exp_dir, 'raw')

        os.makedirs(self.matrices_dir, exist_ok=True)
        os.makedirs(self.results_dir,  exist_ok=True)
        if save_raw:
            os.makedirs(self.raw_dir, exist_ok=True)

        self.logged_files = []
        self._start_time  = time.time()

    def log_metadata(self, metadata):
        """Save metadata as metadata.json (structured) and parameters.csv (flat, 1 row)."""
        json_path = os.path.join(self.exp_dir, 'metadata.json')
        with open(json_path, 'w') as f:
            json.dump(metadata, f, indent=2, default=str)
        self.logged_files.append(json_path)

        csv_path = os.path.join(self.exp_dir, 'parameters.csv')
        pd.DataFrame([metadata]).to_csv(csv_path, index=False)
        self.logged_files.append(csv_path)

        return self

    def log_matrices(self, f1_base, f2_base,
                     f1_adaptation, f2_adaptation,
                     f1_payoff, f2_payoff):
        """
        Save all six matrices as CSVs with state labels as row/column
        headers (index=from_state, columns=to_state), with a row_sum
        column on the stochastic (base/adaptation) matrices to verify
        row-stochasticity.
        """
        matrix_specs = [
            (f1_base,       'f1_base.csv',       'Base transition'),
            (f2_base,       'f2_base.csv',       'Base transition'),
            (f1_adaptation, 'f1_adaptation.csv', 'Adaptation'),
            (f2_adaptation, 'f2_adaptation.csv', 'Adaptation'),
            (f1_payoff,     'f1_payoff.csv',     'Payoff'),
            (f2_payoff,     'f2_payoff.csv',     'Payoff'),
        ]

        for matrix, filename, matrix_type in matrix_specs:
            path = os.path.join(self.matrices_dir, filename)
            df   = pd.DataFrame(
                matrix,
                index   = STATES,
                columns = STATES
            )
            df.index.name = 'from_state'

            if matrix_type in ('Base transition', 'Adaptation'):
                df['row_sum'] = df.sum(axis=1)

            df.to_csv(path)
            self.logged_files.append(path)

        return self

    def log_cumulative_fitness(self, analysis):
        """Mean/std/CI/median of cumulative fitness over time for both fighters."""
        path  = os.path.join(
            self.results_dir, 'cumulative_fitness.csv'
        )
        n     = len(analysis['f1_cumulative']['mean'])
        steps = np.arange(n)

        df = pd.DataFrame({
            'step':           steps,
            'f1_mean':        analysis['f1_cumulative']['mean'],
            'f1_std':         analysis['f1_cumulative']['std'],
            'f1_ci_lower':    analysis['f1_cumulative']['ci_lower'],
            'f1_ci_upper':    analysis['f1_cumulative']['ci_upper'],
            'f1_median':      analysis['f1_cumulative']['median'],
            'f2_mean':        analysis['f2_cumulative']['mean'],
            'f2_std':         analysis['f2_cumulative']['std'],
            'f2_ci_lower':    analysis['f2_cumulative']['ci_lower'],
            'f2_ci_upper':    analysis['f2_cumulative']['ci_upper'],
            'f2_median':      analysis['f2_cumulative']['median'],
            'fitness_diff':   (analysis['f1_cumulative']['mean'] -
                               analysis['f2_cumulative']['mean']),
        })
        df.to_csv(path, index=False)
        self.logged_files.append(path)
        return self

    def log_per_step_fitness(self, analysis):
        """Per-step payoff statistics over time."""
        path  = os.path.join(
            self.results_dir, 'per_step_fitness.csv'
        )
        n     = len(analysis['f1_fitness']['mean'])
        steps = np.arange(n)

        df = pd.DataFrame({
            'step':        steps,
            'f1_mean':     analysis['f1_fitness']['mean'],
            'f1_std':      analysis['f1_fitness']['std'],
            'f1_ci_lower': analysis['f1_fitness']['ci_lower'],
            'f1_ci_upper': analysis['f1_fitness']['ci_upper'],
            'f2_mean':     analysis['f2_fitness']['mean'],
            'f2_std':      analysis['f2_fitness']['std'],
            'f2_ci_lower': analysis['f2_fitness']['ci_lower'],
            'f2_ci_upper': analysis['f2_fitness']['ci_upper'],
        })
        df.to_csv(path, index=False)
        self.logged_files.append(path)
        return self

    def log_state_frequencies(self, analysis):
        """Mean state occupancy with CI for both fighters."""
        path = os.path.join(
            self.results_dir, 'state_frequencies.csv'
        )
        f1_occ = analysis['f1_occupancy']
        f2_occ = analysis['f2_occupancy']

        df = pd.DataFrame({
            'state':        STATES,
            'f1_mean':      f1_occ['mean'],
            'f1_std':       f1_occ['std'],
            'f1_ci_lower':  f1_occ['ci_lower'],
            'f1_ci_upper':  f1_occ['ci_upper'],
            'f2_mean':      f2_occ['mean'],
            'f2_std':       f2_occ['std'],
            'f2_ci_lower':  f2_occ['ci_lower'],
            'f2_ci_upper':  f2_occ['ci_upper'],
        })
        df.to_csv(path, index=False)
        self.logged_files.append(path)
        return self

    def log_transition_frequencies(self, mc_results):
        """
        Estimate the empirical transition matrix from all MC state
        histories (across all N_simulations x N_steps) and save as
        CSV for both fighters: transition_frequencies_{f1,f2}.csv,
        row-normalized, plus an observation_count column.
        """
        for fighter_key, filename in [
            ('f1_states', 'transition_frequencies_f1.csv'),
            ('f2_states', 'transition_frequencies_f2.csv')
        ]:
            state_array = mc_results[fighter_key]
            N, T = state_array.shape

            counts = np.zeros((N_STATES, N_STATES), dtype=float)
            for i in range(N):
                for t in range(T - 1):
                    s  = state_array[i, t]
                    s2 = state_array[i, t + 1]
                    counts[s, s2] += 1

            row_sums = counts.sum(axis=1, keepdims=True)
            row_sums = np.where(row_sums < 1e-10, 1.0, row_sums)
            freq_matrix = counts / row_sums

            df = pd.DataFrame(
                freq_matrix,
                index   = STATES,
                columns = STATES
            )
            df.index.name = 'from_state'
            df['observation_count'] = counts.sum(axis=1)
            df['row_sum']           = freq_matrix.sum(axis=1)

            path = os.path.join(self.results_dir, filename)
            df.to_csv(path)
            self.logged_files.append(path)

        return self

    def log_lambda_history(self, analysis):
        """Adaptation weight lambda history over time for both fighters."""
        path  = os.path.join(
            self.results_dir, 'lambda_history.csv'
        )
        n     = len(analysis['f1_lambda']['mean'])
        steps = np.arange(n)

        df = pd.DataFrame({
            'step':        steps,
            'f1_mean':     analysis['f1_lambda']['mean'],
            'f1_std':      analysis['f1_lambda']['std'],
            'f1_ci_lower': analysis['f1_lambda']['ci_lower'],
            'f1_ci_upper': analysis['f1_lambda']['ci_upper'],
            'f2_mean':     analysis['f2_lambda']['mean'],
            'f2_std':      analysis['f2_lambda']['std'],
            'f2_ci_lower': analysis['f2_lambda']['ci_lower'],
            'f2_ci_upper': analysis['f2_lambda']['ci_upper'],
        })
        df.to_csv(path, index=False)
        self.logged_files.append(path)
        return self

    def log_outcome_summary(self, mc_results, analysis):
        """
        One-row outcome summary CSV with all key scalar results for
        this experiment — the most important file for aggregating
        results across many experiments later.
        """
        path = os.path.join(
            self.results_dir, 'outcome_summary.csv'
        )

        f1_finals = mc_results['f1_cumulative'][:, -1]
        f2_finals = mc_results['f2_cumulative'][:, -1]
        N         = mc_results['n_simulations']

        f1_wins = int(np.sum(f1_finals > f2_finals))
        f2_wins = int(np.sum(f2_finals > f1_finals))
        ties    = N - f1_wins - f2_wins

        f1_ci_l = float(analysis['f1_cumulative']['ci_lower'][-1])
        f1_ci_u = float(analysis['f1_cumulative']['ci_upper'][-1])
        f2_ci_l = float(analysis['f2_cumulative']['ci_lower'][-1])
        f2_ci_u = float(analysis['f2_cumulative']['ci_upper'][-1])
        overlap = not (f1_ci_u < f2_ci_l or f2_ci_u < f1_ci_l)

        f1_occ = analysis['f1_occupancy']['mean']
        f2_occ = analysis['f2_occupancy']['mean']

        runtime = time.time() - self._start_time

        row = {
            'experiment_id':        self.experiment_id,
            'f1_final_mean':        float(analysis['f1_cumulative']['mean'][-1]),
            'f1_final_std':         float(analysis['f1_cumulative']['std'][-1]),
            'f1_final_ci_lower':    f1_ci_l,
            'f1_final_ci_upper':    f1_ci_u,
            'f2_final_mean':        float(analysis['f2_cumulative']['mean'][-1]),
            'f2_final_std':         float(analysis['f2_cumulative']['std'][-1]),
            'f2_final_ci_lower':    f2_ci_l,
            'f2_final_ci_upper':    f2_ci_u,
            'fitness_diff_mean':    float(
                analysis['f1_cumulative']['mean'][-1] -
                analysis['f2_cumulative']['mean'][-1]
            ),
            'ci_overlap':           overlap,
            'f1_win_rate':          f1_wins / N,
            'f2_win_rate':          f2_wins / N,
            'tie_rate':             ties / N,
            'f1_final_lambda_mean': float(
                analysis['f1_lambda']['mean'][-1]
            ),
            'f2_final_lambda_mean': float(
                analysis['f2_lambda']['mean'][-1]
            ),
            'f1_attack_occ':        float(f1_occ[0]),
            'f1_defend_occ':        float(f1_occ[1]),
            'f1_disengage_occ':     float(f1_occ[2]),
            'f1_feint_occ':         float(f1_occ[3]),
            'f2_attack_occ':        float(f2_occ[0]),
            'f2_defend_occ':        float(f2_occ[1]),
            'f2_disengage_occ':     float(f2_occ[2]),
            'f2_feint_occ':         float(f2_occ[3]),
            'runtime_s':            runtime,
        }

        pd.DataFrame([row]).to_csv(path, index=False)
        self.logged_files.append(path)
        return self

    def log_raw_arrays(self, mc_results):
        """
        Save full MC output arrays as CSVs (N_simulations x N_steps),
        if save_raw=True. Warning: these files can be large.
        """
        if not self.save_raw:
            return self

        raw_specs = [
            (mc_results['f1_cumulative'], 'f1_cumulative_all_runs.csv'),
            (mc_results['f2_cumulative'], 'f2_cumulative_all_runs.csv'),
            (mc_results['f1_states'],     'f1_states_all_runs.csv'),
            (mc_results['f2_states'],     'f2_states_all_runs.csv'),
        ]

        for array, filename in raw_specs:
            path = os.path.join(self.raw_dir, filename)
            N, T = array.shape

            col_headers = [f"step_{t}" for t in range(T)]
            row_headers = [f"run_{i}" for i in range(N)]

            df = pd.DataFrame(
                array,
                index   = row_headers,
                columns = col_headers
            )
            df.index.name = 'simulation_run'
            df.to_csv(path)
            self.logged_files.append(path)

        return self

    def finalize(self):
        """Write a manifest listing all logged files with sizes, then print a summary."""
        manifest_path = os.path.join(
            self.exp_dir, 'manifest.json'
        )
        manifest = {
            'experiment_id': self.experiment_id,
            'total_files':   len(self.logged_files),
            'total_size_kb': sum(
                os.path.getsize(f) / 1024
                for f in self.logged_files
                if os.path.exists(f)
            ),
            'runtime_s':     time.time() - self._start_time,
            'files': [
                {
                    'path': os.path.relpath(f, self.exp_dir),
                    'size_kb': os.path.getsize(f) / 1024
                    if os.path.exists(f) else 0
                }
                for f in self.logged_files
            ]
        }
        with open(manifest_path, 'w') as f:
            json.dump(manifest, f, indent=2)

        print(f"\n── Experiment logged ─────────────────────────────────")
        print(f"  ID:         {self.experiment_id}")
        print(f"  Directory:  {self.exp_dir}")
        print(f"  Files:      {len(self.logged_files)}")
        print(f"  Total size: {manifest['total_size_kb']:.1f} KB")
        print(f"  Runtime:    {manifest['runtime_s']:.1f}s")

        return self


# ---------------------------------------------------------------------------
# SECTION 3 — High-level logging function
# ---------------------------------------------------------------------------

def log_experiment(params, mc_results, analysis,
                   f1_base, f2_base,
                   f1_adaptation, f2_adaptation,
                   f1_payoff, f2_payoff,
                   random_seed, n_simulations, n_steps,
                   notes="", save_raw=True,
                   base_dir=None):
    """
    One-call convenience function that runs the complete logging
    pipeline for a single experiment. Returns (logger, experiment_id).
    """
    experiment_id = generate_experiment_id(params, random_seed)

    metadata = build_experiment_metadata(
        experiment_id  = experiment_id,
        params         = params,
        random_seed    = random_seed,
        n_simulations  = n_simulations,
        n_steps        = n_steps,
        notes          = notes
    )

    logger = ExperimentLogger(
        experiment_id = experiment_id,
        base_dir      = base_dir,
        save_raw      = save_raw
    )

    (logger
        .log_metadata(metadata)
        .log_matrices(f1_base, f2_base,
                      f1_adaptation, f2_adaptation,
                      f1_payoff, f2_payoff)
        .log_cumulative_fitness(analysis)
        .log_per_step_fitness(analysis)
        .log_state_frequencies(analysis)
        .log_transition_frequencies(mc_results)
        .log_lambda_history(analysis)
        .log_outcome_summary(mc_results, analysis)
        .log_raw_arrays(mc_results)
        .finalize()
    )

    return logger, experiment_id


# ---------------------------------------------------------------------------
# SECTION 4 — Experiment registry
# ---------------------------------------------------------------------------

class ExperimentRegistry:
    """
    Tracks all experiments run in a project via a single CSV file
    (outputs/experiments/registry.csv), enabling listing, loading,
    comparing, and finding the best experiment across many runs.
    """

    def __init__(self, base_dir=None):
        self.base_dir     = base_dir or os.path.join(
            OUTPUT_DIR, 'experiments'
        )
        self.registry_path = os.path.join(
            self.base_dir, 'registry.csv'
        )
        os.makedirs(self.base_dir, exist_ok=True)

    def register(self, logger):
        """
        Add a completed experiment to the registry by merging its
        outcome_summary.csv and parameters.csv into one row appended
        to registry.csv.
        """
        outcome_path = os.path.join(
            logger.exp_dir, 'results', 'outcome_summary.csv'
        )
        params_path  = os.path.join(
            logger.exp_dir, 'parameters.csv'
        )

        if not os.path.exists(outcome_path):
            print(f"WARNING: outcome_summary.csv not found "
                  f"for {logger.experiment_id}, skipping registry.")
            return

        outcome_df = pd.read_csv(outcome_path)
        params_df  = pd.read_csv(params_path)

        merged = pd.concat(
            [params_df.reset_index(drop=True),
             outcome_df.reset_index(drop=True)],
            axis=1
        )

        merged = merged.loc[:, ~merged.columns.duplicated()]

        if os.path.exists(self.registry_path):
            existing = pd.read_csv(self.registry_path)
            updated  = pd.concat(
                [existing, merged], ignore_index=True
            )
        else:
            updated = merged

        updated.to_csv(self.registry_path, index=False)
        print(f"  Registered: {logger.experiment_id} → "
              f"{self.registry_path}")

    def load_as_dataframe(self):
        """Load the full registry as a DataFrame, or None if empty."""
        if not os.path.exists(self.registry_path):
            print("No experiments registered yet.")
            return None
        return pd.read_csv(self.registry_path)

    def list_experiments(self, n=20):
        """Print a table of the most recent N experiments."""
        df = self.load_as_dataframe()
        if df is None:
            return

        display_cols = [
            'experiment_id', 'timestamp',
            'selection_strength', 'memory_decay',
            'memory_growth', 'steepness',
            'n_simulations', 'n_steps',
            'f1_final_mean', 'f2_final_mean',
            'f1_win_rate', 'ci_overlap'
        ]

        available = [c for c in display_cols if c in df.columns]

        print(f"\nExperiment Registry — {len(df)} total experiments")
        print(f"Showing most recent {min(n, len(df))}:")
        print(df[available].tail(n).to_string(index=False))

    def find_best_experiment(self, metric='f1_final_mean',
                              minimize=False):
        """
        Find the experiment with the highest (or lowest) value of a
        given metric. Returns (experiment_id, metric_value).
        """
        df = self.load_as_dataframe()
        if df is None or metric not in df.columns:
            return None, None

        if minimize:
            idx = df[metric].idxmin()
        else:
            idx = df[metric].idxmax()

        row = df.loc[idx]
        return row['experiment_id'], row[metric]

    def load_experiment(self, experiment_id):
        """
        Load all CSV/JSON files from a specific experiment directory
        into a dict of DataFrames (plus metadata dict and matrices).
        """
        exp_dir = os.path.join(self.base_dir, experiment_id)
        if not os.path.exists(exp_dir):
            raise FileNotFoundError(
                f"Experiment {experiment_id} not found at {exp_dir}"
            )

        data = {}

        with open(os.path.join(exp_dir, 'metadata.json')) as f:
            data['metadata'] = json.load(f)

        results_dir = os.path.join(exp_dir, 'results')
        csv_files = {
            'parameters':           os.path.join(exp_dir, 'parameters.csv'),
            'cumulative_fitness':   os.path.join(results_dir, 'cumulative_fitness.csv'),
            'per_step_fitness':     os.path.join(results_dir, 'per_step_fitness.csv'),
            'state_frequencies':    os.path.join(results_dir, 'state_frequencies.csv'),
            'transition_f1':        os.path.join(results_dir, 'transition_frequencies_f1.csv'),
            'transition_f2':        os.path.join(results_dir, 'transition_frequencies_f2.csv'),
            'lambda_history':       os.path.join(results_dir, 'lambda_history.csv'),
            'outcome_summary':      os.path.join(results_dir, 'outcome_summary.csv'),
        }
        for key, path in csv_files.items():
            if os.path.exists(path):
                data[key] = pd.read_csv(path)

        matrices_dir = os.path.join(exp_dir, 'matrices')
        data['matrices'] = {}
        for name in ['f1_base', 'f2_base', 'f1_adaptation',
                     'f2_adaptation', 'f1_payoff', 'f2_payoff']:
            path = os.path.join(matrices_dir, f"{name}.csv")
            if os.path.exists(path):
                data['matrices'][name] = pd.read_csv(
                    path, index_col=0
                )

        return data

    def build_comparison_table(self, metric_cols=None):
        """
        Build a clean comparison table across all experiments with
        the most informative columns for analysis.
        """
        df = self.load_as_dataframe()
        if df is None:
            return None

        if metric_cols is None:
            metric_cols = [
                'experiment_id', 'timestamp',
                'selection_strength', 'memory_decay',
                'memory_growth', 'steepness',
                'n_simulations',
                'f1_final_mean', 'f2_final_mean',
                'fitness_diff_mean', 'ci_overlap',
                'f1_win_rate', 'f2_win_rate',
                'f1_final_lambda_mean', 'f2_final_lambda_mean',
                'f1_attack_occ', 'f1_feint_occ',
                'f2_defend_occ', 'f2_attack_occ',
                'runtime_s'
            ]

        available = [c for c in metric_cols if c in df.columns]
        return df[available].copy()


# ---------------------------------------------------------------------------
# SECTION 5 — Integration with pipeline
# ---------------------------------------------------------------------------

def log_to_registry(params, mc_results, analysis,
                    fighters_dict, random_seed,
                    n_simulations, n_steps,
                    notes="", save_raw=False,
                    base_dir=None):
    """
    Combined convenience function: log_experiment() then register the
    result. Returns (logger, experiment_id, registry).

    fighters_dict needs keys: f1_base, f2_base, f1_adaptation,
    f2_adaptation, f1_payoff, f2_payoff (all np.ndarray).

    save_raw defaults to False here to keep routine logging
    lightweight — pass save_raw=True explicitly for the full arrays.
    """
    logger, exp_id = log_experiment(
        params         = params,
        mc_results     = mc_results,
        analysis       = analysis,
        f1_base        = fighters_dict['f1_base'],
        f2_base        = fighters_dict['f2_base'],
        f1_adaptation  = fighters_dict['f1_adaptation'],
        f2_adaptation  = fighters_dict['f2_adaptation'],
        f1_payoff      = fighters_dict['f1_payoff'],
        f2_payoff      = fighters_dict['f2_payoff'],
        random_seed    = random_seed,
        n_simulations  = n_simulations,
        n_steps        = n_steps,
        notes          = notes,
        save_raw       = save_raw,
        base_dir       = base_dir
    )

    registry = ExperimentRegistry(base_dir=base_dir)
    registry.register(logger)

    return logger, exp_id, registry
