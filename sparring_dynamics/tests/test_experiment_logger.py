"""Tests for sparring_dynamics.analysis.experiment_logger."""
import os
import shutil
import tempfile

import numpy as np

from sparring_dynamics.analysis.experiment_logger import (
    generate_experiment_id, build_experiment_metadata,
    ExperimentLogger, log_experiment,
    ExperimentRegistry, log_to_registry,
)
from sparring_dynamics.analysis.monte_carlo import run_monte_carlo, analyze_monte_carlo
from sparring_dynamics.analysis.statistics import compute_statistics, compute_state_occupancy
from sparring_dynamics.simulation.fighter import Fighter
from sparring_dynamics.simulation.match import SparringMatch
from sparring_dynamics.config import (
    N_STATES, DEFAULT_START_STATE,
    F1_BASE_DEFAULT, F2_BASE_DEFAULT,
    F1_ADAPTATION_DEFAULT, F2_ADAPTATION_DEFAULT,
    F1_PAYOFF_DEFAULT, F2_PAYOFF_DEFAULT,
)


def _make_match(selection_strength=1.0):
    f1 = Fighter.from_matrices(
        "CJ", F1_BASE_DEFAULT, F1_ADAPTATION_DEFAULT, F1_PAYOFF_DEFAULT,
        color='crimson',
    )
    f2 = Fighter.from_matrices(
        "Counter-Puncher", F2_BASE_DEFAULT, F2_ADAPTATION_DEFAULT, F2_PAYOFF_DEFAULT,
        color='steelblue',
    )
    match = SparringMatch(f1, f2, selection_strength=selection_strength)
    return f1, f2, match


def _run_small_experiment(base_dir, selection_strength=1.0, random_seed=42,
                           n_simulations=5, n_steps=20, save_raw=True):
    f1, f2, match = _make_match(selection_strength)
    mc_results = run_monte_carlo(
        match, n_simulations=n_simulations, n_steps=n_steps,
        start_state=DEFAULT_START_STATE, random_seed=random_seed,
    )
    analysis = analyze_monte_carlo(mc_results)

    params = {'selection_strength': selection_strength}
    fighters_dict = {
        'f1_base': F1_BASE_DEFAULT, 'f2_base': F2_BASE_DEFAULT,
        'f1_adaptation': F1_ADAPTATION_DEFAULT, 'f2_adaptation': F2_ADAPTATION_DEFAULT,
        'f1_payoff': F1_PAYOFF_DEFAULT, 'f2_payoff': F2_PAYOFF_DEFAULT,
    }

    logger, exp_id = log_experiment(
        params=params, mc_results=mc_results, analysis=analysis,
        f1_base=F1_BASE_DEFAULT, f2_base=F2_BASE_DEFAULT,
        f1_adaptation=F1_ADAPTATION_DEFAULT, f2_adaptation=F2_ADAPTATION_DEFAULT,
        f1_payoff=F1_PAYOFF_DEFAULT, f2_payoff=F2_PAYOFF_DEFAULT,
        random_seed=random_seed, n_simulations=n_simulations, n_steps=n_steps,
        save_raw=save_raw, base_dir=base_dir,
    )
    return logger, exp_id, mc_results, analysis, params, fighters_dict


def test_experiment_id_deterministic():
    """Same params + seed -> same hash portion of the ID (timestamp differs)."""
    params = {'selection_strength': 1.0, 'memory_decay': 0.9}
    id1 = generate_experiment_id(params, random_seed=42)
    id2 = generate_experiment_id(params, random_seed=42)
    hash1 = id1.split('_')[1]
    hash2 = id2.split('_')[1]
    assert hash1 == hash2

    id3 = generate_experiment_id({'selection_strength': 2.0, 'memory_decay': 0.9}, random_seed=42)
    hash3 = id3.split('_')[1]
    assert hash3 != hash1
    print("  test_experiment_id_deterministic: PASS")


def test_logger_creates_all_files():
    tmpdir = tempfile.mkdtemp()
    try:
        logger, exp_id, mc_results, analysis, params, fighters_dict = _run_small_experiment(tmpdir)

        expected = [
            'metadata.json', 'parameters.csv', 'manifest.json',
            os.path.join('matrices', 'f1_base.csv'),
            os.path.join('matrices', 'f2_base.csv'),
            os.path.join('matrices', 'f1_adaptation.csv'),
            os.path.join('matrices', 'f2_adaptation.csv'),
            os.path.join('matrices', 'f1_payoff.csv'),
            os.path.join('matrices', 'f2_payoff.csv'),
            os.path.join('results', 'cumulative_fitness.csv'),
            os.path.join('results', 'per_step_fitness.csv'),
            os.path.join('results', 'state_frequencies.csv'),
            os.path.join('results', 'transition_frequencies_f1.csv'),
            os.path.join('results', 'transition_frequencies_f2.csv'),
            os.path.join('results', 'lambda_history.csv'),
            os.path.join('results', 'outcome_summary.csv'),
        ]
        for rel_path in expected:
            full_path = os.path.join(logger.exp_dir, rel_path)
            assert os.path.exists(full_path), f"Missing expected file: {rel_path}"
        print("  test_logger_creates_all_files: PASS")
    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)


def test_matrices_saved_with_correct_shape():
    tmpdir = tempfile.mkdtemp()
    try:
        logger, exp_id, *_ = _run_small_experiment(tmpdir)
        import pandas as pd
        df = pd.read_csv(os.path.join(logger.matrices_dir, 'f1_base.csv'), index_col=0)
        # N_STATES data columns + row_sum
        assert df.shape[0] == N_STATES
        assert 'row_sum' in df.columns
        assert df.shape[1] == N_STATES + 1
        np.testing.assert_allclose(df['row_sum'].values, 1.0, atol=1e-6)
        print("  test_matrices_saved_with_correct_shape: PASS")
    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)


def test_transition_frequencies_row_normalized():
    tmpdir = tempfile.mkdtemp()
    try:
        logger, exp_id, *_ = _run_small_experiment(tmpdir, n_simulations=8, n_steps=30)
        import pandas as pd
        for fname in ('transition_frequencies_f1.csv', 'transition_frequencies_f2.csv'):
            df = pd.read_csv(os.path.join(logger.results_dir, fname), index_col=0)
            assert 'row_sum' in df.columns
            assert 'observation_count' in df.columns
            np.testing.assert_allclose(df['row_sum'].values, 1.0, atol=1e-6)
            assert (df['observation_count'].values >= 0).all()
        print("  test_transition_frequencies_row_normalized: PASS")
    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)


def test_outcome_summary_has_all_columns():
    tmpdir = tempfile.mkdtemp()
    try:
        logger, exp_id, *_ = _run_small_experiment(tmpdir)
        import pandas as pd
        df = pd.read_csv(os.path.join(logger.results_dir, 'outcome_summary.csv'))
        required_cols = {
            'experiment_id', 'f1_final_mean', 'f2_final_mean',
            'fitness_diff_mean', 'ci_overlap', 'f1_win_rate', 'f2_win_rate',
            'tie_rate', 'f1_final_lambda_mean', 'f2_final_lambda_mean',
            'f1_attack_occ', 'f2_attack_occ', 'runtime_s',
        }
        assert required_cols.issubset(set(df.columns)), f"Missing: {required_cols - set(df.columns)}"
        assert len(df) == 1
        print("  test_outcome_summary_has_all_columns: PASS")
    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)


def test_registry_append():
    tmpdir = tempfile.mkdtemp()
    try:
        registry = ExperimentRegistry(base_dir=tmpdir)
        for strength in (0.5, 1.5):
            logger, exp_id, mc_results, analysis, params, fighters_dict = _run_small_experiment(
                tmpdir, selection_strength=strength, random_seed=int(strength * 100)
            )
            registry.register(logger)

        df = registry.load_as_dataframe()
        assert df is not None
        assert len(df) == 2
        print("  test_registry_append: PASS")
    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)


def test_load_experiment_returns_all_keys():
    tmpdir = tempfile.mkdtemp()
    try:
        logger, exp_id, *_ = _run_small_experiment(tmpdir)
        registry = ExperimentRegistry(base_dir=tmpdir)
        data = registry.load_experiment(exp_id)

        required_keys = {
            'metadata', 'parameters', 'cumulative_fitness', 'per_step_fitness',
            'state_frequencies', 'transition_f1', 'transition_f2',
            'lambda_history', 'outcome_summary', 'matrices',
        }
        assert required_keys.issubset(set(data.keys())), f"Missing: {required_keys - set(data.keys())}"

        matrix_keys = {'f1_base', 'f2_base', 'f1_adaptation', 'f2_adaptation', 'f1_payoff', 'f2_payoff'}
        assert matrix_keys.issubset(set(data['matrices'].keys()))
        print("  test_load_experiment_returns_all_keys: PASS")
    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)


def test_find_best_experiment():
    tmpdir = tempfile.mkdtemp()
    try:
        registry = ExperimentRegistry(base_dir=tmpdir)
        exp_ids = []
        for strength in (0.5, 1.0, 2.0):
            logger, exp_id, *_ = _run_small_experiment(
                tmpdir, selection_strength=strength, random_seed=int(strength * 100)
            )
            registry.register(logger)
            exp_ids.append(exp_id)

        best_id, best_val = registry.find_best_experiment(metric='f1_final_mean', minimize=False)
        worst_id, worst_val = registry.find_best_experiment(metric='f1_final_mean', minimize=True)

        assert best_id in exp_ids
        assert worst_id in exp_ids
        assert best_val >= worst_val
        print("  test_find_best_experiment: PASS")
    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)


def test_raw_arrays_shape():
    tmpdir = tempfile.mkdtemp()
    try:
        n_sim, n_steps = 6, 15
        logger, exp_id, *_ = _run_small_experiment(
            tmpdir, n_simulations=n_sim, n_steps=n_steps, save_raw=True
        )
        import pandas as pd
        for fname in ('f1_cumulative_all_runs.csv', 'f2_cumulative_all_runs.csv',
                      'f1_states_all_runs.csv', 'f2_states_all_runs.csv'):
            path = os.path.join(logger.raw_dir, fname)
            assert os.path.exists(path)
            df = pd.read_csv(path, index_col=0)
            assert df.shape == (n_sim, n_steps)
        print("  test_raw_arrays_shape: PASS")
    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)


def test_reproducibility():
    """Same params + seed logged twice (different base_dirs) -> identical outcome_summary values."""
    tmpdir1 = tempfile.mkdtemp()
    tmpdir2 = tempfile.mkdtemp()
    try:
        logger1, exp_id1, *_ = _run_small_experiment(tmpdir1, selection_strength=1.0, random_seed=7)
        logger2, exp_id2, *_ = _run_small_experiment(tmpdir2, selection_strength=1.0, random_seed=7)

        assert exp_id1.split('_')[1] == exp_id2.split('_')[1]

        import pandas as pd
        df1 = pd.read_csv(os.path.join(logger1.results_dir, 'outcome_summary.csv'))
        df2 = pd.read_csv(os.path.join(logger2.results_dir, 'outcome_summary.csv'))

        numeric_cols = [c for c in df1.columns if c not in ('experiment_id', 'runtime_s')]
        for col in numeric_cols:
            np.testing.assert_allclose(df1[col].values, df2[col].values, atol=1e-9)
        print("  test_reproducibility: PASS")
    finally:
        shutil.rmtree(tmpdir1, ignore_errors=True)
        shutil.rmtree(tmpdir2, ignore_errors=True)


def run_all():
    print("test_experiment_logger.py")
    test_experiment_id_deterministic()
    test_logger_creates_all_files()
    test_matrices_saved_with_correct_shape()
    test_transition_frequencies_row_normalized()
    test_outcome_summary_has_all_columns()
    test_registry_append()
    test_load_experiment_returns_all_keys()
    test_find_best_experiment()
    test_raw_arrays_shape()
    test_reproducibility()


if __name__ == "__main__":
    run_all()
    print("ALL test_experiment_logger.py TESTS PASSED")
