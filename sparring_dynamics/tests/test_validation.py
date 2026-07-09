"""Tests for sparring_dynamics.analysis.validation."""
import numpy as np

from sparring_dynamics.analysis.validation import (
    ObservedSparringData, ValidationReport, MultiOpponentValidation,
    transition_frequency_error, state_occupancy_error, compute_rmse,
    jensen_shannon_divergence, compute_row_jsd,
    kl_divergence, compute_row_kl,
    generate_placeholder_observed_data,
)
from sparring_dynamics.config import N_STATES, STATES


def _make_observed_with_fixed_stats(matrix, occupancy, fighter_name="CJ", opponent_name="Test"):
    """
    Build an ObservedSparringData with a trivial sequence (just to
    satisfy the constructor) then overwrite its computed matrix/
    occupancy directly — lets tests set up an exact target distribution
    without fighting the Laplace-smoothing bias in the sequence-based
    estimator.
    """
    obs = ObservedSparringData(fighter_name, [[STATES[0], STATES[1]]], opponent_name)
    obs.transition_matrix = matrix.copy()
    obs.state_occupancy = occupancy.copy()
    return obs


def _random_stochastic_matrix(rng):
    rows = [rng.dirichlet(np.ones(N_STATES)) for _ in range(N_STATES)]
    return np.array(rows)


def _random_distribution(rng):
    return rng.dirichlet(np.ones(N_STATES))


def test_perfect_match():
    """When simulated == observed, all error metrics should be 0, JSD/KL 0, score 1.0."""
    rng = np.random.default_rng(1)
    matrix = _random_stochastic_matrix(rng)
    occupancy = _random_distribution(rng)

    obs = _make_observed_with_fixed_stats(matrix, occupancy)
    report = ValidationReport("CJ", "Test", matrix, occupancy, obs, n_simulations=100)

    assert report.tf_error['mean_error'] < 1e-9
    assert report.tf_error['max_error'] < 1e-9
    assert report.occ_error['mean_error'] < 1e-9
    assert report.rmse_result['rmse'] < 1e-9
    assert report.jsd_occ['jsd'] < 1e-9
    assert report.jsd_rows['mean_jsd'] < 1e-9
    assert abs(report.kl_occ['kl_obs_sim']) < 1e-9
    assert abs(report.overall_score() - 1.0) < 1e-6
    print("  test_perfect_match: PASS")


def test_uniform_vs_uniform():
    """Both uniform distributions — JSD and KL should be 0."""
    occ = np.ones(N_STATES) / N_STATES
    jsd_result = jensen_shannon_divergence(occ, occ)
    kl_result = kl_divergence(occ, occ)
    assert jsd_result['jsd'] < 1e-9
    assert abs(kl_result['kl_obs_sim']) < 1e-9
    assert abs(kl_result['kl_sim_obs']) < 1e-9
    print("  test_uniform_vs_uniform: PASS")


def test_jsd_bounded():
    """JSD should always be in [0, 1] for any two distributions. Test 100 random pairs."""
    rng = np.random.default_rng(2)
    for _ in range(100):
        p = _random_distribution(rng)
        q = _random_distribution(rng)
        jsd = jensen_shannon_divergence(p, q)['jsd']
        assert -1e-9 <= jsd <= 1.0 + 1e-9, f"JSD out of bounds: {jsd}"
    print("  test_jsd_bounded: PASS (100/100 pairs)")


def test_kl_asymmetry():
    """KL(P||Q) != KL(Q||P) in general."""
    # A reversed pair like [0.7,0.1,0.1,0.1] vs [0.1,0.1,0.1,0.7] is
    # symmetric under KL by coincidence (the two middle log(1)=0 terms
    # cancel identically either direction) — use a genuinely lopsided
    # pair instead so the asymmetry actually shows up.
    p = np.array([0.85, 0.07, 0.05, 0.03])
    q = np.array([0.25, 0.25, 0.25, 0.25])
    result = kl_divergence(p, q)  # kl_obs_sim = KL(p||q), kl_sim_obs = KL(q||p)
    assert not np.isclose(result['kl_obs_sim'], result['kl_sim_obs'])
    print(f"  test_kl_asymmetry: PASS (KL(obs||sim)={result['kl_obs_sim']:.4f} != "
          f"KL(sim||obs)={result['kl_sim_obs']:.4f})")


def test_rmse_matches_mae_uniform_errors():
    """For a matrix where all errors are equal constant c, RMSE == MAE == c."""
    c = 0.05
    observed = np.zeros((N_STATES, N_STATES))
    simulated = np.full((N_STATES, N_STATES), c)

    mae_result = transition_frequency_error(simulated, observed)
    rmse_result = compute_rmse(simulated, observed)

    assert abs(mae_result['mean_error'] - c) < 1e-9
    assert abs(rmse_result['rmse'] - c) < 1e-9
    print("  test_rmse_matches_mae_uniform_errors: PASS")


def test_overall_score_range():
    """overall_score() should always be in [0, 1]. Test 20 random pairs."""
    rng = np.random.default_rng(3)
    for _ in range(20):
        sim_matrix = _random_stochastic_matrix(rng)
        obs_matrix = _random_stochastic_matrix(rng)
        sim_occ = _random_distribution(rng)
        obs_occ = _random_distribution(rng)

        obs = _make_observed_with_fixed_stats(obs_matrix, obs_occ)
        report = ValidationReport("CJ", "Test", sim_matrix, sim_occ, obs, n_simulations=100)
        score = report.overall_score()
        assert -1e-9 <= score <= 1.0 + 1e-9, f"Score out of bounds: {score}"
    print("  test_overall_score_range: PASS (20/20 pairs)")


def test_multi_opponent_comparison():
    """Create 3 opponents with different fit quality; verify best/worst are identified correctly."""
    rng = np.random.default_rng(4)
    sim_matrix = _random_stochastic_matrix(rng)
    sim_occ = _random_distribution(rng)

    # Best: identical to simulated.
    best_obs = _make_observed_with_fixed_stats(sim_matrix, sim_occ, opponent_name="Best")

    # Moderate: shifted a bit.
    moderate_matrix = np.clip(sim_matrix + 0.1, 1e-6, None)
    moderate_matrix = moderate_matrix / moderate_matrix.sum(axis=1, keepdims=True)
    moderate_occ = np.clip(sim_occ + 0.1, 1e-6, None)
    moderate_occ = moderate_occ / moderate_occ.sum()
    moderate_obs = _make_observed_with_fixed_stats(moderate_matrix, moderate_occ, opponent_name="Moderate")

    # Worst: maximally different (reverse rows).
    worst_matrix = sim_matrix[::-1].copy()
    worst_occ = np.array([sim_occ[3], sim_occ[2], sim_occ[1], sim_occ[0]])
    worst_obs = _make_observed_with_fixed_stats(worst_matrix, worst_occ, opponent_name="Worst")

    reports = [
        ValidationReport("CJ", "Best", sim_matrix, sim_occ, best_obs, n_simulations=100),
        ValidationReport("CJ", "Moderate", sim_matrix, sim_occ, moderate_obs, n_simulations=100),
        ValidationReport("CJ", "Worst", sim_matrix, sim_occ, worst_obs, n_simulations=100),
    ]
    multi = MultiOpponentValidation(reports)

    best = max(multi.reports, key=lambda r: r.overall_score())
    worst = min(multi.reports, key=lambda r: r.overall_score())

    assert best.opponent_name == "Best"
    assert worst.opponent_name == "Worst"
    assert best.overall_score() >= reports[1].overall_score() >= worst.overall_score()
    print("  test_multi_opponent_comparison: PASS")


def test_placeholder_data_generation():
    """generate_placeholder_observed_data() returns valid transition matrix and occupancy."""
    for style in ('aggressive', 'counter', 'balanced'):
        obs = generate_placeholder_observed_data(
            "CJ", "Test", n_sequences=6, seq_length=20, style=style, seed=7
        )
        assert np.allclose(obs.transition_matrix.sum(axis=1), 1.0)
        assert abs(obs.state_occupancy.sum() - 1.0) < 1e-9
        assert obs.n_sequences == 6
    print("  test_placeholder_data_generation: PASS")


def test_to_dict_keys():
    """ValidationReport.to_dict() should contain all required keys for MultiOpponentValidation."""
    rng = np.random.default_rng(5)
    sim_matrix = _random_stochastic_matrix(rng)
    sim_occ = _random_distribution(rng)
    obs = _make_observed_with_fixed_stats(sim_matrix, sim_occ)
    report = ValidationReport("CJ", "Test", sim_matrix, sim_occ, obs, n_simulations=250)

    required_keys = {
        'fighter', 'opponent', 'n_simulations', 'n_observed',
        'mae', 'max_ae', 'rmse', 'occ_mae',
        'jsd_occupancy', 'jsd_rows_mean', 'kl_obs_sim', 'kl_sim_obs',
        'overall_score', 'quality',
    }
    d = report.to_dict()
    assert required_keys.issubset(set(d.keys())), f"Missing keys: {required_keys - set(d.keys())}"
    print("  test_to_dict_keys: PASS")


def run_all():
    print("test_validation.py")
    test_perfect_match()
    test_uniform_vs_uniform()
    test_jsd_bounded()
    test_kl_asymmetry()
    test_rmse_matches_mae_uniform_errors()
    test_overall_score_range()
    test_multi_opponent_comparison()
    test_placeholder_data_generation()
    test_to_dict_keys()


if __name__ == "__main__":
    run_all()
    print("ALL test_validation.py TESTS PASSED")
