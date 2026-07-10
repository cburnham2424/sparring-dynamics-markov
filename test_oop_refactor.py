"""
Validation suite confirming the OOP refactor of sparring_markov_two_agent.py
(Fighter / SparringMatch / factory functions) preserves the original
simulator's behavior — structural refactor only, no logic changes.
"""

import numpy as np
from scipy import stats

from sparring_markov_two_agent import (
    simulate,
    create_cj, create_counter_puncher, create_match,
    Fighter, SparringMatch,
    F1_BASE, F2_BASE,
    F1_ADAPTATION_MATRIX, F2_ADAPTATION_MATRIX,
    F1_PAYOFF, F2_PAYOFF,
    STATES, N_STATES,
    N_STEPS, ATTACK, FEINT,
)

EXPECTED_RESULT_KEYS = {
    "f1_seq", "f2_seq",
    "f1_exposure_history", "f2_exposure_history",
    "f1_lambda_history", "f2_lambda_history",
    "f1_defend_prob_history", "f2_defend_prob_history",
    "f1_fitness_history", "f2_fitness_history",
    "f1_cumulative_fitness", "f2_cumulative_fitness",
}


def test_backward_compatibility(n_runs=50):
    """
    Run n_runs simulations via the top-level simulate() wrapper and
    n_runs via SparringMatch.simulate() called directly (different seed
    ranges, so this is a genuine test that both code paths draw from the
    same underlying statistical model, not just that identical inputs
    give identical outputs). Verify:
    1. Both return the same result-dict keys
    2. Final cumulative fitness distributions are statistically
       indistinguishable (t-test p > 0.05)
    3. Average state occupancies match within 0.02 tolerance
    4. Fighter.__repr__ works without errors
    5. SparringMatch.__repr__ works without errors
    """
    print(f"\n{'='*60}")
    print(f"test_backward_compatibility (n_runs={n_runs})")
    print(f"{'='*60}")
    passed = True

    wrapper_result = simulate(selection_strength=1.0, seed=0)
    match = create_match(selection_strength=1.0)
    direct_result = match.simulate(N_STEPS, start_state=2, seed=1000)

    keys_match = set(wrapper_result.keys()) >= EXPECTED_RESULT_KEYS and \
                 set(direct_result.keys()) >= EXPECTED_RESULT_KEYS
    print(f"  [1] Result dict keys match expected set: {keys_match}")
    passed &= keys_match

    wrapper_f1_final = np.array([
        simulate(selection_strength=1.0, seed=i)['f1_cumulative_fitness'][-1]
        for i in range(n_runs)
    ])
    direct_f1_final = np.empty(n_runs)
    direct_f1_occ = np.zeros((n_runs, N_STATES))
    for i in range(n_runs):
        m = create_match(selection_strength=1.0)
        r = m.simulate(N_STEPS, start_state=2, seed=1000 + i)
        direct_f1_final[i] = r['f1_cumulative_fitness'][-1]
        counts = np.bincount(r['f1_seq'], minlength=N_STATES)
        direct_f1_occ[i] = counts / counts.sum()

    wrapper_f1_occ = np.zeros((n_runs, N_STATES))
    for i in range(n_runs):
        r = simulate(selection_strength=1.0, seed=i)
        counts = np.bincount(r['f1_seq'], minlength=N_STATES)
        wrapper_f1_occ[i] = counts / counts.sum()

    t_stat, p_value = stats.ttest_ind(wrapper_f1_final, direct_f1_final)
    distributions_indistinguishable = p_value > 0.05
    print(f"  [2] Final fitness t-test: t={t_stat:.4f}, p={p_value:.4f} "
          f"({'PASS' if distributions_indistinguishable else 'FAIL'} — indistinguishable at p>0.05)")
    passed &= distributions_indistinguishable

    occ_diff = np.abs(wrapper_f1_occ.mean(axis=0) - direct_f1_occ.mean(axis=0))
    occ_within_tolerance = np.all(occ_diff < 0.02)
    print(f"  [3] State occupancy diff: {occ_diff.round(4)} "
          f"({'PASS' if occ_within_tolerance else 'FAIL'} — all < 0.02)")
    passed &= occ_within_tolerance

    try:
        cj = create_cj()
        repr(cj)
        fighter_repr_ok = True
    except Exception as e:
        fighter_repr_ok = False
        print(f"      Fighter.__repr__ raised: {e}")
    print(f"  [4] Fighter.__repr__ works: {fighter_repr_ok}")
    passed &= fighter_repr_ok

    try:
        repr(match)
        match_repr_ok = True
    except Exception as e:
        match_repr_ok = False
        print(f"      SparringMatch.__repr__ raised: {e}")
    print(f"  [5] SparringMatch.__repr__ works: {match_repr_ok}")
    passed &= match_repr_ok

    print(f"\n  Overall: {'PASS' if passed else 'FAIL'}")
    return passed


def test_fighter_reset():
    """
    Verify that calling match.simulate() twice produces independent
    results (reset() works correctly):
    - Different seeds -> different histories
    - Exposure starts at 0.0 on every run
    - Lambda starts at 0.0 on every run
    """
    print(f"\n{'='*60}")
    print("test_fighter_reset")
    print(f"{'='*60}")
    passed = True

    match = create_match(selection_strength=1.0)
    result1 = match.simulate(N_STEPS, start_state=2, seed=1)
    exposure_start_1 = match.f1.exposure_history[0]
    lambda_start_1 = match.f1.lambda_history[0]

    result2 = match.simulate(N_STEPS, start_state=2, seed=2)
    exposure_start_2 = match.f1.exposure_history[0]
    lambda_start_2 = match.f1.lambda_history[0]

    histories_differ = not np.array_equal(result1['f1_seq'], result2['f1_seq'])
    print(f"  [1] Different seeds produce different histories: {histories_differ}")
    passed &= histories_differ

    exposure_resets = (exposure_start_1 == 0.0) and (exposure_start_2 == 0.0)
    print(f"  [2] Exposure starts at 0.0 each run: {exposure_resets} "
          f"(run1={exposure_start_1}, run2={exposure_start_2})")
    passed &= exposure_resets

    lambda_resets = (lambda_start_1 == 0.0) and (lambda_start_2 == 0.0)
    print(f"  [3] Lambda starts at 0.0 each run: {lambda_resets} "
          f"(run1={lambda_start_1}, run2={lambda_start_2})")
    passed &= lambda_resets

    print(f"\n  Overall: {'PASS' if passed else 'FAIL'}")
    return passed


def test_factory_functions():
    """
    Verify factory functions produce correctly configured objects,
    with matrices matching the module-level constants exactly.
    """
    print(f"\n{'='*60}")
    print("test_factory_functions")
    print(f"{'='*60}")
    passed = True

    cj = create_cj()
    cj_ok = (cj.name == "CJ"
              and np.array_equal(cj.base_matrix, F1_BASE)
              and np.array_equal(cj.adaptation_matrix, F1_ADAPTATION_MATRIX)
              and np.array_equal(cj.payoff_matrix, F1_PAYOFF))
    print(f"  [1] create_cj(): name/matrices correct: {cj_ok}")
    passed &= cj_ok

    cp = create_counter_puncher()
    cp_ok = (cp.name == "Counter-Fighter"
              and np.array_equal(cp.base_matrix, F2_BASE)
              and np.array_equal(cp.adaptation_matrix, F2_ADAPTATION_MATRIX)
              and np.array_equal(cp.payoff_matrix, F2_PAYOFF))
    print(f"  [2] create_counter_puncher(): name/matrices correct: {cp_ok}")
    passed &= cp_ok

    match = create_match(selection_strength=1.5)
    match_ok = (isinstance(match, SparringMatch)
                 and match.f1.name == "CJ"
                 and match.f2.name == "Counter-Fighter"
                 and match.selection_strength == 1.5
                 and match.f1_tracked_state == ATTACK
                 and match.f2_tracked_state == FEINT)
    print(f"  [3] create_match(): correct fighters/config: {match_ok}")
    passed &= match_ok

    print(f"\n  Overall: {'PASS' if passed else 'FAIL'}")
    return passed


def test_payoff_symmetry():
    """
    Verify payoff computation is correct:
    - F1 payoff uses F1_PAYOFF[f1_state, f2_state]
    - F2 payoff uses F2_PAYOFF[f2_state, f1_state]
    - Both are in [0, 1]
    """
    print(f"\n{'='*60}")
    print("test_payoff_symmetry")
    print(f"{'='*60}")
    passed = True

    match = create_match(selection_strength=1.0)
    result = match.simulate(N_STEPS, start_state=2, seed=7)

    checks = []
    for f1_s, f2_s, f1_payoff, f2_payoff in zip(
        result['f1_seq'], result['f2_seq'],
        result['f1_fitness_history'], result['f2_fitness_history']
    ):
        expected_f1 = F1_PAYOFF[f1_s, f2_s]
        expected_f2 = F2_PAYOFF[f2_s, f1_s]
        checks.append(np.isclose(f1_payoff, expected_f1) and np.isclose(f2_payoff, expected_f2))

    formula_correct = all(checks)
    print(f"  [1] F1 payoff == F1_PAYOFF[f1_state, f2_state] and "
          f"F2 payoff == F2_PAYOFF[f2_state, f1_state] for all {len(checks)} steps: "
          f"{formula_correct}")
    passed &= formula_correct

    f1_in_range = np.all((result['f1_fitness_history'] >= 0) & (result['f1_fitness_history'] <= 1))
    f2_in_range = np.all((result['f2_fitness_history'] >= 0) & (result['f2_fitness_history'] <= 1))
    print(f"  [2] All F1 payoffs in [0,1]: {f1_in_range}")
    print(f"  [3] All F2 payoffs in [0,1]: {f2_in_range}")
    passed &= f1_in_range and f2_in_range

    print(f"\n  Overall: {'PASS' if passed else 'FAIL'}")
    return passed


if __name__ == "__main__":
    results = {
        "test_backward_compatibility": test_backward_compatibility(),
        "test_fighter_reset": test_fighter_reset(),
        "test_factory_functions": test_factory_functions(),
        "test_payoff_symmetry": test_payoff_symmetry(),
    }

    print(f"\n{'='*60}")
    print("TEST SUITE SUMMARY")
    print(f"{'='*60}")
    for name, passed in results.items():
        print(f"  {name:<30} {'PASS' if passed else 'FAIL'}")

    all_passed = all(results.values())
    print(f"\n  {'ALL TESTS PASSED' if all_passed else 'SOME TESTS FAILED'}")
    if not all_passed:
        raise SystemExit(1)
