"""
Standalone verification of the fixed apply_memory_effect: proportional
redistribution instead of blind subtract-and-clip, which silently lost
probability mass whenever cost_state didn't have enough to give up.

Run before patching sparring_markov_two_agent.py to validate the algorithm
in isolation.
"""

import numpy as np

STATES = ["Attack", "Defend", "Disengage", "Feint"]
ATTACK, DEFEND, DISENGAGE, FEINT = range(4)
MAX_EXPOSURE = 10.0


def apply_memory_effect(base_probs, exposure, boost_state, cost_state, max_boost=0.25):
    exposure_ratio = exposure / MAX_EXPOSURE
    intended_boost = max_boost * exposure_ratio

    # Never take more from cost_state than it actually has.
    actual_boost = min(intended_boost, base_probs[cost_state])

    new_probs = base_probs.copy()
    new_probs[boost_state] += actual_boost
    new_probs[cost_state] -= actual_boost

    # If cost_state couldn't cover the intended boost, drain the shortfall
    # proportionally from every other state instead of losing it.
    remaining = intended_boost - actual_boost
    if remaining > 1e-8:
        donor_mask = np.ones(len(new_probs), dtype=bool)
        donor_mask[boost_state] = False
        donor_mask[cost_state] = False
        donor_probs = new_probs[donor_mask]

        if donor_probs.sum() > 1e-8:
            donor_fraction = donor_probs / donor_probs.sum()
            drain = donor_fraction * min(remaining, donor_probs.sum())
            new_probs[donor_mask] -= drain
            new_probs[boost_state] += drain.sum()

    total = new_probs.sum()
    if total > 0:
        new_probs = new_probs / total

    assert np.all(new_probs >= -1e-10), f"Negative probability detected: {new_probs}"
    new_probs = np.clip(new_probs, 0, None)

    return new_probs, exposure_ratio, intended_boost, actual_boost


def run_case(name, base_probs, exposure, boost_state, cost_state, max_boost, expect):
    base_probs = np.array(base_probs)
    out, ratio, intended, actual = apply_memory_effect(
        base_probs, exposure, boost_state, cost_state, max_boost
    )

    row_sum = out.sum()
    sum_ok = abs(row_sum - 1.0) < 1e-9
    no_negatives = np.all(out >= -1e-10)
    expect_ok = expect(out, actual, intended) if expect else True
    passed = sum_ok and no_negatives and expect_ok

    print(f"--- {name} ---")
    print(f"Input probabilities:  {dict(zip(STATES, base_probs.round(4)))}")
    print(f"Exposure ratio used:  {ratio:.4f}  (exposure={exposure}, max_boost={max_boost})")
    print(f"Intended boost:       {intended:.4f}")
    print(f"Actual boost applied: {actual:.4f}")
    print(f"Output probabilities: {dict(zip(STATES, out.round(6)))}")
    print(f"Row sum:              {row_sum:.6f}")
    print(f"Result: {'PASS' if passed else 'FAIL'}")
    print()
    return passed


def main():
    results = []

    # 1. Normal case: cost_state has plenty of room, boost applies in full.
    results.append(run_case(
        "Case 1: Normal case (plenty of room in cost_state)",
        base_probs=[0.40, 0.10, 0.20, 0.30],
        exposure=5.0, boost_state=DEFEND, cost_state=FEINT, max_boost=0.25,
        expect=lambda out, actual, intended: abs(actual - intended) < 1e-9,
    ))

    # 2. Constrained case: cost_state can only cover part of the boost.
    results.append(run_case(
        "Case 2: Constrained case (cost_state nearly empty)",
        base_probs=[0.45, 0.30, 0.20, 0.05],
        exposure=10.0, boost_state=DEFEND, cost_state=FEINT, max_boost=0.25,
        expect=lambda out, actual, intended: abs(actual - 0.05) < 1e-9 and (intended - actual) > 0.19,
    ))

    # 3. Fully suppressed case: cost_state AND one donor already at 0.
    results.append(run_case(
        "Case 3: Fully suppressed (cost_state=0 and a donor=0)",
        base_probs=[0.00, 0.90, 0.10, 0.00],
        exposure=10.0, boost_state=DEFEND, cost_state=FEINT, max_boost=0.25,
        expect=lambda out, actual, intended: abs(actual) < 1e-9,
    ))

    # 4. Extreme case: boost_state already dominant, cost_state empty, donors thin.
    results.append(run_case(
        "Case 4: Extreme case (boost_state=0.95, cost_state=0, thin donors)",
        base_probs=[0.95, 0.03, 0.02, 0.00],
        exposure=10.0, boost_state=ATTACK, cost_state=FEINT, max_boost=0.25,
        expect=lambda out, actual, intended: abs(actual) < 1e-9,
    ))

    print("=" * 60)
    print(f"{sum(results)}/{len(results)} cases passed")
    if not all(results):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
