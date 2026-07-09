"""
Extend estimate_payoffs.py to handle richer annotated exchange data:
points scored (0-4) and Double/None outcomes, instead of simple
win/loss/draw. Replaces the binary win-rate reward with a
points-weighted reward system.
"""

import csv
import os

import numpy as np
import pandas as pd

states = ['Attack', 'Defend', 'Disengage', 'Feint']
STATE_INDEX = {s: i for i, s in enumerate(states)}
n = len(states)

VALID_WINNERS = ('F1', 'F2', 'Double', 'None')


def load_exchanges_from_csv(filepath):
    """
    Load annotated exchange data from CSV.

    Validates:
    - All f1_state and f2_state values are in STATE_INDEX
    - winner is one of: 'F1', 'F2', 'Double', 'None'
    - f1_points and f2_points are non-negative integers
    - f1_points == 0 when winner == 'F2' (and vice versa,
      except for Double where both can be > 0)

    Returns list of dicts with keys:
    f1_state, f2_state, winner, f1_points, f2_points

    Raises ValueError with clear message if validation fails.
    """
    exchanges = []
    with open(filepath, newline='') as f:
        reader = csv.DictReader(f)
        for row_num, row in enumerate(reader, start=2):  # header is row 1
            f1_state = row['f1_state']
            f2_state = row['f2_state']
            winner = row['winner']

            if f1_state not in STATE_INDEX:
                raise ValueError(f"Row {row_num}: unknown f1_state '{f1_state}'")
            if f2_state not in STATE_INDEX:
                raise ValueError(f"Row {row_num}: unknown f2_state '{f2_state}'")
            if winner not in VALID_WINNERS:
                raise ValueError(f"Row {row_num}: unknown winner '{winner}'")

            try:
                f1_points = int(row['f1_points'])
                f2_points = int(row['f2_points'])
            except ValueError:
                raise ValueError(f"Row {row_num}: f1_points/f2_points must be integers")

            if f1_points < 0 or f2_points < 0:
                raise ValueError(f"Row {row_num}: points must be non-negative")

            if winner == 'F2' and f1_points != 0:
                raise ValueError(
                    f"Row {row_num}: winner is 'F2' but f1_points={f1_points} (expected 0)")
            if winner == 'F1' and f2_points != 0:
                raise ValueError(
                    f"Row {row_num}: winner is 'F1' but f2_points={f2_points} (expected 0)")
            if winner == 'None' and (f1_points != 0 or f2_points != 0):
                raise ValueError(
                    f"Row {row_num}: winner is 'None' but points were scored "
                    f"(f1_points={f1_points}, f2_points={f2_points})")

            exchanges.append({
                'f1_state': f1_state,
                'f2_state': f2_state,
                'winner': winner,
                'f1_points': f1_points,
                'f2_points': f2_points,
            })
    return exchanges


def compute_rewards(exchanges, n, max_points=4):
    """
    Compute expected reward for each (f1_state, f2_state) pair
    using points scored rather than binary win/loss.

    F1 reward contribution:
    - winner == 'F1':     f1_points / max_points
    - winner == 'F2':     0.0
    - winner == 'Double': f1_points / max_points * 0.5
    - winner == 'None':   0.0

    F2 reward contribution (symmetric).

    Returns f1_reward_sum, f2_reward_sum, totals (all n x n arrays)
    """
    f1_reward_sum = np.zeros((n, n), dtype=float)
    f2_reward_sum = np.zeros((n, n), dtype=float)
    totals = np.zeros((n, n), dtype=float)

    for ex in exchanges:
        i = STATE_INDEX[ex['f1_state']]
        j = STATE_INDEX[ex['f2_state']]
        winner = ex['winner']

        if winner == 'F1':
            f1_contribution = ex['f1_points'] / max_points
            f2_contribution = 0.0
        elif winner == 'F2':
            f1_contribution = 0.0
            f2_contribution = ex['f2_points'] / max_points
        elif winner == 'Double':
            f1_contribution = ex['f1_points'] / max_points * 0.5
            f2_contribution = ex['f2_points'] / max_points * 0.5
        else:  # None
            f1_contribution = 0.0
            f2_contribution = 0.0

        f1_reward_sum[i, j] += f1_contribution
        f2_reward_sum[i, j] += f2_contribution
        totals[i, j] += 1

    return f1_reward_sum, f2_reward_sum, totals


def smooth_rewards(reward_sum, totals, alpha=0.5):
    """
    Smooth reward estimates to handle sparse state pairs.

    Expected reward = (reward_sum + alpha * 0.5) / (totals + alpha)

    - Zero observations: expected reward = 0.5 (neutral)
    - Few observations: pulled toward 0.5
    - Many observations: dominated by actual data

    alpha=0.5 recommended — Jeffreys prior for proportion estimation.
    """
    numerator = reward_sum + alpha * 0.5
    denominator = totals + alpha
    return numerator / denominator


def apply_confidence_weighting(smoothed_rewards, totals, min_obs=5):
    """
    Blend smoothed estimate toward 0.5 for sparse pairs.

    min_obs=5 here (higher than v1) because points-weighted
    estimates need more data to stabilize than binary win/loss.
    """
    confidence = np.minimum(totals / min_obs, 1.0)
    return confidence * smoothed_rewards + (1 - confidence) * 0.5


def normalize_to_unit_interval(matrix):
    """
    Min-max normalize so min=0.0 and max=1.0.
    Degenerate case (all identical): return 0.5 uniformly.
    """
    min_val = matrix.min()
    max_val = matrix.max()
    if max_val - min_val < 1e-8:
        return np.full_like(matrix, 0.5)
    return (matrix - min_val) / (max_val - min_val)


def build_payoff_matrices_v2(filepath, alpha=0.5, min_obs=5,
                              max_points=4,
                              use_confidence_weighting=True):
    """
    Full pipeline from CSV path to F1_PAYOFF and F2_PAYOFF.

    1. Load and validate CSV
    2. Compute points-weighted rewards
    3. Apply Laplace smoothing
    4. Apply confidence weighting
    5. Normalize to [0,1]
    6. Return matrices + diagnostics
    """
    exchanges = load_exchanges_from_csv(filepath)

    f1_raw, f2_raw, totals = compute_rewards(exchanges, n, max_points)

    f1_smooth = smooth_rewards(f1_raw, totals, alpha)
    f2_smooth = smooth_rewards(f2_raw, totals, alpha)

    if use_confidence_weighting:
        f1_weighted = apply_confidence_weighting(f1_smooth, totals, min_obs)
        f2_weighted = apply_confidence_weighting(f2_smooth, totals, min_obs)
    else:
        f1_weighted = f1_smooth
        f2_weighted = f2_smooth

    f1_final = normalize_to_unit_interval(f1_weighted)
    f2_final = normalize_to_unit_interval(f2_weighted)

    return f1_final, f2_final, totals, exchanges


def print_diagnostics(f1_final, f2_final, totals, exchanges, min_obs=5):
    """
    Print comprehensive diagnostics about the loaded exchanges and
    resulting payoff matrices.
    """
    print(f"\n{'='*60}")
    print(f"1. Total exchanges loaded: {len(exchanges)}")

    print("\n2. Outcome distribution:")
    outcome_counts = {}
    for ex in exchanges:
        outcome_counts[ex['winner']] = outcome_counts.get(ex['winner'], 0) + 1
    for outcome in VALID_WINNERS:
        count = outcome_counts.get(outcome, 0)
        pct = count / len(exchanges) * 100
        print(f"   {outcome:8s}: {count:3d}  ({pct:5.1f}%)")

    print("\n3. Average points per exchange:")
    f1_points = [ex['f1_points'] for ex in exchanges]
    f2_points = [ex['f2_points'] for ex in exchanges]
    print(f"   Fighter 1 (CJ):              {np.mean(f1_points):.3f}")
    print(f"   Fighter 2 (Counter-Puncher): {np.mean(f2_points):.3f}")

    print("\n4. Observation count matrix (totals):")
    print(pd.DataFrame(totals, index=states, columns=states).to_string())

    print("\n5. F1_PAYOFF matrix:")
    print(pd.DataFrame(f1_final, index=states, columns=states).round(4).to_string())

    print("\n6. F2_PAYOFF matrix:")
    print(pd.DataFrame(f2_final, index=states, columns=states).round(4).to_string())

    print(f"\n7. Sparsity report (fewer than min_obs={min_obs} observations, "
          f"confidence-weighted toward 0.5):")
    sparse_pairs = []
    for i in range(n):
        for j in range(n):
            if totals[i, j] < min_obs:
                sparse_pairs.append((states[i], states[j], int(totals[i, j])))
    if sparse_pairs:
        for f1_s, f2_s, count in sparse_pairs:
            print(f"   {f1_s:10s} vs {f2_s:10s}: {count} observation(s)")
    else:
        print("   None — every state pair meets min_obs.")

    print("\n8. Highest payoff state pair:")
    f1_max_idx = np.unravel_index(f1_final.argmax(), f1_final.shape)
    f2_max_idx = np.unravel_index(f2_final.argmax(), f2_final.shape)
    print(f"   Fighter 1 (CJ): {states[f1_max_idx[0]]} vs {states[f1_max_idx[1]]} "
          f"= {f1_final[f1_max_idx]:.4f} — CJ's most rewarding state pairing observed.")
    print(f"   Fighter 2 (Counter-Puncher): {states[f2_max_idx[0]]} vs {states[f2_max_idx[1]]} "
          f"= {f2_final[f2_max_idx]:.4f} — Fighter 2's most rewarding state pairing observed.")

    print("\n9. Lowest payoff state pair:")
    f1_min_idx = np.unravel_index(f1_final.argmin(), f1_final.shape)
    f2_min_idx = np.unravel_index(f2_final.argmin(), f2_final.shape)
    print(f"   Fighter 1 (CJ): {states[f1_min_idx[0]]} vs {states[f1_min_idx[1]]} "
          f"= {f1_final[f1_min_idx]:.4f} — CJ's least rewarding state pairing observed.")
    print(f"   Fighter 2 (Counter-Puncher): {states[f2_min_idx[0]]} vs {states[f2_min_idx[1]]} "
          f"= {f2_final[f2_min_idx]:.4f} — Fighter 2's least rewarding state pairing observed.")


def validate_payoff_matrices(f1, f2):
    """
    Confirm:
    - All values in [0, 1]
    - No NaN or Inf
    - Matrices are 4x4
    Print PASS/FAIL for each check.
    """
    print(f"\n{'='*60}")
    print("Validating payoff matrices:")
    for name, matrix in [("F1_PAYOFF", f1), ("F2_PAYOFF", f2)]:
        in_range = np.all((matrix >= 0) & (matrix <= 1))
        no_nan = not np.any(np.isnan(matrix))
        no_inf = not np.any(np.isinf(matrix))
        right_shape = matrix.shape == (n, n)
        status = "PASS" if (in_range and no_nan and no_inf and right_shape) else "FAIL"
        print(f"\n  {name}:")
        print(f"    All values in [0,1]: {in_range}")
        print(f"    No NaN values:       {no_nan}")
        print(f"    No Inf values:       {no_inf}")
        print(f"    Shape is 4x4:        {right_shape}")
        print(f"    Overall: {status}")


def compare_with_handcrafted(f1_learned, f2_learned, f1_handcrafted, f2_handcrafted):
    """
    Element-wise absolute difference. Print largest discrepancies with
    interpretation. Flag any cell where difference > 0.2 as
    'significant revision'.
    """
    print(f"\n{'='*60}")
    for name, learned, handcrafted in [
        ("Fighter 1 (CJ)", f1_learned, f1_handcrafted),
        ("Fighter 2 (Counter-Puncher)", f2_learned, f2_handcrafted),
    ]:
        diff = np.abs(learned - handcrafted)
        print(f"\n{name} — Absolute Difference (Learned vs Hand-Crafted):")
        print(pd.DataFrame(diff, index=states, columns=states).round(4).to_string())

        max_idx = np.unravel_index(diff.argmax(), diff.shape)
        print(f"\n  Largest discrepancy: {states[max_idx[0]]} vs {states[max_idx[1]]} "
              f"= {diff[max_idx]:.4f}")
        print(f"  Hand-crafted: {handcrafted[max_idx]:.4f}   Learned: {learned[max_idx]:.4f}")
        print(f"  Mean absolute difference: {diff.mean():.4f}")

        significant = [(states[i], states[j], diff[i, j])
                       for i in range(n) for j in range(n) if diff[i, j] > 0.2]
        if significant:
            print("  Significant revisions (|diff| > 0.2):")
            for f1_s, f2_s, d in sorted(significant, key=lambda x: -x[2]):
                print(f"    {f1_s} vs {f2_s}: {d:.4f}")
        else:
            print("  No cells exceed the 0.2 significant-revision threshold.")


def generate_placeholder_csv(filepath, n_exchanges=60):
    """
    Generate a realistic placeholder CSV of annotated exchanges,
    consistent with the fighting styles already modeled in the
    simulator (CJ: aggressive feint-heavy blitzer; Fighter 2: patient
    counter-puncher). Structured, not randomly sampled, so the stated
    win counts and point averages are hit exactly/closely.
    """
    rows = [
        # Feint vs Defend: 10 exchanges, F1 wins 7, avg f1_points ~2.5
        ('Feint', 'Defend', 'F1', 3, 0),
        ('Feint', 'Defend', 'F1', 2, 0),
        ('Feint', 'Defend', 'F1', 3, 0),
        ('Feint', 'Defend', 'F1', 2, 0),
        ('Feint', 'Defend', 'F1', 3, 0),
        ('Feint', 'Defend', 'F1', 3, 0),
        ('Feint', 'Defend', 'F1', 2, 0),
        ('Feint', 'Defend', 'F2', 0, 1),
        ('Feint', 'Defend', 'F2', 0, 2),
        ('Feint', 'Defend', 'None', 0, 0),

        # Attack vs Defend: 10 exchanges, F1 wins 6, avg f1_points ~1.8
        ('Attack', 'Defend', 'F1', 2, 0),
        ('Attack', 'Defend', 'F1', 1, 0),
        ('Attack', 'Defend', 'F1', 2, 0),
        ('Attack', 'Defend', 'F1', 2, 0),
        ('Attack', 'Defend', 'F1', 1, 0),
        ('Attack', 'Defend', 'F1', 3, 0),
        ('Attack', 'Defend', 'F2', 0, 1),
        ('Attack', 'Defend', 'F2', 0, 1),
        ('Attack', 'Defend', 'F2', 0, 2),
        ('Attack', 'Defend', 'Double', 1, 1),

        # Defend vs Attack: 8 exchanges, F1 wins 5 (counter), avg f1_points ~1.0
        ('Defend', 'Attack', 'F1', 1, 0),
        ('Defend', 'Attack', 'F1', 1, 0),
        ('Defend', 'Attack', 'F1', 1, 0),
        ('Defend', 'Attack', 'F1', 1, 0),
        ('Defend', 'Attack', 'F1', 1, 0),
        ('Defend', 'Attack', 'F2', 0, 2),
        ('Defend', 'Attack', 'F2', 0, 1),
        ('Defend', 'Attack', 'None', 0, 0),

        # Feint vs Attack: 7 exchanges, split 4-3 F2, avg points ~1.5 each
        ('Feint', 'Attack', 'F2', 0, 2),
        ('Feint', 'Attack', 'F2', 0, 1),
        ('Feint', 'Attack', 'F2', 0, 2),
        ('Feint', 'Attack', 'F2', 0, 1),
        ('Feint', 'Attack', 'F1', 2, 0),
        ('Feint', 'Attack', 'F1', 1, 0),
        ('Feint', 'Attack', 'F1', 2, 0),

        # Attack vs Feint: 6 exchanges, F1 wins 4, avg f1_points = 2.0
        ('Attack', 'Feint', 'F1', 2, 0),
        ('Attack', 'Feint', 'F1', 2, 0),
        ('Attack', 'Feint', 'F1', 2, 0),
        ('Attack', 'Feint', 'F1', 2, 0),
        ('Attack', 'Feint', 'F2', 0, 1),
        ('Attack', 'Feint', 'Double', 1, 1),

        # Disengage vs Attack: 6 exchanges, F2 wins 4, avg f2_points ~1.5
        ('Disengage', 'Attack', 'F2', 0, 2),
        ('Disengage', 'Attack', 'F2', 0, 1),
        ('Disengage', 'Attack', 'F2', 0, 2),
        ('Disengage', 'Attack', 'F2', 0, 1),
        ('Disengage', 'Attack', 'F1', 1, 0),
        ('Disengage', 'Attack', 'None', 0, 0),

        # Attack vs Attack: 5 exchanges, mixed Double/F1/F2
        ('Attack', 'Attack', 'Double', 1, 1),
        ('Attack', 'Attack', 'Double', 2, 2),
        ('Attack', 'Attack', 'F1', 2, 0),
        ('Attack', 'Attack', 'F1', 1, 0),
        ('Attack', 'Attack', 'F2', 0, 2),

        # Disengage vs Disengage: 4 exchanges, mostly None
        ('Disengage', 'Disengage', 'None', 0, 0),
        ('Disengage', 'Disengage', 'None', 0, 0),
        ('Disengage', 'Disengage', 'None', 0, 0),
        ('Disengage', 'Disengage', 'Double', 1, 1),

        # Remaining 4: other state pairs with realistic outcomes
        ('Feint', 'Feint', 'Double', 1, 1),
        ('Disengage', 'Defend', 'F1', 1, 0),
        ('Defend', 'Defend', 'None', 0, 0),
        ('Defend', 'Feint', 'F1', 1, 0),
    ]

    assert len(rows) == n_exchanges, (
        f"Placeholder row count {len(rows)} does not match n_exchanges={n_exchanges}")

    with open(filepath, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['f1_state', 'f2_state', 'winner', 'f1_points', 'f2_points'])
        writer.writerows(rows)


def create_annotation_template(filepath, n_rows=20):
    """
    Generate an empty CSV template with correct headers, instructions,
    and one example row, ready for manual annotation of real match
    footage. Instruction/placeholder lines are '#'-commented so they
    are easy to spot and remove; only real data rows should be
    uncommented before loading with load_exchanges_from_csv.
    """
    lines = [
        "f1_state,f2_state,winner,f1_points,f2_points",
        "# Instructions:",
        "# f1_state/f2_state: Attack, Defend, Disengage, or Feint",
        "# winner: F1, F2, Double, or None",
        "# f1_points: 0-4 (head kick=4, body kick=3, punch=1 in WT rules)",
        "# f2_points: 0-4",
        "# Double: both fighters score on same exchange",
        "# None: no points scored (reset, out of bounds, no contact)",
        "Attack,Defend,F1,2,0",
    ]
    for i in range(2, n_rows + 1):
        lines.append(f"# row {i}: add your annotated exchange here")

    with open(filepath, 'w') as f:
        f.write("\n".join(lines) + "\n")


if __name__ == "__main__":
    F1_HANDCRAFTED = np.array([
        [0.2, 0.8, 0.5, 0.9],
        [0.6, 0.1, 0.3, 0.4],
        [0.3, 0.4, 0.2, 0.6],
        [0.5, 0.7, 0.4, 0.3]
    ])

    F2_HANDCRAFTED = np.array([
        [0.7, 0.2, 0.8, 0.3],
        [0.8, 0.1, 0.2, 0.6],
        [0.2, 0.5, 0.3, 0.4],
        [0.4, 0.3, 0.7, 0.2]
    ])

    # Generate placeholder CSV for testing
    placeholder_path = 'placeholder_exchanges.csv'
    generate_placeholder_csv(placeholder_path, n_exchanges=60)
    print(f"Generated placeholder CSV: {placeholder_path}")

    # Generate annotation template for future real data
    create_annotation_template('annotation_template.csv')
    print("Generated annotation_template.csv for real match coding")

    # Run full pipeline
    f1_final, f2_final, totals, exchanges = build_payoff_matrices_v2(
        placeholder_path, alpha=0.5, min_obs=5, max_points=4
    )

    print_diagnostics(f1_final, f2_final, totals, exchanges, min_obs=5)
    validate_payoff_matrices(f1_final, f2_final)
    compare_with_handcrafted(f1_final, f2_final, F1_HANDCRAFTED, F2_HANDCRAFTED)

    print(f"\n{'='*60}")
    print("DROP-IN REPLACEMENT FOR sparring_markov_two_agent.py:")
    print("\n# Estimated from points-weighted observed exchange outcomes")
    print("F1_PAYOFF_MATRIX = np.array([")
    for row in f1_final:
        print(f"    {[round(float(x), 4) for x in row]},")
    print("])")
    print("\nF2_PAYOFF_MATRIX = np.array([")
    for row in f2_final:
        print(f"    {[round(float(x), 4) for x in row]},")
    print("])")
