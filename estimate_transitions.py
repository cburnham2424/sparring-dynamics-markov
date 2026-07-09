"""
Estimate F1_BASE and F2_BASE transition matrices for
sparring_markov_two_agent.py from observed state sequences, instead of
hand-assigned values.
"""

import numpy as np
import pandas as pd

states = ['Attack', 'Defend', 'Disengage', 'Feint']
STATE_INDEX = {s: i for i, s in enumerate(states)}
n = len(states)  # 4

# EXAMPLE OBSERVED SEQUENCES (placeholder — replace with real data later)
# f1_observed reflects CJ's aggressive feint-heavy style.
# f2_observed reflects the counter-puncher's defend-heavy reactive style.
# These are placeholder sequences — structured to be consistent with the
# hand-crafted matrices but replaceable with real observed data later.
f1_observed = [
    ['Attack', 'Feint', 'Attack', 'Disengage', 'Feint', 'Attack',
     'Defend', 'Attack', 'Feint', 'Feint', 'Attack', 'Disengage',
     'Attack', 'Feint', 'Attack', 'Attack', 'Disengage', 'Feint',
     'Attack', 'Defend', 'Disengage', 'Attack', 'Feint', 'Attack'],
    ['Feint', 'Attack', 'Feint', 'Attack', 'Disengage', 'Feint',
     'Attack', 'Disengage', 'Attack', 'Feint', 'Attack', 'Defend',
     'Feint', 'Attack', 'Feint', 'Disengage', 'Attack', 'Feint']
]

f2_observed = [
    ['Defend', 'Attack', 'Defend', 'Feint', 'Defend', 'Attack',
     'Defend', 'Disengage', 'Feint', 'Defend', 'Attack', 'Defend',
     'Feint', 'Attack', 'Defend', 'Disengage', 'Defend', 'Attack',
     'Defend', 'Feint', 'Defend', 'Attack', 'Defend', 'Disengage'],
    ['Defend', 'Feint', 'Defend', 'Attack', 'Defend', 'Feint',
     'Disengage', 'Defend', 'Attack', 'Defend', 'Feint', 'Defend',
     'Attack', 'Disengage', 'Defend', 'Attack', 'Defend', 'Feint']
]


def count_transitions(sequences, n_states):
    """
    Count transitions across all observed sequences.
    sequences: list of lists of state name strings
    Returns: n_states x n_states count matrix
    """
    counts = np.zeros((n_states, n_states), dtype=float)
    for seq in sequences:
        for t in range(len(seq) - 1):
            i = STATE_INDEX[seq[t]]
            j = STATE_INDEX[seq[t + 1]]
            counts[i, j] += 1
    return counts


def smooth_counts(counts, alpha=1.0):
    """
    Add alpha to every cell before normalizing.
    alpha=1.0 is standard Laplace smoothing.
    alpha=0.5 is Jeffreys prior (gentler smoothing).
    For sparse data (few observations), alpha=1.0 prevents zero probabilities.
    For rich data (many observations), alpha=0.1 preserves observed structure.
    """
    return counts + alpha


def counts_to_matrix(counts, alpha=1.0):
    """
    Apply Laplace smoothing and normalize each row to sum to 1.0.
    Rows with zero total (unseen states) get uniform distribution.
    """
    smoothed = smooth_counts(counts, alpha)
    row_sums = smoothed.sum(axis=1, keepdims=True)
    # Guard against zero rows (unseen states in data)
    row_sums = np.where(row_sums == 0, 1, row_sums)
    matrix = smoothed / row_sums
    return matrix


def validate_matrix(matrix, name):
    """
    Check that all rows sum to 1.0 and no negative values exist.
    Print a clear pass/fail for each row.
    """
    print(f"\nValidating {name}:")
    all_pass = True
    for i, state in enumerate(states):
        row_sum = matrix[i].sum()
        has_negatives = np.any(matrix[i] < 0)
        status = "PASS" if abs(row_sum - 1.0) < 1e-8 and not has_negatives else "FAIL"
        if status == "FAIL":
            all_pass = False
        print(f"  Row {state:12s}: sum={row_sum:.8f} | negatives={has_negatives} | {status}")
    print(f"  Overall: {'ALL PASS' if all_pass else 'FAILURES DETECTED'}")
    return all_pass


def estimate_transition_matrix(sequences, alpha=1.0, name="Fighter"):
    counts = count_transitions(sequences, n)
    matrix = counts_to_matrix(counts, alpha)

    print(f"\n{'='*50}")
    print(f"{name} — Raw Transition Counts:")
    print(pd.DataFrame(counts, index=states, columns=states).to_string())

    print(f"\n{name} — Estimated Transition Matrix (alpha={alpha}):")
    df = pd.DataFrame(matrix, index=states, columns=states)
    print(df.round(4).to_string())

    validate_matrix(matrix, name)
    return matrix


def compare_matrices(learned, handcrafted, name):
    """
    Compute element-wise absolute difference between learned and
    hand-crafted matrices. Print a summary showing which transitions
    differ most and what this means in sparring terms.
    """
    diff = np.abs(learned - handcrafted)
    print(f"\n{name} — Absolute Difference (Learned vs Hand-Crafted):")
    df_diff = pd.DataFrame(diff, index=states, columns=states)
    print(df_diff.round(4).to_string())

    max_diff_idx = np.unravel_index(diff.argmax(), diff.shape)
    max_diff_val = diff[max_diff_idx]
    from_state = states[max_diff_idx[0]]
    to_state = states[max_diff_idx[1]]
    print(f"\n  Largest discrepancy: {from_state} → {to_state} = {max_diff_val:.4f}")
    print(f"  Hand-crafted: {handcrafted[max_diff_idx]:.4f}")
    print(f"  Learned:      {learned[max_diff_idx]:.4f}")

    mean_diff = diff.mean()
    print(f"  Mean absolute difference across all transitions: {mean_diff:.4f}")
    if mean_diff < 0.05:
        print(f"  Interpretation: Learned matrix closely matches hand-crafted design.")
    elif mean_diff < 0.15:
        print(f"  Interpretation: Moderate discrepancy — observed sequences suggest "
              f"some transition probabilities should be adjusted.")
    else:
        print(f"  Interpretation: Large discrepancy — observed data tells a "
              f"meaningfully different story than hand-crafted assumptions.")


def load_sequences_from_file(filepath):
    """
    Load observed state sequences from a plain text file where each line
    is a comma-separated state sequence, e.g.:

    Attack,Feint,Attack,Disengage,Feint,Attack
    Defend,Attack,Defend,Feint,Defend,Attack

    This is the future interface for plugging in real observed match data.
    """
    sequences = []
    with open(filepath) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            seq = line.split(',')
            for state in seq:
                assert state in STATE_INDEX, f"Unknown state: {state}"
            sequences.append(seq)
    return sequences


if __name__ == "__main__":
    # Hand-crafted matrices from the main simulator, for comparison
    F1_HANDCRAFTED = np.array([
        [0.25, 0.15, 0.35, 0.25],
        [0.45, 0.10, 0.25, 0.20],
        [0.30, 0.10, 0.20, 0.40],
        [0.60, 0.08, 0.17, 0.15]
    ])

    F2_HANDCRAFTED = np.array([
        [0.15, 0.30, 0.35, 0.20],
        [0.55, 0.15, 0.20, 0.10],
        [0.15, 0.30, 0.25, 0.30],
        [0.40, 0.25, 0.20, 0.15]
    ])

    # Estimate from observed sequences
    f1_learned = estimate_transition_matrix(f1_observed, alpha=1.0,
                                             name="Fighter 1 (CJ)")
    f2_learned = estimate_transition_matrix(f2_observed, alpha=1.0,
                                             name="Fighter 2 (Counter-Puncher)")

    # Compare learned vs hand-crafted
    compare_matrices(f1_learned, F1_HANDCRAFTED, "Fighter 1 (CJ)")
    compare_matrices(f2_learned, F2_HANDCRAFTED, "Fighter 2 (Counter-Puncher)")

    print("\n" + "="*50)
    print("DROP-IN REPLACEMENT FOR sparring_markov_two_agent.py:")
    print("\nF1_BASE = np.array([")
    for row in f1_learned:
        print(f"    {[round(float(x), 4) for x in row]},")
    print("])")
    print("\nF2_BASE = np.array([")
    for row in f2_learned:
        print(f"    {[round(float(x), 4) for x in row]},")
    print("])")
