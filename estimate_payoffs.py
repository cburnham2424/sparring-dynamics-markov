"""
Estimate F1_PAYOFF and F2_PAYOFF matrices for sparring_markov_two_agent.py
from observed sparring exchange outcomes, instead of hand-assigned values.
"""

import csv

import numpy as np
import pandas as pd

states = ['Attack', 'Defend', 'Disengage', 'Feint']
STATE_INDEX = {s: i for i, s in enumerate(states)}
n = len(states)

# EXAMPLE OBSERVED EXCHANGES (placeholder — replace with real data later)
exchanges = [
    # CJ attacks into counter-puncher's defense — common exchange
    {'f1_state': 'Attack',    'f2_state': 'Defend',    'winner': 'F1'},
    {'f1_state': 'Attack',    'f2_state': 'Defend',    'winner': 'F2'},
    {'f1_state': 'Attack',    'f2_state': 'Defend',    'winner': 'F1'},
    {'f1_state': 'Attack',    'f2_state': 'Defend',    'winner': 'F1'},
    {'f1_state': 'Attack',    'f2_state': 'Defend',    'winner': 'F2'},

    # CJ feints into counter-puncher's defense — high CJ payoff
    {'f1_state': 'Feint',     'f2_state': 'Defend',    'winner': 'F1'},
    {'f1_state': 'Feint',     'f2_state': 'Defend',    'winner': 'F1'},
    {'f1_state': 'Feint',     'f2_state': 'Defend',    'winner': 'F1'},
    {'f1_state': 'Feint',     'f2_state': 'Defend',    'winner': 'F2'},
    {'f1_state': 'Feint',     'f2_state': 'Defend',    'winner': 'F1'},

    # Counter-puncher attacks into CJ's feint — dangerous for CJ
    {'f1_state': 'Feint',     'f2_state': 'Attack',    'winner': 'F2'},
    {'f1_state': 'Feint',     'f2_state': 'Attack',    'winner': 'F1'},
    {'f1_state': 'Feint',     'f2_state': 'Attack',    'winner': 'F2'},
    {'f1_state': 'Feint',     'f2_state': 'Attack',    'winner': 'F2'},
    {'f1_state': 'Feint',     'f2_state': 'Attack',    'winner': 'F1'},

    # CJ attacks into counter-puncher's feint (pump kick) — CJ reads it
    {'f1_state': 'Attack',    'f2_state': 'Feint',     'winner': 'F1'},
    {'f1_state': 'Attack',    'f2_state': 'Feint',     'winner': 'F1'},
    {'f1_state': 'Attack',    'f2_state': 'Feint',     'winner': 'F2'},
    {'f1_state': 'Attack',    'f2_state': 'Feint',     'winner': 'F1'},
    {'f1_state': 'Attack',    'f2_state': 'Feint',     'winner': 'F1'},

    # CJ disengages while counter-puncher attacks — counter-puncher wins
    {'f1_state': 'Disengage', 'f2_state': 'Attack',    'winner': 'F2'},
    {'f1_state': 'Disengage', 'f2_state': 'Attack',    'winner': 'F2'},
    {'f1_state': 'Disengage', 'f2_state': 'Attack',    'winner': 'F1'},
    {'f1_state': 'Disengage', 'f2_state': 'Attack',    'winner': 'F2'},
    {'f1_state': 'Disengage', 'f2_state': 'Attack',    'winner': 'F2'},

    # CJ defends into counter-puncher's attack — CJ counters well
    {'f1_state': 'Defend',    'f2_state': 'Attack',    'winner': 'F1'},
    {'f1_state': 'Defend',    'f2_state': 'Attack',    'winner': 'F1'},
    {'f1_state': 'Defend',    'f2_state': 'Attack',    'winner': 'F2'},
    {'f1_state': 'Defend',    'f2_state': 'Attack',    'winner': 'F1'},
    {'f1_state': 'Defend',    'f2_state': 'Attack',    'winner': 'Draw'},

    # Both disengage — low stakes, usually draw
    {'f1_state': 'Disengage', 'f2_state': 'Disengage', 'winner': 'Draw'},
    {'f1_state': 'Disengage', 'f2_state': 'Disengage', 'winner': 'Draw'},
    {'f1_state': 'Disengage', 'f2_state': 'Disengage', 'winner': 'F1'},
    {'f1_state': 'Disengage', 'f2_state': 'Disengage', 'winner': 'Draw'},
    {'f1_state': 'Disengage', 'f2_state': 'Disengage', 'winner': 'F2'},

    # CJ feints into counter-puncher's feint — unpredictable
    {'f1_state': 'Feint',     'f2_state': 'Feint',     'winner': 'F1'},
    {'f1_state': 'Feint',     'f2_state': 'Feint',     'winner': 'F2'},
    {'f1_state': 'Feint',     'f2_state': 'Feint',     'winner': 'Draw'},
    {'f1_state': 'Feint',     'f2_state': 'Feint',     'winner': 'F1'},
    {'f1_state': 'Feint',     'f2_state': 'Feint',     'winner': 'F2'},

    # Counter-puncher defends CJ's disengage — low payoff both sides
    {'f1_state': 'Disengage', 'f2_state': 'Defend',    'winner': 'Draw'},
    {'f1_state': 'Disengage', 'f2_state': 'Defend',    'winner': 'F1'},
    {'f1_state': 'Disengage', 'f2_state': 'Defend',    'winner': 'Draw'},
    {'f1_state': 'Disengage', 'f2_state': 'Feint',     'winner': 'F1'},
    {'f1_state': 'Disengage', 'f2_state': 'Feint',     'winner': 'F2'},
    {'f1_state': 'Defend',    'f2_state': 'Defend',    'winner': 'Draw'},
    {'f1_state': 'Defend',    'f2_state': 'Feint',     'winner': 'F1'},
    {'f1_state': 'Defend',    'f2_state': 'Disengage', 'winner': 'F2'},
    {'f1_state': 'Attack',    'f2_state': 'Attack',    'winner': 'F1'},
    {'f1_state': 'Attack',    'f2_state': 'Disengage', 'winner': 'F1'},
]


def accumulate_outcomes(exchanges, n):
    """
    For each (f1_state, f2_state) pair, count:
    - f1_wins: number of exchanges F1 won
    - f2_wins: number of exchanges F2 won
    - draws: number of draws
    - total: total exchanges observed for this pair

    Returns three n×n matrices: f1_wins, f2_wins, totals
    """
    f1_wins  = np.zeros((n, n), dtype=float)
    f2_wins  = np.zeros((n, n), dtype=float)
    draws    = np.zeros((n, n), dtype=float)
    totals   = np.zeros((n, n), dtype=float)

    for ex in exchanges:
        i = STATE_INDEX[ex['f1_state']]
        j = STATE_INDEX[ex['f2_state']]
        totals[i, j] += 1
        if ex['winner'] == 'F1':
            f1_wins[i, j] += 1
        elif ex['winner'] == 'F2':
            f2_wins[i, j] += 1
        else:
            draws[i, j] += 0.5  # draws award 0.5 to each fighter
            f1_wins[i, j] += 0.5
            f2_wins[i, j] += 0.5

    return f1_wins, f2_wins, totals


def estimate_raw_payoffs(wins, totals, alpha=0.5):
    """
    Expected payoff for fighter at (i,j) = wins[i,j] / totals[i,j]

    Apply Laplace smoothing to handle unseen state pairs:
    smoothed_payoff = (wins + alpha) / (totals + 2*alpha)

    2*alpha in denominator because each exchange has two possible
    outcomes (win or loss), so smoothing adds alpha pseudocounts
    to both sides.

    alpha=0.5 (Jeffreys prior) is recommended here since payoffs
    are proportions — more principled than alpha=1.0 for this case.
    """
    smoothed_wins   = wins   + alpha
    smoothed_totals = totals + 2 * alpha
    raw_payoffs = smoothed_wins / smoothed_totals
    return raw_payoffs


def normalize_payoffs(raw_payoffs):
    """
    Min-max normalize so the lowest payoff = 0.0 and
    highest payoff = 1.0 across the entire matrix.

    If all values are identical (degenerate case),
    return a matrix of 0.5 uniformly.
    """
    min_val = raw_payoffs.min()
    max_val = raw_payoffs.max()

    if max_val - min_val < 1e-8:
        return np.full_like(raw_payoffs, 0.5)

    return (raw_payoffs - min_val) / (max_val - min_val)


def confidence_weighted_payoffs(normalized, totals, min_obs=3, alpha=0.5):
    """
    For state pairs with very few observations, blend the
    estimated payoff toward 0.5 (neutral/uncertain) based
    on how many observations exist.

    confidence = min(totals[i,j] / min_obs, 1.0)
    final[i,j] = confidence * normalized[i,j] + (1-confidence) * 0.5

    This means:
    - 0 observations: payoff = 0.5 (maximum uncertainty)
    - min_obs observations: payoff = fully estimated value
    - >min_obs observations: payoff = fully estimated value

    min_obs=3 means we need at least 3 exchanges to fully trust
    an estimated payoff.
    """
    confidence = np.minimum(totals / min_obs, 1.0)
    return confidence * normalized + (1 - confidence) * 0.5


def build_payoff_matrices(exchanges, alpha=0.5, min_obs=3,
                           use_confidence_weighting=True):
    """
    Full pipeline: exchanges → estimated F1_PAYOFF and F2_PAYOFF
    ready to drop into sparring_markov_two_agent.py
    """
    f1_wins, f2_wins, totals = accumulate_outcomes(exchanges, n)

    f1_raw = estimate_raw_payoffs(f1_wins, totals, alpha)
    f2_raw = estimate_raw_payoffs(f2_wins, totals, alpha)

    f1_norm = normalize_payoffs(f1_raw)
    f2_norm = normalize_payoffs(f2_raw)

    if use_confidence_weighting:
        f1_final = confidence_weighted_payoffs(f1_norm, totals, min_obs)
        f2_final = confidence_weighted_payoffs(f2_norm, totals, min_obs)
    else:
        f1_final = f1_norm
        f2_final = f2_norm

    return f1_final, f2_final, totals


def validate_payoffs(matrix, name):
    """
    Confirm all values are in [0, 1] and no NaN/Inf values exist.
    """
    print(f"\nValidating {name}:")
    in_range = np.all((matrix >= 0) & (matrix <= 1))
    no_nan   = not np.any(np.isnan(matrix))
    no_inf   = not np.any(np.isinf(matrix))
    status   = "PASS" if (in_range and no_nan and no_inf) else "FAIL"
    print(f"  All values in [0,1]: {in_range}")
    print(f"  No NaN values:       {no_nan}")
    print(f"  No Inf values:       {no_inf}")
    print(f"  Overall: {status}")
    return status == "PASS"


def compare_payoffs(learned, handcrafted, name):
    diff = np.abs(learned - handcrafted)
    print(f"\n{name} — Absolute Difference (Learned vs Hand-Crafted):")
    df = pd.DataFrame(diff, index=states, columns=states)
    print(df.round(4).to_string())
    max_idx = np.unravel_index(diff.argmax(), diff.shape)
    print(f"\n  Largest discrepancy: "
          f"{states[max_idx[0]]} vs {states[max_idx[1]]} = "
          f"{diff[max_idx]:.4f}")
    print(f"  Mean absolute difference: {diff.mean():.4f}")


def load_exchanges_from_csv(filepath):
    """
    Load exchange data from a CSV with columns:
    f1_state, f2_state, winner

    Example CSV content:
    f1_state,f2_state,winner
    Attack,Defend,F1
    Feint,Defend,F1
    Attack,Feint,F2
    """
    exchanges = []
    with open(filepath, newline='') as f:
        reader = csv.DictReader(f)
        for row in reader:
            assert row['f1_state'] in STATE_INDEX, \
                f"Unknown state: {row['f1_state']}"
            assert row['f2_state'] in STATE_INDEX, \
                f"Unknown state: {row['f2_state']}"
            assert row['winner'] in ('F1', 'F2', 'Draw'), \
                f"Unknown winner: {row['winner']}"
            exchanges.append(row)
    return exchanges


if __name__ == "__main__":
    F1_HANDCRAFTED_PAYOFF = np.array([
        [0.2, 0.8, 0.5, 0.9],
        [0.6, 0.1, 0.3, 0.4],
        [0.3, 0.4, 0.2, 0.6],
        [0.5, 0.7, 0.4, 0.3]
    ])

    F2_HANDCRAFTED_PAYOFF = np.array([
        [0.7, 0.2, 0.8, 0.3],
        [0.8, 0.1, 0.2, 0.6],
        [0.2, 0.5, 0.3, 0.4],
        [0.4, 0.3, 0.7, 0.2]
    ])

    f1_final, f2_final, totals = build_payoff_matrices(
        exchanges, alpha=0.5, min_obs=3,
        use_confidence_weighting=True
    )

    print("Observation counts per state pair:")
    print(pd.DataFrame(totals, index=states, columns=states).to_string())

    print("\nEstimated F1 Payoff Matrix:")
    print(pd.DataFrame(f1_final, index=states,
                        columns=states).round(4).to_string())

    print("\nEstimated F2 Payoff Matrix:")
    print(pd.DataFrame(f2_final, index=states,
                        columns=states).round(4).to_string())

    validate_payoffs(f1_final, "F1_PAYOFF_MATRIX")
    validate_payoffs(f2_final, "F2_PAYOFF_MATRIX")

    compare_payoffs(f1_final, F1_HANDCRAFTED_PAYOFF, "Fighter 1 (CJ)")
    compare_payoffs(f2_final, F2_HANDCRAFTED_PAYOFF,
                    "Fighter 2 (Counter-Puncher)")

    print("\n" + "="*60)
    print("DROP-IN REPLACEMENT FOR sparring_markov_two_agent.py:")
    print("\n# Estimated from observed exchange outcomes")
    print("F1_PAYOFF_MATRIX = np.array([")
    for row in f1_final:
        print(f"    {[round(float(x), 4) for x in row]},")
    print("])")
    print("\nF2_PAYOFF_MATRIX = np.array([")
    for row in f2_final:
        print(f"    {[round(float(x), 4) for x in row]},")
    print("])")
