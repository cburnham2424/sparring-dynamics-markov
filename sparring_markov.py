"""
Markov chain model of a traditional point sparring exchange.

States: Attack, Defend, Disengage, Feint

Clinch is intentionally excluded since traditional point sparring does not
involve clinching. Feint is included as a distinct state representing a
deliberate deceptive movement designed to provoke a reaction before a real
attack.
"""

import numpy as np
import matplotlib.pyplot as plt

STATES = ["Attack", "Defend", "Disengage", "Feint"]
N_STATES = len(STATES)

# Rows: current state. Columns: next state. Each row sums to 1.0.
TRANSITION_MATRIX = np.array([
    [0.20, 0.40, 0.25, 0.15],  # From Attack
    [0.45, 0.10, 0.30, 0.15],  # From Defend
    [0.35, 0.15, 0.20, 0.30],  # From Disengage
    [0.55, 0.10, 0.20, 0.15],  # From Feint
])

N_STEPS = 300
START_STATE = "Disengage"


def simulate(transition_matrix, start_state, n_steps, seed=42):
    rng = np.random.default_rng(seed)
    start_idx = STATES.index(start_state)

    sequence = np.empty(n_steps, dtype=int)
    sequence[0] = start_idx

    for t in range(1, n_steps):
        current = sequence[t - 1]
        sequence[t] = rng.choice(N_STATES, p=transition_matrix[current])

    return sequence


def theoretical_stationary_distribution(transition_matrix):
    eigenvalues, eigenvectors = np.linalg.eig(transition_matrix.T)
    idx = np.argmin(np.abs(eigenvalues - 1.0))
    stationary = np.real(eigenvectors[:, idx])
    stationary = stationary / stationary.sum()
    return stationary


def empirical_distribution(sequence):
    counts = np.bincount(sequence, minlength=N_STATES)
    return counts / counts.sum()


def plot_results(theoretical, empirical, sequence, filename="sparring_markov.png"):
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    # Left: theoretical vs empirical distribution
    ax = axes[0]
    x = np.arange(N_STATES)
    width = 0.35
    ax.bar(x - width / 2, theoretical, width, label="Theoretical", color="#4C72B0")
    ax.bar(x + width / 2, empirical, width, label="Empirical", color="#DD8452")
    ax.set_xticks(x)
    ax.set_xticklabels(STATES)
    ax.set_ylabel("Probability")
    ax.set_title("Stationary Distribution: Theoretical vs Empirical")
    ax.legend()
    ax.grid(axis="y", linestyle="--", alpha=0.4)

    # Right: step plot of first 60 exchanges
    ax = axes[1]
    n_show = 60
    ax.step(range(n_show), sequence[:n_show], where="mid", color="#4C72B0")
    ax.set_yticks(range(N_STATES))
    ax.set_yticklabels(STATES)
    ax.set_xlabel("Exchange step")
    ax.set_title("First 60 Exchanges of Simulation")
    ax.grid(axis="y", linestyle="--", alpha=0.4)

    fig.tight_layout()
    fig.savefig(filename, dpi=150)
    plt.close(fig)


def main():
    row_sums = TRANSITION_MATRIX.sum(axis=1)
    assert np.allclose(row_sums, 1.0), f"Transition rows must sum to 1.0, got {row_sums}"

    sequence = simulate(TRANSITION_MATRIX, START_STATE, N_STEPS)
    theoretical = theoretical_stationary_distribution(TRANSITION_MATRIX)
    empirical = empirical_distribution(sequence)

    print("Sparring Exchange Markov Chain")
    print("=" * 60)
    print(f"States: {STATES}")
    print(f"Start state: {START_STATE}")
    print(f"Simulation steps: {N_STEPS}")
    print()
    print(f"{'State':<12}{'Theoretical':>15}{'Empirical':>15}")
    print("-" * 42)
    for state, t_prob, e_prob in zip(STATES, theoretical, empirical):
        print(f"{state:<12}{t_prob:>15.4f}{e_prob:>15.4f}")
    print("-" * 42)
    print(f"{'Sum':<12}{theoretical.sum():>15.4f}{empirical.sum():>15.4f}")

    plot_results(theoretical, empirical, sequence)
    print("\nSaved plot to sparring_markov.png")


if __name__ == "__main__":
    main()
