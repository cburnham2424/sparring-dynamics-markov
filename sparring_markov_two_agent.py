"""
Two-agent interacting Markov chain model of a traditional point sparring exchange.

States (both fighters): Attack, Defend, Disengage, Feint

Fighter 1 (CJ) is an aggressive, timing-based blitzer: heavy feinter (cut kick
as a guard-breaker before a round kick combo), level-switches to confuse
defensive reads, and disengages strategically to reset timing rather than to
hide. He rarely sits in pure defend, always looking to counter or reset.

Fighter 2 is a patient counter-puncher working a Philly-shell-style stance: a
pump kick maps to Feint (a range finder, not a scoring tool), attacks
primarily off a defensive read, and rarely self-initiates from Disengage.

INTERACTION LAYER
-----------------
Each fighter's base transition row is nudged by the opponent's *current*
state before a next state is drawn, modeling how each fighter reads and
reacts to what the other is doing right now.

Fighter 1's row is adjusted based on Fighter 2's current state:
  - Fighter 2 Defend:    Feint +0.10, Attack -0.10  (feints more to break the shell)
  - Fighter 2 Attack:    Defend +0.15, Disengage +0.05, Attack -0.20
  - Fighter 2 Feint:     Disengage +0.10, Attack -0.10  (reads the pump kick, resets)
  - Fighter 2 Disengage: Attack +0.15, Disengage -0.15  (blitzes into the reset)

Fighter 2's row is adjusted based on Fighter 1's current state:
  - Fighter 1 Feint:     Defend +0.15, Attack -0.15  (reads the setup)
  - Fighter 1 Attack:    Defend +0.10, Attack +0.10, Disengage -0.20  (counter chance)
  - Fighter 1 Disengage: Disengage +0.10, Defend -0.10  (stays patient, does not blitz)
  - Fighter 1 Defend:    no adjustment (Fighter 2 has no defined reaction)

Every adjustment set sums to zero, so each adjusted row still sums to 1.0.
"""

import numpy as np
import matplotlib.pyplot as plt

STATES = ["Attack", "Defend", "Disengage", "Feint"]
N_STATES = len(STATES)
ATTACK, DEFEND, DISENGAGE, FEINT = range(N_STATES)

# Rows: current state. Columns: next state. Each row sums to 1.0.
F1_BASE = np.array([
    [0.25, 0.15, 0.35, 0.25],  # From Attack
    [0.45, 0.10, 0.25, 0.20],  # From Defend
    [0.30, 0.10, 0.20, 0.40],  # From Disengage
    [0.60, 0.08, 0.17, 0.15],  # From Feint
])

F2_BASE = np.array([
    [0.15, 0.30, 0.35, 0.20],  # From Attack
    [0.55, 0.15, 0.20, 0.10],  # From Defend
    [0.15, 0.30, 0.25, 0.30],  # From Disengage
    [0.40, 0.25, 0.20, 0.15],  # From Feint
])

# adjustment[opponent_state] = delta vector applied to [Attack, Defend, Disengage, Feint]
F1_ADJUSTMENTS = {
    DEFEND:    np.array([-0.10,  0.00,  0.00,  0.10]),  # opponent (F2) in Defend
    ATTACK:    np.array([-0.20,  0.15,  0.05,  0.00]),  # opponent (F2) in Attack
    FEINT:     np.array([-0.10,  0.00,  0.10,  0.00]),  # opponent (F2) in Feint
    DISENGAGE: np.array([ 0.15,  0.00, -0.15,  0.00]),  # opponent (F2) in Disengage
}

F2_ADJUSTMENTS = {
    FEINT:     np.array([-0.15,  0.15,  0.00,  0.00]),  # opponent (F1) in Feint
    ATTACK:    np.array([ 0.10,  0.10, -0.20,  0.00]),  # opponent (F1) in Attack
    DISENGAGE: np.array([ 0.00, -0.10,  0.10,  0.00]),  # opponent (F1) in Disengage
    DEFEND:    np.array([ 0.00,  0.00,  0.00,  0.00]),  # opponent (F1) in Defend: no reaction
}

N_STEPS = 300
START_STATE = "Disengage"


def adjusted_row(base_matrix, own_state, opponent_state, adjustments):
    row = base_matrix[own_state] + adjustments[opponent_state]
    return row


def simulate(seed=42):
    rng = np.random.default_rng(seed)
    start_idx = STATES.index(START_STATE)

    f1_seq = np.empty(N_STEPS, dtype=int)
    f2_seq = np.empty(N_STEPS, dtype=int)
    f1_seq[0] = start_idx
    f2_seq[0] = start_idx

    for t in range(1, N_STEPS):
        f1_prev, f2_prev = f1_seq[t - 1], f2_seq[t - 1]

        f1_row = adjusted_row(F1_BASE, f1_prev, f2_prev, F1_ADJUSTMENTS)
        f2_row = adjusted_row(F2_BASE, f2_prev, f1_prev, F2_ADJUSTMENTS)

        f1_seq[t] = rng.choice(N_STATES, p=f1_row)
        f2_seq[t] = rng.choice(N_STATES, p=f2_row)

    return f1_seq, f2_seq


def build_joint_transition_matrix():
    """16x16 transition matrix over joint states (f1, f2), assuming each
    fighter draws its next state independently given the current joint state."""
    n_joint = N_STATES * N_STATES
    joint = np.zeros((n_joint, n_joint))

    for f1 in range(N_STATES):
        for f2 in range(N_STATES):
            i = f1 * N_STATES + f2
            f1_row = adjusted_row(F1_BASE, f1, f2, F1_ADJUSTMENTS)
            f2_row = adjusted_row(F2_BASE, f2, f1, F2_ADJUSTMENTS)
            for f1_next in range(N_STATES):
                for f2_next in range(N_STATES):
                    j = f1_next * N_STATES + f2_next
                    joint[i, j] = f1_row[f1_next] * f2_row[f2_next]

    return joint


def joint_stationary_distribution():
    """4x4 theoretical joint stationary distribution P(F1=i, F2=j)."""
    joint = build_joint_transition_matrix()
    eigenvalues, eigenvectors = np.linalg.eig(joint.T)
    idx = np.argmin(np.abs(eigenvalues - 1.0))
    stationary = np.real(eigenvectors[:, idx])
    stationary = stationary / stationary.sum()
    return stationary.reshape(N_STATES, N_STATES)  # [f1_state, f2_state]


def empirical_distribution(sequence):
    counts = np.bincount(sequence, minlength=N_STATES)
    return counts / counts.sum()


def empirical_joint_distribution(f1_seq, f2_seq):
    """4x4 empirical joint distribution of (F1 state, F2 state) pairs."""
    counts = np.zeros((N_STATES, N_STATES))
    for f1, f2 in zip(f1_seq, f2_seq):
        counts[f1, f2] += 1
    return counts / counts.sum()


def print_distribution_table(title, theoretical, empirical):
    print(title)
    print(f"{'State':<12}{'Theoretical':>15}{'Empirical':>15}")
    print("-" * 42)
    for state, t_prob, e_prob in zip(STATES, theoretical, empirical):
        print(f"{state:<12}{t_prob:>15.4f}{e_prob:>15.4f}")
    print("-" * 42)
    print(f"{'Sum':<12}{theoretical.sum():>15.4f}{empirical.sum():>15.4f}")
    print()


def plot_results(f1_theoretical, f1_empirical, f1_seq,
                  f2_theoretical, f2_empirical, f2_seq,
                  filename="sparring_markov_two_agent.png"):
    fig, axes = plt.subplots(2, 2, figsize=(13, 10))
    x = np.arange(N_STATES)
    width = 0.35
    n_show = 60

    fighters = [
        ("Fighter 1 (CJ)", f1_theoretical, f1_empirical, f1_seq, "#55A868"),
        ("Fighter 2 (Counter-Puncher)", f2_theoretical, f2_empirical, f2_seq, "#C44E52"),
    ]

    for row, (name, theoretical, empirical, seq, line_color) in enumerate(fighters):
        ax_bar = axes[row, 0]
        ax_bar.bar(x - width / 2, theoretical, width, label="Theoretical", color="#4C72B0")
        ax_bar.bar(x + width / 2, empirical, width, label="Empirical", color="#DD8452")
        ax_bar.set_xticks(x)
        ax_bar.set_xticklabels(STATES)
        ax_bar.set_ylabel("Probability")
        ax_bar.set_title(f"{name}: Theoretical vs Empirical")
        ax_bar.legend()
        ax_bar.grid(axis="y", linestyle="--", alpha=0.4)

        ax_step = axes[row, 1]
        ax_step.step(range(n_show), seq[:n_show], where="mid", color=line_color)
        ax_step.set_yticks(range(N_STATES))
        ax_step.set_yticklabels(STATES)
        ax_step.set_xlabel("Exchange step")
        ax_step.set_title(f"{name}: First 60 Exchanges")
        ax_step.grid(axis="y", linestyle="--", alpha=0.4)

    fig.tight_layout()
    fig.savefig(filename, dpi=150)
    plt.close(fig)


def plot_joint_heatmaps(theoretical_joint, empirical_joint,
                         filename="sparring_markov_two_agent_heatmap.png"):
    fig, axes = plt.subplots(1, 2, figsize=(12, 5.5))
    panels = [
        ("Theoretical Joint Distribution", theoretical_joint),
        ("Empirical Joint Distribution", empirical_joint),
    ]
    vmax = max(theoretical_joint.max(), empirical_joint.max())

    for ax, (title, matrix) in zip(axes, panels):
        im = ax.imshow(matrix, cmap="Blues", vmin=0, vmax=vmax)
        ax.set_xticks(range(N_STATES))
        ax.set_xticklabels(STATES)
        ax.set_yticks(range(N_STATES))
        ax.set_yticklabels(STATES)
        ax.set_xlabel("Fighter 2 state")
        ax.set_ylabel("Fighter 1 state")
        ax.set_title(title)

        for i in range(N_STATES):
            for j in range(N_STATES):
                value = matrix[i, j]
                text_color = "white" if value > vmax * 0.6 else "black"
                ax.text(j, i, f"{value:.3f}", ha="center", va="center", color=text_color)

        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="Probability")

    fig.tight_layout()
    fig.savefig(filename, dpi=150)
    plt.close(fig)


def main():
    for name, matrix in [("Fighter 1", F1_BASE), ("Fighter 2", F2_BASE)]:
        row_sums = matrix.sum(axis=1)
        assert np.allclose(row_sums, 1.0), f"{name} base rows must sum to 1.0, got {row_sums}"

    f1_seq, f2_seq = simulate()
    theoretical_joint = joint_stationary_distribution()
    f1_theoretical = theoretical_joint.sum(axis=1)
    f2_theoretical = theoretical_joint.sum(axis=0)
    f1_empirical = empirical_distribution(f1_seq)
    f2_empirical = empirical_distribution(f2_seq)
    empirical_joint = empirical_joint_distribution(f1_seq, f2_seq)

    print("Two-Agent Interacting Sparring Markov Chain")
    print("=" * 60)
    print(f"States: {STATES}")
    print(f"Start state (both fighters): {START_STATE}")
    print(f"Simulation steps: {N_STEPS}")
    print()
    print_distribution_table("Fighter 1 (CJ) — Marginal Distribution", f1_theoretical, f1_empirical)
    print_distribution_table("Fighter 2 (Counter-Puncher) — Marginal Distribution", f2_theoretical, f2_empirical)

    plot_results(f1_theoretical, f1_empirical, f1_seq, f2_theoretical, f2_empirical, f2_seq)
    plot_joint_heatmaps(theoretical_joint, empirical_joint)
    print("Saved plot to sparring_markov_two_agent.png")
    print("Saved plot to sparring_markov_two_agent_heatmap.png")


if __name__ == "__main__":
    main()
