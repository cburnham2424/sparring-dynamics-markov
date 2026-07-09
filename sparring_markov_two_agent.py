"""
Two-agent Markov chain model of a traditional point sparring exchange,
driven by Evolutionary Game Theory (EGT) replicator dynamics.

States (both fighters): Attack, Defend, Disengage, Feint

Fighter 1 (CJ) is an aggressive, timing-based blitzer: heavy feinter (cut kick
as a guard-breaker before a round kick combo), level-switches to confuse
defensive reads, and disengages strategically to reset timing rather than to
hide. He rarely sits in pure defend, always looking to counter or reset.

Fighter 2 is a patient counter-puncher working a Philly-shell-style stance: a
pump kick maps to Feint (a range finder, not a scoring tool), attacks
primarily off a defensive read, and rarely self-initiates from Disengage.

REPLICATOR DYNAMICS INTERACTION LAYER
--------------------------------------
Rather than nudging each fighter's base transition row with hand-tuned
adjustment deltas, each fighter's next-state probabilities are reshaped by
replicator dynamics: actions that pay off better against the opponent's
*current* state grow relatively more likely, and worse-paying actions shrink,
scaled by `selection_strength`. At `selection_strength = 0.0` this reduces
exactly to the unmodified base transition matrix.

ADAPTATION MEMORY LAYER
------------------------
On top of the per-step EGT adjustment, each fighter accumulates "exposure" to
a specific opponent tell, decaying it when not reinforced:
  - f2_attack_exposure: how much F1 has been repeatedly exposed to F2 Attack.
    As it grows, F1's Defend probability is boosted (reading the pattern) at
    the cost of Feint (less time spent setting up).
  - f1_feint_exposure: how much F2 has been repeatedly exposed to F1 Feint.
    As it grows, F2's Defend probability is boosted (reading CJ's setups) at
    the cost of Disengage (less passive waiting).
This models a fighter "learning" an opponent's tendency within a single
sparring session, and is loosely analogous to adaptive-immune priming under
repeated antigen exposure in mathematical oncology.
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

# Payoff[i, j] = fitness a fighter gets playing state i while the opponent plays state j.
F1_PAYOFF = np.array([
    [0.2, 0.8, 0.5, 0.9],  # F1 Attack vs F2: Atk, Def, Dis, Fnt
    [0.6, 0.1, 0.3, 0.4],  # F1 Defend
    [0.3, 0.4, 0.2, 0.6],  # F1 Disengage
    [0.5, 0.7, 0.4, 0.3],  # F1 Feint
])

F2_PAYOFF = np.array([
    [0.7, 0.2, 0.8, 0.3],  # F2 Attack vs F1: Atk, Def, Dis, Fnt
    [0.8, 0.1, 0.2, 0.6],  # F2 Defend
    [0.2, 0.5, 0.3, 0.4],  # F2 Disengage
    [0.4, 0.3, 0.7, 0.2],  # F2 Feint
])

N_STEPS = 500
START_STATE = "Disengage"
SELECTION_STRENGTHS = [0.0, 1.0, 2.0]

F1_COLOR = "steelblue"
F2_COLOR = "coral"

# Adaptation memory: how much each fighter has been repeatedly exposed to a
# specific opponent tell, decaying each step it isn't reinforced.
MEMORY_DECAY = 0.95
MEMORY_GROWTH = 1.5
MAX_EXPOSURE = 10.0
F1_MEMORY_BOOST_MAX = 0.25  # F1's Defend boost as f2_attack_exposure grows
F2_MEMORY_BOOST_MAX = 0.20  # F2's Defend boost as f1_feint_exposure grows

ROLLING_WINDOW = 20
MEMORY_STAT_STEPS = [100, 250, 500]
MEMORY_REFERENCE_STRENGTH = 1.0

JOINT_STATE_DESCRIPTIONS = {
    (ATTACK, ATTACK): "Mutual blitz - both fighters attack at once, a scramble.",
    (ATTACK, DEFEND): "CJ presses forward into Fighter 2's shell.",
    (ATTACK, DISENGAGE): "CJ blitzes just as Fighter 2 tries to reset the distance.",
    (ATTACK, FEINT): "CJ attacks while Fighter 2 is mid pump-kick - caught probing.",
    (DEFEND, ATTACK): "Fighter 2 attacks into CJ's guard; CJ covers up.",
    (DEFEND, DEFEND): "Mutual standoff - both fighters shelled up, nobody commits.",
    (DEFEND, DISENGAGE): "CJ covers up as Fighter 2 backs out to reset.",
    (DEFEND, FEINT): "CJ reads Fighter 2's pump kick and stays covered.",
    (DISENGAGE, ATTACK): "Fighter 2 counters just as CJ tries to create distance.",
    (DISENGAGE, DEFEND): "Mutual reset - both fighters at range, guards up.",
    (DISENGAGE, DISENGAGE): "Both fighters circle at range - no engagement.",
    (DISENGAGE, FEINT): "CJ resets while Fighter 2 probes with a pump kick.",
    (FEINT, ATTACK): "CJ feints just as Fighter 2 attacks - a mistimed setup.",
    (FEINT, DEFEND): "CJ feints to break Fighter 2's shell - the classic setup.",
    (FEINT, DISENGAGE): "CJ feints while Fighter 2 backs out of range.",
    (FEINT, FEINT): "Mutual feinting - both probing, neither commits.",
}


def apply_replicator_dynamics(base_probs, payoff_matrix, opponent_state, selection_strength=1.0):
    """
    Modify transition probabilities using replicator dynamics.

    For each action i in the current fighter's state distribution:
    - fitness[i] = payoff_matrix[i, opponent_state]
    - mean_fitness = sum(base_probs * fitness)
    - new_prob[i] = base_probs[i] * (1 + selection_strength * (fitness[i] - mean_fitness))
    - Clip negatives to 0 and renormalize to sum to 1.0

    selection_strength controls how strongly payoffs influence transitions:
    - 0.0 = no EGT influence (pure base transition matrix)
    - 1.0 = moderate EGT influence (default)
    - 2.0 = strong EGT influence
    """
    fitness = payoff_matrix[:, opponent_state]
    mean_fitness = np.sum(base_probs * fitness)
    new_probs = base_probs * (1 + selection_strength * (fitness - mean_fitness))
    new_probs = np.clip(new_probs, 0, None)
    return new_probs / new_probs.sum()


def apply_memory_effect(base_probs, exposure, boost_state, cost_state, max_boost=0.25):
    """
    As exposure grows toward MAX_EXPOSURE, gradually boost one state's
    probability and reduce another's, using proportional redistribution.

    Blindly subtracting the boost from cost_state and clipping negatives to
    0 silently destroys probability mass whenever cost_state can't cover the
    full boost (e.g. cost_state=0.1, boost=0.25 loses 0.15 before
    renormalizing, distorting every other state unintentionally). Instead,
    cap the amount taken from cost_state at what it actually holds, and if
    that falls short of the intended boost, drain the shortfall
    proportionally from the remaining states.
    """
    exposure_ratio = exposure / MAX_EXPOSURE
    intended_boost = max_boost * exposure_ratio
    actual_boost = min(intended_boost, base_probs[cost_state])

    new_probs = base_probs.copy()
    new_probs[boost_state] += actual_boost
    new_probs[cost_state] -= actual_boost

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
    return np.clip(new_probs, 0, None)


def simulate(selection_strength, seed=42):
    rng = np.random.default_rng(seed)
    start_idx = STATES.index(START_STATE)

    f1_seq = np.empty(N_STEPS, dtype=int)
    f2_seq = np.empty(N_STEPS, dtype=int)
    f1_seq[0] = start_idx
    f2_seq[0] = start_idx

    f2_attack_exposure = 0.0   # how much F1 has been exposed to F2's Attack
    f1_feint_exposure = 0.0    # how much F2 has been exposed to F1's Feint
    f2_exposure_history = [f2_attack_exposure]
    f1_exposure_history = [f1_feint_exposure]
    f1_defend_prob_history = []
    f2_defend_prob_history = []

    f1_fitness_history = [F1_PAYOFF[f1_seq[0], f2_seq[0]]]
    f2_fitness_history = [F2_PAYOFF[f2_seq[0], f1_seq[0]]]
    f1_cumulative_fitness = [f1_fitness_history[0]]
    f2_cumulative_fitness = [f2_fitness_history[0]]

    for t in range(1, N_STEPS):
        f1_prev, f2_prev = f1_seq[t - 1], f2_seq[t - 1]

        f1_row = apply_replicator_dynamics(F1_BASE[f1_prev], F1_PAYOFF, f2_prev, selection_strength)
        f2_row = apply_replicator_dynamics(F2_BASE[f2_prev], F2_PAYOFF, f1_prev, selection_strength)

        # F1 reads F2's repeated attacks: more Defend, less time feinting.
        f1_row = apply_memory_effect(f1_row, f2_attack_exposure, DEFEND, FEINT, F1_MEMORY_BOOST_MAX)
        # F2 reads CJ's repeated feints: more Defend, less passive disengaging.
        f2_row = apply_memory_effect(f2_row, f1_feint_exposure, DEFEND, DISENGAGE, F2_MEMORY_BOOST_MAX)

        f1_defend_prob_history.append(f1_row[DEFEND])
        f2_defend_prob_history.append(f2_row[DEFEND])

        f1_seq[t] = rng.choice(N_STATES, p=f1_row)
        f2_seq[t] = rng.choice(N_STATES, p=f2_row)

        if f2_seq[t] == ATTACK:
            f2_attack_exposure = min(f2_attack_exposure + MEMORY_GROWTH, MAX_EXPOSURE)
        else:
            f2_attack_exposure *= MEMORY_DECAY

        if f1_seq[t] == FEINT:
            f1_feint_exposure = min(f1_feint_exposure + MEMORY_GROWTH, MAX_EXPOSURE)
        else:
            f1_feint_exposure *= MEMORY_DECAY

        f2_exposure_history.append(f2_attack_exposure)
        f1_exposure_history.append(f1_feint_exposure)

        f1_payoff_this_step = F1_PAYOFF[f1_seq[t], f2_seq[t]]
        f2_payoff_this_step = F2_PAYOFF[f2_seq[t], f1_seq[t]]
        f1_fitness_history.append(f1_payoff_this_step)
        f2_fitness_history.append(f2_payoff_this_step)
        f1_cumulative_fitness.append(f1_cumulative_fitness[-1] + f1_payoff_this_step)
        f2_cumulative_fitness.append(f2_cumulative_fitness[-1] + f2_payoff_this_step)

    return {
        "f1_seq": f1_seq,
        "f2_seq": f2_seq,
        "f2_exposure_history": np.array(f2_exposure_history),
        "f1_exposure_history": np.array(f1_exposure_history),
        "f1_defend_prob_history": np.array(f1_defend_prob_history),
        "f2_defend_prob_history": np.array(f2_defend_prob_history),
        "f1_fitness_history": np.array(f1_fitness_history),
        "f2_fitness_history": np.array(f2_fitness_history),
        "f1_cumulative_fitness": np.array(f1_cumulative_fitness),
        "f2_cumulative_fitness": np.array(f2_cumulative_fitness),
    }


def empirical_distribution(sequence):
    counts = np.bincount(sequence, minlength=N_STATES)
    return counts / counts.sum()


def empirical_joint_distribution(f1_seq, f2_seq):
    """4x4 empirical joint distribution of (F1 state, F2 state) pairs."""
    counts = np.zeros((N_STATES, N_STATES))
    for f1, f2 in zip(f1_seq, f2_seq):
        counts[f1, f2] += 1
    return counts / counts.sum()


def top_joint_states(joint_dist, n=3):
    flat_indices = np.argsort(joint_dist, axis=None)[::-1][:n]
    results = []
    for flat_idx in flat_indices:
        f1_idx, f2_idx = np.unravel_index(flat_idx, joint_dist.shape)
        results.append((f1_idx, f2_idx, joint_dist[f1_idx, f2_idx]))
    return results


def print_marginal_table(f1_dist, f2_dist):
    print(f"{'State':<12}{'Fighter 1':>12}{'Fighter 2':>12}")
    print("-" * 36)
    for state, f1_p, f2_p in zip(STATES, f1_dist, f2_dist):
        print(f"{state:<12}{f1_p:>12.4f}{f2_p:>12.4f}")
    print("-" * 36)
    print(f"{'Sum':<12}{f1_dist.sum():>12.4f}{f2_dist.sum():>12.4f}")


def summarize_change(prev_f1, prev_f2, curr_f1, curr_f2, prev_strength, curr_strength):
    f1_deltas = curr_f1 - prev_f1
    f2_deltas = curr_f2 - prev_f2
    f1_idx = np.argmax(np.abs(f1_deltas))
    f2_idx = np.argmax(np.abs(f2_deltas))

    f1_dir = "up" if f1_deltas[f1_idx] > 0 else "down"
    f2_dir = "up" if f2_deltas[f2_idx] > 0 else "down"

    return (
        f"Compared to strength {prev_strength}: CJ's {STATES[f1_idx]} probability moved "
        f"{f1_dir} from {prev_f1[f1_idx]:.3f} to {curr_f1[f1_idx]:.3f} "
        f"({f1_deltas[f1_idx]:+.3f}), while Fighter 2's {STATES[f2_idx]} probability moved "
        f"{f2_dir} from {prev_f2[f2_idx]:.3f} to {curr_f2[f2_idx]:.3f} "
        f"({f2_deltas[f2_idx]:+.3f})."
    )


def run_all_simulations():
    results = {}
    for strength in SELECTION_STRENGTHS:
        sim = simulate(strength)
        sim["f1_dist"] = empirical_distribution(sim["f1_seq"])
        sim["f2_dist"] = empirical_distribution(sim["f2_seq"])
        sim["joint"] = empirical_joint_distribution(sim["f1_seq"], sim["f2_seq"])
        results[strength] = sim
    return results


def print_analysis(results):
    print("Two-Agent Sparring: EGT Replicator Dynamics")
    print("=" * 60)
    print(f"States: {STATES}")
    print(f"Start state (both fighters): {START_STATE}")
    print(f"Simulation steps per run: {N_STEPS}")
    print()

    prev = None
    for strength in SELECTION_STRENGTHS:
        r = results[strength]
        print(f"### Selection strength = {strength}")
        print()
        print_marginal_table(r["f1_dist"], r["f2_dist"])
        print()
        print("Top 3 joint states:")
        for rank, (f1_idx, f2_idx, prob) in enumerate(top_joint_states(r["joint"]), start=1):
            desc = JOINT_STATE_DESCRIPTIONS[(f1_idx, f2_idx)]
            print(f"  {rank}. {STATES[f1_idx]} vs {STATES[f2_idx]} - {prob:.1%} - {desc}")
        print()

        if prev is None:
            print("Baseline run: pure Markov chain dynamics, no EGT influence.")
        else:
            print(summarize_change(prev["f1_dist"], prev["f2_dist"], r["f1_dist"], r["f2_dist"],
                                    prev_strength=prev["strength"], curr_strength=strength))
        print()

        r["strength"] = strength
        prev = r


def rolling_average(values, window=ROLLING_WINDOW):
    kernel = np.ones(window) / window
    avg = np.convolve(values, kernel, mode="valid")
    x = np.arange(window - 1, len(values))
    return x, avg


def describe_exposure_level(ratio):
    if ratio < 0.33:
        return "low"
    elif ratio < 0.66:
        return "moderate"
    return "high"


def print_memory_stats(f1_exposure_history, f2_exposure_history):
    print(f"Adaptation Memory Snapshots (selection_strength={MEMORY_REFERENCE_STRENGTH} reference run)")
    print("=" * 60)
    for step in MEMORY_STAT_STEPS:
        idx = step - 1
        f2_exp = f2_exposure_history[idx]
        f1_exp = f1_exposure_history[idx]
        f2_ratio = f2_exp / MAX_EXPOSURE
        f1_ratio = f1_exp / MAX_EXPOSURE
        f1_boost = F1_MEMORY_BOOST_MAX * f2_ratio
        f2_boost = F2_MEMORY_BOOST_MAX * f1_ratio

        print(f"--- Step {step} ---")
        print(f"F2 attack exposure: {f2_exp:.3f} / {MAX_EXPOSURE:.1f}  ({f2_ratio:.1%})")
        print(f"F1 feint exposure:  {f1_exp:.3f} / {MAX_EXPOSURE:.1f}  ({f1_ratio:.1%})")
        print(f"F1 Defend boost currently applied: +{f1_boost:.3f} (Feint -{f1_boost:.3f})")
        print(f"F2 Defend boost currently applied: +{f2_boost:.3f} (Disengage -{f2_boost:.3f})")
        print(
            f"Sparring read: CJ's read on Fighter 2's attacks is {describe_exposure_level(f2_ratio)} "
            f"({f2_ratio:.0%}) - his Defend is sharpening at the cost of feint setups. Fighter 2's "
            f"read on CJ's feints is {describe_exposure_level(f1_ratio)} ({f1_ratio:.0%}) - trading "
            f"passive resets for active covers."
        )
        print(
            f"Tumor-immune parallel: repeated antigen exposure (F2's Attack) primes F1's adaptive "
            f"response (Defend) at {f2_ratio:.0%} strength, while sustained pressure (F1's Feint) "
            f"drives F2's immune remodeling (Defend) at {f1_ratio:.0%} strength - akin to "
            f"immunoediting under chronic antigenic stimulation."
        )
        print()


def plot_memory_grid(f2_exposure_history, f1_exposure_history,
                      f1_defend_prob_history, f2_defend_prob_history,
                      filename="sparring_memory.png"):
    fig, axes = plt.subplots(2, 2, figsize=(13, 10))
    steps = np.arange(len(f2_exposure_history))

    ax = axes[0, 0]
    ax.plot(steps, f2_exposure_history, color=F2_COLOR)
    for frac in (0.25, 0.5, 0.75, 1.0):
        ax.axhline(frac * MAX_EXPOSURE, color="gray", linestyle="--", alpha=0.5)
    ax.set_xlabel("Exchange step")
    ax.set_ylabel("Exposure level")
    ax.set_ylim(0, MAX_EXPOSURE * 1.05)
    ax.set_title("Fighter 2 Attack Exposure (F1 Adaptation Memory)")
    ax.grid(axis="y", linestyle=":", alpha=0.2)

    ax = axes[0, 1]
    ax.plot(steps, f1_exposure_history, color=F1_COLOR)
    for frac in (0.25, 0.5, 0.75, 1.0):
        ax.axhline(frac * MAX_EXPOSURE, color="gray", linestyle="--", alpha=0.5)
    ax.set_xlabel("Exchange step")
    ax.set_ylabel("Exposure level")
    ax.set_ylim(0, MAX_EXPOSURE * 1.05)
    ax.set_title("Fighter 1 Feint Exposure (F2 Adaptation Memory)")
    ax.grid(axis="y", linestyle=":", alpha=0.2)

    ax = axes[1, 0]
    x, avg = rolling_average(f1_defend_prob_history)
    ax.plot(x, avg, color=F1_COLOR)
    ax.set_xlabel("Exchange step")
    ax.set_ylabel("Probability")
    ax.set_title(f"F1 Defend Probability (Rolling {ROLLING_WINDOW}-step avg)")
    ax.grid(axis="y", linestyle="--", alpha=0.4)

    ax = axes[1, 1]
    x, avg = rolling_average(f2_defend_prob_history)
    ax.plot(x, avg, color=F2_COLOR)
    ax.set_xlabel("Exchange step")
    ax.set_ylabel("Probability")
    ax.set_title(f"F2 Defend Probability (Rolling {ROLLING_WINDOW}-step avg)")
    ax.grid(axis="y", linestyle="--", alpha=0.4)

    fig.tight_layout()
    fig.savefig(filename, dpi=150)
    plt.close(fig)


def find_last_lead_change(f1_cumulative_fitness, f2_cumulative_fitness):
    diff = f1_cumulative_fitness - f2_cumulative_fitness
    sign = np.sign(diff)
    nonzero = sign[sign != 0]
    if len(nonzero) < 2:
        return None
    changes = np.where(np.diff(sign) != 0)[0]
    return int(changes[-1]) + 1 if len(changes) > 0 else None


def print_fitness_summary(f1_fitness_history, f2_fitness_history,
                           f1_cumulative_fitness, f2_cumulative_fitness):
    print(f"Cumulative Fitness Summary (selection_strength={MEMORY_REFERENCE_STRENGTH} reference run)")
    print("=" * 60)

    f1_total = f1_cumulative_fitness[-1]
    f2_total = f2_cumulative_fitness[-1]
    margin = f1_total - f2_total

    print(f"Total cumulative fitness — Fighter 1 (CJ): {f1_total:.3f}")
    print(f"Total cumulative fitness — Fighter 2 (Counter-Puncher): {f2_total:.3f}")
    if margin > 1e-6:
        print(f"Winner on total fitness: Fighter 1 (CJ), by {margin:.3f}")
    elif margin < -1e-6:
        print(f"Winner on total fitness: Fighter 2 (Counter-Puncher), by {-margin:.3f}")
    else:
        print("Result: dead even on total fitness")

    lead_change_step = find_last_lead_change(f1_cumulative_fitness, f2_cumulative_fitness)
    if lead_change_step is None:
        leader = "Fighter 1 (CJ)" if margin > -1e-6 else "Fighter 2 (Counter-Puncher)"
        print(f"Lead never changed hands — {leader} led throughout")
    else:
        print(f"Lead last changed hands at step {lead_change_step}")

    f1_avg = f1_fitness_history.mean()
    f2_avg = f2_fitness_history.mean()
    print(f"Average per-step payoff — Fighter 1 (CJ): {f1_avg:.4f}")
    print(f"Average per-step payoff — Fighter 2 (Counter-Puncher): {f2_avg:.4f}")
    print()

    if margin > 1e-6:
        sparring_note = "CJ's blitz-and-feint pressure is outscoring the counter-punching patience"
        tumor_note = ("the aggressor phenotype (CJ) is accumulating a fitness edge, analogous to a "
                       "tumor clone outcompeting immune surveillance")
    elif margin < -1e-6:
        sparring_note = "the counter-puncher's patient reads are outscoring the blitz"
        tumor_note = ("the adaptive/defensive phenotype (Fighter 2) is accumulating a fitness edge, "
                       "analogous to immune surveillance suppressing a tumor clone")
    else:
        sparring_note = "both styles are scoring evenly"
        tumor_note = "the two phenotypes are in a fitness stalemate, analogous to immune equilibrium"

    print(
        f"Sparring read: over the full exchange, average payoff per exchange is "
        f"{f1_avg:.3f} (CJ) vs {f2_avg:.3f} (Fighter 2) - {sparring_note}."
    )
    print(f"Tumor-immune parallel: cumulative fitness tracks relative selective advantage - {tumor_note}.")
    print()


def plot_fitness(f1_fitness_history, f2_fitness_history,
                  f1_cumulative_fitness, f2_cumulative_fitness,
                  filename="sparring_fitness.png"):
    fig, axes = plt.subplots(2, 1, figsize=(11, 11))
    steps = np.arange(len(f1_fitness_history))

    ax = axes[0]
    x1, avg1 = rolling_average(f1_fitness_history)
    x2, avg2 = rolling_average(f2_fitness_history)
    ax.plot(x1, avg1, color=F1_COLOR, label="Fighter 1 (CJ)")
    ax.plot(x2, avg2, color=F2_COLOR, label="Fighter 2 (Counter-Puncher)")
    ax.axhline(0.5, color="gray", linestyle="--", alpha=0.6)
    ax.set_ylim(0, 1)
    ax.set_xlabel("Exchange step")
    ax.set_ylabel("Payoff")
    ax.set_title(f"Per-Step Payoff — Rolling {ROLLING_WINDOW}-Step Average")
    ax.legend()
    ax.grid(axis="y", linestyle="--", alpha=0.4)

    ax = axes[1]
    ax.plot(steps, f1_cumulative_fitness, color=F1_COLOR, label="Fighter 1 (CJ)")
    ax.plot(steps, f2_cumulative_fitness, color=F2_COLOR, label="Fighter 2 (Counter-Puncher)")
    ax.fill_between(steps, f1_cumulative_fitness, f2_cumulative_fitness,
                     where=f1_cumulative_fitness >= f2_cumulative_fitness,
                     color="green", alpha=0.2, interpolate=True)
    ax.fill_between(steps, f1_cumulative_fitness, f2_cumulative_fitness,
                     where=f1_cumulative_fitness < f2_cumulative_fitness,
                     color="red", alpha=0.2, interpolate=True)
    ax.set_xlabel("Exchange step")
    ax.set_ylabel("Cumulative payoff")
    ax.set_title("Cumulative Fitness Over Match Duration")
    ax.legend()
    ax.grid(axis="y", linestyle="--", alpha=0.4)

    fig.tight_layout()
    fig.savefig(filename, dpi=150)
    plt.close(fig)


def plot_marginals_grid(results, filename="sparring_egt.png"):
    fig, axes = plt.subplots(3, 2, figsize=(11, 13))
    x = np.arange(N_STATES)

    for row, strength in enumerate(SELECTION_STRENGTHS):
        r = results[strength]

        ax_f1 = axes[row, 0]
        ax_f1.bar(x, r["f1_dist"], color=F1_COLOR)
        ax_f1.set_xticks(x)
        ax_f1.set_xticklabels(STATES)
        ax_f1.set_ylabel("Probability")
        ax_f1.set_title(f"Fighter 1 (CJ) — selection_strength={strength}")
        ax_f1.grid(axis="y", linestyle="--", alpha=0.4)

        ax_f2 = axes[row, 1]
        ax_f2.bar(x, r["f2_dist"], color=F2_COLOR)
        ax_f2.set_xticks(x)
        ax_f2.set_xticklabels(STATES)
        ax_f2.set_ylabel("Probability")
        ax_f2.set_title(f"Fighter 2 (Counter-Puncher) — selection_strength={strength}")
        ax_f2.grid(axis="y", linestyle="--", alpha=0.4)

    fig.tight_layout()
    fig.savefig(filename, dpi=150)
    plt.close(fig)


def plot_joint_heatmaps_grid(results, filename="sparring_egt_heatmaps.png"):
    fig, axes = plt.subplots(1, 3, figsize=(17, 5.5))
    vmax = max(results[s]["joint"].max() for s in SELECTION_STRENGTHS)

    for ax, strength in zip(axes, SELECTION_STRENGTHS):
        matrix = results[strength]["joint"]
        im = ax.imshow(matrix, cmap="Blues", vmin=0, vmax=vmax)
        ax.set_xticks(range(N_STATES))
        ax.set_xticklabels(STATES)
        ax.set_yticks(range(N_STATES))
        ax.set_yticklabels(STATES)
        ax.set_xlabel("Fighter 2 state")
        ax.set_ylabel("Fighter 1 state")
        ax.set_title(f"Joint Distribution — selection_strength={strength}")

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

    results = run_all_simulations()
    print_analysis(results)

    plot_marginals_grid(results)
    plot_joint_heatmaps_grid(results)
    print("Saved plot to sparring_egt.png")
    print("Saved plot to sparring_egt_heatmaps.png")
    print()

    reference = results[MEMORY_REFERENCE_STRENGTH]
    print_memory_stats(reference["f1_exposure_history"], reference["f2_exposure_history"])
    plot_memory_grid(reference["f2_exposure_history"], reference["f1_exposure_history"],
                      reference["f1_defend_prob_history"], reference["f2_defend_prob_history"])
    print("Saved plot to sparring_memory.png")
    print()

    print_fitness_summary(reference["f1_fitness_history"], reference["f2_fitness_history"],
                           reference["f1_cumulative_fitness"], reference["f2_cumulative_fitness"])
    plot_fitness(reference["f1_fitness_history"], reference["f2_fitness_history"],
                 reference["f1_cumulative_fitness"], reference["f2_cumulative_fitness"])
    print("Saved plot to sparring_fitness.png")


if __name__ == "__main__":
    main()
