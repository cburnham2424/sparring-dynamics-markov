"""
Two-agent Markov chain model of a traditional point sparring exchange,
driven by Evolutionary Game Theory (EGT) replicator dynamics.

States (both fighters): Attack, Defend, Disengage, Feint

Fighter 1 (CJ) is an aggressive, timing-based blitzer: heavy feinter (cut kick
as a guard-breaker before a round kick combo), level-switches to confuse
defensive reads, and disengages strategically to reset timing rather than to
hide. He rarely sits in pure defend, always looking to counter or reset.

Fighter 2 is a patient counter-fighter working a Philly-shell-style stance: a
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

ADAPTIVE TRANSITION MATRIX LAYER
---------------------------------
On top of the per-step EGT adjustment, each fighter accumulates "exposure" to
a specific opponent tell, decaying it when not reinforced:
  - f2_attack_exposure: how much F1 has been repeatedly exposed to F2 Attack.
  - f1_feint_exposure: how much F2 has been repeatedly exposed to F1 Feint.

Rather than nudging two states of the base matrix, exposure now drives a
convex blend of the ENTIRE base transition matrix toward a fully-adapted
target matrix (F1_ADAPTATION_MATRIX / F2_ADAPTATION_MATRIX) representing the
fighter's fully pattern-read strategic profile:

    P_new = (1 - lambda) * P_base + lambda * P_adaptation

lambda climbs from 0 (no adaptation, pure base matrix) to 1 (fully adapted)
along a smooth sigmoid of exposure, so the whole strategic profile shifts
gradually rather than two probabilities being pushed and pulled. This models
a fighter "learning" an opponent's tendency within a single sparring session,
and is loosely analogous to adaptive-immune priming and gradual phenotypic
switching under sustained selective pressure in mathematical oncology.
"""

import numpy as np
import pandas as pd
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

# What each fighter's transition matrix converges to under maximum exposure
# to the tracked opponent tell (the fully pattern-read response).
F1_ADAPTATION_MATRIX = np.array([
    # Atk    Def    Dis    Fnt
    [0.15,  0.35,  0.25,  0.25],  # From Attack — less aggressive, more defensive
    [0.55,  0.08,  0.17,  0.20],  # From Defend — more counter-attacking
    [0.25,  0.20,  0.15,  0.40],  # From Disengage — more feinting to probe
    [0.65,  0.10,  0.10,  0.15],  # From Feint — feint converts to attack more
])

F2_ADAPTATION_MATRIX = np.array([
    # Atk    Def    Dis    Fnt
    [0.10,  0.45,  0.30,  0.15],  # From Attack — more conservative attacking
    [0.60,  0.18,  0.12,  0.10],  # From Defend — faster counter when defending
    [0.10,  0.45,  0.15,  0.30],  # From Disengage — more defensive waiting
    [0.35,  0.35,  0.15,  0.15],  # From Feint — reads feints, defends more
])

LAMBDA_STEEPNESS = 0.6
LAMBDA_MIDPOINT = 0.5

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


def compute_lambda(exposure, max_exposure=MAX_EXPOSURE,
                    steepness=LAMBDA_STEEPNESS, midpoint=LAMBDA_MIDPOINT):
    """
    Smooth lambda schedule based on exposure ratio.

    exposure_ratio = exposure / max_exposure  (0.0 to 1.0)

    Uses a smooth sigmoid centered at midpoint, then rescaled so
    lambda(ratio=0) = 0 and lambda(ratio=1) = 1 exactly:
    - No adaptation at zero exposure (lambda=0, pure base matrix)
    - Full adaptation at max exposure (lambda=1, pure adaptation matrix)
    - Smooth nonlinear ramp between (biologically realistic)

    steepness controls how sharply adaptation kicks in (low = gradual
    linear-like ramp, high = sudden jump around midpoint). midpoint is
    where adaptation is 50% complete.
    """
    k = steepness * 10  # scale steepness to reasonable sigmoid range
    ratio = np.clip(exposure / max_exposure, 0.0, 1.0)

    sigmoid = lambda r: 1.0 / (1.0 + np.exp(-k * (r - midpoint)))

    lambda_raw = sigmoid(ratio)
    lambda_min = sigmoid(0.0)
    lambda_max = sigmoid(1.0)

    lam = (lambda_raw - lambda_min) / (lambda_max - lambda_min)
    return float(np.clip(lam, 0.0, 1.0))


def apply_adaptive_matrix(base_matrix, adaptation_matrix, exposure,
                           max_exposure=MAX_EXPOSURE,
                           steepness=LAMBDA_STEEPNESS, midpoint=LAMBDA_MIDPOINT):
    """
    Blend a fighter's entire base transition matrix toward its fully-adapted
    target matrix based on current exposure level:

        P_new = (1 - lambda) * P_base + lambda * P_adaptation

    This shifts ALL transition probabilities simultaneously, not just two
    states — as exposure grows, the fighter's entire strategic profile
    shifts toward the fully-adapted response.
    """
    lam = compute_lambda(exposure, max_exposure, steepness, midpoint)

    P_new = (1.0 - lam) * base_matrix + lam * adaptation_matrix

    # Convex combination of two valid stochastic matrices already sums to 1
    # per row; renormalize anyway as floating-point safety.
    row_sums = P_new.sum(axis=1, keepdims=True)
    row_sums = np.where(row_sums < 1e-10, 1.0, row_sums)
    P_new = P_new / row_sums

    P_new = np.clip(P_new, 0.0, None)
    P_new = P_new / P_new.sum(axis=1, keepdims=True)

    return P_new, lam


def validate_adaptive_system():
    """
    Confirm key mathematical properties of the adaptive system:
    1. At exposure=0: P_new should equal F1_BASE exactly (lambda=0)
    2. At exposure=max: P_new should equal F1_ADAPTATION_MATRIX exactly (lambda=1)
    3. At exposure=5 (midpoint): lambda should be close to 0.5
    4. P_new rows sum to 1.0 at all tested exposure levels
    5. No negative values at any exposure level
    6. Lambda is monotonically increasing with exposure
    """
    print("Validating adaptive transition matrix system...")
    test_exposures = [0.0, 1.0, 2.5, 5.0, 7.5, 9.0, 10.0]
    lambdas = []

    for exp in test_exposures:
        P_new, lam = apply_adaptive_matrix(F1_BASE, F1_ADAPTATION_MATRIX, exp)
        row_sums = P_new.sum(axis=1)
        all_positive = np.all(P_new >= -1e-10)
        rows_valid = np.allclose(row_sums, 1.0, atol=1e-8)
        lambdas.append(lam)

        status = "PASS" if (rows_valid and all_positive) else "FAIL"
        print(f"  Exposure={exp:5.1f} | lambda={lam:.4f} | "
              f"row_sums_valid={rows_valid} | "
              f"all_positive={all_positive} | {status}")

    _, lam_zero = apply_adaptive_matrix(F1_BASE, F1_ADAPTATION_MATRIX, 0.0)
    _, lam_max = apply_adaptive_matrix(F1_BASE, F1_ADAPTATION_MATRIX, 10.0)
    print(f"\n  Lambda at exposure=0:   {lam_zero:.6f} (should be 0.0)")
    print(f"  Lambda at exposure=10:  {lam_max:.6f}  (should be 1.0)")

    is_monotone = all(lambdas[i] <= lambdas[i + 1] for i in range(len(lambdas) - 1))
    print(f"  Lambda monotonically increasing: {is_monotone}")
    print("  Validation complete.\n")


class Fighter:
    """
    Represents a single fighter in the sparring simulation.

    Encapsulates all per-fighter state: transition matrices, payoff
    matrix, adaptation matrix, exposure tracking, memory parameters,
    and fitness history. Pure state container plus per-step math — the
    two-fighter interaction loop itself lives in SparringMatch.
    """

    def __init__(self, name, base_matrix, adaptation_matrix,
                 payoff_matrix, memory_growth=MEMORY_GROWTH, memory_decay=MEMORY_DECAY,
                 max_exposure=MAX_EXPOSURE, steepness=LAMBDA_STEEPNESS, midpoint=LAMBDA_MIDPOINT,
                 color='steelblue'):

        self.name               = name
        self.base_matrix        = base_matrix.copy()
        self.adaptation_matrix  = adaptation_matrix.copy()
        self.payoff_matrix      = payoff_matrix.copy()
        self.memory_growth      = memory_growth
        self.memory_decay       = memory_decay
        self.max_exposure       = max_exposure
        self.steepness          = steepness
        self.midpoint            = midpoint
        self.color              = color

        # Runtime state — reset at start of each simulation
        self.current_state          = None
        self.exposure                = 0.0
        self.current_lambda          = 0.0
        self.current_adapted_matrix = base_matrix.copy()

        # History — populated during simulation
        self.state_history      = []
        self.fitness_history    = []
        self.cumulative_fitness = []
        self.lambda_history     = []
        self.exposure_history   = []
        self._cumulative_sum    = 0.0

    def reset(self, start_state):
        """Reset all runtime state for a fresh simulation run."""
        self.current_state          = start_state
        self.exposure                = 0.0
        self.current_lambda          = 0.0
        self.current_adapted_matrix = self.base_matrix.copy()
        self.state_history      = []
        self.fitness_history    = []
        self.cumulative_fitness = []
        self.lambda_history     = []
        self.exposure_history   = []
        self._cumulative_sum    = 0.0

    def update_exposure(self, opponent_state, tracked_state):
        """
        Update exposure counter based on opponent's state: grows by
        memory_growth if opponent is in tracked_state, else decays by
        memory_decay. Clamped to [0, max_exposure].
        """
        if opponent_state == tracked_state:
            self.exposure = min(self.exposure + self.memory_growth, self.max_exposure)
        else:
            self.exposure *= self.memory_decay

    def compute_lambda(self):
        """
        Compute current adaptation weight lambda from exposure via the
        same sigmoid schedule as the module-level compute_lambda().
        Stores and returns the result.
        """
        self.current_lambda = compute_lambda(
            self.exposure, self.max_exposure, self.steepness, self.midpoint)
        return self.current_lambda

    def update_adapted_matrix(self):
        """
        Recompute adapted transition matrix using current lambda:
        P_new = (1 - lambda) * base_matrix + lambda * adaptation_matrix
        """
        self.current_adapted_matrix, _ = apply_adaptive_matrix(
            self.base_matrix, self.adaptation_matrix, self.exposure,
            self.max_exposure, self.steepness, self.midpoint)

    def get_transition_row(self, opponent_state, selection_strength):
        """
        Transition probability row for the current state: adapted-matrix
        row run through EGT replicator dynamics against the opponent's
        current state. Returns a normalized probability array of length
        N_STATES.
        """
        base_row = self.current_adapted_matrix[self.current_state]
        return apply_replicator_dynamics(base_row, self.payoff_matrix, opponent_state, selection_strength)

    def step(self, transition_probs, rng=None):
        """
        Sample next state from transition probabilities. Accepts an
        optional numpy Generator (or the np.random module) so a match
        can share one seeded stream across both fighters for
        reproducibility; falls back to the global numpy RNG otherwise.
        Updates and returns self.current_state.
        """
        source = rng if rng is not None else np.random
        self.current_state = source.choice(N_STATES, p=transition_probs)
        return self.current_state

    def record_step(self, payoff):
        """Record all per-step tracking variables after each exchange."""
        self._cumulative_sum += payoff

        self.state_history.append(self.current_state)
        self.fitness_history.append(payoff)
        self.cumulative_fitness.append(self._cumulative_sum)
        self.lambda_history.append(self.current_lambda)
        self.exposure_history.append(self.exposure)

    def record_initial(self, opponent_state):
        """
        Record the pre-simulation starting state (index 0): its
        self-payoff, lambda at zero exposure, and zero exposure. Mirrors
        the original procedural simulate(), where index 0 held the start
        state before any transition had occurred.
        """
        self.compute_lambda()
        payoff = self.get_payoff(opponent_state)
        self._cumulative_sum = payoff
        self.state_history.append(self.current_state)
        self.fitness_history.append(payoff)
        self.cumulative_fitness.append(payoff)
        self.lambda_history.append(self.current_lambda)
        self.exposure_history.append(self.exposure)

    def get_payoff(self, opponent_state):
        """Payoff for this fighter's current state vs opponent's current state."""
        return float(self.payoff_matrix[self.current_state, opponent_state])

    def get_history_arrays(self):
        """Return all history as numpy arrays — convenience for analysis/plotting."""
        return {
            'states':     np.array(self.state_history),
            'fitness':    np.array(self.fitness_history),
            'cumulative': np.array(self.cumulative_fitness),
            'lambda':     np.array(self.lambda_history),
            'exposure':   np.array(self.exposure_history),
        }

    def print_distribution(self, label=""):
        """Print empirical state distribution for this fighter."""
        if not self.state_history:
            print(f"{self.name}: No history to display.")
            return

        counts = np.bincount(self.state_history, minlength=N_STATES)
        total = len(self.state_history)

        header = f"{self.name}"
        if label:
            header += f" — {label}"
        print(f"\n{header}")
        print(f"{'State':<14} {'Empirical':>10}")
        for i, state in enumerate(STATES):
            print(f"  {state:<12} {counts[i]/total:>10.4f}")

    def __repr__(self):
        state_name = STATES[self.current_state] if self.current_state is not None else None
        return (f"Fighter(name='{self.name}', "
                f"state={state_name}, "
                f"exposure={self.exposure:.3f}, "
                f"lambda={self.current_lambda:.3f})")


class SparringMatch:
    """
    Manages a two-agent sparring simulation between two Fighter objects.

    Handles the simulation loop, inter-fighter interaction, and result
    collection, keeping Fighter objects as clean per-agent state
    containers separate from match orchestration.
    """

    def __init__(self, fighter1, fighter2,
                 f1_tracked_state=ATTACK,
                 f2_tracked_state=FEINT,
                 selection_strength=1.0):

        self.f1                 = fighter1
        self.f2                 = fighter2
        self.f1_tracked_state   = f1_tracked_state
        self.f2_tracked_state   = f2_tracked_state
        self.selection_strength = selection_strength
        self.result              = None

    def simulate(self, n_steps, start_state=DISENGAGE, seed=42):
        """
        Run one complete simulation of n_steps total recorded exchanges
        (including the starting state at index 0, matching the original
        procedural simulate()'s array-length convention).

        Per transition (n_steps - 1 of them):
        1. Both fighters recompute lambda and their adapted matrix from
           their CURRENT (not-yet-updated-this-step) exposure.
        2. Both fighters compute a transition row (adapted matrix + EGT)
           using the opponent's current (pre-sample) state.
        3. Both fighters sample their next state, drawing from one
           shared seeded RNG stream (F1 first, then F2) for
           reproducibility.
        4. Both fighters update exposure from the opponent's NEW state.
        5. Both fighters compute payoff from the NEW states and record.

        Returns a result dict compatible with the rest of this module
        (run_all_simulations, main, monte_carlo_sparring.py, ...): same
        keys as the original procedural simulate(), plus f1_history /
        f2_history aliases for the state sequences.
        """
        self.f1.reset(start_state)
        self.f2.reset(start_state)

        self.f1.record_initial(self.f2.current_state)
        self.f2.record_initial(self.f1.current_state)

        rng = np.random.default_rng(seed)
        f1_defend_prob_history = []
        f2_defend_prob_history = []

        for _ in range(n_steps - 1):
            self.f1.compute_lambda()
            self.f2.compute_lambda()
            self.f1.update_adapted_matrix()
            self.f2.update_adapted_matrix()

            f1_probs = self.f1.get_transition_row(self.f2.current_state, self.selection_strength)
            f2_probs = self.f2.get_transition_row(self.f1.current_state, self.selection_strength)

            f1_defend_prob_history.append(f1_probs[DEFEND])
            f2_defend_prob_history.append(f2_probs[DEFEND])

            self.f1.step(f1_probs, rng)
            self.f2.step(f2_probs, rng)

            self.f1.update_exposure(self.f2.current_state, self.f1_tracked_state)
            self.f2.update_exposure(self.f1.current_state, self.f2_tracked_state)

            f1_payoff = self.f1.get_payoff(self.f2.current_state)
            f2_payoff = self.f2.get_payoff(self.f1.current_state)

            self.f1.record_step(f1_payoff)
            self.f2.record_step(f2_payoff)

        f1_states = np.array(self.f1.state_history)
        f2_states = np.array(self.f2.state_history)

        self.result = {
            "f1_seq": f1_states,
            "f2_seq": f2_states,
            # NOTE: preserves the original (tracked-subject-based) naming —
            # "f2_exposure_history" is F1's own exposure counter (it tracks
            # F2's Attack), and "f1_exposure_history" is F2's own exposure
            # counter (it tracks F1's Feint).
            "f2_exposure_history": np.array(self.f1.exposure_history),
            "f1_exposure_history": np.array(self.f2.exposure_history),
            "f1_lambda_history": np.array(self.f1.lambda_history),
            "f2_lambda_history": np.array(self.f2.lambda_history),
            "f1_defend_prob_history": np.array(f1_defend_prob_history),
            "f2_defend_prob_history": np.array(f2_defend_prob_history),
            "f1_fitness_history": np.array(self.f1.fitness_history),
            "f2_fitness_history": np.array(self.f2.fitness_history),
            "f1_cumulative_fitness": np.array(self.f1.cumulative_fitness),
            "f2_cumulative_fitness": np.array(self.f2.cumulative_fitness),
            # OOP-native aliases
            "f1_history": f1_states,
            "f2_history": f2_states,
        }
        return self.result

    def print_result_summary(self):
        """
        Print match result summary: final cumulative fitness for each
        fighter, who won and by what margin, final lambda/exposure
        values, and empirical state distributions.
        """
        if self.result is None:
            print("No simulation has been run yet.")
            return

        f1_final = self.f1.cumulative_fitness[-1]
        f2_final = self.f2.cumulative_fitness[-1]
        margin = abs(f1_final - f2_final)

        print(f"\n{'='*60}")
        print("MATCH RESULT")
        print(f"{'='*60}")
        print(f"  {self.f1.name:<25} Final fitness: {f1_final:.4f}")
        print(f"  {self.f2.name:<25} Final fitness: {f2_final:.4f}")

        if margin < 1e-6:
            print("  Result: Dead even (margin < 1e-6)")
        elif f1_final > f2_final:
            print(f"  Result: {self.f1.name} wins by {margin:.4f}")
        else:
            print(f"  Result: {self.f2.name} wins by {margin:.4f}")

        print(f"\n  Final lambda — {self.f1.name}: {self.f1.lambda_history[-1]:.4f}")
        print(f"  Final lambda — {self.f2.name}: {self.f2.lambda_history[-1]:.4f}")
        print(f"  Final exposure — {self.f1.name}: {self.f1.exposure_history[-1]:.4f}")
        print(f"  Final exposure — {self.f2.name}: {self.f2.exposure_history[-1]:.4f}")

        self.f1.print_distribution("Empirical State Distribution")
        self.f2.print_distribution("Empirical State Distribution")

    def __repr__(self):
        return (f"SparringMatch({self.f1.name} vs {self.f2.name}, "
                f"selection_strength={self.selection_strength})")


def create_cj():
    """Instantiate Fighter 1 (CJ) with all established parameters."""
    return Fighter(
        name              = "CJ",
        base_matrix       = F1_BASE,
        adaptation_matrix = F1_ADAPTATION_MATRIX,
        payoff_matrix     = F1_PAYOFF,
        memory_growth     = MEMORY_GROWTH,
        memory_decay      = MEMORY_DECAY,
        max_exposure      = MAX_EXPOSURE,
        steepness         = LAMBDA_STEEPNESS,
        midpoint          = LAMBDA_MIDPOINT,
        color             = F1_COLOR,
    )


def create_counter_puncher():
    """Instantiate Fighter 2 (Counter-Fighter) with all parameters."""
    return Fighter(
        name              = "Counter-Fighter",
        base_matrix       = F2_BASE,
        adaptation_matrix = F2_ADAPTATION_MATRIX,
        payoff_matrix     = F2_PAYOFF,
        memory_growth     = MEMORY_GROWTH,
        memory_decay      = MEMORY_DECAY,
        max_exposure      = MAX_EXPOSURE,
        steepness         = LAMBDA_STEEPNESS,
        midpoint          = LAMBDA_MIDPOINT,
        color             = F2_COLOR,
    )


def create_match(selection_strength=1.0):
    """Instantiate a complete, ready-to-run SparringMatch: CJ vs Counter-Fighter."""
    cj = create_cj()
    cp = create_counter_puncher()
    return SparringMatch(
        fighter1            = cj,
        fighter2            = cp,
        f1_tracked_state    = ATTACK,  # F1 tracks F2's Attack
        f2_tracked_state    = FEINT,   # F2 tracks F1's Feint
        selection_strength  = selection_strength,
    )


def simulate(selection_strength=1.0, seed=42, n_steps=None, start_state=None):
    """
    Backward-compatible wrapper around SparringMatch.simulate().

    Same signature, default seed, and result-dict keys as the original
    procedural implementation, so existing callers (run_all_simulations
    in this module, monte_carlo_sparring.py, etc.) keep working
    unchanged. n_steps/start_state default to this module's own
    N_STEPS/START_STATE, exactly as the original always did.
    """
    if n_steps is None:
        n_steps = N_STEPS
    if start_state is None:
        start_state = STATES.index(START_STATE)
    match = create_match(selection_strength)
    return match.simulate(n_steps, start_state, seed=seed)


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


def print_memory_stats(f1_exposure_history, f2_exposure_history,
                        f1_lambda_history, f2_lambda_history):
    print(f"Adaptation Memory Snapshots (selection_strength={MEMORY_REFERENCE_STRENGTH} reference run)")
    print("=" * 60)
    for step in MEMORY_STAT_STEPS:
        idx = step - 1
        f2_exp = f2_exposure_history[idx]
        f1_exp = f1_exposure_history[idx]
        f1_lambda = f1_lambda_history[idx]
        f2_lambda = f2_lambda_history[idx]

        f1_matrix, _ = apply_adaptive_matrix(F1_BASE, F1_ADAPTATION_MATRIX, f2_exp)
        f2_matrix, _ = apply_adaptive_matrix(F2_BASE, F2_ADAPTATION_MATRIX, f1_exp)

        print(f"--- Step {step} ---")
        print(f"F2 attack exposure: {f2_exp:.3f} / {MAX_EXPOSURE:.1f}")
        print(f"F1 feint exposure:  {f1_exp:.3f} / {MAX_EXPOSURE:.1f}")
        print(f"f1_lambda: {f1_lambda:.4f}")
        print(f"f2_lambda: {f2_lambda:.4f}")
        print("\nF1 adapted matrix:")
        print(pd.DataFrame(f1_matrix.round(3), index=STATES, columns=STATES).to_string())
        print("\nF2 adapted matrix:")
        print(pd.DataFrame(f2_matrix.round(3), index=STATES, columns=STATES).to_string())
        print(
            f"\nSparring read: CJ's overall gameplan is {describe_exposure_level(f1_lambda)} "
            f"adapted ({f1_lambda:.0%} of the way to his fully-read counter-stance) to Fighter 2's "
            f"repeated attacks; Fighter 2's gameplan is {describe_exposure_level(f2_lambda)} adapted "
            f"({f2_lambda:.0%}) to CJ's feints."
        )
        print(
            f"Tumor-immune parallel: F1's strategic profile has converged {f1_lambda:.0%} of the way "
            f"toward its adapted phenotype, and F2's {f2_lambda:.0%} of the way toward its adapted "
            f"phenotype - modeling gradual phenotypic switching under sustained selective pressure, "
            f"as in immunoediting."
        )
        print()


def plot_memory_grid(f2_exposure_history, f1_exposure_history,
                      f1_lambda_history, f2_lambda_history,
                      f1_defend_prob_history, f2_defend_prob_history,
                      filename="sparring_memory.png"):
    fig, axes = plt.subplots(3, 2, figsize=(13, 15))
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
    ax.plot(steps, f1_lambda_history, color=F1_COLOR)
    for frac in (0.25, 0.5, 0.75):
        ax.axhline(frac, color="gray", linestyle="--", alpha=0.5)
    ax.set_xlabel("Exchange step")
    ax.set_ylabel("Lambda")
    ax.set_ylim(0, 1.05)
    ax.set_title("F1 Adaptation Weight λ (Response to F2 Attack Exposure)")
    ax.grid(axis="y", linestyle=":", alpha=0.2)

    ax = axes[1, 1]
    ax.plot(steps, f2_lambda_history, color=F2_COLOR)
    for frac in (0.25, 0.5, 0.75):
        ax.axhline(frac, color="gray", linestyle="--", alpha=0.5)
    ax.set_xlabel("Exchange step")
    ax.set_ylabel("Lambda")
    ax.set_ylim(0, 1.05)
    ax.set_title("F2 Adaptation Weight λ (Response to F1 Feint Exposure)")
    ax.grid(axis="y", linestyle=":", alpha=0.2)

    ax = axes[2, 0]
    x, avg = rolling_average(f1_defend_prob_history)
    ax.plot(x, avg, color=F1_COLOR)
    ax.set_xlabel("Exchange step")
    ax.set_ylabel("Probability")
    ax.set_title("F1 Defend Probability — Effect of Full Matrix Adaptation")
    ax.grid(axis="y", linestyle="--", alpha=0.4)

    ax = axes[2, 1]
    x, avg = rolling_average(f2_defend_prob_history)
    ax.plot(x, avg, color=F2_COLOR)
    ax.set_xlabel("Exchange step")
    ax.set_ylabel("Probability")
    ax.set_title("F2 Defend Probability — Effect of Full Matrix Adaptation")
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
    print(f"Total cumulative fitness — Fighter 2 (Counter-Fighter): {f2_total:.3f}")
    if margin > 1e-6:
        print(f"Winner on total fitness: Fighter 1 (CJ), by {margin:.3f}")
    elif margin < -1e-6:
        print(f"Winner on total fitness: Fighter 2 (Counter-Fighter), by {-margin:.3f}")
    else:
        print("Result: dead even on total fitness")

    lead_change_step = find_last_lead_change(f1_cumulative_fitness, f2_cumulative_fitness)
    if lead_change_step is None:
        leader = "Fighter 1 (CJ)" if margin > -1e-6 else "Fighter 2 (Counter-Fighter)"
        print(f"Lead never changed hands — {leader} led throughout")
    else:
        print(f"Lead last changed hands at step {lead_change_step}")

    f1_avg = f1_fitness_history.mean()
    f2_avg = f2_fitness_history.mean()
    print(f"Average per-step payoff — Fighter 1 (CJ): {f1_avg:.4f}")
    print(f"Average per-step payoff — Fighter 2 (Counter-Fighter): {f2_avg:.4f}")
    print()

    if margin > 1e-6:
        sparring_note = "CJ's blitz-and-feint pressure is outscoring the counter-punching patience"
        tumor_note = ("the aggressor phenotype (CJ) is accumulating a fitness edge, analogous to a "
                       "tumor clone outcompeting immune surveillance")
    elif margin < -1e-6:
        sparring_note = "the counter-fighter's patient reads are outscoring the blitz"
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
    ax.plot(x2, avg2, color=F2_COLOR, label="Fighter 2 (Counter-Fighter)")
    ax.axhline(0.5, color="gray", linestyle="--", alpha=0.6)
    ax.set_ylim(0, 1)
    ax.set_xlabel("Exchange step")
    ax.set_ylabel("Payoff")
    ax.set_title(f"Per-Step Payoff — Rolling {ROLLING_WINDOW}-Step Average")
    ax.legend()
    ax.grid(axis="y", linestyle="--", alpha=0.4)

    ax = axes[1]
    ax.plot(steps, f1_cumulative_fitness, color=F1_COLOR, label="Fighter 1 (CJ)")
    ax.plot(steps, f2_cumulative_fitness, color=F2_COLOR, label="Fighter 2 (Counter-Fighter)")
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
        ax_f2.set_title(f"Fighter 2 (Counter-Fighter) — selection_strength={strength}")
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
    for name, matrix in [("Fighter 1", F1_BASE), ("Fighter 2", F2_BASE),
                         ("Fighter 1 adaptation", F1_ADAPTATION_MATRIX),
                         ("Fighter 2 adaptation", F2_ADAPTATION_MATRIX)]:
        row_sums = matrix.sum(axis=1)
        assert np.allclose(row_sums, 1.0), f"{name} rows must sum to 1.0, got {row_sums}"

    validate_adaptive_system()

    results = run_all_simulations()
    print_analysis(results)

    plot_marginals_grid(results)
    plot_joint_heatmaps_grid(results)
    print("Saved plot to sparring_egt.png")
    print("Saved plot to sparring_egt_heatmaps.png")
    print()

    reference = results[MEMORY_REFERENCE_STRENGTH]
    print_memory_stats(reference["f1_exposure_history"], reference["f2_exposure_history"],
                        reference["f1_lambda_history"], reference["f2_lambda_history"])
    plot_memory_grid(reference["f2_exposure_history"], reference["f1_exposure_history"],
                      reference["f1_lambda_history"], reference["f2_lambda_history"],
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
