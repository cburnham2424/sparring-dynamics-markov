"""
Fighter: per-agent state container plus the per-step math (EGT
replicator dynamics + sigmoid adaptive-matrix blending) that turns a
fighter's current state and exposure into a next-state distribution.
"""
import numpy as np

from sparring_dynamics.config import (
    N_STATES,
    DEFAULT_MEMORY_GROWTH, DEFAULT_MEMORY_DECAY, DEFAULT_MAX_EXPOSURE,
    DEFAULT_STEEPNESS, DEFAULT_MIDPOINT,
    STATES,
)


def apply_replicator_dynamics(base_probs, payoff_matrix, opponent_state, selection_strength=1.0):
    """
    Modify transition probabilities using replicator dynamics.

    fitness[i] = payoff_matrix[i, opponent_state]
    mean_fitness = sum(base_probs * fitness)
    new_prob[i] = base_probs[i] * (1 + selection_strength * (fitness[i] - mean_fitness))
    Clipped to >=0 and renormalized to sum to 1.0.
    """
    fitness = payoff_matrix[:, opponent_state]
    mean_fitness = np.sum(base_probs * fitness)
    new_probs = base_probs * (1 + selection_strength * (fitness - mean_fitness))
    new_probs = np.clip(new_probs, 0, None)
    return new_probs / new_probs.sum()


def compute_lambda(exposure, max_exposure=DEFAULT_MAX_EXPOSURE,
                    steepness=DEFAULT_STEEPNESS, midpoint=DEFAULT_MIDPOINT):
    """
    Smooth sigmoid lambda schedule based on exposure ratio, rescaled so
    lambda(ratio=0) = 0 and lambda(ratio=1) = 1 exactly.
    """
    k = steepness * 10
    ratio = np.clip(exposure / max_exposure, 0.0, 1.0)

    sigmoid = lambda r: 1.0 / (1.0 + np.exp(-k * (r - midpoint)))

    lambda_raw = sigmoid(ratio)
    lambda_min = sigmoid(0.0)
    lambda_max = sigmoid(1.0)

    lam = (lambda_raw - lambda_min) / (lambda_max - lambda_min)
    return float(np.clip(lam, 0.0, 1.0))


def apply_adaptive_matrix(base_matrix, adaptation_matrix, exposure,
                           max_exposure=DEFAULT_MAX_EXPOSURE,
                           steepness=DEFAULT_STEEPNESS, midpoint=DEFAULT_MIDPOINT):
    """
    Blend a fighter's entire base transition matrix toward its
    fully-adapted target matrix: P_new = (1-lambda)*P_base + lambda*P_adaptation.
    """
    lam = compute_lambda(exposure, max_exposure, steepness, midpoint)

    P_new = (1.0 - lam) * base_matrix + lam * adaptation_matrix

    row_sums = P_new.sum(axis=1, keepdims=True)
    row_sums = np.where(row_sums < 1e-10, 1.0, row_sums)
    P_new = P_new / row_sums

    P_new = np.clip(P_new, 0.0, None)
    P_new = P_new / P_new.sum(axis=1, keepdims=True)

    return P_new, lam


class Fighter:
    """
    Represents a single fighter in the sparring simulation.

    Encapsulates all per-fighter state: transition matrices, payoff
    matrix, adaptation matrix, exposure tracking, memory parameters,
    and fitness history.
    """

    def __init__(self, name, base_matrix, adaptation_matrix,
                 payoff_matrix, memory_growth=DEFAULT_MEMORY_GROWTH, memory_decay=DEFAULT_MEMORY_DECAY,
                 max_exposure=DEFAULT_MAX_EXPOSURE, steepness=DEFAULT_STEEPNESS, midpoint=DEFAULT_MIDPOINT,
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

    @classmethod
    def from_matrices(cls, name, base_matrix, adaptation_matrix,
                       payoff_matrix, color='steelblue', **kwargs):
        """
        Construct a Fighter from explicit matrix arguments. All
        memory/sigmoid parameters use config defaults unless overridden
        via kwargs (memory_growth, memory_decay, max_exposure,
        steepness, midpoint).
        """
        return cls(
            name=name,
            base_matrix=base_matrix,
            adaptation_matrix=adaptation_matrix,
            payoff_matrix=payoff_matrix,
            memory_growth=kwargs.get('memory_growth', DEFAULT_MEMORY_GROWTH),
            memory_decay=kwargs.get('memory_decay', DEFAULT_MEMORY_DECAY),
            max_exposure=kwargs.get('max_exposure', DEFAULT_MAX_EXPOSURE),
            steepness=kwargs.get('steepness', DEFAULT_STEEPNESS),
            midpoint=kwargs.get('midpoint', DEFAULT_MIDPOINT),
            color=color
        )

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
        """Compute current adaptation weight lambda from exposure. Stores and returns it."""
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
        self-payoff, lambda at zero exposure, and zero exposure.
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
