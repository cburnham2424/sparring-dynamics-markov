"""
SparringMatch: manages a two-agent sparring simulation between two
Fighter objects — the interaction loop, shared seeded RNG, and result
collection.
"""
import numpy as np

from sparring_dynamics.config import ATTACK, DEFEND, FEINT, DISENGAGE, DEFAULT_RANDOM_SEED


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

    def simulate(self, n_steps, start_state=DISENGAGE, seed=DEFAULT_RANDOM_SEED):
        """
        Run one complete simulation of n_steps total recorded exchanges
        (including the starting state at index 0).

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

        Returns a result dict: f1_seq/f2_seq (+ f1_history/f2_history
        aliases), f1/f2_exposure_history, f1/f2_lambda_history,
        f1/f2_defend_prob_history, f1/f2_fitness_history,
        f1/f2_cumulative_fitness.
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
            # NOTE: naming is tracked-subject-based, matching this project's
            # established convention — "f2_exposure_history" is F1's own
            # exposure counter (it tracks F2's Attack), and
            # "f1_exposure_history" is F2's own exposure counter (it tracks
            # F1's Feint).
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
