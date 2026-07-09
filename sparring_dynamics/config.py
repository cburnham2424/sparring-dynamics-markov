"""
Central configuration for the sparring dynamics simulation.
All tunable parameters live here — no magic numbers elsewhere.
"""
import numpy as np

# ── States ──────────────────────────────────────────────────
STATES = ['Attack', 'Defend', 'Disengage', 'Feint']
STATE_INDEX = {s: i for i, s in enumerate(STATES)}
N_STATES = len(STATES)

ATTACK    = 0
DEFEND    = 1
DISENGAGE = 2
FEINT     = 3

# ── Simulation defaults ──────────────────────────────────────
DEFAULT_N_STEPS          = 500
DEFAULT_START_STATE      = DISENGAGE
DEFAULT_SELECTION        = 1.0
DEFAULT_N_SIMULATIONS    = 500
DEFAULT_CONFIDENCE       = 0.95
DEFAULT_RANDOM_SEED      = 42

# ── Memory / adaptation defaults ────────────────────────────
DEFAULT_MEMORY_GROWTH    = 1.5
DEFAULT_MEMORY_DECAY     = 0.95
DEFAULT_MAX_EXPOSURE     = 10.0
DEFAULT_STEEPNESS        = 0.6
DEFAULT_MIDPOINT         = 0.5

# ── Estimation defaults ──────────────────────────────────────
DEFAULT_TRANSITION_ALPHA = 1.0
DEFAULT_PAYOFF_ALPHA     = 0.5
DEFAULT_MIN_OBS          = 5
DEFAULT_MAX_POINTS       = 4

# ── Fallback hand-crafted matrices (used when CSV is absent) ─
F1_BASE_DEFAULT = np.array([
    [0.25, 0.15, 0.35, 0.25],
    [0.45, 0.10, 0.25, 0.20],
    [0.30, 0.10, 0.20, 0.40],
    [0.60, 0.08, 0.17, 0.15]
])

F2_BASE_DEFAULT = np.array([
    [0.15, 0.30, 0.35, 0.20],
    [0.55, 0.15, 0.20, 0.10],
    [0.15, 0.30, 0.25, 0.30],
    [0.40, 0.25, 0.20, 0.15]
])

F1_ADAPTATION_DEFAULT = np.array([
    [0.15, 0.35, 0.25, 0.25],
    [0.55, 0.08, 0.17, 0.20],
    [0.25, 0.20, 0.15, 0.40],
    [0.65, 0.10, 0.10, 0.15]
])

F2_ADAPTATION_DEFAULT = np.array([
    [0.10, 0.45, 0.30, 0.15],
    [0.60, 0.18, 0.12, 0.10],
    [0.10, 0.45, 0.15, 0.30],
    [0.35, 0.35, 0.15, 0.15]
])

F1_PAYOFF_DEFAULT = np.array([
    [0.2, 0.8, 0.5, 0.9],
    [0.6, 0.1, 0.3, 0.4],
    [0.3, 0.4, 0.2, 0.6],
    [0.5, 0.7, 0.4, 0.3]
])

F2_PAYOFF_DEFAULT = np.array([
    [0.7, 0.2, 0.8, 0.3],
    [0.8, 0.1, 0.2, 0.6],
    [0.2, 0.5, 0.3, 0.4],
    [0.4, 0.3, 0.7, 0.2]
])

# ── Visualization ────────────────────────────────────────────
F1_COLOR = 'steelblue'
F2_COLOR = 'coral'
FIGURE_DPI = 150
OUTPUT_DIR = 'outputs'
