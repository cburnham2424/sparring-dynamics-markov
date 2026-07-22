# Sparring Dynamics Markov

Traditional Taekwondo point sparring is a sequential decision-making process in which fighters continuously adapt their tactics based on previous exchanges and opponent behavior. As a second-degree black belt and multiple-time world champion in point sparring, I have experienced this adaptive process firsthand, in which matches are often decided by pattern recognition and a single well-timed tactical adjustment.
This project translates those adaptive dynamics into a data-driven mathematical framework. Using Markov chains, evolutionary game theory, and adaptive state-transition dynamics, it models a sparring match as a stochastic system in which fighters evolve their tactical decisions over repeated exchanges. Rather than relying on fixed assumptions, the framework is designed to learn fighter-specific transition probabilities and scoring effectiveness from annotated sparring footage, allowing simulations to become increasingly representative of real competitive behavior as additional match data is incorporated.
The model combines learned transition matrices, payoff estimation, adaptive memory, Monte Carlo simulation, and statistical validation to study how strategies emerge, adapt, and succeed against different opponent styles. While developed using Taekwondo as an observable competitive system, the underlying mathematics is applicable to a broad class of adaptive stochastic systems. The same concepts of state transitions, evolutionary selection, and adaptive dynamics appear in fields ranging from behavioral modeling to mathematical oncology, in which similar frameworks are used to study tumor evolution, treatment resistance, and population dynamics. This repository represents both an applied sports analytics project and the foundation for future research in adaptive mathematical modeling.

## Getting Started

### Requirements
- Python 3.9+
- pip

### Installation

```bash
git clone https://github.com/cburnham2424/sparring-dynamics-markov.git
cd sparring-dynamics-markov
pip install -r requirements.txt
```

### Run the full pipeline (uses hand-crafted defaults)

```bash
python scripts/run_pipeline.py --no-estimate --n-sims 50 --n-steps 200
```

### Run with real match annotation data

The `--csv` flag currently supports only the legacy
`f1_state,f2_state,winner,f1_points,f2_points` schema; it does not yet
parse the richer schema defined in `annotation_format.py` and used by
`data/processed/combined_annotations.csv` (see Data Status below).
Until this integration is implemented, load the annotated dataset
directly through the estimation API:

```bash
python3 <<'EOF'
from sparring_dynamics.data.annotation_format import (
    load_annotation_csv, annotations_to_exchanges, annotations_to_sequences,
)
from sparring_dynamics.estimation.transitions import (
    estimate_both_transition_matrices, create_hybrid_transition_matrix,
)
from sparring_dynamics.estimation.payoffs import (
    estimate_payoff_matrices, create_hybrid_payoff_matrix,
)
from sparring_dynamics.simulation.fighter import Fighter
from sparring_dynamics.simulation.match import SparringMatch
from sparring_dynamics.analysis.monte_carlo import run_monte_carlo, print_summary, analyze_monte_carlo
from sparring_dynamics.config import (
    DEFAULT_START_STATE, F1_BASE_DEFAULT, F2_BASE_DEFAULT,
    F1_PAYOFF_DEFAULT, F2_PAYOFF_DEFAULT,
)

annotations, _ = load_annotation_csv('data/processed/combined_annotations.csv', strict=False)
exchanges = annotations_to_exchanges(annotations)
f1_seqs, f2_seqs = annotations_to_sequences(annotations)

trans = estimate_both_transition_matrices(f1_seqs, f2_seqs)
payoff = estimate_payoff_matrices(exchanges)

f1_base, _ = create_hybrid_transition_matrix(trans['f1_matrix'], trans['f1_counts'], F1_BASE_DEFAULT)
f2_base, _ = create_hybrid_transition_matrix(trans['f2_matrix'], trans['f2_counts'], F2_BASE_DEFAULT)
f1_payoff, _ = create_hybrid_payoff_matrix(payoff['f1_matrix'], payoff['totals'], F1_PAYOFF_DEFAULT)
f2_payoff, _ = create_hybrid_payoff_matrix(payoff['f2_matrix'], payoff['totals'], F2_PAYOFF_DEFAULT)

f1 = Fighter.from_matrices('CJ', f1_base, f1_base, f1_payoff, color='crimson')
f2 = Fighter.from_matrices('Counter-Fighter', f2_base, f2_base, f2_payoff, color='steelblue')
match = SparringMatch(f1, f2)

results = run_monte_carlo(match, n_simulations=100, n_steps=500, start_state=DEFAULT_START_STATE, random_seed=42)
analysis = analyze_monte_carlo(results)
print_summary(results, analysis)
EOF
```

### Generate a blank annotation template for new match footage

```bash
python scripts/run_pipeline.py --template
```

### Run the test suite

```bash
python -m pytest tests/ -v
```

### Run sensitivity analysis (parameter sweep)

```bash
python scripts/run_sensitivity.py --n-sims-1d 50 --n-sims-2d 20
```

### Run a standalone Monte Carlo simulation (hand-crafted defaults only)

```bash
python scripts/run_monte_carlo.py --n-sims 100 --n-steps 300
```

### What This Toolchain Produces

Running the analysis modules above (individually or in combination)
generates the following:
- Estimated transition matrices from annotated match footage (or
  hand-crafted defaults if no CSV is provided)
- Estimated payoff matrices from scored exchanges
- Monte Carlo simulation results across N independent match
  realizations with 95% confidence intervals
- Sensitivity analysis heatmaps sweeping selection strength, memory
  decay, memory growth, and adaptation rate
- Validation reports comparing simulated dynamics to observed match
  data using Jensen-Shannon divergence, KL divergence, and RMSE
- Experiment logs saved to `outputs/experiments/` for full
  reproducibility
- Publication-quality plots saved to `outputs/figures/`
- Sensitivity sweep data (CSV/JSON) saved to `outputs/simulations/sensitivity/`

## Data Status — Current

Six matches annotated, 87 exchanges total:
- `MATCH_20260613_M1`: 14 exchanges, 6-3 (post-break, June 2026)
- `MATCH_20260613_M2`: 9 exchanges, 7-2 (post-break, June 2026)
- `MATCH_20230700_M1`: 8 exchanges, 6-3 (pre-break, 2023)
- `MATCH_20230700_M2`: 20 exchanges, 4-3 (pre-break, 2023; same opponent as `MATCH_20260613_M1`, confirmed by the athlete)
- `MATCH_20230700_M3`: 13 exchanges, 5-2 (pre-break, 2023; different opponent)
- `MATCH_20260711_M1`: 23 exchanges, 7-1 (post-break, July 2026 Worlds)

**Empirically grounded transition rows — F1 (CJ):**
- Attack: 32 observations — data-driven
- Defend: 14 observations — data-driven
- Disengage: 24 observations — data-driven
- Feint: 11 observations — data-driven (note: includes consolidated multi-event windows from 2fps footage — treat with caution)

All four F1 tactical states are now empirically grounded for the first time.

**Empirically grounded transition rows — F2:**
- Attack: 15 observations — data-driven
- Defend: 38 observations — data-driven
- Disengage: 24 observations — data-driven
- Feint: 4 observations — hand-crafted default

**Empirically grounded payoff cells (4 of 16):**
- (Attack, Defend): learned
- (Defend, Attack): learned
- (Disengage, Disengage): learned
- (Feint, Defend): learned
- Remaining 12 cells: hand-crafted defaults

**Key findings:**
- Attack→Disengage: 0.60 pre-break vs. 0.17 post-break (diff=0.43) — CJ reset distance far more frequently after attacking in 2023 peak training than in 2026 post-break return
- Feint→Attack persistence: 0.667 — once CJ commits to a feint sequence the most likely next state is a full attack, consistent with the cut-kick-to-roundhouse combo pattern observed in competition
- New state pair: Feint vs Feint added in match 6 (simultaneous deception — both fighters probing without committing)

**Methodological notes:**
- Round field repurposed as match boundary marker in combined dataset
- Feint row includes consolidated multi-event windows due to 2fps footage limitations in matches 5 and 6
- Opponent identity in the cross-year comparison confirmed by the athlete, not by footage analysis
- Sample size insufficient for statistical inference — dataset is a research prototype pending additional match annotation
