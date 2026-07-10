"""
Validation framework: compares simulated sparring behavior (from Monte
Carlo runs) against observed (real or placeholder) sparring data, using
transition-frequency error, RMSE, Jensen-Shannon divergence, and KL
divergence, rolled up into a composite fit-quality score per matchup.
"""
import os

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from scipy.spatial.distance import jensenshannon
from scipy.stats import entropy

from sparring_dynamics.config import (
    STATES, N_STATES, STATE_INDEX,
    F1_COLOR, F2_COLOR, FIGURE_DPI, OUTPUT_DIR
)

# ── Dark mode theme ──────────────────────────────────────────
BG_COLOR = '#1a1a19'
GRID_COLOR = '#2c2c2a'
TICK_COLOR = '#c3c2b7'
TITLE_COLOR = '#ffffff'

plt.style.use('dark_background')
plt.rcParams.update({
    'figure.facecolor': BG_COLOR,
    'axes.facecolor': BG_COLOR,
    'savefig.facecolor': BG_COLOR,
    'axes.edgecolor': GRID_COLOR,
    'axes.labelcolor': TICK_COLOR,
    'axes.titlecolor': TITLE_COLOR,
    'xtick.color': TICK_COLOR,
    'ytick.color': TICK_COLOR,
    'text.color': TITLE_COLOR,
    'axes.grid': True,
    'grid.color': GRID_COLOR,
    'grid.alpha': 0.5,
})


def _darken_figure(fig, axes):
    fig.patch.set_facecolor(BG_COLOR)
    for ax in np.atleast_1d(axes).flat:
        ax.set_facecolor(BG_COLOR)


# ---------------------------------------------------------------------------
# SECTION 1 — Observed data structures
# ---------------------------------------------------------------------------

class ObservedSparringData:
    """
    Container for observed (real or placeholder) sparring data.
    Stores raw state sequences and computes derived statistics needed
    for validation.
    """

    def __init__(self, fighter_name, sequences, opponent_name="Unknown"):
        self.fighter_name  = fighter_name
        self.opponent_name = opponent_name
        self.sequences     = sequences

        self.transition_matrix = self._compute_transition_matrix()
        self.state_occupancy   = self._compute_state_occupancy()
        self.n_transitions     = sum(
            len(s) - 1 for s in sequences if len(s) > 1
        )
        self.n_sequences       = len(sequences)

    def _compute_transition_matrix(self):
        """
        MLE transition matrix from observed sequences. Uses Laplace
        smoothing alpha=0.1 (near-zero — we want to preserve observed
        structure here, not regularize heavily).
        """
        alpha  = 0.1
        counts = np.zeros((N_STATES, N_STATES))
        for seq in self.sequences:
            for t in range(len(seq) - 1):
                i = STATE_INDEX[seq[t]]
                j = STATE_INDEX[seq[t + 1]]
                counts[i, j] += 1

        smoothed = counts + alpha
        row_sums = smoothed.sum(axis=1, keepdims=True)
        row_sums = np.where(row_sums < 1e-10, 1.0, row_sums)
        return smoothed / row_sums

    def _compute_state_occupancy(self):
        """Fraction of time in each state across all sequences, summing to 1.0."""
        counts = np.zeros(N_STATES)
        total  = 0
        for seq in self.sequences:
            for state in seq:
                counts[STATE_INDEX[state]] += 1
                total += 1
        if total == 0:
            return np.ones(N_STATES) / N_STATES
        return counts / total

    def __repr__(self):
        return (f"ObservedSparringData("
                f"fighter='{self.fighter_name}', "
                f"opponent='{self.opponent_name}', "
                f"sequences={self.n_sequences}, "
                f"transitions={self.n_transitions})")


# ---------------------------------------------------------------------------
# SECTION 2 — Core metric functions
# ---------------------------------------------------------------------------

def transition_frequency_error(simulated_matrix, observed_matrix):
    """
    Element-wise absolute error between simulated and observed
    transition matrices.

    Returns error_matrix, mean_error, max_error, worst_pair, row_errors.
    """
    error_matrix = np.abs(simulated_matrix - observed_matrix)
    mean_error   = error_matrix.mean()
    max_idx      = np.unravel_index(error_matrix.argmax(), error_matrix.shape)
    worst_pair   = (STATES[max_idx[0]], STATES[max_idx[1]])
    row_errors   = error_matrix.mean(axis=1)

    return {
        'error_matrix': error_matrix,
        'mean_error':   float(mean_error),
        'max_error':    float(error_matrix.max()),
        'worst_pair':   worst_pair,
        'row_errors':   row_errors
    }


def state_occupancy_error(simulated_occupancy, observed_occupancy):
    """
    Compare simulated vs observed state occupancy distributions.

    Returns absolute_error, mean_error, max_error, worst_state,
    relative_error.
    """
    abs_error = np.abs(simulated_occupancy - observed_occupancy)
    max_idx   = abs_error.argmax()

    # Relative error — avoid division by zero for unseen states
    denom    = np.where(observed_occupancy > 1e-8,
                         observed_occupancy, 1e-8)
    rel_err  = abs_error / denom

    return {
        'absolute_error': abs_error,
        'mean_error':     float(abs_error.mean()),
        'max_error':      float(abs_error.max()),
        'worst_state':    STATES[max_idx],
        'relative_error': rel_err
    }


def compute_rmse(simulated_matrix, observed_matrix):
    """
    Root Mean Square Error between simulated and observed transition
    matrices: RMSE = sqrt(mean((P_sim - P_obs)^2)). Also computes
    per-row RMSE.
    """
    diff      = simulated_matrix - observed_matrix
    squared   = diff ** 2
    rmse      = float(np.sqrt(squared.mean()))
    row_rmse  = np.sqrt(squared.mean(axis=1))

    return {
        'rmse':      rmse,
        'row_rmse':  row_rmse,
        'worst_row': STATES[row_rmse.argmax()]
    }


def jensen_shannon_divergence(simulated_occupancy, observed_occupancy):
    """
    Jensen-Shannon divergence between simulated and observed state
    occupancy distributions.

    JSD is a symmetric measure of distributional similarity, bounded
    in [0,1] when computed with a base-2 logarithm (JSD=0 identical,
    JSD=1 maximally different) — scipy.spatial.distance.jensenshannon
    defaults to natural log (bounded [0, ln 2] instead), so base=2 is
    passed explicitly to keep this contract honest and make the
    interpretation thresholds below meaningful on a [0,1] scale.

    scipy's jensenshannon returns the JS *distance* (sqrt of the
    divergence), so it's squared to recover the divergence itself.

    Returns jsd, jsd_distance, interpretation.
    """
    eps = 1e-10
    p   = simulated_occupancy + eps
    q   = observed_occupancy  + eps
    p   = p / p.sum()
    q   = q / q.sum()

    jsd_dist = float(jensenshannon(p, q, base=2))
    jsd      = jsd_dist ** 2

    if jsd < 0.01:
        interpretation = "Excellent fit — distributions nearly identical"
    elif jsd < 0.05:
        interpretation = "Good fit — minor distributional differences"
    elif jsd < 0.10:
        interpretation = "Moderate fit — noticeable differences"
    elif jsd < 0.20:
        interpretation = "Poor fit — substantial differences"
    else:
        interpretation = "Very poor fit — distributions are very different"

    return {
        'jsd':            jsd,
        'jsd_distance':   jsd_dist,
        'interpretation': interpretation
    }


def compute_row_jsd(simulated_matrix, observed_matrix):
    """
    Jensen-Shannon divergence for each row of the transition matrix
    independently (base=2, see jensen_shannon_divergence for why).

    Returns row_jsd, mean_jsd, worst_row.
    """
    eps     = 1e-10
    row_jsd = np.zeros(N_STATES)

    for i in range(N_STATES):
        p = simulated_matrix[i] + eps
        q = observed_matrix[i]  + eps
        p = p / p.sum()
        q = q / q.sum()
        row_jsd[i] = jensenshannon(p, q, base=2) ** 2

    return {
        'row_jsd':  row_jsd,
        'mean_jsd': float(row_jsd.mean()),
        'worst_row': STATES[row_jsd.argmax()]
    }


def kl_divergence(simulated_occupancy, observed_occupancy):
    """
    KL divergence KL(observed || simulated): how many extra bits are
    needed to encode samples from the observed distribution using the
    simulated distribution as a code. Lower is better. Asymmetric —
    KL(obs||sim) is the relevant direction for model validation (how
    well the simulation approximates the true observed distribution).

    Returns kl_obs_sim, kl_sim_obs (reverse direction), interpretation.
    """
    eps = 1e-10
    p   = observed_occupancy   + eps
    q   = simulated_occupancy  + eps
    p   = p / p.sum()
    q   = q / q.sum()

    kl_obs_sim = float(entropy(p, q))  # KL(obs || sim)
    kl_sim_obs = float(entropy(q, p))  # KL(sim || obs)

    if kl_obs_sim < 0.01:
        interpretation = "Excellent — simulation closely approximates observed"
    elif kl_obs_sim < 0.05:
        interpretation = "Good — minor divergence from observed"
    elif kl_obs_sim < 0.15:
        interpretation = "Moderate — meaningful divergence"
    else:
        interpretation = "High — simulation may not represent observed dynamics"

    return {
        'kl_obs_sim':     kl_obs_sim,
        'kl_sim_obs':     kl_sim_obs,
        'interpretation': interpretation
    }


def compute_row_kl(simulated_matrix, observed_matrix):
    """KL(observed_row_i || simulated_row_i) for each row i."""
    eps    = 1e-10
    row_kl = np.zeros(N_STATES)

    for i in range(N_STATES):
        p = observed_matrix[i]   + eps
        q = simulated_matrix[i]  + eps
        p = p / p.sum()
        q = q / q.sum()
        row_kl[i] = float(entropy(p, q))

    return {
        'row_kl':   row_kl,
        'mean_kl':  float(row_kl.mean()),
        'worst_row': STATES[row_kl.argmax()]
    }


# ---------------------------------------------------------------------------
# SECTION 3 — Full validation report
# ---------------------------------------------------------------------------

class ValidationReport:
    """
    Complete validation comparing one fighter's simulated behavior
    against observed data for a specific opponent matchup.
    """

    def __init__(self, fighter_name, opponent_name,
                 simulated_matrix, simulated_occupancy,
                 observed_data, n_simulations=500):

        self.fighter_name        = fighter_name
        self.opponent_name       = opponent_name
        self.simulated_matrix    = simulated_matrix
        self.simulated_occupancy = simulated_occupancy
        self.observed            = observed_data
        self.n_simulations       = n_simulations

        self.tf_error   = transition_frequency_error(
            simulated_matrix, observed_data.transition_matrix
        )
        self.occ_error  = state_occupancy_error(
            simulated_occupancy, observed_data.state_occupancy
        )
        self.rmse_result = compute_rmse(
            simulated_matrix, observed_data.transition_matrix
        )
        self.jsd_occ    = jensen_shannon_divergence(
            simulated_occupancy, observed_data.state_occupancy
        )
        self.jsd_rows   = compute_row_jsd(
            simulated_matrix, observed_data.transition_matrix
        )
        self.kl_occ     = kl_divergence(
            simulated_occupancy, observed_data.state_occupancy
        )
        self.kl_rows    = compute_row_kl(
            simulated_matrix, observed_data.transition_matrix
        )

    def overall_score(self):
        """
        Composite validation score in [0, 1] where 1.0 = perfect match.

        Weighted combination (weights sum to 1.0, so with all-non-negative
        metrics the result is guaranteed in [0,1]):
        - Occupancy JSD:          weight 0.30
        - Transition MAE:         weight 0.25
        - Transition RMSE:        weight 0.20
        - Occupancy KL(obs||sim): weight 0.15
        - Row JSD mean:           weight 0.10

        Each metric normalized via score_i = max(0, 1 - metric_i / threshold_i),
        threshold being the value at which we'd call fit "poor".
        """
        thresholds = {
            'occ_jsd':     0.20,
            'trans_mae':   0.15,
            'trans_rmse':  0.15,
            'kl':          0.30,
            'row_jsd':     0.20
        }
        weights = {
            'occ_jsd':     0.30,
            'trans_mae':   0.25,
            'trans_rmse':  0.20,
            'kl':          0.15,
            'row_jsd':     0.10
        }

        raw = {
            'occ_jsd':    self.jsd_occ['jsd'],
            'trans_mae':  self.tf_error['mean_error'],
            'trans_rmse': self.rmse_result['rmse'],
            'kl':         self.kl_occ['kl_obs_sim'],
            'row_jsd':    self.jsd_rows['mean_jsd']
        }

        score = 0.0
        for key in weights:
            component = max(0.0, 1.0 - raw[key] / thresholds[key])
            score    += weights[key] * component

        return float(score)

    def quality_label(self):
        score = self.overall_score()
        if score >= 0.85:
            return "Excellent"
        elif score >= 0.70:
            return "Good"
        elif score >= 0.50:
            return "Moderate"
        elif score >= 0.30:
            return "Poor"
        else:
            return "Very Poor"

    def print_report(self):
        """Print comprehensive validation report with all metrics and a composite score."""
        print(f"\n{'='*65}")
        print(f"VALIDATION REPORT — {self.fighter_name} vs {self.opponent_name}")
        print(f"{'='*65}")
        print(f"  Simulated from:  {self.n_simulations} Monte Carlo runs")
        print(f"  Observed from:   {self.observed.n_sequences} sequences "
              f"({self.observed.n_transitions} transitions)")

        print(f"\n── Transition Frequency Error ────────────────────────")
        print(f"  Mean Absolute Error: {self.tf_error['mean_error']:.4f}")
        print(f"  Max Absolute Error:  {self.tf_error['max_error']:.4f}")
        print(f"  Worst transition:    "
              f"{self.tf_error['worst_pair'][0]} → "
              f"{self.tf_error['worst_pair'][1]}")
        print(f"  Per-state MAE:")
        for i, state in enumerate(STATES):
            bar = '█' * int(self.tf_error['row_errors'][i] * 40)
            print(f"    {state:12s}: {self.tf_error['row_errors'][i]:.4f}  {bar}")

        print(f"\n── State Occupancy Error ─────────────────────────────")
        print(f"  Mean Absolute Error: {self.occ_error['mean_error']:.4f}")
        print(f"  Max Absolute Error:  {self.occ_error['max_error']:.4f}")
        print(f"  Worst state:         {self.occ_error['worst_state']}")
        print(f"  Per-state comparison (simulated vs observed):")
        for i, state in enumerate(STATES):
            sim = self.simulated_occupancy[i]
            obs = self.observed.state_occupancy[i]
            print(f"    {state:12s}: sim={sim:.4f}  obs={obs:.4f}  "
                  f"Δ={abs(sim-obs):.4f}")

        print(f"\n── RMSE ──────────────────────────────────────────────")
        print(f"  Matrix RMSE:    {self.rmse_result['rmse']:.4f}")
        print(f"  Per-row RMSE:")
        for i, state in enumerate(STATES):
            print(f"    {state:12s}: {self.rmse_result['row_rmse'][i]:.4f}")
        print(f"  Worst row:      {self.rmse_result['worst_row']}")

        print(f"\n── Jensen-Shannon Divergence ─────────────────────────")
        print(f"  Occupancy JSD:      {self.jsd_occ['jsd']:.6f}")
        print(f"  Occupancy JS Dist:  {self.jsd_occ['jsd_distance']:.6f}")
        print(f"  Interpretation:     {self.jsd_occ['interpretation']}")
        print(f"  Per-row JSD (transition matrices):")
        for i, state in enumerate(STATES):
            print(f"    {state:12s}: {self.jsd_rows['row_jsd'][i]:.6f}")
        print(f"  Mean row JSD:       {self.jsd_rows['mean_jsd']:.6f}")
        print(f"  Worst row:          {self.jsd_rows['worst_row']}")

        print(f"\n── KL Divergence ─────────────────────────────────────")
        print(f"  KL(obs || sim):  {self.kl_occ['kl_obs_sim']:.6f}")
        print(f"  KL(sim || obs):  {self.kl_occ['kl_sim_obs']:.6f}")
        print(f"  Interpretation:  {self.kl_occ['interpretation']}")
        print(f"  Per-row KL(obs||sim):")
        for i, state in enumerate(STATES):
            print(f"    {state:12s}: {self.kl_rows['row_kl'][i]:.6f}")

        score = self.overall_score()
        print(f"\n── Composite Validation Score ────────────────────────")
        print(f"  Score:   {score:.4f} / 1.0000")
        print(f"  Quality: {self.quality_label()}")
        bar_len = int(score * 40)
        print(f"  [{'█'*bar_len}{'░'*(40-bar_len)}]")

        print(f"\n── Tumor-Immune Parallel ─────────────────────────────")
        if score >= 0.70:
            print(f"  This validation score indicates the model captures "
                  f"the essential dynamics of this matchup. In a tumor-immune "
                  f"context, a model at this fidelity level would be considered "
                  f"adequate for generating testable hypotheses about treatment "
                  f"response dynamics.")
        elif score >= 0.50:
            print(f"  Moderate fit — the model captures broad trends but "
                  f"misses specific transition patterns. In oncology modeling, "
                  f"this would warrant additional biological constraints or "
                  f"more training data before clinical inference.")
        else:
            print(f"  Poor fit — the model does not adequately reproduce "
                  f"observed dynamics. More observed sequences or revised "
                  f"transition assumptions are needed.")
        print(f"{'='*65}")

    def to_dict(self):
        """Serialize all metrics to a flat dict for comparison across multiple opponents."""
        return {
            'fighter':        self.fighter_name,
            'opponent':       self.opponent_name,
            'n_simulations':  self.n_simulations,
            'n_observed':     self.observed.n_transitions,
            'mae':            self.tf_error['mean_error'],
            'max_ae':         self.tf_error['max_error'],
            'rmse':           self.rmse_result['rmse'],
            'occ_mae':        self.occ_error['mean_error'],
            'jsd_occupancy':  self.jsd_occ['jsd'],
            'jsd_rows_mean':  self.jsd_rows['mean_jsd'],
            'kl_obs_sim':     self.kl_occ['kl_obs_sim'],
            'kl_sim_obs':     self.kl_occ['kl_sim_obs'],
            'overall_score':  self.overall_score(),
            'quality':        self.quality_label()
        }


# ---------------------------------------------------------------------------
# SECTION 4 — Multi-opponent comparison
# ---------------------------------------------------------------------------

class MultiOpponentValidation:
    """
    Compare validation results across multiple opponent matchups.

    Answers: does the model generalize across opponent styles, or does
    it only fit one specific matchup? (Tumor-immune parallel: does the
    model generalize across treatment modalities, or only fit one drug?)
    """

    def __init__(self, reports):
        self.reports         = reports
        self.opponent_names  = [r.opponent_name for r in reports]
        self.summary_df      = self._build_summary()

    def _build_summary(self):
        """Build summary dict of all metrics across opponents."""
        return [r.to_dict() for r in self.reports]

    def print_comparison_table(self):
        """Print side-by-side metric comparison across all opponents."""
        metrics = ['mae', 'rmse', 'occ_mae',
                   'jsd_occupancy', 'kl_obs_sim', 'overall_score']
        labels  = ['Trans MAE', 'Trans RMSE', 'Occ MAE',
                   'JSD (occ)', 'KL(obs||sim)', 'Overall Score']

        print(f"\n{'='*70}")
        print(f"MULTI-OPPONENT VALIDATION COMPARISON — {self.reports[0].fighter_name}")
        print(f"{'='*70}")

        col_w = max(len(n) for n in self.opponent_names) + 2
        header = f"{'Metric':<18}" + "".join(
            f"{n:>{col_w}}" for n in self.opponent_names
        )
        print(header)
        print("─" * len(header))

        for metric, label in zip(metrics, labels):
            row = f"{label:<18}"
            for r in self.reports:
                d   = r.to_dict()
                val = d[metric]
                row += f"{val:>{col_w}.4f}"
            print(row)

        print(f"\n  Best overall fit:  "
              f"{max(self.reports, key=lambda r: r.overall_score()).opponent_name}")
        print(f"  Worst overall fit: "
              f"{min(self.reports, key=lambda r: r.overall_score()).opponent_name}")

        scores = [r.overall_score() for r in self.reports]
        print(f"  Score range:       {min(scores):.4f} – {max(scores):.4f}")
        print(f"  Score std dev:     {np.std(scores):.4f}")
        if np.std(scores) < 0.05:
            print(f"  Generalization:    Good — model fits consistently "
                  f"across all opponents")
        else:
            print(f"  Generalization:    Variable — model fits some opponents "
                  f"better than others")
        print(f"{'='*70}")


# ---------------------------------------------------------------------------
# SECTION 5 — Publication-quality plots
# ---------------------------------------------------------------------------

def plot_validation_report(report, filepath=None):
    """
    Publication-quality validation figure for a single matchup: 3x3 grid
    of transition-matrix heatmaps, occupancy/error bar charts, and a
    composite-score gauge.
    """
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    if filepath is None:
        safe_name = (f"{report.fighter_name}_vs_"
                     f"{report.opponent_name}").replace(' ', '_')
        filepath = os.path.join(
            OUTPUT_DIR, f"validation_{safe_name}.png"
        )

    score   = report.overall_score()
    quality = report.quality_label()

    fig = plt.figure(
        figsize=(16, 14),
        constrained_layout=True
    )
    fig.patch.set_facecolor(BG_COLOR)
    fig.suptitle(
        f"Validation: {report.fighter_name} vs {report.opponent_name} "
        f"— {quality} (score={score:.2f})",
        fontsize=14, fontweight='bold', y=1.01
    )

    gs = gridspec.GridSpec(3, 3, figure=fig,
                            hspace=0.4, wspace=0.35)

    # ── Row 1: Transition matrices ──────────────────────────
    # Simulated and observed share one color scale so equal-valued
    # cells render identically and the two panels are honestly
    # comparable; the error panel gets its own scale (different quantity).
    shared_vmax = max(report.simulated_matrix.max(),
                       report.observed.transition_matrix.max())
    matrices = [
        (report.simulated_matrix, 'Simulated Transition Matrix', 'Blues', shared_vmax),
        (report.observed.transition_matrix, 'Observed Transition Matrix', 'Blues', shared_vmax),
        (report.tf_error['error_matrix'], 'Absolute Error (|Sim − Obs|)', 'Reds',
         report.tf_error['error_matrix'].max())
    ]

    for col, (mat, title, cmap, vmax) in enumerate(matrices):
        ax  = fig.add_subplot(gs[0, col])
        im  = ax.imshow(mat, cmap=cmap, vmin=0,
                         vmax=vmax, aspect='auto')
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        for i in range(N_STATES):
            for j in range(N_STATES):
                ax.text(j, i, f"{mat[i,j]:.2f}",
                         ha='center', va='center',
                         fontsize=8,
                         color='white' if mat[i, j] > vmax * 0.6
                               else 'black')
        ax.set_xticks(range(N_STATES))
        ax.set_yticks(range(N_STATES))
        ax.set_xticklabels(STATES, rotation=45, ha='right', fontsize=9)
        ax.set_yticklabels(STATES, fontsize=9)
        ax.set_xlabel('To state', fontsize=9)
        ax.set_ylabel('From state', fontsize=9)
        ax.set_title(title, fontsize=10, fontweight='bold')

    # ── Row 2: Occupancy and per-row MAE/JSD ────────────────
    ax = fig.add_subplot(gs[1, 0])
    x  = np.arange(N_STATES)
    w  = 0.35

    color = (F1_COLOR if 'CJ' in report.fighter_name
              else F2_COLOR)

    ax.bar(x - w/2, report.simulated_occupancy, w,
            color=color, alpha=0.8, label='Simulated')
    ax.bar(x + w/2, report.observed.state_occupancy, w,
            color='gray', alpha=0.7, label='Observed')
    ax.set_xticks(x)
    ax.set_xticklabels(STATES, rotation=45, ha='right', fontsize=9)
    ax.set_ylabel('Fraction of time', fontsize=9)
    ax.set_title('State Occupancy: Simulated vs Observed',
                  fontsize=10, fontweight='bold')
    ax.legend(fontsize=8)

    ax = fig.add_subplot(gs[1, 1])
    ax.bar(STATES, report.tf_error['row_errors'],
            color='steelblue', alpha=0.8, edgecolor='white')
    ax.axhline(report.tf_error['mean_error'], color='red',
                linestyle='--', linewidth=1,
                label=f"Mean={report.tf_error['mean_error']:.4f}")
    ax.set_title('Per-State Transition MAE',
                  fontsize=10, fontweight='bold')
    ax.set_ylabel('Mean Absolute Error', fontsize=9)
    ax.tick_params(axis='x', rotation=45)
    ax.legend(fontsize=8)

    ax = fig.add_subplot(gs[1, 2])
    ax.bar(STATES, report.jsd_rows['row_jsd'],
            color='coral', alpha=0.8, edgecolor='white')
    ax.axhline(report.jsd_rows['mean_jsd'], color='red',
                linestyle='--', linewidth=1,
                label=f"Mean={report.jsd_rows['mean_jsd']:.4f}")
    ax.set_title('Per-State Row JSD',
                  fontsize=10, fontweight='bold')
    ax.set_ylabel('Jensen-Shannon Divergence', fontsize=9)
    ax.tick_params(axis='x', rotation=45)
    ax.legend(fontsize=8)

    # ── Row 3: RMSE, KL, composite score gauge ───────────────
    ax = fig.add_subplot(gs[2, 0])
    ax.bar(STATES, report.rmse_result['row_rmse'],
            color='mediumseagreen', alpha=0.8, edgecolor='white')
    ax.axhline(report.rmse_result['rmse'], color='red',
                linestyle='--', linewidth=1,
                label=f"Overall={report.rmse_result['rmse']:.4f}")
    ax.set_title('Per-State Row RMSE',
                  fontsize=10, fontweight='bold')
    ax.set_ylabel('RMSE', fontsize=9)
    ax.tick_params(axis='x', rotation=45)
    ax.legend(fontsize=8)

    ax = fig.add_subplot(gs[2, 1])
    ax.bar(STATES, report.kl_rows['row_kl'],
            color='mediumpurple', alpha=0.8, edgecolor='white')
    ax.axhline(report.kl_rows['mean_kl'], color='red',
                linestyle='--', linewidth=1,
                label=f"Mean={report.kl_rows['mean_kl']:.4f}")
    ax.set_title('Per-State KL Divergence KL(obs||sim)',
                  fontsize=10, fontweight='bold')
    ax.set_ylabel('KL Divergence', fontsize=9)
    ax.tick_params(axis='x', rotation=45)
    ax.legend(fontsize=8)

    # Composite score gauge (horizontal bar)
    ax = fig.add_subplot(gs[2, 2])
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_aspect('auto')

    bands = [
        (0.00, 0.30, '#ffcccc', 'Very Poor'),
        (0.30, 0.50, '#ffd9b3', 'Poor'),
        (0.50, 0.70, '#fff3b3', 'Moderate'),
        (0.70, 0.85, '#ccffcc', 'Good'),
        (0.85, 1.00, '#b3ffb3', 'Excellent'),
    ]
    for x0, x1, fc, label in bands:
        ax.barh(0.5, x1-x0, left=x0, height=0.3,
                 color=fc, edgecolor='gray', linewidth=0.5)
        ax.text((x0+x1)/2, 0.35, label,
                 ha='center', va='top', fontsize=7, color='gray')

    ax.barh(0.5, score, height=0.12,
             color='navy', alpha=0.9)
    ax.axvline(score, color='navy', linewidth=2)
    ax.text(score, 0.72,
             f"{score:.3f}",
             ha='center', va='bottom',
             fontsize=12, fontweight='bold', color='navy')

    ax.set_xlabel('Composite Score', fontsize=9)
    ax.set_yticks([])
    ax.set_xlim(0, 1)
    ax.set_title(f'Overall Fit Quality: {quality}',
                  fontsize=10, fontweight='bold')

    plt.savefig(filepath, dpi=FIGURE_DPI, bbox_inches='tight')
    plt.close()
    print(f"Saved: {filepath}")
    return filepath


def plot_multi_opponent_comparison(multi_validation, filepath=None):
    """
    Publication-quality figure comparing validation metrics across
    multiple opponent matchups: 2x3 grid of per-metric bar charts plus
    a radar/spider chart of the error profile per opponent.
    """
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    if filepath is None:
        filepath = os.path.join(
            OUTPUT_DIR, 'validation_multi_opponent.png'
        )

    reports  = multi_validation.reports
    names    = [r.opponent_name for r in reports]
    n_opp    = len(reports)

    palette = plt.cm.tab10(np.linspace(0, 0.5, n_opp))

    fig, axes = plt.subplots(2, 3, figsize=(16, 10),
                               constrained_layout=True)
    _darken_figure(fig, axes)
    fig.suptitle(
        f"Multi-Opponent Validation — "
        f"{reports[0].fighter_name}",
        fontsize=13, fontweight='bold'
    )

    x = np.arange(n_opp)

    metric_configs = [
        ('overall_score', 'Overall Fit Score', axes[0,0], True),
        ('jsd_occupancy', 'Occupancy JSD', axes[0,1], False),
        ('kl_obs_sim', 'KL Divergence KL(obs||sim)', axes[0,2], False),
        ('mae', 'Transition MAE', axes[1,0], False),
        ('rmse', 'Transition RMSE', axes[1,1], False),
    ]

    for metric, title, ax, higher_is_better in metric_configs:
        vals = [r.to_dict()[metric] for r in reports]
        bars = ax.bar(x, vals, color=palette,
                       edgecolor='white', linewidth=0.5)

        for bar, val in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width()/2,
                     bar.get_height() + max(vals)*0.01,
                     f"{val:.4f}",
                     ha='center', va='bottom', fontsize=8)

        ax.set_xticks(x)
        ax.set_xticklabels(names, rotation=30,
                             ha='right', fontsize=9)
        ax.set_title(title, fontsize=10, fontweight='bold')

        if higher_is_better:
            ax.set_ylabel('Score (higher = better)', fontsize=9)
        else:
            ax.set_ylabel('Error (lower = better)', fontsize=9)

    # Radar chart for the last cell — remove the blank Cartesian axes
    # plt.subplots already placed there before adding a polar one in
    # its spot, so the two don't end up overlapping.
    axes[1, 2].remove()
    ax_radar = fig.add_subplot(2, 3, 6, projection='polar')

    radar_metrics = ['mae', 'rmse', 'occ_mae',
                      'jsd_occupancy', 'kl_obs_sim']
    radar_labels  = ['Trans MAE', 'RMSE', 'Occ MAE',
                      'JSD', 'KL']

    angles = np.linspace(0, 2*np.pi, len(radar_metrics),
                          endpoint=False).tolist()
    angles += angles[:1]

    for i, r in enumerate(reports):
        d      = r.to_dict()
        values = [d[m] for m in radar_metrics]
        values += values[:1]
        ax_radar.plot(angles, values, 'o-', linewidth=1.5,
                       color=palette[i], label=r.opponent_name)
        ax_radar.fill(angles, values, alpha=0.1, color=palette[i])

    ax_radar.set_xticks(angles[:-1])
    ax_radar.set_xticklabels(radar_labels, fontsize=8)
    ax_radar.set_title('Error Profile per Opponent\n(lower = better)',
                         fontsize=9, fontweight='bold', pad=15)
    ax_radar.legend(loc='upper right',
                     bbox_to_anchor=(1.3, 1.1), fontsize=8)

    plt.savefig(filepath, dpi=FIGURE_DPI, bbox_inches='tight')
    plt.close()
    print(f"Saved: {filepath}")
    return filepath


# ---------------------------------------------------------------------------
# SECTION 6 — Placeholder observed data generator
# ---------------------------------------------------------------------------

def generate_placeholder_observed_data(fighter_name,
                                        opponent_name,
                                        n_sequences=5,
                                        seq_length=30,
                                        style='aggressive',
                                        seed=None):
    """
    Generate realistic placeholder observed sequences for testing.

    style options:
    'aggressive':    high Attack/Feint, low Defend
    'counter':       high Defend/Attack, moderate Feint
    'balanced':      roughly equal across all states

    Returns ObservedSparringData object.
    """
    if seed is not None:
        np.random.seed(seed)

    style_probs = {
        'aggressive': [0.35, 0.10, 0.20, 0.35],
        'counter':    [0.30, 0.35, 0.20, 0.15],
        'balanced':   [0.25, 0.25, 0.25, 0.25]
    }

    style_matrices = {
        'aggressive': np.array([
            [0.25, 0.15, 0.30, 0.30],
            [0.50, 0.10, 0.20, 0.20],
            [0.30, 0.10, 0.20, 0.40],
            [0.60, 0.10, 0.15, 0.15]
        ]),
        'counter': np.array([
            [0.15, 0.30, 0.30, 0.25],
            [0.55, 0.15, 0.20, 0.10],
            [0.15, 0.35, 0.20, 0.30],
            [0.35, 0.30, 0.20, 0.15]
        ]),
        'balanced': np.array([
            [0.25, 0.25, 0.25, 0.25],
            [0.25, 0.25, 0.25, 0.25],
            [0.25, 0.25, 0.25, 0.25],
            [0.25, 0.25, 0.25, 0.25]
        ])
    }

    T   = style_matrices[style]
    sequences = []

    for _ in range(n_sequences):
        start = np.random.choice(N_STATES, p=style_probs[style])
        seq   = [STATES[start]]
        current = start

        for _ in range(seq_length - 1):
            current = np.random.choice(N_STATES, p=T[current])
            seq.append(STATES[current])

        sequences.append(seq)

    return ObservedSparringData(
        fighter_name  = fighter_name,
        sequences     = sequences,
        opponent_name = opponent_name
    )


# ---------------------------------------------------------------------------
# SECTION 7 — Integration with pipeline
# ---------------------------------------------------------------------------

def _estimate_matrix_from_history(state_history_array):
    """
    Estimate mean transition matrix from MC state history array.
    state_history_array: shape (N_simulations, N_steps)
    """
    counts = np.zeros((N_STATES, N_STATES))
    N, T = state_history_array.shape

    for i in range(N):
        for t in range(T - 1):
            s  = state_history_array[i, t]
            s2 = state_history_array[i, t + 1]
            counts[s, s2] += 1

    row_sums = counts.sum(axis=1, keepdims=True)
    row_sums = np.where(row_sums < 1e-10, 1.0, row_sums)
    return counts / row_sums


def validate_pipeline(mc_results, mc_analysis,
                       observed_f1=None, observed_f2=None,
                       opponent_name="Counter-Fighter"):
    """
    Run validation comparing Monte Carlo simulation results against
    observed data. If observed_f1/observed_f2 are None, generates
    placeholder data for testing. Returns (f1_report, f2_report).
    """
    if observed_f1 is None:
        observed_f1 = generate_placeholder_observed_data(
            fighter_name  = "CJ",
            opponent_name = opponent_name,
            n_sequences   = 8,
            style         = 'aggressive',
            seed          = 42
        )

    if observed_f2 is None:
        observed_f2 = generate_placeholder_observed_data(
            fighter_name  = "Counter-Fighter",
            opponent_name = "CJ",
            n_sequences   = 8,
            style         = 'counter',
            seed          = 43
        )

    # Mean empirical transition matrix from MC runs, from state histories.
    f1_sim_T = _estimate_matrix_from_history(mc_results['f1_states'])
    f2_sim_T = _estimate_matrix_from_history(mc_results['f2_states'])

    f1_report = ValidationReport(
        fighter_name         = "CJ",
        opponent_name        = opponent_name,
        simulated_matrix     = f1_sim_T,
        simulated_occupancy  = mc_analysis['f1_occupancy']['mean'],
        observed_data        = observed_f1,
        n_simulations        = mc_results['n_simulations']
    )

    f2_report = ValidationReport(
        fighter_name         = "Counter-Fighter",
        opponent_name        = "CJ",
        simulated_matrix     = f2_sim_T,
        simulated_occupancy  = mc_analysis['f2_occupancy']['mean'],
        observed_data        = observed_f2,
        n_simulations        = mc_results['n_simulations']
    )

    f1_report.print_report()
    f2_report.print_report()

    plot_validation_report(f1_report)
    plot_validation_report(f2_report)

    return f1_report, f2_report
