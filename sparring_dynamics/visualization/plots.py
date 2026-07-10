"""
All visualization functions. Each function takes analysis results and
fighter objects, saves a PNG, and returns the filepath. No plt.show()
calls — only plt.savefig().
"""
import os
import numpy as np
import matplotlib.pyplot as plt

from sparring_dynamics.config import (
    STATES, OUTPUT_DIR, FIGURE_DPI, F1_COLOR, F2_COLOR
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
    """Explicitly set fig/axes facecolors (belt-and-suspenders on top
    of the rcParams above, in case a caller overrides them later)."""
    fig.patch.set_facecolor(BG_COLOR)
    for ax in np.atleast_1d(axes).flat:
        ax.set_facecolor(BG_COLOR)


def ensure_output_dir():
    os.makedirs(OUTPUT_DIR, exist_ok=True)


def plot_monte_carlo_summary(results, analysis, filepath=None):
    """
    3x2 grid: cumulative fitness bands, per-step payoff bands,
    state occupancy with error bars, lambda curves.
    Saves to outputs/monte_carlo_summary.png by default.
    """
    ensure_output_dir()
    if filepath is None:
        filepath = os.path.join(OUTPUT_DIR, 'monte_carlo_summary.png')

    fig, axes = plt.subplots(3, 2, figsize=(14, 16))
    _darken_figure(fig, axes)
    steps = np.arange(results['n_steps'])

    # Top row — cumulative fitness with CI bands
    for col, (key, label, color) in enumerate([
        ('f1_cumulative', 'Fighter 1 (CJ)', F1_COLOR),
        ('f2_cumulative', 'Fighter 2 (Counter-Fighter)', F2_COLOR)
    ]):
        ax = axes[0, col]
        s  = analysis[key]
        ax.fill_between(steps, s['ci_lower'], s['ci_upper'],
                         alpha=0.15, color=color, label='95% CI')
        ax.fill_between(steps, s['q25'], s['q75'],
                         alpha=0.15, color=color, label='IQR')
        for i in range(min(5, results['n_simulations'])):
            ax.plot(steps, results[key][i], color=color, alpha=0.05, linewidth=0.5)
        ax.plot(steps, s['mean'], color=color,
                linewidth=2, label='Mean')
        ax.set_title(f"{label}\nCumulative Fitness (N={results['n_simulations']}, 95% CI)")
        ax.set_xlabel('Exchange step')
        ax.set_ylabel('Cumulative fitness')
        ax.legend(fontsize=8)

    # Middle row — rolling per-step payoff
    window = 20
    for col, (key, label, color) in enumerate([
        ('f1_fitness', 'Fighter 1 (CJ)', F1_COLOR),
        ('f2_fitness', 'Fighter 2 (Counter-Fighter)', F2_COLOR)
    ]):
        ax = axes[1, col]
        s  = analysis[key]
        ax.fill_between(steps, s['ci_lower'], s['ci_upper'],
                         alpha=0.15, color=color)
        ax.plot(steps, s['mean'], color=color, linewidth=2)
        ax.axhline(0.5, color=TICK_COLOR, linestyle='--',
                    linewidth=0.8, alpha=0.6)
        ax.set_title(f"{label}\nPer-Step Payoff (Rolling {window}-step avg)")
        ax.set_xlabel('Exchange step')
        ax.set_ylabel('Payoff')

    # Bottom left — state occupancy with error bars
    ax = axes[2, 0]
    x  = np.arange(len(STATES))
    w  = 0.35
    f1_occ = analysis['f1_occupancy']
    f2_occ = analysis['f2_occupancy']

    ax.bar(x - w/2, f1_occ['mean'], w, color=F1_COLOR,
            label='Fighter 1 (CJ)',
            yerr=[f1_occ['mean'] - f1_occ['ci_lower'],
                  f1_occ['ci_upper'] - f1_occ['mean']],
            capsize=4)
    ax.bar(x + w/2, f2_occ['mean'], w, color=F2_COLOR,
            label='Fighter 2',
            yerr=[f2_occ['mean'] - f2_occ['ci_lower'],
                  f2_occ['ci_upper'] - f2_occ['mean']],
            capsize=4)
    ax.set_xticks(x)
    ax.set_xticklabels(STATES)
    ax.set_title('Average State Occupancy ± 95% CI')
    ax.set_ylabel('Fraction of time')
    ax.legend()

    # Bottom right — lambda curves
    ax = axes[2, 1]
    for key, label, color in [
        ('f1_lambda', 'Fighter 1 (CJ)', F1_COLOR),
        ('f2_lambda', 'Fighter 2', F2_COLOR)
    ]:
        s = analysis[key]
        ax.fill_between(steps, s['ci_lower'], s['ci_upper'],
                         alpha=0.15, color=color)
        ax.plot(steps, s['mean'], color=color,
                linewidth=2, label=label)
    for y in [0.25, 0.5, 0.75]:
        ax.axhline(y, color=TICK_COLOR, linestyle='--',
                    linewidth=0.6, alpha=0.5)
    ax.set_title('Adaptation Weight λ (Mean ± 95% CI)')
    ax.set_xlabel('Exchange step')
    ax.set_ylabel('λ')
    ax.set_ylim(0, 1)
    ax.legend()

    plt.tight_layout()
    plt.savefig(filepath, dpi=FIGURE_DPI, bbox_inches='tight')
    plt.close()
    print(f"Saved: {filepath}")
    return filepath


def plot_distributions(results, analysis, filepath=None):
    """
    2x2 distribution plots: F1 histogram, F2 histogram,
    F1 vs F2 scatter, final lambda distribution.
    Saves to outputs/monte_carlo_distributions.png
    """
    ensure_output_dir()
    if filepath is None:
        filepath = os.path.join(OUTPUT_DIR, 'monte_carlo_distributions.png')

    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    _darken_figure(fig, axes)

    f1_finals = results['f1_cumulative'][:, -1]
    f2_finals = results['f2_cumulative'][:, -1]
    f1_lams   = results['f1_lambda'][:, -1]
    f2_lams   = results['f2_lambda'][:, -1]

    # F1 histogram
    ax = axes[0, 0]
    ax.hist(f1_finals, bins=30, color=F1_COLOR, alpha=0.7, edgecolor='white')
    ax.axvline(f1_finals.mean(), color=TITLE_COLOR, linestyle='--', linewidth=1.5)
    ci = analysis['f1_cumulative']
    ax.axvspan(ci['ci_lower'][-1], ci['ci_upper'][-1],
                alpha=0.15, color=F1_COLOR)
    ax.set_title('Distribution of F1 Final Fitness')
    ax.set_xlabel('Final cumulative fitness')

    # F2 histogram
    ax = axes[0, 1]
    ax.hist(f2_finals, bins=30, color=F2_COLOR, alpha=0.7, edgecolor='white')
    ax.axvline(f2_finals.mean(), color=TITLE_COLOR, linestyle='--', linewidth=1.5)
    ci = analysis['f2_cumulative']
    ax.axvspan(ci['ci_lower'][-1], ci['ci_upper'][-1],
                alpha=0.15, color=F2_COLOR)
    ax.set_title('Distribution of F2 Final Fitness')
    ax.set_xlabel('Final cumulative fitness')

    # Scatter F1 vs F2
    ax = axes[1, 0]
    f1_wins = f1_finals > f2_finals
    f2_wins = f2_finals > f1_finals
    ties    = ~f1_wins & ~f2_wins

    ax.scatter(f1_finals[f1_wins], f2_finals[f1_wins],
                c=F1_COLOR, alpha=0.4, s=8, label='F1 wins')
    ax.scatter(f1_finals[f2_wins], f2_finals[f2_wins],
                c=F2_COLOR, alpha=0.4, s=8, label='F2 wins')
    ax.scatter(f1_finals[ties], f2_finals[ties],
                c='gray', alpha=0.4, s=8, label='Tie')

    lim = [min(f1_finals.min(), f2_finals.min()) * 0.98,
            max(f1_finals.max(), f2_finals.max()) * 1.02]
    ax.plot(lim, lim, '--', color=TICK_COLOR, linewidth=0.8, alpha=0.5)
    ax.set_xlim(lim); ax.set_ylim(lim)
    ax.set_xlabel('F1 final fitness')
    ax.set_ylabel('F2 final fitness')
    ax.set_title(f'F1 vs F2 Final Fitness\n(N={results["n_simulations"]})')
    ax.legend(fontsize=8)

    # Lambda distribution
    ax = axes[1, 1]
    ax.hist(f1_lams, bins=25, color=F1_COLOR, alpha=0.6,
             label='Fighter 1 (CJ)', edgecolor='white')
    ax.hist(f2_lams, bins=25, color=F2_COLOR, alpha=0.6,
             label='Fighter 2', edgecolor='white')
    ax.set_title('Distribution of Final Adaptation Weight λ')
    ax.set_xlabel('λ')
    ax.legend()

    plt.tight_layout()
    plt.savefig(filepath, dpi=FIGURE_DPI, bbox_inches='tight')
    plt.close()
    print(f"Saved: {filepath}")
    return filepath
