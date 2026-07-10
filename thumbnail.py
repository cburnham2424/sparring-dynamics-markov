"""
Generates a high-resolution (1280x640) dark-mode thumbnail for the
project, suitable for LinkedIn: two fighter clusters (F1 left, F2
right), the four tactical states as colored nodes, interaction arrows
between the clusters, and three center labels naming the core
mathematical machinery.

Arrow thickness is not decorative — it is derived from the real,
committed dataset (data/combined_annotations.csv): the three
"dominant" (>= DEFAULT_MIN_OBS) state-pair arrows are exactly the
three empirically-grounded payoff cells this project has been
tracking all along ((Attack,Defend), (Defend,Attack),
(Disengage,Disengage)); everything else renders as a thin, non-dominant
connector.

Run directly: `python thumbnail.py` (writes outputs/thumbnail.png).
"""
import os

import matplotlib.pyplot as plt
import matplotlib.patheffects as path_effects
import numpy as np
from matplotlib.patches import Circle, FancyArrowPatch

from sparring_dynamics.config import OUTPUT_DIR, DEFAULT_MIN_OBS
from sparring_dynamics.data.annotation_format import (
    load_annotation_csv, annotations_to_exchanges,
)
from sparring_dynamics.estimation.payoffs import estimate_payoff_matrices

# ── Palette ───────────────────────────────────────────────────
BG_COLOR = '#1a1a19'
STATE_COLORS = {
    'Attack': '#7F77DD',
    'Defend': '#1D9E75',
    'Disengage': '#888780',
    'Feint': '#EF9F27',
}
ARROW_COLOR = '#c3c2b7'
DOMINANT_ARROW_COLOR = '#ffffff'
TEXT_PRIMARY = '#ffffff'
TEXT_SECONDARY = '#898781'
LABEL_BOX_FACE = '#2c2c2a'
LABEL_BOX_EDGE = '#4a4a47'

FIG_WIDTH_PX, FIG_HEIGHT_PX = 1280, 640
DPI = 100

# Node layout: Attack faces the opponent (center), Defend faces
# outward, Disengage sits below, Feint sits above.
NODE_ANGLES_F1 = {'Attack': 0, 'Defend': 180, 'Disengage': 270, 'Feint': 90}
NODE_ANGLES_F2 = {'Attack': 180, 'Defend': 0, 'Disengage': 270, 'Feint': 90}


def _get_dominant_pairs():
    """
    Load the real annotated dataset and return a list of
    (f1_state, f2_state, observation_count, is_dominant) tuples for
    every state pair that was ever observed, so arrow thickness in the
    thumbnail reflects genuine data rather than arbitrary decoration.
    Falls back to a small illustrative set if the dataset is missing.
    """
    csv_path = os.path.join('data', 'combined_annotations.csv')
    if not os.path.exists(csv_path):
        return [
            ('Attack', 'Defend', 29, True),
            ('Disengage', 'Disengage', 15, True),
            ('Defend', 'Attack', 12, True),
            ('Feint', 'Defend', 4, False),
        ]

    annotations, _ = load_annotation_csv(csv_path, strict=False)
    exchanges = annotations_to_exchanges(annotations)
    payoff_result = estimate_payoff_matrices(exchanges)
    totals = payoff_result['totals']

    states = ['Attack', 'Defend', 'Disengage', 'Feint']
    pairs = []
    for i, s1 in enumerate(states):
        for j, s2 in enumerate(states):
            count = int(totals[i, j])
            if count > 0:
                pairs.append((s1, s2, count, count >= DEFAULT_MIN_OBS))
    return pairs


def _node_position(cluster_center, angle_deg, radius):
    angle = np.radians(angle_deg)
    return (cluster_center[0] + radius * np.cos(angle),
            cluster_center[1] + radius * np.sin(angle))


def _draw_cluster(ax, cluster_center, radius, angles, node_radius=0.032):
    """Draw the 4 tactical-state nodes for one fighter's cluster. Returns
    a dict of state -> (x, y) node position for arrow-drawing."""
    positions = {}
    for state, angle_deg in angles.items():
        x, y = _node_position(cluster_center, angle_deg, radius)
        positions[state] = (x, y)
        ax.add_patch(Circle((x, y), node_radius,
                             facecolor=STATE_COLORS[state],
                             edgecolor=BG_COLOR, linewidth=2, zorder=4))
        # Label placed just outside the node, away from cluster center
        label_x, label_y = _node_position(cluster_center, angle_deg,
                                           radius + node_radius + 0.045)
        ha = 'center'
        if label_x < cluster_center[0] - 0.01:
            ha = 'right'
        elif label_x > cluster_center[0] + 0.01:
            ha = 'left'
        txt = ax.text(label_x, label_y, state, color=TEXT_PRIMARY,
                       fontsize=9.5, ha=ha, va='center', zorder=10)
        # Dark stroke so the label stays legible over arrows of any
        # color, including the white dominant arrows (same color as
        # the label text, otherwise invisible where they cross).
        txt.set_path_effects([
            path_effects.withStroke(linewidth=3, foreground=BG_COLOR)
        ])
    return positions


def _draw_arrow(ax, start, end, color, linewidth, curve=0.0, zorder=2, alpha=1.0):
    arrow = FancyArrowPatch(
        start, end,
        connectionstyle=f"arc3,rad={curve}",
        arrowstyle='-|>', mutation_scale=12 + linewidth * 1.5,
        linewidth=linewidth, color=color, alpha=alpha,
        zorder=zorder, shrinkA=8, shrinkB=8,
    )
    ax.add_patch(arrow)


def generate_thumbnail(filepath=None):
    if filepath is None:
        os.makedirs(OUTPUT_DIR, exist_ok=True)
        filepath = os.path.join(OUTPUT_DIR, 'thumbnail.png')

    plt.style.use('dark_background')

    fig, ax = plt.subplots(
        figsize=(FIG_WIDTH_PX / DPI, FIG_HEIGHT_PX / DPI), dpi=DPI
    )
    fig.patch.set_facecolor(BG_COLOR)
    ax.set_facecolor(BG_COLOR)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis('off')

    f1_center = (0.18, 0.58)
    f2_center = (0.82, 0.58)
    cluster_radius = 0.15

    # Cluster labels
    ax.text(f1_center[0], f1_center[1] + cluster_radius + 0.16,
            "Fighter 1 — CJ", color=TEXT_PRIMARY, fontsize=13,
            ha='center', va='center', fontweight='bold')
    ax.text(f2_center[0], f2_center[1] + cluster_radius + 0.16,
            "Fighter 2 — Counter-Fighter", color=TEXT_PRIMARY, fontsize=13,
            ha='center', va='center', fontweight='bold')

    f1_nodes = _draw_cluster(ax, f1_center, cluster_radius, NODE_ANGLES_F1)
    f2_nodes = _draw_cluster(ax, f2_center, cluster_radius, NODE_ANGLES_F2)

    # ── Interaction arrows, weighted by the real dataset ─────────
    pairs = _get_dominant_pairs()
    max_count = max((p[2] for p in pairs), default=1)

    for f1_state, f2_state, count, is_dominant in pairs:
        if f1_state not in f1_nodes or f2_state not in f2_nodes:
            continue
        start = f1_nodes[f1_state]
        end = f2_nodes[f2_state]
        if is_dominant:
            width = 2.0 + 6.0 * (count / max_count)
            color = DOMINANT_ARROW_COLOR
            alpha = 0.9
            zorder = 3
        else:
            width = 0.8
            color = ARROW_COLOR
            alpha = 0.5
            zorder = 2
        curve = 0.08 if f1_state == f2_state else 0.0
        _draw_arrow(ax, start, end, color, width, curve=curve,
                    zorder=zorder, alpha=alpha)

    # ── Center labels: the core mathematical machinery ───────────
    # Placed in a horizontal row in the clear band below the node
    # clusters and above the title — arrows only ever span the space
    # between the two clusters (roughly y=0.35-0.80), so this band
    # never crosses an arrow path.
    center_labels = ['EGT replicator', 'memory decay λ', 'Nash equilibrium']
    label_xs = [0.28, 0.5, 0.72]
    label_y = 0.255
    for label, x in zip(center_labels, label_xs):
        ax.text(x, label_y, label, color=TEXT_PRIMARY, fontsize=11,
                ha='center', va='center', zorder=6,
                bbox=dict(boxstyle='round,pad=0.45',
                          facecolor=LABEL_BOX_FACE,
                          edgecolor=LABEL_BOX_EDGE,
                          linewidth=1.2))

    # ── Title / subtitle ──────────────────────────────────────────
    ax.text(0.5, 0.14, "Sparring Dynamics Markov Model",
            color=TEXT_PRIMARY, fontsize=24, fontweight='bold',
            ha='center', va='center')
    ax.text(0.5, 0.065, "two-agent · stochastic · adaptive",
            color=TEXT_SECONDARY, fontsize=13,
            ha='center', va='center')

    fig.savefig(filepath, dpi=DPI, facecolor=BG_COLOR)
    plt.close(fig)
    print(f"Saved: {filepath}")
    return filepath


if __name__ == "__main__":
    generate_thumbnail()
