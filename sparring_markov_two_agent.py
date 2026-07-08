import os
import numpy as np
import matplotlib.pyplot as plt

STATES = ['Attack', 'Defend', 'Disengage', 'Feint']
N = 4
OUT_DIR = os.path.dirname(os.path.abspath(__file__))

P1_BASE = np.array([
    [0.25, 0.15, 0.35, 0.25],
    [0.45, 0.10, 0.25, 0.20],
    [0.30, 0.10, 0.20, 0.40],
    [0.60, 0.08, 0.17, 0.15],
], dtype=float)

P2_BASE = np.array([
    [0.15, 0.30, 0.35, 0.20],
    [0.55, 0.15, 0.20, 0.10],
    [0.15, 0.30, 0.25, 0.30],
    [0.40, 0.25, 0.20, 0.15],
], dtype=float)

# Column adjustments applied to every row of F1 when F2 is in a given state
# [Attack, Defend, Disengage, Feint]
F1_ADJ = {
    0: np.array([-0.20, +0.15, +0.05,  0.00]),  # F2 in Attack
    1: np.array([-0.10,  0.00,  0.00, +0.10]),  # F2 in Defend
    2: np.array([+0.15,  0.00, -0.15,  0.00]),  # F2 in Disengage
    3: np.array([-0.10,  0.00, +0.10,  0.00]),  # F2 in Feint
}

# Column adjustments applied to every row of F2 when F1 is in a given state
F2_ADJ = {
    0: np.array([+0.10, +0.10, -0.20,  0.00]),  # F1 in Attack
    1: np.array([+0.20,  0.00, -0.20,  0.00]),  # F1 in Defend
    2: np.array([ 0.00, -0.10,  0.00, +0.10]),  # F1 in Disengage
    3: np.array([-0.15, +0.15,  0.00,  0.00]),  # F1 in Feint
}


def apply_adjustment(base, adj):
    m = base + adj  # adj broadcast over all rows
    m = np.clip(m, 0, None)
    m /= m.sum(axis=1, keepdims=True)
    return m


def adjusted_P1(f2_state):
    return apply_adjustment(P1_BASE, F1_ADJ[f2_state])


def adjusted_P2(f1_state):
    return apply_adjustment(P2_BASE, F2_ADJ[f1_state])


# --- Build 16x16 joint transition matrix ---
P_joint = np.zeros((N * N, N * N))
for s1 in range(N):
    for s2 in range(N):
        p1 = adjusted_P1(s2)
        p2 = adjusted_P2(s1)
        for s1n in range(N):
            for s2n in range(N):
                P_joint[s1 * N + s2, s1n * N + s2n] = p1[s1, s1n] * p2[s2, s2n]

# --- Theoretical stationary distribution via left eigenvector ---
eigenvalues, eigenvectors = np.linalg.eig(P_joint.T)
idx = np.argmin(np.abs(eigenvalues - 1.0))
pi_joint = np.real(eigenvectors[:, idx])
pi_joint = np.abs(pi_joint)
pi_joint /= pi_joint.sum()

pi_joint_4x4 = pi_joint.reshape(N, N)          # [F1 state, F2 state]
theoretical_F1 = pi_joint_4x4.sum(axis=1)
theoretical_F2 = pi_joint_4x4.sum(axis=0)

# --- Simulation: 500 steps ---
N_STEPS = 500
rng = np.random.default_rng(42)
s1, s2 = 2, 2  # both start in Disengage

s1_hist = np.empty(N_STEPS, dtype=int)
s2_hist = np.empty(N_STEPS, dtype=int)
s1_hist[0], s2_hist[0] = s1, s2

for t in range(1, N_STEPS):
    p1 = adjusted_P1(s2)
    p2 = adjusted_P2(s1)
    s1 = rng.choice(N, p=p1[s1])
    s2 = rng.choice(N, p=p2[s2])
    s1_hist[t], s2_hist[t] = s1, s2

# --- Empirical distributions ---
empirical_F1 = np.array([(s1_hist == i).mean() for i in range(N)])
empirical_F2 = np.array([(s2_hist == i).mean() for i in range(N)])

empirical_joint = np.zeros((N, N))
for i in range(N):
    for j in range(N):
        empirical_joint[i, j] = ((s1_hist == i) & (s2_hist == j)).mean()

# --- Top 5 joint states ---
pairs = sorted(
    [(empirical_joint[i, j], STATES[i], STATES[j]) for i in range(N) for j in range(N)],
    reverse=True,
)
print("Top 5 most common joint states (empirical):")
for prob, f1s, f2s in pairs[:5]:
    print(f"  F1={f1s:10s}  F2={f2s:10s}  {prob*100:.1f}%")

print("\nTheoretical marginal — Fighter 1 (CJ):")
for s, p in zip(STATES, theoretical_F1):
    print(f"  {s:10s}: {p:.4f}")

print("\nEmpirical marginal — Fighter 1 (CJ):")
for s, p in zip(STATES, empirical_F1):
    print(f"  {s:10s}: {p:.4f}")

print("\nTheoretical marginal — Fighter 2 (Counter-Puncher):")
for s, p in zip(STATES, theoretical_F2):
    print(f"  {s:10s}: {p:.4f}")

print("\nEmpirical marginal — Fighter 2 (Counter-Puncher):")
for s, p in zip(STATES, empirical_F2):
    print(f"  {s:10s}: {p:.4f}")


# ---------------------------------------------------------------
# Figure 1: 2×2 summary grid
# ---------------------------------------------------------------
fig, axes = plt.subplots(2, 2, figsize=(13, 10))
fig.suptitle('Two-Agent Sparring Markov Chain — CJ vs Counter-Puncher', fontsize=14, fontweight='bold')

x = np.arange(N)
w = 0.35

# Top-left: F1 theoretical vs empirical bar chart
ax = axes[0, 0]
ax.bar(x - w / 2, theoretical_F1, w, label='Theoretical', color='steelblue', alpha=0.9)
ax.bar(x + w / 2, empirical_F1,   w, label='Empirical',   color='steelblue', alpha=0.5)
ax.set_xticks(x); ax.set_xticklabels(STATES)
ax.set_ylabel('Probability')
ax.set_title('Fighter 1 (CJ) — State Distribution')
ax.legend()
ax.set_ylim(0, 0.55)

# Top-right: F1 first 60 steps
ax = axes[0, 1]
ax.step(range(60), s1_hist[:60], color='green', where='post', linewidth=1.5)
ax.set_yticks(range(N)); ax.set_yticklabels(STATES)
ax.set_xlabel('Exchange')
ax.set_title('Fighter 1 (CJ) — First 60 Exchanges')
ax.grid(axis='y', alpha=0.3)

# Bottom-left: F2 theoretical vs empirical bar chart
ax = axes[1, 0]
ax.bar(x - w / 2, theoretical_F2, w, label='Theoretical', color='coral', alpha=0.9)
ax.bar(x + w / 2, empirical_F2,   w, label='Empirical',   color='coral', alpha=0.5)
ax.set_xticks(x); ax.set_xticklabels(STATES)
ax.set_ylabel('Probability')
ax.set_title('Fighter 2 (Counter-Puncher) — State Distribution')
ax.legend()
ax.set_ylim(0, 0.55)

# Bottom-right: F2 first 60 steps
ax = axes[1, 1]
ax.step(range(60), s2_hist[:60], color='red', where='post', linewidth=1.5)
ax.set_yticks(range(N)); ax.set_yticklabels(STATES)
ax.set_xlabel('Exchange')
ax.set_title('Fighter 2 (Counter-Puncher) — First 60 Exchanges')
ax.grid(axis='y', alpha=0.3)

plt.tight_layout()
out1 = os.path.join(OUT_DIR, 'sparring_two_agent.png')
plt.savefig(out1, dpi=150, bbox_inches='tight')
plt.close()
print(f"\nSaved: {out1}")


# ---------------------------------------------------------------
# Figure 2: Side-by-side joint heatmaps
# ---------------------------------------------------------------
fig, axes = plt.subplots(1, 2, figsize=(14, 6))
fig.suptitle('Joint State Distribution — Fighter 1 (CJ) vs Fighter 2 (Counter-Puncher)',
             fontsize=13, fontweight='bold')

vmax = max(pi_joint_4x4.max(), empirical_joint.max())

for ax, data, title in zip(
    axes,
    [pi_joint_4x4, empirical_joint],
    ['Theoretical Joint Distribution', 'Empirical Joint Distribution'],
):
    im = ax.imshow(data, cmap='Blues', aspect='auto', vmin=0, vmax=vmax)
    ax.set_xticks(range(N)); ax.set_xticklabels(STATES, fontsize=10)
    ax.set_yticks(range(N)); ax.set_yticklabels(STATES, fontsize=10)
    ax.set_xlabel('Fighter 2 State', fontsize=11)
    ax.set_ylabel('Fighter 1 State', fontsize=11)
    ax.set_title(title, fontsize=12, fontweight='bold')
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    for i in range(N):
        for j in range(N):
            color = 'white' if data[i, j] > vmax * 0.6 else 'black'
            ax.text(j, i, f'{data[i, j]:.3f}', ha='center', va='center',
                    fontsize=9, color=color)

plt.tight_layout()
out2 = os.path.join(OUT_DIR, 'sparring_joint_heatmap.png')
plt.savefig(out2, dpi=150, bbox_inches='tight')
plt.close()
print(f"Saved: {out2}")
