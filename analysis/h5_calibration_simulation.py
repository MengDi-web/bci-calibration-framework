#!/usr/bin/env python3
"""
Framework 2.0 — H5: Calibration Strategy Simulation
=====================================================
H4 finding: Drift rate correlates with learning rate (rho=+0.633, p=0.067).
  High-drift (top 5): S1, S4, S7, S8, S9
  Low-drift (bottom 4): S2, S11, S12, S13

H5 question: Does drift-informed stratified calibration outperform uniform?
Method: Simulate cumulative accuracy across 11 sessions per subject.
  Decay: acc(i->j) = acc_i - delta * gap, where delta = subject's H4 drift rate.
  Calibration resets decoder to current session's accuracy.

Strategies:
  S1_uniform_3:    All every 3 sessions
  S2_uniform_5:    All every 5 sessions
  S3_stratified:   High-drift every 2, low-drift every 5 (ceiling-direction)
  S4_reverse:      High-drift every 5, low-drift every 2 (control: reverse direction)
  S5_no_calib:     Never recalibrate (floor)

Comparison: Paired bootstrap S3 - S1 with MID = 0.01 (1% accuracy).
  MID = 0.01: engineering judgment; <1% difference unlikely to be practically
  noticeable. No online behavioral metrics available to anchor a clinical MID.

Caveats:
  - n=9, exploratory. Drift rates estimated from H4 on the same 9 subjects;
    strategy evaluation is not fully independent of drift estimation.
  - Linear decay assumption: constant delta per session. Sensitivity to decay
    functional form not tested.
  - Offline simulation: absolute cumulative accuracy not predictive of online
    deployment; relative strategy ranking is the intended output.

Input:  data/accuracy_matrix_10subjects.csv, results/h4_results.json
Output: results/h5_results.json, figures/h5_calibration_simulation.png
"""

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os, json, warnings
warnings.filterwarnings('ignore')

# ============================================================
# Configuration
# ============================================================
DATA_DIR = os.path.expanduser("~/Desktop/bci_project/calibration_framework/data")
FIG_DIR = os.path.expanduser("~/Desktop/bci_project/calibration_framework/figures")
RESULT_DIR = os.path.expanduser("~/Desktop/bci_project/calibration_framework/results")
H4_RESULTS = os.path.join(RESULT_DIR, "h4_results.json")
ACC_PATH = os.path.join(DATA_DIR, "accuracy_matrix_10subjects.csv")

os.makedirs(FIG_DIR, exist_ok=True)
os.makedirs(RESULT_DIR, exist_ok=True)

SUBJECTS = ['S1', 'S2', 'S4', 'S7', 'S8', 'S9', 'S11', 'S12', 'S13']
N_SESSIONS = 11
MID = 0.01          # 1% cumulative accuracy: engineering judgment (see docstring)
N_BOOTSTRAP = 10000
RANDOM_SEED = 42

# H4 drift-based groups: top 5 vs bottom 4
HIGH_DRIFT = ['S1', 'S4', 'S7', 'S8', 'S9']
LOW_DRIFT  = ['S2', 'S11', 'S12', 'S13']

# ============================================================
# Load
# ============================================================
print("H5: Calibration Strategy Simulation")
print(f"  High-drift: {HIGH_DRIFT}")
print(f"  Low-drift:  {LOW_DRIFT}")
print(f"  MID = {MID} (engineering judgment, no online anchor)")
print("-" * 50)

df_acc = pd.read_csv(ACC_PATH, index_col=0)
with open(H4_RESULTS, 'r') as f:
    drift_rates = {s: v for s, v in json.load(f)['drift_rates'].items() if v is not None}

print("Drift rates (/session):")
for s in SUBJECTS:
    d = drift_rates.get(s, np.nan)
    print(f"  {s} [{'HIGH' if s in HIGH_DRIFT else 'LOW'}]: {d:.5f}")

# ============================================================
# Strategy definitions
# ============================================================
def get_calib_sessions(interval, n_sessions=N_SESSIONS):
    """0-indexed calibration session indices."""
    if interval >= 999:
        return [0]
    sessions = [0]
    nxt = interval
    while nxt < n_sessions:
        sessions.append(nxt)
        nxt += interval
    return sessions

STRATEGIES = {
    'S1_uniform_3':  {'label': 'Uniform-3',            'color': '#2c7bb6',
                       'high_interval': 3,  'low_interval': 3},
    'S2_uniform_5':  {'label': 'Uniform-5',            'color': '#fdae61',
                       'high_interval': 5,  'low_interval': 5},
    'S3_stratified': {'label': 'Stratified (drift)',   'color': '#1a9641',
                       'high_interval': 2,  'low_interval': 5},
    'S4_reverse':    {'label': 'Reverse',              'color': '#d7191c',
                       'high_interval': 5,  'low_interval': 2},
    'S5_no_calib':   {'label': 'No calibration',       'color': '#999999',
                       'high_interval': 999, 'low_interval': 999}
}

def simulate(acc_vec, calib_sessions, delta):
    """Cumulative mean accuracy under linear decay with recalibration."""
    n = len(acc_vec)
    effective = np.zeros(n)
    last_calib = 0
    for i in range(n):
        if i in calib_sessions:
            effective[i] = acc_vec[i]
            last_calib = i
        else:
            effective[i] = max(0.0, acc_vec[last_calib] - delta * (i - last_calib))
    return np.mean(effective)

# ============================================================
# Run simulation
# ============================================================
print(f"\nSimulating {len(STRATEGIES)} x {len(SUBJECTS)} subjects ...")
all_results = {}
for name, info in STRATEGIES.items():
    subj_means = []
    for subj in SUBJECTS:
        acc_vec = df_acc.loc[subj].values
        delta = drift_rates.get(subj, 0.003)  # fallback: typical SMR-BCI drift
        interval = info['high_interval'] if subj in HIGH_DRIFT else info['low_interval']
        subj_means.append(simulate(acc_vec, get_calib_sessions(interval), delta))
    m, s = np.mean(subj_means), np.std(subj_means)
    all_results[name] = {'mean': m, 'std': s, 'individual': np.array(subj_means)}
    print(f"  {info['label']:<22s}: {m:.4f} +/- {s:.4f}")

# ============================================================
# S3 vs S1: Paired bootstrap
# ============================================================
print(f"\n{'='*50}")
print(f"S3 vs S1: Paired Bootstrap")
print(f"{'='*50}")

s3 = all_results['S3_stratified']['individual']
s1 = all_results['S1_uniform_3']['individual']
diff = s3 - s1
mean_diff = np.mean(diff)

np.random.seed(RANDOM_SEED)
n_subj = len(diff)
boot = np.zeros(N_BOOTSTRAP)
for b in range(N_BOOTSTRAP):
    idx = np.random.choice(n_subj, n_subj, replace=True)
    boot[b] = np.mean(diff[idx])

ci_low, ci_high = np.percentile(boot, [2.5, 97.5])
p_pos = np.mean(boot > 0)
p_above_mid = np.mean(boot > MID)

print(f"  S3 - S1 = {mean_diff:.4f}")
print(f"  95% CI:  [{ci_low:.4f}, {ci_high:.4f}]")
print(f"  P(S3 > S1)     = {p_pos:.4f}")
print(f"  P(S3 > S1+{MID}) = {p_above_mid:.4f}")

# Decision framework
if ci_low > MID:
    decision = 'STRONG_STRATIFIED'
elif ci_low > 0 and p_above_mid > 0.5:
    decision = 'WEAK_STRATIFIED'
elif ci_high > 0:
    decision = 'MARGINAL_STRATIFIED'
else:
    decision = 'USE_UNIFORM'

print(f"  Decision: {decision}")
if decision == 'MARGINAL_STRATIFIED':
    print(f"  Interpretation: stratified ranks first but functionally equivalent to uniform")

# Also S3 vs S2
s2 = all_results['S2_uniform_5']['individual']
diff_32 = np.mean(s3 - s2)
print(f"\n  S3 vs S2 (Uniform-5): diff = {diff_32:.4f}")

# Rankings
print(f"\nStrategy Rankings:")
for rank, (name, res) in enumerate(
    sorted(all_results.items(), key=lambda x: x[1]['mean'], reverse=True), 1
):
    print(f"  #{rank}: {STRATEGIES[name]['label']:<22s} {res['mean']:.4f}")

# ============================================================
# Figure
# ============================================================
fig, axes = plt.subplots(1, 2, figsize=(16, 6))
fig.subplots_adjust(wspace=0.30)

# Panel A: Bar chart
ax = axes[0]
names = list(STRATEGIES.keys())
means = [all_results[n]['mean'] for n in names]
stds  = [all_results[n]['std'] for n in names]
colors = [STRATEGIES[n]['color'] for n in names]
labels = [STRATEGIES[n]['label'] for n in names]
bars = ax.bar(range(len(names)), means, color=colors, alpha=0.85, edgecolor='white')
ax.errorbar(range(len(names)), means, yerr=stds, fmt='none', color='black', capsize=5)
for bar, m in zip(bars, means):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.001,
            f'{m:.4f}', ha='center', va='bottom', fontsize=8)
ax.set_xticks(range(len(names)))
ax.set_xticklabels(labels, fontsize=8, rotation=12)
ax.set_ylabel('Mean Cumulative Accuracy', fontsize=10)
ax.set_title('H5: Strategy Performance', fontsize=11, fontweight='bold')
ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False)

# Panel B: Bootstrap distribution of S3-S1
ax = axes[1]
ax.hist(boot, bins=50, color='#2c7bb6' if mean_diff > 0 else '#d7191c',
        alpha=0.7, edgecolor='white', density=True)
ax.axvline(x=0, color='gray', linestyle='--', linewidth=1.2)
ax.axvline(x=mean_diff, color='#d7191c', linewidth=2, label=f'Mean = {mean_diff:.4f}')
ax.axvline(x=MID, color='#1a9641', linestyle=':', linewidth=1, label=f'MID = {MID}')
ax.axvspan(ci_low, ci_high, alpha=0.15, color='#2c7bb6', label='95% CI')
ax.set_xlabel('S3 - S1 (Cumulative Accuracy)', fontsize=10)
ax.set_ylabel('Bootstrap Density', fontsize=10)
ax.set_title(f'S3 vs S1: {decision}\nP(S3>S1)={p_pos:.3f}, P(>MID)={p_above_mid:.3f}',
             fontsize=11, fontweight='bold')
ax.legend(fontsize=8)
ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False)

fig.suptitle(f'H5: Calibration Strategy Simulation\n'
             f'{len(SUBJECTS)} subjects, H4 drift rates, {N_BOOTSTRAP} bootstrap iterations',
             fontsize=13, fontweight='bold', y=1.02)
fig_path = os.path.join(FIG_DIR, 'h5_calibration_simulation.png')
fig.savefig(fig_path, dpi=150, bbox_inches='tight', facecolor='white')
plt.close()
print(f"\nFigure: {fig_path}")

# Save
h5_out = {
    'framework': '2.0', 'stage': 'H5',
    'n_subjects': len(SUBJECTS),
    'caveat': 'n=9. Drift rates from H4 (same subjects). MID=0.01: engineering judgment.',
    'stratification': {'high_drift': HIGH_DRIFT, 'low_drift': LOW_DRIFT},
    'strategies': {n: {'label': STRATEGIES[n]['label'],
                        'mean': float(all_results[n]['mean']),
                        'std': float(all_results[n]['std'])} for n in STRATEGIES},
    's3_vs_s1': {'mean_diff': float(mean_diff), 'ci_95': [float(ci_low), float(ci_high)],
                  'p_pos': float(p_pos), 'p_above_mid': float(p_above_mid), 'decision': decision}
}
with open(os.path.join(RESULT_DIR, 'h5_results.json'), 'w') as f:
    json.dump(h5_out, f, indent=2)
print(f"Results: {os.path.join(RESULT_DIR, 'h5_results.json')}")
print(f"\nH5 complete. Final decision: {decision}")
