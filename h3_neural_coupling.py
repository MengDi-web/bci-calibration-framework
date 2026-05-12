#!/usr/bin/env python3
"""
Framework 2.0 — H3: Neural-Behavioral Coupling (Heterogeneity Path)
====================================================================
H2 diagnosis: Significant individual heterogeneity (3 up, 2 down, 4 flat).

H3 question: Do neural stability metrics correlate with behavioral learning rate?
Design: Continuous Spearman correlation, NOT group comparison (n=2-3 per group
  is too small for valid statistical inference).

Metrics (all computed per session on full training data, not cross-validated):
  1. Weight stability: mean cosine similarity of LDA coefficient vectors
     between consecutive sessions. Dimension mismatch (different PCA n_components
     across sessions) handled by truncating to min common dimension.
  2. PCA stability: mean cosine similarity of first principal component
     between consecutive sessions.
  3. Decoding entropy: mean LDA posterior entropy across all trials in a session.
     Lower entropy = more confident (less uncertain) classifications.

Caveats:
  - n=9, exploratory. 80% power only detects |rho| >= 0.83.
  - All metrics computed on training data (not CV). Entropy and LDA weights
    may be overfit relative to out-of-sample performance.
  - PCA component sign ambiguity: cosine similarity uses absolute value,
    accepting that PC1 sign is arbitrary.

Input:  Raw .mat files (9 subjects) + results/h1_results.json
Output: results/h3_results.json, figures/h3_neural_coupling.png
"""

import numpy as np
import pandas as pd
from scipy.io import loadmat
from scipy.spatial.distance import cosine
from scipy import stats
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os, json, warnings, time
warnings.filterwarnings('ignore')

# ============================================================
# Configuration
# ============================================================
DATA_DIR = os.path.expanduser("~/mne_data/MNE-Stieger2021-data")
FIG_DIR = os.path.expanduser("~/Desktop/bci_project/calibration_framework/figures")
RESULT_DIR = os.path.expanduser("~/Desktop/bci_project/calibration_framework/results")
H1_RESULTS = os.path.join(RESULT_DIR, "h1_results.json")

os.makedirs(FIG_DIR, exist_ok=True)
os.makedirs(RESULT_DIR, exist_ok=True)

SUBJECTS = ['S1', 'S2', 'S4', 'S7', 'S8', 'S9', 'S11', 'S12', 'S13']
N_SESSIONS = 11
PCA_VARIANCE = 0.95
RANDOM_SEED = 42
FS = 1000
TIME_START_IDX = 2500; TIME_END_IDX = 5200   # Stage 0: 500-3200ms post-cue
REQUIRED_LEN = TIME_END_IDX

print("H3: Neural-Behavioral Coupling (Continuous Correlation)")
print(f"  Subjects: {len(SUBJECTS)}")
print(f"  Caveat: n=9. Min detectable |rho| ~0.83 (80% power)")
print("-" * 50)

# ============================================================
# 1. Behavioral slopes from H1
# ============================================================
with open(H1_RESULTS, 'r') as f:
    h1 = json.load(f)
slopes = {s: h1['individual_slopes'][s]['mean_accuracy_change'] for s in SUBJECTS}

print("Behavioral slopes (acc change / 11 sessions):")
for s in SUBJECTS:
    print(f"  {s}: {slopes[s]:+.5f}")

# ============================================================
# 2. Extract neural metrics from raw data
# ============================================================
print("\nExtracting neural metrics ...")
t0 = time.time()

neural = {}
for subj in SUBJECTS:
    sess_data = {}
    for sess in range(1, N_SESSIONS + 1):
        fp = os.path.join(DATA_DIR, f"{subj}_Session_{sess}.mat")
        if not os.path.exists(fp): continue
        try:
            mat = loadmat(fp); bci = mat['BCI'][0,0]; data = bci['data']; td = bci['TrialData']
            bidx, blab = [], []
            for i in range(td.shape[1]):
                tn = td[0,i]['targetnumber']
                if isinstance(tn, np.ndarray) and tn.size==1:
                    v = int(tn.item())
                    if v in [1,2]: bidx.append(i); blab.append(v)
            if len(bidx) < 15: continue
            X, y = [], []
            for idx, lab in zip(bidx, blab):
                eeg = data[0,idx]
                if eeg.shape[1] < REQUIRED_LEN: continue
                X.append(eeg[:, TIME_START_IDX:TIME_END_IDX].ravel())
                y.append(lab)
            if len(X) < 15: continue
            X, y = np.array(X), np.array(y)

            scaler = StandardScaler().fit(X)
            X_sc = scaler.transform(X)
            mc = min(len(y), X.shape[1]) - 1
            pca = PCA(n_components=PCA_VARIANCE, random_state=RANDOM_SEED)
            X_pca = pca.fit_transform(X_sc)
            if X_pca.shape[1] > mc:
                pca = PCA(n_components=mc, random_state=RANDOM_SEED)
                X_pca = pca.fit_transform(X_sc)
            lda = LinearDiscriminantAnalysis()
            lda.fit(X_pca, y)
            proba = np.clip(lda.predict_proba(X_pca), 1e-10, 1-1e-10)
            entropy = -np.mean(np.sum(proba * np.log(proba), axis=1))

            sess_data[sess] = {
                'lda_coef': lda.coef_.copy().flatten(),
                'pca_comp1': pca.components_[0].copy(),
                'entropy': float(entropy),
                'n_comp': pca.n_components_
            }
        except Exception: continue

    sessions = sorted(sess_data.keys())
    if len(sessions) < 3:
        neural[subj] = None; continue

    wsim, psim, evals = [], [], []
    for i in range(len(sessions)):
        s = sessions[i]
        evals.append(sess_data[s]['entropy'])
        if i > 0:
            sp = sessions[i-1]
            # Weight stability: truncate to min dimension to handle PCA mismatch
            w1 = sess_data[sp]['lda_coef']; w2 = sess_data[s]['lda_coef']
            md = min(len(w1), len(w2))
            if md > 0 and np.any(w1[:md]) and np.any(w2[:md]):
                wsim.append(1 - cosine(w1[:md], w2[:md]))
            # PCA stability
            pc1 = sess_data[sp]['pca_comp1']; pc2 = sess_data[s]['pca_comp1']
            if len(pc1) == len(pc2) and np.any(pc1) and np.any(pc2):
                psim.append(1 - cosine(pc1, pc2))

    neural[subj] = {
        'weight_stability': float(np.nanmean(wsim)) if wsim else np.nan,
        'pca_stability': float(np.nanmean(psim)) if psim else np.nan,
        'entropy_mean': float(np.nanmean(evals)) if evals else np.nan
    }
    print(f"  {subj}: w_stab={neural[subj]['weight_stability']:.3f}, "
          f"pca_stab={neural[subj]['pca_stability']:.3f}, entropy={neural[subj]['entropy_mean']:.3f}")

print(f"  Extraction: {time.time()-t0:.0f}s")

# ============================================================
# 3. Spearman correlation (prespecified primary analysis)
# ============================================================
print(f"\n{'='*50}")
print(f"Spearman: Neural Stability vs Behavioral Slope")
print(f"{'='*50}")

slope_arr = np.array([slopes[s] for s in SUBJECTS])
wstab = np.array([neural[s]['weight_stability'] if neural[s] else np.nan for s in SUBJECTS])
pstab = np.array([neural[s]['pca_stability'] if neural[s] else np.nan for s in SUBJECTS])
ent = np.array([neural[s]['entropy_mean'] if neural[s] else np.nan for s in SUBJECTS])

metrics = {'Weight Stability': wstab, 'PCA Stability': pstab, 'Decoding Entropy': ent}
corr_results = {}

for name, vals in metrics.items():
    mask = ~np.isnan(vals) & ~np.isnan(slope_arr)
    if mask.sum() < 5:
        print(f"\n{name}: insufficient ({mask.sum()} pairs)")
        continue
    rho, p = stats.spearmanr(slope_arr[mask], vals[mask])
    print(f"\n{name}:")
    print(f"  Spearman rho = {rho:+.3f}, p = {p:.4f}")
    corr_results[name] = {'spearman_rho': float(rho), 'spearman_p': float(p), 'n': int(mask.sum())}

# ============================================================
# Figure
# ============================================================
fig, axes = plt.subplots(1, 3, figsize=(18, 5.5))
fig.subplots_adjust(wspace=0.35)

for ax, (name, vals) in zip(axes, metrics.items()):
    mask = ~np.isnan(vals) & ~np.isnan(slope_arr)
    if mask.sum() < 3:
        ax.text(0.5, 0.5, 'Insufficient data', ha='center', va='center', transform=ax.transAxes)
        ax.set_title(name, fontsize=11, fontweight='bold')
        continue
    x, y = slope_arr[mask], vals[mask]
    ax.scatter(x, y, s=80, c='#2c7bb6', edgecolors='white', linewidth=0.5, zorder=5)
    for i, s in enumerate(np.array(SUBJECTS)[mask]):
        ax.annotate(s, (x[i], y[i]), textcoords="offset points", xytext=(5,5), fontsize=7)
    if len(x) >= 3:
        z = np.polyfit(x, y, 1)
        ax.plot(np.linspace(x.min(), x.max(), 50),
                np.polyval(z, np.linspace(x.min(), x.max(), 50)),
                '--', color='#d7191c', linewidth=1.5, alpha=0.7)
    if name in corr_results:
        ax.set_title(f'{name}\nSpearman rho={corr_results[name]["spearman_rho"]:+.3f}, '
                     f'p={corr_results[name]["spearman_p"]:.3f}', fontsize=11, fontweight='bold')
    ax.set_xlabel('Behavioral Slope (acc change / 11 sessions)', fontsize=9)
    ax.set_ylabel(name, fontsize=9)
    ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False)

fig.suptitle(f'H3: Neural-Behavioral Coupling\n'
             f'n={len(SUBJECTS)}, exploratory | Min detectable |rho| ~0.83',
             fontsize=13, fontweight='bold', y=1.02)
fig_path = os.path.join(FIG_DIR, 'h3_neural_coupling.png')
fig.savefig(fig_path, dpi=150, bbox_inches='tight', facecolor='white')
plt.close()
print(f"\nFigure: {fig_path}")

# Save
h3_out = {
    'framework': '2.0', 'stage': 'H3',
    'n_subjects': len(SUBJECTS),
    'caveat': 'n=9. Min detectable |rho| ~0.83 (80% power).',
    'neural_metrics': neural,
    'correlations': corr_results
}
with open(os.path.join(RESULT_DIR, 'h3_results.json'), 'w') as f:
    json.dump(h3_out, f, indent=2, default=str)
print(f"Results: {os.path.join(RESULT_DIR, 'h3_results.json')}")
print(f"\nH3 complete.")
