#!/usr/bin/env python3
"""
Framework 2.0 — H4: Decoder Drift vs Learning Rate
====================================================
H3 conclusion: Neural stability does not correlate with behavioral learning rate.

H4 question: Does decoder performance decay across sessions correlate with learning?
Method: Full cross-session decoding matrix per subject.
  - Train decoder (PCA+LDA) on session i, test on session j (all i != j).
  - Diagonal: pre-computed 5-fold CV accuracy from accuracy matrix.
  - Drift = time-weighted mean performance drop per session of separation.
  - Spearman correlation with H1 individual learning slope, bootstrap 95% CI.

Design decisions:
  - Pre-load all sessions: avoids repeated .mat file I/O (efficiency).
  - Time-weighted drift: performance_drop / session_distance, averaged over all
    off-diagonal pairs. Accounts for larger drops across longer intervals.
  - Bootstrap CI (10k iterations): better uncertainty quantification than single
    p-value when n=9.
  - Explicit caveat: n=9, 80% power only detects |rho| >= 0.83.

Input:  Raw .mat files + data/accuracy_matrix_10subjects.csv +
        results/h1_results.json
Output: results/h4_results.json, figures/h4_drift_learning.png
"""

import numpy as np
import pandas as pd
from scipy.io import loadmat
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
ACC_PATH = os.path.expanduser("~/Desktop/bci_project/calibration_framework/data/accuracy_matrix_10subjects.csv")
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

print("H4: Decoder Drift vs Learning Rate")
print(f"  Subjects: {len(SUBJECTS)}")
print(f"  Caveat: n=9. Min detectable |rho| ~0.83 (80% power)")
print("-" * 50)

# ============================================================
# 1. Behavioral slopes from H1
# ============================================================
with open(H1_RESULTS, 'r') as f:
    h1 = json.load(f)
slopes = {s: h1['individual_slopes'][s]['mean_accuracy_change'] for s in SUBJECTS}

# Diagonal: pre-computed CV accuracy
df_acc = pd.read_csv(ACC_PATH, index_col=0)
diag = {s: df_acc.loc[s].values for s in SUBJECTS}

print("Behavioral slopes (acc change / 11 sessions):")
for s in SUBJECTS:
    print(f"  {s}: {slopes[s]:+.4f}")

# ============================================================
# 2. Pre-load all session data
# ============================================================
print("\nPre-loading session data ...")
t0 = time.time()
all_data = {}
for subj in SUBJECTS:
    all_data[subj] = {}
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
            all_data[subj][sess] = (np.array(X), np.array(y))
        except Exception: continue
    print(f"  {subj}: {len(all_data[subj])}/{N_SESSIONS}")
print(f"  Loaded in {time.time()-t0:.0f}s")

# ============================================================
# 3. Cross-session decoding matrix
# ============================================================
print("\nComputing cross-session decoding ...")
t0 = time.time()
drift_rates = {}

for subj in SUBJECTS:
    sessions = sorted(all_data[subj].keys())
    n_valid = len(sessions)
    if n_valid < 5: drift_rates[subj] = np.nan; continue

    dec_mat = np.full((n_valid, n_valid), np.nan)

    for i_idx, i_sess in enumerate(sessions):
        X_train, y_train = all_data[subj][i_sess]
        n_train = len(y_train)

        scaler = StandardScaler().fit(X_train)
        X_tr = scaler.transform(X_train)
        max_comp = min(n_train, X_train.shape[1]) - 1
        pca = PCA(n_components=PCA_VARIANCE, random_state=RANDOM_SEED)
        X_tr_pca = pca.fit_transform(X_tr)
        if X_tr_pca.shape[1] > max_comp:
            pca = PCA(n_components=max_comp, random_state=RANDOM_SEED)
            X_tr_pca = pca.fit_transform(X_tr)
        lda = LinearDiscriminantAnalysis()
        lda.fit(X_tr_pca, y_train)

        for j_idx, j_sess in enumerate(sessions):
            if i_sess == j_sess:
                dec_mat[i_idx, j_idx] = diag[subj][j_sess - 1]
            else:
                X_test, y_test = all_data[subj][j_sess]
                X_te = scaler.transform(X_test)
                dec_mat[i_idx, j_idx] = lda.score(pca.transform(X_te), y_test)

    # Time-weighted drift: drop per session of separation
    diag_vals = np.array([dec_mat[i,i] for i in range(n_valid)])
    wdrifts = []
    for i in range(n_valid):
        for j in range(n_valid):
            if i != j and not np.isnan(dec_mat[i,j]):
                dist = abs(sessions[i] - sessions[j])
                drop = dec_mat[i,i] - dec_mat[i,j]
                if dist > 0: wdrifts.append(drop / dist)

    drift_rates[subj] = float(np.nanmean(wdrifts)) if wdrifts else np.nan
    dtag = f"{drift_rates[subj]:.5f}/session" if not np.isnan(drift_rates[subj]) else "NaN"
    print(f"  {subj}: diag_mean={np.nanmean(diag_vals):.4f}, "
          f"n_pairs={len(wdrifts)}, drift={dtag}")

print(f"  Done in {time.time()-t0:.0f}s")

# ============================================================
# 4. Correlation
# ============================================================
print(f"\n{'='*50}")
print(f"Spearman: Drift Rate vs Behavioral Slope")
print(f"{'='*50}")

slope_arr = np.array([slopes[s] for s in SUBJECTS])
drift_arr = np.array([drift_rates[s] for s in SUBJECTS])

missing = [s for s in SUBJECTS if np.isnan(drift_rates[s])]
if missing: print(f"  Missing: {', '.join(missing)}")

mask = ~np.isnan(drift_arr) & ~np.isnan(slope_arr)

if mask.sum() >= 5:
    rho, p = stats.spearmanr(slope_arr[mask], drift_arr[mask])
    print(f"  Spearman rho = {rho:+.3f}, p = {p:.4f}")

    np.random.seed(42)
    bs = np.zeros(10000)
    for b in range(10000):
        idx = np.random.choice(mask.sum(), mask.sum(), replace=True)
        bs[b], _ = stats.spearmanr(slope_arr[mask][idx], drift_arr[mask][idx])
    ci = (np.percentile(bs, 2.5), np.percentile(bs, 97.5))
    print(f"  Bootstrap 95% CI: [{ci[0]:+.3f}, {ci[1]:+.3f}]")
    print(f"  Valid: {mask.sum()}/{len(SUBJECTS)}")

    for s in SUBJECTS:
        i = SUBJECTS.index(s)
        print(f"  {s}: slope={slope_arr[i]:+.4f}, drift={drift_arr[i]:+.5f}")
else:
    rho, p, ci = np.nan, np.nan, (np.nan, np.nan)
    print(f"  Insufficient: {mask.sum()}")

# ============================================================
# Figure
# ============================================================
fig, ax = plt.subplots(figsize=(9, 6))
fig.subplots_adjust(left=0.12, right=0.92, top=0.92, bottom=0.12)

if mask.sum() >= 3:
    x, y = slope_arr[mask], drift_arr[mask]
    ax.scatter(x, y, s=100, c='#2c7bb6', edgecolors='white', linewidth=0.5, zorder=5)
    for i, s in enumerate(np.array(SUBJECTS)[mask]):
        ax.annotate(s, (x[i], y[i]), textcoords="offset points", xytext=(7, 5), fontsize=10)
    if len(x) >= 3:
        z = np.polyfit(x, y, 1)
        xl = np.linspace(x.min(), x.max(), 50)
        ax.plot(xl, np.polyval(z, xl), '--', color='#d7191c', linewidth=2, alpha=0.6)
    ax.set_xlabel('Behavioral Slope (acc change / 11 sessions)', fontsize=11)
    ax.set_ylabel('Time-Weighted Drift Rate (/session)', fontsize=11)
    ax.set_title(f'H4: Decoder Drift vs Learning Rate\n'
                 f'Spearman rho={rho:+.3f}, 95% CI [{ci[0]:+.3f}, {ci[1]:+.3f}], n={mask.sum()}',
                 fontsize=11, fontweight='bold')
else:
    ax.text(0.5, 0.5, 'Insufficient data', ha='center', va='center', transform=ax.transAxes, fontsize=14)

ax.axhline(y=0, color='gray', linestyle=':', alpha=0.3)
ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False)
fig_path = os.path.join(FIG_DIR, 'h4_drift_learning.png')
fig.savefig(fig_path, dpi=150, bbox_inches='tight', facecolor='white')
plt.close()
print(f"\nFigure: {fig_path}")

# Save
h4_out = {
    'framework': '2.0', 'stage': 'H4',
    'n_subjects': len(SUBJECTS),
    'caveat': 'n=9. Min detectable |rho| ~0.83 (80% power).',
    'drift_rates': {s: float(v) if not np.isnan(v) else None for s, v in drift_rates.items()},
    'correlation': {'spearman_rho': float(rho), 'spearman_p': float(p),
                     'bootstrap_ci_95': [float(ci[0]), float(ci[1])],
                     'n_pairs': int(mask.sum())} if mask.sum() >= 5 else None
}
with open(os.path.join(RESULT_DIR, 'h4_results.json'), 'w') as f:
    json.dump(h4_out, f, indent=2, default=str)
print(f"Results: {os.path.join(RESULT_DIR, 'h4_results.json')}")
print(f"\nH4 complete.")
