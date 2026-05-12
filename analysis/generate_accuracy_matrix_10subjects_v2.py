#!/usr/bin/env python3
"""Correct 10 subjects: S1-S13 (numerical order), first 10 with complete 11 sessions."""
import numpy as np
import pandas as pd
from scipy.io import loadmat
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.decomposition import PCA
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
import os, json, warnings, time
warnings.filterwarnings('ignore')

DATA_DIR = os.path.expanduser("~/mne_data/MNE-Stieger2021-data")
OUTPUT_DIR = os.path.expanduser("~/Desktop/bci_project/calibration_framework/data")
OUTPUT_PATH = os.path.join(OUTPUT_DIR, "accuracy_matrix_10subjects.csv")

# Correct numerical order: first 10 with complete 11 sessions
SUBJECTS = ['S1', 'S2', 'S4', 'S7', 'S8', 'S9', 'S10', 'S11', 'S12', 'S13']
N_SESSIONS = 11; N_FOLDS = 5; PCA_VARIANCE = 0.95; RANDOM_SEED = 42; FS = 1000
TIME_START_IDX = 2500
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Step 0
print("Step 0: Scanning trial durations...")
all_dur = []
for subj in SUBJECTS:
    for sess in range(1, N_SESSIONS+1):
        fp = os.path.join(DATA_DIR, f"{subj}_Session_{sess}.mat")
        if not os.path.exists(fp): continue
        try:
            mat = loadmat(fp); bci = mat['BCI'][0,0]; data, td = bci['data'], bci['TrialData']
            for i in range(data.shape[1]):
                tn = td[0,i]['targetnumber']
                if isinstance(tn, np.ndarray) and tn.size==1:
                    if int(tn.item()) not in [1,2]: continue
                else: continue
                all_dur.append(data[0,i].shape[1] - 2000)
        except: continue
all_dur = np.array(all_dur)
p5, p1 = np.percentile(all_dur, 5), np.percentile(all_dur, 1)
p_target = p5 if p5 >= 2000 else p1
WINDOW_ACTUAL = min(2700, int(p_target))
TIME_END_IDX = TIME_START_IDX + WINDOW_ACTUAL
REQUIRED_LEN = TIME_END_IDX
cov = np.sum(all_dur >= WINDOW_ACTUAL)/len(all_dur)*100
print(f"  {len(all_dur)} trials, P5={p5:.0f}ms, P1={p1:.0f}ms, Window: {WINDOW_ACTUAL}ms ({WINDOW_ACTUAL/FS:.2f}s), Coverage: {cov:.1f}%")

# Step 1
results, trial_counts, quality = {}, {}, {}
t0 = time.time()
for subj in SUBJECTS:
    s_acc, s_trial, s_qual = [], [], []
    print(f"\n{subj}:")
    for sess in range(1, N_SESSIONS+1):
        fp = os.path.join(DATA_DIR, f"{subj}_Session_{sess}.mat")
        if not os.path.exists(fp):
            s_acc.append(np.nan); s_trial.append(0); s_qual.append('MISSING')
            continue
        try:
            mat = loadmat(fp); bci = mat['BCI'][0,0]; data = bci['data']; td = bci['TrialData']
            bidx, blab = [], []
            for i in range(td.shape[1]):
                tn = td[0,i]['targetnumber']
                if isinstance(tn, np.ndarray) and tn.size==1:
                    v = int(tn.item())
                    if v in [1,2]: bidx.append(i); blab.append(v)
            if len(bidx) < 15:
                s_acc.append(np.nan); s_trial.append(len(bidx)); s_qual.append(f'FEW({len(bidx)})')
                print(f"  S{sess:02d}: FEW BINARY ({len(bidx)}) -> NaN")
                continue
            X_list, y_list, skipped = [], [], 0
            for idx, lab in zip(bidx, blab):
                eeg = data[0,idx]
                if eeg.shape[1] < REQUIRED_LEN: skipped+=1; continue
                X_list.append(eeg[:, TIME_START_IDX:TIME_END_IDX].ravel())
                y_list.append(lab)
            if len(X_list) < 15:
                s_acc.append(np.nan); s_trial.append(len(X_list)); s_qual.append(f'FEW_VALID({len(X_list)})')
                print(f"  S{sess:02d}: FEW VALID ({len(X_list)}, skip={skipped}) -> NaN")
                continue
            X, y = np.array(X_list), np.array(y_list); n_trials = len(y)
            skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=RANDOM_SEED)
            fold_accs, fold_errs = [], 0
            for tr, te in skf.split(X, y):
                try:
                    Xtr = StandardScaler().fit_transform(X[tr]); Xte = StandardScaler().fit(X[tr]).transform(X[te])
                    mc = min(len(tr), X.shape[1])-1
                    pca = PCA(n_components=PCA_VARIANCE, random_state=RANDOM_SEED)
                    Xtr_pca = pca.fit_transform(Xtr)
                    if Xtr_pca.shape[1] > mc: pca = PCA(n_components=mc, random_state=RANDOM_SEED); Xtr_pca = pca.fit_transform(Xtr)
                    Xte_pca = pca.transform(Xte)
                    lda = LinearDiscriminantAnalysis(); lda.fit(Xtr_pca, y[tr])
                    fold_accs.append(lda.score(Xte_pca, y[te]))
                except: fold_errs+=1; fold_accs.append(np.nan)
            valid = [a for a in fold_accs if not np.isnan(a)]
            if len(valid) < 3:
                s_acc.append(np.nan); s_trial.append(n_trials); s_qual.append(f'FOLD_ERR({fold_errs})')
                print(f"  S{sess:02d}: FOLD ERRORS ({fold_errs}/{N_FOLDS}) -> NaN")
                continue
            mean_acc = np.mean(valid); pca_var = pca.explained_variance_ratio_.sum()
            s_acc.append(round(mean_acc,4)); s_trial.append(n_trials)
            qtags = []
            if skipped>0: qtags.append(f'skip{skipped}')
            if fold_errs>0: qtags.append(f'fe{fold_errs}')
            s_qual.append('+'.join(qtags) if qtags else 'OK')
            status = ' [!]' if qtags else ''
            print(f"  S{sess:02d}: n={n_trials}, PCA={pca.n_components_} ({pca_var:.1%}), acc={mean_acc:.4f}{status}")
        except Exception as e:
            s_acc.append(np.nan); s_trial.append(0); s_qual.append('ERROR')
            print(f"  S{sess:02d}: ERROR {str(e)[:60]}")
    results[subj] = s_acc; trial_counts[subj] = s_trial; quality[subj] = s_qual

sl = [f"Session_{i}" for i in range(1,N_SESSIONS+1)]
pd.DataFrame(results, index=sl).T.to_csv(OUTPUT_PATH)
pd.DataFrame(trial_counts, index=sl).T.to_csv(os.path.join(OUTPUT_DIR, "trial_counts_10subjects.csv"))
pd.DataFrame(quality, index=sl).T.to_csv(os.path.join(OUTPUT_DIR, "quality_flags_10subjects.csv"))
print(f"\nSaved: {OUTPUT_PATH}, Time: {time.time()-t0:.0f}s")
print(f"\nSummary:")
df = pd.read_csv(OUTPUT_PATH, index_col=0)
for subj in SUBJECTS:
    v = df.loc[subj]
    print(f"  {subj}: mean={v.mean():.4f}, S1={v.iloc[0]:.4f}, valid={v.notna().sum()}/{N_SESSIONS}")
