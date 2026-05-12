#!/usr/bin/env python3
"""
Framework 2.0 — Stage 0: Data Quality Diagnosis
=================================================
Pre-hypothesis quality control. Four independent dimensions, each must PASS.

D1: Signal Strength
    Wilcoxon signed-rank (one-sided: median > 0.50), Holm-Bonferroni across
    11 sessions. Threshold: >50% sessions significant.
    Chosen over t-test because accuracy is bounded [0,1], violating normality
    near ceiling/floor.

D2: Within-Subject Reliability
    (a) 3-sigma rule: per-subject outlier detection, adapts to individual
        variability rather than applying a fixed threshold across subjects.
    (b) CV > 0.20: at mean=0.70, allows SD up to 0.14 before flagging.
    (c) Recovery jump: deviation > 2*SD that returns within 2 sessions.
        Distinguishes noise from genuine learning or performance shifts.

D3: Cross-Subject Comparability
    Per-subject Wilcoxon (one-sided: median > 0.50), Holm-corrected across
    subjects. Allows at most 1 failure: n=11 per subject limits statistical
    power for detecting moderate effects in individual Wilcoxon tests.

D4: Data Completeness
    >= 80% subjects complete AND missing unrelated to baseline accuracy
    (Mann-Whitney test). Four dimensions judged independently; no weighted
    composite that could mask single-dimension failures.

Input:  data/accuracy_matrix_10subjects.csv
Output: data/stage0_quality_report.json, figures/stage0_quality_summary.png
"""

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os, json
from scipy import stats

# ============================================================
# Paths
# ============================================================
DATA_DIR = os.path.expanduser("~/Desktop/bci_project/calibration_framework/data")
FIG_DIR = os.path.expanduser("~/Desktop/bci_project/calibration_framework/figures")
ACC_PATH = os.path.join(DATA_DIR, "accuracy_matrix_10subjects.csv")
REPORT_PATH = os.path.join(DATA_DIR, "stage0_quality_report.json")
os.makedirs(FIG_DIR, exist_ok=True)

# ============================================================
# Load
# ============================================================
df = pd.read_csv(ACC_PATH, index_col=0)
SUBJECTS = df.index.tolist()
N_SUBJECTS = len(SUBJECTS)
N_SESSIONS = df.shape[1]

print(f"Stage 0: Data Quality Diagnosis")
print(f"  Subjects: {N_SUBJECTS}, Sessions: {N_SESSIONS}")
print()

# ============================================================
# Holm-Bonferroni correction
# ============================================================
def holm_correct(p_values):
    """Return boolean array: True if significant after Holm correction (alpha=0.05)."""
    n = len(p_values)
    idx = np.argsort(p_values)
    sorted_p = np.array(p_values)[idx]
    sig = np.zeros(n, dtype=bool)
    for k, (i, p) in enumerate(zip(idx, sorted_p)):
        if p < 0.05 / (n - k):
            sig[i] = True
        else:
            break
    return sig

# ============================================================
# D1: Signal Strength
# ============================================================
print("D1: Signal Strength (Wilcoxon signed-rank, Holm-corrected)")
print("-" * 50)

p_vals_d1, wilcoxon_res = [], {}
for sess in df.columns:
    vals = df[sess].dropna().values
    if len(vals) < 5:
        wilcoxon_res[sess] = {'n': len(vals), 'median': np.nan, 'p_raw': np.nan, 'sig': False}
        p_vals_d1.append(1.0)
    else:
        r = stats.wilcoxon(vals - 0.50, alternative='greater')
        p_vals_d1.append(r.pvalue)
        wilcoxon_res[sess] = {'n': len(vals), 'median': round(np.median(vals), 4),
                               'mean': round(np.mean(vals), 4), 'p_raw': round(r.pvalue, 6), 'sig': None}

holm_sig = holm_correct(p_vals_d1)
sessions_above = []
for i, sess in enumerate(df.columns):
    wilcoxon_res[sess]['sig'] = bool(holm_sig[i])
    if holm_sig[i]:
        sessions_above.append(sess)
    status = 'PASS' if wilcoxon_res[sess]['sig'] else 'FAIL'
    print(f"  {status} {sess}: median={wilcoxon_res[sess]['median']:.4f}, "
          f"n={wilcoxon_res[sess]['n']}, p={wilcoxon_res[sess]['p_raw']}")

n_above = len(sessions_above)
dim1_pass = n_above > N_SESSIONS / 2
print(f"  -> {n_above}/{N_SESSIONS} sessions above chance -> {'PASS' if dim1_pass else 'FAIL'}")
print()

# ============================================================
# D2: Within-Subject Reliability
# ============================================================
print("D2: Within-Subject Reliability (3-sigma, CV, recovery jumps)")
print("-" * 50)

reliability_flags = {}
for subj in SUBJECTS:
    vals = df.loc[subj].dropna()
    n_valid = len(vals)
    if n_valid < 3:
        reliability_flags[subj] = ['INSUFF_DATA']
        print(f"  {subj}: insufficient data (n={n_valid})")
        continue

    m, s = vals.mean(), vals.std()
    cv = s / m if m > 0 else float('inf')
    flags = []

    # 3-sigma: per-subject, adapts to individual variability level
    upper, lower = m + 3*s, m - 3*s
    outliers = [f"{ss}={v:.3f}" for ss, v in vals.items() if v > upper or v < lower]
    if outliers:
        flags.append(f'3SIGMA({len(outliers)}):{",".join(outliers[:3])}')

    # CV > 0.20: at mean=0.70, allows SD up to 0.14 before flagging
    if cv > 0.20:
        flags.append(f'CV({cv:.3f})')

    # Recovery jumps: large deviation (>2*SD) that returns within 2 sessions
    if n_valid >= 4:
        diffs = np.diff(vals.values)
        threshold = 2 * s
        for j in range(len(diffs)):
            if abs(diffs[j]) > threshold:
                pre = vals.values[j]
                recovery = False
                for ahead in range(1, min(3, len(vals.values) - j)):
                    if abs(vals.values[j+ahead] - pre) < s:
                        recovery = True
                        break
                if recovery:
                    flags.append(f'JUMP({vals.index[j]}->{vals.index[j+1]}: '
                                 f'{pre:.3f}->{vals.values[j+1]:.3f})')

    reliability_flags[subj] = flags if flags else ['OK']
    if flags:
        print(f"  FLAG {subj}: {' | '.join(flags)}  (mean={m:.3f}, sd={s:.3f})")
    else:
        print(f"  OK   {subj}: mean={m:.4f}, sd={s:.4f}")

n_flagged_d2 = sum(1 for v in reliability_flags.values() if v != ['OK'] and 'INSUFF_DATA' not in v)
dim2_pass = n_flagged_d2 <= N_SUBJECTS * 0.2
print(f"  -> {n_flagged_d2}/{N_SUBJECTS} flagged -> {'PASS' if dim2_pass else 'FAIL'}")
print()

# ============================================================
# D3: Cross-Subject Comparability
# ============================================================
print("D3: Cross-Subject Comparability (per-subject Wilcoxon + Holm)")
print("-" * 50)

p_vals_d3, comp_res = [], {}
for subj in SUBJECTS:
    vals = df.loc[subj].dropna()
    if len(vals) < 5:
        comp_res[subj] = {'n': len(vals), 'median': np.nan, 'p_raw': np.nan, 'sig': None}
        p_vals_d3.append(1.0)
    else:
        r = stats.wilcoxon(vals - 0.50, alternative='greater')
        p_vals_d3.append(r.pvalue)
        comp_res[subj] = {'n': len(vals), 'median': round(np.median(vals), 4),
                            'mean': round(np.mean(vals), 4),
                            'baseline': round(vals.iloc[0], 4) if len(vals) > 0 else None,
                            'p_raw': round(r.pvalue, 6), 'sig': None}

holm_sig_d3 = holm_correct(p_vals_d3)
near_chance = []
for i, subj in enumerate(SUBJECTS):
    comp_res[subj]['sig'] = bool(holm_sig_d3[i])
    if not holm_sig_d3[i]:
        near_chance.append(subj)
    status = 'PASS' if comp_res[subj]['sig'] else 'NEAR_CHANCE'
    print(f"  {status} {subj}: median={comp_res[subj]['median']}, "
          f"mean={comp_res[subj]['mean']}, p={comp_res[subj]['p_raw']}")

# Allow 1 failure: n=11 per subject limits Wilcoxon power for moderate effects
dim3_pass = len(near_chance) <= 1
print(f"  -> {len(near_chance)}/{N_SUBJECTS} near-chance -> {'PASS' if dim3_pass else 'FAIL'}")
if near_chance:
    print(f"     List: {', '.join(near_chance)}")
print()

# ============================================================
# D4: Data Completeness
# ============================================================
print("D4: Data Completeness")
print("-" * 50)

completeness = {}
for subj in SUBJECTS:
    n_valid = int(df.loc[subj].notna().sum())
    completeness[subj] = {'valid': n_valid, 'missing': N_SESSIONS - n_valid}

for subj in SUBJECTS:
    c = completeness[subj]
    print(f"  {'OK' if c['missing'] <= 2 else 'FLAG'} {subj}: "
          f"{c['valid']}/{N_SESSIONS} sessions")

subjects_miss = [s for s in SUBJECTS if completeness[s]['missing'] > 0]
subjects_comp = [s for s in SUBJECTS if completeness[s]['missing'] == 0]
nonrandom = False
if len(subjects_miss) >= 3 and len(subjects_comp) >= 3:
    bl_m = df.loc[subjects_miss].iloc[:, 0].dropna()
    bl_c = df.loc[subjects_comp].iloc[:, 0].dropna()
    if len(bl_m) >= 3 and len(bl_c) >= 3:
        mw = stats.mannwhitneyu(bl_m, bl_c, alternative='two-sided')
        nonrandom = mw.pvalue < 0.05
        print(f"\n  Missing-pattern test: p = {mw.pvalue:.4f}"
              f"{' * NON-RANDOM' if nonrandom else ' (random)'}")

n_complete = sum(1 for c in completeness.values() if c['missing'] == 0)
complete_rate = n_complete / N_SUBJECTS
dim4_pass = complete_rate >= 0.80 and not nonrandom
print(f"  -> {n_complete}/{N_SUBJECTS} complete ({complete_rate:.1%}), "
      f"non-random: {nonrandom} -> {'PASS' if dim4_pass else 'FAIL'}")
print()

# ============================================================
# Verdict
# ============================================================
print("=" * 50)
print("Verdict")
print("=" * 50)

dim_results = {'D1_SignalStrength': dim1_pass, 'D2_Reliability': dim2_pass,
               'D3_Comparability': dim3_pass, 'D4_Completeness': dim4_pass}
all_pass = all(dim_results.values())
for dim, result in dim_results.items():
    print(f"  {dim}: {'PASS' if result else 'FAIL'}")

if all_pass:
    print("\n  All dimensions PASS. Proceed to H1.")
else:
    failed = [d for d, r in dim_results.items() if not r]
    print(f"\n  Failed: {', '.join(failed)}")
    if not dim3_pass:
        print(f"  Near-chance: {near_chance}")
        print(f"  Remaining n = {N_SUBJECTS - len(near_chance)}")

# ============================================================
# Save report
# ============================================================
report = {
    'framework': '2.0', 'stage': 0,
    'n_subjects': N_SUBJECTS, 'n_sessions': N_SESSIONS,
    'overall_pass': all_pass,
    'dimensions': {
        'D1': {'pass': dim1_pass, 'n_above_chance': n_above},
        'D2': {'pass': dim2_pass, 'n_flagged': n_flagged_d2, 'details': reliability_flags},
        'D3': {'pass': dim3_pass, 'near_chance': near_chance},
        'D4': {'pass': dim4_pass, 'complete_rate': round(complete_rate, 4)}
    },
    'high_baseline_note': 'Subjects with Session 1 >= 0.80 noted for transparency.',
    'high_baseline_subjects': [s for s in SUBJECTS
                                if not pd.isna(df.loc[s].iloc[0]) and df.loc[s].iloc[0] >= 0.80]
}
with open(REPORT_PATH, 'w') as f:
    json.dump(report, f, indent=2, default=str)
print(f"\nReport: {REPORT_PATH}")

# ============================================================
# Figure: 2x2 panel (loose layout, no Chinese fonts)
# ============================================================
fig, axes = plt.subplots(2, 2, figsize=(16, 11))
fig.subplots_adjust(hspace=0.38, wspace=0.30)

# Panel A: Per-session accuracy
ax = axes[0, 0]
for i, sess in enumerate(df.columns):
    vals = df[sess].dropna().values
    jitter = np.random.normal(i+1, 0.06, size=len(vals))
    ax.scatter(jitter, vals, alpha=0.35, s=12, color='#2c7bb6', edgecolors='none')
ax.boxplot([df[s].dropna().values for s in df.columns],
           positions=range(1, N_SESSIONS+1), widths=0.5,
           patch_artist=True, showfliers=False,
           boxprops=dict(facecolor='#b3d9ff', alpha=0.6),
           medianprops=dict(color='#003366', linewidth=2))
ax.axhline(y=0.50, color='gray', linestyle='--', linewidth=0.8, alpha=0.5)
ax.set_xticks(range(1, N_SESSIONS+1))
ax.set_xticklabels([f'S{i+1}' for i in range(N_SESSIONS)], fontsize=6.5)
ax.set_ylabel('Decoding Accuracy', fontsize=9)
ax.set_title(f'D1: Signal Strength ({n_above}/{N_SESSIONS} above chance)',
             fontsize=10, fontweight='bold')
ax.set_ylim(0.30, 1.02)
ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False)

# Panel B: Per-subject mean
ax = axes[0, 1]
subj_means = df.mean(axis=1).sort_values()
colors = ['#d7191c' if s in near_chance else '#2c7bb6' for s in subj_means.index]
ax.barh(range(len(subj_means)), subj_means.values, color=colors, alpha=0.75,
        edgecolor='white', height=0.7)
ax.axvline(x=0.50, color='gray', linestyle='--', linewidth=0.8, alpha=0.5)
ax.set_yticks(range(len(subj_means)))
ax.set_yticklabels(subj_means.index, fontsize=7)
ax.set_xlabel('Mean Accuracy', fontsize=9)
ax.set_title(f'D3: Comparability ({len(near_chance)} near-chance)',
             fontsize=10, fontweight='bold')
ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False)

# Panel C: Missing data matrix
ax = axes[1, 0]
ax.imshow(df.isna().astype(int), aspect='auto', cmap='Reds', vmin=0, vmax=1, alpha=0.8)
ax.set_xticks(range(N_SESSIONS))
ax.set_xticklabels([f'S{i+1}' for i in range(N_SESSIONS)], fontsize=6.5)
ax.set_yticks(range(N_SUBJECTS))
ax.set_yticklabels(SUBJECTS, fontsize=7)
ax.set_title(f'D4: Missing Data ({n_complete}/{N_SUBJECTS} complete)',
             fontsize=10, fontweight='bold')

# Panel D: Verdict summary
ax = axes[1, 1]
ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.axis('off')
dim_labels = [
    'D1: Signal Strength (Wilcoxon + Holm)',
    'D2: Reliability (3-sigma, CV, jumps)',
    'D3: Comparability (per-subject Wilcoxon)',
    'D4: Completeness (>=80%, random missing)'
]
dim_passes = [dim1_pass, dim2_pass, dim3_pass, dim4_pass]
for i, (label, passed) in enumerate(zip(dim_labels, dim_passes)):
    y = 0.88 - i * 0.19
    color = '#1a9641' if passed else '#d7191c'
    symbol = 'PASS' if passed else 'FAIL'
    ax.text(0.05, y, label, fontsize=9, va='center', fontfamily='monospace')
    ax.text(0.78, y, symbol, fontsize=11, fontweight='bold', color=color, va='center',
            bbox=dict(boxstyle='round,pad=0.3', facecolor=color, alpha=0.12,
                      edgecolor=color, linewidth=1))
verdict_text = 'ALL PASS -> H1' if all_pass else 'REVIEW BEFORE H1'
vc = '#1a9641' if all_pass else '#d7191c'
ax.text(0.5, 0.06, verdict_text, fontsize=14, fontweight='bold', color=vc,
        ha='center', va='bottom',
        bbox=dict(boxstyle='round,pad=0.5', facecolor=vc, alpha=0.08,
                  edgecolor=vc, linewidth=1.5))

fig.suptitle(f'Stage 0: Data Quality Diagnosis\n'
             f'{N_SUBJECTS} subjects x {N_SESSIONS} sessions',
             fontsize=13, fontweight='bold', y=1.01)
fig_path = os.path.join(FIG_DIR, 'stage0_quality_summary.png')
fig.savefig(fig_path, dpi=150, bbox_inches='tight', facecolor='white')
plt.close()
print(f"Figure: {fig_path}")
print("Stage 0 complete.")
