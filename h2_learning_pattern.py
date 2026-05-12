#!/usr/bin/env python3
"""
Framework 2.0 — H2: Learning Pattern Discrimination
=====================================================
Hypothesis: What shape does learning take — linear, plateau, stage-like, or null?

Method: Compare 4 models on group-average accuracy (AIC).
  M_linear:    acc = a + b * session
  M_plateau:   acc = asymptote - (asymptote - y0) * exp(-c * (session-1))
  M_stage:     two-segment linear, breakpoint searched over sessions 4-8
  M_null:      acc = constant

Decision thresholds (Burnham & Anderson, 2004):
  delta_AIC < 2: models indistinguishable; choose simpler model
  delta_AIC >= 2: best model selected

Individual heterogeneity diagnosis (framework 2.0 optimized):
  Per-subject linear fit reported regardless of group result.
  If individual patterns diverge (e.g. some up, some down, some flat),
  this is noted as evidence of heterogeneity for downstream analysis.

Role: Descriptive router. H1 already modeled individual differences via
  hierarchical regression. H2 guides path choice only.

Input:  data/accuracy_matrix_10subjects.csv
Output: results/h2_results.json, figures/h2_learning_pattern.png
"""

import numpy as np
import pandas as pd
from scipy.optimize import curve_fit
from scipy import stats
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
ACC_PATH = os.path.join(DATA_DIR, "accuracy_matrix_10subjects.csv")

os.makedirs(FIG_DIR, exist_ok=True)
os.makedirs(RESULT_DIR, exist_ok=True)

EXCLUDED = ['S10']
N_SESSIONS = 11
DELTA_AIC_THRESHOLD = 2.0  # Burnham & Anderson (2004)

# ============================================================
# Load
# ============================================================
print("H2: Learning Pattern Discrimination")
print("Model comparison: Linear vs Plateau vs Stage vs Null")
print("-" * 50)

df = pd.read_csv(ACC_PATH, index_col=0)
df_clean = df.drop(index=EXCLUDED, errors='ignore')
SUBJECTS = df_clean.index.tolist()
N_SUBJECTS = len(SUBJECTS)
print(f"  Subjects: {N_SUBJECTS} (excluded: {EXCLUDED})")

group_acc = df_clean.mean(axis=0).values
sessions = np.arange(1, N_SESSIONS + 1)
n = len(sessions)

print(f"\n  Group mean accuracy:")
for s, acc in zip(sessions, group_acc):
    print(f"    S{s:2d}: {acc:.4f}")

# ============================================================
# Models
# ============================================================
def linear(x, a, b):
    return a + b * x

def plateau(x, asymptote, y0, rate):
    """Exponential approach: asymptote - (asymptote - y0) * exp(-rate*(x-1))."""
    return asymptote - (asymptote - y0) * np.exp(-rate * (x - 1))

def stage_two_line(x, a1, b1, a2, b2, k):
    y = np.zeros_like(x, dtype=float)
    y[x <= k] = a1 + b1 * x[x <= k]
    y[x > k]  = a2 + b2 * x[x > k]
    return y

# ============================================================
# Fit
# ============================================================
results = {}

# Null
rss_null = np.sum((group_acc - np.mean(group_acc))**2)
aic_null = n * np.log(rss_null / n) + 2 * 1
bic_null = n * np.log(rss_null / n) + 1 * np.log(n)
results['Null'] = {'aic': aic_null, 'bic': bic_null, 'n_params': 1, 'rss': rss_null}

# Linear
try:
    p, _ = curve_fit(linear, sessions, group_acc, p0=[0.75, 0.01])
    rss = np.sum((group_acc - linear(sessions, *p))**2)
    results['Linear'] = {'aic': n*np.log(rss/n)+4, 'bic': n*np.log(rss/n)+2*np.log(n),
                          'n_params': 2, 'rss': rss, 'params': {'a': p[0], 'b': p[1]}}
except Exception as e:
    print(f"  Linear fit failed: {e}")

# Plateau
try:
    p, _ = curve_fit(plateau, sessions, group_acc, p0=[group_acc[-1], group_acc[0], 0.3], maxfev=5000)
    rss = np.sum((group_acc - plateau(sessions, *p))**2)
    results['Plateau'] = {'aic': n*np.log(rss/n)+6, 'bic': n*np.log(rss/n)+3*np.log(n),
                           'n_params': 3, 'rss': rss,
                           'params': {'asymptote': p[0], 'y0': p[1], 'rate': p[2]}}
except Exception as e:
    print(f"  Plateau fit failed: {e}")

# Stage (search k in 4-8)
best = None
best_aic = np.inf
for k in range(4, 9):
    try:
        p, _ = curve_fit(lambda x, a1, b1, a2, b2: stage_two_line(x, a1, b1, a2, b2, k),
                         sessions, group_acc, p0=[group_acc[0], 0.02, group_acc[-1], 0.0])
        rss = np.sum((group_acc - stage_two_line(sessions, *p, k))**2)
        aic = n*np.log(rss/n) + 10  # 4 params + 1 breakpoint = 5
        if aic < best_aic:
            best_aic = aic
            bic = n*np.log(rss/n) + 5*np.log(n)
            best = {'aic': aic, 'bic': bic, 'n_params': 5, 'rss': rss,
                    'params': {'a1': p[0], 'b1': p[1], 'a2': p[2], 'b2': p[3], 'k': k}}
    except: continue
if best:
    results['Stage'] = best

# ============================================================
# Select
# ============================================================
print(f"\n{'Model':<10} {'AIC':>10} {'BIC':>10} {'Params':>8} {'RSS':>10}")
print("-" * 52)
for name in ['Null', 'Linear', 'Plateau', 'Stage']:
    if name in results:
        r = results[name]
        print(f"{name:<10} {r['aic']:>10.2f} {r['bic']:>10.2f} {r['n_params']:>8} {r['rss']:>10.4f}")

sorted_by_aic = sorted([(n, r) for n, r in results.items()], key=lambda x: x[1]['aic'])
best_name = sorted_by_aic[0][0]
best_aic = sorted_by_aic[0][1]['aic']
second_name = sorted_by_aic[1][0] if len(sorted_by_aic) > 1 else 'N/A'
second_aic = sorted_by_aic[1][1]['aic'] if len(sorted_by_aic) > 1 else float('inf')
delta_aic = second_aic - best_aic

simplicity = {'Null': 0, 'Linear': 1, 'Plateau': 2, 'Stage': 3}

if delta_aic < DELTA_AIC_THRESHOLD:
    chosen = sorted([(n, r) for n, r in results.items()], key=lambda x: simplicity.get(x[0], 99))[0][0]
    print(f"\n  delta_AIC={delta_aic:.2f} < {DELTA_AIC_THRESHOLD}: models indistinguishable.")
    print(f"  Choosing simpler: {chosen}")
else:
    chosen = best_name
    print(f"\n  delta_AIC={delta_aic:.2f} >= {DELTA_AIC_THRESHOLD}: {chosen} selected.")

if chosen == 'Stage':
    decision = "SKIP_TO_H5"
    reason = f"Stage transition at session {results['Stage']['params']['k']}."
elif chosen == 'Null':
    decision = "TERMINATE"
    reason = "Null model best: no detectable group pattern."
else:
    decision = "PROCEED_TO_H3"
    reason = f"{chosen} pattern: proceed to H3 (neural coupling)."

print(f"  Decision: {decision}")
print(f"  {reason}")

# ============================================================
# Individual heterogeneity
# ============================================================
print(f"\nIndividual linear fits (OLS, descriptive):")
n_up, n_down, n_flat = 0, 0, 0
for subj in SUBJECTS:
    vals = df_clean.loc[subj].values
    valid = ~np.isnan(vals)
    if valid.sum() >= 5:
        slope, _, r_val, _, _ = stats.linregress(sessions[valid], vals[valid])
        r2 = r_val**2
        if r2 > 0.2 and slope > 0:
            n_up += 1; tag = 'up'
        elif r2 > 0.2 and slope < 0:
            n_down += 1; tag = 'down'
        else:
            n_flat += 1; tag = 'flat'
        print(f"  {subj}: slope={slope*100:+.2f}%/sess, R²={r2:.3f} [{tag}]")
print(f"  Summary: {n_up} up, {n_down} down, {n_flat} flat")

# ============================================================
# Figure
# ============================================================
fig, ax = plt.subplots(figsize=(10, 6))
for subj in SUBJECTS:
    vals = df_clean.loc[subj].values
    valid = ~np.isnan(vals)
    ax.plot(sessions[valid], vals[valid], 'o-', alpha=0.35, linewidth=0.8, markersize=3, color='gray')

ax.plot(sessions, group_acc, 'o-', color='black', linewidth=2.5, markersize=8,
        markerfacecolor='white', markeredgewidth=2, label='Group mean', zorder=5)

smooth_x = np.linspace(1, N_SESSIONS, 100)
if chosen == 'Linear':
    p = results['Linear']['params']
    ax.plot(smooth_x, linear(smooth_x, p['a'], p['b']), '-', color='#d7191c', linewidth=2,
            label=f'Linear (AIC={results["Linear"]["aic"]:.1f})')
elif chosen == 'Plateau':
    p = results['Plateau']['params']
    ax.plot(smooth_x, plateau(smooth_x, p['asymptote'], p['y0'], p['rate']), '-', color='#d7191c', linewidth=2,
            label=f'Plateau (AIC={results["Plateau"]["aic"]:.1f})')
elif chosen == 'Stage':
    p = results['Stage']['params']
    ax.plot(smooth_x, stage_two_line(smooth_x, p['a1'], p['b1'], p['a2'], p['b2'], p['k']),
            '-', color='#d7191c', linewidth=2, label=f'Stage (k={p["k"]}, AIC={results["Stage"]["aic"]:.1f})')

ax.axhline(y=0.50, color='gray', linestyle=':', alpha=0.5)
ax.set_xlabel('Session', fontsize=11)
ax.set_ylabel('Decoding Accuracy', fontsize=11)
ax.set_title(f'H2: Learning Pattern — {chosen} Best (delta_AIC={delta_aic:.1f})\n'
             f'{N_SUBJECTS} subjects: {n_up} up, {n_down} down, {n_flat} flat',
             fontsize=11, fontweight='bold')
ax.legend(fontsize=9)
ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False)
fig.tight_layout()
fig_path = os.path.join(FIG_DIR, 'h2_learning_pattern.png')
fig.savefig(fig_path, dpi=150, bbox_inches='tight', facecolor='white')
plt.close()
print(f"\nFigure: {fig_path}")

# Save
h2_results = {
    'framework': '2.0', 'stage': 'H2',
    'n_subjects': N_SUBJECTS, 'excluded': EXCLUDED,
    'group_mean': {f'S{i+1}': float(v) for i, v in enumerate(group_acc)},
    'model_comparison': {n: {'aic': float(r['aic']), 'bic': float(r['bic']),
                              'n_params': r['n_params'], 'rss': float(r['rss'])}
                          for n, r in results.items()},
    'chosen_model': chosen, 'delta_aic': float(delta_aic),
    'decision': decision, 'reason': reason,
    'individual_summary': f'{n_up} up, {n_down} down, {n_flat} flat'
}
with open(os.path.join(RESULT_DIR, 'h2_results.json'), 'w') as f:
    json.dump(h2_results, f, indent=2)
print(f"Results: {os.path.join(RESULT_DIR, 'h2_results.json')}")
print(f"\nH2 complete: {decision}")
