#!/usr/bin/env python3
"""
Framework 2.0 — H1: Group-level Learning Existence
===================================================
Hypothesis: Decoding accuracy improves across sessions at the group level.

Model: Bayesian hierarchical Beta regression
  Level 1: accuracy_ij ~ Beta(mu_ij * kappa, (1 - mu_ij) * kappa)
            logit(mu_ij) = alpha_i + beta_i * session_c_j
  Level 2: alpha_i ~ Normal(mu_alpha, sigma_alpha)
            beta_i ~ Normal(mu_beta, sigma_beta)

Falsification: 95% HDI of mu_beta includes zero -> H1 not supported.
If HDI spans zero but P(mu_beta > 0) >= 0.75: marginal evidence, proceed to H2
with caution.

Priors (weakly informative):
  mu_beta ~ Normal(0, 0.02): 95% mass within +/- 0.04 logit/session,
    equivalent to +/- 2.2%/session on accuracy scale. Covers typical
    SMR-BCI learning rates (0-3%/session; Stieger et al. 2021).
  kappa ~ HalfNormal(50): Beta precision, allows moderate overdispersion.

Design decisions:
  - Beta likelihood: accuracy bounded [0,1]; avoids Normal's support outside [0,1].
  - Session centered at session 6: minimizes posterior correlation between
    alpha_i and beta_i, improving sampling geometry.
  - Non-centered parameterization: avoids Neal's funnel degeneracy.
  - Accuracy compressed to [0.005, 0.995] for Beta support.

Effect size: Simulated accuracy change over 11 sessions (session 1 vs 11)
  from posterior draws, avoiding local linear approximation.

Input:  data/accuracy_matrix_10subjects.csv
Output: results/h1_results.json, figures/h1_results.png, h1_individual_slopes.png
"""

import numpy as np
import pandas as pd
import pymc as pm
import arviz as az
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os, json, warnings
warnings.filterwarnings('ignore')

# ============================================================
# Configuration
# ============================================================
RANDOM_SEED = 42
DATA_DIR = os.path.expanduser("~/Desktop/bci_project/calibration_framework/data")
FIG_DIR = os.path.expanduser("~/Desktop/bci_project/calibration_framework/figures")
RESULT_DIR = os.path.expanduser("~/Desktop/bci_project/calibration_framework/results")
ACC_PATH = os.path.join(DATA_DIR, "accuracy_matrix_10subjects.csv")

os.makedirs(FIG_DIR, exist_ok=True)
os.makedirs(RESULT_DIR, exist_ok=True)

# Stage 0: S10 excluded (CV=0.24, unstable signal, PCA frequently 1-3)
EXCLUDED = ['S10']

N_CHAINS = 4        # 4 chains for robust R-hat and ESS
N_TUNE = 2000       # warmup sufficient for hierarchical model convergence
N_DRAWS = 2000      # 8000 total posterior samples
TARGET_ACCEPT = 0.90  # higher than default 0.80 for better sampling in hierarchical models
EPSILON = 0.005       # clip accuracy to [0.005, 0.995] for Beta support

# ============================================================
# Load data
# ============================================================
print("H1: Group-level Learning Existence")
print("Model: Bayesian hierarchical Beta regression")
print("-" * 50)

df = pd.read_csv(ACC_PATH, index_col=0)
df_clean = df.drop(index=EXCLUDED, errors='ignore')
SUBJECTS = df_clean.index.tolist()
N_SUBJECTS = len(SUBJECTS)
N_SESSIONS = df_clean.shape[1]

data_list = []
for subj in SUBJECTS:
    for sess_idx in range(N_SESSIONS):
        acc = df_clean.loc[subj].iloc[sess_idx]
        if not pd.isna(acc):
            data_list.append({
                'subject_idx': SUBJECTS.index(subj),
                'session_c': sess_idx - (N_SESSIONS - 1) / 2,  # centered at session 6
                'accuracy': acc
            })

data = pd.DataFrame(data_list)
subject_idx = data['subject_idx'].values
session_c = data['session_c'].values
accuracy_beta = np.clip(data['accuracy'].values, EPSILON, 1 - EPSILON)
N_OBS = len(accuracy_beta)

print(f"  Subjects: {N_SUBJECTS} (excluded: {', '.join(EXCLUDED)})")
print(f"  Observations: {N_OBS} ({N_SESSIONS} sessions each)")
print(f"  Accuracy: mean={accuracy_beta.mean():.4f}, SD={accuracy_beta.std():.4f}")
print()

# ============================================================
# Prior predictive check
# ============================================================
print("Prior predictive check ...")

with pm.Model() as model_prior:
    mu_alpha = pm.Normal('mu_alpha', mu=1.0, sigma=0.5)
    sigma_alpha = pm.HalfNormal('sigma_alpha', sigma=0.5)
    mu_beta = pm.Normal('mu_beta', mu=0.0, sigma=0.02)
    sigma_beta = pm.HalfNormal('sigma_beta', sigma=0.015)
    kappa = pm.HalfNormal('kappa', sigma=50)
    
    z_alpha = pm.Normal('z_alpha', mu=0, sigma=1, shape=N_SUBJECTS)
    z_beta = pm.Normal('z_beta', mu=0, sigma=1, shape=N_SUBJECTS)
    
    alpha_i = mu_alpha + z_alpha * sigma_alpha
    beta_i = mu_beta + z_beta * sigma_beta
    
    mu_ij = pm.math.invlogit(alpha_i[subject_idx] + beta_i[subject_idx] * session_c)
    y_pred = pm.Beta('y_pred', alpha=mu_ij * kappa, beta=(1 - mu_ij) * kappa, shape=N_OBS)

with model_prior:
    prior_pred = pm.sample_prior_predictive(samples=500, random_seed=RANDOM_SEED)

prior_acc = prior_pred.prior['y_pred'].values.flatten()
print(f"  Prior predictive: mean={prior_acc.mean():.3f}, SD={prior_acc.std():.3f}")
print()

# ============================================================
# H1 model
# ============================================================
print("Fitting H1 model ...")

with pm.Model() as model_h1:
    # Group-level hyperparameters
    mu_alpha = pm.Normal('mu_alpha', mu=1.0, sigma=0.5)
    sigma_alpha = pm.HalfNormal('sigma_alpha', sigma=0.5)
    
    mu_beta = pm.Normal('mu_beta', mu=0.0, sigma=0.02)  # focal parameter
    sigma_beta = pm.HalfNormal('sigma_beta', sigma=0.015)
    
    kappa = pm.HalfNormal('kappa', sigma=50)
    
    # Non-centered parameterization
    z_alpha = pm.Normal('z_alpha', mu=0, sigma=1, shape=N_SUBJECTS)
    alpha_i = pm.Deterministic('alpha_i', mu_alpha + z_alpha * sigma_alpha)
    
    z_beta = pm.Normal('z_beta', mu=0, sigma=1, shape=N_SUBJECTS)
    beta_i = pm.Deterministic('beta_i', mu_beta + z_beta * sigma_beta)
    
    # Likelihood
    logit_mu = alpha_i[subject_idx] + beta_i[subject_idx] * session_c
    mu = pm.math.invlogit(logit_mu)
    y_obs = pm.Beta('y_obs', alpha=mu * kappa, beta=(1 - mu) * kappa, observed=accuracy_beta)
    
    trace = pm.sample(draws=N_DRAWS, tune=N_TUNE, chains=N_CHAINS,
                      target_accept=TARGET_ACCEPT, random_seed=RANDOM_SEED, progressbar=True)

# ============================================================
# Diagnostics
# ============================================================
print()
rhat = az.rhat(trace)
for var in ['mu_beta', 'mu_alpha', 'kappa']:
    if var in rhat:
        val = rhat[var].values
        if hasattr(val, 'ravel'): val = val.ravel()[0]
        print(f"  R-hat {var}: {val:.4f}")
div = int(trace.sample_stats['diverging'].sum().values)
print(f"  Divergences: {div} / {N_CHAINS * N_DRAWS}")
ess = az.ess(trace)
if hasattr(ess, 'to_dict'):
    ess_val = ess['mu_beta'].values if 'mu_beta' in ess else float('nan')
    print(f"  ESS (mu_beta): {ess_val:.0f}")

# ============================================================
# H1 test
# ============================================================
print()
print("H1 Test:")
print("-" * 50)

mu_beta_samples = trace.posterior['mu_beta'].values.flatten()
mu_beta_mean = float(np.mean(mu_beta_samples))
mu_beta_hdi = az.hdi(mu_beta_samples, hdi_prob=0.95)
hdi_low, hdi_high = float(mu_beta_hdi[0]), float(mu_beta_hdi[1])

print(f"  mu_beta (logit/session):")
print(f"    Mean:     {mu_beta_mean:.5f}")
print(f"    SD:       {np.std(mu_beta_samples):.5f}")
print(f"    95% HDI:  [{hdi_low:.5f}, {hdi_high:.5f}]")
print(f"    P(mu_beta > 0):  {np.mean(mu_beta_samples > 0):.4f}")

# Simulated 11-session effect
acc_start = 1 / (1 + np.exp(-mu_beta_samples * (-5.0)))
acc_end   = 1 / (1 + np.exp(-mu_beta_samples * 5.0))
acc_diff  = acc_end - acc_start
beta_acc_mean = float(np.mean(acc_diff))
beta_acc_hdi_lo = float(az.hdi(acc_diff, hdi_prob=0.95)[0])
beta_acc_hdi_hi = float(az.hdi(acc_diff, hdi_prob=0.95)[1])

print(f"\n  Accuracy change over 11 sessions (simulated):")
print(f"    Mean:     {beta_acc_mean:.4f} ({beta_acc_mean*100:.2f} pp)")
print(f"    95% HDI:  [{beta_acc_hdi_lo:.4f}, {beta_acc_hdi_hi:.4f}]")

hdi_contains_zero = hdi_low <= 0 <= hdi_high
h1_supported = not hdi_contains_zero

print(f"\n  HDI contains zero: {hdi_contains_zero}")
print(f"  H1: {'SUPPORTED' if h1_supported else 'NOT SUPPORTED'}")
if not h1_supported:
    print(f"  -> Proceed to H2 (learning pattern).")
print()

# ============================================================
# Individual slopes
# ============================================================
beta_subj = trace.posterior['beta_i'].values.reshape(-1, N_SUBJECTS).T

print("Individual Learning Effects (11-session simulated change):")
print("  Note: n=11 per subject -> wide uncertainty on individual estimates.")
print(f"{'Subject':<8} {'Mean':>9} {'95% HDI':>28} {'P(>0)':>8}")
print("-" * 57)
for i, subj in enumerate(SUBJECTS):
    b_s = beta_subj[i]
    s1  = 1 / (1 + np.exp(-b_s * (-5.0)))
    s11 = 1 / (1 + np.exp(-b_s * 5.0))
    diff = s11 - s1
    m = np.mean(diff) * 100
    h = az.hdi(diff, hdi_prob=0.95)
    p = np.mean(diff > 0)
    print(f"{subj:<8} {m:>+8.2f}%  [{h[0]*100:>8.2f}, {h[1]*100:>8.2f}]  {p:>8.3f}")

# ============================================================
# Posterior predictive check
# ============================================================
print()
print("Posterior predictive check ...")
with model_h1:
    post_pred = pm.sample_posterior_predictive(trace, random_seed=RANDOM_SEED)
ppc_acc = post_pred.posterior_predictive['y_obs'].values.flatten()
print(f"  Observed:  mean={accuracy_beta.mean():.4f}, SD={accuracy_beta.std():.4f}")
print(f"  Predicted: mean={ppc_acc.mean():.4f}, SD={ppc_acc.std():.4f}")

# ============================================================
# Save
# ============================================================
results = {
    'framework': '2.0', 'stage': 'H1',
    'n_subjects': N_SUBJECTS, 'n_sessions': N_SESSIONS,
    'excluded': EXCLUDED, 'subjects': SUBJECTS,
    'mu_beta': {
        'mean': mu_beta_mean,
        'sd': float(np.std(mu_beta_samples)),
        'hdi_95': [hdi_low, hdi_high],
        'p_gt_0': float(np.mean(mu_beta_samples > 0)),
        'accuracy_change_11sessions_mean': beta_acc_mean,
        'accuracy_change_11sessions_hdi': [beta_acc_hdi_lo, beta_acc_hdi_hi]
    },
    'h1_supported': h1_supported,
    'individual_slopes': {
        subj: {
            'mean_accuracy_change': float(np.mean(
                1/(1+np.exp(-beta_subj[i]*5)) - 1/(1+np.exp(-beta_subj[i]*(-5)))
            )),
            'p_gt_0': float(np.mean(beta_subj[i] > 0))
        }
        for i, subj in enumerate(SUBJECTS)
    }
}
with open(os.path.join(RESULT_DIR, 'h1_results.json'), 'w') as f:
    json.dump(results, f, indent=2)
print(f"\nResults: {os.path.join(RESULT_DIR, 'h1_results.json')}")

# ============================================================
# Figures (loose layout, no Chinese)
# ============================================================
fig, axes = plt.subplots(1, 2, figsize=(16, 5.5))
fig.subplots_adjust(wspace=0.30)

# Panel A: Posterior of 11-session accuracy change
ax = axes[0]
ax.hist(acc_diff * 100, bins=60, color='#2c7bb6', alpha=0.7, edgecolor='white', density=True)
ax.axvline(x=0, color='gray', linestyle='--', linewidth=1.2)
ax.axvline(x=beta_acc_mean * 100, color='#d7191c', linewidth=2,
           label=f'Mean = {beta_acc_mean*100:.2f} pp')
ax.axvspan(beta_acc_hdi_lo * 100, beta_acc_hdi_hi * 100,
           alpha=0.15, color='#2c7bb6', label='95% HDI')
ax.set_xlabel('Accuracy Change over 11 Sessions (pp)', fontsize=10)
ax.set_ylabel('Posterior Density', fontsize=10)
ax.set_title(f'H1: Group-level Learning\nP(mu_beta > 0) = {np.mean(mu_beta_samples > 0):.3f}',
             fontsize=11, fontweight='bold')
ax.legend(fontsize=8)
ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False)

# Panel B: Prior + posterior predictive checks
ax = axes[1]
bins = 35
ax.hist(accuracy_beta, bins=bins, color='gray', alpha=0.5, density=True, label='Observed')
ax.hist(prior_acc, bins=bins, color='#fdae61', alpha=0.3, density=True, label='Prior predictive')
ax.hist(ppc_acc, bins=bins, color='#2c7bb6', alpha=0.3, density=True, label='Posterior predictive')
ax.set_xlabel('Decoding Accuracy', fontsize=10)
ax.set_ylabel('Density', fontsize=10)
ax.set_title('Prior & Posterior Predictive Checks', fontsize=11, fontweight='bold')
ax.legend(fontsize=8)
ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False)

fig.suptitle(f'H1: Bayesian Hierarchical Beta Regression\n'
             f'{N_SUBJECTS} subjects x {N_SESSIONS} sessions | '
             f'H1 {"SUPPORTED" if h1_supported else "NOT SUPPORTED"}',
             fontsize=13, fontweight='bold', y=1.02)
fig.savefig(os.path.join(FIG_DIR, 'h1_results.png'), dpi=150, bbox_inches='tight', facecolor='white')
plt.close()

# Individual forest plot
fig, ax = plt.subplots(figsize=(10, 6))
fig.subplots_adjust(left=0.15, right=0.92, top=0.93, bottom=0.08)

slope_acc, slope_hdis = [], []
for i in range(N_SUBJECTS):
    b_s = beta_subj[i]
    diff = 1/(1+np.exp(-b_s*5)) - 1/(1+np.exp(-b_s*(-5)))
    slope_acc.append(np.mean(diff) * 100)
    slope_hdis.append((float(az.hdi(diff, hdi_prob=0.95)[0]) * 100,
                       float(az.hdi(diff, hdi_prob=0.95)[1]) * 100))

sort_idx = np.argsort(slope_acc)
for i, idx in enumerate(sort_idx):
    color = '#2c7bb6' if slope_acc[idx] > 0 else '#d7191c'
    hdi_has_zero = slope_hdis[idx][0] <= 0 <= slope_hdis[idx][1]
    ax.errorbar(slope_acc[idx], i,
                xerr=[[slope_acc[idx] - slope_hdis[idx][0]], [slope_hdis[idx][1] - slope_acc[idx]]],
                fmt='o', color=color, capsize=3, markersize=7,
                markerfacecolor='white' if hdi_has_zero else color,
                markeredgewidth=2, markeredgecolor=color)
ax.axvline(x=0, color='gray', linestyle='--', linewidth=0.8)
ax.set_yticks(range(N_SUBJECTS))
ax.set_yticklabels([SUBJECTS[i] for i in sort_idx], fontsize=9)
ax.set_xlabel('Accuracy Change over 11 Sessions (pp)', fontsize=10)
ax.set_title('Individual Learning Effects (sorted)\n'
             'Note: n=11 per subject; wide uncertainty',
             fontsize=11, fontweight='bold')
ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False)
fig.savefig(os.path.join(FIG_DIR, 'h1_individual_slopes.png'), dpi=150, bbox_inches='tight', facecolor='white')
plt.close()

print(f"Figures: {FIG_DIR}/h1_results.png, h1_individual_slopes.png")
print(f"\nH1 complete: {'SUPPORTED' if h1_supported else 'NOT SUPPORTED'}")
