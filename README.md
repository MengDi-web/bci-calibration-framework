Framework: Five-Stage Falsifiable Framework for BCI Calibration Scheduling

Overview
========
Tests the logical chain from "does learning exist?" to "when to calibrate?"
using five interconnected hypothesis tests with pre-specified falsification paths.

Pipeline
========
data/accuracy_matrix_10subjects.csv (10 subjects x 11 sessions)
    |
    Stage 0: stage0_quality_diagnosis.py
    4-dimension quality diagnosis -> exclude S10
    |
    H1: h1_learning_existence.py
    Bayesian hierarchical Beta regression -> marginal positive (P>0=0.80)
    |
    H2: h2_learning_pattern.py
    Group + individual model comparison -> significant heterogeneity
    (3/9 linear up, 2/9 linear down, 4/9 flat)
    |
    H3: h3_neural_coupling.py
    Neural stability vs learning rate -> not supported (all |rho|<0.32, p>0.4)
    |
    H4: h4_drift_learning.py
    Decoder drift vs learning rate -> strong trend (rho=+0.633, p=0.067)
    |
    H5: h5_calibration_simulation.py
    Drift-informed stratification vs uniform -> functionally equivalent
    (difference = +0.0005, 95% CI [-0.033, +0.036])

Key Findings
============
  - Group learning: marginal positive (+3.19 pp/11 sessions, P>0=0.80)
  - Learning patterns: significant individual heterogeneity
  - Neural coupling: no predictive power for learning rate
  - Decoder drift: strong positive correlation with learning (rho=+0.633)
  - Calibration: stratified functionally equivalent to uniform every 3 sessions

Subjects
========
First 10 with complete 11-session records (numerical order):
S1, S2, S4, S7, S8, S9, S10, S11, S12, S13
Excluded: S10 (unstable signal, CV=0.24, PCA frequently 1-3 components)

Directory Structure
===================
analysis/   Executable scripts (7 files)
data/       Accuracy matrix, quality report, trial counts (4 files)
results/    Per-stage JSON outputs (5 files)
figures/    Visualizations (7 PNG files)
_archive/   Deprecated scripts and intermediate figures

Reproducibility
===============
Python 3.9, PyMC 5.12, arviz 0.15
All random seeds fixed at 42
Run order: Stage 0 -> H1 -> H2 -> H3 -> H4 -> H5
Raw data: Stieger2021 dataset (Scientific Data, 2021)

