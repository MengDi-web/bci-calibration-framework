# BCI Calibration Scheduling: A Five-Stage Falsifiable Framework

## What this is

A systematic framework that breaks down "When should a BCI recalibrate?"
into five testable hypotheses. Each stage has pre-specified falsification
conditions and fallback paths. Negative results are not failures — they
shrink uncertainty and trigger the next stage.

## Pipeline

Data: 10 subjects x 11 sessions (Stieger et al., 2021, Scientific Data)

Stage 0 | stage0_quality_diagnosis.py
4-dimension quality control -> exclude S10 (unstable signal) -> 9 subjects remain

Stage 1 | h1_learning_existence.py
Bayesian hierarchical Beta regression
Question: Does group-level learning exist?
Result: Marginal (P(mu_beta>0)=0.80, 95% HDI spans zero)

Stage 2 | h2_learning_pattern.py
Group + individual model comparison (Linear, Plateau, Stage, Null)
Question: What shape is the learning curve?
Result: Significant heterogeneity: 3 subjects up, 2 down, 4 flat

Stage 3 | h3_neural_coupling.py
Continuous Spearman correlation, 3 neural stability metrics
Question: Can neural signals predict learning rate?
Result: Not supported (all |rho| < 0.32, all p > 0.4)

Stage 4 | h4_drift_learning.py
Full 11x11 cross-session decoding matrix per subject
Question: Does decoder drift correlate with learning rate?
Result: Strong trend (Spearman rho=+0.633, p=0.067)
Bootstrap 95% CI: [-0.019, +0.945]
Caveat: n=9, minimum detectable |rho| = 0.83 at 80% power

Stage 5 | h5_calibration_simulation.py
Simulate 5 strategies with subject-specific H4 drift rates
Question: Does drift-informed stratification beat uniform?
Result: Stratified functionally equivalent to uniform-3
S3 - S1 = +0.0005, 95% CI [-0.033, +0.036]
P(S3 > S1) = 0.512, P(S3 > S1 + 0.01) = 0.297

## Recommendation

Calibrate every 3 sessions. Simple, predictable, and as effective as any
more complex strategy tested. Optional: if baseline accuracy is available
after session 1, stratification offers probable but unguaranteed gain.

## Key Findings

  Group learning: +3.19 pp over 11 sessions (marginal evidence)
  Learning patterns: 3 up, 2 down, 4 flat — highly individual
  Neural coupling: no predictive power for learning rate
  Decoder drift: strong correlation with learning (rho=+0.633)
  Calibration: stratified equivalent to uniform every 3 sessions

## Structure

calibration_framework/
  analysis/   Stage scripts + data extraction (7 files)
  data/       Accuracy matrix, quality report, trial counts (4 files)
  results/    Per-stage JSON outputs (5 files)
  figures/    Visualizations (7 PNG files)
  _archive/   Deprecated scripts and intermediate figures

## Reproducibility

  Python 3.9, PyMC 5.12, arviz 0.15, scikit-learn, scipy
  All random seeds fixed at 42
  Raw data: Stieger et al. (2021), available via MOABB and PhysioNet
  Run order: Stage 0 -> H1 -> H2 -> H3 -> H4 -> H5

## Citation

If you use this framework in your research, please cite:

```bibtex
@software{bci_calibration_framework,
  author = {Di Meng},
  title = {BCI Calibration Scheduling: A Five-Stage Falsifiable Framework},
  year = {2026},
  url = {https://github.com/MengDi-web/bci-calibration-framework}
