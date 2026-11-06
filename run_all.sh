#!/usr/bin/env bash
# Full pipeline, in the order of the article. Run from the repository root.
# Steps marked [GPU] fit NumPyro models: run them in the Docker image (see README) or on a machine with
# a CUDA-enabled JAX. To reuse the saved model outputs instead of refitting, skip the [GPU] and [fit]
# steps and place the outputs in results/fits/ (see README, "Reusing the saved model outputs").
set -euo pipefail
PY=${PY:-python}

$PY code/01_descriptives.py                          # data, early responses, Figures 1-2
$PY code/02_ddm_assumption_check.py                  # per-session DDM fits (~20-40 min, CPU)
$PY code/03_behavioral_models.py all                 # [fit] Bambi models (~5-10 min each)
$PY code/04_behavioral_contrasts.py                  # Tables 3, 4A, 5 (behavioural rows), Figure 5
$PY code/05_early_responding.py                      # individual differences in early responding

$PY code/06_fit_model1.py main                       # [GPU] Model 1
$PY code/07_model1_results.py                        # Model 1 estimates, contrasts, Figure 3

$PY code/08_fit_alternative_models.py model2         # [GPU] Model 2
$PY code/08_fit_alternative_models.py model3         # [GPU] Model 3 (long: 8 chains x 5,000 iterations)
$PY code/08_fit_alternative_models.py ddm_trunc all  # [GPU] hierarchical truncated DDM
$PY code/09_model_comparison.py                      # PSIS-LOO, Table 2
$PY code/10_loo_truncated.py                         # truncated DDM vs Model 1 on responses >= 150 ms
$PY code/11_posterior_predictive.py                  # predictive checks by condition, Figure 4

$PY code/12_parameter_recovery.py simulate           # simulated data set
$PY code/06_fit_model1.py recovery                   # [GPU] refit
$PY code/12_parameter_recovery.py evaluate           # recovery of parameters and contrasts

$PY code/06_fit_model1.py half0                      # [GPU] split-half fits of Model 1
$PY code/06_fit_model1.py half1                      # [GPU]
$PY code/08_fit_alternative_models.py ddm_trunc half0  # [GPU] split-half fits of the truncated DDM
$PY code/08_fit_alternative_models.py ddm_trunc half1  # [GPU]
$PY code/13_reliability.py                           # Table 6

$PY code/14_truncated_vs_model1.py                   # does the treatment of early responses change conclusions?
