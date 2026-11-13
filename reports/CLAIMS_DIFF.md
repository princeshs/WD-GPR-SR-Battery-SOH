# Empirical Claims & Metric Differences: Old vs. New Regenerated

**Author:** Senior ML Research Engineer (Student: Prince Kumar, Roll No: 2401AI38)  
**Date:** 2026-11-10  
**Project:** WD-GPR-SR Battery State of Health (SOH) Estimation  
**Status:** Phase 2B Verification Completed Across Seeds $\{42, 123, 999\}$  

---

## 1. Executive Summary of Protocol Repair
During the strict Phase 1 audit and Phase 2A/2B code refactoring, three fundamental methodological flaws were corrected:
1. **Confounded Model Comparison:** Previously, Config C was trained on an 85% train split with early stopping, while Config E's base GP was trained on 100% without early stopping. Both now use the exact same canonical `train_gp_model()` recipe ($\max |\Delta \mu_{\text{GP}}| = 0.0$).
2. **SR Over-fitting & Degradation:** Previously, adding unconstrained SR worsened base GP performance (from 1.73% to 2.53% on CALCE, and from 3.33% to 4.93% on NASA). With nested leave-one-training-cell-out selection (P3 requirement), candidate $g = 0$ is evaluated against candidate parsimonies. Because genetic programming failed to improve cross-cell generalization out-of-fold, $g = 0$ was honestly selected across all folds, preserving base GP accuracy.
3. **Hardcoded Metrics & Confounded Calibration:** The master runner previously hardcoded $\text{MAPE} = \text{MAE} \times 1.12$, while Config E was the only model evaluated with recalibrated variance $s$. Calibration is now computed strictly from inner OOF residuals and reported symmetrically across all probabilistic models.

---

## 2. Quantitative Claims Difference Table

| Metric / Model | Old Value (`results/archive_old/`) | New Regenerated Value (LOCO $\times$ Seeds) | Reason for Discrepancy & File:Line Evidence |
| :--- | :--- | :--- | :--- |
| **NASA Proposed (Config E) RMSE** | **$4.93 \pm 1.48\%$** | **$3.33 \pm 2.51\%$** (cells) / $\pm 0.23\%$ (seeds) | Old 4.93% degraded base GP due to unconstrained SR fitting on train residuals. Nested inner-LOCO selection honestly selected $g = 0$ across all 12 folds ($E = C$). |
| **NASA Unweighted SR (Config D) RMSE** | **$4.31 \pm 1.86\%$** | **$4.14 \pm 1.39\%$** (cells) / $\pm 0.44\%$ (seeds) | Evaluated with actual discovered unweighted equations across 12 folds. |
| **NASA WD-GPR (Config C) RMSE** | **$3.41 \pm 2.22\%$** | **$3.33 \pm 2.51\%$** (cells) / $\pm 0.23\%$ (seeds) | Retrained with single canonical `train_gp_model` (`src/gpr_models.py#L369`). |
| **CALCE Proposed (Config E) RMSE** | **$2.53 \pm 1.56\%$** | **$1.64 \pm 0.72\%$** (cells) / $\pm 0.15\%$ (seeds) | Old 2.53% degraded base GP; nested selection selected $g = 0$ across all 12 folds ($E = C$). |
| **CALCE Unweighted SR (Config D) RMSE** | **$3.65 \pm 3.02\%$** | **$3.94 \pm 3.41\%$** (cells) / $\pm 0.91\%$ (seeds) | Severe over-fitting on CS2_37 (RMSE 8.18% vs base 1.79%). |
| **CALCE WD-GPR (Config C) RMSE** | **$3.30 \pm 1.15\%$** | **$1.64 \pm 0.72\%$** (cells) / $\pm 0.15\%$ (seeds) | Old 3.30% was an artifact of 85% train split in Stage 4; canonical training on all training cells achieves 1.64%. |
| **CALCE Random Forest RMSE** | **$3.39 \pm 0.78\%$** | **$2.31 \pm 0.79\%$** | CALCE cohort restriction to 4 benchmark cells (`CS2_35`–`CS2_38`, excluding `CS2_33`/`34`) and removal of 257 corrupted outlier cycles (`src/run_stage3.py#L55`). |
| **CALCE SVR (RBF) RMSE** | **$3.87 \pm 0.80\%$** | **$1.96 \pm 0.69\%$** | Outlier filtering (`data/processed/features_calce.csv` 124 outliers on `CS2_35`, 133 on `CS2_38` with corrupted IC peaks removed). |
| **CALCE BiLSTM RMSE** | **$3.08 \pm 1.11\%$** | **$1.34 \pm 0.41\%$** | Cleaned sequence length without corrupted cycles (`src/baseline_models.py#L650`). |
| **CALCE Linear Regression RMSE** | **$4.61 \pm 1.10\%$** | **$2.82 \pm 0.42\%$** | Elimination of extreme tail artifacts and non-benchmark training cells. |
| **CALCE Exact GPR (Matérn-5/2) RMSE** | **$3.82 \pm 0.79\%$** | **$2.03 \pm 0.61\%$** | Clean 4-cell benchmark cohort with zero-leakage standard scaling. |
| **CALCE Exact GPR (ARD) RMSE** | **$3.55 \pm 0.84\%$** | **$2.04 \pm 0.60\%$** | Clean 4-cell benchmark cohort (`src/run_stage3.py#L55`). |
| **CALCE Exact GPR (RBF) RMSE** | **$3.93 \pm 0.85\%$** | **$2.17 \pm 0.62\%$** | Clean 4-cell benchmark cohort (`src/run_stage3.py#L55`). |
| **CALCE DeepNN RMSE** | **$3.53 \pm 0.94\%$** | **$1.64 \pm 0.52\%$** | Clean 4-cell benchmark cohort; DeepNN performs competitively (1.637%) with base GP. |
| **CALCE WideNN RMSE** | **$3.79 \pm 0.76\%$** | **$2.08 \pm 0.60\%$** | Clean 4-cell benchmark cohort (`src/run_stage3.py#L55`). |
| **CALCE Wide & Deep NN RMSE** | **$3.74 \pm 1.09\%$** | **$2.69 \pm 0.75\%$** | Clean 4-cell benchmark cohort (`src/run_stage3.py#L55`). |
| **CALCE Deep Kernel GPR RMSE** | **$3.51 \pm 0.83\%$** | **$1.56 \pm 0.58\%$** | Clean 4-cell benchmark cohort (`src/run_stage3.py#L55`). |
| **CALCE Sparse GPR (SVGP) RMSE** | **$3.69 \pm 1.01\%$** | **$1.81 \pm 0.64\%$** | Clean 4-cell benchmark cohort (`src/run_stage3.py#L55`). |
| **Standalone SR CALCE RMSE** | **$55.51 \pm 34.99\%$** | **$4.40 \pm 0.70\%$** | Target standardization, safe operator guarding, and $[0, 110\%]$ prediction clamping (`src/baseline_models.py#L550`). |
| **Standalone SR NASA RMSE** | **$28.40 \pm 14.20\%$** | **$6.20 \pm 2.92\%$** | Target standardization prevents asymptotic pole divergence. |
| **CALCE 95% PICP (Config E)** | **$90.6\%$** (recalibrated only) | **$91.1\%$** (raw) $\to$ **$94.5\%$** (recalibrated) | Recalibration factor $s=1.34$ brings empirical coverage to within 0.5% of nominal 95%. |
| **NASA 95% PICP (Config E)** | **$77.4\%$** (raw only) | **$76.8\%$** (raw) $\to$ **$74.4\%$** (variance-recal) / **$95.0\%$** (quantile-recal) | Variance scaling collapses due to heavy tails; quantile scaling restores 95% nominal coverage. |
| **NASA MAPE (Config E)** | Hardcoded $\approx \text{MAE} \times 1.12$ | Full: **$3.81\%$** <br> $\text{SOH} \ge 50\%$: **$3.81\%$** <br> sMAPE: **$3.69\%$** | True MAPE computed without hardcoded scaling; NASA has no cycles $< 50\%$ SOH. |
| **CALCE MAPE (Config E)** | Hardcoded $\approx \text{MAE} \times 1.12$ | Full: **$2.26\%$** <br> $\text{SOH} \ge 50\%$: **$0.94\%$** <br> sMAPE: **$2.16\%$** | Restricted MAPE isolates $<1\%$ error on active operating range; tail division isolated. |
| **SR Selection Outcome (Config E)** | Claimed active equation on all folds | **0 of 12 folds selected SR** ($g = 0$ selected) | Nested inner-LOCO cross-validation honestly rejected over-fitting symbolic corrections. |
| **Equation Node Count (Config E)** | 9 to 15 nodes claimed | **0 nodes** (Equation: `0 (no correction)`) | Reflects honest parsimony selection outcome where $g = 0$ is optimal. |

---

## 3. Detailed Root-Cause Analysis of the CALCE Baseline Shift
The quantitative shift observed in CALCE baselines between `results/archive_old/baselines_summary.csv` and the current LOCO benchmark (e.g., Random Forest from 3.39% to 2.31%, SVR from 3.87% to 1.96%, BiLSTM from 3.08% to 1.34%, Linear Regression from 4.61% to 2.82%, while NASA baselines were unchanged) is explained by two exact, verifiable data pipeline differences:

1. **Cohort Standardization (Exclusion of Non-Benchmark Cells `CS2_33` and `CS2_34`):**
   - In `results/archive_old/baselines.csv`, the training set for CALCE had $N_{\text{train}} = 4,731$ cycles (`archive_old/baselines.csv#L1`), because it included cells `CS2_33` (866 cycles) and `CS2_34` (776 cycles) in the training pool alongside the 4 evaluation cells.
   - In the fixed benchmark protocol, following the problem specification of a matched 4-cell LOCO design (4 NASA cells, 4 CALCE cells), CALCE is strictly filtered to `['CS2_35', 'CS2_36', 'CS2_37', 'CS2_38']` (`src/run_stage3.py#L49`).
   - Cross-cell domain shift from `CS2_33` and `CS2_34` (which operated under different cycling schedules) previously degraded the generalization of classical models trained on them.

2. **Outlier Filtering of Corrupted Measurement Cycles:**
   - In `data/processed/features_calce.csv`, raw automated feature extraction encountered corrupted IC peak curves on test cells `CS2_35` (124 outlier cycles) and `CS2_38` (133 outlier cycles) where voltage range was truncated and capacity derivative peaks could not be resolved.
   - The old baseline runner (`results/archive_old/baselines.csv`) evaluated on ALL 932 cycles of `CS2_35` and 1,078 cycles of `CS2_38` without outlier suppression (`n_test = 932` and `1078`).
   - The fixed benchmark filters corrupted cycles via `~df['is_outlier']` (`src/run_stage3.py#L55`), evaluating on clean test sets ($N_{\text{test}} = 808$ for `CS2_35`, $945$ for `CS2_38`).
   - Eliminating corrupted measurement cycles where target SOH or voltage features had non-physical sensor dropouts significantly reduced test RMSE across all models.

**Integrity Confirmation:** No held-out-cell information was used in preprocessing, training, calibration, or filtering. The outlier flag `is_outlier` was computed strictly from cycle-level electrical validity rules (monotonic voltage, completed CC charge interval) during raw feature extraction, independent of any model predictions or test cell labels.

