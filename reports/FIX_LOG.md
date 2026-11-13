# Fix Log: Methodological, Codebase, and Metric Auditing

## Phase 1: Audit Only
- **Status:** COMPLETED
- **Findings:** See `reports/AUDIT_FINDINGS.md` for full breakdown.

## Phase 2A: Code Fixes and Unit Testing
- **Status:** COMPLETED
- **Key Outcomes:** Single GP recipe in `src/gpr_models.py`, grouped weight decay, shared inner OOF calibration, nested SR selection with $g=0$ candidate, exact C equation fidelity ($\Delta = 0$), standalone SR standardization and pole suppression, unbiased MAPE calculations. 31/31 pytest unit tests pass.

## Phase 2B: Full LOCO Benchmark Rerun
- **Status:** COMPLETED
- **Key Outcomes:** 504 evaluation runs executed across NASA and CALCE across seeds {42, 123, 999}. Real numbers saved to `results/metrics/full_benchmark_summary.csv`, `results/metrics/ablation_table.csv`, and `results/metrics/per_cell.csv`. $g=0$ honestly selected across all 12 folds.

## Phase 3A: Ablation Breakdown, XAI Regeneration, Edge Deployment & Dynamic Report Verification
- **Status:** COMPLETED
- **Files Modified / Created:**
  - `results/metrics/paired_cell_differences.csv`: Paired per-cell differences ($E - C$, $E - D$, $E - \text{best baseline}$).
  - `results/metrics/ablation_breakdown.csv`: Disentangles effect of SR on mean ($\Delta \text{RMSE} = 0.0$ for E vs C, +2.30% CALCE / +0.81% NASA for D vs C) from effect of calibration on uncertainty ($\Delta \text{PICP} = +3.40\%$ on CALCE, raising coverage to 94.5%).
  - `src/explainability.py`: Re-ran XAI with ARD ranked by $1/\ell$, $4 \times 4$ active rank correlation matrix, disclosure of attribution vs causation, and PDP/ICE method pair.
  - `results/metrics/xai_feature_relevance.csv` & `results/metrics/xai_rank_correlations.csv`: Full feature relevance and Spearman correlation matrix (SHAP vs ARD $\rho=0.90$).
  - `deployment/benchmark.py` & `results/metrics/deployment_metrics.csv`: Process-level peak RSS (559.75 MB), model heap (0.59 MB), median latency (0.1227 ms), ONNX max diff ($5.82 \times 10^{-10}$), C exact ($0.000$). Explicitly labeled development computer, NOT Jetson.
  - `scripts/generate_guideline_report.py`: 100% dynamic metrics loading from CSVs; zero hand-typed metrics; prohibited phrases ("zero data leakage", "Rank 1") removed and replaced with compliant phrasing ("no held-out-cell information is used in preprocessing, training, calibration or SR selection").
  - `tests/test_report_metrics.py`: Automated test asserting every report metric matches CSVs (8/8 pass; 39/39 total suite tests pass).
  - `2401AI38_MLProject.pdf`: Regenerated, strictly 8 pages, zero tofu glyphs.
  - Poster generation stopped as instructed.

## Phase 3B: Bug Fixes, Fair SR Benchmark, Methodological Corrections & Complete Protocol Audit
- **Status:** COMPLETED (41/41 unit tests passing)
- **Detailed Executions & File:Line Evidence:**
  1. **B1 Timing & Latency Profiling:**
     - Replaced hardcoded fallback dict `profiling_stats` (`0.50` s, `0.050` ms) in `scripts/run_full_evaluation.py#L53-L185` with measured per-fold wall-clock training time (`time.perf_counter()`), median per-sample inference latency measured over $\ge 1,000$ timed predictions after 50-sample warm-up, and peak memory via `resource.getrusage(resource.RUSAGE_SELF).ru_maxrss`.
     - Integrated empirical timing into `src/run_stage3.py#L140-L165`, `src/run_stage4.py#L145-L175`, `src/run_stage5.py#L190-L245`. Labeled explicitly as `"development computer, CPU, NOT Jetson"`.
  2. **B2 Config D Recalibration Bug Fix & Unit Test:**
     - Fixed bug in `src/run_stage5.py#L220` where Config D was borrowing Config E/C's recalibrated uncertainty metrics. Config D now evaluates its own recalibrated metrics `m_after_unw = apply_shared_calibration(y_test, y_pred_unweighted, std_gp, s_factor=s_factor, s_quantile=s_quantile)` and saves distinct `nll_recal_unw`, `picp_recal_unw`, `mpiw_recal_unw` columns.
     - Wired Config D records in `scripts/run_full_evaluation.py#L140-L155` to read `row["nll_recal_unw"]`, `row["picp_recal_unw"]`, `row["mpiw_recal_unw"]`.
     - Added automated unit test `test_recalibrated_metrics_differ_when_means_differ` in `tests/test_evaluation.py#L149-L168` asserting that recalibrated NLL and PICP differ between models with different means.
  3. **B3 Honest "Proposed" Row Naming:**
     - Updated summary reporting so that where nested selection chooses $g = 0$ (NASA 12/12 folds), the model is reported honestly as `"WD-GPR (SR gate selected g=0 in 12/12 folds)"` (identical to Config C), while preserving a separate explicit row for active SR variants ($E_{\text{forced}}$ and $D_{\text{forced}}$).
  4. **B4 Updating `reports/CLAIMS_DIFF.md`:**
     - Corrected old NASA Config E ($4.93 \pm 1.48\%$ RMSE) and old NASA Config D ($4.31 \pm 1.86\%$ RMSE). Fixed cell CS2_38 dataset assignment to CALCE. Added diff rows for all 12 baselines sourced strictly from `results/archive_old/baselines_summary.csv` (`reports/CLAIMS_DIFF.md#L20-L75`).
  5. **Task 1: CALCE Baseline Shift & Inner Validation Tuning:**
     - Documented exact root cause with file:line citations (`reports/CLAIMS_DIFF.md#L65-L85`): exclusion of non-benchmark cells `CS2_33` and `CS2_34` from the training set, and filtering of 257 corrupted outlier cycles (`is_outlier == 0` in `src/run_stage3.py#L55`). Confirmed no held-out-cell data was used.
     - Implemented nested leave-one-training-cell-out cross-validation tuning for baselines with small search budgets (Ridge $\alpha \in \{0.01, 0.1, 1.0, 10.0\}$, SVR $C \in \{1, 10, 100\}$, RF $\text{max\_depth} \in \{6, 10, 14\}$, XGBoost) in `src/run_stage3.py#L40-L95`.
  6. **Task 2: Cell-Blocked Early-Stopping Split:**
     - Replaced random per-cycle permutation `perm = np.random.RandomState(seed).permutation(n_samples)` in `src/gpr_models.py#L170-L195` with a cell-blocked split: for each training cell, the last contiguous 15% of cycles are held out for early stopping validation.
     - Documented protocol rule in `reports/EVAL_PROTOCOL.md#L18`. Re-ran Variants A, B, C across all 24 folds.
  7. **Task 3: NASA Calibration & Quantile Scaling:**
     - Formalized dual calibration rule in `reports/EVAL_PROTOCOL.md#L25-L35`: variance scaling $s_{\text{var}}$ vs quantile/conformal scaling $s_{\text{quantile}} = \frac{Q_{0.95}(|r_{\text{oof}}| / \sigma_{\text{oof}})}{1.95996}$.
     - Implemented `compute_quantile_calibration_factor` in `src/evaluation.py#L175-L195` and updated `apply_shared_calibration` (`src/evaluation.py#L198-L245`). Reported raw, variance-scaled, and quantile-scaled metrics side-by-side for all probabilistic models.
  8. **Task 4: Fair SR Benchmark (5 Rules):**
     - Evaluated $E_{\text{sel}}$, $E_{\text{forced}}$, $D_{\text{forced}}$, $D_{\text{sel}}$, and $E_{\text{shrunk}}$ ($\alpha \in \{0, 0.25, 0.5, 1\}$) across 24 LOCO folds under fixed budget (pop=1000, gen=30, restricted operators `add, sub, mul, div`).
     - Saved results to `results/metrics/sr_fair_test.csv` and documented fold 0 discovered equations (nodes, operators) in `results/equations/discovered_equations_fair_sr.txt`.
  9. **Task 5: Residual Variance Diagnostic:**
     - Implemented `scripts/run_residual_diagnostic.py` and saved `results/metrics/residual_variance_diagnostic.csv`.
     - Findings: on NASA, 73.9% of residual variance is cell-specific constant offset ($R^2_{\text{cross-cell}} = -1.459$), proving why SR overfits out-of-fold. On CALCE, offset share is 11.9% and inner cross-cell $R^2 = -0.019$.
  10. **Task 6: Pooled 8-Cell LOCO Cross-Validation:**
      - Evaluated all 8 benchmark cells (4 NASA + 4 CALCE) under LOCO-CV for WD-GPR (5.24% RMSE), BiLSTM (2.20% RMSE), Random Forest (3.10% RMSE), and XGBoost (3.19% RMSE) in `scripts/run_pooled_8cell.py`. Saved to `results/metrics/pooled_8cell_summary.csv`.
  11. **Task 7: Honest SOTA Disclosures:**
      - Confirmed that DeepNN (1.64% RMSE) and BiLSTM (1.34% RMSE) beat WD-GPR (1.64% / 6.34% under cell-blocked split) on CALCE ("within noise"). Removed all claims of beating all non-sequential baselines.

## Phase 3C: Refit Protocol, Full SR Variant Metrics, Quantile Calibration & Audit Verification
- **Status:** COMPLETED (All 42/42 unit and integration tests passing)
- **Detailed Executions & File:Line Evidence:**
  1. **Task 1: Protocol Fix in GP Training (Validation for Stopping Epoch + 100% Data Refit):**
     - Stated protocol in `reports/EVAL_PROTOCOL.md#L18-L25` before execution.
     - In `src/gpr_models.py#L180-L390`, implemented `training_regime="refit_blocked"` (default): step 1 uses blocked validation (last 15% of cycles of each training cell) to determine optimal stopping epoch $E^*$; step 2 refits model on 100% of training cycles of all training cells for exactly $E^*$ epochs. Added tracking attributes `stopping_epoch`, `y_train_min`, `y_fit_min`, `y_val_min`.
     - Wired canonical function `train_gp_model` (`src/gpr_models.py#L476-L516`) across Configs A, B, C (`src/run_stage4.py#L126,L155`), base GP in D and E (`src/run_stage5.py#L126,L292`), and inner calibration folds.
  2. **Task 2: Regime Comparison for Config C:**
     - Created and executed `scripts/compare_gp_regimes.py#L1-L195`.
     - Evaluated regimes across NASA and CALCE (LOCO $\times$ 3 seeds, 96 total runs). Saved to `results/metrics/regime_comparison_per_fold.csv` and `results/metrics/regime_comparison_summary.csv`.
     - Selected by pre-written rule (lowest inner leave-one-training-cell-out RMSE on training cells only): on NASA, `d_refit_blocked` achieves lowest inner OOF RMSE (4.01%). On CALCE, `d_refit_blocked` achieves 3.62% inner OOF RMSE and 2.99% test RMSE (down from 6.34% under `b_blocked_no_refit`), resolving the end-of-life tail truncation issue.
  3. **Task 3: Fix SR-Variant Metric Export & Integration Test:**
     - In `src/run_stage5.py#L310-L470`, exported full metrics (`mae`, `mse`, `rmse`, `r2`, `mape`, `mape_soh50`, `smape`, `nll`, `picp`, `mpiw`, `nll_recal`, `picp_recal`, `mpiw_recal`, `nll_recal_q`, `picp_recal_q`, `mpiw_recal_q`) individually for all 5 SR variants (`E_sel`, `E_forced`, `E_shrunk`, `D_sel`, `D_forced`).
     - In `scripts/run_full_evaluation.py#L140-L240`, updated parser to read strictly each variant's own columns and never fall back.
     - Added integration test `test_per_cell_variants_differ_in_recal_metrics` in `tests/test_evaluation.py#L170-L195`. Verified it failed on Phase 3B CSVs (6 export violations) and passes on Phase 3C CSVs.
  4. **Task 4: Export Quantile-Scaled Calibration Metrics:**
     - Exported `NLL_recal_q`, `PICP_recal_q`, `MPIW_recal_q`, `s_quantile` into `results/metrics/per_cell.csv` and `results/metrics/full_benchmark_summary.csv` for all probabilistic models next to raw and variance-scaled values.
  5. **Task 5: Data Filter Audit:**
     - Created and ran `scripts/audit_data_filter.py#L1-L160`. Generated `results/metrics/data_filter_audit_counts.csv`, `results/metrics/data_filter_impact_comparison.csv`, and report `reports/DATA_FILTER_AUDIT.md`.
     - Confirmed anomaly rule is local detrended rolling IQR (window 15, IQR 2.5) on raw capacity; zero use of SOH label or model output; applied identically to all cells before partitioning. Documented impact of excluding non-benchmark cells `CS2_33` and `CS2_34`.
  6. **Task 6: Full Benchmark Rerun & Residual Diagnostic:**
     - Re-ran Stage 4 (`src/run_stage4.py`), Stage 5 (`src/run_stage5.py`), pooled 8-cell LOCO (`scripts/run_pooled_8cell.py`), and residual diagnostic (`scripts/run_residual_diagnostic.py`).
     - Residual diagnostic saved with exact columns `cross_cell_gbdt_r2` and `inner_gbdt_r2` (`results/metrics/residual_variance_diagnostic.csv`). Findings: NASA is 71.4% constant offset (`cross_cell_gbdt_r2` = -1.341, `inner_gbdt_r2` = -1.200); CALCE is 15.7% constant offset (`cross_cell_gbdt_r2` = -0.354, `inner_gbdt_r2` = -2.649).
     - Pooled 8-cell LOCO: WD-GPR achieves 3.82% RMSE (Random Forest 3.10%, XGBoost 3.19%, BiLSTM 2.20%).
  7. **Task 7: SR Compactness & Pareto Front:**
     - Enforced `max_size <= 15`, rounded constants to 3 sig figs in displayed equations (`src/symbolic_regression.py#L120-L160`).
     - Extracted non-dominated Pareto front for fold NASA B0005 seed 42 in `results/metrics/sr_pareto_front_b0005.csv` (12 non-dominated points from complexity 1 to 31; complexity $\le 15$ RMSE = 3.93%).
     - Nested selection chose $g = 0$ in 12/12 folds on both NASA and CALCE.
  8. **Task 8: Reporting Rules Compliance:**
     - Stated "no difference beyond seed and cell variation" when gap < std across cells.
     - State B0006 calibration failure as a limitation.
     - Quoted PyTorch per-sample latency and ONNX/C pipeline latency as separate rows labeled "development computer, CPU, NOT Jetson".
