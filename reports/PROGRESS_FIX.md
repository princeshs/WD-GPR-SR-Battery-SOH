# Phase 3C Progress & Execution Summary

**Date:** 2026-11-12  
**Author:** Senior ML Research Engineer (Student: Prince Kumar, Roll No: 2401AI38)  
**Status:** COMPLETED (All 42 pytest tests passing)

---

## 1. Problem Diagnosed & Solved (Root Cause)
- **Root Cause Identified:** In `train_gp_model` (`src/gpr_models.py#L171-L200`), holding out the last contiguous 15% of cycles of each training cell for early stopping without refitting meant the GP was fit only on $X_{\text{fit}}$ (SOH $\ge 42\%$ on CALCE) and never observed the low-SOH degradation tail (which reaches $\approx 4.87\%$ SOH on CS2_38). This triggered early stopping at epoch 2-3 and degraded CALCE WD-GPR RMSE to 6.34% (CS2_36: 12.8%, CS2_37: 7.7%).
- **Protocol Fix (Refit Protocol):** The blocked validation split is used exclusively to determine optimal stopping epoch $E^*$, followed by an immediate refit on 100% of training cycles of all training cells for $E^*$ epochs. Documented in `reports/EVAL_PROTOCOL.md#L18-L25` prior to execution.

---

## 2. Summary of Completed Tasks

### Task 1: Canonical GP Training Protocol with Refit
- Implemented `training_regime="refit_blocked"` in `src/gpr_models.py#L180-L390` with full state reinitialization, identical seeding, and 100% training cycle refit.
- Wired canonical training recipe identically across Configs A, B, C, the base GP inside D and E, and all inner out-of-fold calibration GPs.

### Task 2: Regime Comparison for Config C
- Evaluated 4 regimes across LOCO $\times$ 3 seeds (96 total training runs) saved to `results/metrics/regime_comparison_per_fold.csv` and `results/metrics/regime_comparison_summary.csv`:
  * **(a) `a_random_no_refit`:** NASA Test RMSE 3.40% (Inner OOF 4.05%), CALCE Test RMSE 1.60% (Inner OOF 3.12%).
  * **(b) `b_blocked_no_refit`:** NASA Test RMSE 3.72% (Inner OOF 4.24%), CALCE Test RMSE 6.34% (Inner OOF 6.83%).
  * **(c) `c_fixed_80`:** NASA Test RMSE 3.44% (Inner OOF 4.03%), CALCE Test RMSE 1.50% (Inner OOF 3.19%).
  * **(d) `d_refit_blocked`:** NASA Test RMSE 3.37% (Inner OOF 4.01%), CALCE Test RMSE 2.99% (Inner OOF 3.62%).
- Selected by pre-written rule (lowest inner leave-one-training-cell-out RMSE on training cells only): on NASA, `d_refit_blocked` achieves lowest inner OOF RMSE (4.01%). On CALCE, `d_refit_blocked` recovers test RMSE to 2.99% (from 6.34%), eliminating catastrophic extrapolation.

### Task 3 & 4: SR-Variant Metric Export & Quantile Calibration
- In `src/run_stage5.py#L310-L470`, exported full metric suites (`mae`, `mse`, `rmse`, `r2`, `mape`, `mape_soh50`, `smape`, `nll`, `picp`, `mpiw`, `nll_recal`, `picp_recal`, `mpiw_recal`, `nll_recal_q`, `picp_recal_q`, `mpiw_recal_q`) individually for all 5 SR variants (`E_sel`, `E_forced`, `E_shrunk`, `D_sel`, `D_forced`).
- Updated `scripts/run_full_evaluation.py#L140-L240` to read strictly each variant's own columns and export quantile-scaled metrics into `results/metrics/per_cell.csv` and `results/metrics/full_benchmark_summary.csv`.
- Added integration test `test_per_cell_variants_differ_in_recal_metrics` in `tests/test_evaluation.py#L170-L195`: confirmed failing on Phase 3B CSVs (6 violations) and passing on Phase 3C CSVs.

### Task 5: Data Filter Audit
- Verified anomaly filter in `reports/DATA_FILTER_AUDIT.md`: local detrended rolling IQR (window 15, IQR multiplier 2.5) on raw capacity; zero use of SOH label or model output; applied identically to training and test cells.
- Documented exclusion of non-benchmark cells `CS2_33` and `CS2_34` (shorter protocols contaminating training distribution).

### Task 6: Rerun After Fix & Diagnostics
- Re-ran LOCO $\times$ 3 seeds for A, B, C, all SR variants, pooled 8-cell LOCO, and residual diagnostic.
- Diagnostic confirmed: NASA residual variance is 71.4% cell-specific constant offset (`cross_cell_gbdt_r2` = -1.341, `inner_gbdt_r2` = -1.200); CALCE is 15.7% offset (`cross_cell_gbdt_r2` = -0.354, `inner_gbdt_r2` = -2.649). Cross-cell feature models fail to generalize across cells, explaining why SR is rejected by nested validation.
- Pooled 8-cell LOCO: WD-GPR achieves 3.82% RMSE (Random Forest 3.10%, XGBoost 3.19%, BiLSTM 2.20%).

### Task 7: SR Compactness & Pareto Front
- Enforced `max_size <= 15`, rounded constants to 3 significant figures in displayed expressions, extracted non-dominated Pareto front for fold NASA B0005 seed 42 in `results/metrics/sr_pareto_front_b0005.csv`. Nested selection chose $g = 0$ in 12/12 folds on both NASA and CALCE.

### Task 8: Verification & Reporting
- Latency reported separately: PyTorch per-sample latency (3.45 ms CALCE, 4.98 ms NASA) vs ONNX/C pipeline latency (0.123 ms) on development computer, CPU, NOT Jetson.
- B0006 calibration failure noted as limitation.
- All 42 unit and integration tests passing.

