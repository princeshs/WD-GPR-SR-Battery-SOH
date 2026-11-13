# Progress Log: Strict Codebase & Protocol Fixes

## Current Status: Phase 3B (Bug Fixes, Fair SR Benchmark, Methodological Corrections & Complete Protocol Audit) Complete
- **Date:** 2026-11-11
- **Author / Reviewer:** Senior ML Research Engineer (Student: Prince Kumar, Roll No: 2401AI38)
- **Active Workspace:** `/home/prince/Desktop/ML_Project`

---

## Phase Execution Summary

### Phase 1: Audit Only
- **Status:** COMPLETED
- **Files Created:** `PROGRESS_FIX.md`, `reports/FIX_LOG.md`, `reports/AUDIT_FINDINGS.md`

### Phase 2A: Code Fixes & Unit Testing
- **Status:** COMPLETED (31/31 unit tests passing)
- **Key Outcomes:** Single GP recipe `train_gp_model()`, parameter-grouped weight decay, shared calibration from inner OOF residuals, nested SR selection with g=0 candidate, exact equation fidelity (max |Δ| = 0.0 on C, <= 3.55e-15 on SymPy), StandaloneSR standardization and clipping, unbiased MAPE, and trivial reference baselines.

### Phase 2B: Full LOCO Benchmark Rerun
- **Status:** COMPLETED
- **Executions Completed:** 504 total records compiled into `results/metrics/full_results_long.csv`, `results/metrics/per_cell.csv`, `results/metrics/ablation_table.csv`, and `results/metrics/full_benchmark_summary.csv`.
- **Files Created:** `reports/EVAL_PROTOCOL.md`, `reports/CLAIMS_DIFF.md`.

### Phase 3A: Redo Ablation Breakdown, XAI Regeneration, Edge Deployment & Dynamic Report Verification
- **Status:** COMPLETED (39/39 pytest unit tests passing)
- **Key Executions:** Ablation breakdown and paired differences, ARD ranked by 1/ℓ, edge deployment with process RSS, report dynamically loading from CSVs, prohibited phrasing removed.

### Phase 3B: Bug Fixes, Fair SR Benchmark, Calibration & Protocol Realignment
- **Status:** COMPLETED (41/41 pytest unit tests passing)
- **Key Executions:**
  1. **B1 Empirical Timings & Profiling:** Eliminated static fallback profiling dictionary. All training times measured via `time.perf_counter()` wall-clock per fold, per-sample inference latency measured over $\ge 1,000$ timed single-sample predictions after warm-up, peak memory via `resource.getrusage`. All labeled "development computer, CPU, NOT Jetson".
  2. **B2 Config D Recalibrated Metrics Bug Fix:** Config D unweighted SR evaluates its own recalibrated metrics with its corrected mean $\hat{y}_D = \mu_{\text{GP}} + g(x)$ and saves distinct `nll_recal_unw`, `picp_recal_unw`, `mpiw_recal_unw`. Automated unit test added in `tests/test_evaluation.py` (passes).
  3. **B3 Honest Proposed Row Naming:** Where nested selection chooses $g = 0$ (NASA 12/12 folds), reported as `"WD-GPR (SR gate selected g=0 in 12/12 folds)"` (Config C), preserving separate explicit rows for active SR variants ($E_{\text{forced}}$, $D_{\text{forced}}$).
  4. **B4 Updating `reports/CLAIMS_DIFF.md`:** Corrected old NASA Config E (4.93%) and Config D (4.31%). Corrected CS2_38 to CALCE. Added diff rows for all 12 baselines sourced strictly from `results/archive_old/baselines_summary.csv`.
  5. **Task 1: CALCE Baseline Shift & Inner Validation Tuning:** Documented root cause of CALCE baseline shift (exclusion of `CS2_33`/`34` and filtering of 257 corrupted outlier cycles). Implemented nested inner leave-one-training-cell-out validation tuning for baselines with small search budgets.
  6. **Task 2: Cell-Blocked Early-Stopping Split:** Replaced random per-cycle permutation with holding out the last contiguous 15% of cycles of each training cell in `src/gpr_models.py`. Documented in `reports/EVAL_PROTOCOL.md`.
  7. **Task 3: Dual Calibration Protocol:** Formalized quantile scaling $s_{\text{quantile}} = \frac{Q_{0.95}(|r_{\text{oof}}| / \sigma_{\text{oof}})}{1.95996}$ side-by-side with variance scaling.
  8. **Task 4: Fair SR Benchmark (5 Rules):** Evaluated $E_{\text{sel}}$, $E_{\text{forced}}$, $D_{\text{forced}}$, $D_{\text{sel}}$, and $E_{\text{shrunk}}$ under fixed budget (pop=1000, gen=30, restricted ops `add, sub, mul, div`). Saved to `results/metrics/sr_fair_test.csv` and documented fold 0 discovered equations.
  9. **Task 5: Residual Variance Diagnostic:** Evaluated in `scripts/run_residual_diagnostic.py`: 73.9% of NASA residual variance is cell-specific constant offset ($R^2_{\text{cross-cell}} = -1.459$), explaining why SR overfits out-of-fold.
  10. **Task 6: Pooled 8-Cell LOCO:** Evaluated all 8 benchmark cells under LOCO-CV for WD-GPR (5.24% RMSE), BiLSTM (2.20%), Random Forest (3.10%), and XGBoost (3.19%).
  11. **Task 7: SOTA Claims Realignment:** Honest disclosure that BiLSTM (1.34%) and DeepNN (1.64%) outperform WD-GPR on CALCE ("within noise"). Removed claims of beating all non-sequential baselines.

---

## Key Verified Benchmark Numbers (Phase 3B)
- **CALCE Base GP (WD-GPR C):** RMSE 6.34 ± 5.00% (seed std 0.21%), MAE 3.29%, 95% PICP 94.8% (recalibrated), MPIW 26.65%.
- **CALCE SR Active (E_forced):** RMSE 3.08 ± 0.72% (seed std 0.17%), MAE 1.65%, 95% PICP 98.7%, MPIW 26.65%, nodes 20.3.
- **CALCE Proposed (E_sel, 9/12 folds SR active):** RMSE 4.18 ± 2.49% (seed std 0.11%), MAE 2.16%, nodes 11.4.
- **NASA Base GP (WD-GPR C):** RMSE 3.72 ± 2.84% (seed std 0.19%), MAE 3.31%, 95% PICP 75.0% (recalibrated), MPIW 9.51%.
- **NASA SR Active (E_forced):** RMSE 4.62 ± 1.30% (seed std 0.14%), MAE 3.88% (SR hurts base GP).
- **NASA Proposed (E_sel, 0/12 folds SR active):** RMSE 3.72 ± 2.84% (seed std 0.19%), nodes 0 ($g=0$ honestly selected across all 12 folds).
- **Hardware Profiling:** Wall-clock training and inference latency measured over $\ge 1,000$ predictions on development computer CPU, NOT Jetson.

---

## Key Verified Benchmark Numbers (Phase 3A)
- **CALCE Proposed (WD-GPR-SR):** RMSE 1.64 ± 0.72% (seed std 0.15%), MAE 0.97 ± 0.48%, 95% PICP 94.5% (recalibrated, s=1.34), MPIW 6.50%, rMAPE 0.94%, equation nodes 0 (g=0 selected across all 12 folds).
- **NASA Proposed (WD-GPR-SR):** RMSE 3.33 ± 2.51% (seed std 0.23%), MAE 2.94 ± 2.54%, 95% PICP 74.4% (recalibrated, s=0.74), MPIW 8.84%, equation nodes 0 (g=0 selected across all 12 folds).
- **XAI Consistency:** Spearman ρ(SHAP, ARD) = 0.90; top drivers: `cc_charge_time` and `cycle`.
- **Edge Deployment (Dev PC):** Median latency 0.1227 ms, model heap 0.59 MB, process RSS 559.75 MB, ONNX parity 5.82e-10, C parity 0.000.

---

## Next Steps
- Await user prompt for Phase 4 (Notebook synchronization, README update). Do not package or build poster.
