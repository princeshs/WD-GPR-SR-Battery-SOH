# Comprehensive Technical Audit Findings: WD-GPR-SR

**Project:** WD-GPR-SR (Battery State of Health Prognostics)  
**Author / Student:** Prince Kumar | Roll No: 2401AI38  
**Department:** Computer Science & Engineering, IIT Patna (IMPACT Lab)  
**Audit Conducted By:** Senior ML Research Engineer & Strict Reviewer  
**Audit Date:** 2026-11-10  
**Status:** Phase 1 Complete (Audit Only — Zero Code Fixes Applied)

---

## Executive Summary of Audit Findings

The audit investigated 15 specific architectural, methodological, numerical, and documentation areas across `src/`, `scripts/`, `deployment/`, `2401AI38_MLProject.ipynb`, and `results/metrics/`. All 15 suspected issues were rigorously verified using targeted inspection scripts and empirical numerical tests. 

Key takeaways:
1. **Zero Test Data Leakage in Calibration:** Recalibration factor $s$ is computed strictly from inner out-of-fold residuals of training cells (no test leakage). However, the notebook omitted $s$ completely, and Config E is the *only* configuration evaluated with recalibration in the master evaluation, creating an unfair, confounded ablation comparison (M1).
2. **Artificial Superiority of Proposed Model:** The claimed CALCE RMSE drop from 3.30% (Config C) to 2.53% (Config E) is misleading. In Stage 5, the base GP alone already achieved **1.73% RMSE**. Adding Symbolic Regression actually **worsened** error from **1.73% to 2.53%** (+0.79% degradation). The 3.30% figure was an artifact of comparing an 85%-trained GP with early stopping (Stage 4) against a 100%-trained GP without early stopping (Stage 5) (L3, L4).
3. **Severe Equation Inconsistencies:** The exported SymPy parsing logic does not implement `gplearn`'s protected operators, producing numerical discrepancies of up to **13.85** on near-zero inputs and possessing an unsafe fallback `Symbol(prog_str)` (C1).
4. **Optimizer Degradation:** `Adam(weight_decay=1e-4)` was applied globally to all model parameters, improperly decaying spatial inducing points $Z$, unconstrained lengthscales, and variational parameters towards zero (C4).
5. **Multiple Inconsistent Claims:** Latency values conflict across documents (0.08 ms vs 0.098 ms vs 0.123 ms vs 0.245 ms live); `requirements.txt` is unpinned (`>=`); and memory was measured strictly on the Python heap (`tracemalloc`) rather than full process RSS (R2).

---

## Detailed Audit Findings per Item

### L1. Recalibration Factor $s$ Computation & Data Isolation
- **Status:** **CONFIRMED (Isolated on inner training OOF; omitted in notebook)**
- **Evidence:**
  In `src/run_stage5.py`, lines 194–196:
  ```python
  # Optional: variance recalibration factor fitted on OOF
  s_factor = recalibrate_variance(r_oof, var_oof)
  std_gp_recal = std_gp * s_factor
  ```
  Where `r_oof` and `var_oof` are built across lines 88–121:
  ```python
  for inner_val_cell in train_cells:
      inner_train_cells = [c for c in train_cells if c != inner_val_cell]
      ...
      oof_res = y_in_val - mu_in
      oof_residuals_list.append(oof_res)
      oof_variances_list.append(var_in)
  ```
  Formula in `src/symbolic_regression.py`, lines 165–176:
  ```python
  def recalibrate_variance(residuals_oof: np.ndarray, variances_oof: np.ndarray) -> float:
      ratio = (residuals_oof ** 2) / np.maximum(variances_oof, 1e-6)
      s2 = float(np.mean(np.clip(ratio, 0.05, 20.0)))
      return float(np.sqrt(s2))
  ```
- **Finding:** No held-out test cell data is used to compute $s$. It is computed strictly from inner out-of-fold residuals across the 3 training cells. However, in `scripts/build_complete_notebook.py` (Cell 8, lines 618–620), recalibration $s$ is omitted entirely:
  ```python
  ci_lower = np.maximum(0.0, y_final - 1.96 * std_gp)
  ci_upper = np.minimum(100.0, y_final + 1.96 * std_gp)
  ```
  The notebook reports unscaled GP variance while the CSV results report variance scaled by $s$.

---

### L2. Hyperparameter Selection & Leakage Audit
- **Status:** **CONFIRMED (No search loop; hardcoded ad-hoc)**
- **Evidence:**
  - `configs/config.yaml`: `n_inducing: 64, epochs: 100, lr: 0.01, population_size: 500, generations: 20, parsimony_coefficient: 0.005`.
  - `src/run_stage4.py:103-114`: `num_inducing=32, epochs=80, lr=0.015`.
  - `src/run_stage5.py:100-108,174-182`: Inner GP `num_inducing=24, epochs=70`; Final GP `num_inducing=32, epochs=80`; SR `population_size=400, generations=15, parsimony_coefficient=0.005`.
  - `scripts/build_complete_notebook.py:460-468,597,608`: Inner GP `epochs=25, M=64`; Final GP `epochs=40, M=64`; SR `population_size=150, generations=10, parsimony_coefficient=0.001`.
- **Finding:** There is no automated hyperparameter tuning loop (such as `GridSearchCV` or validation-based selection). Hyperparameters were chosen manually by trial and error. While test cell features are not directly fed into an automated tuning loop, manual tuning by past developers inspecting held-out test results violates strict blind evaluation. Phase 2 must add a formal inner LOCO validation selection loop.

---

### L3. Base GP Equivalence: Config C vs Config E
- **Status:** **CONFIRMED (Not identical; large discrepancy)**
- **Evidence:**
  - In `src/run_stage4.py` (lines 92–97), Config C splits training data into 85% train and 15% validation:
    ```python
    val_size = int(0.15 * n_tr)
    perm = np.random.RandomState(seed).permutation(n_tr)
    idx_tr, idx_val = perm[val_size:], perm[:val_size]
    model.fit(X_tr_inner, y_tr_inner, X_val=X_val_inner, y_val=y_val_inner, ...)
    ```
  - In `src/run_stage5.py` (line 183), Config E's final GP trains on 100% of the training cells with `X_val=None`:
    ```python
    final_gp.fit(X_train_full, y_train_full, seed=seed)
    ```
  - Additionally, in Stage 5, before `final_gp.fit` is called, three inner GPs and two SR models are fitted, advancing PyTorch and NumPy RNG states.
- **Empirical Measurement:**
  Testing on held-out cell `CS2_35`, seed 42:
  - Stage 4 Config C RMSE: **2.1718%**
  - Stage 5 100%-train base GP RMSE: **2.3137%**
  - $\max |\mu_{\text{GP, Stage 4}} - \mu_{\text{GP, Stage 5}}| =$ **4.9863% SOH**
  - $\text{mean} |\mu_{\text{GP, Stage 4}} - \mu_{\text{GP, Stage 5}}| =$ **0.5950% SOH**
  The base GP mean is NOT identical between Config C and Config E.

---

### L4. Decomposition of the Claimed CALCE RMSE Drop (3.30% to 2.53%)
- **Status:** **CONFIRMED (SR degrades base GP; apparent gain was an artifact)**
- **Evidence:**
  Analysis of `results/metrics/sr_results.csv` on CALCE held-out cells:
  ```text
  Held-Out Cell  Seed   RMSE (Base GP)   RMSE (After Weighted SR)   Delta RMSE (Degradation)
  CS2_35          42        0.9782                1.5173                     +0.5391
  CS2_35         123        1.0093                1.0479                     +0.0386
  CS2_35         999        0.8318                1.6191                     +0.7873
  CS2_36          42        2.9407                1.6921                     -1.2486  (Improved)
  CS2_36         123        2.9292                2.9807                     +0.0515
  CS2_36         999        2.8399                2.8729                     +0.0330
  CS2_37          42        1.6832                4.2302                     +2.5470
  CS2_37         123        1.7317                4.3784                     +2.6467
  CS2_37         999        1.8032                5.8976                     +4.0944
  CS2_38          42        1.2920                1.2946                     +0.0026
  CS2_38         123        1.4793                1.4729                     -0.0065
  CS2_38         999        1.2991                1.3170                     +0.0179
  -----------------------------------------------------------------------------------------
  CALCE Average:            1.7348%               2.5267%                    +0.7919%
  ```
  On NASA:
  - Mean Base GP RMSE: **3.3590%**
  - Mean After Weighted SR: **4.9283%**
  - Average Degradation: **+1.5693% RMSE**
- **Finding:** In Stage 5, the base GP alone reached **1.73% RMSE** on CALCE. Symbolic Regression actually **worsened** error from **1.73% to 2.53%** (and to 3.65% under unweighted SR). The claimed drop from 3.30% to 2.53% in `README.md` and report was an artifact of comparing the 85%-trained GP in Stage 4 (3.30%) with the 100%-trained GP + SR in Stage 5 (2.53%).

---

### M1. Confounded Calibration in Ablation Table
- **Status:** **CONFIRMED (Confounded uncertainty comparison)**
- **Evidence:**
  In `scripts/run_full_evaluation.py`:
  - Lines 101–103 (Configs A, B, C):
    ```python
    "nll": float(row["nll"]), "picp": float(row["picp"]), "mpiw": float(row["mpiw"])
    ```
    Raw, uncalibrated GP variance from Stage 4 (e.g., Config C: PICP = 99.5%, MPIW = 17.25%).
  - Lines 132–134 (Config D: Unweighted SR):
    ```python
    "nll": float(row["nll_before"]), "picp": float(row["picp_before"]), "mpiw": float(row["mpiw_before"])
    ```
    Raw, uncalibrated GP variance (PICP = 99.5%, MPIW = 17.25%).
  - Lines 157–159 (Config E: Proposed WD-GPR-SR):
    ```python
    "nll": float(row["nll_recal"]), "picp": float(row["picp_recal"]), "mpiw": float(row["mpiw_recal"])
    ```
    Recalibrated variance using $s \approx 0.4$ on CALCE, shrinking MPIW to 6.68% and PICP to 90.6%.
- **Finding:** Config E was the *only* model given variance recalibration. The reduction in MPIW (from 17.2% to 6.68%) was caused entirely by applying $s$, not by the architectural contribution of Symbolic Regression.

---

### M2. Standalone SR Baseline Failure (RMSE 55, R² -9)
- **Status:** **CONFIRMED (Unstandardized targets, division pole explosion)**
- **Evidence:**
  In `src/baseline_models.py`, lines 586–609:
  ```python
  self.model = SymbolicRegressor(
      population_size=self.population_size,
      generations=self.generations,
      ...
      function_set=("add", "sub", "mul", "div", "sqrt", "log", "abs"),
      metric="mean absolute error",
      random_state=seed,
  )
  self.model.fit(X_tr, y_tr)
  ```
- **Finding:** `StandaloneSR` fits on raw SOH values (60–100%) without target standardization (`StandardScaler`) and without output clipping. On held-out test cells, unconstrained division terms explode, yielding massive errors on CALCE (CS2_38 seed 123: RMSE = **134.49%**, R² = **-44.7**; average RMSE = **55.51%**). Meanwhile, `SymbolicResidualRegressor.predict` in `src/symbolic_regression.py:125` hard-clips predictions to `[-30, 30]`. If applied directly to raw SOH, clipping to 30 creates a guaranteed floor error of $\ge 50\%$.

---

### M3. MAPE Function Behavior and True SOH Ranges
- **Status:** **CONFIRMED (Asymmetric tail inflation; hardcoded MAPE in master runner)**
- **Evidence:**
  1. True SOH Ranges Across Datasets:
     - **NASA B0005:** Min = 69.73%, Max = 100.55%, Mean = 85.17%
     - **NASA B0006:** Min = **56.97%**, Max = 100.50%, Mean = 76.38% (Falls well below 70%)
     - **NASA B0007:** Min = 74.47%, Max = 100.55%, Mean = 87.44%
     - **NASA B0018:** Min = 72.76%, Max = 100.64%, Mean = 84.51%
     - **CALCE CS2_35:** Min = 21.52%, Max = 100.88%, Mean = 74.90%
     - **CALCE CS2_36:** Min = 8.08%, Max = 100.91%, Mean = 71.66%
     - **CALCE CS2_37:** Min = 4.89%, Max = 100.90%, Mean = 73.28%
     - **CALCE CS2_38:** Min = **4.87%**, Max = 100.93%, Mean = 73.91%
  2. The claim that SOH remains in the range "70% to 100%" is false. CALCE cells degrade down to ~5% SOH.
  3. In `src/evaluation.py:92`:
     ```python
     mape = float(np.mean(np.abs((y_true[nonzero_mask] - y_pred[nonzero_mask]) / y_true[nonzero_mask])) * 100.0)
     ```
     When $y_{\text{true}} \approx 5\%$, an error of $1.5\%$ yields a percentage error of $30\%$, heavily inflating MAPE relative to MAE.
  4. In `scripts/run_full_evaluation.py` (lines 131 and 156), MAPE was literally hardcoded:
     ```python
     "mape": float(row["mae_after_unweighted"] * 1.15)  # Config D
     "mape": float(row["mae_after_weighted"] * 1.12)    # Config E
     ```
     This fabrication bypassed true MAPE evaluation for the proposed model while baselines suffered tail division inflation.

---

### M4. ARD Relevance Table Ordering Inconsistency
- **Status:** **CONFIRMED (Misaligned table ordering)**
- **Evidence:**
  Values in `results/metrics/xai_feature_relevance.csv`:
  1. `cc_charge_time`: ARD relevance ($1/\ell$) = **0.3125** (Rank 1)
  2. `cycle`: ARD relevance ($1/\ell$) = **0.2111** (Rank 2)
  3. `ic_peak_voltage`: ARD relevance ($1/\ell$) = **0.1656** (Rank 3)
  4. `cv_charge_time`: ARD relevance ($1/\ell$) = **0.1602** (Rank 4)
  5. `ic_peak_height`: ARD relevance ($1/\ell$) = **0.1506** (Rank 5)
  In `scripts/generate_guideline_report.py`, line 737:
  ```python
  ["ARD Lengthscales (1/ℓ)", "cycle (0.211)", "cc_charge_time (0.313)", "ic_peak_voltage (0.166)"]
  ```
- **Finding:** The table lists `cycle (0.211)` before `cc_charge_time (0.313)`, violating descending numerical order.

---

### M5. Discrepancies Between Documentation and XAI CSVs
- **Status:** **CONFIRMED (Text mismatches identified)**
- **Evidence:**
  1. `README.md`, line 164:
     > "Symbolic sensitivities specifically target non-linear electrochemical polarization in `cv_charge_time` and `ic_peak_voltage` ($33.8\%$ each)."
     In `results/metrics/xai_feature_relevance.csv`:
     - `cv_charge_time`: **55.85%** (0.5585)
     - `ic_peak_voltage`: **21.91%** (0.2191)
  2. `README.md`, line 162:
     > "`cc_charge_time` and `cycle` are the dominant global degradation drivers across both SHAP and ARD kernels (Spearman $\rho = 0.50$)."
     In SHAP, `cycle` is only **3.19%** (Rank 4), while `ic_peak_voltage` is dominant at **54.66%** (Rank 1).
  3. Correlation between ARD and Symbolic sensitivities in `results/metrics/xai_consistency.csv` is **0.00** (orthogonal), not correlated.

---

### C1. Equation Fidelity: gplearn vs SymPy vs C Implementation
- **Status:** **CONFIRMED (Max abs diff 13.85, unsafe Symbol fallback)**
- **Evidence:**
  In `src/symbolic_regression.py`, lines 134–151:
  ```python
  "div": lambda a, b: a / b if b != 0 else 1,
  "log": lambda a: sp.log(sp.Abs(a) + 1e-6),
  ...
  except Exception:
      return sp.Symbol(prog_str)
  ```
  In `gplearn.functions`:
  - `div(x1, x2)` returns `1.0` if $|x2| \le 0.001$.
  - `log(x1)` returns `0.0` if $|x1| \le 0.001$.
- **Empirical Numerical Test:**
  Evaluating on inputs with features near zero ($|x| = 0.0002$):
  - `gplearn.predict`: **7.6918**
  - `sp.lambdify(sympy_expr)`: **21.5419**
  - Max absolute difference: **13.8501**
  Furthermore, the fallback `return sp.Symbol(prog_str)` silently turns an entire unparseable program tree into an atomic undefined Symbol, causing runtime failures when evaluated downstream.

---

### C2. SR Weights Quantile Clipping Missing in Notebook
- **Status:** **CONFIRMED (Notebook omits quantile clipping)**
- **Evidence:**
  In `src/symbolic_regression.py` (lines 29–31):
  ```python
  q_low = float(np.quantile(w_raw, clip_quantiles[0]))
  q_high = float(np.quantile(w_raw, clip_quantiles[1]))
  w_clip = np.clip(w_raw, q_low, q_high)
  ```
  In `scripts/build_complete_notebook.py` (Cell 8, lines 604–605):
  ```python
  sample_weights = 1.0 / (oof_std**2 + 1e-6)
  sample_weights = sample_weights / np.mean(sample_weights)
  ```
- **Finding:** The notebook directly standardizes raw inverse variances without clipping extreme quantiles $[P_1, P_{99}]$, leaving the genetic program vulnerable to outlier residual weights.

---

### C3. Epoch & Inducing Point Discrepancies: Inner vs Final GPs
- **Status:** **CONFIRMED (Mismatched training regimes)**
- **Evidence:**
  - Notebook (`scripts/build_complete_notebook.py`):
    - Final GP (Cell 6, line 466): `num_inducing=64, epochs=40` (trained on 3 cells)
    - Inner GPs (Cell 8, line 597): `num_inducing=64, epochs=25` (trained on 2 cells)
  - Full Evaluation (`src/run_stage5.py`):
    - Inner GPs (lines 102–103): `num_inducing=24, epochs=70`
    - Final GP (lines 176–177): `num_inducing=32, epochs=80`
- **Finding:** Inner out-of-fold residuals are extracted from GPs with different capacities and convergence states than the primary predictor.

---

### C4. Unconstrained Weight Decay on Non-Neural Parameters
- **Status:** **CONFIRMED (Inducing points, lengthscales, and variational params decayed)**
- **Evidence:**
  In `src/gpr_models.py`, lines 185–188:
  ```python
  optimizer = torch.optim.Adam([
      {"params": self.model.parameters()},
      {"params": self.likelihood.parameters()},
  ], lr=self.lr, weight_decay=self.weight_decay)
  ```
  And in `scripts/build_complete_notebook.py`, line 433:
  ```python
  opt = torch.optim.Adam([
      {'params': self.model.parameters()},
      {'params': self.likelihood.parameters()}
  ], lr=self.lr, weight_decay=1e-4)
  ```
- **Finding:** `weight_decay=1e-4` is applied indiscriminately to all parameters, including:
  1. Inducing points $Z$ (forcing spatial inducing coordinates towards the origin in $\mathbb{R}^D$),
  2. Raw ARD and Matérn lengthscales (forcing lengthscales to shrink towards zero),
  3. Variational distribution parameters $\mathbf{m}$ and $\mathbf{S}$,
  4. Likelihood noise variance $\sigma_n^2$.
  Weight decay must be applied exclusively to the neural network weights $\phi_{\boldsymbol{\theta}}$.

---

### R1. Notebook Data Ingestion & Live Computation vs CSV Reading
- **Status:** **CONFIRMED (Starts from processed data; tables read from CSV)**
- **Evidence:**
  - Data Loading (Cell 3, lines 148–156): Reads `data/processed/features_nasa.csv` and `data/processed/features_calce.csv` rather than raw aging matrices.
  - Live Computations:
    - Cell 4: Live Pearson correlations and EDA visualizations.
    - Cell 7: Live training of `wd_svgp` on fold `B0005`.
    - Cell 8: Live inner LOCO OOF residuals, SR fitting, and prediction on `B0005`.
    - Cell 12: Live 50-warmup + 1,000-iteration inference benchmark on host CPU.
    - Cell 13: Live document compilation and PyMuPDF page count assertions.
  - CSV-Read Tables:
    - Cell 5: Reads `results/metrics/baselines_summary.csv` and `baselines.csv`.
    - Cell 9: Reads `results/metrics/summary_mean_std_over_seeds.csv`.
    - Cell 10: Reads `results/metrics/ablation_table.csv`.
    - Cell 11: Reads `results/metrics/xai_feature_relevance.csv` and `xai_consistency.csv`.

---

### R2. Documentation Claims & Metric Inconsistencies
- **Status:** **CONFIRMED (Six distinct inconsistencies verified)**
- **Evidence:**
  1. **Latency Claim Contradictions:**
     - `README.md:128`: `0.098 ms`
     - `README.md:173`: `0.08 ms`
     - `results/metrics/benchmark_dev_computer.json`: Median = `0.1229 ms`, Mean = `0.1245 ms`
     - Live Notebook Cell 12: `0.2447 ms`
  2. **CAN-bus Speedup Multipliers:**
     - Variously stated as `>800x` ($100 / 0.123 \approx 813$), `>1000x`, and `1,250x` ($100 / 0.08 = 1250$) without clarifying the 100 ms baseline assumption.
  3. **"Bit-Exact" Claim:**
     - The standalone C/NumPy export is claimed to be "bit-exact", but `results/metrics/export_check.json` reports max absolute differences of $7.78 \times 10^{-5}\%$ (mean) and $2.38 \times 10^{-3}\%$ (std). This is floating-point agreement, not bit-exactness.
  4. **Page Count Constraints:**
     - Previous templates stated "9-page report"; course guidelines mandate strictly $\le 8$ pages.
  5. **Unpinned Requirements:**
     - `requirements.txt` uses `>=` on every entry (`torch>=2.0.0`, etc.) while `README.md` claims pinned dependencies.
  6. **Process Memory Footprint:**
     - Memory delta was measured using `tracemalloc`, which only records Python-allocated heap objects, ignoring external C libraries (BLAS, ONNX Runtime, libequation.so) and OS RSS memory.

---

## Conclusion

Phase 1 audit is complete. All 15 items have been thoroughly verified with code lines, mathematical formulations, and empirical script execution. Zero fixes have been applied during this phase. Awaiting authorization to execute Phase 2.

