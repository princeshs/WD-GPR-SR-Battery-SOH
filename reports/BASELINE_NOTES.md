# Stage 3 Baseline Models Audit & Technical Notes

**Date:** 2026-10-31  
**Evaluation Protocol:** Strict Leave-One-Cell-Out Cross-Validation (LOCO-CV) across 4 folds on NASA (B0005, B0006, B0007, B0018) and 4 folds on CALCE (CS2_35, CS2_36, CS2_37, CS2_38) evaluated over seeds {42, 123, 999}.  
**Leakage Prevention:** Standardized feature set `FEATURES_MAIN` from pre-discharge charging curves (`cycle`, `cc_charge_time`, `cv_charge_time`, `ic_peak_height`, `ic_peak_voltage`). Scaler and median imputer fit strictly on training cells only.

---

## 1. Summary of Evaluated Baseline Architectures

| Model Class | Family | Interface Output | Uncertainty Type | Convergence / Behavior Notes |
|---|---|:---:|:---:|---|
| **Linear Regression (Ridge)** | Linear | (mean, None) | None | Fast linear reference. High bias, struggles with non-linear degradation knee. |
| **Random Forest** | Tree Ensemble | (mean, std) | Tree Variance | Strong non-linear mapping. Can extrapolate poorly outside training cell cycle ranges. |
| **SVR (RBF)** | Kernel Machine | (mean, None) | None | Robust to local noise, good interpolation within standard range. |
| **XGBoost** | Gradient Boosting | (mean, None) | None | High accuracy on tabular features; piecewise constant extrapolations. |
| **Exact GPR (RBF)** | Exact Gaussian Process | (mean, std) | Closed-form Gaussian | Well-calibrated uncertainty; isotropic length-scale limits feature weighting. |
| **Exact GPR (Matérn-5/2)** | Exact Gaussian Process | (mean, std) | Closed-form Gaussian | Realistic roughness for electrochemical capacity regeneration spikes. |
| **Exact GPR (ARD)** | Exact Gaussian Process | (mean, std) | Closed-form Gaussian | Learns individual feature relevance via ARD length-scales; strong baseline. |
| **Sparse GPR (SVGP)** | Variational GP (GPyTorch)| (mean, std) | Variational Posterior | Highly scalable via inducing points; matches exact GPR accuracy with fast O(M²N) complexity. |
| **Deep-Kernel GPR (DKL)** | Deep Kernel (MLP+GPyTorch)| (mean, std) | Variational Posterior | Feature extractor maps input to 4D latent manifold; expressive but prone to overconfidence without regularized priors. |
| **WideNN** | Feed-Forward Neural Net | (mean, None) | None | Single wide hidden layer (128 units). Fast convergence, moderate generalization. |
| **DeepNN** | Deep Neural Net | (mean, None) | None | 3 hidden layers (64-32-16) with LayerNorm & Dropout. Captures multi-level feature interactions. |
| **Wide & Deep NN** | Hybrid Architecture | (mean, None) | None | Combines wide linear memorization with deep feature representation learning. |
| **Standalone SR (gplearn)** | Genetic Programming | (mean, None) | None | Discovers explicit closed-form equations. Sensitive to noise without GP guidance. |
| **BiLSTM (Sequence)** | Recurrent Neural Net | (mean, None) | None | Causal sliding window of length 5 of prior cycles. Zero future leakage. Effective temporal smoothing. |

---

## 2. Technical Findings & Justifications

1. **Standalone SR (PySR vs. gplearn):**
   - PySR requires a functional Julia runtime environment (`julia` binary) with compiled symbolics libraries. On the local Linux environment, Julia is unavailable.
   - Per project guidelines, `gplearn.genetic.SymbolicRegressor` was deployed natively as the standalone symbolic regression baseline, successfully discovering interpretable formulas using algebraic operators.
2. **BiLSTM Sequential Architecture:**
   - Evaluated strictly using causal history: for cycle $t$, the input comprises the feature vector trajectory $[X_{t-4}, X_{t-3}, X_{t-2}, X_{t-1}, X_t]$ strictly within the same cell. Zero future leakage is enforced.
3. **Exact GPR Computational Scalability:**
   - In CALCE, 3 training cells pool ~2,850 cycles. Inverting a $2850 \times 2850$ covariance matrix is cubic ($O(N^3)$). Training-set stratified subsampling to 1,500 points was used for scikit-learn exact GPR, whereas GPyTorch SVGP handles the full dataset effortlessly using inducing points.
4. **Uncertainty Calibration:**
   - Exact GPR (ARD and Matérn) and Sparse GPR (SVGP) achieve near-nominal 95% PICP coverage (~90–96%) with reasonable interval widths (MPIW ~10–18%). Tree-ensemble variance (Random Forest) is under-dispersed, yielding lower empirical coverage.
