# Evaluation Protocol & Metric Definitions: WD-GPR-SR Benchmark

**Author:** Senior ML Research Engineer (Student: Prince Kumar, Roll No: 2401AI38)  
**Date:** 2026-11-10  
**Project:** WD-GPR-SR Battery State of Health (SOH) Estimation  
**Benchmark Sets:** NASA 4-cell benchmark (`B0005`, `B0006`, `B0007`, `B0018`: 636 cycles) & CALCE 4-cell benchmark (`CS2_35`, `CS2_36`, `CS2_37`, `CS2_38`: 4,021 cycles).  
**Protocol:** Leave-One-Cell-Out Cross-Validation (LOCO-CV) evaluated independently across random seeds $\{42, 123, 999\}$.

---

## 1. Held-Out Isolation & Validation Boundaries (No held-out-cell information is used in preprocessing, training, calibration or SR selection)

1. **Held-Out Test Cell Isolation:**
   - The held-out cell is strictly isolated during every cross-validation fold.
   - All feature preprocessing (median imputation, standard scaling) is fit exclusively on the remaining training cells.
   - Variational inducing points ($M=64$) are selected via K-Means fit strictly on training features.
   - Neural network weights, variational parameters, kernel hyperparameters, and symbolic regression equations are learned without observing held-out cell data.
   - **Phase 3C GP Training Protocol (Validation-for-Epoch-Selection + Full-Data Refit):**
     * The cell-blocked validation split (last contiguous 15% of cycles of each training cell) is used EXCLUSIVELY to determine the optimal stopping epoch $E^* = \mathrm{argmin}_{e \in [1, E_{\max}]} \mathcal{L}_{\mathrm{val}}(e)$.
     * The model is NEVER deployed using weights fit only on the earlier 85% sub-split, because holding out the last 15% of cycles systematically removes the critical low-SOH degradation tail (which reaches ~5% SOH on CALCE).
     * Once $E^*$ is determined, the GP model is refit from scratch on 100% of the training cycles of all training cells for exactly $E^*$ epochs (with seed reset to ensure identical weight initialization and reproducible shuffling).
     * This guarantees that (1) hyperparameter early stopping is governed by blocked validation without temporal cycle interpolation leakage, and (2) the final deployed model observes 100% of the training cycles including the end-of-life tail.
     * This protocol is applied identically across Configs A, B, C, the base GP inside D and E, and all inner out-of-fold GPs used for SR residuals and calibration.
   - Early stopping uses an internal cell-blocked split: for each training cell, the last contiguous 15% of cycles are assigned to validation, while the earlier 85% are used for fitting. Random per-cycle permutation is strictly prohibited to eliminate temporal cycle interpolation leakage.

2. **Inner Cross-Validation for Calibration & Symbolic Selection:**
   - Out-of-fold residuals $r_{\text{oof}} = y_i - \mu_{\text{GP}}^{(-k)}(x_i)$ and predictive variances $\sigma_{\text{GP}, i}^2$ are computed via inner LOCO-CV among the training cells.
   - **Calibration Rules (Defined Pre-Evaluation):**
     * **Temperature / Variance Scaling:** $s_{\text{var}} = \sqrt{\frac{1}{N_{\text{tr}}} \sum \text{clip}\left(\frac{r_i^2}{\sigma_i^2}, 0.05, 20.0\right)}$.
     * **Quantile / Conformal Scaling:** $s_{\text{quantile}} = \frac{Q_{0.95}(|r_{\text{oof}}| / \sigma_{\text{oof}})}{1.95996}$. By construction, scaling standard deviation by $s_{\text{quantile}}$ achieves exactly $\ge 95\%$ empirical coverage on inner out-of-fold residuals, preventing the variance collapse observed under Gaussian variance scaling on heavy-tailed residuals.
     * Both raw ($s=1$), variance-scaled ($s_{\text{var}}$), and quantile-scaled ($s_{\text{quantile}}$) are reported side-by-side for all probabilistic models without tuning on held-out cells.
   - Symbolic regression complexity and parsimony selection is nested: candidate models (including candidate $g = 0$) are fit on 2 training cells and scored on the 3rd training cell, rotating across all 3 training cells. If $g = 0$ achieves lower out-of-fold error, $g = 0$ is selected.

---

## 2. Evaluation Ranges & Metric Hierarchies

### A. Headline Metric Suite (Full Evaluation Range: All Test Cycles)
The primary evaluation suite evaluates models over the complete lifetime trajectory of each battery cell:
- **Root Mean Squared Error (RMSE, %):** $\sqrt{\frac{1}{N} \sum_{i=1}^N (y_i - \hat{y}_i)^2}$ (Headline point accuracy metric).
- **Mean Absolute Error (MAE, %):** $\frac{1}{N} \sum_{i=1}^N |y_i - \hat{y}_i|$.
- **Mean Squared Error (MSE, %$^2$):** $\frac{1}{N} \sum_{i=1}^N (y_i - \hat{y}_i)^2$.
- **Coefficient of Determination ($R^2$):** $1 - \frac{\sum (y_i - \hat{y}_i)^2}{\sum (y_i - \bar{y})^2}$.
- **Gaussian Negative Log-Likelihood (NLL):** $\frac{1}{N} \sum_{i=1}^N \left[ \frac{1}{2} \ln(2\pi \sigma_i^2) + \frac{(y_i - \hat{y}_i)^2}{2\sigma_i^2} \right]$ (Reported for both raw $\sigma_{\text{GP}}$ and recalibrated $s \cdot \sigma_{\text{GP}}$).
- **Prediction Interval Coverage Probability (PICP, % at 95% nominal):** Percentage of ground truth observations falling within $\hat{y}_i \pm 1.95996 \sigma_i$.
- **Mean Prediction Interval Width (MPIW, % SOH):** $\frac{1}{N} \sum_{i=1}^N 2 \times 1.95996 \sigma_i$.
- **Equation Node Count & Operator Count:** Direct measure of structural interpretability and edge execution complexity ($0$ for models without symbolic residual correction).

### B. Percentage Error Hierarchy (Tail Sensitivity Mitigation)
Because CALCE cells degrade deeply past standard 80% end-of-life (reaching $4.87\%$ SOH on CS2_38), standard percentage division $\frac{|y - \hat{y}|}{y}$ creates severe tail distortions when $y \to 0$. We establish a three-tiered percentage error reporting standard:
1. **Standard MAPE (Full Range, %):**
   $$\text{MAPE}_{\text{full}} = \frac{100\%}{N} \sum_{i=1}^N \frac{|y_i - \hat{y}_i|}{|y_i|}$$
   Guarded against exact division by zero with threshold $|y_i| > 10^{-4}$. Reflects total-range percentage deviation.
2. **Restricted MAPE ($\text{SOH} \ge 50\%$, %):**
   $$\text{MAPE}_{\text{SOH}\ge 50} = \frac{100\%}{N_{50}} \sum_{i: y_i \ge 50\%} \frac{|y_i - \hat{y}_i|}{y_i}$$
   Restricted to active operational lifecycle (above 50% SOH), preventing end-of-life tail division explosion and allowing direct comparison with automotive and aerospace literature.
3. **Symmetric MAPE (sMAPE, %):**
   $$\text{sMAPE} = \frac{100\%}{N} \sum_{i=1}^N \frac{|\hat{y}_i - y_i|}{\frac{|y_i| + |\hat{y}_i|}{2}}$$
   Naturally bounded in $[0\%, 200\%]$, treating over-predictions and under-predictions symmetrically without asymptotic tail divergence.

---

## 3. Immutability Guarantee
This protocol is recorded prior to model re-execution. All baseline models and proposed configurations (A through E) are evaluated identically under this exact protocol.

