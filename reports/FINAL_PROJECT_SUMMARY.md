# WD-GPR-SR: Final Project Technical Summary Report

**Project Title:** WD-GPR-SR: When Uncertainty Meets Equations in Battery Health Prediction  
**Date:** 2026-10-31  
**Target Hardware:** NVIDIA Jetson Orin Nano Developer Kit (8GB)  
**Author / Student:** Prince Kumar  
**Roll Number:** 2401AI38  

---

## 1. Problem Statement & Motivation
Accurate State of Health (SOH) estimation is essential for lithium-ion battery management systems (BMS). Existing data-driven methodologies face three critical shortcomings:
1. **Black-box Neural Networks:** Lack Bayesian uncertainty quantification, suffer from covariate shift across individual cells, and lack interpretability.
2. **Exact Gaussian Process Regression:** Scales cubically $\mathcal{O}(N^3)$, limiting edge viability. Stationary kernels (e.g. standard RBF) revert to constant prior means on unseen trajectories, failing to extrapolate macroscopic non-stationary capacity fade.
3. **Symbolic Regression Vulnerability:** Pure genetic programming tends to overfit noisy, high-dimensional spaces with bloated, non-physical mathematical expressions.

---

## 2. Methodology & Proposed Architecture
We introduced the **WD-GPR-SR** framework, combining:
1. **Wide-and-Deep Variational Gaussian Process (WD-SVGP):**
   - Composite PSD kernel:
     $$k_{\text{WD}}(x, x') = \sigma_w^2 [k_{\text{linear}}(x, x') + k_{\text{RBF-ARD}}(x, x')] + \sigma_d^2 k_{\text{Matérn-2.5}}(\phi_\theta(x), \phi_\theta(x'))$$
   - **Wide Branch:** Non-stationary linear kernel + ARD-RBF kernel over 5 features to establish the global macroscopic degradation trajectory.
   - **Deep Branch:** Regularized neural feature extractor $\phi_\theta: \mathbb{R}^5 \to \mathbb{R}^4$ (2 hidden layers, LayerNorm, Dropout $p=0.1$) followed by Matérn-5/2 kernel to model non-linear health indicator interactions.
   - **Variational ELBO:** 32 inducing points initialized via k-means, trained via Adam ($\eta=0.015$, weight decay $10^{-4}$) with early stopping and automatic target standardization.
2. **Uncertainty-Weighted Symbolic Regression (UW-SR):**
   - Solves the *near-zero-residual trap* via inner Leave-One-Training-Cell-Out CV, extracting genuine out-of-fold residuals $r_i = y_i - \mu_{\text{GP}}(x_i)$ and predictive uncertainties $\sigma_i^2$.
   - Inverse-variance sample weights: $w_i = \text{clip}(1 / (\sigma_i^2 + \epsilon))$, normalized to mean 1.
   - Parsimonious genetic programming: $\mathcal{L}_{\text{SR}}(g) = \frac{1}{N}\sum w_i (r_i - g(x_i))^2 + \lambda \mathcal{C}(g)$.
   - Variance recalibration factor $s = \sqrt{\frac{1}{N}\sum (r_i - g(x_i))^2 / \sigma_{\text{GP}, i}^2}$ applied to predictive bounds.

---

## 3. Data & Zero-Leakage Preprocessing
- **NASA Ames Dataset:** 4 cylindrical 18650 cells (`B0005`, `B0006`, `B0007`, `B0018`), 616 total cycles.
- **CALCE CS2 Dataset:** 4 prismatic cells (`CS2_35`, `CS2_36`, `CS2_37`, `CS2_38`), 4,041 total cycles.
- **Features Extracted Strictly During Charging:**
  1. `cycle`: Cumulative cycle index
  2. `cc_charge_time`: Constant-current charging duration
  3. `cv_charge_time`: Constant-voltage taper duration
  4. `ic_peak_height`: Incremental capacity peak height via Savitzky-Golay filtering
  5. `ic_peak_voltage`: Voltage corresponding to the main $dQ/dV$ phase transition peak
- **Strict Leakage Prevention:** Scaler and imputer fitted strictly on training cells in each Leave-One-Cell-Out fold.

---

## 4. Actual Cross-Validation Results

Full Leave-One-Cell-Out Cross-Validation (LOCO-CV) across NASA (4 folds) and CALCE (4 folds) over seeds {42, 123, 999} (24 outer runs per model):

| Model | NASA RMSE (%) | NASA MAE (%) | NASA $R^2$ | CALCE RMSE (%) | CALCE MAE (%) | CALCE $R^2$ | CALCE 95% PICP | Latency |
|---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **Sparse GPR (Wide Only)** | **$3.34 \pm 2.17$** | $2.97 \pm 2.18$ | **0.879** | $3.31 \pm 1.18$ | $1.24 \pm 0.43$ | 0.974 | 99.5% | 0.038 ms |
| **WD-GPR (Wide + Deep)** | $3.41 \pm 2.22$ | $3.03 \pm 2.24$ | 0.874 | $3.30 \pm 1.15$ | $1.25 \pm 0.34$ | 0.974 | 99.5% | 0.082 ms |
| Random Forest | $3.55 \pm 1.75$ | $3.08 \pm 1.69$ | 0.870 | $4.05 \pm 1.37$ | $1.41 \pm 0.49$ | 0.959 | - | 0.106 ms |
| BiLSTM | $3.59 \pm 2.23$ | $3.04 \pm 2.28$ | 0.859 | **$3.16 \pm 1.21$** | $1.28 \pm 0.28$ | **0.976** | - | 0.017 ms |
| Exact GPR (Matérn-5/2) | $3.59 \pm 2.02$ | $3.17 \pm 2.08$ | 0.863 | $3.51 \pm 1.28$ | $1.17 \pm 0.26$ | 0.971 | 99.2% | 0.061 ms |
| Deep Kernel GPR | $3.65 \pm 2.48$ | $3.12 \pm 2.34$ | 0.852 | $3.90 \pm 1.22$ | $1.88 \pm 0.54$ | 0.965 | 99.0% | 0.044 ms |
| XGBoost | $3.70 \pm 2.00$ | $3.26 \pm 1.94$ | 0.857 | $3.49 \pm 1.21$ | $1.39 \pm 0.40$ | 0.971 | - | 0.006 ms |
| SVR (RBF) | $4.21 \pm 3.33$ | $3.58 \pm 2.97$ | 0.791 | $3.42 \pm 1.07$ | **$1.04 \pm 0.32$** | 0.973 | - | 0.010 ms |
| WD-GPR + Unweighted SR | $4.31 \pm 1.86$ | $3.59 \pm 2.01$ | 0.805 | $3.30 \pm 1.24$ | $1.32 \pm 0.42$ | 0.974 | 99.2% | 0.095 ms |
| **Proposed WD-GPR-SR** | $4.93 \pm 1.48$ | $4.08 \pm 1.70$ | 0.740 | **$3.29 \pm 1.23$** | $1.27 \pm 0.45$ | 0.974 | **96.0%** | 0.098 ms |

- **Secondary Pooled 8-Cell LOCO:** WD-GPR across NASA + CALCE yields RMSE = $3.67 \pm 1.40\%$, MAE = $2.35 \pm 1.60\%$, $R^2 = 0.907$.

---

## 5. Final Discovered Symbolic Equations
For NASA Ames (Cell B0005, Seed 42), the uncertainty-weighted genetic program discovered:
$$g(x) = 3 \cdot x_0 - x_1 \cdot x_3 + x_1 - 5 \cdot x_4 - \ln(|x_2| + 10^{-6}) - \sqrt{|x_4 + 0.492|}$$
where $x_0 = \text{cycle}$, $x_1 = \text{cc\_charge\_time}$, $x_2 = \text{cv\_charge\_time}$, $x_3 = \text{ic\_peak\_height}$, $x_4 = \text{ic\_peak\_voltage}$.
Complexity: 34 nodes, 18 operators. Variance recalibration factor: $s = 0.9981$.

---

## 6. Explainable AI (XAI) Findings
1. **Global Attribution:** SHAP highlights `ic_peak_voltage` ($47.5\%$) and `cc_charge_time` ($34.8\%$). ARD kernel relevance identifies `cc_charge_time` ($31.3\%$, $\ell=1.390$) and `cycle` ($21.1\%$, $\ell=2.058$) as primary drivers.
2. **Lifecycle Dynamic Shifts (LIME):**
   - Early Life (Cycle 10): Strongly positive `cc_charge_time` (+4.35%) keeps prediction near 100% SOH.
   - Mid Life (Cycle 80): Penalized by `cycle` (-3.55%) and polarization voltage.
   - Late Life (Cycle 150): Cumulative aging penalty dominates as the cell enters the knee.
3. **Rank Consistency:** Spearman rank correlation between SHAP and ARD is $\rho = 0.50$; SHAP and Symbolic sensitivities is $\rho = 0.70$.

---

## 7. Edge Deployment Status & Verification

### What Was Tested & Verified on Development Computer:
1. **Model Export:** Deep feature extractor $\phi_\theta$ exported to ONNX (opset 17, dynamic batch). ONNX Runtime vs PyTorch max abs diff: **$1.397 \times 10^{-9}$**.
2. **SVGP Arrays:** Inducing points $Z$, weights $\alpha = L_Z^{-T} m$, and covariance matrix $A_{\text{cov}}$ saved in `svgp_arrays.npz`.
3. **C Code Compilation:** Symbolic equation compiled with `gcc -O3 -shared -fPIC` into `libequation.so` invoked via `ctypes`.
4. **Numerical Verification:** End-to-end exported pipeline matches PyTorch across all 168 test cycles of NASA B0005 within tolerance:
   - Predictive Mean Max Abs Diff: **$7.7845 \times 10^{-5}\%$ SOH**
   - Predictive Std Dev Max Abs Diff: **$2.3849 \times 10^{-3}\%$**
   - Verification Status: **PASSED** (`results/metrics/export_check.json`).
5. **Functional Benchmark (Dev Computer, NOT Jetson):**
   - Median Latency: **$0.1232\text{ ms}$** per cycle (p95: $0.1393\text{ ms}$, p99: $0.1635\text{ ms}$).
   - Throughput: **$8,114.8\text{ inferences/sec}$**, Peak Memory: **$0.59\text{ MB}$**.

### What Requires Physical Jetson Hardware:
- Physical electrical power draw (Watts) and energy per inference (mJ) require execution on an actual NVIDIA Jetson Orin Nano Developer Kit using onboard INA3221 power monitors via `jtop` or `tegrastats`.
- Full setup instructions documented in `deployment/README_JETSON.md`.

---

## 8. Critical Limitations & Future Work
1. **Cell Heterogeneity & Sample Size ($N=4+4$):** Because dataset size is limited to 4 NASA and 4 CALCE cells, inter-cell variation exceeds algorithmic differences. No claims of formal statistical significance are made (`reports/LIMITATIONS_STATS.md`).
2. **Residual Neutrality on Calm Degradation:** On CALCE CS2, where base WD-SVGP captures degradation with extreme precision (MAE $\approx 1.25\%$), residuals are Gaussian white noise; genetic programming discovers constant offsets ($\pm 0.1\%$) with no functional change in RMSE ($3.30\% \to 3.29\%$).
3. **Future Work:** Extend evaluation to dynamic drive cycles (e.g. Oxford, Stanford battery datasets) and execute physical power profiling on NVIDIA Jetson hardware under thermal stress.

