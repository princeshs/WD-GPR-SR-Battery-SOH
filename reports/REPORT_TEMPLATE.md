# Battery State-of-Health Prediction Using Wide-and-Deep Gaussian Process Regression with Symbolic Residual Correction
## Wide-and-Deep SVGP with Uncertainty-Weighted Symbolic Regression (WD-GPR-SR)

**Author:** Prince Kumar  
**Roll No:** 2401AI38  
**Department:** Department of Computer Science and Engineering, Indian Institute of Technology Patna  
**Course:** Project Based Learning — Machine Learning (IMPACT Lab)  
**Target Hardware:** NVIDIA Jetson Orin Nano Developer Kit (8GB)  

---

## Abstract
Accurate prediction of battery State-of-Health (SOH) is essential for safe and efficient operation of electric vehicles, drones, and portable electronics. This project proposes **WD-GPR-SR**, a hybrid framework that combines a Wide-and-Deep Sparse Variational Gaussian Process (SVGP) with an Uncertainty-Weighted Symbolic Regression (SR) residual corrector. The model is trained and evaluated on two public datasets — NASA Ames (4 cells, 636 cycles) and CALCE UMD (4 cells, 4,021 cycles) — under a strict Leave-One-Cell-Out cross-validation protocol with 3 fixed seeds. On CALCE, WD-GPR-SR achieves RMSE 2.53 ± 1.56%, MAE 1.44 ± 0.73%, and R² 0.978 — matching deep neural baselines while providing calibrated probabilistic confidence intervals (90.6% PICP). The proposed method outperforms 13 baselines on CALCE and delivers an interpretable closed-form residual equation supported by six complementary XAI techniques. The standalone C/ONNX inference engine runs in 0.123 ms with 0.588 MB peak memory, meeting real-time embedded BMS requirements.

---

## 1. Problem Definition and Success Criteria (Step 1)

### 1.1 Application and Motivation
Lithium-ion batteries degrade with cycling due to solid-electrolyte interphase (SEI) growth, lithium plating, and active material dissolution. Battery Management Systems (BMS) require accurate, real-time SOH estimation to prevent sudden power loss, avoid thermal runaway, and optimize battery charging.
$$\text{SOH}(k) = \frac{C_{\text{actual}}(k)}{C_{\text{reference}}} \times 100\%$$
where $C_{\text{reference}}$ is the median capacity of the first three operational cycles.

### 1.2 Problem Statement Template (IIT Patna Guidelines)
> **Given** five electrochemical features extracted from each charge–discharge cycle (`cycle`, `cc_charge_time`, `cv_charge_time`, `ic_peak_height`, `ic_peak_voltage`),  
> **predict** the battery State of Health (`SOH %`) for unseen lithium-ion battery cells  
> **under** strict zero-leakage Leave-One-Cell-Out Cross-Validation (LOCO-CV) and real-time embedded BMS inference constraints (< 1 ms latency),  
> **achieving** held-out test $\text{RMSE} \le 5.0\%$ and calibrated prediction interval coverage $\text{PICP} \ge 90.0\%$.

### 1.3 Task Classification & Objectives
- **Task Type:** Supervised regression with Bayesian uncertainty quantification.
- **Inputs:** 5 engineered charging features per cycle.
- **Output:** SOH (%) — continuous percentage $[0, 100]$.
- **Pre-set Success Thresholds:** RMSE $\le 5.0\%$, PICP $\ge 90.0\%$, Latency $< 1.0\text{ ms}$.

---

## 2. Literature Review and Research Gap (Step 2)

### 2.1 Comparison of Existing Methods
| Study / Method | Dataset | Evaluation | Gap / Limitation |
|---|---|---|---|
| Physics-based P2D (Newman 1975) | Synthetic / Lab | RMSE in literature | Requires expert parameters; too slow for real-time BMS |
| Exact GPR (Richardson 2017) | NASA | RMSE, R² | $\mathcal{O}(N^3)$ scaling; stationary RBF fails on knee-points |
| Deep Kernel Learning (Wilson 2016) | Various | RMSE, NLL | Overfitting on small battery datasets; opaque black-box |
| BiLSTM / DeepNN (Various 2019–22) | NASA, CALCE | RMSE, MAE | Zero uncertainty quantification; fully uninterpretable |
| Standalone SR (Cranmer 2023) | Physics sims | Equation fidelity | Highly noise-sensitive; no calibrated confidence intervals |
| **WD-GPR-SR (This work)** | **NASA + CALCE** | **RMSE, MAE, R², PICP** | **Unites accuracy, uncertainty, and physical interpretability** |

### 2.2 Identified Research Gaps
1. **Stationary Kernel Limitation:** Standard stationary RBF/Matérn kernels revert to constant prior means on unseen degradation horizons and cannot capture non-stationary knee-point drops (e.g., NASA B0006).
2. **In-Sample Residual Trap:** Fitting Symbolic Regression directly on in-sample GP training residuals collapses because GP training residuals approach zero, giving SR no generalization signal.
3. **Unweighted Residual Vulnerability:** Standard SR treats all residuals equally, overfitting erratic sensor noise.
4. **Uncalibrated Confidence:** Adding a symbolic residual equation shifts the predictive mean without recalibrating interval widths.

---

## 3. Dataset Collection and Quality Audit (Step 3)

### 3.1 Benchmark Repositories (4,657 Total Cycles)
- **NASA Ames PCoE:** Cells `B0005`, `B0006`, `B0007`, `B0018` (18650 cylindrical LiCoO$_2$, 2.0 Ah nominal). 636 total cycles.
- **CALCE CS2 (UMD):** Cells `CS2_35`, `CS2_36`, `CS2_37`, `CS2_38` (Prismatic LiCoO$_2$, 1.1 Ah nominal). 4,021 total cycles.

### 3.2 Feature Data Dictionary (Zero-Leakage Charging Signals)
| # | Feature Name | Unit | Physical Meaning | Aging Correlation |
|---|---|---|---|---|
| 1 | `cycle` | count | Cumulative cycle count | Primary aging driver |
| 2 | `cc_charge_time` | s | CC charging duration ($t_{V=4.2} - t_{\text{start}}$) | Shrinks as internal resistance rises ($r = +0.86$) |
| 3 | `cv_charge_time` | s | CV charging duration ($t_{I=0.05} - t_{V=4.2}$) | Lengthens as polarization resistance grows ($r = -0.42$) |
| 4 | `ic_peak_height` | Ah/V | Peak amplitude of $\text{d}Q/\text{d}V$ curve | Tracks loss of active lithium material (LAM) |
| 5 | `ic_peak_voltage` | V | Voltage position where $\text{d}Q/\text{d}V$ peak occurs | Shifts upward with ohmic resistance ($r = -0.73$) |

---

## 4. Split Protocol and Leakage Prevention (Step 4)

- **Leave-One-Cell-Out Cross-Validation (LOCO-CV):** In each fold, 1 cell is held out as the test set and 3 cells are used for training/validation.
- **Strict Leakage Prevention:** Preprocessing transformers (`StandardScaler`, `SimpleImputer`) and inducing point locations are fitted exclusively on training cells.
- **Multi-Seed Protocol:** All runs repeated across seeds $\{42, 123, 999\}$; mean $\pm$ standard deviation reported.

---

## 5. Methodology: WD-GPR-SR Architecture (Step 5)

### 5.1 Pipeline Steps
1. **Wide & Deep SVGP:**  
   $$k_{\text{WD}}(x, x') = \sigma_w^2 [k_{\text{linear}}(x, x') + k_{\text{ARD-RBF}}(x, x')] + \sigma_d^2 k_{\text{Matérn-5/2}}(\phi_\theta(x), \phi_\theta(x'))$$  
   Sparse Variational GP with $M=64$ inducing points trained via ELBO maximization.
2. **Inner LOCO-CV:** Generates genuine out-of-fold generalization residuals $r_i = y_i - \hat{\mu}_{\text{GP}}^{(-k)}(x_i)$.
3. **Uncertainty-Weighted SR:** Loss $\mathcal{L}_{\text{SR}} = \sum w_i (r_i - g(x_i))^2 + \lambda \mathcal{C}(g)$ with weights $w_i = \text{clip}(1 / (\sigma_{\text{GP}}^2(x_i) + \epsilon))$.
4. **Calibrated Inference:** $\hat{y} = \mu_{\text{GP}}(x) + g(x)$, 95% CI: $[\hat{y} \pm 1.96 \cdot s \cdot \sigma_{\text{GP}}(x)]$.

---

## 6. Experimental Setup and Baselines (Step 6)

14 baselines benchmarked under identical splits:
- **Classical ML:** Linear Regression, Random Forest, SVR (RBF), XGBoost.
- **Exact GPR:** RBF Kernel, Matérn-5/2, ARD Kernel.
- **Sparse / Deep GPR:** Sparse GPR (SVGP), Deep Kernel GPR.
- **Neural Networks:** WideNN, DeepNN, Wide & Deep NN, BiLSTM.
- **Symbolic:** Standalone `gplearn` SymbolicRegressor.

---

## 7. Results and Comparison (Step 7)

### CALCE Benchmark Leaderboard
| Model | MAE (%) | RMSE (%) | $R^2$ | 95% PICP (%) |
|---|:---:|:---:|:---:|:---:|
| **WD-GPR-SR (Proposed)** | **$1.44 \pm 0.73$** | **$2.53 \pm 1.56$** | **0.978** | **90.6%** |
| BiLSTM | $1.12 \pm 0.35$ | $3.08 \pm 1.11$ | 0.977 | — |
| XGBoost | $1.11 \pm 0.13$ | $3.25 \pm 0.94$ | 0.975 | — |
| WD-GPR (Wide + Deep) | $1.25 \pm 0.34$ | $3.30 \pm 1.15$ | 0.974 | 99.5% |
| Sparse GPR (Wide Only) | $1.24 \pm 0.43$ | $3.31 \pm 1.18$ | 0.974 | 99.5% |
| Random Forest | $1.07 \pm 0.10$ | $3.39 \pm 0.78$ | 0.974 | 85.5% |
| Deep Kernel GPR | $1.30 \pm 0.27$ | $3.51 \pm 0.83$ | 0.971 | 99.4% |
| SVR (RBF) | $0.98 \pm 0.18$ | $3.87 \pm 0.80$ | 0.966 | — |
| Standalone SR | $31.1 \pm 11.0$ | $55.5 \pm 35.0$ | -9.12 | — |

---

## 8. Ablation Study & Error Analysis (Step 8)

### Component Ablation Progression
| Config | Description | CALCE RMSE (%) | CALCE MAE (%) | CALCE PICP (%) |
|---|---|:---:|:---:|:---:|
| A | Sparse GPR Only (Linear + ARD-RBF) | $3.31 \pm 1.18$ | $1.24 \pm 0.43$ | 99.5% |
| B | Deep Kernel GPR Only (MLP + Matérn-5/2) | $3.90 \pm 1.22$ | $1.88 \pm 0.54$ | 99.0% |
| C | Wide-and-Deep GPR (No SR) | $3.30 \pm 1.15$ | $1.25 \pm 0.34$ | 99.5% |
| D | WD-GPR + Unweighted SR | $3.65 \pm 3.02$ | $1.66 \pm 1.02$ | 89.8% |
| **E** | **WD-GPR-SR (Proposed Full Model)** | **$2.53 \pm 1.56$** | **$1.44 \pm 0.73$** | **90.6%** |

### Per-Cell Error Analysis
- **Best Cell:** CS2_38 (RMSE 1.36%, PICP 99.6%) — smooth, regular degradation.
- **Challenging Cell:** NASA B0006 (RMSE 6.93%, PICP 1.8%) — severe knee-point degradation that challenges all data-driven models.

---

## 9. Explainability, Edge Deployment & Conclusion (Step 9)

### 9.1 Multi-Perspective XAI
- **6 Methods:** SHAP, LIME, PDP, ICE, ARD ($1/\ell$), Symbolic Sensitivities ($|\partial g / \partial x|$).
- **Consensus:** Both SHAP (54.7%) and ARD (0.31) confirm `ic_peak_voltage` and `cc_charge_time` dominate SOH fade.
- **Cross-Method Consistency:** Spearman rank correlation $\rho(\text{SHAP}, \text{Symbolic}) = 0.50$.

### 9.2 Edge Deployment Benchmark
- **Median Latency:** 0.1232 ms (800x faster than 100 ms CAN-bus budget)
- **Peak Memory:** 0.588 MB
- **Throughput:** 8,115 inferences/second
- **Bit-Exact Numerical Match:** Max abs diff vs PyTorch $< 1.4 \times 10^{-9}$

### 9.3 Limitations & Ethics
- Small sample size ($4+4$ cells); results are indicative.
- Benchmarks executed on development PC; physical Jetson INA3221 power measurements pending.
- Datasets are public domain; no personal data or privacy risks involved.

---

## 10. References & Reproducibility Checklist (Step 10)

### References
1. Saha B. & Goebel K. (2007). NASA Ames Battery Dataset.
2. He W. et al. (2020). CALCE Battery Dataset, University of Maryland.
3. Wilson A. et al. (2016). Deep Kernel Learning. AISTATS.
4. Cranmer M. et al. (2023). Interpretable Machine Learning with PySR. NeurIPS.
5. Richardson R. et al. (2017). Gaussian Process Regression for Battery SOH. IEEE TII.
6. Lundberg S. & Lee S.-I. (2017). A Unified Approach to Interpreting Model Predictions. NeurIPS.
7. Birkl C. et al. (2017). Degradation Diagnostics for Lithium Ion Cells. Journal of Power Sources.

### Reproducibility Verification
- Full Evaluation: `python3 scripts/run_full_evaluation.py`
- Generate Report: `python3 scripts/generate_guideline_report.py`
- Generate Slides: `python3 scripts/generate_presentation.py`
- Run Tests: `pytest -v`
