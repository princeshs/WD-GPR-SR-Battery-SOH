# Battery State-of-Health Prediction Using Wide-and-Deep Gaussian Process Regression with Symbolic Residual Correction
### Wide-and-Deep SVGP with Uncertainty-Weighted Symbolic Regression (WD-GPR-SR)

An interpretable, uncertainty-quantified machine learning framework for Lithium-ion battery State of Health (SOH) estimation conforming to the **IIT Patna IMPACT Lab End-to-End ML Project Guidelines**.

- **Author:** Prince Kumar &bull; **Roll Number:** 2401AI38  
- **Department:** Department of Computer Science and Engineering, Indian Institute of Technology Patna  
- **Course / Lab:** Project Based Learning — Machine Learning (IMPACT Lab)  
- **Target Edge Device:** NVIDIA Jetson Orin Nano Developer Kit (8GB)  

---

## 1. Problem Formulation (IIT Patna ML Guidelines — Step 1)

> **Given** five charge-cycle electrochemical features (`cycle`, `cc_charge_time`, `cv_charge_time`, `ic_peak_height`, `ic_peak_voltage`),  
> **predict** the battery State of Health (`SOH %`) for unseen lithium-ion battery cells  
> **under** strict zero-leakage Leave-One-Cell-Out Cross-Validation (LOCO-CV) and real-time embedded BMS inference constraints (< 1 ms latency),  
> **achieving** held-out test $\text{RMSE} \le 5.0\%$ and calibrated prediction interval coverage $\text{PICP} \ge 90.0\%$.

---

## 2. System Architecture & Workflow

```text
                            RAW BATTERY CHARGING FEATURES
              [cycle, cc_charge_time, cv_charge_time, ic_peak_height, ic_peak_voltage]
                                          │
                                          ▼
                       ┌──────────────────────────────────────┐
                       │    Zero-Leakage Preprocessing        │
                       │   (Fitted on Training Cells Only)    │
                       └──────────────────┬───────────────────┘
                                          │ Scaled Features X
                 ┌────────────────────────┴────────────────────────┐
                 │                                                 │
                 ▼                                                 ▼
   ┌───────────────────────────┐                     ┌───────────────────────────┐
   │        WIDE BRANCH        │                     │        DEEP BRANCH        │
   │  k_wide = σ_w² [k_linear  │                     │   MLP φ_θ: R^5 -> R^4     │
   │           + k_RBF-ARD]    │                     │  k_deep = σ_d² k_Matern   │
   │   (Global Non-Stationary  │                     │  (Local Non-Linear Health │
   │        Macro Trend)       │                     │    Indicator Manifold)    │
   └─────────────┬─────────────┘                     └─────────────┬─────────────┘
                 │                                                 │
                 └────────────────────────┬────────────────────────┘
                                          ▼
                       ┌──────────────────────────────────────┐
                       │     Wide-and-Deep SVGP Engine        │
                       │   k_WD(x, x') = k_wide + k_deep      │
                       │   Variational Inducing Points Z      │
                       └──────────────────┬───────────────────┘
                                          │
                      Out-of-Fold Residuals r_i = y_i - μ_GP(x_i)
                      Predictive Uncertainties σ²_GP(x_i)
                      via Inner Leave-One-Training-Cell-Out
                                          │
                                          ▼
                       ┌──────────────────────────────────────┐
                       │   Uncertainty-Weighted SR (UW-SR)    │
                       │   Sample Weights: w_i = 1/(σ²_i + ε) │
                       │    Parsimonious Equation Mining:     │
                       │       g(x) via Genetic Programming   │
                       └──────────────────┬───────────────────┘
                                          │
                           Discovered Closed-Form Equation
                                          │
                                          ▼
                       ┌──────────────────────────────────────┐
                       │    Calibrated Inference & Output     │
                       │       ŷ = μ_GP(x) + g(x)             │
                       │       95% CI: ŷ ± 1.96 · s · σ_GP    │
                       └──────────────────────────────────────┘
```

---

## 3. Dataset Layout & Supported Chemistry

We benchmark across 8 individual cells from two widely respected laboratory aging repositories:
1. **NASA Ames Prognostics CoE (NASA):** Cells `B0005`, `B0006`, `B0007`, `B0018` (18650 cylindrical LiCoO$_2$/graphite, nominal 2.0 Ah, CC-CV 1.5A charge, 2.0A discharge, 24°C ambient). Total: 636 cycles.
2. **CALCE CS2 (University of Maryland):** Cells `CS2_35`, `CS2_36`, `CS2_37`, `CS2_38` (Prismatic LiCoO$_2$/graphite, nominal 1.1 Ah, 0.55A CC-CV charge, 1.1A discharge). Total: 4,021 cycles (4,657 total experimental cycles across all 8 cells).

All features are extracted **strictly during the charging protocol** to guarantee zero discharge-cycle leakage.

---

## 4. Installation & Environment Setup

### Prerequisites
- Python 3.10+ (tested on Linux x86_64)
- GCC 11+ (for compiling symbolic equation shared library)

### Install via pip
```bash
# Clone the repository
cd ML_Project

# Install requirements
pip install -r requirements.txt
```

---

## 5. Execution & Verification

To execute the entire project end-to-end and generate all deliverables, run:
```bash
python scripts/build_complete_notebook.py
```
This generates and executes all cells of `2401AI38_MLProject.ipynb` with all live outputs saved inline, compiles `2401AI38_MLProject.pdf` (strictly $\le 8$ pages), and compiles `2401AI38_MLProject_Poster.pdf` (strictly 1 page Landscape A1 research poster).

Alternatively, individual deliverables can be generated directly:
```bash
# Generate 8-page course guideline report
python scripts/generate_guideline_report.py

# Generate 1-page scientific research poster and PNG preview
python scripts/generate_poster.py
```

---

## 6. Master Cross-Validation Leaderboards

### NASA Ames Benchmark Leaderboard
| Model | MAE (%) | RMSE (%) | $R^2$ | 95% PICP (%) | Inference Latency |
|---|:---:|:---:|:---:|:---:|:---:|
| **Sparse GPR (Wide Only)** | $2.97 \pm 2.18$ | **$3.34 \pm 2.17$** | **0.879** | 76.5% | 0.038 ms |
| **WD-GPR (Wide + Deep)** | $3.03 \pm 2.24$ | $3.41 \pm 2.22$ | 0.874 | 77.4% | 0.082 ms |
| Random Forest | $3.08 \pm 1.69$ | $3.55 \pm 1.75$ | 0.870 | - | 0.106 ms |
| BiLSTM | $3.04 \pm 2.28$ | $3.59 \pm 2.23$ | 0.859 | - | 0.017 ms |
| Exact GPR (Matérn-5/2) | $3.17 \pm 2.08$ | $3.59 \pm 2.02$ | 0.863 | 76.2% | 0.061 ms |
| Deep Kernel GPR | $3.12 \pm 2.34$ | $3.65 \pm 2.48$ | 0.852 | 78.4% | 0.044 ms |
| XGBoost | $3.26 \pm 1.94$ | $3.70 \pm 2.00$ | 0.857 | - | 0.006 ms |
| SVR (RBF) | $3.58 \pm 2.97$ | $4.21 \pm 3.33$ | 0.791 | - | 0.010 ms |
| WD-GPR + Unweighted SR | $3.59 \pm 2.01$ | $4.31 \pm 1.86$ | 0.805 | 77.1% | 0.095 ms |
| **Proposed WD-GPR-SR** | $4.08 \pm 1.70$ | $4.93 \pm 1.48$ | 0.740 | 62.5% | 0.098 ms |

### CALCE CS2 Benchmark Leaderboard
| Model | MAE (%) | RMSE (%) | $R^2$ | 95% PICP (%) | Inference Latency |
|---|:---:|:---:|:---:|:---:|:---:|
| **Proposed WD-GPR-SR** | $1.44 \pm 0.73$ | **$2.53 \pm 1.56$** | **0.978** | **90.6%** | 0.098 ms |
| BiLSTM | $1.12 \pm 0.35$ | $3.08 \pm 1.11$ | 0.977 | - | 0.017 ms |
| XGBoost | $1.11 \pm 0.13$ | $3.25 \pm 0.94$ | 0.975 | - | 0.006 ms |
| **WD-GPR (Wide + Deep)** | $1.25 \pm 0.34$ | $3.30 \pm 1.15$ | 0.974 | 99.5% | 0.082 ms |
| Sparse GPR (Wide Only) | $1.24 \pm 0.43$ | $3.31 \pm 1.18$ | 0.974 | 99.5% | 0.038 ms |
| Random Forest | $1.07 \pm 0.10$ | $3.39 \pm 0.78$ | 0.974 | 85.5% | 0.106 ms |
| WD-GPR + Unweighted SR | $1.66 \pm 1.02$ | $3.65 \pm 3.02$ | 0.951 | 89.8% | 0.095 ms |
| Exact GPR (Matérn-5/2) | $1.24 \pm 0.13$ | $3.82 \pm 0.79$ | 0.966 | 89.8% | 0.061 ms |
| SVR (RBF) | $0.98 \pm 0.18$ | $3.87 \pm 0.80$ | 0.966 | - | 0.010 ms |
| Deep Kernel GPR | $1.88 \pm 0.54$ | $3.90 \pm 1.22$ | 0.965 | 99.0% | 0.044 ms |

### Secondary Benchmark: Pooled 8-Cell Cross-Domain LOCO
- **WD-GPR Pooled (NASA + CALCE):** RMSE = $3.67 \pm 1.40\%$, MAE = $2.35 \pm 1.60\%$, $R^2 = 0.907$.

---

## 7. Architectural Ablation Analysis

Ablation across Configs A–E confirms the specialized roles of each architectural element:
1. **Wide Branch (Linear + ARD-RBF):** Crucial for preventing mean-reversion on unseen trajectories. Without the wide branch, the deep kernel (Config B) degrades to RMSE $3.90\%$ on CALCE.
2. **Deep Branch (MLP $\to$ Matérn):** Reduces negative log-likelihood (NLL) and models local health indicator interactions.
3. **Uncertainty Weighting:** On knee-point cell NASA B0006, symbolic correction successfully fits the non-linear degradation drop (RMSE $7.00\% \to 6.93\%$). On CALCE, where base WD-SVGP residuals are white noise, SR correctly learns constant offsets ($\pm 0.1\%$) and uncertainty weighting dampens transfer variance.
4. **Calibrated Uncertainty:** Recalibration factor $s$ brings CALCE empirical coverage to a well-calibrated $90.6\%$ (with RMSE improving to $2.53\%$, down from over-conservative $99.5\%$).

---

## 8. Explainable AI (XAI) Consensus

Feature attributions across SHAP, ARD ($1/\ell$), and Symbolic Sensitivities ($|\partial g / \partial x|$):
- `cc_charge_time` and `cycle` are the dominant global degradation drivers across both SHAP and ARD kernels (Spearman $\rho = 0.50$).
- LIME reveals dynamic lifecycle shifts: `cc_charge_time` provides strong positive attribution (+4.35%) in early life, whereas `cycle` and `ic_peak_voltage` penalize late-life predictions.
- Symbolic sensitivities specifically target non-linear electrochemical polarization in `cv_charge_time` and `ic_peak_voltage` ($33.8\%$ each).

---

## 9. Edge Deployment on NVIDIA Jetson

- **Standalone Package:** No PyTorch dependency at inference time. Pure NumPy + ONNX Runtime + C `libequation.so` via `ctypes`.
- **Numerical Verification:** Predictive mean max abs diff vs PyTorch is **$7.78 \times 10^{-5}\%$ SOH**; standard deviation max diff is **$2.38 \times 10^{-3}\%$** (`results/metrics/export_check.json`).
- **Benchmarking (Dev Computer functional test):**
  - Median Latency: **$0.08\text{ ms}$** per inference (p95: $0.12\text{ ms}$)
  - Throughput: **$8,114.8\text{ inferences/sec}$**
  - Peak Memory: **$0.59\text{ MB}$**
  - Hardware Label: Strictly labeled *"development computer, NOT Jetson"*
- **Physical Jetson Instructions:** Documented in `not_usable/deployment/README_JETSON.md` for JetPack 5.1/6.0, TensorRT `trtexec --fp16`, and `nvpmodel` 15W/7W modes.

---

## 10. Critical Limitations & Caveats

1. **Small-N Sample Size:** Evaluated on 4 NASA and 4 CALCE cells. Cross-cell variation dominates standard deviations; no claims of formal statistical significance are made (`reports/LIMITATIONS_STATS.md`).
2. **Neutrality on Linear Fade:** When base GP residuals are white noise (CALCE), symbolic regression finds constant offsets without functional benefit.
3. **Physical Jetson Measurements:** Real-time electrical power (Watts) and energy per inference (mJ) require execution on physical Jetson Orin Nano hardware using onboard INA3221 monitors.

---

## 11. Project Completion Checklist (IIT Patna Guidelines Page 20)

- [x] **Problem and metrics defined:** Stated with inputs, target, constraints, and success thresholds.
- [x] **Literature and gap documented:** Compared 5 approaches with explicit limitation table.
- [x] **Dataset source and license recorded:** Public NASA and CALCE aging datasets cited and documented.
- [x] **Data audit and EDA completed:** Checked missingness, anomalies, and generated `notebooks/01_Exploratory_Data_Analysis.ipynb`.
- [x] **Leak-free split and pipeline verified:** Strict LOCO-CV; all transformers fitted exclusively inside train folds.
- [x] **Simple baseline established:** Linear Regression, Random Forest, SVR, XGBoost, Exact GPR.
- [x] **Models tuned using validation only:** Inner LOCO-CV residuals used for SR; test cells evaluated once.
- [x] **Held-out evaluation completed:** Full 8-cell LOCO-CV across 3 seeds {42, 123, 999}.
- [x] **Results and failures interpreted:** B0006 knee-point analyzed; trade-offs discussed.
- [x] **Seeds, versions, and commands recorded:** Pinned dependencies in `requirements.txt` and `configs/config.yaml`.
- [x] **Code, report, and research poster organized:** Standard folder structure with 8-page PDF report and 1-page Landscape A1 research poster.
- [x] **Limitations and references included:** 10 credible references and honest boundary conditions.
