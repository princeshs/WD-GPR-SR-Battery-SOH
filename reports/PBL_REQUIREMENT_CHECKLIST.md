# PBL Project Requirement Checklist: WD-GPR-SR

**Project Title:** WD-GPR-SR: When Uncertainty Meets Equations in Battery Health Prediction  
**Student:** Prince Kumar  
**Roll Number:** 2401AI38  
**Problem Statement:** Lithium-ion Battery State of Health (SOH) estimation with Gaussian Process Regression, Wide & Deep feature learning, and uncertainty-weighted Symbolic Regression residual correction.  
**Hardware Target:** NVIDIA Jetson Orin Nano Super Developer Kit (8GB, 7–25W, up to 67 INT8 TOPS)  
**Evaluation Protocol:** Leave-One-Cell-Out Cross-Validation (LOCO-CV) across 8 cells (NASA B0005, B0006, B0007, B0018 and CALCE CS2_35, CS2_36, CS2_37, CS2_38); fixed random seed; mean ± standard deviation over at least 3 random seeds.  
**Audit Date:** 2026-11-03  

---

## 1. Deliverables Checklist

| ID | Deliverable | Format / Constraints | Status | Evidence / File Path |
|---|---|---|:---:|---|
| **DEL-01** | Presentation | `.pptx` (Max 10 slides) | **Completed** | Generated and verified (10 slides) in `2401AI38_MLProject.pptx` |
| **DEL-02** | Project Report | `.pdf` (Max 8 pages) | **Completed** | Generated and verified (8 pages) in `2401AI38_MLProject.pdf` |
| **DEL-03** | Code Notebook | `.ipynb` (Self-contained) | **Completed** | Executed end-to-end with all cell outputs in `2401AI38_MLProject.ipynb` (and `WD-GPR-SR-Implementation/2401AI38_MLProject.ipynb`) |

---

## 2. Rules & Governance Checklist

| ID | Rule Category | Requirement Specification | Status | Evidence / File Path |
|---|---|---|:---:|---|
| **RUL-01** | Public Data Only | NASA Ames and CALCE CS2 battery datasets; fixed split with explicit random seed (`random_state=42`). | **Completed** | `data/processed/*.csv`, `src/config.py` |
| **RUL-02** | No Data Leakage | Test data strictly held out; preprocessing fitted strictly on training cells; inner LOCO for SR out-of-fold residuals. | **Completed** | `src/evaluation.py`, `tests/test_evaluation.py` |
| **RUL-03** | Fair Baselines | All 14 baselines and proposed model use identical data pipeline, preprocessing, and seeds {42, 123, 999}. | **Completed** | `src/baseline_models.py`, `results/metrics/baselines.csv` |
| **RUL-04** | Statistical Reliability | Report final performance metrics as mean ± standard deviation across 3 distinct random seeds. | **Completed** | `results/metrics/summary_mean_std_over_seeds.csv` |
| **RUL-05** | Ablation Study | Systematic ablation of Wide branch, Deep branch, Unweighted SR, and Uncertainty-Weighted SR (Configs A–E). | **Completed** | `results/metrics/ablation_table.csv`, `scripts/run_full_evaluation.py` |
| **RUL-06** | Edge Profiling | ≥100 warmup, ≥1,000 timed runs, median & p95 latency, peak RAM, power modes (15W). Labeled strictly "dev computer, NOT Jetson" without hardware. | **Completed** (Functional) / **Requires physical hardware** (Power) | `deployment/benchmark.py`, `results/metrics/benchmark_dev_computer.json`, `deployment/README_JETSON.md` |
| **RUL-07** | Relative Paths & Immutability | Exclusive relative paths; raw original data files never modified or overwritten. | **Completed** | `reports/CODE_AUDIT.md`, `src/data_loading.py` |

---

## 3. Performance Metrics Checklist

| ID | Metric Group | Metric | Formula / Meaning | Status | Evidence / File Path |
|---|---|---|---|:---:|---|
| **MET-01** | Predictive Accuracy | **MAE** | $\frac{1}{N}\sum \|y - \hat{y}\|$ (Mean Absolute Error) | **Completed** | `results/metrics/full_results_long.csv` |
| **MET-02** | Predictive Accuracy | **MSE** | $\frac{1}{N}\sum (y - \hat{y})^2$ (Mean Squared Error) | **Completed** | `results/metrics/full_results_long.csv` |
| **MET-03** | Predictive Accuracy | **RMSE** | $\sqrt{\text{MSE}}$ (Root Mean Squared Error) | **Completed** | `results/metrics/full_results_long.csv` |
| **MET-04** | Predictive Accuracy | **$R^2$** | Coefficient of Determination | **Completed** | `results/metrics/full_results_long.csv` |
| **MET-05** | Predictive Accuracy | **MAPE** | $\frac{100\%}{N}\sum \|\frac{y - \hat{y}}{y}\|$ | **Completed** | `results/metrics/full_results_long.csv` |
| **MET-06** | Uncertainty Quality | **NLL** | Gaussian Negative Log-Likelihood with $10^{-6}$ variance floor | **Completed** | `results/metrics/full_results_long.csv`, `src/evaluation.py` |
| **MET-07** | Uncertainty Quality | **PICP** | 95% Prediction Interval Coverage Probability | **Completed** | `results/metrics/full_results_long.csv`, `src/evaluation.py` |
| **MET-08** | Uncertainty Quality | **MPIW** | Mean Prediction Interval Width at 95% ($2 \times 1.96 \times \sigma$) | **Completed** | `results/metrics/full_results_long.csv`, `src/evaluation.py` |
| **MET-09** | Interpretability | **Node Count** | Symbolic equation complexity / syntactic tree size | **Completed** | `results/metrics/sr_results.csv`, `results/equations/*.txt` |
| **MET-10** | Edge Performance | **Latency** | Median ($0.123\text{ ms}$) and P95 ($0.139\text{ ms}$) over 1,000 runs | **Completed** | `results/metrics/benchmark_dev_computer.json` |
| **MET-11** | Edge Performance | **Power & Energy** | Peak memory: $0.59\text{ MB}$; Physical Watts & mJ require Jetson | **Requires physical hardware** | `deployment/README_JETSON.md`, `results/metrics/benchmark_dev_computer.json` |

---

## 4. Model Architecture & Baselines Checklist

| ID | Component | Models Required | Status | Evidence / File Path |
|---|---|---|:---:|---|
| **MOD-01** | Baseline GPR | Standard GPR with RBF, Matérn-5/2, and ARD kernels | **Completed** | `src/baseline_models.py`, `results/metrics/baselines.csv` |
| **MOD-02** | Baseline Sparse GPR | Sparse Variational GPR (SVGP / 32 Inducing points) | **Completed** | `src/baseline_models.py`, `results/metrics/baselines.csv` |
| **MOD-03** | Baseline Deep-Kernel GPR | Deep Kernel Learning (DKL) GPR (MLP $\to$ Matérn-5/2) | **Completed** | `src/baseline_models.py`, `results/metrics/baselines.csv` |
| **MOD-04** | Classical ML Baselines | SVR (RBF), Random Forest, XGBoost | **Completed** | `src/baseline_models.py`, `results/metrics/baselines.csv` |
| **MOD-05** | Neural Baselines | WideNN, DeepNN, Wide & Deep NN | **Completed** | `src/baseline_models.py`, `results/metrics/baselines.csv` |
| **MOD-06** | Recurrent Baseline | BiLSTM operating on cycle sequence representations | **Completed** | `src/baseline_models.py`, `results/metrics/baselines.csv` |
| **MOD-07** | Symbolic Baseline | Standalone Symbolic Regression (`gplearn`) | **Completed** | `src/baseline_models.py`, `results/metrics/baselines.csv` |
| **MOD-08** | Proposed Model | **WD-GPR-SR**: Wide-and-Deep SVGP + Uncertainty-Weighted Symbolic Regression | **Completed** | `src/gpr_models.py`, `src/symbolic_regression.py`, `results/metrics/sr_results.csv` |

---

## 5. Explainable AI (XAI) Checklist (At Least 5 Required)

| ID | Method | Description | Status | Evidence / File Path |
|---|---|---|:---:|---|
| **XAI-01** | SHAP | KernelExplainer with k-means background summary ($k=25$) | **Completed** | `src/explainability.py`, `results/figures/explainability/shap_summary.png` |
| **XAI-02** | LIME | Local explanations across Early, Mid, and Late life stages | **Completed** | `src/explainability.py`, `results/figures/explainability/lime_local_explanations.png` |
| **XAI-03** | PDP | Partial Dependence Plots showing marginal health indicator effects | **Completed** | `src/explainability.py`, `results/figures/explainability/pdp_ice_curves.png` |
| **XAI-04** | ICE | Individual Conditional Expectation curves showing sample variance | **Completed** | `src/explainability.py`, `results/figures/explainability/pdp_ice_curves.png` |
| **XAI-05** | ARD Relevance | Wide branch ARD inverse length-scales ($1/\ell_d$) | **Completed** | `src/explainability.py`, `results/figures/explainability/ard_lengthscales.png` |
| **XAI-06** | Sensitivity Analysis | Closed-form analytic partial derivatives $|\partial g / \partial x_j|$ | **Completed** | `src/explainability.py`, `results/figures/explainability/symbolic_sensitivities.png` |

---

## 6. Edge Deployment Checklist (Jetson Orin Nano)

| ID | Deployment Stage | Details | Status | Evidence / File Path |
|---|---|---|:---:|---|
| **EDG-01** | Deep Feature Extractor | Export PyTorch $\phi_\theta$ to ONNX (opset 17, dynamic batch); verified vs PyTorch ($< 1.4 \times 10^{-9}$ diff) | **Completed** | `deployment/export_model.py`, `deployment/phi_theta.onnx` |
| **EDG-02** | Sparse GP Component | Precompute & serialize inducing points $Z$, $\alpha$, and $A_{\text{cov}}$ matrices | **Completed** | `deployment/export_model.py`, `deployment/svgp_arrays.npz` |
| **EDG-03** | Symbolic Equation | Transpile mathematical equation to C code, compile to `libequation.so` | **Completed** | `deployment/equation.c`, `deployment/libequation.so`, `deployment/Makefile` |
| **EDG-04** | Hardware Benchmarking | 100 warmups, 1,000 timed runs, median latency $0.123\text{ ms}$, peak RAM $0.59\text{ MB}$; physical power requires Jetson | **Completed** (Functional) / **Requires physical hardware** (Power) | `deployment/benchmark.py`, `results/metrics/benchmark_dev_computer.json`, `deployment/README_JETSON.md` |

---

## 7. Checklist Count Summary
- **Total Requirements Tracked:** 36 items
- **Completed:** 34 items (100% of non-physical software/documentation requirements)
- **Requires Physical Hardware:** 2 items (physical onboard INA3221 power and energy measurement on Jetson)
- **Partially Completed:** 0 items
- **Not Started:** 0 items
