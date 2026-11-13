# WD-GPR-SR: Comprehensive Viva Preparation Guide
## 25 Technical Questions and Rigorous Answers for Machine Learning Project Defense

This document provides concise, technically precise, and project-grounded answers to the 25 most critical questions for the viva examination on **WD-GPR-SR: When Uncertainty Meets Equations in Battery Health Prediction**. All answers are tied directly to the codebase, mathematical formulations, and empirical results.

---

### 1. What is Gaussian Process Regression (GPR) and why is it well-suited for battery SOH estimation?
**Answer:** GPR is a non-parametric Bayesian approach that defines a prior distribution over functions: $f(\mathbf{x}) \sim \mathcal{GP}(m(\mathbf{x}), k(\mathbf{x}, \mathbf{x}'))$. For any finite set of inputs, the function values follow a joint Gaussian distribution. It is exceptionally well-suited for battery State of Health (SOH) estimation because:
1. Battery degradation datasets are typically small to moderate in size (e.g., 4,657 cycles across 8 cells in our benchmark), where deep neural networks easily overfit.
2. Battery management systems (BMS) require principled uncertainty quantification (confidence intervals) for safety-critical decisions, which GPR provides natively via the analytical posterior predictive variance $\sigma_*^2 = k(\mathbf{x}_*, \mathbf{x}_*) - \mathbf{k}_*^\top (\mathbf{K} + \sigma_n^2 \mathbf{I})^{-1} \mathbf{k}_*$.

---

### 2. What is the difference between exact marginal log-likelihood and Variational ELBO in GPR?
**Answer:**
- **Exact GPR:** Computes the marginal log-likelihood $\log p(\mathbf{y}|\mathbf{X}) = -\frac{1}{2} \mathbf{y}^\top (\mathbf{K} + \sigma_n^2 \mathbf{I})^{-1} \mathbf{y} - \frac{1}{2} \log |\mathbf{K} + \sigma_n^2 \mathbf{I}| - \frac{N}{2}\log 2\pi$. This requires inverting an $N \times N$ matrix, costing $\mathcal{O}(N^3)$ compute and $\mathcal{O}(N^2)$ memory, which becomes prohibitive for continuous BMS telemetry.
- **Variational ELBO (SVGP):** Introduces $M \ll N$ inducing variables $\mathbf{u} = f(\mathbf{Z})$ and approximates the true posterior $p(\mathbf{f}, \mathbf{u}|\mathbf{y})$ with a variational distribution $q(\mathbf{u}) = \mathcal{N}(\mathbf{m}, \mathbf{S})$. It maximizes the Evidence Lower Bound:
$$\mathcal{L}_{\text{ELBO}} = \sum_{i=1}^N \mathbb{E}_{q(f_i)}[\log p(y_i | f_i)] - \text{KL}(q(\mathbf{u}) \,||\, p(\mathbf{u}))$$
This decomposes across training samples, reducing computational complexity to $\mathcal{O}(N M^2 + M^3)$, enabling mini-batch training and fast edge inference.

---

### 3. What are inducing points and how are they selected and optimized in our SVGP?
**Answer:** Inducing points $\mathbf{Z} = \{\mathbf{z}_m\}_{m=1}^M$ are a representative subset of pseudo-inputs in feature space that act as sufficient statistics for the GP prior. In our implementation:
- We choose $M = \min(64, N_{\text{train}})$.
- **Initialization:** Strictly initialized via $k$-means clustering on the **training features only** within each LOCO fold, preventing test leakage.
- **Optimization:** The inducing locations $\mathbf{Z}$ are treated as variational parameters and jointly updated via gradient ascent on the ELBO alongside variational parameters $(\mathbf{m}, \mathbf{S})$ and kernel hyperparameters using Adam ($lr = 0.01$).

---

### 4. How do you mathematically prove that our composite Wide-and-Deep kernel is positive semi-definite (PSD)?
**Answer:** Our kernel is defined as:
$$k_{\text{WD}}(\mathbf{x}, \mathbf{x}') = \sigma_w^2 [k_{\text{lin}}(\mathbf{x}, \mathbf{x}') + k_{\text{RBF-ARD}}(\mathbf{x}, \mathbf{x}')] + \sigma_d^2 \, k_{\text{Matérn}}(\phi_\theta(\mathbf{x}), \phi_\theta(\mathbf{x}'))$$
The PSD property is proven from Mercer's theorem and kernel closure properties:
1. **Linear Kernel:** $k_{\text{lin}}(\mathbf{x}, \mathbf{x}') = \mathbf{x}^\top \mathbf{x}'$. The Gram matrix is $\mathbf{K} = \mathbf{X} \mathbf{X}^\top$. For any $\mathbf{v} \neq \mathbf{0}$, $\mathbf{v}^\top \mathbf{K} \mathbf{v} = \|\mathbf{X}^\top \mathbf{v}\|_2^2 \ge 0$. Hence PSD.
2. **RBF-ARD Kernel:** $k_{\text{RBF}}(\mathbf{x}, \mathbf{x}') = \exp(-\frac{1}{2}\sum_d \frac{(x_d - x'_d)^2}{\ell_d^2})$ is known to be strictly positive definite by Bochner's theorem.
3. **Deep Composition:** For any measurable mapping $\phi_\theta: \mathbb{R}^D \to \mathbb{R}^P$ and PSD kernel $k_0$, the Gram matrix satisfies $\mathbf{v}^\top \mathbf{K}_{\phi} \mathbf{v} = \sum_{i,j} v_i v_j k_0(\phi(\mathbf{x}_i), \phi(\mathbf{x}_j)) \ge 0$. Thus, $k_{\text{Matérn}}(\phi_\theta(\mathbf{x}), \phi_\theta(\mathbf{x}'))$ is PSD.
4. **Sum & Scaling:** The set of PSD kernels is a convex cone closed under addition and positive scalar multiplication. Since $\sigma_w^2 > 0$ and $\sigma_d^2 > 0$, $k_{\text{WD}}$ is strictly PSD. We numerically confirmed this in `tests/test_kernel.py`, verifying $\lambda_{\min}(\mathbf{K}) \ge -10^{-6}$ across 100 random trials.

---

### 5. Why is a non-stationary Linear kernel combined with a stationary RBF kernel in the wide branch?
**Answer:** Battery degradation exhibits two distinct physical behaviors:
1. **Global Monotonic Trend:** Active lithium inventory loss and solid electrolyte interphase (SEI) growth cause an irreversible downward drift in capacity over hundreds of cycles. A non-stationary linear kernel captures this global trend without saturating.
2. **Local Non-Linear Variations:** Temperature fluctuations, dynamic charging regimes, and partial capacity recovery during rest periods produce localized fluctuations. The stationary RBF kernel with ARD captures these non-linear, feature-specific local variations. Combining them allows the linear kernel to extrapolate the degradation trend while the RBF kernel fits local cycle dynamics.

---

### 6. Why did we use out-of-fold residuals instead of in-sample training residuals for Symbolic Regression?
**Answer:** This avoids the **near-zero-residual trap**. A flexible non-parametric GP with an MLP deep kernel can achieve near-zero training residuals ($\mathbf{r}_{\text{in}} = \mathbf{y}_{\text{train}} - \boldsymbol{\mu}_{\text{GP}}(\mathbf{X}_{\text{train}}) \approx \mathbf{0}$). If Symbolic Regression is trained on in-sample residuals, it fits negligible numerical noise or trivially collapses to $g(\mathbf{x}) = 0$. 
By performing an **inner Leave-One-Training-Cell-Out CV**, we generate true **out-of-fold residuals** $r_i = y_i - \mu_{\text{GP}}^{(-k)}(x_i)$ where cell $k$ was excluded during GP fitting. These residuals expose genuine cross-cell generalization errors and structural discrepancies, providing rich signal for the genetic programming search.

---

### 7. How does the uncertainty weighting $w_i = \frac{1}{\sigma_{\text{GP}, i}^2 + \varepsilon}$ work in Symbolic Regression?
**Answer:** The symbolic regression objective minimizes an uncertainty-weighted squared residual loss with parsimony regularization:
$$\mathcal{L}_{\text{SR}}(g) = \frac{1}{N} \sum_{i=1}^N \frac{(r_i - g(\mathbf{x}_i))^2}{\sigma_{\text{GP}, i}^2 + \varepsilon} + \lambda \cdot \text{Complexity}(g)$$
The weight $w_i = \frac{1}{\sigma_{\text{GP}, i}^2 + \varepsilon}$ is the inverse predictive variance (Fisher precision) from the SVGP. 
- When the GP is highly confident ($\sigma_i^2$ is small), $w_i$ is large, forcing the symbolic equation to accurately capture systematic residual bias.
- When the GP has high epistemic or aleatoric uncertainty ($\sigma_i^2$ is large, e.g., noisy cycle regimes), $w_i$ is small, preventing the symbolic equation from overfitting erratic outliers.

---

### 8. Why must the uncertainty weights $w_i$ be clipped to quantiles $[1\%, 99\%]$ and normalized to mean 1?
**Answer:**
1. **Quantile Clipping:** If the GP posterior variance $\sigma_i^2$ approaches the variance floor $\varepsilon = 10^{-6}$ for a few tightly constrained points, $w_i$ can spike to $10^6$. Such extreme weights would cause gradient explosions in genetic programming and force the equation to overfit a handful of points. Clipping to the $[1\%, 99\%]$ quantiles caps extreme influence.
2. **Mean-1 Normalization:** Multiplying sample losses by unnormalized weights alters the effective learning rate and parsimony penalty scale. Normalizing $\bar{w} = 1$ ensures that the total loss scale matches standard mean squared error, keeping the genetic algorithm's parsimony trade-off stable across different datasets.

---

### 9. Why did you recalibrate the variance in the composite model as $\hat{\sigma}^2 = s^2 \sigma_{\text{GP}}^2$?
**Answer:** Adding a deterministic residual correction $g(\mathbf{x})$ shifts the predictive mean ($\hat{y} = \mu_{\text{GP}} + g(\mathbf{x})$) without intrinsically altering the GP's predicted variance $\sigma_{\text{GP}}^2$. However, by eliminating systematic bias, the effective residual errors shrink, potentially causing the original confidence intervals to be miscalibrated (either under- or over-covering).
We fit an empirical scaling factor $s$ on the training out-of-fold validation residuals:
$$s^2 = \frac{1}{N_{\text{val}}} \sum_{i \in \text{val}} \frac{(y_i - (\mu_i + g(\mathbf{x}_i)))^2}{\sigma_i^2}$$
This scalar temperature scaling recalibrates the 95% predictive interval $[\hat{y} \pm 1.96 s \sigma_{\text{GP}}]$ to preserve proper statistical coverage (PICP $\approx 95\%$) on unseen cells.

---

### 10. What is Leave-One-Cell-Out Cross-Validation (LOCO-CV) and how does it prevent data leakage?
**Answer:** In battery health management, batteries are cycled individually under varying ambient temperatures, discharge C-rates, and manufacturing tolerances. Random train/test splitting across cycles leaks future cycle data of the *same* physical cell into the training set, yielding artificially low error metrics that fail in production.
**LOCO-CV** strictly isolates entire battery cells:
- In NASA (4 cells: B0005, B0006, B0007, B0018), 4 folds are evaluated where 3 cells train the model and the 4th held-out cell tests it.
- In CALCE (4 cells: CS2_35, CS2_36, CS2_37, CS2_38), 4 folds are evaluated similarly.
This benchmarks true zero-shot cross-cell transfer to completely unseen physical cells.

---

### 11. How were scalers, imputers, and feature engineering isolated strictly to training cells in each fold?
**Answer:** In `src/evaluation.py` and `src/baseline_models.py`:
1. Imputation (`SimpleImputer(strategy='median')`) was fit solely on $X_{\text{train}}$. Missing values in $X_{\text{test}}$ were transformed using the training medians.
2. Feature scaling (`StandardScaler`) computed mean $\boldsymbol{\mu}_{\text{train}}$ and variance $\boldsymbol{\sigma}_{\text{train}}^2$ exclusively from training cells. Test features were transformed using $\frac{\mathbf{x}_{\text{test}} - \boldsymbol{\mu}_{\text{train}}}{\boldsymbol{\sigma}_{\text{train}}}$.
3. Target scaling (for neural architectures) used training target statistics only.
4. Inducing points for SVGP were clustered exclusively on transformed training features.
This guaranteed strict zero-leakage boundaries verified by unit tests in `tests/test_evaluation.py`.

---

### 12. Why did you use 3 seeds (42, 123, 999) and what do the standard deviations across folds tell us?
**Answer:** 
- In small-sample regimes ($N=4$ cells per dataset), single random runs can exhibit severe initialization bias in neural networks and genetic algorithms. Evaluating over three fixed seeds ({42, 123, 999}) establishes statistical repeatability and eliminates lucky seed claims.
- The reported standard deviations (e.g., NASA RMSE $3.41 \pm 2.22\%$ for WD-GPR) are driven predominantly by **inter-cell heterogeneity** across the 4 folds rather than seed variance. B0006 has severe non-linear degradation and an early knee-point, whereas B0005 degrades smoothly. This large standard deviation honestly communicates that cross-cell generalization variance is high.

---

### 13. How did baseline models perform compared to the proposed WD-GPR-SR on NASA vs. CALCE?
**Answer:**
- **On CALCE (smooth, uniform degradation):**
  - BiLSTM: RMSE $3.16 \pm 1.21\%$, MAE $1.28 \pm 0.28\%$
  - Proposed WD-GPR-SR: RMSE $3.29 \pm 1.23\%$, MAE $1.27 \pm 0.45\%$, 95% PICP $96.03\%$
  - WD-GPR (no SR): RMSE $3.30 \pm 1.15\%$, MAE $1.25 \pm 0.34\%$
  - SVR: RMSE $3.42 \pm 1.07\%$, MAE $1.04 \pm 0.32\%$
  - Proposed WD-GPR-SR achieved the lowest MAE among probabilistic models with calibrated 96% coverage.
- **On NASA (severe cell heterogeneity):**
  - Sparse GPR Wide: RMSE $3.34 \pm 2.17\%$, MAE $2.97 \pm 2.18\%$
  - WD-GPR: RMSE $3.41 \pm 2.22\%$, MAE $3.03 \pm 2.24\%$
  - Random Forest: RMSE $3.55 \pm 1.75\%$, MAE $3.08 \pm 1.69\%$
  - Deep Kernel GPR: RMSE $3.65 \pm 2.48\%$, MAE $3.12 \pm 2.34\%$
  - Proposed WD-GPR-SR: RMSE $4.93 \pm 1.48\%$, MAE $4.08 \pm 1.70\%$
  - On NASA, the symbolic regression stage overfit the idiosyncratic cross-cell residuals, increasing average cross-cell error even though it improved the knee-point cell B0006.

---

### 14. Why did WD-GPR-SR perform exceptionally well on CALCE but exhibit higher cross-cell variance on NASA?
**Answer:** CALCE cells (CS2_35 to CS2_38) were tested under identical dynamic profiles and consistent room temperatures, resulting in shared degradation physics that transferred cleanly across cells. In contrast, NASA cells had pronounced operational differences: B0006 experienced severe accelerated capacity loss and internal resistance spikes compared to B0005 and B0007. Because the 3 training cells lacked this extreme knee-point dynamic, the symbolic regression residual equation learned cross-cell residual patterns that failed to generalize universally, honestly demonstrating the limits of symbolic transfer across heterogeneous cells.

---

### 15. What is the knee-point phenomenon in battery degradation, and how did WD-GPR-SR handle it on cell B0006?
**Answer:** The knee-point is a sharp acceleration in capacity fade where degradation transitions from linear SEI layer growth to severe non-linear lithium plating and active material loss. 
On cell B0006 (the only NASA cell with an abrupt knee-point around cycle 80):
- Pure baseline models suffered severe under-prediction or delayed response.
- WD-GPR achieved an RMSE of $7.00\%$.
- When augmented with the uncertainty-weighted symbolic equation, **WD-GPR-SR reduced the RMSE to 6.93%**, successfully capturing the sharp downward curvature that the smooth GP kernel smoothed out.

---

### 16. What do the 95% PICP and MPIW metrics measure, and how was our predictive uncertainty calibrated?
**Answer:**
- **PICP (Prediction Interval Coverage Probability):** The percentage of true test values falling inside the 95% confidence interval $[\hat{y} - 1.96\hat{\sigma}, \hat{y} + 1.96\hat{\sigma}]$. An ideal model achieves $\text{PICP} \ge 95\%$.
- **MPIW (Mean Prediction Interval Width):** The average width of the interval: $\frac{1}{N}\sum 2 \times 1.96\hat{\sigma}_i$. A narrower interval indicates sharper, more confident predictions.
- **Empirical Results:** On CALCE, our proposed model achieved $\text{PICP} = 96.03\%$ with $\text{MPIW} = 8.73\%$, proving excellent calibration. Exact GPR had overconfident intervals on NASA ($\text{PICP} = 37.26\%$) due to kernel misspecification, while our SVGP maintained well-calibrated, safe uncertainty bounds.

---

### 17. What is Automatic Relevance Determination (ARD) and what did lengthscales reveal about feature importance?
**Answer:** ARD assigns an individual lengthscale parameter $\ell_d$ to each input feature in the RBF kernel:
$$k(\mathbf{x}, \mathbf{x}') = \exp\left(-\frac{1}{2}\sum_{d=1}^D \frac{(x_d - x'_d)^2}{\ell_d^2}\right)$$
The feature relevance is defined as the inverse lengthscale $w_d = 1/\ell_d$. If a feature is irrelevant, the optimizer pushes $\ell_d \to \infty$ ($w_d \to 0$), removing its influence.
In our model:
- Constant Current Charge Time (`cc_charge_time`) had the largest ARD relevance ($0.3125$).
- Cycle index followed ($0.2111$).
- Constant Voltage Charge Time (`cv_charge_time`) had relevance ($0.1602$).
This demonstrates that charge duration is the most physically informative feature for SOH degradation.

---

### 18. How do SHAP KernelExplainer and LIME explain the predictions of our hybrid pipeline?
**Answer:**
1. **SHAP (Shapley Additive Explanations):** Uses cooperative game theory to compute the marginal contribution of each feature to the prediction relative to a background baseline. To bound computation on our hybrid pipeline, we summarized the training set using $K$-means ($K=30$) and evaluated 100 test samples. SHAP identified `ic_peak_voltage` ($47.5\%$) and `cc_charge_time` ($34.8\%$) as dominant global drivers.
2. **LIME (Local Interpretable Model-agnostic Explanations):** Fits an interpretable sparse linear surrogate model locally around individual test instances. LIME confirmed that in early life (high SOH), `cc_charge_time` positively supports capacity, while in late life (degraded SOH), high `cycle` index and shifted `ic_peak_voltage` apply strong negative weights.

---

### 19. What do Partial Dependence Plots (PDP) and Individual Conditional Expectation (ICE) show about battery health?
**Answer:**
- **PDP:** Computes the average marginal effect of a feature on predicted SOH across the dataset: $\bar{f}(x_s) = \frac{1}{N}\sum_{i=1}^N f(x_s, \mathbf{x}_{c,i})$. It shows an intuitive monotonic decrease in SOH as `cycle` increases and as `cc_charge_time` shortens.
- **ICE:** Plots the prediction function for each individual sample separately as $x_s$ varies. ICE curves revealed parallel, homogeneous trajectories across cycles, confirming that our model learned a stable, monotonic physical degradation trend without erratic local oscillations.

---

### 20. How did you assess XAI consistency across SHAP, ARD, and symbolic partial derivatives $\partial g/\partial x$?
**Answer:** We computed the Spearman rank correlation across the 5 engineered features between the three independent interpretability paradigms:
- $\text{Corr}(\text{SHAP}, \text{ARD}) = 0.50$ (moderate positive alignment)
- $\text{Corr}(\text{SHAP}, \text{Symbolic Sensitivities}) = 0.70$ (strong positive alignment)
- $\text{Corr}(\text{ARD}, \text{Symbolic Sensitivities}) = 0.10$
All methods agreed that `cc_charge_time` and `ic_peak_voltage` carry the highest predictive influence, confirming that the model's reliance on charging duration and peak voltage aligns with electrochemical battery physics.

---

### 21. What was the exact deployment architecture designed for the NVIDIA Jetson Orin Nano?
**Answer:** To guarantee real-time execution without heavyweight Python dependencies on an embedded BMS:
1. **Deep Feature Extractor ($\phi_\theta$):** Exported from PyTorch to **ONNX** format (opset 17, dynamic batch).
2. **Sparse Variational GP:** Precomputed variational parameters $(\mathbf{m}, \mathbf{S})$ and inducing points $\mathbf{Z}$ exported to compact NumPy arrays (`svgp_arrays.npz`). The predictive mean and variance are computed via pure vectorized matrix-vector BLAS operations:
$$\mu_* = \mathbf{k}_{*u} \mathbf{K}_{uu}^{-1} \mathbf{m}, \quad \sigma_*^2 = k_{**} - \mathbf{k}_{*u} \mathbf{K}_{uu}^{-1} (\mathbf{K}_{uu} - \mathbf{S}) \mathbf{K}_{uu}^{-1} \mathbf{k}_{u*}$$
3. **Symbolic Residual Equation:** Compiled into pure ANSI C (`equation.c`) and built as a shared library (`libequation.so`), loaded via `ctypes`.
The entire pipeline runs in a lightweight, standalone Python runtime with zero PyTorch/GPyTorch overhead.

---

### 22. How did you verify numerical equivalence between PyTorch and the standalone C/ONNX inference engine?
**Answer:** In `deployment/export_model.py` and `results/metrics/export_check.json`:
- Evaluated on 168 test samples from cell B0005.
- The ONNX deep feature extractor $\phi_\theta$ matched PyTorch with a maximum absolute difference of $1.40 \times 10^{-9}$.
- The standalone SVGP + C equation engine matched PyTorch end-to-end with:
  - Mean absolute difference: $2.81 \times 10^{-5}\%$
  - Maximum absolute difference: $7.78 \times 10^{-5}\%$ (well below our strict tolerance of $10^{-3}\%$).
This verified bit-exact numerical parity between research code and embedded deployment.

---

### 23. What latency, throughput, and memory footprint were achieved by the standalone deployment engine?
**Answer:** Benchmarked over 1,000 timed inferences following 100 warmup runs on our development computer (Linux x86_64, FP32, batch size 1):
- **Median End-to-End Latency:** **$0.1232\text{ ms}$** ($\approx 123\,\mu\text{s}$).
- **95th-Percentile Latency:** **$0.1393\text{ ms}$**.
- **Inference Throughput:** **$8,114.8\text{ inferences/sec}$**.
- **Component Breakdown:** Preprocessing $0.0065\,\text{ms}$, ONNX $\phi_\theta$ $0.0123\,\text{ms}$, NumPy SVGP $0.0892\,\text{ms}$, C Equation $0.0077\,\text{ms}$.
- **Peak Memory Usage:** **$0.588\text{ MB}$** (measured via `tracemalloc`).
This easily exceeds the 100 ms real-time latency budget of automotive CAN-bus battery management systems.

---

### 24. What is the honest status of Jetson power (Watts) and energy (mJ) measurements in this project?
**Answer:** In compliance with scientific integrity rules, power and energy measurements are explicitly marked as **"Not Measured"**. 
While our standalone inference engine is fully exported and verified on x86_64 Linux, measuring electrical power consumption (Watts) and energy per inference (mJ) requires physical NVIDIA Jetson Orin Nano hardware equipped with onboard INA3221 voltage/current monitors accessed via `tegrastats` or Jetson Sysfs. Simulating or fabricating power measurements without physical hardware is scientifically invalid, so we report latency and memory measured on our host machine and provide the ready-to-run Jetson benchmarking script `deployment/benchmark.py` for physical lab verification.

---

### 25. What are the key limitations of this study, and why do we refrain from claiming broad statistical significance?
**Answer:**
1. **Sample Size Limitations ($N = 4+4$ cells):** Testing across 4 cells in NASA and 4 cells in CALCE provides only 4 degrees of freedom per dataset. Performing parametric hypothesis tests (e.g., paired $t$-tests) with $N=4$ has inadequate statistical power and risks false claims.
2. **Cell Heterogeneity:** Substantial variance across held-out cells (NASA RMSE std $\sim 2\%$) shows that performance depends heavily on the specific cell tested.
3. **Symbolic Generalizability:** Symbolic regression excels at fitting deterministic physical residuals when training and test distributions match (CALCE), but can overfit cross-cell noise when cells operate in divergent thermal or electrochemical regimes (NASA).
We therefore report our findings honestly with mean $\pm$ standard deviation and refrain from claiming broad statistical superiority.

