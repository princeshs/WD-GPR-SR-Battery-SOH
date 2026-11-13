# Uncertainty-Weighted Symbolic Regression: Mathematical Framework

**Project:** WD-GPR-SR: When Uncertainty Meets Equations in Battery Health Prediction  
**Component:** Stage 5 - Uncertainty-Weighted Symbolic Residual Discovery  

---

## 1. Problem Formulation & Hybrid Architecture

Let $x_i \in \mathbb{R}^D$ denote the pre-discharge battery health indicators for cycle $i$, and $y_i \in \mathbb{R}$ denote the true State of Health (SOH %).  
The proposed Wide-and-Deep Variational Gaussian Process (WD-SVGP) models the global degradation trajectory and yields an analytical predictive distribution:

$$y_i \sim \mathcal{N}\left(\mu_{\text{GP}}(x_i), \sigma_{\text{GP}}^2(x_i)\right)$$

where $\mu_{\text{GP}}(x_i)$ is the predictive mean and $\sigma_{\text{GP}}^2(x_i) = \mathbb{V}[f(x_i)] + \sigma_{\text{obs}}^2$ is the observation-noise-inclusive predictive variance.

The residual error $r_i$ represents the discrepancy between the true battery state and the GP prior:

$$r_i = y_i - \mu_{\text{GP}}(x_i)$$

Our goal is to discover a compact, interpretable mathematical function $g: \mathbb{R}^D \to \mathbb{R}$ via genetic programming such that the corrected prediction is:

$$\hat{y}(x) = \mu_{\text{GP}}(x) + g(x)$$

---

## 2. Objective Function with Uncertainty Weighting

Directly minimizing unweighted squared residuals treats all data points equally, exposing the symbolic search to overfit noisy or high-uncertainty cycles (such as localized sensor fluctuations).  
We formulate the **Uncertainty-Weighted Symbolic Regression** objective:

$$\mathcal{L}_{\text{SR}}(g) = \frac{1}{N} \sum_{i=1}^N \frac{\left(r_i - g(x_i)\right)^2}{\sigma_{\text{GP}}^2(x_i) + \epsilon} + \lambda \cdot \mathcal{C}(g)$$

where:
- $\sigma_{\text{GP}}^2(x_i) + \epsilon$ acts as a sample-specific variance regularizer ($\epsilon = 10^{-4}$ prevents division by zero).
- $\mathcal{C}(g)$ measures the structural complexity (node count / syntax tree size) of candidate equation $g$.
- $\lambda > 0$ is the parsimony coefficient penalizing over-parameterized equations (Occam's razor).

### Equivalent Sample Weights
The objective can be written in standard weighted mean squared error form:

$$\mathcal{L}_{\text{SR}}(g) = \frac{1}{N} \sum_{i=1}^N w_i \left(r_i - g(x_i)\right)^2 + \lambda \cdot \mathcal{C}(g)$$

The sample weights $w_i$ are computed via:
1. **Inverse Variance Weighting:**
   $$w_i^{\text{raw}} = \frac{1}{\sigma_{\text{GP}}^2(x_i) + \epsilon}$$
2. **Quantile Clipping:** To prevent a few highly confident points from dominating the genetic tournament selection, weights are clipped to the $[1\%, 99\%]$ empirical quantiles:
   $$w_i^{\text{clip}} = \min\left(\max\left(w_i^{\text{raw}}, q_{0.01}\right), q_{0.99}\right)$$
3. **Mean-1 Normalization:**
   $$w_i = \frac{w_i^{\text{clip}}}{\frac{1}{N}\sum_{j=1}^N w_j^{\text{clip}}} \quad \implies \quad \frac{1}{N} \sum_{i=1}^N w_i = 1$$

### Physical Meaning
- **Low Uncertainty, High Residual:** If $\sigma_{\text{GP}}^2(x_i)$ is small but $|r_i| > 0$, the GP is overly confident in an erroneous prediction (systematic bias). The weight $w_i \gg 1$ forces the symbolic regressor to find a physical correction term for this region.
- **High Uncertainty:** If $\sigma_{\text{GP}}^2(x_i)$ is large (e.g. knee-point transition or unobserved degradation state), $w_i \ll 1$. The symbolic regressor is discouraged from fitting high-order polynomials to stochastic noise.

---

## 3. Strict Leakage Prevention & The Near-Zero-Residual Trap

### The Trap
If $r_i$ is computed in-sample on the training data of an SVGP, the GP's flexible variational distribution interpolates the training points, resulting in $r_i^{\text{in-sample}} \approx 0$. Symbolic regression trained on near-zero noise collapses to the trivial zero function $g(x) \equiv 0$ and fails entirely when tested out-of-fold.

### The Inner LOCO Solution
To expose the true generalization errors to the symbolic regressor without leaking the outer held-out cell:
1. **Outer LOCO:** Let cell $c_{\text{test}}$ be the held-out test cell, and $\mathcal{D}_{\text{train}} = \{c_1, c_2, c_3\}$ be the remaining training cells.
2. **Inner LOCO:** For each training cell $k \in \{1, 2, 3\}$:
   - Fit a WD-SVGP model on the remaining 2 training cells: $\mathcal{D}_{\text{inner}} = \mathcal{D}_{\text{train}} \setminus \{c_k\}$.
   - Predict out-of-fold mean $\mu_{\text{OOF}}(x_i)$ and variance $\sigma_{\text{OOF}}^2(x_i)$ for all cycles $x_i \in c_k$.
   - Compute out-of-fold residual: $r_i = y_i - \mu_{\text{OOF}}(x_i)$.
3. **Pooled SR Training:** Pool the genuine out-of-fold residuals $\{ (x_i, r_i, w_i) \}_{i \in \mathcal{D}_{\text{train}}}$.
4. **Genetic Search:** Fit the symbolic regressor $g$ on the pooled out-of-fold residuals using weights $w_i$.
5. **Final GP Fitting:** Train the primary WD-SVGP model on all cells in $\mathcal{D}_{\text{train}}$.
6. **Held-Out Evaluation:** Predict cell $c_{\text{test}}$ via:
   $$\hat{y}_{\text{test}}(x) = \mu_{\text{GP, full}}(x) + g(x)$$
   The held-out cell $c_{\text{test}}$ is **never seen** by the GP, the residuals, the weights, or the symbolic search.

---

## 4. Parsimony, Complexity & Search Constraints

To ensure fast convergence and human-interpretable battery degradation formulas:
- **Operator Set:** Binary operators $\{+, -, \times, \div\}$ (protected division with $\div(a, b) = 1$ if $|b| < 10^{-6}$), unary operators $\{\text{square}, \text{sqrt}, \log, \text{exp}\}$.
- **Maximum Tree Size:** Max syntax tree length $\le 15$ nodes to prevent bloat.
- **Parsimony Penalty:** Parsimony coefficient $\lambda = 0.005$ dynamically penalizes tree size during tournament selection.
- **Ablation Comparison:** An unweighted version ($w_i = 1$) is trained identically to evaluate the direct empirical benefit of uncertainty-weighted loss.

---

## 5. Post-Correction Uncertainty Metrics

Because the symbolic equation $g(x)$ is a deterministic mean correction, the predictive variance $\sigma_{\text{GP}}^2(x)$ remains governed by the GP:

$$y \sim \mathcal{N}\left(\hat{y}(x), \sigma_{\text{GP}}^2(x)\right) = \mathcal{N}\left(\mu_{\text{GP}}(x) + g(x), \sigma_{\text{GP}}^2(x)\right)$$

Negative Log-Likelihood (NLL), 95% Prediction Interval Coverage Probability (PICP), and Mean Prediction Interval Width (MPIW) are re-evaluated using $\hat{y}(x)$ to rigorously determine whether mean residual correction improves or degrades uncertainty calibration.

