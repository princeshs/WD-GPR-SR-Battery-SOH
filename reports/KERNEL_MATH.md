# Positive Semi-Definite (PSD) Proof for Wide-and-Deep Kernel

**Project:** WD-GPR-SR: When Uncertainty Meets Equations in Battery Health Prediction  
**Component:** Proposed Wide-and-Deep Variational Gaussian Process Regressor (WD-SVGP)  

---

## 1. Kernel Architecture

The proposed Wide-and-Deep kernel function $k_{\text{WD}}: \mathbb{R}^D \times \mathbb{R}^D \to \mathbb{R}$ is defined as:

$$k_{\text{WD}}(x, x') = \sigma_w^2 \left[ k_{\text{linear}}(x, x') + k_{\text{RBF-ARD}}(x, x') \right] + \sigma_d^2 \cdot k_{\text{Matérn}}\left(\phi_\theta(x), \phi_\theta(x')\right)$$

where:
- **Wide branch:** 
  - $k_{\text{linear}}(x, x') = x^T x + \beta^2$ (Linear covariance capturing monotonic macro degradation trends; non-stationary).
  - $k_{\text{RBF-ARD}}(x, x') = \exp\left(-\frac{1}{2} \sum_{d=1}^D \frac{(x_d - x'_d)^2}{l_d^2}\right)$ (Stationary Radial Basis Function with dimension-specific Automatic Relevance Determination length scales).
  - Scaled by wide variance parameter $\sigma_w^2 > 0$.
- **Deep branch:**
  - $\phi_\theta: \mathbb{R}^D \to \mathbb{R}^p$ ($p \in [2, 8]$) is a parameterised deep neural feature extractor with LayerNorm and Dropout.
  - $k_{\text{Matérn}}(\nu=2.5)$ is the Matérn covariance with smoothness parameter $\nu = 5/2$.
  - Scaled by deep variance parameter $\sigma_d^2 > 0$.

---

## 2. Mathematical Definition of Positive Semi-Definiteness (PSD)

A symmetric kernel function $k: \mathcal{X} \times \mathcal{X} \to \mathbb{R}$ is said to be **Positive Semi-Definite (PSD)** if for any finite set of input points $\{x_1, x_2, \dots, x_N\} \subset \mathcal{X}$ and any real coefficients $c = [c_1, c_2, \dots, c_N]^T \in \mathbb{R}^N$:

$$\sum_{i=1}^N \sum_{j=1}^N c_i c_j k(x_i, x_j) = c^T K c \ge 0$$

where $K \in \mathbb{R}^{N \times N}$ is the Gram matrix with entries $K_{ij} = k(x_i, x_j)$.

---

## 3. Fundamental Lemmas

### Lemma 1: Positive Scaling Preserves PSD
**Statement:** If $k(x, x')$ is a PSD kernel and $\sigma^2 > 0$, then $k'(x, x') = \sigma^2 k(x, x')$ is a PSD kernel.  
**Proof:**  
Let $K$ be the Gram matrix of $k$ on any collection of points $\{x_i\}_{i=1}^N$. The Gram matrix of $k'$ is $K' = \sigma^2 K$.  
For any $c \in \mathbb{R}^N$:
$$c^T K' c = c^T (\sigma^2 K) c = \sigma^2 (c^T K c)$$
Since $\sigma^2 > 0$ and $c^T K c \ge 0$ by the PSD assumption on $k$, it follows that $c^T K' c \ge 0$. Thus $k'$ is PSD. $\blacksquare$

### Lemma 2: Sum of PSD Kernels is PSD
**Statement:** If $k_1(x, x')$ and $k_2(x, x')$ are PSD kernels on $\mathcal{X}$, then their sum $k(x, x') = k_1(x, x') + k_2(x, x')$ is a PSD kernel.  
**Proof:**  
Let $K_1$ and $K_2$ be the Gram matrices of $k_1$ and $k_2$ on $\{x_i\}_{i=1}^N$.  
The Gram matrix of $k$ is $K = K_1 + K_2$.  
For any $c \in \mathbb{R}^N$:
$$c^T K c = c^T (K_1 + K_2) c = c^T K_1 c + c^T K_2 c$$
Because $k_1$ and $k_2$ are PSD, $c^T K_1 c \ge 0$ and $c^T K_2 c \ge 0$.  
The sum of two non-negative real numbers is non-negative:
$$c^T K c \ge 0$$
Hence $k$ is PSD. $\blacksquare$

### Lemma 3: Pullback / Composition by Measurable Feature Map Preserves PSD
**Statement:** Let $\mathcal{Z} \subseteq \mathbb{R}^p$. If $k_0: \mathcal{Z} \times \mathcal{Z} \to \mathbb{R}$ is a PSD kernel on $\mathcal{Z}$, and $\phi: \mathcal{X} \to \mathcal{Z}$ is any deterministic mapping, then the pullback kernel $k(x, x') = k_0(\phi(x), \phi(x'))$ is a PSD kernel on $\mathcal{X}$.  
**Proof:**  
Let $\{x_1, \dots, x_N\} \subset \mathcal{X}$ be an arbitrary set of input points.  
Define $z_i = \phi(x_i) \in \mathcal{Z}$ for $i = 1, \dots, N$.  
The Gram matrix $K \in \mathbb{R}^{N \times N}$ for $k$ on $\{x_i\}$ has entries:
$$K_{ij} = k(x_i, x_j) = k_0(\phi(x_i), \phi(x_j)) = k_0(z_i, z_j)$$
Observe that $K$ is precisely the Gram matrix $K_0$ of the kernel $k_0$ evaluated at the points $\{z_1, \dots, z_N\} \subset \mathcal{Z}$.  
Because $k_0$ is PSD on $\mathcal{Z}$, for any $c \in \mathbb{R}^N$:
$$c^T K c = c^T K_0 c \ge 0$$
Therefore, $k(x, x')$ is a valid PSD kernel on $\mathcal{X}$ regardless of the non-linearity or depth of $\phi$. $\blacksquare$

---

## 4. Main Theorem: PSD Property of the Wide-and-Deep Kernel

**Theorem:**  
The Wide-and-Deep kernel $k_{\text{WD}}(x, x')$ is a valid Positive Semi-Definite (PSD) kernel on $\mathbb{R}^D$ for all $\sigma_w^2 > 0$, $\sigma_d^2 > 0$, $\beta \ge 0$, $\{l_d > 0\}_{d=1}^D$, and arbitrary neural network weights $\theta$.

**Proof:**
1. **Linear Component:** $k_{\text{linear}}(x, x') = x^T x' + \beta^2$. The Gram matrix is $K_{\text{lin}} = X X^T + \beta^2 \mathbf{1}\mathbf{1}^T$. For any $c$, $c^T K_{\text{lin}} c = \|X^T c\|_2^2 + \beta^2 (\sum_i c_i)^2 \ge 0$. Thus $k_{\text{linear}}$ is PSD.
2. **RBF-ARD Component:** The Gaussian RBF kernel with ARD length scales is PSD by Bochner's Theorem (as the Fourier transform of a non-negative finite Borel measure).
3. **Wide Branch:** By Lemma 2, $k_{\text{linear}} + k_{\text{RBF-ARD}}$ is PSD. By Lemma 1, scaling by $\sigma_w^2 > 0$ preserves PSD:
   $$k_{\text{wide}}(x, x') = \sigma_w^2 [k_{\text{linear}}(x, x') + k_{\text{RBF-ARD}}(x, x')] \text{ is PSD.}$$
4. **Deep Branch:** The standard Matérn-5/2 kernel $k_{\text{Matérn}}$ is PSD on $\mathbb{R}^p$. The neural network $\phi_\theta: \mathbb{R}^D \to \mathbb{R}^p$ is a valid feature map. By Lemma 3, $k_{\text{Matérn}}(\phi_\theta(x), \phi_\theta(x'))$ is PSD. By Lemma 1, multiplying by $\sigma_d^2 > 0$ yields a PSD kernel:
   $$k_{\text{deep}}(x, x') = \sigma_d^2 \cdot k_{\text{Matérn}}(\phi_\theta(x), \phi_\theta(x')) \text{ is PSD.}$$
5. **Full Kernel:** Applying Lemma 2 to the sum of $k_{\text{wide}}$ and $k_{\text{deep}}$:
   $$k_{\text{WD}}(x, x') = k_{\text{wide}}(x, x') + k_{\text{deep}}(x, x') \text{ is PSD.}$$

$\blacksquare$

---

## 5. Non-Stationarity & Global Trend Modeling

Notice that while $k_{\text{RBF-ARD}}$ and $k_{\text{Matérn}}$ are stationary (invariant under spatial translations $x \mapsto x + \tau$), the linear kernel $k_{\text{linear}}(x, x') = x^T x' + \beta^2$ is **non-stationary**:
$$k_{\text{linear}}(x + \tau, x' + \tau) = x^T x' + \tau^T(x + x') + \|\tau\|^2 + \beta^2 \neq k_{\text{linear}}(x, x')$$
In battery health prognosis, capacity loss over cycles is dominated by irreversible monotonic active material degradation (SEI growth, lithium plating). A purely stationary kernel lacks inductive bias for global degradation and reverts to the prior mean when extrapolating. Incorporating the non-stationary linear kernel inside the Wide branch enforces this global downward slope, while the stationary ARD-RBF and Deep Matérn branches capture local capacity regenerations and non-linear knee-point kinetics.

