# Statistical Limitations and Small-Sample Caveats

**Project:** WD-GPR-SR: When Uncertainty Meets Equations in Battery Health Prediction  
**Stage:** 6 - Full Evaluation & Ablation Analysis  

---

## 1. Small Sample Size ($N=4+4$ Cells) and Statistical Power

A central methodological limitation of battery prognostic benchmarks (including NASA Ames and CALCE CS2) is the small number of distinct physical cells available:
- **NASA Ames:** 4 commercial 18650 cells (`B0005`, `B0006`, `B0007`, `B0018`).
- **CALCE CS2:** 4 prismatic $\text{LiCoO}_2$ cells (`CS2_35`, `CS2_36`, `CS2_37`, `CS2_38`).

### Implications:
1. **Low Statistical Power:** With $K = 4$ cross-validation folds per dataset, paired difference tests (such as Student's paired $t$-test or Wilcoxon signed-rank test) suffer from severely degraded statistical power. Small sample variations between single cells can dominate aggregate ranking.
2. **Absence of Significance Claims:** In strict accordance with rigorous scientific reporting standards, **we make no claims of formal "statistical significance" ($p < 0.05$)**. We report empirical means, sample standard deviations over seeds and folds, and honest confidence intervals.

---

## 2. High Inter-Fold Variance & Cell Heterogeneity

Battery degradation is fundamentally stochastic, governed by microstructural variations in electrode porosity, active material loading, and solid electrolyte interphase (SEI) growth. This manifests as substantial variance across held-out cells:

### NASA Cell Heterogeneity:
- Cells `B0005` and `B0007` exhibit steady, gradual capacity fade, allowing the Wide-and-Deep GP to achieve low prediction error ($\text{MAE} \approx 1.8\%$).
- Cell `B0006` exhibits accelerated degradation with significant capacity recovery spikes after prolonged resting periods ($\text{MAE} \approx 6.6\%$).
- Cell `B0018` was cycled with a shorter discharge cutoff (reaching 80% SOH within ~132 cycles vs ~168 cycles for B0005). Extrapolating to `B0018` presents an out-of-distribution knee point, causing higher predictive variance.
- Result: Standard deviations across LOCO folds on NASA are large ($\sigma \approx 1.5\% - 2.5\%$), reflecting true physical divergence rather than modeling instability.

### CALCE Cell Heterogeneity:
- CALCE cells were cycled under tightly controlled continuous laboratory conditions without thermal cycling.
- All four cells display nearly identical monotonic fade curves, resulting in much tighter LOCO variance ($\sigma \approx 0.3\% - 0.5\%$) and lower overall MAE ($\approx 1.2\%$).

---

## 3. Dataset Incompatibility & Cross-Domain Challenges

| Property | NASA Ames (18650) | CALCE CS2 (Prismatic) |
|---|---|---|
| **Cathode Chemistry** | $\text{LiCoO}_2$ | $\text{LiCoO}_2$ |
| **Form Factor** | Cylindrical 18650 | Prismatic pouch/can |
| **Nominal Capacity** | $2.0\text{ Ah}$ | $1.1\text{ Ah}$ |
| **Nominal CC Current** | $1.5\text{ A}$ ($0.75\text{ C}$) | $0.55\text{ A}$ ($0.5\text{ C}$) |
| **Cycle Life** | 132 – 168 cycles | 800 – 1,200 cycles |
| **Temperature Logged?** | Yes (thermocouple) | No (missing, set to NaN) |
| **Degradation Pattern** | Non-monotonic, knee + spikes | Monotonic smooth decline |

Because operating currents, form factors, and cycle lifespans differ by orders of magnitude (168 vs 1,200 cycles), naive pooling without dataset-specific scaling risks conflating operational regimes. Consequently, primary evaluation must be performed within-dataset, with pooled multi-dataset evaluation presented strictly as a secondary analysis.

---

## 4. Symbolic Residual Transferability & Overfitting Risk

The Stage 5 empirical results revealed an essential scientific insight regarding symbolic regression applied to battery health residuals:
1. **The Overfitting Penalty on Unseen Cells:** When an out-of-fold GP residual $r_i = y_i - \mu_{\text{GP}}(x_i)$ is fit with unconstrained symbolic regression, the discovered polynomial or rational terms can fit specific artifacts of the training cells that fail to generalize to a held-out cell with distinct knee-point kinetics.
2. **Mitigation via Uncertainty Weighting:** Weighting samples by $w_i = 1 / (\sigma_{\text{GP}}^2(x_i) + \epsilon)$ prevents the genetic algorithm from fitting noisy high-uncertainty regions. While unweighted SR substantially degrades transfer error on unseen cells, uncertainty-weighted SR dampens this degradation.
3. **Honest Reporting:** When the base GP already captures degradation with high precision (e.g. CALCE MAE $\approx 1.18\%$), the residuals are dominated by Gaussian white noise, where no meaningful functional correction exists beyond a scalar DC offset.

