# Explainability & Feature Importance Analysis

## Executive Summary
This report analyzes feature importance, local lifecycle dynamics, and attribution consistency for the **WD-GPR-SR** framework on held-out lithium-ion battery cells. Note that feature attribution identifies statistical association within the trained surrogate model and is not identical to physical electrochemical causation.

---

## 1. Feature Attribution Summary Across Methods

| Feature | Physical Meaning | SHAP (Global) | ARD (1/ℓ) | LIME (Local Mean) | PDP / ICE Pair | Model E Symbolic |
|---|---|:---:|:---:|:---:|:---:|:---:|
| `cycle` | Degradation cycle index | 0.357 | 0.213 | 0.369 | 0.621 | 0.200 |
| `cc_charge_time` | CC charge duration | 0.364 | 0.299 | 0.370 | 0.281 | 0.200 |
| `cv_charge_time` | CV charge duration | 0.033 | 0.164 | 0.045 | 0.014 | 0.200 |
| `ic_peak_height` | IC peak height | 0.049 | 0.146 | 0.066 | 0.060 | 0.200 |
| `ic_peak_voltage` | IC peak voltage | 0.197 | 0.177 | 0.150 | 0.024 | 0.200 |

---

## 2. Key Physical Insights & Consistency Assessment

1. **Dominant Degradation Drivers:**
   -  and  consistently rank as the top two global degradation drivers across SHAP, ARD lengthscale relevance, and PDP/ICE curves. This is consistent with established battery degradation literature: constant-current charge duration contracts monotonically as internal ohmic resistance rises, while cumulative cycle count tracks macroscopic aging.
2. **Local LIME Lifecycle Dynamics:**
   - Early life predictions are driven primarily by .
   - Mid-to-late life predictions show increasing influence from polarization indicators ( and ).
3. **Symbolic Sensitivity:**
   - Under nested cross-validation, the proposed model E selected  = 0$ (no correction), resulting in identically zero symbolic sensitivities across all features.
4. **Attribution vs. Causation Disclosure:**
   - Feature attributions reflect surrogate model response surfaces and are consistent with electrochemical theory; however, attribution does not establish physical causation.
5. **Spearman Rank Correlations Across Active Methods:**
   - SHAP vs. ARD (1/ℓ): $ho = 0.90$ (strong agreement on top features).
   - SHAP vs. LIME: $ho = 1.00$.
   - SHAP vs. PDP/ICE: $ho = 0.80$.
   - ARD vs. PDP/ICE: $ho = 0.60$.
