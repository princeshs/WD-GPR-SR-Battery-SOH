# Feature Leakage & Inference Validity Analysis

**Date:** 2026-10-31  
**Project:** WD-GPR-SR (Battery State of Health Prediction)  
**Target Variable:** $\text{SOH}(\%) = \frac{Q_{\text{discharge}}}{C_{\text{ref}}} \times 100$

---

## 1. Executive Summary & Problem Formulation

In machine learning for battery health prognosis, **data leakage** occurs whenever information that is strictly unavailable prior to battery discharge—or information that tautologically computes the target variable—is fed into the model during training.

### The Tautological Leakage Trap
The State of Health (SOH) is defined directly by the usable discharge capacity $Q_{\text{discharge}}$:
$$Q_{\text{discharge}} = \int_0^{t_{\text{dis}}} |I_{\text{discharge}}(\tau)| \, d\tau \approx I_{\text{mean}} \times t_{\text{discharge}}$$
Under constant current (CC) discharge protocols (e.g. 2.0 A for NASA, 1.1 A for CALCE):
$$\text{SOH} \approx \frac{I_{\text{constant}} \times t_{\text{discharge}}}{C_{\text{ref}}} \times 100$$
If any feature derived directly from current-cycle discharge duration or total discharged charge ($t_{\text{dis}} \times I_{\text{dis}}$) is provided to an ML model, the regression task collapses to an elementary arithmetic division ($y = a \cdot x + b$). While such a model will achieve an artificial $R^2 \approx 0.999$, it is completely useless in practical BMS deployment because the vehicle operator cannot know the discharge duration until the battery is already fully depleted!

---

## 2. Comprehensive Feature Leakage Audit Table

| Feature Name | Signal Source | Physical Description | Available at Inference Time? | Leakage Classification | Inclusion Decision for Modeling |
|---|---|---|:---:|:---:|:---:|
| `cycle` | Cycler log | Cumulative cycle counter ($1, 2, \dots$) | **Yes** (Known prior to trip) | **Inference-Valid** | **Included in `FEATURES_MAIN`** |
| `cc_charge_time` | Charge phase | Duration of constant-current charging (s) | **Yes** (Measured during charging) | **Inference-Valid** | **Included in `FEATURES_MAIN`** |
| `cv_charge_time` | Charge phase | Duration of constant-voltage taper (s) | **Yes** (Measured during charging) | **Inference-Valid** | **Included in `FEATURES_MAIN`** |
| `ic_peak_height` | Charge phase | Peak height of $dQ/dV$ incremental capacity curve | **Yes** (Extracted from charge curve) | **Inference-Valid** | **Included in `FEATURES_MAIN`** |
| `ic_peak_voltage` | Charge phase | Voltage coordinate of $dQ/dV$ peak (V) | **Yes** (Extracted from charge curve) | **Inference-Valid** | **Included in `FEATURES_MAIN`** |
| `temp_charge_mean` | Charge phase | Mean cell temperature during charging (°C) | **Yes** (Where thermocouple logged) | **Inference-Valid (NASA only)** | Included in thermal ablations; NaN for CALCE |
| `temp_charge_max` | Charge phase | Maximum cell temperature during charging (°C) | **Yes** (Where thermocouple logged) | **Inference-Valid (NASA only)** | Included in thermal ablations; NaN for CALCE |
| `temp_charge_rise` | Charge phase | $\Delta T = T_{\max} - T_{\text{initial}}$ during charging (°C) | **Yes** (Where thermocouple logged) | **Inference-Valid (NASA only)** | Included in thermal ablations; NaN for CALCE |
| `discharge_time` | Discharge phase | Total time elapsed during discharge (s) | **No** (Requires full depletion) | **Tautological Leakage** | **EXCLUDED** from prospective SOH models |
| `discharge_current_mean` | Discharge phase | Mean discharge current magnitude (A) | **No** (Measured during driving/load) | **Use With Caution** | Excluded from primary inference set |
| `discharge_current_max` | Discharge phase | Peak discharge current magnitude (A) | **No** (Measured during driving/load) | **Use With Caution** | Excluded from primary inference set |
| `voltage_plateau_duration`| Discharge phase | Time spent within $[V_{\text{low}}, V_{\text{high}}]$ | **No** (Requires mid-to-late discharge) | **Use With Caution** | Diagnostic analysis only; excluded from `FEATURES_MAIN` |
| `voltage_mean` | Discharge phase | Mean discharge voltage (V) | **No** (Computed post-discharge) | **Use With Caution** | Diagnostic analysis only |
| `voltage_std` | Discharge phase | Standard deviation of discharge voltage (V) | **No** (Computed post-discharge) | **Use With Caution** | Diagnostic analysis only |
| `voltage_skew` | Discharge phase | Skewness of discharge voltage curve | **No** (Computed post-discharge) | **Use With Caution** | Diagnostic analysis only |
| `voltage_min` | Discharge phase | Minimum cut-off voltage reached (V) | **No** (Known only at end of discharge)| **Use With Caution** | Diagnostic analysis only |
| `voltage_slope` | Discharge phase | Linear slope $dV/dt$ across active discharge | **No** (Requires full discharge profile)| **Use With Caution** | Diagnostic analysis only |
| `temp_discharge_rise` | Discharge phase | $\Delta T$ rise during DISCHARGE phase (°C) | **No** (Measured post-discharge) | **Use With Caution** | Diagnostic analysis only |
| `capacity` | Target | Measured discharge capacity (Ah) | **No** (Ground truth target) | **STRICT TARGET** | **NEVER an input** |
| `SOH` | Target | $\frac{\text{capacity}}{C_{\text{ref}}} \times 100$ (%) | **No** (Ground truth target) | **STRICT TARGET** | **Target variable $y$** |

---

## 3. Operational Policy & Configuration Mapping

### Primary Feature Set (`FEATURES_MAIN`)
Configured in `src/config.py: FEATURES_MAIN`:
```python
FEATURES_MAIN = [
    "cycle",
    "cc_charge_time",
    "cv_charge_time",
    "ic_peak_height",
    "ic_peak_voltage",
]
```
These 5 features represent a battery health estimation paradigm where:
1. The EV or storage pack completes its charging cycle (overnight or at a depot).
2. The onboard BMS logs the charging voltage/current trajectory.
3. The feature extractor derives the CC time, CV time, and incremental capacity (IC) peak height and voltage.
4. The model predicts the available SOH and usable capacity **BEFORE the battery enters operation**.

### Cross-Dataset Compatibility
Because CALCE CS2 does not log temperature, `FEATURES_MAIN` is 100% compatible and non-null across both NASA Ames and CALCE datasets. Temperature features (`FEATURES_NASA_TEMP`) are preserved in the processed tables for NASA-specific thermal sensitivity analysis and ablation studies.

