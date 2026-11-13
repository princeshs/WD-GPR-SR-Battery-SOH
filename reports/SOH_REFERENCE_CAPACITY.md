# Battery State of Health (SOH) Reference Capacity Determination

**Date:** 2026-10-31  
**Project:** WD-GPR-SR (Battery State of Health Prediction)  
**Standard:** $\text{SOH}(\%) = \frac{C_{\text{discharge}}}{C_{\text{ref}}} \times 100$

---

## 1. Terminology: Distinguishing Battery Capacity Metrics

In battery prognostic literature, three distinct capacity measures are often conflated:

1. **Nominal Capacity ($C_{\text{nom}}$):**
   - The manufacturer's nominal rating stamped on the cell datasheet (e.g., 2.000 Ah for NASA 18650 Li-ion cells, 1.100 Ah for CALCE CS2 prismatic cells).
   - **Limitation:** Cell-to-cell manufacturing tolerances, chemistry variations, and test conditions (e.g., C-rates, cut-off voltages) cause actual fresh cell capacity to deviate by 5–15% from $C_{\text{nom}}$.
2. **Initial Capacity ($C_1$):**
   - The capacity measured during cycle 1 of the test profile.
   - **Limitation:** The first cycle frequently exhibits transient artifacts due to transport history, storage self-discharge, cell relaxation, or test setup equilibration, making $C_1$ vulnerable to single-point measurement distortion.
3. **Reference Capacity ($C_{\text{ref}}$):**
   - The empirically validated baseline capacity representing the true initial usable capacity of the cell under the specific cycling protocol.
   - Defined rigorously as:
     $$C_{\text{ref}} = \text{median}(C_1, C_2, C_3)$$
     computed across the first 3 valid discharge cycles.

---

## 2. Quantitative Cell-by-Cell Comparison

| Cell ID | Dataset | Cell Chemistry / Form Factor | Nominal $C_{\text{nom}}$ (Ah) | Initial $C_1$ (Ah) | Reference $C_{\text{ref}}$ (Ah) | Initial SOH using $C_{\text{nom}}$ (%) | Initial SOH using $C_{\text{ref}}$ (%) |
|---|---|---|:---:|:---:|:---:|:---:|:---:|
| **B0005** | NASA | LiCoO2 / 18650 cylindrical | 2.0000 | 1.8565 | **1.8463** | 92.82% | 100.55% |
| **B0006** | NASA | LiCoO2 / 18650 cylindrical | 2.0000 | 2.0353 | **2.0251** | 101.77% | 100.50% |
| **B0007** | NASA | LiCoO2 / 18650 cylindrical | 2.0000 | 1.8911 | **1.8807** | 94.55% | 100.55% |
| **B0018** | NASA | LiCoO2 / 18650 cylindrical | 2.0000 | 1.8550 | **1.8432** | 92.75% | 100.64% |
| **CS2_35** | CALCE | LiCoO2 / Prismatic | 1.1000 | 1.1385 | **1.1262** | 103.50% | 101.09% |
| **CS2_36** | CALCE | LiCoO2 / Prismatic | 1.1000 | 1.1408 | **1.1334** | 103.71% | 100.65% |
| **CS2_37** | CALCE | LiCoO2 / Prismatic | 1.1000 | 1.1311 | **1.1240** | 102.83% | 100.63% |
| **CS2_38** | CALCE | LiCoO2 / Prismatic | 1.1000 | 1.1332 | **1.1266** | 103.02% | 100.59% |

---

## 3. Justification for Using $C_{\text{ref}} = \text{median}(C_1, C_2, C_3)$

1. **Avoids Severe Baseline Distortion:**
   - If nominal capacity $C_{\text{nom}} = 2.0\text{ Ah}$ were used for NASA cell B0005, its initial SOH would be calculated as 92.8%. If the End-of-Life (EOL) threshold is defined at 70% or 80% of rated capacity, cell B0005 would appear to have lost over 7% of its life before the test even began!
   - Similarly, CALCE cells would begin at >103% SOH under $C_{\text{nom}} = 1.1\text{ Ah}$.
2. **Robustness to Transient Cycle 1 Noise:**
   - In real laboratory testing, cycle 1 may experience incomplete initial state-of-charge alignment. Taking the median across cycles 1, 2, and 3 completely eliminates susceptibility to a single abnormal cycle while reflecting the actual fresh cell capacity.
3. **Cross-Cell Comparability:**
   - Anchoring all cells to their respective empirical $C_{\text{ref}}$ ensures that SOH starts uniformly near 100.0% ($\pm 0.6\%$). This aligns with standard industrial battery management system (BMS) practices and facilitates direct cross-cell model generalization and transfer learning.

