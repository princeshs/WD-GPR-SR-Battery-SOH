# Project Inspection Report: Battery Datasets & Baseline Architecture

**Date:** 2026-10-31  
**Project:** WD-GPR-SR (Battery State of Health Prediction with Uncertainty Quantification and Symbolic Regression)  
**Scope:** NASA Ames Prognostics Data (B0005, B0006, B0007, B0018), CALCE Battery Data (CS2_35, CS2_36, CS2_37, CS2_38), Existing Notebooks, and Reference Paper.

---

## 1. Directory Structure Overview

```text
ML_Project/
├── A_Data-Driven_Approach_With_Uncertainty_Quantifica.pdf  # Reference paper (Liu et al., IEEE TIE 2021)
├── PBL.pdf                                                # Project requirements and evaluation guidelines
├── B0005.mat, B0006.mat, B0007.mat, B0018.mat             # NASA Ames raw MATLAB files
├── BatteryDatasetImplementation-master/                   # Reference NASA notebook & helper scripts
│   └── Battery.ipynb
├── NASA-Battery-Dataset-main/                             # Reference extraction notebook
│   └── NASA_Battery_Data_Extraction.ipynb
├── CALCE-main/                                            # CALCE raw dataset & baseline notebooks
│   ├── dataset/
│   │   ├── CS2_35/ (25 .xlsx files)
│   │   ├── CS2_36/ (26 .xlsx files)
│   │   ├── CS2_37/ (27 .xlsx files)
│   │   ├── CS2_38/ (28 .xlsx files)
│   │   └── CALCE.npy (pre-extracted summary dictionary)
│   ├── CALCE.ipynb, MLP.ipynb, RNN & LSTM - CALCE.ipynb, Gaussian fitting.ipynb
├── WD-GPR-SR-Implementation/                              # Core implementation workspace
│   ├── PROGRESS.md
│   ├── reports/ -> ../reports
│   ├── src/ -> ../src
│   ├── data/ -> ../data
│   └── results/ -> ../results
```

---

## 2. Dataset Formats & Inspection Details

### 2.1 NASA Ames Prognostics Center Li-ion Dataset
- **File Format:** MATLAB `.mat` struct (v7 format).
- **Cells & Nominal Capacity:** Commercial 18650 Li-ion cells, Nominal Capacity = **2.0 Ah**.
  - Rated voltage: 4.2 V cut-off charge, 2.7 V cut-off discharge (2.5 V for B0018).
  - Ambient temperature: ~24 °C constant chamber.
- **Top-Level Struct Keys:** `['__header__', '__version__', '__globals__', '<cell_id>']`.
- **Cycle Counts by Operation Type:**
  - **B0005:** 616 total cycles (170 charge, 168 discharge, 278 impedance).
  - **B0006:** 616 total cycles (170 charge, 168 discharge, 278 impedance).
  - **B0007:** 616 total cycles (170 charge, 168 discharge, 278 impedance).
  - **B0018:** 319 total cycles (134 charge, 132 discharge, 53 impedance).
- **Discharge Cycle Struct & Array Shapes:**
  - `fields`: `Voltage_measured`, `Current_measured`, `Temperature_measured`, `Current_load`, `Voltage_load`, `Time`, `Capacity`
  - Array shapes (e.g. B0005 cycle 1): Time series vectors are `(1, 197)` (or up to `(1, 366)` for B0018); `Capacity` is scalar `(1, 1)` in Ah.
  - Measured initial discharge capacities: B0005 = 1.8565 Ah, B0006 = 2.0353 Ah, B0007 = 1.8911 Ah, B0018 = 1.8550 Ah.

### 2.2 CALCE (Center for Advanced Life Cycle Engineering) Dataset
- **File Format:** Microsoft Excel OpenXML (`.xlsx`).
- **Cells & Nominal Capacity:** Prismatic LiCoO2/graphite cells (CS2 series), Nominal Capacity = **1.1 Ah**.
  - CC-CV charging: 0.55 A (0.5 C) to 4.2 V, hold at 4.2 V until current drops to 0.05 A.
  - CC discharge: 1.10 A (1.0 C) to 2.7 V.
- **File Counts per Cell:**
  - **CS2_35:** 25 files (936 raw cycle segments).
  - **CS2_36:** 26 files (976 raw cycle segments).
  - **CS2_37:** 27 files (1043 raw cycle segments).
  - **CS2_38:** 28 files (1082 raw cycle segments).
- **Sheet Architecture:**
  - Sheet 0: `Info` (metadata).
  - Sheet 1: `Channel_1-008` (CS2_35), `Channel_1-009` (CS2_36), `Channel_1-010` (CS2_37), `Channel_1-011` (CS2_38).
  - Sheet 2 (optional): `Statistics_1-...`.
- **Columns & Units:**
  - `Data_Point` (integer index)
  - `Test_Time(s)`, `Step_Time(s)` (seconds)
  - `Date_Time` (ISO timestamp)
  - `Step_Index` (step identifier: 2=CC charge, 4=CV charge, 7=CC discharge)
  - `Cycle_Index` (local test cycle number)
  - `Current(A)` (Amperes)
  - `Voltage(V)` (Volts)
  - `Charge_Capacity(Ah)`, `Discharge_Capacity(Ah)` (Ampere-hours)
  - `Internal_Resistance(Ohm)` (Ohms)
- **Cycle Offset & Concatenation Requirement:**
  - **YES.** Each `.xlsx` file resets `Cycle_Index` starting from 1.
  - Files MUST be sorted chronologically using the first `Date_Time` timestamp in each file (alphabetical filename sorting incorrectly places October before August).
  - Cumulative cycle indices ($1, 2, \dots, N$) must be assigned across files.

---

## 3. Existing Notebooks Audit

1. `BatteryDatasetImplementation-master/Battery.ipynb`: Parses NASA .mat files for B0005/B0006, extracts capacity, normalizes SOH relative to cycle 1, and trains a basic MLP/LSTM regression model for remaining life.
2. `NASA-Battery-Dataset-main/NASA_Battery_Data_Extraction.ipynb`: Implements an interactive file extraction utility with `ipyfilechooser` to convert NASA `.mat` discharge profiles into tabular DataFrames.
3. `CALCE-main/CALCE.ipynb`: Traverses CALCE Excel files, computes CCCT, CVCT, and discharge capacity, applies a 40-step rolling 2-$\sigma$ outlier filter, and saves the output to `CALCE.npy`.
4. `CALCE-main/MLP.ipynb`: Loads `CALCE.npy` and trains a multi-layer perceptron in PyTorch to predict capacity degradation.
5. `CALCE-main/RNN & LSTM - CALCE.ipynb`: Implements standard RNN and LSTM sequence models in PyTorch on CALCE capacity series.
6. `CALCE-main/Gaussian fitting.ipynb`: Evaluates Gaussian process and parametric Gaussian curve fitting for degradation trajectory smoothing on CALCE batteries.

---

## 4. Reference Paper Summary (Liu et al., IEEE TIE 2021)

> **Paper:** *A Data-Driven Approach With Uncertainty Quantification for Predicting Future Capacities and Remaining Useful Life of Lithium-ion Battery*, IEEE Trans. Ind. Electron., 68(4):3170–3180, 2021.
>
> 1. Utilizes Empirical Mode Decomposition (EMD) to decompose non-monotonic battery capacity degradation into intrinsic mode functions (IMFs) and a monotonic residual trend.
> 2. Deploys an LSTM neural network on the residual trend to capture long-term irreversible aging dynamics.
> 3. Employs Gaussian Process Regression (GPR) to model the high-frequency IMFs, which capture capacity regeneration phenomena following rest periods.
> 4. GPR naturally provides predictive variance, enabling principled uncertainty quantification and prediction intervals for future capacity.
> 5. Computes RUL as the cycle where predicted capacity drops below the 80% or 70% end-of-life threshold.
> 6. Validates the hybrid LSTM+GPR model against solo GPR, solo LSTM, and EMD baselines on NASA (B0005, B0006, B0007, B0018) and CALCE (CS2_35–38).

---

## 5. Critical Issues & Data Engineering Decisions

1. **Temperature Availability:** NASA includes high-resolution thermocouple temperature data (`Temperature_measured`), whereas CALCE CS2 does not log cell temperature. The loader must treat temperature as optional/nullable.
2. **Cycle Index Reset in CALCE:** Without chronological timestamp sorting and cumulative cycle indexing, cycles from different test runs overlap and corrupt temporal ordering.
3. **Flawed Outlier Handling in Existing Code:** Previous scripts used `range_ = np.arange(1, count, bins)` in `drop_outlier`, silently dropping cycle 0 and the entire final chunk, and discarding rows rather than flagging them. Stage 1 adheres to the strict requirement: **FLAG outliers in a dedicated column, never delete them silently.**
4. **Reference Capacity ($C_{\text{ref}}$) Selection:** Initial cycle 1 is often an unconditioned or formation cycle. Nominal capacity (2.0 Ah or 1.1 Ah) differs from actual measured fresh capacity. $C_{\text{ref}}$ is rigorously defined as the median measured capacity of the first 3 valid cycles.

