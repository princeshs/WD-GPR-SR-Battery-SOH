# Data Filtering and Outlier Detection Audit Report


1. OUTLIER DETECTION RULES:
   - Algorithm: Local Rolling Detrended Median IQR + Cutoff Threshold.
   - Reference signal: Raw measured discharge capacity `capacity` (in Ampere-hours, Ah).
   - Rolling Window: 15 cycles (center=True, min_periods=3, with edge fill).
   - Anomaly Criterion 1 (Capacity IQR Outlier):
       residual_i = |capacity_i - median_15(capacity_i)|
       thresh = Q75(residual) + 2.5 * IQR(residual)
       Flagged if: residual_i > thresh AND capacity_i < median_15(capacity_i)
   - Anomaly Criterion 2 (Interrupted Test Cutoff):
       Flagged if: capacity_i < 0.35 * median_15(capacity_i) AND median_15(capacity_i) > 0.5 Ah
   - Physical Validity Bounds:
       Capacity: 0.0 Ah <= capacity <= 5.0 Ah (unit error check if median > 50.0 Ah)
       Voltage: 2.0 V <= voltage <= 4.5 V
       Temperature: -20 C <= temperature <= 80 C
       Timestamp Monotonicity: cycle dates and intra-cycle time series strictly non-decreasing.

2. LABEL & MODEL OUTPUT INDEPENDENCE:
   - Uses SOH label? NO. SOH (%) = (capacity / nominal_capacity) * 100 is NOT used for detection;
     filtering operates purely on raw discharge capacity curves.
   - Uses model predictions or residuals? NO. Outlier flags are computed once during Stage 1
     ingestion before any feature normalization, scaling, model training, or evaluation.
   - Held-out cell leakage? NO. Outlier detection operates cell-by-cell on time series,
     completely independent of any other cell.

3. UNIFORM APPLICATION:
   - Training vs. Test: Applied identically to all cells prior to LOCO partitioning.
     No cell is selectively cleaned based on whether it is held out.
   - Model Uniformity: Every downstream pipeline (Stage 3 Baselines, Stage 4 Ablation,
     Stage 5 Symbolic Regression, Secondary Pooled Evaluation) loads the exact same
     filtered data table.


## Per-Cell Cycle Counts

| dataset   | cohort        | cell_id   |   total_raw |   outliers_removed |   valid_cycles |   pct_removed |
|:----------|:--------------|:----------|------------:|-------------------:|---------------:|--------------:|
| NASA      | Benchmark     | B0005     |         168 |                  0 |            168 |       0       |
| NASA      | Benchmark     | B0006     |         168 |                  0 |            168 |       0       |
| NASA      | Benchmark     | B0007     |         168 |                  0 |            168 |       0       |
| NASA      | Benchmark     | B0018     |         132 |                  0 |            132 |       0       |
| CALCE     | Benchmark     | CS2_35    |         932 |                124 |            808 |      13.3047  |
| CALCE     | Benchmark     | CS2_36    |         973 |                 29 |            944 |       2.98047 |
| CALCE     | Benchmark     | CS2_37    |        1038 |                 30 |           1008 |       2.89017 |
| CALCE     | Benchmark     | CS2_38    |        1078 |                133 |            945 |      12.3377  |
| CALCE     | Non-Benchmark | CS2_33    |         866 |                 39 |            827 |       4.50346 |
| CALCE     | Non-Benchmark | CS2_34    |         776 |                 25 |            751 |       3.22165 |

## Key Models Performance: Filter Impact (CALCE LOCO)

| model         | dataset   | filter_active   | condition         |   RMSE_mean |   RMSE_std |   MAE_mean |   MAE_std |
|:--------------|:----------|:----------------|:------------------|------------:|-----------:|-----------:|----------:|
| Random_Forest | CALCE     | True            | Clean (Filter ON) |     2.31568 |   1.25287  |   1.12913  |  0.63075  |
| Random_Forest | CALCE     | False           | Raw (Filter OFF)  |     4.07227 |   1.4143   |   1.42382  |  0.52513  |
| DeepNN        | CALCE     | True            | Clean (Filter ON) |     1.77061 |   0.902714 |   1.19019  |  0.751448 |
| DeepNN        | CALCE     | False           | Raw (Filter OFF)  |     3.28097 |   1.34476  |   1.39839  |  0.526058 |
| BiLSTM        | CALCE     | True            | Clean (Filter ON) |     1.40383 |   0.427101 |   0.942827 |  0.23792  |
| BiLSTM        | CALCE     | False           | Raw (Filter OFF)  |     3.17577 |   1.26411  |   1.23068  |  0.263206 |
| WD_GPR        | CALCE     | True            | Clean (Filter ON) |     3.02179 |   0.929213 |   1.8353   |  0.782966 |
| WD_GPR        | CALCE     | False           | Raw (Filter OFF)  |     4.14558 |   1.58692  |   2.11103  |  0.919131 |

## Exclusion of Non-Benchmark Cells CS2_33 and CS2_34


1. Cross-Cell Cycling Schedule Inconsistency:
   - CS2_35, CS2_36, CS2_37, and CS2_38 were cycled under identical 0.5C constant-current /
     constant-voltage charging followed by 1C discharge cycles until end-of-life (>900 cycles).
   - In contrast, CS2_33 (866 cycles) and CS2_34 (776 cycles) were operated under different
     experimental protocols with interrupted testing and fewer operational cycles.

2. Impact of Training Set Contamination:
   - In earlier legacy runs (results/archive_old/baselines.csv), CS2_33 and CS2_34 were bundled
     into the training set (N_train = 4,731), but evaluation was performed ONLY on CS2_35-38.
   - This introduced severe domain shift and irregular capacity plateaus into the training data,
     degrading Random Forest from 2.31% to 3.39% RMSE and BiLSTM from 1.34% to 3.08% RMSE.
   - Restricting the training cohort strictly to the 4 benchmark cells under standard LOCO-CV
     (train on 3 cells, test on 1 held-out cell) eliminates training domain shift and aligns
     our experimental protocol with peer-reviewed battery degradation benchmarks.
