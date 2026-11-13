"""
Patch 2401AI38_MLProject.ipynb:
- Fix stale CSV column names in Cells 16 & 18 (NLL_mean -> NLL_raw_mean etc.)
- Fix stale conclusions in Cell 25 with verified Phase 3C numbers
"""
import json

NB_PATH = "2401AI38_MLProject.ipynb"

with open(NB_PATH, "r") as f:
    nb = json.load(f)

# ─── Cell 16: Leaderboard ──────────────────────────────────────────────────
NEW_CELL16 = r"""# 9. Master Benchmark Leaderboard across All Mandatory PBL Metrics
summary_path = 'results/metrics/summary_mean_std_over_seeds.csv'
if not os.path.exists(summary_path) and os.path.exists('../' + summary_path):
    summary_path = '../' + summary_path

df_sum = pd.read_csv(summary_path)
for dset in ['CALCE', 'NASA']:
    sub = df_sum[df_sum['dataset'] == dset].sort_values('RMSE_mean')
    print(f'\n========================================================================================================================')
    print(f'                             {dset} MASTER BENCHMARK LEADERBOARD (All Mandatory PBL Metrics)                            ')
    print(f'========================================================================================================================')
    disp = pd.DataFrame({
        'Model': sub['model'],
        'MAE (%)': sub['MAE_mean'].apply(lambda x: f'{x:.2f}'),
        'MSE': sub['MSE_mean'].apply(lambda x: f'{x:.2f}' if pd.notna(x) else '-'),
        'RMSE (%)': sub['RMSE_mean'].apply(lambda x: f'{x:.2f}'),
        'R\u00b2': sub['R2_mean'].round(3),
        'MAPE (%)': sub['MAPE_mean'].apply(lambda x: f'{x:.2f}' if pd.notna(x) else '-'),
        'NLL': sub['NLL_raw_mean'].apply(lambda x: f'{x:.2f}' if pd.notna(x) else '-'),
        '95% PICP': sub['PICP_raw_mean'].apply(lambda x: f'{x:.1f}%' if pd.notna(x) else '-'),
        'MPIW (%)': sub['MPIW_raw_mean'].apply(lambda x: f'{x:.2f}' if pd.notna(x) else '-'),
        'Nodes': sub['equation_nodes'].apply(lambda x: f'{int(x)}' if pd.notna(x) else '0'),
        'Latency': sub['inf_latency_ms'].apply(lambda x: f'{x:.3f} ms' if pd.notna(x) else '-')
    })
    print(disp.to_string(index=False))

# Compact Hyperparameter Specification Table
hp_data = [
    {'Module': 'Wide Branch', 'Formulation': 'Linear + ARD-RBF Kernel', 'Key Hyperparameters': 'ARD lengthscales (5 features), sigma_w^2 init=1.0'},
    {'Module': 'Deep Branch (phi_theta)', 'Formulation': 'MLP: 5 -> 32 -> 16 -> 4', 'Key Hyperparameters': 'ReLU, LayerNorm, Dropout(0.05), lr=0.015, Adam'},
    {'Module': 'Variational SVGP', 'Formulation': 'Sparse GP + Matern-5/2', 'Key Hyperparameters': 'M=64 inducing points (k-means init), 80 epochs, batch=128, refit_blocked protocol'},
    {'Module': 'Symbolic Corrector', 'Formulation': 'Parsimonious GP (gplearn)', 'Key Hyperparameters': 'Pop=1000, Gen=30, maxsize=15, Inv-variance weights, nested gate (g=0 candidate)'},
]
print('\n=== Model Hyperparameter & Architecture Specification ===')
print(pd.DataFrame(hp_data).to_string(index=False))

# Display ELBO Loss Convergence Curves Plot
elbo_fig_path = 'results/figures/final/final_elbo_convergence.png'
if not os.path.exists(elbo_fig_path) and os.path.exists('../' + elbo_fig_path):
    elbo_fig_path = '../' + elbo_fig_path
if os.path.exists(elbo_fig_path):
    img = plt.imread(elbo_fig_path)
    plt.figure(figsize=(10, 3.5))
    plt.imshow(img)
    plt.axis('off')
    plt.title('Variational ELBO Training Loss Convergence (M=64 Inducing Points)', fontsize=11, fontweight='bold')
    plt.tight_layout()
    plt.show()
"""

# ─── Cell 18: Ablation ─────────────────────────────────────────────────────
NEW_CELL18 = r"""# 10. Display Verified Ablation Analysis Table with All Mandatory PBL Metrics
abl_path = 'results/metrics/ablation_table.csv'
if not os.path.exists(abl_path) and os.path.exists('../' + abl_path):
    abl_path = '../' + abl_path

df_abl = pd.read_csv(abl_path)
for dset in ['CALCE', 'NASA']:
    sub = df_abl[df_abl['dataset'] == dset]
    print(f'\n=== {dset} Architectural Ablation (Configs A\u2013E across All Metrics) ===')
    disp = pd.DataFrame({
        'Config': sub['model'],
        'MAE (%)': sub['MAE_mean'].apply(lambda x: f'{x:.2f}'),
        'MSE': sub['MSE_mean'].apply(lambda x: f'{x:.2f}' if pd.notna(x) else '-'),
        'RMSE (%)': sub['RMSE_mean'].apply(lambda x: f'{x:.2f}'),
        'R\u00b2': sub['R2_mean'].round(3),
        'MAPE (%)': sub['MAPE_mean'].apply(lambda x: f'{x:.2f}' if pd.notna(x) else '-'),
        'NLL': sub['NLL_raw_mean'].apply(lambda x: f'{x:.2f}' if pd.notna(x) else '-'),
        '95% PICP': sub['PICP_raw_mean'].apply(lambda x: f'{x:.1f}%' if pd.notna(x) else '-'),
        'MPIW (%)': sub['MPIW_raw_mean'].apply(lambda x: f'{x:.2f}' if pd.notna(x) else '-'),
        'Nodes': sub['equation_nodes'].apply(lambda x: f'{int(x)}' if pd.notna(x) else '0'),
        'Peak Mem': sub['peak_memory_mb'].apply(lambda x: f'{x:.2f} MB' if pd.notna(x) else '-')
    })
    print(disp.to_string(index=False))
"""

# ─── Cell 25: Discussion & Conclusions ─────────────────────────────────────
NEW_CELL25 = r"""## 14. Discussion, Limitations & Conclusions

### Summary of Empirical Evidence (Phase 3C Verified Results):

| Verdict | Finding |
|:---|:---|
| **(a) Does the deep branch help?** | **No difference beyond seed and cell variation.** Config C (Wide+Deep, NASA 3.37%, CALCE 2.99%) vs. Config A (Wide-only, NASA 3.51%, CALCE 2.95%) — gap is smaller than the cross-cell standard deviation. |
| **(b) Does uncertainty weighting help SR?** | **No.** $E_\text{forced}$ (weighted) underperformed $D_\text{forced}$ (unweighted) on both datasets (NASA: 4.69% vs. 4.29%; CALCE: 4.61% vs. 4.23%). |
| **(c) Does any SR correction help?** | **No.** Nested cross-validation gate selected $g=0$ in **12/12 folds on both NASA and CALCE**. Forcing SR always increased test error. The proposed WD-GPR-SR reduces to WD-GPR. |

### Key Benchmark Results (LOCO-CV $\times$ Seeds {42, 123, 999}):

**NASA:** WD-GPR achieves **3.37% ± 2.48% RMSE** (Config C). BiLSTM achieves 3.59% ± 2.46%. Random Forest achieves 3.56% ± 1.95%. Differences are within cross-cell variation — **no model is conclusively superior** on NASA.

**CALCE:** BiLSTM (**1.34% ± 0.45% RMSE**) and DeepNN (**1.64% ± 0.60%**) outperform WD-GPR (2.99% ± 0.80%). Sequential architectures that model temporal dynamics hold a genuine advantage on this long-horizon degradation dataset.

**Pooled 8-Cell LOCO:** BiLSTM 2.20%, Random Forest 3.10%, XGBoost 3.19%, WD-GPR 3.82% RMSE.

### Real-Time Edge Deployment:
Standalone ONNX + C engine executes in **0.123 ms** per sample with **0.59 MB** memory on the development computer (CPU, NOT Jetson), operating >800× faster than the 100 ms CAN-bus budget.

### Residual Variance Diagnostic:
On NASA, **71.4% of GP out-of-fold residual variance is a per-cell constant offset** (cross-cell GBDT $R^2 = -1.34$), which explains why SR fails to generalize: the dominant error is a cell-specific domain offset that symbolic regression cannot cross-cell-correct. On CALCE, the offset is 15.7% (cross-cell GBDT $R^2 = -0.35$).

### Honest Limitations & Next Steps:
- **Small-Sample Boundary:** Evaluated on $N=4+4$ cells; inter-cell variance is physical rather than stochastic. Results may not generalize across different cycling protocols or chemistries.
- **B0006 Calibration Failure:** Cell B0006 shows severe calibration failure across all probabilistic models (PICP ≈ 37.5% raw), caused by abrupt early-failure dynamics not seen in training cells. Cited as a fundamental cross-cell domain shift limitation — not tuned.
- **GP Training Protocol Fix (Phase 3C):** The original blocked-split-without-refit protocol held out the low-SOH tail of CALCE (~5% SOH end-of-life), causing RMSE to degrade to 6.34%. The corrected `refit_blocked` protocol restores CALCE WD-GPR RMSE to 2.99%.
- **Hardware Power Telemetry:** Electrical power (Watts) and energy consumption require physical Jetson Orin Nano hardware with INA3221 power rail monitors.
"""

# Apply patches
nb['cells'][16]['source'] = NEW_CELL16
nb['cells'][18]['source'] = NEW_CELL18
nb['cells'][25]['source'] = NEW_CELL25

# Clear stale outputs so re-execution is clean
for i in [16, 18]:
    nb['cells'][i]['outputs'] = []
    nb['cells'][i]['execution_count'] = None

with open(NB_PATH, "w") as f:
    json.dump(nb, f, indent=1, ensure_ascii=False)

print("Patched cells 16, 18, 25 successfully.")

