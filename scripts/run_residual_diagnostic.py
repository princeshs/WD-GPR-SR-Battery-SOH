"""
scripts/run_residual_diagnostic.py
==================================
Diagnostic: on the inner OOF residuals, report the share of residual variance
that is a per-cell constant offset versus explained by a cross-cell-validated
feature model (Gradient Boosting / Ridge).
"""

import sys
import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.linear_model import Ridge

sys.path.insert(0, ".")
from src.config import FEATURES_MAIN, TARGET_COLUMN
from src.evaluation import PreprocessingPipeline, set_seed
from src.gpr_models import train_gp_model

def run_diagnostic():
    print("=== Task 5: Residual Variance Diagnostic ===")
    
    nasa_cells = ["B0005", "B0006", "B0007", "B0018"]
    calce_cells = ["CS2_35", "CS2_36", "CS2_37", "CS2_38"]
    
    df_nasa = pd.read_csv("data/processed/features_nasa.csv")
    df_calce = pd.read_csv("data/processed/features_calce.csv")
    
    df_nasa_clean = df_nasa[df_nasa["cell_id"].isin(nasa_cells) & (~df_nasa["is_outlier"])].copy()
    df_calce_clean = df_calce[df_calce["cell_id"].isin(calce_cells) & (~df_calce["is_outlier"])].copy()
    
    datasets = {
        "NASA": {"df": df_nasa_clean, "cells": nasa_cells},
        "CALCE": {"df": df_calce_clean, "cells": calce_cells},
    }
    
    results = []
    
    for dset_name, dset_info in datasets.items():
        df_dset = dset_info["df"]
        cells = dset_info["cells"]
        
        # Step 1: Compute cross-cell GP out-of-fold residuals (LOCO over all 4 cells)
        oof_residuals = []
        oof_cell_ids = []
        oof_features = []
        
        for held_out in cells:
            tr_df = df_dset[df_dset["cell_id"] != held_out]
            val_df = df_dset[df_dset["cell_id"] == held_out]
            
            pipe = PreprocessingPipeline()
            X_tr = pipe.fit_transform(tr_df[FEATURES_MAIN].values)
            y_tr = tr_df[TARGET_COLUMN].values
            X_val = pipe.transform(val_df[FEATURES_MAIN].values)
            y_val = val_df[TARGET_COLUMN].values
            
            gp = train_gp_model(
                X_tr, y_tr,
                variant="C",
                num_inducing=64,
                epochs=80,
                lr=0.015,
                patience=15,
                val_fraction=0.15,
                cell_ids=tr_df["cell_id"].values,
                seed=42,
            )
            mu_val, _ = gp.predict(X_val)
            res = y_val - mu_val
            
            oof_residuals.append(res)
            oof_cell_ids.append(np.full(len(res), held_out))
            oof_features.append(X_val)
            
        r_all = np.concatenate(oof_residuals)
        cell_all = np.concatenate(oof_cell_ids)
        X_all = np.vstack(oof_features)
        
        # Total residual variance
        var_total = float(np.var(r_all))
        mean_total = float(np.mean(r_all))
        
        # Share explained by per-cell constant offset
        cell_means = {}
        offset_pred = np.zeros_like(r_all)
        for c in cells:
            idx = np.where(cell_all == c)[0]
            c_mean = float(np.mean(r_all[idx]))
            cell_means[c] = c_mean
            offset_pred[idx] = c_mean
            
        var_offset = float(np.var(offset_pred))
        share_offset = (var_offset / var_total) * 100.0 if var_total > 0 else 0.0
        
        # Share explained by cross-cell-validated feature model (GBDT)
        gbdt_pred = np.zeros_like(r_all)
        for c in cells:
            tr_idx = np.where(cell_all != c)[0]
            val_idx = np.where(cell_all == c)[0]
            
            gbdt = GradientBoostingRegressor(n_estimators=50, max_depth=3, random_state=42)
            gbdt.fit(X_all[tr_idx], r_all[tr_idx])
            gbdt_pred[val_idx] = gbdt.predict(X_all[val_idx])
            
        mse_gbdt = float(np.mean((r_all - gbdt_pred) ** 2))
        r2_gbdt = 1.0 - (mse_gbdt / var_total) if var_total > 0 else 0.0
        share_gbdt = max(0.0, r2_gbdt) * 100.0
        
        # Also check inner LOCO rotations (within 3 training cells)
        inner_offset_shares = []
        inner_gbdt_r2s = []
        for outer_held in cells:
            in_cells = [c for c in cells if c != outer_held]
            in_mask = np.isin(cell_all, in_cells)
            r_in = r_all[in_mask]
            cell_in = cell_all[in_mask]
            X_in = X_all[in_mask]
            
            var_in_tot = float(np.var(r_in))
            off_in = np.zeros_like(r_in)
            for c in in_cells:
                idx = np.where(cell_in == c)[0]
                off_in[idx] = float(np.mean(r_in[idx]))
            inner_offset_shares.append((float(np.var(off_in)) / var_in_tot) * 100.0)
            
            # GBDT cross-cell inside inner
            in_gb_pred = np.zeros_like(r_in)
            for c in in_cells:
                tr_i = np.where(cell_in != c)[0]
                val_i = np.where(cell_in == c)[0]
                gb = GradientBoostingRegressor(n_estimators=30, max_depth=2, random_state=42)
                gb.fit(X_in[tr_i], r_in[tr_i])
                in_gb_pred[val_i] = gb.predict(X_in[val_i])
            mse_in_gb = float(np.mean((r_in - in_gb_pred) ** 2))
            inner_gbdt_r2s.append(1.0 - mse_in_gb / var_in_tot)

        print(f"\n--- {dset_name} Diagnostic ---")
        print(f"  Total OOF Residual Variance: {var_total:.4f} (%^2) [Std: {np.sqrt(var_total):.2f}%]")
        print(f"  Per-Cell Mean Offsets: { {k: round(v, 3) for k, v in cell_means.items()} }")
        print(f"  Share of Residual Variance Explained by Per-Cell Constant Offset: {share_offset:.1f}%")
        print(f"  Share Explained by Cross-Cell GBDT Feature Model (R2): {r2_gbdt*100.0:.1f}% (Cross-cell R2 = {r2_gbdt:.3f})")
        print(f"  Inner-Fold Avg Offset Share: {np.mean(inner_offset_shares):.1f}%, Inner Cross-Cell GBDT R2: {np.mean(inner_gbdt_r2s):.3f}")
        
        results.append({
            "dataset": dset_name,
            "total_residual_variance": var_total,
            "per_cell_offset_share_pct": share_offset,
            "cross_cell_gbdt_r2": r2_gbdt,
            "cross_cell_gbdt_share_pct": share_gbdt,
            "inner_offset_share_pct": np.mean(inner_offset_shares),
            "inner_gbdt_r2": np.mean(inner_gbdt_r2s),
        })
        
    df_res = pd.DataFrame(results)
    df_res.to_csv("results/metrics/residual_variance_diagnostic.csv", index=False)
    print("\nSaved diagnostic to results/metrics/residual_variance_diagnostic.csv")

if __name__ == "__main__":
    run_diagnostic()

