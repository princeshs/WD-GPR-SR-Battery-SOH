#!/usr/bin/env python3
"""
Regime comparison for Config C (WD-GPR) on NASA and CALCE (LOCO x seeds 42, 123, 999).
Task 2 of Phase 3C:
  (a) random_no_refit: old random 15% validation
  (b) blocked_no_refit: current blocked split without refit
  (c) fixed_80: fixed 80 epochs, no early stopping
  (d) refit_blocked: blocked validation to choose epochs, then refit on all training cycles

Evaluation includes:
  - Inner leave-one-training-cell-out RMSE on training cells only (selection criterion)
  - Held-out test cell RMSE
  - Stopping epoch
  - SOH range of training part vs validation part
"""

import os
import sys
import time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
import pandas as pd

from src.config import FEATURES_MAIN, TARGET_COLUMN
from src.evaluation import PreprocessingPipeline
from src.gpr_models import train_gp_model

NASA_CELLS = ["B0005", "B0006", "B0007", "B0018"]
CALCE_CELLS = ["CS2_35", "CS2_36", "CS2_37", "CS2_38"]
RANDOM_SEEDS = [42, 123, 999]

def run_regime_comparison():
    os.makedirs("results/metrics", exist_ok=True)
    
    # Load data
    df_nasa = pd.read_csv("data/processed/features_nasa.csv")
    df_calce = pd.read_csv("data/processed/features_calce.csv")
    
    # Clean filters
    df_nasa = df_nasa[df_nasa["cell_id"].isin(NASA_CELLS) & (~df_nasa["is_outlier"])].copy()
    df_calce = df_calce[df_calce["cell_id"].isin(CALCE_CELLS) & (~df_calce["is_outlier"])].copy()
    
    regimes = [
        ("a_random_no_refit", "random_no_refit"),
        ("b_blocked_no_refit", "blocked_no_refit"),
        ("c_fixed_80", "fixed_80"),
        ("d_refit_blocked", "refit_blocked"),
    ]
    
    datasets = [
        ("NASA", df_nasa, NASA_CELLS),
        ("CALCE", df_calce, CALCE_CELLS),
    ]
    
    all_results = []
    
    print("=" * 80)
    print("Starting Phase 3C Task 2: Regime Comparison for Config C")
    print("=" * 80)
    
    for dset_name, df_data, cells in datasets:
        for reg_label, reg_key in regimes:
            print(f"\n--- Running Dataset: {dset_name} | Regime: {reg_label} ---")
            
            for seed in RANDOM_SEEDS:
                for test_cell in cells:
                    train_cells = [c for c in cells if c != test_cell]
                    
                    df_train = df_data[df_data["cell_id"].isin(train_cells)].copy()
                    df_test = df_data[df_data["cell_id"] == test_cell].copy()
                    
                    pipe = PreprocessingPipeline()
                    X_train = pipe.fit_transform(df_train[FEATURES_MAIN].values)
                    y_train = df_train[TARGET_COLUMN].values
                    cell_ids_train = df_train["cell_id"].values
                    
                    X_test = pipe.transform(df_test[FEATURES_MAIN].values)
                    y_test = df_test[TARGET_COLUMN].values
                    
                    # 1. Inner leave-one-training-cell-out validation (selection criterion)
                    inner_rmses = []
                    for inner_val_cell in train_cells:
                        inner_sub_tr = [c for c in train_cells if c != inner_val_cell]
                        df_in_tr = df_train[df_train["cell_id"].isin(inner_sub_tr)]
                        df_in_val = df_train[df_train["cell_id"] == inner_val_cell]
                        
                        pipe_in = PreprocessingPipeline().fit(df_in_tr[FEATURES_MAIN].values)
                        X_in_tr = pipe_in.transform(df_in_tr[FEATURES_MAIN].values)
                        y_in_tr = df_in_tr[TARGET_COLUMN].values
                        c_in_tr = df_in_tr["cell_id"].values
                        
                        X_in_val = pipe_in.transform(df_in_val[FEATURES_MAIN].values)
                        y_in_val = df_in_val[TARGET_COLUMN].values
                        
                        m_inner = train_gp_model(
                            X_in_tr, y_in_tr,
                            variant="C", epochs=80, patience=15,
                            cell_ids=c_in_tr, seed=seed,
                            training_regime=reg_key
                        )
                        mu_in, _ = m_inner.predict(X_in_val)
                        in_rmse = float(np.sqrt(np.mean((y_in_val - mu_in) ** 2)))
                        inner_rmses.append(in_rmse)
                        
                    inner_oof_rmse = float(np.mean(inner_rmses))
                    
                    # 2. Outer training on all 3 training cells and evaluation on held-out test cell
                    t0 = time.perf_counter()
                    model = train_gp_model(
                        X_train, y_train,
                        variant="C", epochs=80, patience=15,
                        cell_ids=cell_ids_train, seed=seed,
                        training_regime=reg_key
                    )
                    train_time = time.perf_counter() - t0
                    
                    mu_test, std_test = model.predict(X_test)
                    test_rmse = float(np.sqrt(np.mean((y_test - mu_test) ** 2)))
                    test_mae = float(np.mean(np.abs(y_test - mu_test)))
                    
                    rec = {
                        "dataset": dset_name,
                        "regime_label": reg_label,
                        "regime_key": reg_key,
                        "seed": seed,
                        "test_cell": test_cell,
                        "inner_oof_rmse": inner_oof_rmse,
                        "test_rmse": test_rmse,
                        "test_mae": test_mae,
                        "stopping_epoch": model.stopping_epoch,
                        "train_time_sec": train_time,
                        "soh_train_min": model.y_train_min,
                        "soh_train_max": model.y_train_max,
                        "soh_fit_min": model.y_fit_min,
                        "soh_fit_max": model.y_fit_max,
                        "soh_val_min": model.y_val_min,
                        "soh_val_max": model.y_val_max,
                    }
                    all_results.append(rec)
                    print(f"  [{dset_name}] {reg_label} seed={seed} test={test_cell} | stop_ep={model.stopping_epoch:2d} | Inner-OOF RMSE: {inner_oof_rmse:.2f}% | Test RMSE: {test_rmse:.2f}% | SOH fit:[{model.y_fit_min:.1f}, {model.y_fit_max:.1f}] val:[{model.y_val_min:.1f}, {model.y_val_max:.1f}]")

    df_res = pd.DataFrame(all_results)
    df_res.to_csv("results/metrics/regime_comparison_per_fold.csv", index=False)
    print("\nSaved per-fold results to results/metrics/regime_comparison_per_fold.csv")
    
    # Aggregate summary
    summary_rows = []
    for (dset, reg_label), grp in df_res.groupby(["dataset", "regime_label"]):
        # Mean across folds
        inner_mean = grp["inner_oof_rmse"].mean()
        inner_std = grp["inner_oof_rmse"].std()
        test_mean = grp["test_rmse"].mean()
        test_std_cells = grp.groupby("test_cell")["test_rmse"].mean().std()
        test_std_seeds = grp.groupby("seed")["test_rmse"].mean().std()
        test_mae_mean = grp["test_mae"].mean()
        stop_epoch_mean = grp["stopping_epoch"].mean()
        time_mean = grp["train_time_sec"].mean()
        
        # Per cell RMSE
        cell_dict = {}
        for c, c_grp in grp.groupby("test_cell"):
            cell_dict[f"RMSE_{c}"] = c_grp["test_rmse"].mean()
            
        row = {
            "dataset": dset,
            "regime": reg_label,
            "inner_oof_rmse_mean": inner_mean,
            "inner_oof_rmse_std": inner_std,
            "test_rmse_mean": test_mean,
            "test_rmse_std_cells": test_std_cells,
            "test_rmse_std_seeds": test_std_seeds,
            "test_mae_mean": test_mae_mean,
            "stopping_epoch_mean": stop_epoch_mean,
            "train_time_sec_mean": time_mean,
            "soh_fit_min_avg": grp["soh_fit_min"].mean(),
            "soh_fit_max_avg": grp["soh_fit_max"].mean(),
            "soh_val_min_avg": grp["soh_val_min"].mean(),
            "soh_val_max_avg": grp["soh_val_max"].mean(),
        }
        row.update(cell_dict)
        summary_rows.append(row)
        
    df_summary = pd.DataFrame(summary_rows)
    df_summary.to_csv("results/metrics/regime_comparison_summary.csv", index=False)
    print("Saved summary table to results/metrics/regime_comparison_summary.csv")
    
    print("\n" + "=" * 90)
    print("REGIME COMPARISON SUMMARY TABLE (Phase 3C Task 2)")
    print("=" * 90)
    display_cols = ["dataset", "regime", "inner_oof_rmse_mean", "test_rmse_mean", "test_rmse_std_cells", "stopping_epoch_mean", "soh_fit_min_avg", "soh_val_min_avg"]
    print(df_summary[display_cols].to_string(index=False))
    print("=" * 90)

if __name__ == "__main__":
    run_regime_comparison()
