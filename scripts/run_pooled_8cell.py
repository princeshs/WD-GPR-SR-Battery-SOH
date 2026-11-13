"""
scripts/run_pooled_8cell.py
===========================
Executes Pooled 8-Cell Leave-One-Cell-Out Cross-Validation (LOCO-CV)
across all 8 benchmark cells (4 NASA + 4 CALCE):
- NASA: B0005, B0006, B0007, B0018
- CALCE: CS2_35, CS2_36, CS2_37, CS2_38
Evaluates:
  1. WD_GPR (Proposed Wide-and-Deep Variational GP)
  2. Random_Forest
  3. XGBoost
  4. BiLSTM
Outputs:
  - results/metrics/pooled_8cell_results.csv
  - results/metrics/pooled_8cell_summary.csv
"""

import os
import sys
import time
import resource
import numpy as np
import pandas as pd

sys.path.insert(0, ".")
from src.config import FEATURES_MAIN, TARGET_COLUMN
from src.evaluation import (
    PreprocessingPipeline,
    evaluate_uncertainty_metrics,
    set_seed,
    compute_recalibration_factor,
    compute_quantile_calibration_factor,
    apply_shared_calibration,
)
from src.gpr_models import train_gp_model
from src.baseline_models import RandomForestBaseline, XGBoostBaseline, BiLSTMSeq


def run_pooled_8cell_loco():
    print("=== Executing Task 6: Pooled 8-Cell LOCO Cross-Validation ===")
    os.makedirs("results/metrics", exist_ok=True)

    nasa_cells = ["B0005", "B0006", "B0007", "B0018"]
    calce_cells = ["CS2_35", "CS2_36", "CS2_37", "CS2_38"]
    all_cells = nasa_cells + calce_cells

    df_nasa = pd.read_csv("data/processed/features_nasa.csv")
    df_calce = pd.read_csv("data/processed/features_calce.csv")

    df_nasa_clean = df_nasa[df_nasa["cell_id"].isin(nasa_cells) & (~df_nasa["is_outlier"])].copy()
    df_calce_clean = df_calce[df_calce["cell_id"].isin(calce_cells) & (~df_calce["is_outlier"])].copy()

    df_nasa_clean["source_dataset"] = "NASA"
    df_calce_clean["source_dataset"] = "CALCE"

    df_pooled = pd.concat([df_nasa_clean, df_calce_clean], ignore_index=True)
    print(f"Pooled dataset: {len(df_pooled)} cycles across {len(all_cells)} cells.")

    models = {
        "WD_GPR": None,  # handled via train_gp_model
        "Random_Forest": RandomForestBaseline,
        "XGBoost": XGBoostBaseline,
        "BiLSTM": BiLSTMSeq,
    }

    seeds = [42, 123, 999]
    records = []

    for m_name, m_cls in models.items():
        print(f"\n--- Running Pooled 8-Cell LOCO for {m_name} ---")
        for held_out in all_cells:
            tr_df = df_pooled[df_pooled["cell_id"] != held_out]
            val_df = df_pooled[df_pooled["cell_id"] == held_out]

            X_tr_raw = tr_df[FEATURES_MAIN].values
            y_tr = tr_df[TARGET_COLUMN].values
            X_val_raw = val_df[FEATURES_MAIN].values
            y_val = val_df[TARGET_COLUMN].values

            for seed in seeds:
                set_seed(seed)
                pipe = PreprocessingPipeline()
                X_tr = pipe.fit_transform(X_tr_raw)
                X_val = pipe.transform(X_val_raw)

                t0 = time.perf_counter()
                if m_name == "WD_GPR":
                    model = train_gp_model(
                        X_tr, y_tr,
                        variant="C",
                        num_inducing=64,
                        epochs=80,
                        lr=0.015,
                        patience=15,
                        val_fraction=0.15,
                        cell_ids=tr_df["cell_id"].values,
                        seed=seed,
                    )
                    y_pred, y_std = model.predict(X_val)
                else:
                    inst = m_cls()
                    inst.fit(X_tr, y_tr, seed=seed)
                    y_pred, y_std = inst.predict(X_val)
                train_time = float(time.perf_counter() - t0)

                # Inference latency over 500 samples
                single_x = X_val[:1]
                for _ in range(20):
                    if m_name == "WD_GPR":
                        model.predict(single_x)
                    else:
                        inst.predict(single_x)
                lats = []
                for _ in range(500):
                    t_inf0 = time.perf_counter()
                    if m_name == "WD_GPR":
                        model.predict(single_x)
                    else:
                        inst.predict(single_x)
                    lats.append((time.perf_counter() - t_inf0) * 1000.0)
                inf_lat = float(np.median(lats))
                peak_mem = float(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0)

                # Regression & calibration metrics
                m = evaluate_uncertainty_metrics(y_val, y_pred, y_std)
                calib = apply_shared_calibration(y_val, y_pred, y_std, s_factor=1.0)

                records.append({
                    "model": m_name,
                    "dataset": "POOLED_8CELL",
                    "held_out_cell": held_out,
                    "cell_origin": "NASA" if held_out in nasa_cells else "CALCE",
                    "seed": seed,
                    "n_train": len(X_tr),
                    "n_test": len(X_val),
                    "mae": m["mae"],
                    "mse": m["mse"],
                    "rmse": m["rmse"],
                    "r2": m["r2"],
                    "mape": m["mape"],
                    "mape_soh50": m["mape_soh50"],
                    "smape": m["smape"],
                    "nll": calib["nll_raw"],
                    "picp": calib["picp_raw"],
                    "mpiw": calib["mpiw_raw"],
                    "train_time_sec": train_time,
                    "inference_latency_ms": inf_lat,
                    "peak_memory_mb": peak_mem,
                    "hardware": "development computer, CPU, NOT Jetson",
                })

        # Progress print
        sub_m = [r for r in records if r["model"] == m_name]
        print(f"  Completed {len(sub_m)} runs: Mean RMSE = {np.mean([r['rmse'] for r in sub_m]):.2f} ± {np.std([r['rmse'] for r in sub_m]):.2f}%")

    df_res = pd.DataFrame(records)
    df_res.to_csv("results/metrics/pooled_8cell_results.csv", index=False)
    print("Saved pooled 8-cell results to results/metrics/pooled_8cell_results.csv")

    # Summary
    sum_rows = []
    for m_name, grp in df_res.groupby("model"):
        # Std across cells (average per cell first, then std)
        cell_means = grp.groupby("held_out_cell")["rmse"].mean()
        seed_means = grp.groupby("seed")["rmse"].mean()
        sum_rows.append({
            "model": m_name,
            "dataset": "POOLED_8CELL",
            "MAE_mean": grp["mae"].mean(),
            "MAE_std_cells": grp.groupby("held_out_cell")["mae"].mean().std(),
            "RMSE_mean": grp["rmse"].mean(),
            "RMSE_std_cells": cell_means.std(),
            "RMSE_std_seeds": seed_means.std(),
            "R2_mean": grp["r2"].mean(),
            "MAPE_mean": grp["mape"].mean(),
            "MAPE_SOH50_mean": grp["mape_soh50"].mean(),
            "sMAPE_mean": grp["smape"].mean(),
            "train_time_sec_mean": grp["train_time_sec"].mean(),
            "inference_latency_ms_mean": grp["inference_latency_ms"].mean(),
            "peak_memory_mb_mean": grp["peak_memory_mb"].mean(),
        })
    df_sum = pd.DataFrame(sum_rows).sort_values("RMSE_mean")
    df_sum.to_csv("results/metrics/pooled_8cell_summary.csv", index=False)
    print("Saved pooled 8-cell summary to results/metrics/pooled_8cell_summary.csv")


if __name__ == "__main__":
    run_pooled_8cell_loco()

