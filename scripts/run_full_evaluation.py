"""
scripts/run_full_evaluation.py
==============================
Stage 6 Master Runner: Full Cross-Validation, Ablation Analysis, and Benchmarking.
Executes under strict LOCO within NASA and CALCE across seeds {42, 123, 999}.
Integrates secondary pooled 8-cell LOCO evaluation.
Compiles regression metrics, uncertainty metrics, equation complexity, runtime,
inference latency per sample, and peak memory via tracemalloc.
Generates all 4 comprehensive CSV reports and 6 publication-ready figures.
"""

import os
import sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, ".")
from src.config import FEATURES_MAIN, TARGET_COLUMN
from src.evaluation import PreprocessingPipeline, evaluate_uncertainty_metrics
from src.gpr_models import WideAndDeepSVGPRegressor


def run_full_evaluation():
    print("=== Executing Stage 6: Full Evaluation & Ablation Master Runner ===")
    os.makedirs("results/metrics", exist_ok=True)
    os.makedirs("results/figures/final", exist_ok=True)

    long_path = "results/metrics/full_results_long.csv"

    # Profiling is measured by the stage runners.  Never substitute historical
    # values here: that would make a regenerated report look empirically timed
    # when it was not.
    def measured(row, column):
        value = row.get(column, np.nan)
        return float(value) if pd.notna(value) else np.nan

    all_records = []

    # 2. Compile Baselines (Stage 3)
    df_baselines = pd.read_csv("results/metrics/baselines.csv")
    for _, row in df_baselines.iterrows():
        m_name = row["model"]
        train_t = measured(row, "train_time_sec")
        inf_lat = measured(row, "inf_latency_ms")
        peak_m = measured(row, "peak_memory_mb")
        all_records.append({
            "model": m_name,
            "dataset": row["dataset"],
            "held_out_cell": row["held_out_cell"],
            "seed": int(row["seed"]),
            "n_train": int(row["n_train"]),
            "n_test": int(row["n_test"]),
            "mae": float(row["mae"]),
            "mse": float(row["mse"]),
            "rmse": float(row["rmse"]),
            "r2": float(row["r2"]),
            "mape": float(row.get("mape", np.nan)),
            "mape_soh50": float(row.get("mape_soh50", np.nan)),
            "smape": float(row.get("smape", np.nan)),
            "nll": float(row.get("nll_raw", row.get("nll", np.nan))),
            "picp": float(row.get("picp_raw", row.get("picp", np.nan))),
            "mpiw": float(row.get("mpiw_raw", row.get("mpiw", np.nan))),
            "nll_recal": float(row.get("nll_recal", np.nan)),
            "picp_recal": float(row.get("picp_recal", np.nan)),
            "mpiw_recal": float(row.get("mpiw_recal", np.nan)),
            "nll_recal_q": float(row.get("nll_recal_q", np.nan)),
            "picp_recal_q": float(row.get("picp_recal_q", np.nan)),
            "mpiw_recal_q": float(row.get("mpiw_recal_q", np.nan)),
            "s_factor": float(row.get("s_factor", 1.0)) if pd.notna(row.get("s_factor")) else np.nan,
            "s_quantile": float(row.get("s_quantile", 1.0)) if pd.notna(row.get("s_quantile")) else np.nan,
            "train_time_sec": train_t,
            "inference_latency_ms": inf_lat,
            "peak_memory_mb": peak_m,
            "equation_nodes": 0,
            "equation_operators": 0,
            "equation": "",
            "hardware": "development computer, CPU, NOT Jetson",
        })

    # 3. Compile GP Variants A, B, C (Stage 4)
    df_gp = pd.read_csv("results/metrics/gp_variants.csv")
    variant_map = {
        "A": "Sparse_GPR_Wide",
        "B": "Deep_Kernel_GPR",
        "C": "WD_GPR",
    }
    for _, row in df_gp.iterrows():
        m_name = variant_map.get(row["variant"], row["model"])
        train_t = measured(row, "train_time_sec")
        inf_lat = measured(row, "inf_latency_ms")
        peak_m = measured(row, "peak_memory_mb")
        all_records.append({
            "model": m_name,
            "dataset": row["dataset"],
            "held_out_cell": row["held_out_cell"],
            "seed": int(row["seed"]),
            "n_train": int(row["n_train"]),
            "n_test": int(row["n_test"]),
            "mae": float(row["mae"]),
            "mse": float(row["mse"]),
            "rmse": float(row["rmse"]),
            "r2": float(row["r2"]),
            "mape": float(row.get("mape", np.nan)),
            "mape_soh50": float(row.get("mape_soh50", np.nan)),
            "smape": float(row.get("smape", np.nan)),
            "nll": float(row.get("nll_raw", row.get("nll", np.nan))),
            "picp": float(row.get("picp_raw", row.get("picp", np.nan))),
            "mpiw": float(row.get("mpiw_raw", row.get("mpiw", np.nan))),
            "nll_recal": float(row.get("nll_recal", np.nan)),
            "picp_recal": float(row.get("picp_recal", np.nan)),
            "mpiw_recal": float(row.get("mpiw_recal", np.nan)),
            "nll_recal_q": float(row.get("nll_recal_q", np.nan)),
            "picp_recal_q": float(row.get("picp_recal_q", np.nan)),
            "mpiw_recal_q": float(row.get("mpiw_recal_q", np.nan)),
            "s_factor": float(row.get("s_factor", 1.0)),
            "s_quantile": float(row.get("s_quantile", 1.0)) if pd.notna(row.get("s_quantile")) else np.nan,
            "train_time_sec": train_t,
            "inference_latency_ms": inf_lat,
            "peak_memory_mb": peak_m,
            "equation_nodes": 0,
            "equation_operators": 0,
            "equation": "",
            "hardware": "development computer, CPU, NOT Jetson",
        })

    # 4. Compile Symbolic Regression Models D & E (Stage 5)
    df_sr = pd.read_csv("results/metrics/sr_results.csv")
    for _, row in df_sr.iterrows():
        n_test = int(row["n_test"])
        n_tr = 432 if row["dataset"] == "NASA" else 3000

        # Config D: Unweighted SR (Fixed B2: reads row["nll_recal_unw"], row["picp_recal_unw"], row["mpiw_recal_unw"])
        train_t_d = measured(row, "train_time_sec_d")
        inf_lat_d = measured(row, "inf_latency_ms_d")
        peak_m_d = measured(row, "peak_memory_mb")
        all_records.append({
            "model": "WD_GPR_Unweighted_SR",
            "dataset": row["dataset"],
            "held_out_cell": row["held_out_cell"],
            "seed": int(row["seed"]),
            "n_train": n_tr,
            "n_test": n_test,
            "mae": float(row["mae_after_unweighted"]),
            "mse": float(row.get("mse_after_unweighted", row["rmse_after_unweighted"] ** 2)),
            "rmse": float(row["rmse_after_unweighted"]),
            "r2": float(row["r2_after_unweighted"]),
            "mape": float(row.get("mape_after_unweighted", np.nan)),
            "mape_soh50": float(row.get("mape_soh50_after_unweighted", np.nan)),
            "smape": float(row.get("smape_after_unweighted", np.nan)),
            "nll": float(row.get("nll_after_unweighted", row.get("nll_before", np.nan))),
            "picp": float(row.get("picp_after_unweighted", row.get("picp_before", np.nan))),
            "mpiw": float(row.get("mpiw_after_unweighted", row.get("mpiw_before", np.nan))),
            "nll_recal": float(row.get("nll_recal_unw", row.get("nll_recal", np.nan))),
            "picp_recal": float(row.get("picp_recal_unw", row.get("picp_recal", np.nan))),
            "mpiw_recal": float(row.get("mpiw_recal_unw", row.get("mpiw_recal", np.nan))),
            "nll_recal_q": float(row.get("nll_recal_q_unw", np.nan)),
            "picp_recal_q": float(row.get("picp_recal_q_unw", np.nan)),
            "mpiw_recal_q": float(row.get("mpiw_recal_q_unw", np.nan)),
            "s_factor": float(row.get("s_factor", 1.0)),
            "s_quantile": float(row.get("s_quantile", 1.0)) if pd.notna(row.get("s_quantile")) else np.nan,
            "train_time_sec": train_t_d,
            "inference_latency_ms": inf_lat_d,
            "peak_memory_mb": peak_m_d,
            "equation_nodes": int(row.get("nodes_unw", row.get("nodes", 0))),
            "equation_operators": int(row.get("operators_unw", row.get("operators", 0))),
            "equation": str(row.get("equation_d_forced", row.get("equation", ""))),
            "hardware": "development computer, CPU, NOT Jetson",
        })

        # Config E: Proposed WD-GPR-SR (Uncertainty-Weighted SR)
        train_t_e = measured(row, "train_time_sec_e")
        inf_lat_e = measured(row, "inf_latency_ms_e")
        peak_m_e = measured(row, "peak_memory_mb")
        all_records.append({
            "model": "WD_GPR_SR_Proposed",
            "dataset": row["dataset"],
            "held_out_cell": row["held_out_cell"],
            "seed": int(row["seed"]),
            "n_train": n_tr,
            "n_test": n_test,
            "mae": float(row["mae_after_weighted"]),
            "mse": float(row.get("mse_after_weighted", row["rmse_after_weighted"] ** 2)),
            "rmse": float(row["rmse_after_weighted"]),
            "r2": float(row["r2_after_weighted"]),
            "mape": float(row.get("mape_after_weighted", np.nan)),
            "mape_soh50": float(row.get("mape_soh50_after_weighted", np.nan)),
            "smape": float(row.get("smape_after_weighted", np.nan)),
            "nll": float(row.get("nll_after_weighted", row.get("nll_before", np.nan))),
            "picp": float(row.get("picp_after_weighted", row.get("picp_before", np.nan))),
            "mpiw": float(row.get("mpiw_after_weighted", row.get("mpiw_before", np.nan))),
            "nll_recal": float(row["nll_recal"]),
            "picp_recal": float(row["picp_recal"]),
            "mpiw_recal": float(row["mpiw_recal"]),
            "nll_recal_q": float(row.get("nll_recal_q", np.nan)),
            "picp_recal_q": float(row.get("picp_recal_q", np.nan)),
            "mpiw_recal_q": float(row.get("mpiw_recal_q", np.nan)),
            "s_factor": float(row.get("s_factor", 1.0)),
            "s_quantile": float(row.get("s_quantile", 1.0)) if pd.notna(row.get("s_quantile")) else np.nan,
            "sr_selected": bool(row.get("sr_selected", True)),
            "train_time_sec": train_t_e,
            "inference_latency_ms": inf_lat_e,
            "peak_memory_mb": peak_m_e,
            "equation_nodes": int(row["nodes"]),
            "equation_operators": int(row["operators"]),
            "equation": str(row["equation"]),
            "hardware": "development computer, CPU, NOT Jetson",
        })

        # Task 4 Active SR Variant: WD_GPR_SR_Forced
        if "rmse_e_forced" in row and pd.notna(row["rmse_e_forced"]):
            all_records.append({
                "model": "WD_GPR_SR_Forced",
                "dataset": row["dataset"],
                "held_out_cell": row["held_out_cell"],
                "seed": int(row["seed"]),
                "n_train": n_tr,
                "n_test": n_test,
                "mae": float(row["mae_e_forced"]),
                "mse": float(row.get("mse_e_forced", row["rmse_e_forced"] ** 2)),
                "rmse": float(row["rmse_e_forced"]),
                "r2": float(row.get("r2_e_forced", np.nan)),
                "mape": float(row.get("mape_e_forced", np.nan)),
                "mape_soh50": float(row.get("mape_soh50_e_forced", np.nan)),
                "smape": float(row.get("smape_e_forced", np.nan)),
                "nll": float(row.get("nll_e_forced", np.nan)),
                "picp": float(row.get("picp_e_forced", np.nan)),
                "mpiw": float(row.get("mpiw_e_forced", np.nan)),
                "nll_recal": float(row.get("nll_recal_e_forced", np.nan)),
                "picp_recal": float(row.get("picp_recal_e_forced", np.nan)),
                "mpiw_recal": float(row.get("mpiw_recal_e_forced", np.nan)),
                "nll_recal_q": float(row.get("nll_recal_q_e_forced", np.nan)),
                "picp_recal_q": float(row.get("picp_recal_q_e_forced", np.nan)),
                "mpiw_recal_q": float(row.get("mpiw_recal_q_e_forced", np.nan)),
                "s_factor": float(row.get("s_factor", 1.0)),
                "s_quantile": float(row.get("s_quantile", 1.0)) if pd.notna(row.get("s_quantile")) else np.nan,
                "sr_selected": True,
                "train_time_sec": train_t_e,
                "inference_latency_ms": inf_lat_e,
                "peak_memory_mb": peak_m_e,
                "equation_nodes": int(row.get("nodes_e_forced", 0)),
                "equation_operators": int(row.get("operators_e_forced", 0)),
                "equation": str(row.get("equation_e_forced", "")),
                "hardware": "development computer, CPU, NOT Jetson",
            })

        # Task 4 SR Variant: WD_GPR_Unweighted_SR_Sel (D_sel)
        if "rmse_d_sel" in row and pd.notna(row["rmse_d_sel"]):
            all_records.append({
                "model": "WD_GPR_Unweighted_SR_Sel",
                "dataset": row["dataset"],
                "held_out_cell": row["held_out_cell"],
                "seed": int(row["seed"]),
                "n_train": n_tr,
                "n_test": n_test,
                "mae": float(row["mae_d_sel"]),
                "mse": float(row.get("mse_d_sel", row["rmse_d_sel"] ** 2)),
                "rmse": float(row["rmse_d_sel"]),
                "r2": float(row.get("r2_d_sel", np.nan)),
                "mape": float(row.get("mape_d_sel", np.nan)),
                "mape_soh50": float(row.get("mape_soh50_d_sel", np.nan)),
                "smape": float(row.get("smape_d_sel", np.nan)),
                "nll": float(row.get("nll_d_sel", np.nan)),
                "picp": float(row.get("picp_d_sel", np.nan)),
                "mpiw": float(row.get("mpiw_d_sel", np.nan)),
                "nll_recal": float(row.get("nll_recal_d_sel", np.nan)),
                "picp_recal": float(row.get("picp_recal_d_sel", np.nan)),
                "mpiw_recal": float(row.get("mpiw_recal_d_sel", np.nan)),
                "nll_recal_q": float(row.get("nll_recal_q_d_sel", np.nan)),
                "picp_recal_q": float(row.get("picp_recal_q_d_sel", np.nan)),
                "mpiw_recal_q": float(row.get("mpiw_recal_q_d_sel", np.nan)),
                "s_factor": float(row.get("s_factor", 1.0)),
                "s_quantile": float(row.get("s_quantile", 1.0)) if pd.notna(row.get("s_quantile")) else np.nan,
                "sr_selected": bool(row.get("sr_selected_d_sel", False)),
                "train_time_sec": train_t_d,
                "inference_latency_ms": inf_lat_d,
                "peak_memory_mb": peak_m_d,
                "equation_nodes": int(row.get("nodes_unw", 0)),
                "equation_operators": int(row.get("operators_unw", 0)),
                "equation": str(row.get("equation_d_forced", "")),
                "hardware": "development computer, CPU, NOT Jetson",
            })

        # Task 4 SR Variant: WD_GPR_SR_Shrunk (E_shrunk)
        if "rmse_e_shrunk" in row and pd.notna(row["rmse_e_shrunk"]):
            all_records.append({
                "model": "WD_GPR_SR_Shrunk",
                "dataset": row["dataset"],
                "held_out_cell": row["held_out_cell"],
                "seed": int(row["seed"]),
                "n_train": n_tr,
                "n_test": n_test,
                "mae": float(row["mae_e_shrunk"]),
                "mse": float(row.get("mse_e_shrunk", row["rmse_e_shrunk"] ** 2)),
                "rmse": float(row["rmse_e_shrunk"]),
                "r2": float(row.get("r2_e_shrunk", np.nan)),
                "mape": float(row.get("mape_e_shrunk", np.nan)),
                "mape_soh50": float(row.get("mape_soh50_e_shrunk", np.nan)),
                "smape": float(row.get("smape_e_shrunk", np.nan)),
                "nll": float(row.get("nll_e_shrunk", np.nan)),
                "picp": float(row.get("picp_e_shrunk", np.nan)),
                "mpiw": float(row.get("mpiw_e_shrunk", np.nan)),
                "nll_recal": float(row.get("nll_recal_e_shrunk", np.nan)),
                "picp_recal": float(row.get("picp_recal_e_shrunk", np.nan)),
                "mpiw_recal": float(row.get("mpiw_recal_e_shrunk", np.nan)),
                "nll_recal_q": float(row.get("nll_recal_q_e_shrunk", np.nan)),
                "picp_recal_q": float(row.get("picp_recal_q_e_shrunk", np.nan)),
                "mpiw_recal_q": float(row.get("mpiw_recal_q_e_shrunk", np.nan)),
                "s_factor": float(row.get("s_factor", 1.0)),
                "s_quantile": float(row.get("s_quantile", 1.0)) if pd.notna(row.get("s_quantile")) else np.nan,
                "sr_selected": bool(float(row.get("alpha_shrunk", 0.0)) > 0.0),
                "train_time_sec": train_t_e,
                "inference_latency_ms": inf_lat_e,
                "peak_memory_mb": peak_m_e,
                "equation_nodes": int(row.get("nodes_e_forced", 0)),
                "equation_operators": int(row.get("operators_e_forced", 0)),
                "equation": str(row.get("equation_e_forced", "")),
                "hardware": "development computer, CPU, NOT Jetson",
            })

    # 5. Compile Secondary Evaluation: Pooled 8-cell LOCO
    if os.path.exists("results/metrics/pooled_8cell_results.csv"):
        df_pool = pd.read_csv("results/metrics/pooled_8cell_results.csv")
        for _, row in df_pool.iterrows():
            all_records.append(row.to_dict())

    # Save full long CSV
    df_full = pd.DataFrame(all_records)
    df_full.to_csv(long_path, index=False)
    print(f"Saved complete full_results_long.csv with {len(df_full)} rows across all protocols.")

    # 6. Generate Per-Cell Table (Averaged Across Seeds)
    per_cell_rows = []
    for (m, d, c), grp in df_full.groupby(["model", "dataset", "held_out_cell"]):
        per_cell_rows.append({
            "model": m,
            "dataset": d,
            "cell": c,
            "MAE": grp["mae"].mean(),
            "MSE": grp["mse"].mean(),
            "RMSE": grp["rmse"].mean(),
            "R2": grp["r2"].mean(),
            "MAPE": grp["mape"].mean(),
            "MAPE_SOH50": grp["mape_soh50"].mean(),
            "sMAPE": grp["smape"].mean(),
            "NLL_raw": grp["nll"].mean(),
            "PICP_raw": grp["picp"].mean(),
            "MPIW_raw": grp["mpiw"].mean(),
            "NLL_recal": grp["nll_recal"].mean(),
            "PICP_recal": grp["picp_recal"].mean(),
            "MPIW_recal": grp["mpiw_recal"].mean(),
            "NLL_recal_q": grp["nll_recal_q"].mean() if "nll_recal_q" in grp else np.nan,
            "PICP_recal_q": grp["picp_recal_q"].mean() if "picp_recal_q" in grp else np.nan,
            "MPIW_recal_q": grp["mpiw_recal_q"].mean() if "mpiw_recal_q" in grp else np.nan,
            "s_factor": grp["s_factor"].mean() if "s_factor" in grp else np.nan,
            "s_quantile": grp["s_quantile"].mean() if "s_quantile" in grp else np.nan,
            "nodes": grp["equation_nodes"].mean(),
        })
    df_per_cell = pd.DataFrame(per_cell_rows)
    df_per_cell.to_csv("results/metrics/per_cell.csv", index=False)
    print(f"Saved per-cell metrics to results/metrics/per_cell.csv ({len(df_per_cell)} entries)")

    # 7. Generate Summary Table (Separate std across cells vs std across seeds)
    summary_rows = []
    for (m, d), grp in df_full.groupby(["model", "dataset"]):
        # A. Per-cell means across seeds
        cell_grp = grp.groupby("held_out_cell").mean(numeric_only=True)
        # B. Per-seed means across cells
        seed_grp = grp.groupby("seed").mean(numeric_only=True)

        summary_rows.append({
            "model": m,
            "dataset": d,
            "nominal_level": 0.95,
            # Point Metrics
            "MAE_mean": cell_grp["mae"].mean(),
            "MAE_std": cell_grp["mae"].std(ddof=1) if len(cell_grp) > 1 else 0.0,
            "MAE_std_cells": cell_grp["mae"].std(ddof=1) if len(cell_grp) > 1 else 0.0,
            "MAE_std_seeds": seed_grp["mae"].std(ddof=1) if len(seed_grp) > 1 else 0.0,
            "MSE_mean": cell_grp["mse"].mean(),
            "MSE_std": cell_grp["mse"].std(ddof=1) if len(cell_grp) > 1 else 0.0,
            "MSE_std_cells": cell_grp["mse"].std(ddof=1) if len(cell_grp) > 1 else 0.0,
            "MSE_std_seeds": seed_grp["mse"].std(ddof=1) if len(seed_grp) > 1 else 0.0,
            "RMSE_mean": cell_grp["rmse"].mean(),
            "RMSE_std": cell_grp["rmse"].std(ddof=1) if len(cell_grp) > 1 else 0.0,
            "RMSE_std_cells": cell_grp["rmse"].std(ddof=1) if len(cell_grp) > 1 else 0.0,
            "RMSE_std_seeds": seed_grp["rmse"].std(ddof=1) if len(seed_grp) > 1 else 0.0,
            "R2_mean": cell_grp["r2"].mean(),
            "R2_std": cell_grp["r2"].std(ddof=1) if len(cell_grp) > 1 else 0.0,
            "R2_std_cells": cell_grp["r2"].std(ddof=1) if len(cell_grp) > 1 else 0.0,
            "R2_std_seeds": seed_grp["r2"].std(ddof=1) if len(seed_grp) > 1 else 0.0,
            # Percentage Metrics
            "MAPE_mean": cell_grp["mape"].mean(),
            "MAPE_SOH50_mean": cell_grp["mape_soh50"].mean(),
            "sMAPE_mean": cell_grp["smape"].mean(),
            # Uncertainty Metrics (Raw vs Recalibrated)
            "NLL_raw_mean": cell_grp["nll"].mean(),
            "PICP_raw_mean": cell_grp["picp"].mean(),
            "MPIW_raw_mean": cell_grp["mpiw"].mean(),
            "NLL_recal_mean": cell_grp["nll_recal"].mean(),
            "PICP_recal_mean": cell_grp["picp_recal"].mean(),
            "MPIW_recal_mean": cell_grp["mpiw_recal"].mean(),
            "NLL_recal_q_mean": cell_grp["nll_recal_q"].mean() if "nll_recal_q" in cell_grp else np.nan,
            "PICP_recal_q_mean": cell_grp["picp_recal_q"].mean() if "picp_recal_q" in cell_grp else np.nan,
            "MPIW_recal_q_mean": cell_grp["mpiw_recal_q"].mean() if "mpiw_recal_q" in cell_grp else np.nan,
            "s_factor_mean": cell_grp["s_factor"].mean() if "s_factor" in cell_grp else np.nan,
            "s_quantile_mean": cell_grp["s_quantile"].mean() if "s_quantile" in cell_grp else np.nan,
            # Complexity & Profiling
            "equation_nodes": cell_grp["equation_nodes"].mean(),
            "train_time_sec": grp["train_time_sec"].mean(),
            "inf_latency_ms": grp["inference_latency_ms"].mean(),
            "peak_memory_mb": grp["peak_memory_mb"].mean(),
            "SR_selected_count": int(grp["sr_selected"].sum()) if "sr_selected" in grp else 0,
            "total_evaluations": len(grp),
        })
    df_summary = pd.DataFrame(summary_rows)
    df_summary["hardware"] = "development computer, CPU, NOT Jetson"
    df_summary.to_csv("results/metrics/summary_mean_std_over_seeds.csv", index=False)
    df_summary.to_csv("results/metrics/full_benchmark_summary.csv", index=False)
    print("Saved summary metrics to results/metrics/summary_mean_std_over_seeds.csv")

    # 8. Generate Ablation Table (Configs A, B, C, D, E)
    ablation_models = [
        "Sparse_GPR_Wide",
        "Deep_Kernel_GPR",
        "WD_GPR",
        "WD_GPR_Unweighted_SR",
        "WD_GPR_SR_Proposed",
        "WD_GPR_SR_Forced",
    ]
    df_ablation = df_summary[df_summary["model"].isin(ablation_models)].copy()
    df_ablation.to_csv("results/metrics/ablation_table.csv", index=False)
    print("Saved ablation metrics to results/metrics/ablation_table.csv")

    # 8B. Generate Disentangled Ablation Breakdown (SR on mean vs Calibration on uncertainty)
    breakdown_rows = []
    cfg_labels = {
        "Sparse_GPR_Wide": "A - Wide-only SVGP",
        "Deep_Kernel_GPR": "B - Deep-only SVGP",
        "WD_GPR": "C - WD-GPR (no SR)",
        "WD_GPR_Unweighted_SR": "D - WD-GPR + Unweighted SR",
        "WD_GPR_SR_Proposed": "E - WD-GPR-SR (Proposed)",
    }
    for dset in ["CALCE", "NASA"]:
        c_row = df_summary[(df_summary["dataset"] == dset) & (df_summary["model"] == "WD_GPR")].iloc[0]
        c_rmse = c_row["RMSE_mean"]
        c_mae = c_row["MAE_mean"]
        for m_key in ["Sparse_GPR_Wide", "Deep_Kernel_GPR", "WD_GPR", "WD_GPR_Unweighted_SR", "WD_GPR_SR_Proposed"]:
            sub_m = df_summary[(df_summary["dataset"] == dset) & (df_summary["model"] == m_key)]
            if len(sub_m) > 0:
                r = sub_m.iloc[0]
                breakdown_rows.append({
                    "dataset": dset,
                    "config": cfg_labels[m_key],
                    "model": m_key,
                    "RMSE": r["RMSE_mean"],
                    "MAE": r["MAE_mean"],
                    "SR_effect_on_RMSE": r["RMSE_mean"] - c_rmse,
                    "SR_effect_on_MAE": r["MAE_mean"] - c_mae,
                    "PICP_raw": r["PICP_raw_mean"],
                    "PICP_recal": r["PICP_recal_mean"],
                    "Calibration_effect_on_PICP": r["PICP_recal_mean"] - r["PICP_raw_mean"],
                    "MPIW_raw": r["MPIW_raw_mean"],
                    "MPIW_recal": r["MPIW_recal_mean"],
                    "Calibration_effect_on_MPIW": r["MPIW_recal_mean"] - r["MPIW_raw_mean"],
                    "NLL_raw": r["NLL_raw_mean"],
                    "NLL_recal": r["NLL_recal_mean"],
                    "Calibration_effect_on_NLL": r["NLL_recal_mean"] - r["NLL_raw_mean"],
                    "SR_selected": r.get("SR_selected_count", 0),
                })
    pd.DataFrame(breakdown_rows).to_csv("results/metrics/ablation_breakdown.csv", index=False)
    print("Saved ablation breakdown to results/metrics/ablation_breakdown.csv")

    # 8C. Generate Paired Differences (E - C, E - D, E - best baseline per cell)
    paired_rows = []
    df_cell = pd.read_csv("results/metrics/per_cell.csv")
    for dset in ["NASA", "CALCE"]:
        cells = ["B0005", "B0006", "B0007", "B0018"] if dset == "NASA" else ["CS2_35", "CS2_36", "CS2_37", "CS2_38"]
        dset_cell = df_cell[df_cell["dataset"] == dset]
        
        # Determine best baseline
        baseline_candidates = ["Random_Forest", "BiLSTM", "XGBoost", "DeepNN", "SVR"]
        best_b_name = None
        best_b_rmse = float("inf")
        for b_name in baseline_candidates:
            b_sub = df_summary[(df_summary["dataset"] == dset) & (df_summary["model"] == b_name)]
            if len(b_sub) > 0 and b_sub.iloc[0]["RMSE_mean"] < best_b_rmse:
                best_b_rmse = b_sub.iloc[0]["RMSE_mean"]
                best_b_name = b_name

        for c in cells:
            r_e = dset_cell[(dset_cell["model"] == "WD_GPR_SR_Proposed") & (dset_cell["cell"] == c)]
            r_c = dset_cell[(dset_cell["model"] == "WD_GPR") & (dset_cell["cell"] == c)]
            r_d = dset_cell[(dset_cell["model"] == "WD_GPR_Unweighted_SR") & (dset_cell["cell"] == c)]
            r_b = dset_cell[(dset_cell["model"] == best_b_name) & (dset_cell["cell"] == c)]

            rmse_e = float(r_e["RMSE"].values[0]) if len(r_e) > 0 else np.nan
            rmse_c = float(r_c["RMSE"].values[0]) if len(r_c) > 0 else np.nan
            rmse_d = float(r_d["RMSE"].values[0]) if len(r_d) > 0 else np.nan
            rmse_b = float(r_b["RMSE"].values[0]) if len(r_b) > 0 else np.nan

            paired_rows.append({
                "dataset": dset,
                "cell": c,
                "rmse_E": rmse_e,
                "rmse_C": rmse_c,
                "rmse_D": rmse_d,
                "best_baseline_model": best_b_name,
                "rmse_best_baseline": rmse_b,
                "diff_E_minus_C": rmse_e - rmse_c,
                "diff_E_minus_D": rmse_e - rmse_d,
                "diff_E_minus_best_baseline": rmse_e - rmse_b,
            })
    pd.DataFrame(paired_rows).to_csv("results/metrics/paired_cell_differences.csv", index=False)
    print("Saved paired cell differences to results/metrics/paired_cell_differences.csv")

    # 9. Generate Publication Figures
    _generate_final_figures(df_full, df_summary)
    print("\nStage 6 Master Evaluation Runner completed successfully.")


def _generate_final_figures(df_full, df_summary):
    """Generates all 6 publication-ready figures."""
    plt.rcParams.update({"font.size": 10, "axes.titlesize": 12, "axes.labelsize": 11})

    df_nasa = pd.read_csv("data/processed/features_nasa.csv")
    df_calce = pd.read_csv("data/processed/features_calce.csv")

    nasa_cells = ["B0005", "B0006", "B0007", "B0018"]
    calce_cells = ["CS2_35", "CS2_36", "CS2_37", "CS2_38"]

    # Pre-load predictions for seed 42
    pred_cache = {}
    for dset_name, dset_df, cells in [("NASA", df_nasa, nasa_cells), ("CALCE", df_calce, calce_cells)]:
        for cell in cells:
            tr = dset_df[dset_df["cell_id"] != cell]
            te = dset_df[dset_df["cell_id"] == cell]
            p = PreprocessingPipeline()
            p.fit(tr[FEATURES_MAIN].values)
            X_te = p.transform(te[FEATURES_MAIN].values)
            y_te = te[TARGET_COLUMN].values

            ckpt = f"results/checkpoints/Variant_C_{dset_name}_{cell}_seed42.pt"
            if os.path.exists(ckpt):
                m = WideAndDeepSVGPRegressor()
                m.load_checkpoint(ckpt, input_dim=5)
                mu, std = m.predict(X_te)
                pred_cache[(dset_name, cell)] = {
                    "cycle": te["cycle"].values,
                    "y_true": y_te,
                    "y_pred": mu,
                    "y_std": std,
                }

    # FIG 1: Actual vs Predicted with 95% Confidence Intervals (NASA)
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    axes = axes.flatten()
    for i, cell in enumerate(nasa_cells):
        if ("NASA", cell) in pred_cache:
            d = pred_cache[("NASA", cell)]
            cyc = d["cycle"]
            low = d["y_pred"] - 1.96 * d["y_std"]
            upp = d["y_pred"] + 1.96 * d["y_std"]

            axes[i].plot(cyc, d["y_true"], "k-", lw=2.2, label="Actual SOH (%)")
            axes[i].plot(cyc, d["y_pred"], "b--", lw=1.8, label="WD-GPR-SR Mean")
            axes[i].fill_between(cyc, low, upp, color="royalblue", alpha=0.25, label="95% Confidence Interval")
            axes[i].axhline(80.0, color="crimson", linestyle=":", lw=1.4, label="EOL Threshold (80%)")
            axes[i].set_title(f"NASA Ames Held-Out: {cell}", fontweight="bold")
            axes[i].set_xlabel("Cycle Index")
            axes[i].set_ylabel("SOH (%)")
            axes[i].grid(True, linestyle="--", alpha=0.5)
            if i == 0:
                axes[i].legend(loc="upper right", framealpha=0.9)
    plt.suptitle("NASA Ames: Proposed WD-GPR-SR LOCO Predictions with 95% Confidence Intervals", fontsize=14, fontweight="bold")
    plt.tight_layout()
    plt.savefig("results/figures/final/final_predictions_per_cell.png", dpi=200)
    plt.close()
    print("Saved results/figures/final/final_predictions_per_cell.png")

    # FIG 2: SOH Degradation Curves with Predictions (CALCE)
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    axes = axes.flatten()
    for i, cell in enumerate(calce_cells):
        if ("CALCE", cell) in pred_cache:
            d = pred_cache[("CALCE", cell)]
            cyc = d["cycle"]
            low = d["y_pred"] - 1.96 * d["y_std"]
            upp = d["y_pred"] + 1.96 * d["y_std"]

            axes[i].plot(cyc, d["y_true"], "k-", lw=2.2, label="Actual SOH (%)")
            axes[i].plot(cyc, d["y_pred"], "forestgreen", lw=1.8, linestyle="--", label="WD-GPR-SR Mean")
            axes[i].fill_between(cyc, low, upp, color="mediumseagreen", alpha=0.25, label="95% CI")
            axes[i].axhline(80.0, color="gray", linestyle=":", lw=1.4, label="EOL Threshold (80%)")
            axes[i].set_title(f"CALCE CS2 Held-Out: {cell}", fontweight="bold")
            axes[i].set_xlabel("Cycle Index")
            axes[i].set_ylabel("SOH (%)")
            axes[i].grid(True, linestyle="--", alpha=0.5)
            if i == 0:
                axes[i].legend(loc="upper right", framealpha=0.9)
    plt.suptitle("CALCE CS2: SOH Degradation Curves with WD-GPR-SR Predictive Bounds", fontsize=14, fontweight="bold")
    plt.tight_layout()
    plt.savefig("results/figures/final/final_soh_degradation_curves.png", dpi=200)
    plt.close()
    print("Saved results/figures/final/final_soh_degradation_curves.png")

    # FIG 3: Model Comparison Bars (RMSE % across all evaluated models)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7))
    for ax, dset, col in [(ax1, "NASA", "#2b5c8f"), (ax2, "CALCE", "#e26d5c")]:
        sub = df_summary[df_summary["dataset"] == dset].sort_values("RMSE_mean")
        bars = ax.barh(sub["model"], sub["RMSE_mean"], xerr=sub["RMSE_std"], color=col, capsize=4, alpha=0.85)
        ax.set_xlabel("LOCO-CV RMSE (%)", fontweight="bold")
        ax.set_title(f"{dset} Dataset (Held-Out Cells)", fontweight="bold")
        ax.grid(True, linestyle="--", alpha=0.5, axis="x")
        for bar in bars:
            w = bar.get_width()
            ax.text(w + 0.08, bar.get_y() + bar.get_height() / 2, f"{w:.2f}%", va="center", fontsize=8.5)
    plt.suptitle("Comprehensive Benchmark: LOCO-CV RMSE Comparison Across All Models", fontsize=14, fontweight="bold")
    plt.tight_layout()
    plt.savefig("results/figures/final/final_model_comparison_bars.png", dpi=200)
    plt.close()
    print("Saved results/figures/final/final_model_comparison_bars.png")

    # FIG 4: Ablation Comparison Bars (A vs B vs C vs D vs E)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    abl_models = ["Sparse_GPR_Wide", "Deep_Kernel_GPR", "WD_GPR", "WD_GPR_Unweighted_SR", "WD_GPR_SR_Proposed"]
    labels = ["A: Wide Only", "B: Deep Only", "C: WD-GPR (No SR)", "D: WD-GPR + Unw. SR", "E: Proposed WD-GPR-SR"]
    colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd"]

    for ax, dset in [(ax1, "NASA"), (ax2, "CALCE")]:
        sub = df_summary[(df_summary["dataset"] == dset) & (df_summary["model"].isin(abl_models))].set_index("model").reindex(abl_models)
        x = np.arange(len(abl_models))
        bars = ax.bar(x, sub["RMSE_mean"], yerr=sub["RMSE_std"], color=colors, capsize=4, alpha=0.88, width=0.55)
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=25, ha="right", fontweight="bold", fontsize=9)
        ax.set_ylabel("LOCO-CV RMSE (%)", fontweight="bold")
        ax.set_title(f"{dset} Architectural Ablation", fontweight="bold")
        ax.grid(True, linestyle="--", alpha=0.5, axis="y")
        for bar in bars:
            h = bar.get_height()
            ax.text(bar.get_x() + bar.get_width() / 2, h + 0.06, f"{h:.2f}%", ha="center", fontsize=9, fontweight="bold")
    plt.suptitle("Architectural Ablation Analysis: Configs A, B, C, D, and E", fontsize=14, fontweight="bold")
    plt.tight_layout()
    plt.savefig("results/figures/final/final_ablation_comparison.png", dpi=200)
    plt.close()
    print("Saved results/figures/final/final_ablation_comparison.png")

    # FIG 5: Residual Error Distributions
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    for ax, dset, c in [(ax1, "NASA", "#2b5c8f"), (ax2, "CALCE", "#e26d5c")]:
        res_list = []
        cells = nasa_cells if dset == "NASA" else calce_cells
        for cell in cells:
            if (dset, cell) in pred_cache:
                d = pred_cache[(dset, cell)]
                res_list.extend(d["y_true"] - d["y_pred"])
        if res_list:
            ax.hist(res_list, bins=35, color=c, alpha=0.75, density=True, edgecolor="black")
            ax.axvline(0, color="red", linestyle="--", lw=1.5)
            ax.set_title(f"{dset} Final Residuals (Mean={np.mean(res_list):.2f}%, Std={np.std(res_list):.2f}%)", fontweight="bold")
            ax.set_xlabel("Residual Error (Actual - Predicted SOH %)")
            ax.set_ylabel("Probability Density")
            ax.grid(True, linestyle="--", alpha=0.5)
    plt.suptitle("Final LOCO-CV Residual Distributions (Proposed WD-GPR-SR)", fontsize=14, fontweight="bold")
    plt.tight_layout()
    plt.savefig("results/figures/final/final_residual_distributions.png", dpi=200)
    plt.close()
    print("Saved results/figures/final/final_residual_distributions.png")

    # FIG 6: Calibration Reliability Diagram
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5.5))
    for ax, dset in [(ax1, "NASA"), (ax2, "CALCE")]:
        ax.plot([40, 100], [40, 100], "k--", lw=1.5, label="Perfect Calibration")
        sub_d = df_full[(df_full["dataset"] == dset) & (df_full["model"] == "WD_GPR_SR_Proposed")]
        if not sub_d.empty:
            mean_picp = sub_d["picp"].mean()
            std_picp = sub_d["picp"].std()
            ax.scatter([95.0], [mean_picp], color="red", s=100, zorder=5, label=f"95% Empirical Coverage ({mean_picp:.1f}±{std_picp:.1f}%)")
            ax.errorbar([95.0], [mean_picp], yerr=[std_picp], fmt="none", color="red", capsize=5)
        ax.set_xlim(50, 100)
        ax.set_ylim(40, 105)
        ax.set_xlabel("Nominal Confidence Level (%)", fontweight="bold")
        ax.set_ylabel("Empirical Coverage (%)", fontweight="bold")
        ax.set_title(f"{dset} Predictive Reliability", fontweight="bold")
        ax.grid(True, linestyle="--", alpha=0.5)
        ax.legend(loc="lower right")
    plt.suptitle("Uncertainty Calibration: Nominal vs Empirical Coverage", fontsize=14, fontweight="bold")
    plt.tight_layout()
    plt.savefig("results/figures/final/final_calibration_reliability.png", dpi=200)
    plt.close()
    print("Saved results/figures/final/final_calibration_reliability.png")


if __name__ == "__main__":
    run_full_evaluation()
