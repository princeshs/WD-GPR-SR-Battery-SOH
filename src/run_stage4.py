"""
src/run_stage4.py
=================
Executes Stage 4: Proposed Wide-and-Deep SVGP and Ablation Study (Variants A, B, C).
Performs strict Leave-One-Cell-Out Cross-Validation (LOCO-CV) on NASA and CALCE across seeds {42, 123, 999}.
Evaluates regression metrics, uncertainty metrics, calibration curves, and binned coverage.
Saves checkpoints, figures, logs, and comparative tables.
"""

import os
import sys
import time
import resource
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, ".")
from src.config import FEATURES_MAIN, TARGET_COLUMN
from src.evaluation import (
    set_seed,
    evaluate_uncertainty_metrics,
    PreprocessingPipeline,
    compute_recalibration_factor,
    compute_quantile_calibration_factor,
    apply_shared_calibration,
)
from src.gpr_models import WideAndDeepSVGPRegressor, train_gp_model


def run_stage4():
    print("=== Executing Stage 4: Proposed Wide-and-Deep SVGP & Ablations ===")
    os.makedirs("results/metrics", exist_ok=True)
    os.makedirs("results/checkpoints", exist_ok=True)
    os.makedirs("results/figures/gp", exist_ok=True)
    os.makedirs("results/logs", exist_ok=True)

    # 1. Load feature tables and filter strictly to benchmark cells
    nasa_cells = ["B0005", "B0006", "B0007", "B0018"]
    calce_cells = ["CS2_35", "CS2_36", "CS2_37", "CS2_38"]

    df_nasa = pd.read_csv("data/processed/features_nasa.csv")
    df_calce = pd.read_csv("data/processed/features_calce.csv")

    df_nasa = df_nasa[df_nasa["cell_id"].isin(nasa_cells) & (~df_nasa["is_outlier"])].copy()
    df_calce = df_calce[df_calce["cell_id"].isin(calce_cells) & (~df_calce["is_outlier"])].copy()

    datasets = {
        "NASA": {
            "df": df_nasa,
            "cells": nasa_cells,
        },
        "CALCE": {
            "df": df_calce,
            "cells": calce_cells,
        },
    }

    variants = {
        "Variant_A": {"variant": "A", "label": "Wide Only (Linear + ARD-RBF)"},
        "Variant_B": {"variant": "B", "label": "Deep Only (MLP -> Matérn-5/2)"},
        "Variant_C": {"variant": "C", "label": "Full Wide-and-Deep (Proposed)"},
    }

    seeds = [42, 123, 999]
    all_results = []
    predictions_store = {}
    coverage_levels = [0.50, 0.80, 0.90, 0.95]
    reliability_data = []
    binned_cal_data = []

    t_start = time.time()

    for dset_name, dset_info in datasets.items():
        df_dset = dset_info["df"]
        cells = dset_info["cells"]
        print(f"\n=======================================================")
        print(f" LOCO-CV on {dset_name} ({len(cells)} cells, {len(df_dset)} cycles)")
        print(f"=======================================================")

        df_cohort = df_dset[df_dset["cell_id"].isin(cells)].copy()
        for v_name, v_info in variants.items():
            v_code = v_info["variant"]
            print(f"\n--- Model: {v_name} ({v_info['label']}) on {dset_name} ---")

            for held_out_cell in cells:
                train_df = df_cohort[df_cohort["cell_id"] != held_out_cell]
                test_df = df_cohort[df_cohort["cell_id"] == held_out_cell]

                X_train_raw = train_df[FEATURES_MAIN].values
                y_train = train_df[TARGET_COLUMN].values
                X_test_raw = test_df[FEATURES_MAIN].values
                y_test = test_df[TARGET_COLUMN].values

                for seed in seeds:
                    set_seed(seed)

                    # Strict zero-leakage preprocessing on training cells only
                    pipe = PreprocessingPipeline()
                    X_train = pipe.fit_transform(X_train_raw)
                    X_test = pipe.transform(X_test_raw)

                    # 1. Inner LOCO on training cells to compute recalibration factor s
                    train_cells = [c for c in cells if c != held_out_cell]
                    oof_res_list, oof_var_list = [], []
                    for in_val_cell in train_cells:
                        df_in_tr = train_df[train_df["cell_id"] != in_val_cell]
                        df_in_val = train_df[train_df["cell_id"] == in_val_cell]
                        pipe_in = PreprocessingPipeline().fit(df_in_tr[FEATURES_MAIN].values)
                        X_in_tr = pipe_in.transform(df_in_tr[FEATURES_MAIN].values)
                        y_in_tr = df_in_tr[TARGET_COLUMN].values
                        X_in_val = pipe_in.transform(df_in_val[FEATURES_MAIN].values)
                        y_in_val = df_in_val[TARGET_COLUMN].values

                        in_gp = train_gp_model(
                            X_in_tr,
                            y_in_tr,
                            variant=v_code,
                            num_inducing=64,
                            epochs=80,
                            lr=0.015,
                            patience=15,
                            val_fraction=0.15,
                            cell_ids=df_in_tr["cell_id"].values,
                            seed=seed,
                        )
                        mu_in, std_in = in_gp.predict(X_in_val)
                        oof_res_list.append(y_in_val - mu_in)
                        oof_var_list.append(std_in ** 2)

                    r_oof = np.concatenate(oof_res_list)
                    var_oof = np.concatenate(oof_var_list)
                    s_factor = compute_recalibration_factor(r_oof, var_oof)
                    s_quantile = compute_quantile_calibration_factor(r_oof, np.sqrt(np.maximum(var_oof, 1e-6)))

                    log_path = f"results/logs/loss_{v_name}_{dset_name}_{held_out_cell}_seed{seed}.csv"

                    # 2. Canonical GP training function on all training cells (identical recipe)
                    t_tr0 = time.perf_counter()
                    model = train_gp_model(
                        X_train,
                        y_train,
                        variant=v_code,
                        num_inducing=64,
                        latent_dim=4,
                        hidden_dim=32,
                        lr=0.015,
                        weight_decay=1e-4,
                        epochs=80,
                        batch_size=128,
                        patience=15,
                        val_fraction=0.15,
                        cell_ids=train_df["cell_id"].values,
                        seed=seed,
                        log_file=log_path,
                    )
                    train_time_sec = float(time.perf_counter() - t_tr0)

                    # Predict on held-out test cell
                    y_pred, y_std = model.predict(X_test)

                    # Measure median per-sample inference latency over 1000 single-sample predictions
                    single_x = X_test[:1]
                    for _ in range(50):
                        model.predict(single_x)
                    lats = []
                    for _ in range(1000):
                        t_l0 = time.perf_counter()
                        model.predict(single_x)
                        lats.append((time.perf_counter() - t_l0) * 1000.0)
                    inf_latency_ms = float(np.median(lats))
                    peak_mem_mb = float(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0)

                    # Shared calibration procedure (variance and quantile scaling)
                    calib_m = apply_shared_calibration(y_test, y_pred, y_std, s_factor=s_factor, s_quantile=s_quantile)

                    record = {
                        "model": v_name,
                        "variant": v_code,
                        "dataset": dset_name,
                        "held_out_cell": held_out_cell,
                        "seed": seed,
                        "n_train": len(X_train),
                        "n_test": len(X_test),
                        "mae": calib_m["mae"],
                        "mse": calib_m["mse"],
                        "rmse": calib_m["rmse"],
                        "r2": calib_m["r2"],
                        "mape": calib_m["mape"],
                        "mape_soh50": calib_m["mape_soh50"],
                        "smape": calib_m["smape"],
                        "nll_raw": calib_m["nll_raw"],
                        "picp_raw": calib_m["picp_raw"],
                        "mpiw_raw": calib_m["mpiw_raw"],
                        "nll_recal": calib_m["nll_recal"],
                        "picp_recal": calib_m["picp_recal"],
                        "mpiw_recal": calib_m["mpiw_recal"],
                        "nll_recal_q": calib_m["nll_recal_q"],
                        "picp_recal_q": calib_m["picp_recal_q"],
                        "mpiw_recal_q": calib_m["mpiw_recal_q"],
                        "s_factor": s_factor,
                        "s_quantile": s_quantile,
                        "train_time_sec": train_time_sec,
                        "inf_latency_ms": inf_latency_ms,
                        "peak_memory_mb": peak_mem_mb,
                        "num_inducing": 64,
                        "epochs": 80,
                        "hardware": "development computer, CPU, NOT Jetson",
                    }
                    all_results.append(record)

                    # Save checkpoint
                    ckpt_path = f"results/checkpoints/{v_name}_{dset_name}_{held_out_cell}_seed{seed}.pt"
                    model.save_checkpoint(ckpt_path)

                    # Store predictions for seed 42 for trajectory plotting
                    if seed == 42:
                        predictions_store[(v_name, dset_name, held_out_cell)] = {
                            "cycle": test_df["cycle"].values,
                            "y_true": y_test,
                            "y_pred": y_pred,
                            "y_std": y_std,
                        }

                    # Calibration reliability analysis at 50, 80, 90, 95%
                    from scipy.stats import norm
                    for nom_cov in coverage_levels:
                        z_val = norm.ppf(1.0 - (1.0 - nom_cov) / 2.0)
                        lower_cov = y_pred - z_val * y_std
                        upper_cov = y_pred + z_val * y_std
                        in_cov = (y_test >= lower_cov) & (y_test <= upper_cov)
                        emp_cov = float(np.mean(in_cov) * 100.0)
                        mpiw_cov = float(np.mean(upper_cov - lower_cov))
                        reliability_data.append({
                            "model": v_name,
                            "dataset": dset_name,
                            "held_out_cell": held_out_cell,
                            "seed": seed,
                            "nominal_coverage": nom_cov * 100.0,
                            "empirical_coverage": emp_cov,
                            "mpiw": mpiw_cov,
                        })

                    # Binned calibration: High SOH (>90%), Medium (80-90%), Low (<80%)
                    z95 = 1.96
                    lower95 = y_pred - z95 * y_std
                    upper95 = y_pred + z95 * y_std
                    for bin_label, mask in [
                        ("High (>90%)", y_test >= 90.0),
                        ("Medium (80-90%)", (y_test >= 80.0) & (y_test < 90.0)),
                        ("Low (<80%)", y_test < 80.0),
                    ]:
                        if np.sum(mask) > 5:
                            sub_y = y_test[mask]
                            sub_l = lower95[mask]
                            sub_u = upper95[mask]
                            b_picp = float(np.mean((sub_y >= sub_l) & (sub_y <= sub_u)) * 100.0)
                            b_mpiw = float(np.mean(sub_u - sub_l))
                            b_mae = float(np.mean(np.abs(sub_y - y_pred[mask])))
                            binned_cal_data.append({
                                "model": v_name,
                                "dataset": dset_name,
                                "held_out_cell": held_out_cell,
                                "seed": seed,
                                "soh_bin": bin_label,
                                "n_samples": int(np.sum(mask)),
                                "picp": b_picp,
                                "mpiw": b_mpiw,
                                "mae": b_mae,
                            })

            # Print quick progress for this variant
            sub_res = [r for r in all_results if r["model"] == v_name and r["dataset"] == dset_name]
            maes = [r["mae"] for r in sub_res]
            rmses = [r["rmse"] for r in sub_res]
            r2s = [r["r2"] for r in sub_res]
            picps = [r.get("picp_recal", r.get("picp", np.nan)) for r in sub_res]
            nlls = [r.get("nll_recal", r.get("nll", np.nan)) for r in sub_res]
            print(f"  Completed 12 folds: MAE={np.mean(maes):.2f}±{np.std(maes):.2f}%, RMSE={np.mean(rmses):.2f}±{np.std(rmses):.2f}%, R2={np.mean(r2s):.3f}, 95%PICP={np.mean(picps):.1f}%, NLL={np.mean(nlls):.2f}")

    # 1. Save detailed long-format metrics
    results_df = pd.DataFrame(all_results)
    results_df.to_csv("results/metrics/gp_variants.csv", index=False)
    print(f"\nSaved long-format metrics to results/metrics/gp_variants.csv ({len(results_df)} evaluations)")

    # 2. Save summary metrics
    summary_rows = []
    for (m_name, d_name), grp in results_df.groupby(["model", "dataset"]):
        row = {
            "model": m_name,
            "dataset": d_name,
            "MAE_mean": grp["mae"].mean(),
            "MAE_std": grp["mae"].std(),
            "MSE_mean": grp["mse"].mean(),
            "MSE_std": grp["mse"].std(),
            "RMSE_mean": grp["rmse"].mean(),
            "RMSE_std": grp["rmse"].std(),
            "R2_mean": grp["r2"].mean(),
            "R2_std": grp["r2"].std(),
            "MAPE_mean": grp["mape"].mean(),
            "MAPE_std": grp["mape"].std(),
            "MAPE_SOH50_mean": grp["mape_soh50"].mean(),
            "MAPE_SOH50_std": grp["mape_soh50"].std(),
            "sMAPE_mean": grp["smape"].mean(),
            "sMAPE_std": grp["smape"].std(),
            "NLL_raw_mean": grp["nll_raw"].mean(),
            "NLL_raw_std": grp["nll_raw"].std(),
            "PICP_raw_mean": grp["picp_raw"].mean(),
            "PICP_raw_std": grp["picp_raw"].std(),
            "MPIW_raw_mean": grp["mpiw_raw"].mean(),
            "MPIW_raw_std": grp["mpiw_raw"].std(),
            "NLL_recal_mean": grp["nll_recal"].mean(),
            "NLL_recal_std": grp["nll_recal"].std(),
            "PICP_recal_mean": grp["picp_recal"].mean(),
            "PICP_recal_std": grp["picp_recal"].std(),
            "MPIW_recal_mean": grp["mpiw_recal"].mean(),
            "MPIW_recal_std": grp["mpiw_recal"].std(),
            "s_factor_mean": grp["s_factor"].mean(),
            "s_factor_std": grp["s_factor"].std(),
            "s_quantile_mean": grp["s_quantile"].mean() if "s_quantile" in grp else np.nan,
            "s_quantile_std": grp["s_quantile"].std() if "s_quantile" in grp else np.nan,
            "NLL_recal_q_mean": grp["nll_recal_q"].mean() if "nll_recal_q" in grp else np.nan,
            "PICP_recal_q_mean": grp["picp_recal_q"].mean() if "picp_recal_q" in grp else np.nan,
            "MPIW_recal_q_mean": grp["mpiw_recal_q"].mean() if "mpiw_recal_q" in grp else np.nan,
            "train_time_sec_mean": grp["train_time_sec"].mean() if "train_time_sec" in grp else np.nan,
            "train_time_sec_std": grp["train_time_sec"].std() if "train_time_sec" in grp else np.nan,
            "inf_latency_ms_mean": grp["inf_latency_ms"].mean() if "inf_latency_ms" in grp else np.nan,
            "inf_latency_ms_std": grp["inf_latency_ms"].std() if "inf_latency_ms" in grp else np.nan,
            "peak_memory_mb_mean": grp["peak_memory_mb"].mean() if "peak_memory_mb" in grp else np.nan,
            "peak_memory_mb_std": grp["peak_memory_mb"].std() if "peak_memory_mb" in grp else np.nan,
        }
        summary_rows.append(row)
    summary_df = pd.DataFrame(summary_rows)
    summary_df.to_csv("results/metrics/gp_variants_summary.csv", index=False)
    print("Saved summary table to results/metrics/gp_variants_summary.csv")

    # 3. Save calibration data
    df_rel = pd.DataFrame(reliability_data)
    df_rel.to_csv("results/metrics/gp_calibration_reliability.csv", index=False)
    df_bin = pd.DataFrame(binned_cal_data)
    df_bin.to_csv("results/metrics/gp_calibration_binned.csv", index=False)
    print("Saved calibration data to results/metrics/gp_calibration_*.csv")

    # 4. Generate Figures
    _generate_figures(datasets, predictions_store, summary_df, df_rel, df_bin)

    print(f"\nStage 4 Execution completed in {time.time() - t_start:.1f} seconds.")


def _generate_figures(datasets, predictions_store, summary_df, df_rel, df_bin):
    """Generates publication-quality figures for Stage 4."""
    plt.rcParams.update({"font.size": 10, "axes.titlesize": 12, "axes.labelsize": 11})

    # FIGURE 1: LOCO Predictive Trajectories on NASA for Variant C
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    axes = axes.flatten()
    nasa_cells = datasets["NASA"]["cells"]
    for i, cell in enumerate(nasa_cells):
        key = ("Variant_C", "NASA", cell)
        if key in predictions_store:
            data = predictions_store[key]
            cyc = data["cycle"]
            y_t = data["y_true"]
            y_p = data["y_pred"]
            y_s = data["y_std"]
            low = y_p - 1.96 * y_s
            upp = y_p + 1.96 * y_s

            axes[i].plot(cyc, y_t, "k-", lw=2, label="Actual SOH (%)")
            axes[i].plot(cyc, y_p, "b--", lw=1.8, label="WD-SVGP Mean")
            axes[i].fill_between(cyc, low, upp, color="blue", alpha=0.22, label="95% CI (Obs-inclusive)")
            axes[i].axhline(80.0, color="red", linestyle=":", lw=1.2, label="EOL (80%)")
            axes[i].set_title(f"NASA Held-Out Cell: {cell}", fontweight="bold")
            axes[i].set_xlabel("Cycle Index")
            axes[i].set_ylabel("SOH (%)")
            axes[i].grid(True, linestyle="--", alpha=0.5)
            if i == 0:
                axes[i].legend(loc="upper right", framealpha=0.9)
    plt.suptitle("Proposed WD-SVGP (Variant C) LOCO-CV Predictive Trajectories with 95% Confidence Intervals (NASA)", fontsize=14, fontweight="bold")
    plt.tight_layout()
    plt.savefig("results/figures/gp/predictions_variant_C_nasa.png", dpi=200)
    plt.close()
    print("Saved results/figures/gp/predictions_variant_C_nasa.png")

    # FIGURE 2: LOCO Predictive Trajectories on CALCE for Variant C
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    axes = axes.flatten()
    calce_cells = datasets["CALCE"]["cells"]
    for i, cell in enumerate(calce_cells):
        key = ("Variant_C", "CALCE", cell)
        if key in predictions_store:
            data = predictions_store[key]
            cyc = data["cycle"]
            y_t = data["y_true"]
            y_p = data["y_pred"]
            y_s = data["y_std"]
            low = y_p - 1.96 * y_s
            upp = y_p + 1.96 * y_s

            axes[i].plot(cyc, y_t, "k-", lw=2, label="Actual SOH (%)")
            axes[i].plot(cyc, y_p, "crimson", lw=1.8, linestyle="--", label="WD-SVGP Mean")
            axes[i].fill_between(cyc, low, upp, color="crimson", alpha=0.22, label="95% CI (Obs-inclusive)")
            axes[i].axhline(80.0, color="gray", linestyle=":", lw=1.2, label="EOL (80%)")
            axes[i].set_title(f"CALCE Held-Out Cell: {cell}", fontweight="bold")
            axes[i].set_xlabel("Cycle Index")
            axes[i].set_ylabel("SOH (%)")
            axes[i].grid(True, linestyle="--", alpha=0.5)
            if i == 0:
                axes[i].legend(loc="upper right", framealpha=0.9)
    plt.suptitle("Proposed WD-SVGP (Variant C) LOCO-CV Predictive Trajectories with 95% Confidence Intervals (CALCE)", fontsize=14, fontweight="bold")
    plt.tight_layout()
    plt.savefig("results/figures/gp/predictions_variant_C_calce.png", dpi=200)
    plt.close()
    print("Saved results/figures/gp/predictions_variant_C_calce.png")

    # FIGURE 3: Comparative Bar Chart (Variants A, B, C vs Stage 3 Baselines)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7))

    # Load Stage 3 baseline summary for direct comparison
    base_summary_path = "results/metrics/baselines_summary.csv"
    if os.path.exists(base_summary_path):
        df_base = pd.read_csv(base_summary_path)
        # Select key GP baselines
        target_models = ["Exact_GPR_RBF", "Exact_GPR_Matern52", "Exact_GPR_ARD", "Sparse_GPR_SVGP", "Deep_Kernel_GPR"]
        df_base_sub = df_base[df_base["model"].isin(target_models)].copy()
    else:
        df_base_sub = pd.DataFrame()

    for ax, dset, color in [(ax1, "NASA", "#2b5c8f"), (ax2, "CALCE", "#e26d5c")]:
        sub_var = summary_df[summary_df["dataset"] == dset].copy()
        if not df_base_sub.empty:
            sub_base = df_base_sub[df_base_sub["dataset"] == dset].copy()
            combined = pd.concat([sub_base, sub_var], ignore_index=True)
        else:
            combined = sub_var

        combined = combined.sort_values("RMSE_mean", ascending=True)
        # Highlight Variant C in green
        bar_colors = ["#2ca02c" if "Variant_C" in m else ("#1f77b4" if "Variant" in m else "#7f7f7f") for m in combined["model"]]

        bars = ax.barh(combined["model"], combined["RMSE_mean"], xerr=combined["RMSE_std"], color=bar_colors, capsize=4, alpha=0.85)
        ax.set_xlabel("LOCO-CV RMSE (%)", fontweight="bold")
        ax.set_title(f"{dset} Dataset (Held-Out Cells)", fontweight="bold")
        ax.grid(True, linestyle="--", alpha=0.5, axis="x")

        # Add text labels on bars
        for bar in bars:
            w = bar.get_width()
            ax.text(w + 0.1, bar.get_y() + bar.get_height() / 2, f"{w:.2f}%", va="center", fontsize=9)

    plt.suptitle("Ablation Study: Proposed WD-SVGP (Variants A, B, C) vs GP Baselines (RMSE %)", fontsize=14, fontweight="bold")
    plt.tight_layout()
    plt.savefig("results/figures/gp/variants_rmse_mae_comparison.png", dpi=200)
    plt.close()
    print("Saved results/figures/gp/variants_rmse_mae_comparison.png")

    # FIGURE 4: Reliability Diagram (Nominal vs Empirical Coverage at 50, 80, 90, 95%)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    for ax, dset in [(ax1, "NASA"), (ax2, "CALCE")]:
        sub_rel = df_rel[df_rel["dataset"] == dset]
        ax.plot([40, 100], [40, 100], "k--", lw=1.5, label="Ideal Calibration")

        markers = {"Variant_A": "s", "Variant_B": "^", "Variant_C": "o"}
        colors = {"Variant_A": "#1f77b4", "Variant_B": "#ff7f0e", "Variant_C": "#2ca02c"}

        for v_name in ["Variant_A", "Variant_B", "Variant_C"]:
            v_data = sub_rel[sub_rel["model"] == v_name].groupby("nominal_coverage")["empirical_coverage"].agg(["mean", "std"]).reset_index()
            ax.errorbar(
                v_data["nominal_coverage"],
                v_data["mean"],
                yerr=v_data["std"],
                fmt=f"-{markers[v_name]}",
                color=colors[v_name],
                lw=2,
                capsize=4,
                label=f"{v_name} (Mean±Std)",
            )

        ax.set_xlim(45, 100)
        ax.set_ylim(20, 102)
        ax.set_xlabel("Nominal Coverage Level (%)", fontweight="bold")
        ax.set_ylabel("Empirical Coverage (%)", fontweight="bold")
        ax.set_title(f"{dset} Uncertainty Calibration Reliability", fontweight="bold")
        ax.grid(True, linestyle="--", alpha=0.5)
        ax.legend(loc="lower right")

    plt.suptitle("Uncertainty Quantification: Reliability Curves (Empirical vs Nominal Coverage)", fontsize=14, fontweight="bold")
    plt.tight_layout()
    plt.savefig("results/figures/gp/reliability_diagram.png", dpi=200)
    plt.close()
    print("Saved results/figures/gp/reliability_diagram.png")

    # FIGURE 5: Binned Calibration (PICP and MPIW across High, Medium, Low SOH)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    bin_order = ["High (>90%)", "Medium (80-90%)", "Low (<80%)"]
    
    # NASA Binned PICP
    nasa_bins = df_bin[(df_bin["dataset"] == "NASA") & (df_bin["model"] == "Variant_C")]
    if not nasa_bins.empty:
        grouped_p = nasa_bins.groupby("soh_bin")["picp"].mean().reindex(bin_order)
        grouped_w = nasa_bins.groupby("soh_bin")["mpiw"].mean().reindex(bin_order)
        x = np.arange(len(bin_order))
        ax1.bar(x - 0.15, grouped_p.values, width=0.3, color="#2ca02c", label="95% PICP (%)")
        ax1.axhline(95.0, color="red", linestyle="--", label="Nominal 95%")
        ax1.set_xticks(x)
        ax1.set_xticklabels(bin_order)
        ax1.set_ylabel("Coverage (%)")
        ax1.set_title("NASA Variant C: PICP by Degradation State", fontweight="bold")
        ax1.legend(loc="lower left")
        ax1.grid(True, linestyle="--", alpha=0.5, axis="y")

        ax1_twin = ax1.twinx()
        ax1_twin.plot(x + 0.15, grouped_w.values, "b-o", lw=2, label="MPIW (%)")
        ax1_twin.set_ylabel("Mean Interval Width (MPIW %)", color="blue")
        ax1_twin.tick_params(axis="y", labelcolor="blue")

    # CALCE Binned PICP
    calce_bins = df_bin[(df_bin["dataset"] == "CALCE") & (df_bin["model"] == "Variant_C")]
    if not calce_bins.empty:
        grouped_p = calce_bins.groupby("soh_bin")["picp"].mean().reindex(bin_order)
        grouped_w = calce_bins.groupby("soh_bin")["mpiw"].mean().reindex(bin_order)
        x = np.arange(len(bin_order))
        ax2.bar(x - 0.15, grouped_p.values, width=0.3, color="#2ca02c", label="95% PICP (%)")
        ax2.axhline(95.0, color="red", linestyle="--", label="Nominal 95%")
        ax2.set_xticks(x)
        ax2.set_xticklabels(bin_order)
        ax2.set_ylabel("Coverage (%)")
        ax2.set_title("CALCE Variant C: PICP by Degradation State", fontweight="bold")
        ax2.legend(loc="lower left")
        ax2.grid(True, linestyle="--", alpha=0.5, axis="y")

        ax2_twin = ax2.twinx()
        ax2_twin.plot(x + 0.15, grouped_w.values, "b-o", lw=2, label="MPIW (%)")
        ax2_twin.set_ylabel("Mean Interval Width (MPIW %)", color="blue")
        ax2_twin.tick_params(axis="y", labelcolor="blue")

    plt.suptitle("Degradation-Dependent Uncertainty Calibration (Variant C)", fontsize=14, fontweight="bold")
    plt.tight_layout()
    plt.savefig("results/figures/gp/binned_calibration.png", dpi=200)
    plt.close()
    print("Saved results/figures/gp/binned_calibration.png")


if __name__ == "__main__":
    run_stage4()

