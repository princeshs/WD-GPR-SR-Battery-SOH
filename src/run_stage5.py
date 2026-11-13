"""
src/run_stage5.py
=================
Executes Stage 5: Uncertainty-Weighted Symbolic Regression Residual Discovery.
Implements the strict Inner Leave-One-Cell-Out Cross-Validation protocol to prevent
the near-zero-residual trap, followed by outer held-out cell evaluation.

Outputs:
  - results/metrics/sr_results.csv
  - results/metrics/sr_summary.csv
  - results/equations/eq_*.txt (raw & SymPy expressions)
  - results/figures/sr/*.png (residuals before/after & corrected trajectories)
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
from src.symbolic_regression import (
    compute_uncertainty_weights,
    SymbolicResidualRegressor,
    recalibrate_variance,
    select_sr_correction,
)


def run_stage5():
    print("=== Executing Stage 5: Uncertainty-Weighted Symbolic Regression ===")
    os.makedirs("results/metrics", exist_ok=True)
    os.makedirs("results/equations", exist_ok=True)
    os.makedirs("results/figures/sr", exist_ok=True)

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

    seeds = [42, 123, 999]
    all_sr_records = []
    predictions_store = {}

    t_start = time.time()

    for dset_name, dset_info in datasets.items():
        df_dset = dset_info["df"]
        cells = dset_info["cells"]
        print(f"\n=======================================================")
        print(f" Stage 5 LOCO-CV on {dset_name} ({len(cells)} cells, {len(df_dset)} cycles)")
        print(f"=======================================================")

        for held_out_cell in cells:
            train_cells = [c for c in cells if c != held_out_cell]
            test_df = df_dset[df_dset["cell_id"] == held_out_cell]
            X_test_raw = test_df[FEATURES_MAIN].values
            y_test = test_df[TARGET_COLUMN].values

            print(f"\n>>> Outer Held-Out Cell: {held_out_cell} (Training cells: {train_cells})")

            for seed in seeds:
                set_seed(seed)

                # =============================================================
                # STEP 1: INNER LOCO TO GENERATE OUT-OF-FOLD (OOF) RESIDUALS
                # =============================================================
                oof_features_list = []
                oof_residuals_list = []
                oof_variances_list = []
                oof_cell_ids_list = []

                for inner_val_cell in train_cells:
                    inner_train_cells = [c for c in train_cells if c != inner_val_cell]
                    df_inner_tr = df_dset[df_dset["cell_id"].isin(inner_train_cells)]
                    df_inner_val = df_dset[df_dset["cell_id"] == inner_val_cell]

                    pipe_inner = PreprocessingPipeline()
                    X_in_tr = pipe_inner.fit_transform(df_inner_tr[FEATURES_MAIN].values)
                    y_in_tr = df_inner_tr[TARGET_COLUMN].values
                    X_in_val = pipe_inner.transform(df_inner_val[FEATURES_MAIN].values)
                    y_in_val = df_inner_val[TARGET_COLUMN].values

                    # Fit inner WD-SVGP using canonical GP training recipe
                    inner_gp = train_gp_model(
                        X_in_tr,
                        y_in_tr,
                        variant="C",
                        num_inducing=64,
                        latent_dim=4,
                        hidden_dim=32,
                        lr=0.015,
                        weight_decay=1e-4,
                        epochs=80,
                        batch_size=128,
                        patience=15,
                        val_fraction=0.15,
                        cell_ids=df_inner_tr["cell_id"].values,
                        seed=seed,
                    )
                    mu_in, std_in = inner_gp.predict(X_in_val)
                    var_in = std_in ** 2

                    oof_res = y_in_val - mu_in
                    oof_features_list.append(df_inner_val[FEATURES_MAIN].values)
                    oof_residuals_list.append(oof_res)
                    oof_variances_list.append(var_in)
                    oof_cell_ids_list.append(np.full(len(oof_res), inner_val_cell))

                # Pool all out-of-fold records
                X_oof_raw = np.vstack(oof_features_list)
                r_oof = np.concatenate(oof_residuals_list)
                var_oof = np.concatenate(oof_variances_list)
                cell_ids_oof = np.concatenate(oof_cell_ids_list)

                # Fit full outer preprocessing pipeline on all training cells
                df_train_full = df_dset[df_dset["cell_id"].isin(train_cells)]
                pipe_outer = PreprocessingPipeline()
                X_train_full = pipe_outer.fit_transform(df_train_full[FEATURES_MAIN].values)
                y_train_full = df_train_full[TARGET_COLUMN].values
                X_test = pipe_outer.transform(X_test_raw)
                X_oof_scaled = pipe_outer.transform(X_oof_raw)

                # Compute recalibration factors strictly from inner OOF residuals
                s_factor = compute_recalibration_factor(r_oof, var_oof)
                s_quantile = compute_quantile_calibration_factor(r_oof, np.sqrt(np.maximum(var_oof, 1e-6)))

                # =============================================================
                # STEP 2: FAIR SR COMPARISON ACROSS 5 RULES (TASK 4)
                # =============================================================
                # 2A. E_sel: Uncertainty-Weighted SR with nested selection (incl. candidate g=0)
                sr_e_sel, eq_e_sel_str, e_sel_improved, _ = select_sr_correction(
                    X_oof=X_oof_scaled,
                    residuals_oof=r_oof,
                    variances_oof=var_oof,
                    feature_names=FEATURES_MAIN,
                    cell_ids_oof=cell_ids_oof,
                    candidate_parsimonies=(0.001, 0.005, 0.01),
                    population_size=500,
                    generations=20,
                    use_weights=True,
                    seed=seed,
                    n_jobs=2,
                )
                if e_sel_improved:
                    node_cnt_e_sel, op_cnt_e_sel = sr_e_sel.get_node_and_operator_count()
                else:
                    node_cnt_e_sel, op_cnt_e_sel = 0, 0

                # 2B. D_sel: Unweighted SR with nested selection (incl. candidate g=0)
                sr_d_sel, eq_d_sel_str, d_sel_improved, _ = select_sr_correction(
                    X_oof=X_oof_scaled,
                    residuals_oof=r_oof,
                    variances_oof=var_oof,
                    feature_names=FEATURES_MAIN,
                    cell_ids_oof=cell_ids_oof,
                    candidate_parsimonies=(0.001, 0.005, 0.01),
                    population_size=500,
                    generations=20,
                    use_weights=False,
                    seed=seed,
                    n_jobs=2,
                )
                if d_sel_improved:
                    node_cnt_d_sel, op_cnt_d_sel = sr_d_sel.get_node_and_operator_count()
                else:
                    node_cnt_d_sel, op_cnt_d_sel = 0, 0

                # 2C. D_forced: Unweighted SR always on (fixed budget pop=1000, gen=30, restricted op set)
                t_d0 = time.perf_counter()
                sr_d_forced = SymbolicResidualRegressor(
                    population_size=1000,
                    generations=30,
                    tournament_size=20,
                    parsimony_coefficient=0.005,
                    function_set=("add", "sub", "mul", "div"),
                    random_state=seed,
                    n_jobs=2,
                )
                sr_d_forced.fit(
                    X_oof_scaled,
                    r_oof,
                    sample_weight=None,
                    feature_names=FEATURES_MAIN,
                )
                d_forced_time = float(time.perf_counter() - t_d0)
                eq_d_forced_str = str(sr_d_forced.display_expr)
                node_cnt_d_forced, op_cnt_d_forced = sr_d_forced.get_node_and_operator_count()

                # 2D. E_forced: Uncertainty-Weighted SR always on (pop=1000, gen=30, restricted op set)
                t_e0 = time.perf_counter()
                w_full = compute_uncertainty_weights(var_oof)
                sr_e_forced = SymbolicResidualRegressor(
                    population_size=1000,
                    generations=30,
                    tournament_size=20,
                    parsimony_coefficient=0.005,
                    max_size=15,
                    function_set=("add", "sub", "mul", "div"),
                    random_state=seed,
                    n_jobs=2,
                )
                sr_e_forced.fit(
                    X_oof_scaled,
                    r_oof,
                    sample_weight=w_full,
                    feature_names=FEATURES_MAIN,
                )
                e_forced_time = float(time.perf_counter() - t_e0)
                eq_e_forced_str = str(sr_e_forced.display_expr)
                node_cnt_e_forced, op_cnt_e_forced = sr_e_forced.get_node_and_operator_count()

                # Save Pareto front for fold NASA B0005 seed 42 (Task 7 requirement)
                if dset_name == "NASA" and held_out_cell == "B0005" and seed == 42:
                    pfront = sr_e_forced.get_pareto_front(X_oof_scaled, r_oof)
                    df_pfront = pd.DataFrame(pfront)
                    df_pfront.to_csv("results/metrics/sr_pareto_front_b0005.csv", index=False)
                    print(f"  [Pareto Front B0005 seed 42] Extracted {len(df_pfront)} non-dominated points saved to results/metrics/sr_pareto_front_b0005.csv")

                # 2E. E_shrunk: Shrinkage alpha in {0, 0.25, 0.5, 1.0} chosen by nested rotation
                alpha_candidates = [0.0, 0.25, 0.5, 1.0]
                best_alpha = 0.0
                best_alpha_rmse = float("inf")
                # Evaluate shrinkage on inner rotations using inner OOF residuals
                for alpha_cand in alpha_candidates:
                    rot_rmses = []
                    for in_c in train_cells:
                        val_idx = np.where(cell_ids_oof == in_c)[0]
                        tr_idx = np.where(cell_ids_oof != in_c)[0]
                        sr_rot = SymbolicResidualRegressor(
                            population_size=300,
                            generations=15,
                            parsimony_coefficient=0.005,
                            function_set=("add", "sub", "mul", "div"),
                            random_state=seed,
                            n_jobs=1,
                        )
                        sr_rot.fit(X_oof_scaled[tr_idx], r_oof[tr_idx], sample_weight=compute_uncertainty_weights(var_oof[tr_idx]))
                        g_rot_pred = sr_rot.predict(X_oof_scaled[val_idx])
                        rot_rmse = float(np.sqrt(np.mean((r_oof[val_idx] - alpha_cand * g_rot_pred) ** 2)))
                        rot_rmses.append(rot_rmse)
                    mean_rot_rmse = float(np.mean(rot_rmses))
                    if mean_rot_rmse < best_alpha_rmse:
                        best_alpha_rmse = mean_rot_rmse
                        best_alpha = alpha_cand

                # =============================================================
                # STEP 3: FIT PRIMARY GP ON ALL TRAINING CELLS (IDENTICAL RECIPE)
                # =============================================================
                t_gp0 = time.perf_counter()
                final_gp = train_gp_model(
                    X_train_full,
                    y_train_full,
                    variant="C",
                    num_inducing=64,
                    latent_dim=4,
                    hidden_dim=32,
                    lr=0.015,
                    weight_decay=1e-4,
                    epochs=80,
                    batch_size=128,
                    patience=15,
                    val_fraction=0.15,
                    cell_ids=df_train_full["cell_id"].values,
                    seed=seed,
                )
                gp_train_time = float(time.perf_counter() - t_gp0)
                mu_gp, std_gp = final_gp.predict(X_test)

                # Generate predictions for all 5 rules
                g_test_e_sel = sr_e_sel.predict(X_test)
                y_pred_e_sel = mu_gp + g_test_e_sel

                g_test_d_sel = sr_d_sel.predict(X_test)
                y_pred_d_sel = mu_gp + g_test_d_sel

                g_test_d_forced = sr_d_forced.predict(X_test)
                y_pred_d_forced = mu_gp + g_test_d_forced

                g_test_e_forced = sr_e_forced.predict(X_test)
                y_pred_e_forced = mu_gp + g_test_e_forced

                y_pred_e_shrunk = mu_gp + best_alpha * g_test_e_forced

                # Calibration and evaluation across variants
                m_before = apply_shared_calibration(y_test, mu_gp, std_gp, s_factor=s_factor, s_quantile=s_quantile)
                m_e_sel = apply_shared_calibration(y_test, y_pred_e_sel, std_gp, s_factor=s_factor, s_quantile=s_quantile)
                m_d_sel = apply_shared_calibration(y_test, y_pred_d_sel, std_gp, s_factor=s_factor, s_quantile=s_quantile)
                m_d_forced = apply_shared_calibration(y_test, y_pred_d_forced, std_gp, s_factor=s_factor, s_quantile=s_quantile)
                m_e_forced = apply_shared_calibration(y_test, y_pred_e_forced, std_gp, s_factor=s_factor, s_quantile=s_quantile)
                m_e_shrunk = apply_shared_calibration(y_test, y_pred_e_shrunk, std_gp, s_factor=s_factor, s_quantile=s_quantile)

                # Timing & Latency profiling on development computer
                single_x = X_test[:1]
                for _ in range(50):
                    final_gp.predict(single_x)
                    sr_e_sel.predict(single_x)
                    sr_d_forced.predict(single_x)

                lats_gp, lats_d, lats_e = [], [], []
                for _ in range(1000):
                    t_inf = time.perf_counter()
                    final_gp.predict(single_x)
                    lats_gp.append((time.perf_counter() - t_inf) * 1000.0)

                    t_inf_d = time.perf_counter()
                    p_mu, _ = final_gp.predict(single_x)
                    _ = p_mu + sr_d_forced.predict(single_x)
                    lats_d.append((time.perf_counter() - t_inf_d) * 1000.0)

                    t_inf_e = time.perf_counter()
                    p_mu, _ = final_gp.predict(single_x)
                    _ = p_mu + sr_e_sel.predict(single_x)
                    lats_e.append((time.perf_counter() - t_inf_e) * 1000.0)

                inf_lat_gp = float(np.median(lats_gp))
                inf_lat_d = float(np.median(lats_d))
                inf_lat_e = float(np.median(lats_e))
                peak_mem = float(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0)

                # Save equation text for this fold
                eq_file = f"results/equations/eq_{dset_name}_{held_out_cell}_seed{seed}.txt"
                with open(eq_file, "w") as f:
                    f.write(f"Dataset: {dset_name}\n")
                    f.write(f"Held-Out Cell: {held_out_cell}\n")
                    f.write(f"Seed: {seed}\n")
                    f.write(f"E_sel Improved Validation: {e_sel_improved}\n")
                    f.write(f"E_sel Expression: {eq_e_sel_str}\n")
                    f.write(f"D_sel Improved Validation: {d_sel_improved}\n")
                    f.write(f"D_sel Expression: {eq_d_sel_str}\n")
                    f.write(f"D_forced Expression: {eq_d_forced_str}\n")
                    f.write(f"E_forced Expression: {eq_e_forced_str}\n")
                    f.write(f"E_shrunk Chosen Alpha: {best_alpha}\n")
                    f.write(f"Recalibration factor s (var): {s_factor:.4f}, s (quantile): {s_quantile:.4f}\n")

                record = {
                    "dataset": dset_name,
                    "held_out_cell": held_out_cell,
                    "seed": seed,
                    "n_test": len(y_test),
                    # Base GP (Config C)
                    "mae_before": m_before["mae"],
                    "mse_before": m_before["mse"],
                    "rmse_before": m_before["rmse"],
                    "r2_before": m_before["r2"],
                    "mape_before": m_before["mape"],
                    "mape_soh50_before": m_before["mape_soh50"],
                    "smape_before": m_before["smape"],
                    "nll_before": m_before["nll_raw"],
                    "picp_before": m_before["picp_raw"],
                    "mpiw_before": m_before["mpiw_raw"],
                    "nll_recal_before": m_before["nll_recal"],
                    "picp_recal_before": m_before["picp_recal"],
                    "mpiw_recal_before": m_before["mpiw_recal"],
                    "nll_recal_q_before": m_before["nll_recal_q"],
                    "picp_recal_q_before": m_before["picp_recal_q"],
                    "mpiw_recal_q_before": m_before["mpiw_recal_q"],
                    # Proposed WD-GPR-SR (Config E: E_sel)
                    "mae_after_weighted": m_e_sel["mae"],
                    "mse_after_weighted": m_e_sel["mse"],
                    "rmse_after_weighted": m_e_sel["rmse"],
                    "r2_after_weighted": m_e_sel["r2"],
                    "mape_after_weighted": m_e_sel["mape"],
                    "mape_soh50_after_weighted": m_e_sel["mape_soh50"],
                    "smape_after_weighted": m_e_sel["smape"],
                    "nll_after_weighted": m_e_sel["nll_raw"],
                    "picp_after_weighted": m_e_sel["picp_raw"],
                    "mpiw_after_weighted": m_e_sel["mpiw_raw"],
                    "nll_recal": m_e_sel["nll_recal"],
                    "picp_recal": m_e_sel["picp_recal"],
                    "mpiw_recal": m_e_sel["mpiw_recal"],
                    "nll_recal_q": m_e_sel["nll_recal_q"],
                    "picp_recal_q": m_e_sel["picp_recal_q"],
                    "mpiw_recal_q": m_e_sel["mpiw_recal_q"],
                    # Unweighted SR Ablation (Config D: D_forced) - Fixed B2 metrics
                    "mae_after_unweighted": m_d_forced["mae"],
                    "mse_after_unweighted": m_d_forced["mse"],
                    "rmse_after_unweighted": m_d_forced["rmse"],
                    "r2_after_unweighted": m_d_forced["r2"],
                    "mape_after_unweighted": m_d_forced["mape"],
                    "mape_soh50_after_unweighted": m_d_forced["mape_soh50"],
                    "smape_after_unweighted": m_d_forced["smape"],
                    "nll_after_unweighted": m_d_forced["nll_raw"],
                    "picp_after_unweighted": m_d_forced["picp_raw"],
                    "mpiw_after_unweighted": m_d_forced["mpiw_raw"],
                    "nll_recal_unw": m_d_forced["nll_recal"],
                    "picp_recal_unw": m_d_forced["picp_recal"],
                    "mpiw_recal_unw": m_d_forced["mpiw_recal"],
                    "nll_recal_q_unw": m_d_forced["nll_recal_q"],
                    "picp_recal_q_unw": m_d_forced["picp_recal_q"],
                    "mpiw_recal_q_unw": m_d_forced["mpiw_recal_q"],
                    # Task 4 Fair SR Variants - Full Metrics Export (Phase 3C Task 3)
                    # 1. E_forced (Uncertainty-Weighted SR Always Active)
                    "mae_e_forced": m_e_forced["mae"],
                    "mse_e_forced": m_e_forced["mse"],
                    "rmse_e_forced": m_e_forced["rmse"],
                    "r2_e_forced": m_e_forced["r2"],
                    "mape_e_forced": m_e_forced["mape"],
                    "mape_soh50_e_forced": m_e_forced["mape_soh50"],
                    "smape_e_forced": m_e_forced["smape"],
                    "nll_e_forced": m_e_forced["nll_raw"],
                    "picp_e_forced": m_e_forced["picp_raw"],
                    "mpiw_e_forced": m_e_forced["mpiw_raw"],
                    "nll_recal_e_forced": m_e_forced["nll_recal"],
                    "picp_recal_e_forced": m_e_forced["picp_recal"],
                    "mpiw_recal_e_forced": m_e_forced["mpiw_recal"],
                    "nll_recal_q_e_forced": m_e_forced["nll_recal_q"],
                    "picp_recal_q_e_forced": m_e_forced["picp_recal_q"],
                    "mpiw_recal_q_e_forced": m_e_forced["mpiw_recal_q"],

                    # 2. D_sel (Unweighted SR Nested Selection)
                    "mae_d_sel": m_d_sel["mae"],
                    "mse_d_sel": m_d_sel["mse"],
                    "rmse_d_sel": m_d_sel["rmse"],
                    "r2_d_sel": m_d_sel["r2"],
                    "mape_d_sel": m_d_sel["mape"],
                    "mape_soh50_d_sel": m_d_sel["mape_soh50"],
                    "smape_d_sel": m_d_sel["smape"],
                    "nll_d_sel": m_d_sel["nll_raw"],
                    "picp_d_sel": m_d_sel["picp_raw"],
                    "mpiw_d_sel": m_d_sel["mpiw_raw"],
                    "nll_recal_d_sel": m_d_sel["nll_recal"],
                    "picp_recal_d_sel": m_d_sel["picp_recal"],
                    "mpiw_recal_d_sel": m_d_sel["mpiw_recal"],
                    "nll_recal_q_d_sel": m_d_sel["nll_recal_q"],
                    "picp_recal_q_d_sel": m_d_sel["picp_recal_q"],
                    "mpiw_recal_q_d_sel": m_d_sel["mpiw_recal_q"],

                    # 3. E_shrunk (Uncertainty-Weighted SR Shrunk by Alpha)
                    "mae_e_shrunk": m_e_shrunk["mae"],
                    "mse_e_shrunk": m_e_shrunk["mse"],
                    "rmse_e_shrunk": m_e_shrunk["rmse"],
                    "r2_e_shrunk": m_e_shrunk["r2"],
                    "mape_e_shrunk": m_e_shrunk["mape"],
                    "mape_soh50_e_shrunk": m_e_shrunk["mape_soh50"],
                    "smape_e_shrunk": m_e_shrunk["smape"],
                    "nll_e_shrunk": m_e_shrunk["nll_raw"],
                    "picp_e_shrunk": m_e_shrunk["picp_raw"],
                    "mpiw_e_shrunk": m_e_shrunk["mpiw_raw"],
                    "nll_recal_e_shrunk": m_e_shrunk["nll_recal"],
                    "picp_recal_e_shrunk": m_e_shrunk["picp_recal"],
                    "mpiw_recal_e_shrunk": m_e_shrunk["mpiw_recal"],
                    "nll_recal_q_e_shrunk": m_e_shrunk["nll_recal_q"],
                    "picp_recal_q_e_shrunk": m_e_shrunk["picp_recal_q"],
                    "mpiw_recal_q_e_shrunk": m_e_shrunk["mpiw_recal_q"],
                    "alpha_shrunk": best_alpha,
                    "sr_selected": e_sel_improved,
                    "sr_selected_d_sel": d_sel_improved,
                    "equation": eq_e_sel_str,
                    "equation_d_forced": eq_d_forced_str,
                    "equation_e_forced": eq_e_forced_str,
                    "nodes": node_cnt_e_sel,
                    "operators": op_cnt_e_sel,
                    "nodes_unw": node_cnt_d_forced,
                    "operators_unw": op_cnt_d_forced,
                    "nodes_e_forced": node_cnt_e_forced,
                    "operators_e_forced": op_cnt_e_forced,
                    "s_factor": s_factor,
                    "s_quantile": s_quantile,
                    "train_time_sec_gp": gp_train_time,
                    "train_time_sec_d": gp_train_time + d_forced_time,
                    "train_time_sec_e": gp_train_time + e_forced_time,
                    "inf_latency_ms_gp": inf_lat_gp,
                    "inf_latency_ms_d": inf_lat_d,
                    "inf_latency_ms_e": inf_lat_e,
                    "peak_memory_mb": peak_mem,
                    "hardware": "development computer, CPU, NOT Jetson",
                }
                all_sr_records.append(record)

                # Cache predictions for seed 42 for visualization
                if seed == 42:
                    predictions_store[(dset_name, held_out_cell)] = {
                        "cycle": test_df["cycle"].values,
                        "y_true": y_test,
                        "mu_gp": mu_gp,
                        "y_pred_sr": y_pred_e_sel,
                        "std_gp": std_gp,
                        "res_before": y_test - mu_gp,
                        "res_after": y_test - y_pred_e_sel,
                        "eq_str": eq_e_sel_str,
                    }

            # Print cell progress
            sub_c = [r for r in all_sr_records if r["dataset"] == dset_name and r["held_out_cell"] == held_out_cell]
            b_mae = np.mean([r["mae_before"] for r in sub_c])
            a_mae = np.mean([r["mae_after_weighted"] for r in sub_c])
            b_rmse = np.mean([r["rmse_before"] for r in sub_c])
            a_rmse = np.mean([r["rmse_after_weighted"] for r in sub_c])
            print(f"  [{held_out_cell}] Before SR: MAE={b_mae:.2f}%, RMSE={b_rmse:.2f}% | After SR: MAE={a_mae:.2f}%, RMSE={a_rmse:.2f}%")

    # 1. Save results table
    df_sr = pd.DataFrame(all_sr_records)
    df_sr.to_csv("results/metrics/sr_results.csv", index=False)
    print(f"\nSaved long-format metrics to results/metrics/sr_results.csv ({len(df_sr)} evaluations)")

    # 2. Save Fair SR Comparison Table (Task 4)
    fair_rows = []
    variants_info = [
        ("E_sel (Weighted, Nested Selection)", "rmse_after_weighted", "mae_after_weighted", "sr_selected"),
        ("E_forced (Weighted, Always On)", "rmse_e_forced", "mae_e_forced", None),
        ("D_forced (Unweighted, Always On)", "rmse_after_unweighted", "mae_after_unweighted", None),
        ("D_sel (Unweighted, Nested Selection)", "rmse_d_sel", "mae_d_sel", "sr_selected_d_sel"),
        ("E_shrunk (Weighted, Nested Shrinkage)", "rmse_e_shrunk", "mae_e_shrunk", "alpha_shrunk"),
    ]
    for dset in ["NASA", "CALCE"]:
        sub = df_sr[df_sr["dataset"] == dset]
        cells = datasets[dset]["cells"]
        for label, rmse_col, mae_col, sel_col in variants_info:
            cell_rmses = {}
            for c in cells:
                c_sub = sub[sub["held_out_cell"] == c]
                cell_rmses[f"rmse_{c}"] = float(c_sub[rmse_col].mean())

            # Seed std (std across the 3 per-seed means)
            seed_means = [sub[sub["seed"] == s][rmse_col].mean() for s in [42, 123, 999]]
            seed_std = float(np.std(seed_means))

            # Selection count
            if sel_col == "sr_selected" or sel_col == "sr_selected_d_sel":
                sel_count = int(sub[sel_col].sum())
            elif sel_col == "alpha_shrunk":
                sel_count = int((sub[sel_col] > 0).sum())
            else:
                sel_count = len(sub)  # always on

            fair_rows.append({
                "rule": label,
                "dataset": dset,
                "rmse_mean": float(sub[rmse_col].mean()),
                "rmse_std_cells": float(sub[rmse_col].std()),
                "rmse_std_seeds": seed_std,
                "mae_mean": float(sub[mae_col].mean()),
                "mae_std_cells": float(sub[mae_col].std()),
                "folds_sr_selected": f"{sel_count}/{len(sub)}",
                **cell_rmses,
            })

    df_fair = pd.DataFrame(fair_rows)
    df_fair.to_csv("results/metrics/sr_fair_test.csv", index=False)
    print("Saved Fair SR comparison table to results/metrics/sr_fair_test.csv")

    # Save discovered equations for one fold of E_forced and D_forced
    fold0 = df_sr[(df_sr["dataset"] == "NASA") & (df_sr["held_out_cell"] == "B0005") & (df_sr["seed"] == 42)].iloc[0]
    with open("results/equations/discovered_equations_fair_sr.txt", "w") as f:
        f.write("=== Fair SR Discovered Equations (Fold: NASA B0005, Seed 42) ===\n\n")
        f.write(f"E_forced (Weighted, Always On):\n")
        f.write(f"  Nodes: {fold0['nodes_e_forced']}, Operators: {fold0['operators_e_forced']}\n")
        f.write(f"  Equation: {fold0['equation_e_forced']}\n\n")
        f.write(f"D_forced (Unweighted, Always On):\n")
        f.write(f"  Nodes: {fold0['nodes_unw']}, Operators: {fold0['operators_unw']}\n")
        f.write(f"  Equation: {fold0['equation_d_forced']}\n\n")
        f.write(f"E_sel (Weighted, Nested Selection):\n")
        f.write(f"  Nodes: {fold0['nodes']}, Operators: {fold0['operators']}\n")
        f.write(f"  Equation: {fold0['equation']}\n")
        f.write(f"  Selected: {fold0['sr_selected']}\n")

    # 3. Compute and save summary table
    summary_rows = []
    for dset in ["NASA", "CALCE"]:
        sub = df_sr[df_sr["dataset"] == dset]
        row = {
            "dataset": dset,
            # Before SR (Config C)
            "MAE_before_mean": sub["mae_before"].mean(),
            "MAE_before_std": sub["mae_before"].std(),
            "MSE_before_mean": sub["mse_before"].mean(),
            "MSE_before_std": sub["mse_before"].std(),
            "RMSE_before_mean": sub["rmse_before"].mean(),
            "RMSE_before_std": sub["rmse_before"].std(),
            "R2_before_mean": sub["r2_before"].mean(),
            "MAPE_before_mean": sub["mape_before"].mean(),
            "MAPE_before_std": sub["mape_before"].std(),
            "MAPE_SOH50_before_mean": sub["mape_soh50_before"].mean(),
            "sMAPE_before_mean": sub["smape_before"].mean(),
            "NLL_before_mean": sub["nll_before"].mean(),
            "PICP_before_mean": sub["picp_before"].mean(),
            "MPIW_before_mean": sub["mpiw_before"].mean(),
            "NLL_recal_before_mean": sub["nll_recal_before"].mean(),
            "PICP_recal_before_mean": sub["picp_recal_before"].mean(),
            "MPIW_recal_before_mean": sub["mpiw_recal_before"].mean(),
            "NLL_recal_q_before_mean": sub["nll_recal_q_before"].mean(),
            "PICP_recal_q_before_mean": sub["picp_recal_q_before"].mean(),
            "MPIW_recal_q_before_mean": sub["mpiw_recal_q_before"].mean(),
            # Proposed WD-GPR-SR (Config E)
            "MAE_after_weighted_mean": sub["mae_after_weighted"].mean(),
            "MAE_after_weighted_std": sub["mae_after_weighted"].std(),
            "MSE_after_weighted_mean": sub["mse_after_weighted"].mean(),
            "MSE_after_weighted_std": sub["mse_after_weighted"].std(),
            "RMSE_after_weighted_mean": sub["rmse_after_weighted"].mean(),
            "RMSE_after_weighted_std": sub["rmse_after_weighted"].std(),
            "R2_after_weighted_mean": sub["r2_after_weighted"].mean(),
            "MAPE_after_weighted_mean": sub["mape_after_weighted"].mean(),
            "MAPE_after_weighted_std": sub["mape_after_weighted"].std(),
            "MAPE_SOH50_after_weighted_mean": sub["mape_soh50_after_weighted"].mean(),
            "sMAPE_after_weighted_mean": sub["smape_after_weighted"].mean(),
            "NLL_after_weighted_mean": sub["nll_after_weighted"].mean(),
            "PICP_after_weighted_mean": sub["picp_after_weighted"].mean(),
            "MPIW_after_weighted_mean": sub["mpiw_after_weighted"].mean(),
            "NLL_recal_mean": sub["nll_recal"].mean(),
            "PICP_recal_mean": sub["picp_recal"].mean(),
            "MPIW_recal_mean": sub["mpiw_recal"].mean(),
            "NLL_recal_q_mean": sub["nll_recal_q"].mean(),
            "PICP_recal_q_mean": sub["picp_recal_q"].mean(),
            "MPIW_recal_q_mean": sub["mpiw_recal_q"].mean(),
            # Unweighted SR Ablation (Config D) - Fixed B2
            "MAE_after_unweighted_mean": sub["mae_after_unweighted"].mean(),
            "MAE_after_unweighted_std": sub["mae_after_unweighted"].std(),
            "RMSE_after_unweighted_mean": sub["rmse_after_unweighted"].mean(),
            "RMSE_after_unweighted_std": sub["rmse_after_unweighted"].std(),
            "R2_after_unweighted_mean": sub["r2_after_unweighted"].mean(),
            "MAPE_after_unweighted_mean": sub["mape_after_unweighted"].mean(),
            "MAPE_SOH50_after_unweighted_mean": sub["mape_soh50_after_unweighted"].mean(),
            "sMAPE_after_unweighted_mean": sub["smape_after_unweighted"].mean(),
            "NLL_recal_unw_mean": sub["nll_recal_unw"].mean(),
            "PICP_recal_unw_mean": sub["picp_recal_unw"].mean(),
            "MPIW_recal_unw_mean": sub["mpiw_recal_unw"].mean(),
            "NLL_recal_q_unw_mean": sub["nll_recal_q_unw"].mean(),
            "PICP_recal_q_unw_mean": sub["picp_recal_q_unw"].mean(),
            "MPIW_recal_q_unw_mean": sub["mpiw_recal_q_unw"].mean(),
            # Fair SR alternatives
            "RMSE_e_forced_mean": sub["rmse_e_forced"].mean(),
            "RMSE_d_sel_mean": sub["rmse_d_sel"].mean(),
            "RMSE_e_shrunk_mean": sub["rmse_e_shrunk"].mean(),
            "SR_selected_count": int(sub["sr_selected"].sum()),
            "SR_selected_d_sel_count": int(sub["sr_selected_d_sel"].sum()),
            "Total_folds": len(sub),
            "train_time_sec_gp_mean": sub["train_time_sec_gp"].mean(),
            "train_time_sec_d_mean": sub["train_time_sec_d"].mean(),
            "train_time_sec_e_mean": sub["train_time_sec_e"].mean(),
            "inf_latency_ms_gp_mean": sub["inf_latency_ms_gp"].mean(),
            "inf_latency_ms_d_mean": sub["inf_latency_ms_d"].mean(),
            "inf_latency_ms_e_mean": sub["inf_latency_ms_e"].mean(),
            "peak_memory_mb_mean": sub["peak_memory_mb"].mean(),
        }
        summary_rows.append(row)
    df_sr_sum = pd.DataFrame(summary_rows)
    df_sr_sum.to_csv("results/metrics/sr_summary.csv", index=False)
    print("Saved summary table to results/metrics/sr_summary.csv")

    # 3. Generate figures
    _generate_sr_figures(datasets, predictions_store, df_sr)

    print(f"\nStage 5 Execution completed in {time.time() - t_start:.1f} seconds.")


def _generate_sr_figures(datasets, predictions_store, df_sr):
    """Generates comparison plots and residual error distributions."""
    plt.rcParams.update({"font.size": 10, "axes.titlesize": 12, "axes.labelsize": 11})

    # FIGURE 1: NASA LOCO True vs GP vs WD-GPR-SR Predictions
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    axes = axes.flatten()
    for i, cell in enumerate(datasets["NASA"]["cells"]):
        key = ("NASA", cell)
        if key in predictions_store:
            d = predictions_store[key]
            cyc = d["cycle"]
            low = d["y_pred_sr"] - 1.96 * d["std_gp"]
            upp = d["y_pred_sr"] + 1.96 * d["std_gp"]

            axes[i].plot(cyc, d["y_true"], "k-", lw=2, label="Actual SOH (%)")
            axes[i].plot(cyc, d["mu_gp"], "b--", lw=1.5, label="WD-SVGP Base")
            axes[i].plot(cyc, d["y_pred_sr"], "r-", lw=1.8, label="WD-GPR-SR (Corrected)")
            axes[i].fill_between(cyc, low, upp, color="red", alpha=0.18, label="95% CI (Corrected)")
            axes[i].set_title(f"NASA Held-Out: {cell}", fontweight="bold")
            axes[i].set_xlabel("Cycle Index")
            axes[i].set_ylabel("SOH (%)")
            axes[i].grid(True, linestyle="--", alpha=0.5)
            if i == 0:
                axes[i].legend(loc="upper right", framealpha=0.9)
    plt.suptitle("NASA Ames: SOH Estimation Before vs After Uncertainty-Weighted SR Correction", fontsize=14, fontweight="bold")
    plt.tight_layout()
    plt.savefig("results/figures/sr/sr_predictions_nasa.png", dpi=200)
    plt.close()
    print("Saved results/figures/sr/sr_predictions_nasa.png")

    # FIGURE 2: CALCE LOCO True vs GP vs WD-GPR-SR Predictions
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    axes = axes.flatten()
    for i, cell in enumerate(datasets["CALCE"]["cells"]):
        key = ("CALCE", cell)
        if key in predictions_store:
            d = predictions_store[key]
            cyc = d["cycle"]
            low = d["y_pred_sr"] - 1.96 * d["std_gp"]
            upp = d["y_pred_sr"] + 1.96 * d["std_gp"]

            axes[i].plot(cyc, d["y_true"], "k-", lw=2, label="Actual SOH (%)")
            axes[i].plot(cyc, d["mu_gp"], "b--", lw=1.5, label="WD-SVGP Base")
            axes[i].plot(cyc, d["y_pred_sr"], "forestgreen", lw=1.8, label="WD-GPR-SR (Corrected)")
            axes[i].fill_between(cyc, low, upp, color="forestgreen", alpha=0.18, label="95% CI (Corrected)")
            axes[i].set_title(f"CALCE Held-Out: {cell}", fontweight="bold")
            axes[i].set_xlabel("Cycle Index")
            axes[i].set_ylabel("SOH (%)")
            axes[i].grid(True, linestyle="--", alpha=0.5)
            if i == 0:
                axes[i].legend(loc="upper right", framealpha=0.9)
    plt.suptitle("CALCE CS2: SOH Estimation Before vs After Uncertainty-Weighted SR Correction", fontsize=14, fontweight="bold")
    plt.tight_layout()
    plt.savefig("results/figures/sr/sr_predictions_calce.png", dpi=200)
    plt.close()
    print("Saved results/figures/sr/sr_predictions_calce.png")

    # FIGURE 3: Residual Distribution Before vs After SR
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    for ax, dset, color in [(ax1, "NASA", "#2b5c8f"), (ax2, "CALCE", "#e26d5c")]:
        res_b = []
        res_a = []
        for cell in datasets[dset]["cells"]:
            k = (dset, cell)
            if k in predictions_store:
                res_b.extend(predictions_store[k]["res_before"])
                res_a.extend(predictions_store[k]["res_after"])

        ax.hist(res_b, bins=35, alpha=0.55, color="gray", label=f"Before SR (Std={np.std(res_b):.2f}%)", density=True)
        ax.hist(res_a, bins=35, alpha=0.65, color=color, label=f"After SR (Std={np.std(res_a):.2f}%)", density=True)
        ax.axvline(0, color="black", linestyle="--", lw=1.2)
        ax.set_title(f"{dset} Residual Error Distribution", fontweight="bold")
        ax.set_xlabel("Residual Error (Actual - Predicted SOH %)")
        ax.set_ylabel("Probability Density")
        ax.legend(loc="upper right")
        ax.grid(True, linestyle="--", alpha=0.5)

    plt.suptitle("LOCO Residual Error Distribution: Before vs After Uncertainty-Weighted SR", fontsize=14, fontweight="bold")
    plt.tight_layout()
    plt.savefig("results/figures/sr/sr_residuals_distribution.png", dpi=200)
    plt.close()
    print("Saved results/figures/sr/sr_residuals_distribution.png")

    # FIGURE 4: Before vs After RMSE Comparison Bar Chart
    fig, ax = plt.subplots(figsize=(10, 5))
    categories = []
    rmse_before_list = []
    rmse_after_w_list = []
    rmse_after_unw_list = []

    for dset in ["NASA", "CALCE"]:
        sub = df_sr[df_sr["dataset"] == dset]
        categories.append(f"{dset} LOCO")
        rmse_before_list.append(sub["rmse_before"].mean())
        rmse_after_w_list.append(sub["rmse_after_weighted"].mean())
        rmse_after_unw_list.append(sub["rmse_after_unweighted"].mean())

    x = np.arange(len(categories))
    w = 0.25
    ax.bar(x - w, rmse_before_list, width=w, color="#7f7f7f", label="WD-SVGP Base (Before SR)", alpha=0.85)
    ax.bar(x, rmse_after_unw_list, width=w, color="#1f77b4", label="Unweighted SR (w=1)", alpha=0.85)
    ax.bar(x + w, rmse_after_w_list, width=w, color="#2ca02c", label="Uncertainty-Weighted SR (Proposed)", alpha=0.85)

    ax.set_xticks(x)
    ax.set_xticklabels(categories, fontweight="bold")
    ax.set_ylabel("LOCO-CV RMSE (%)", fontweight="bold")
    ax.set_title("RMSE Comparison: WD-SVGP Base vs Unweighted SR vs Proposed Uncertainty-Weighted SR", fontweight="bold")
    ax.legend(loc="upper right")
    ax.grid(True, linestyle="--", alpha=0.5, axis="y")

    for i in range(len(categories)):
        ax.text(x[i] - w, rmse_before_list[i] + 0.08, f"{rmse_before_list[i]:.2f}%", ha="center", fontsize=9)
        ax.text(x[i], rmse_after_unw_list[i] + 0.08, f"{rmse_after_unw_list[i]:.2f}%", ha="center", fontsize=9)
        ax.text(x[i] + w, rmse_after_w_list[i] + 0.08, f"{rmse_after_w_list[i]:.2f}%", ha="center", fontsize=9, fontweight="bold", color="#1c6b1c")

    plt.tight_layout()
    plt.savefig("results/figures/sr/sr_rmse_comparison.png", dpi=200)
    plt.close()
    print("Saved results/figures/sr/sr_rmse_comparison.png")


if __name__ == "__main__":
    run_stage5()

