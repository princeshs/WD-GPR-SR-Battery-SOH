"""
src/run_stage2.py
=================
Executes Stage 2 end-to-end:
1. Extracts all cycle-level features for 4 NASA cells and 4 CALCE cells.
2. Saves data/processed/features_nasa.csv, features_calce.csv, and features_all.csv.
3. Does NOT fit scalers here (leakage prevention: scalers belong inside CV folds).
4. Generates comprehensive EDA figures to results/figures/:
   - eda_soh_curves.png
   - eda_feature_distributions.png
   - eda_correlation_heatmaps.png
   - eda_feature_vs_soh_scatter.png
   - eda_temperature_and_voltage.png
   - eda_missing_values.png
5. Computes NaN counts per dataset and strongest feature-vs-SOH correlations.
"""

import os
import sys
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

sys.path.insert(0, ".")

from src.config import (
    DATASET_CONFIGS,
    FEATURES_MAIN,
    FEATURES_NASA_TEMP,
    FEATURES_DISCHARGE_DIAGNOSTIC,
    TARGET_COLUMN,
)
from src.feature_extraction import extract_all_features_for_cell


def main():
    print("=== Executing Stage 2: Feature Extraction & EDA ===")
    os.makedirs("data/processed", exist_ok=True)
    os.makedirs("results/figures", exist_ok=True)

    nasa_cells = ["B0005", "B0006", "B0007", "B0018"]
    calce_cells = ["CS2_35", "CS2_36", "CS2_37", "CS2_38"]

    nasa_feat_path = "data/processed/features_nasa.csv"
    calce_feat_path = "data/processed/features_calce.csv"
    all_feat_path = "data/processed/features_all.csv"

    if os.path.exists(all_feat_path) and os.path.getsize(all_feat_path) > 500000:
        print(f"Loading pre-extracted features from upgraded dataset: {all_feat_path}")
        df_nasa = pd.read_csv(nasa_feat_path)
        df_calce = pd.read_csv(calce_feat_path)
        df_all = pd.read_csv(all_feat_path)
    else:
        nasa_dfs = []
        calce_dfs = []

        # 1. Extract NASA Features
        for cell in nasa_cells:
            cache_path = f"data/cache/nasa_{cell}.pkl"
            with open(cache_path, "rb") as f:
                cell_data = pickle.load(f)
            cfg = DATASET_CONFIGS[cell]
            print(f"Extracting features for NASA {cell}...")
            df_cell = extract_all_features_for_cell(cell, cell_data, cfg)
            nasa_dfs.append(df_cell)

        df_nasa = pd.concat(nasa_dfs, ignore_index=True)
        df_nasa.to_csv(nasa_feat_path, index=False)
        print(f"-> Saved {nasa_feat_path}: {df_nasa.shape}")

        # 2. Extract CALCE Features
        for cell in calce_cells:
            cache_path = f"data/cache/calce_{cell}.pkl"
            with open(cache_path, "rb") as f:
                cell_data = pickle.load(f)
            cfg = DATASET_CONFIGS[cell]
            print(f"Extracting features for CALCE {cell}...")
            df_cell = extract_all_features_for_cell(cell, cell_data, cfg)
            calce_dfs.append(df_cell)

        df_calce = pd.concat(calce_dfs, ignore_index=True)
        df_calce.to_csv(calce_feat_path, index=False)
        print(f"-> Saved {calce_feat_path}: {df_calce.shape}")

        # 3. Save Unified Table
        df_all = pd.concat([df_nasa, df_calce], ignore_index=True)
        df_all.to_csv(all_feat_path, index=False)
        print(f"-> Saved {all_feat_path}: {df_all.shape}")

    # -------------------------------------------------------------
    # Statistical Analysis: NaN Counts & Feature-vs-SOH Correlations
    # -------------------------------------------------------------
    print("\n--- Missing Values (NaN Counts) ---")
    feat_cols = FEATURES_MAIN + FEATURES_NASA_TEMP + FEATURES_DISCHARGE_DIAGNOSTIC
    nan_summary = pd.DataFrame({
        "Feature": feat_cols,
        "NASA_NaN": [df_nasa[c].isna().sum() for c in feat_cols],
        "NASA_Total": len(df_nasa),
        "CALCE_NaN": [df_calce[c].isna().sum() for c in feat_cols],
        "CALCE_Total": len(df_calce),
    })
    print(nan_summary.to_string(index=False))

    print("\n--- Pearson Correlation with SOH (|r| sorted) ---")
    # Exclude outliers for clean correlation calculation
    valid_nasa = df_nasa[~df_nasa["is_outlier"]]
    valid_calce = df_calce[~df_calce["is_outlier"]]

    corr_nasa = {}
    for c in feat_cols:
        if valid_nasa[c].notna().sum() > 10:
            corr_nasa[c] = float(valid_nasa[c].corr(valid_nasa[TARGET_COLUMN]))

    corr_calce = {}
    for c in feat_cols:
        if valid_calce[c].notna().sum() > 10:
            corr_calce[c] = float(valid_calce[c].corr(valid_calce[TARGET_COLUMN]))

    top_nasa = sorted(corr_nasa.items(), key=lambda x: abs(x[1]), reverse=True)
    top_calce = sorted(corr_calce.items(), key=lambda x: abs(x[1]), reverse=True)

    print("\nTop NASA Correlations with SOH:")
    for k, v in top_nasa[:8]:
        print(f"  {k:25s}: r = {v:+.4f}")

    print("\nTop CALCE Correlations with SOH:")
    for k, v in top_calce[:8]:
        print(f"  {k:25s}: r = {v:+.4f}")

    # -------------------------------------------------------------
    # EDA Figures Generation
    # -------------------------------------------------------------
    plt.rcParams["font.sans-serif"] = "DejaVu Sans"
    plt.rcParams["axes.edgecolor"] = "#cccccc"
    plt.rcParams["axes.linewidth"] = 0.8

    # Figure 1: SOH Degradation Curves per Cell
    fig, axes = plt.subplots(2, 4, figsize=(18, 8), sharey=True)
    axes = axes.flatten()
    all_cells = ["B0005", "B0006", "B0007", "B0018", "CS2_35", "CS2_36", "CS2_37", "CS2_38"]
    palette = {"NASA": "#1f77b4", "CALCE": "#2ca02c"}

    for i, cell in enumerate(all_cells):
        ax = axes[i]
        sub = df_all[df_all["cell_id"] == cell]
        dset = sub["dataset"].iloc[0]
        v_mask = ~sub["is_outlier"]
        o_mask = sub["is_outlier"]

        ax.plot(sub["cycle"][v_mask], sub["SOH"][v_mask], color=palette[dset], lw=1.8, label="Valid")
        if o_mask.sum() > 0:
            ax.scatter(sub["cycle"][o_mask], sub["SOH"][o_mask], color="red", s=20, marker="x", zorder=5, label="Outlier")
        ax.axhline(80.0, color="gray", ls="--", lw=1.0, alpha=0.7)
        ax.set_title(f"{cell} ({dset})", fontweight="bold")
        ax.set_xlabel("Cycle")
        if i % 4 == 0:
            ax.set_ylabel("SOH (%)")
        ax.legend(loc="lower left", fontsize=8)
        ax.grid(True, linestyle=":", alpha=0.6)

    plt.suptitle("EDA 1: SOH Degradation Trajectories Across All 8 Cells", fontsize=14, fontweight="bold", y=0.98)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig("results/figures/eda_soh_curves.png", dpi=200, bbox_inches="tight")
    plt.close()
    print("Saved results/figures/eda_soh_curves.png")

    # Figure 2: Feature Distributions (FEATURES_MAIN)
    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    axes = axes.flatten()
    main_cols = ["cycle", "cc_charge_time", "cv_charge_time", "ic_peak_height", "ic_peak_voltage"]

    for i, col in enumerate(main_cols):
        ax = axes[i]
        for dset, color in palette.items():
            vals = df_all[(df_all["dataset"] == dset) & (~df_all["is_outlier"])][col].dropna()
            ax.hist(vals, bins=30, density=True, alpha=0.45, color=color, label=dset, edgecolor=color)
        ax.set_title(f"Distribution: {col}", fontweight="bold")
        ax.set_xlabel(col)
        ax.set_ylabel("Density")
        ax.legend()
        ax.grid(True, linestyle=":", alpha=0.6)

    # Hide 6th empty subplot
    axes[5].axis("off")
    plt.suptitle("EDA 2: Distributions of Inference-Valid Features by Dataset", fontsize=14, fontweight="bold", y=0.98)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig("results/figures/eda_feature_distributions.png", dpi=200, bbox_inches="tight")
    plt.close()
    print("Saved results/figures/eda_feature_distributions.png")

    # Figure 3: Correlation Heatmaps (per dataset)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 8))
    eval_cols_nasa = [c for c in feat_cols if valid_nasa[c].notna().sum() > 50] + [TARGET_COLUMN]
    eval_cols_calce = [c for c in feat_cols if valid_calce[c].notna().sum() > 50] + [TARGET_COLUMN]

    corr_mat_nasa = valid_nasa[eval_cols_nasa].corr().values
    corr_mat_calce = valid_calce[eval_cols_calce].corr().values

    im1 = ax1.imshow(corr_mat_nasa, cmap="coolwarm", vmin=-1, vmax=1)
    ax1.set_xticks(range(len(eval_cols_nasa)))
    ax1.set_yticks(range(len(eval_cols_nasa)))
    ax1.set_xticklabels(eval_cols_nasa, rotation=90, fontsize=8)
    ax1.set_yticklabels(eval_cols_nasa, fontsize=8)
    ax1.set_title("NASA Ames Correlation Matrix", fontweight="bold", fontsize=11)
    fig.colorbar(im1, ax=ax1, fraction=0.046, pad=0.04)

    im2 = ax2.imshow(corr_mat_calce, cmap="coolwarm", vmin=-1, vmax=1)
    ax2.set_xticks(range(len(eval_cols_calce)))
    ax2.set_yticks(range(len(eval_cols_calce)))
    ax2.set_xticklabels(eval_cols_calce, rotation=90, fontsize=8)
    ax2.set_yticklabels(eval_cols_calce, fontsize=8)
    ax2.set_title("CALCE Correlation Matrix", fontweight="bold", fontsize=11)
    fig.colorbar(im2, ax=ax2, fraction=0.046, pad=0.04)

    plt.suptitle("EDA 3: Feature Correlation Heatmaps by Dataset", fontsize=14, fontweight="bold", y=0.98)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig("results/figures/eda_correlation_heatmaps.png", dpi=200, bbox_inches="tight")
    plt.close()
    print("Saved results/figures/eda_correlation_heatmaps.png")

    # Figure 4: Feature vs SOH Scatter Plots
    fig, axes = plt.subplots(2, 3, figsize=(16, 9))
    axes = axes.flatten()
    scatter_feats = ["cc_charge_time", "cv_charge_time", "ic_peak_height", "ic_peak_voltage", "voltage_plateau_duration", "cycle"]

    for i, col in enumerate(scatter_feats):
        ax = axes[i]
        for dset, color in palette.items():
            sub = df_all[(df_all["dataset"] == dset) & (~df_all["is_outlier"])]
            ax.scatter(sub[col], sub[TARGET_COLUMN], color=color, alpha=0.35, s=15, label=dset)
        ax.set_title(f"SOH vs {col}", fontweight="bold")
        ax.set_xlabel(col)
        ax.set_ylabel("SOH (%)")
        ax.legend()

    plt.suptitle("EDA 4: Key Health Indicators vs. Battery SOH Degradation", fontsize=14, fontweight="bold", y=0.98)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig("results/figures/eda_feature_vs_soh_scatter.png", dpi=200, bbox_inches="tight")
    plt.close()
    print("Saved results/figures/eda_feature_vs_soh_scatter.png")

    # Figure 5: Temperature and Voltage Behavior
    fig, axes = plt.subplots(2, 2, figsize=(14, 9))
    # Subplot 1: NASA Temperature rise vs cycle
    ax = axes[0, 0]
    for cell in nasa_cells:
        sub = df_nasa[df_nasa["cell_id"] == cell]
        ax.plot(sub["cycle"], sub["temp_discharge_rise"], label=f"{cell} Disch Rise", alpha=0.8)
    ax.set_title("NASA: Discharge Temperature Rise vs Cycle", fontweight="bold")
    ax.set_xlabel("Cycle")
    ax.set_ylabel("ΔT (°C)")
    ax.legend()

    # Subplot 2: NASA Charge Temperature Rise vs Cycle
    ax = axes[0, 1]
    for cell in nasa_cells:
        sub = df_nasa[df_nasa["cell_id"] == cell]
        ax.plot(sub["cycle"], sub["temp_charge_rise"], label=f"{cell} Charge Rise", alpha=0.8)
    ax.set_title("NASA: Charge Temperature Rise vs Cycle", fontweight="bold")
    ax.set_xlabel("Cycle")
    ax.set_ylabel("ΔT (°C)")
    ax.legend()

    # Subplot 3: Voltage Plateau Duration vs SOH
    ax = axes[1, 0]
    for cell in all_cells:
        sub = df_all[(df_all["cell_id"] == cell) & (~df_all["is_outlier"])]
        dset = sub["dataset"].iloc[0]
        ax.plot(sub["cycle"], sub["voltage_plateau_duration"], label=cell, alpha=0.7)
    ax.set_title("Voltage Plateau Duration vs Cycle (All Cells)", fontweight="bold")
    ax.set_xlabel("Cycle")
    ax.set_ylabel("Plateau Duration (s)")
    ax.legend(ncol=2, fontsize=7)

    # Subplot 4: Voltage Slope vs SOH
    ax = axes[1, 1]
    for dset, color in palette.items():
        sub = df_all[(df_all["dataset"] == dset) & (~df_all["is_outlier"])]
        ax.scatter(sub["voltage_slope"], sub["SOH"], color=color, alpha=0.4, s=15, label=dset)
    ax.set_title("Discharge Voltage Slope dV/dt vs SOH", fontweight="bold")
    ax.set_xlabel("Voltage Slope (V/s)")
    ax.set_ylabel("SOH (%)")
    ax.legend()

    plt.suptitle("EDA 5: Temperature & Voltage Dynamic Characteristics", fontsize=14, fontweight="bold", y=0.98)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig("results/figures/eda_temperature_and_voltage.png", dpi=200, bbox_inches="tight")
    plt.close()
    print("Saved results/figures/eda_temperature_and_voltage.png")

    # Figure 6: Missing Value Analysis
    fig, ax = plt.subplots(figsize=(12, 6))
    nan_pct_nasa = df_nasa[feat_cols].isna().mean() * 100
    nan_pct_calce = df_calce[feat_cols].isna().mean() * 100

    x = np.arange(len(feat_cols))
    width = 0.38
    ax.bar(x - width/2, nan_pct_nasa, width, label="NASA Ames", color=palette["NASA"])
    ax.bar(x + width/2, nan_pct_calce, width, label="CALCE", color=palette["CALCE"])
    ax.set_xticks(x)
    ax.set_xticklabels(feat_cols, rotation=60, ha="right", fontsize=9)
    ax.set_ylabel("Missing Values (%)")
    ax.set_title("EDA 6: Feature Missingness Analysis Across Datasets (CALCE lacks thermal sensors)", fontweight="bold")
    ax.legend()

    plt.tight_layout()
    plt.savefig("results/figures/eda_missing_values.png", dpi=200, bbox_inches="tight")
    plt.close()
    print("Saved results/figures/eda_missing_values.png")

    print("\nStage 2 Feature Extraction and EDA Complete!")


if __name__ == "__main__":
    main()
