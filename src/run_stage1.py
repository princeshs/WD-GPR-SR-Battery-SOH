"""
src/run_stage1.py
=================
Executes Stage 1 end-to-end:
1. Load NASA cells (B0005, B0006, B0007, B0018)
2. Load CALCE cells (CS2_35, CS2_36, CS2_37, CS2_38)
3. Validate and flag anomalies in each dataset
4. Save data/processed/<cell>.csv and data/processed/dataset_summary.csv
5. Generate reports/DATA_VALIDATION.md
6. Plot SOH vs cycle for all 8 cells to results/figures/soh_curves_check.png
"""

import os
import sys
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Add src to path
sys.path.insert(0, ".")

from src.data_loading import load_nasa, load_calce
from src.data_validation import validate_cell_data, generate_validation_markdown


def main():
    print("=== Executing Stage 1: Data Loading & Validation ===")
    os.makedirs("data/processed", exist_ok=True)
    os.makedirs("results/figures", exist_ok=True)
    os.makedirs("reports", exist_ok=True)

    nasa_cells = ["B0005", "B0006", "B0007", "B0018"]
    calce_cells = ["CS2_33", "CS2_34", "CS2_35", "CS2_36", "CS2_37", "CS2_38"]

    all_summaries = []
    validated_dfs = {}
    time_series_store = {}

    # 1. Process NASA Cells
    for cell in nasa_cells:
        cache_file = f"data/cache/nasa_{cell}.pkl"
        print(f"Loading NASA {cell}...")
        if os.path.exists(cache_file):
            import pickle
            with open(cache_file, "rb") as f:
                data = pickle.load(f)
        else:
            fpath = f"not_usable/raw_mat_files/{cell}.mat" if os.path.exists(f"not_usable/raw_mat_files/{cell}.mat") else f"{cell}.mat"
            data = load_nasa(fpath, force_reload=False)
        df_valid, stats = validate_cell_data(
            data["summary_df"],
            data["time_series"],
            nominal_capacity=data["nominal_capacity"],
            dataset="NASA",
        )
        validated_dfs[cell] = df_valid
        all_summaries.append(stats)
        time_series_store[cell] = data["time_series"]

        # Save to CSV
        csv_path = f"data/processed/{cell}.csv"
        df_valid.to_csv(csv_path, index=False)
        print(f"  -> Saved {csv_path} ({len(df_valid)} cycles, {stats['n_flagged']} flagged)")

    # 2. Process CALCE Cells
    for cell in calce_cells:
        folder = f"data/CALCE/{cell}/{cell}" if os.path.exists(f"data/CALCE/{cell}/{cell}") else f"CALCE-main/dataset/{cell}"
        print(f"Loading CALCE {cell} from {folder}...")
        data = load_calce(folder, force_reload=False)
        df_valid, stats = validate_cell_data(
            data["summary_df"],
            data["time_series"],
            nominal_capacity=data["nominal_capacity"],
            dataset="CALCE",
        )
        validated_dfs[cell] = df_valid
        all_summaries.append(stats)
        time_series_store[cell] = data["time_series"]

        # Save to CSV
        csv_path = f"data/processed/{cell}.csv"
        df_valid.to_csv(csv_path, index=False)
        print(f"  -> Saved {csv_path} ({len(df_valid)} cycles, {stats['n_flagged']} flagged)")

    # 3. Save dataset_summary.csv
    # Format requested: cell, dataset, n_cycles, C_ref, SOH_min, SOH_max, n_flagged
    summary_rows = []
    for s in all_summaries:
        summary_rows.append({
            "cell": s["cell_id"],
            "dataset": s["dataset"],
            "n_cycles": s["total_cycles"],
            "C_ref": round(s["c_ref"], 4),
            "SOH_min": round(s["soh_min"], 2),
            "SOH_max": round(s["soh_max"], 2),
            "n_flagged": s["n_flagged"],
        })

    summary_df = pd.DataFrame(summary_rows)
    summary_csv_path = "data/processed/dataset_summary.csv"
    if os.path.exists(summary_csv_path):
        old_df = pd.read_csv(summary_csv_path)
        other_cells = old_df[~old_df["cell"].isin(summary_df["cell"])]
        summary_df = pd.concat([summary_df, other_cells], ignore_index=True)
    summary_df.to_csv(summary_csv_path, index=False)
    print(f"\nSaved dataset summary to {summary_csv_path}:")
    print(summary_df.to_string(index=False))

    # 4. Write reports/DATA_VALIDATION.md
    validation_md = generate_validation_markdown(all_summaries, validated_dfs)
    with open("reports/DATA_VALIDATION.md", "w") as f:
        f.write(validation_md)
    print("\nSaved reports/DATA_VALIDATION.md")

    # 5. Plot SOH vs cycle for all cells
    all_cells = nasa_cells + calce_cells
    n_c = len(all_cells)
    n_cols = 5
    n_rows = (n_c + n_cols - 1) // n_cols
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(4 * n_cols, 4 * n_rows), sharey=False)
    axes = np.array(axes).flatten()
    for ax in axes[n_c:]:
        ax.axis("off")

    colors = {
        "NASA": "#1f77b4",
        "CALCE": "#2ca02c",
    }

    for idx, cell in enumerate(all_cells):
        ax = axes[idx]
        df_cell = validated_dfs[cell]
        dset = df_cell["dataset"].iloc[0]
        c_ref = df_cell["C_ref"].iloc[0]

        # Valid vs flagged points
        valid_mask = ~df_cell["is_outlier"]
        flag_mask = df_cell["is_outlier"]

        # Plot degradation curve
        ax.plot(
            df_cell["cycle"][valid_mask],
            df_cell["SOH"][valid_mask],
            color=colors[dset],
            linewidth=1.8,
            label="Valid Cycles",
            alpha=0.9,
        )

        # Highlight flagged outliers if any
        if flag_mask.sum() > 0:
            ax.scatter(
                df_cell["cycle"][flag_mask],
                df_cell["SOH"][flag_mask],
                color="red",
                s=25,
                zorder=5,
                label=f"Flagged ({flag_mask.sum()})",
                marker="x",
            )

        # 80% EOL reference line
        ax.axhline(80.0, color="gray", linestyle="--", linewidth=1.0, alpha=0.7, label="80% EOL")

        ax.set_title(f"{cell} ({dset}) | $C_{{ref}}={c_ref:.3f}$ Ah", fontsize=11, fontweight="bold")
        ax.set_xlabel("Cycle Number", fontsize=9)
        ax.set_ylabel("State of Health SOH (%)", fontsize=9)
        ax.grid(True, linestyle=":", alpha=0.6)
        ax.legend(loc="lower left", fontsize=8)

    plt.suptitle("Lithium-ion Battery State of Health (SOH) Degradation Trajectories (All 8 Cells)", fontsize=14, fontweight="bold", y=0.98)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])

    plot_path = "results/figures/soh_curves_check.png"
    plt.savefig(plot_path, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"\nSaved SOH degradation plot to {plot_path}")

    # Compute overall statistics
    total_cycles = sum(s["total_cycles"] for s in all_summaries)
    total_flagged = sum(s["n_flagged"] for s in all_summaries)
    print(f"\nStage 1 Complete: {len(all_cells)} cells processed, {total_cycles} total cycles, {total_flagged} cycles flagged.")


if __name__ == "__main__":
    main()

