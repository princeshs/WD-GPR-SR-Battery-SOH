"""
data/upgrade_dataset.py
=======================
Upgrades the battery dataset from 8 cells (~4.1 MB) to 40 cells (~100 MB):
- 34 NASA Li-ion battery cells (B0005..B0056 from data/cleaned_dataset and MATLAB cache)
- 6 CALCE Li-ion battery cells (CS2_33..CS2_38 from data/CALCE)

Produces in data/processed/:
1. Per-cell summary CSVs for all 40 cells (<cell>.csv)
2. data/processed/dataset_summary.csv
3. data/processed/features_nasa.csv
4. data/processed/features_calce.csv
5. data/processed/features_all.csv
6. data/processed/cycle_timeseries.csv (uniformly resampled charge & discharge curves)
   Targeting ~100 MB total directory size.
"""

import os
import sys
import re
import time
import pickle
import datetime
import numpy as np
import pandas as pd

# Add project root to sys.path so we can import src
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.config import DATASET_CONFIGS, FEATURES_MAIN, TARGET_COLUMN
from src.data_loading import compute_reference_capacity, load_calce
from src.data_validation import validate_cell_data
from src.feature_extraction import extract_cycle_features, extract_all_features_for_cell


def parse_mat_time(s):
    if not isinstance(s, str):
        return None
    nums = [float(x) for x in re.findall(r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?", s)]
    if len(nums) >= 6:
        try:
            return datetime.datetime(
                int(nums[0]), int(nums[1]), int(nums[2]),
                int(nums[3]), int(nums[4]), int(nums[5])
            ).strftime("%Y-%m-%d %H:%M:%S")
        except Exception:
            return None
    return None


def resample_step(t, v, i, temp, n_pts=80):
    """
    Uniformly resamples a single step's time series to n_pts points.
    """
    if t is None or len(t) < 2 or (t[-1] - t[0]) <= 0:
        t_new = np.linspace(0, 1, n_pts)
        v_new = np.full(n_pts, float(v[0]) if v is not None and len(v) > 0 else 0.0)
        i_new = np.full(n_pts, float(i[0]) if i is not None and len(i) > 0 else 0.0)
        temp_new = np.full(n_pts, float(temp[0]) if temp is not None and len(temp) > 0 else np.nan)
        return t_new, v_new, i_new, temp_new

    t_rel = t - t[0]
    duration = float(t_rel[-1])
    t_new = np.linspace(0, duration, n_pts)
    v_new = np.interp(t_new, t_rel, v)
    i_new = np.interp(t_new, t_rel, i)
    if temp is not None and len(temp) == len(t) and not np.all(np.isnan(temp)):
        temp_new = np.interp(t_new, t_rel, temp)
    else:
        temp_new = np.full(n_pts, np.nan)
    return t_new, v_new, i_new, temp_new


def safe_float(val):
    if pd.isna(val):
        return np.nan
    try:
        return float(val)
    except (ValueError, TypeError):
        return np.nan


def load_nasa_from_cleaned(cell_id, meta_df, cleaned_data_dir="data/cleaned_dataset/data"):
    """
    Parses a NASA cell from data/cleaned_dataset/ into the standard cell_data dict format.
    """
    cell_meta = meta_df[meta_df["battery_id"] == cell_id].sort_values("test_id")
    dis_rows = cell_meta[cell_meta["type"] == "discharge"].copy()

    cycle_records = []
    time_series = {}

    nominal_cap = 2.0
    valid_caps = []

    for idx, dis in enumerate(dis_rows.itertuples()):
        cycle_num = idx + 1
        cap_val = safe_float(dis.Capacity)
        if not np.isnan(cap_val) and cap_val > 0:
            valid_caps.append(cap_val)

        # Preceding charge
        chg_candidates = cell_meta[(cell_meta["test_id"] < dis.test_id) & (cell_meta["type"] == "charge")]
        chg_file = chg_candidates.iloc[-1]["filename"] if len(chg_candidates) > 0 else None

        # Load discharge time series
        dis_path = os.path.join(cleaned_data_dir, dis.filename)
        df_dis = pd.read_csv(dis_path)
        v_dis = df_dis["Voltage_measured"].values.astype(float)
        i_dis = df_dis["Current_measured"].values.astype(float)
        t_dis = df_dis["Time"].values.astype(float)
        temp_dis = df_dis["Temperature_measured"].values.astype(float) if "Temperature_measured" in df_dis.columns else None

        # Load charge time series
        if chg_file:
            chg_path = os.path.join(cleaned_data_dir, chg_file)
            df_chg = pd.read_csv(chg_path)
            v_chg = df_chg["Voltage_measured"].values.astype(float)
            i_chg = df_chg["Current_measured"].values.astype(float)
            t_chg = df_chg["Time"].values.astype(float)
            temp_chg = df_chg["Temperature_measured"].values.astype(float) if "Temperature_measured" in df_chg.columns else None
            chg_dur = float(t_chg[-1] - t_chg[0]) if len(t_chg) > 1 else 0.0
        else:
            v_chg, i_chg, t_chg, temp_chg = np.array([]), np.array([]), np.array([]), None
            chg_dur = 0.0

        dis_dur = float(t_dis[-1] - t_dis[0]) if len(t_dis) > 1 else 0.0
        v_min = float(np.min(v_dis)) if len(v_dis) > 0 else np.nan
        v_max = float(np.max(v_chg)) if len(v_chg) > 0 else (float(np.max(v_dis)) if len(v_dis) > 0 else np.nan)
        temp_mean = float(np.mean(temp_dis)) if temp_dis is not None and len(temp_dis) > 0 else np.nan
        temp_max = float(np.max(temp_dis)) if temp_dis is not None and len(temp_dis) > 0 else np.nan
        dt_str = parse_mat_time(dis.start_time)

        cycle_records.append({
            "cell_id": cell_id,
            "dataset": "NASA",
            "cycle": cycle_num,
            "date_time": dt_str,
            "capacity": cap_val,
            "ambient_temp": float(dis.ambient_temperature) if hasattr(dis, "ambient_temperature") else 24.0,
            "charge_duration_s": chg_dur,
            "discharge_duration_s": dis_dur,
            "v_min": v_min,
            "v_max": v_max,
            "temp_mean": temp_mean,
            "temp_max": temp_max,
        })

        time_series[cycle_num] = {
            "charge": {"time": t_chg, "voltage": v_chg, "current": i_chg, "temperature": temp_chg},
            "discharge": {"time": t_dis, "voltage": v_dis, "current": i_dis, "temperature": temp_dis},
        }

    c_ref = float(np.median(valid_caps[:3])) if len(valid_caps) >= 1 and np.median(valid_caps[:3]) > 0 else nominal_cap
    df_summary = pd.DataFrame(cycle_records)
    if len(df_summary) > 0:
        df_summary["C_ref"] = c_ref
        df_summary["SOH"] = (df_summary["capacity"] / c_ref) * 100.0
    else:
        df_summary["C_ref"] = []
        df_summary["SOH"] = []

    return {
        "cell_id": cell_id,
        "dataset": "NASA",
        "nominal_capacity": nominal_cap,
        "c_ref": c_ref,
        "summary_df": df_summary,
        "time_series": time_series,
    }


def main():
    print("=" * 60)
    print("Starting Dataset Upgrade to ~100 MB (40 Cells, NASA + CALCE)")
    print("=" * 60)

    out_dir = "data/processed"
    cache_dir = "data/cache"
    os.makedirs(out_dir, exist_ok=True)
    os.makedirs(cache_dir, exist_ok=True)

    # 1. Identify all CALCE cells
    calce_cells = ["CS2_33", "CS2_34", "CS2_35", "CS2_36", "CS2_37", "CS2_38"]

    # 2. Identify all NASA cells
    meta_df = pd.read_csv("data/cleaned_dataset/metadata.csv")
    nasa_cells = sorted(meta_df["battery_id"].unique().tolist())
    print(f"Discovered {len(calce_cells)} CALCE cells: {calce_cells}")
    print(f"Discovered {len(nasa_cells)} NASA cells: {nasa_cells}")

    all_dataset_summaries = []
    nasa_feature_dfs = []
    calce_feature_dfs = []

    # File to stream uniform time series (charge + discharge)
    ts_csv_path = os.path.join(out_dir, "cycle_timeseries.csv")
    ts_f = open(ts_csv_path, "w", encoding="utf-8")
    ts_f.write("cell_id,dataset,cycle,step_type,step_time_s,voltage,current,temperature,capacity,soh\n")

    # Target points per step: 92 for charge, 92 for discharge => 184 per cycle (~100 MB total directory)
    N_PTS_STEP = 92

    # -------------------------------------------------------------
    # Process NASA cells
    # -------------------------------------------------------------
    print("\n--- Processing 34 NASA Cells ---")
    nasa_defaults = DATASET_CONFIGS["NASA"]

    for idx, cell in enumerate(nasa_cells):
        t0 = time.time()
        cache_file = os.path.join(cache_dir, f"nasa_{cell}.pkl")

        # Load from cache or cleaned_dataset
        if os.path.exists(cache_file):
            with open(cache_file, "rb") as f:
                cell_data = pickle.load(f)
        else:
            cell_data = load_nasa_from_cleaned(cell, meta_df)
            with open(cache_file, "wb") as f:
                pickle.dump(cell_data, f)

        # Validate
        df_valid, stats = validate_cell_data(
            cell_data["summary_df"],
            cell_data["time_series"],
            nominal_capacity=cell_data["nominal_capacity"],
            dataset="NASA",
        )
        cell_data["summary_df"] = df_valid
        all_dataset_summaries.append(stats)

        # Save individual cell CSV
        cell_csv_path = os.path.join(out_dir, f"{cell}.csv")
        df_valid.to_csv(cell_csv_path, index=False)

        # Feature extraction
        cfg = DATASET_CONFIGS.get(cell, nasa_defaults)
        df_feats = extract_all_features_for_cell(cell, cell_data, cfg)
        nasa_feature_dfs.append(df_feats)

        # Resample and stream time series
        ts_dict = cell_data["time_series"]
        for _, row in df_valid.iterrows():
            cyc = int(row["cycle"])
            cap = row["capacity"]
            soh = row["SOH"]
            cap_str = f"{cap:.4f}" if pd.notna(cap) else ""
            soh_str = f"{soh:.2f}" if pd.notna(soh) else ""
            rec = ts_dict.get(cyc, {})

            # Charge
            chg = rec.get("charge", {})
            t_c, v_c, i_c, tmp_c = resample_step(chg.get("time"), chg.get("voltage"), chg.get("current"), chg.get("temperature"), N_PTS_STEP)
            for p in range(N_PTS_STEP):
                tmp_str = f"{tmp_c[p]:.3f}" if not np.isnan(tmp_c[p]) else ""
                ts_f.write(f"{cell},NASA,{cyc},charge,{t_c[p]:.2f},{v_c[p]:.4f},{i_c[p]:.4f},{tmp_str},{cap_str},{soh_str}\n")

            # Discharge
            dis = rec.get("discharge", {})
            t_d, v_d, i_d, tmp_d = resample_step(dis.get("time"), dis.get("voltage"), dis.get("current"), dis.get("temperature"), N_PTS_STEP)
            for p in range(N_PTS_STEP):
                tmp_str = f"{tmp_d[p]:.3f}" if not np.isnan(tmp_d[p]) else ""
                ts_f.write(f"{cell},NASA,{cyc},discharge,{t_d[p]:.2f},{v_d[p]:.4f},{i_d[p]:.4f},{tmp_str},{cap_str},{soh_str}\n")

        print(f"[{idx+1:02d}/{len(nasa_cells)}] NASA {cell:6s}: {len(df_valid):4d} cycles in {time.time()-t0:.2f}s")

    # -------------------------------------------------------------
    # Process CALCE cells
    # -------------------------------------------------------------
    print("\n--- Processing 6 CALCE Cells ---")
    calce_defaults = DATASET_CONFIGS["CALCE"]

    for idx, cell in enumerate(calce_cells):
        t0 = time.time()
        cache_file = os.path.join(cache_dir, f"calce_{cell}.pkl")

        if os.path.exists(cache_file):
            with open(cache_file, "rb") as f:
                cell_data = pickle.load(f)
        else:
            folder = f"data/CALCE/{cell}/{cell}"
            cell_data = load_calce(folder, cache_dir=cache_dir)

        # Validate
        df_valid, stats = validate_cell_data(
            cell_data["summary_df"],
            cell_data["time_series"],
            nominal_capacity=cell_data["nominal_capacity"],
            dataset="CALCE",
        )
        cell_data["summary_df"] = df_valid
        all_dataset_summaries.append(stats)

        # Save individual cell CSV
        cell_csv_path = os.path.join(out_dir, f"{cell}.csv")
        df_valid.to_csv(cell_csv_path, index=False)

        # Feature extraction
        cfg = DATASET_CONFIGS.get(cell, calce_defaults)
        df_feats = extract_all_features_for_cell(cell, cell_data, cfg)
        calce_feature_dfs.append(df_feats)

        # Resample and stream time series
        ts_dict = cell_data["time_series"]
        for _, row in df_valid.iterrows():
            cyc = int(row["cycle"])
            cap = row["capacity"]
            soh = row["SOH"]
            cap_str = f"{cap:.4f}" if pd.notna(cap) else ""
            soh_str = f"{soh:.2f}" if pd.notna(soh) else ""
            rec = ts_dict.get(cyc, {})

            # Charge
            chg = rec.get("charge", {})
            t_c, v_c, i_c, tmp_c = resample_step(chg.get("time"), chg.get("voltage"), chg.get("current"), chg.get("temperature"), N_PTS_STEP)
            for p in range(N_PTS_STEP):
                tmp_str = f"{tmp_c[p]:.3f}" if not np.isnan(tmp_c[p]) else ""
                ts_f.write(f"{cell},CALCE,{cyc},charge,{t_c[p]:.2f},{v_c[p]:.4f},{i_c[p]:.4f},{tmp_str},{cap_str},{soh_str}\n")

            # Discharge
            dis = rec.get("discharge", {})
            t_d, v_d, i_d, tmp_d = resample_step(dis.get("time"), dis.get("voltage"), dis.get("current"), dis.get("temperature"), N_PTS_STEP)
            for p in range(N_PTS_STEP):
                tmp_str = f"{tmp_d[p]:.3f}" if not np.isnan(tmp_d[p]) else ""
                ts_f.write(f"{cell},CALCE,{cyc},discharge,{t_d[p]:.2f},{v_d[p]:.4f},{i_d[p]:.4f},{tmp_str},{cap_str},{soh_str}\n")

        print(f"[{idx+1:02d}/{len(calce_cells)}] CALCE {cell:6s}: {len(df_valid):4d} cycles in {time.time()-t0:.2f}s")

    # Close time series file
    ts_f.close()
    print(f"\n-> Finished cycle_timeseries.csv")

    # -------------------------------------------------------------
    # 3. Save Unified and Feature Tables
    # -------------------------------------------------------------
    print("\n--- Saving Master Datasets ---")

    # Summary table
    summary_rows = []
    for s in all_dataset_summaries:
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
    summary_csv_path = os.path.join(out_dir, "dataset_summary.csv")
    summary_df.to_csv(summary_csv_path, index=False)
    print(f"-> Saved {summary_csv_path} ({len(summary_df)} cells)")

    # features_nasa.csv
    df_nasa_all = pd.concat(nasa_feature_dfs, ignore_index=True)
    df_nasa_all.to_csv(os.path.join(out_dir, "features_nasa.csv"), index=False)
    print(f"-> Saved features_nasa.csv: shape={df_nasa_all.shape}")

    # features_calce.csv
    df_calce_all = pd.concat(calce_feature_dfs, ignore_index=True)
    df_calce_all.to_csv(os.path.join(out_dir, "features_calce.csv"), index=False)
    print(f"-> Saved features_calce.csv: shape={df_calce_all.shape}")

    # features_all.csv
    df_all = pd.concat([df_nasa_all, df_calce_all], ignore_index=True)
    df_all.to_csv(os.path.join(out_dir, "features_all.csv"), index=False)
    print(f"-> Saved features_all.csv: shape={df_all.shape}")

    # -------------------------------------------------------------
    # Check Disk Sizes
    # -------------------------------------------------------------
    print("\n--- Final Size Verification ---")
    total_bytes = 0
    for root, _, files in os.walk(out_dir):
        for f in files:
            fp = os.path.join(root, f)
            sz = os.path.getsize(fp)
            total_bytes += sz

    mb = total_bytes / (1024 * 1024)
    print(f"Total size of {out_dir}: {mb:.2f} MB")
    print("=" * 60)


if __name__ == "__main__":
    main()
