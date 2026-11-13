"""
src/data_loading.py
===================
Robust data loading utilities for NASA Li-ion battery aging dataset (.mat)
and CALCE battery dataset (.xlsx).

Adheres strictly to PBL Project Guidelines:
- Relative paths only.
- Raw original files are NEVER modified.
- Separate, modular parsers for NASA and CALCE.
- Extracts per-cell cycle-level records: raw charge and discharge time series
  (time, voltage, current, temperature where available) plus discharge capacity.
- Fast caching to data/cache/ to accelerate downstream pipelines.
"""

import os
import glob
import pickle
import datetime
import numpy as np
import pandas as pd
import scipy.io as sio


def compute_reference_capacity(capacities: np.ndarray, n_first: int = 3) -> float:
    """
    Computes reference capacity C_ref as the median measured discharge capacity
    of the first n_first valid cycles (default=3).
    """
    valid_caps = [c for c in capacities[:n_first] if c > 0 and not np.isnan(c)]
    if len(valid_caps) == 0:
        raise ValueError("No valid initial cycles found to compute C_ref.")
    return float(np.median(valid_caps))


def load_nasa(path: str, cache_dir: str = "data/cache", force_reload: bool = False):
    """
    Loads and parses a NASA Ames battery aging MATLAB (.mat) file.

    Parameters
    ----------
    path : str
        Relative path to .mat file (e.g., 'B0005.mat').
    cache_dir : str
        Directory to store and load parsed pickle cache.
    force_reload : bool
        If True, ignores existing cache and re-parses from raw file.

    Returns
    -------
    dict
        {
            'cell_id': str,
            'dataset': 'NASA',
            'nominal_capacity': 2.0,
            'c_ref': float,
            'summary_df': pd.DataFrame,
            'time_series': dict {cycle_num: {'charge': dict, 'discharge': dict}}
        }
    """
    os.makedirs(cache_dir, exist_ok=True)
    cell_id = os.path.splitext(os.path.basename(path))[0]
    cache_path = os.path.join(cache_dir, f"nasa_{cell_id}.pkl")

    if not force_reload and os.path.exists(cache_path):
        with open(cache_path, "rb") as f:
            return pickle.load(f)

    if not os.path.exists(path):
        raise FileNotFoundError(f"NASA file not found: {path}")

    mat = sio.loadmat(path)
    # The struct key is the cell name (e.g., B0005)
    keys = [k for k in mat.keys() if not k.startswith("__")]
    if not keys:
        raise ValueError(f"No valid data struct in {path}")
    struct_key = keys[0]
    cycles_raw = mat[struct_key][0, 0]["cycle"][0]

    cycle_records = []
    time_series = {}
    last_charge_data = None
    last_charge_time_s = 0.0

    discharge_cycle_idx = 0

    for i, c in enumerate(cycles_raw):
        ctype = str(c["type"][0]).strip().lower()
        cdata = c["data"]

        # Parse timestamp from cycle['time']
        date_vec = c["time"][0]
        try:
            year, month, day = int(date_vec[0]), int(date_vec[1]), int(date_vec[2])
            hour, minute = int(date_vec[3]), int(date_vec[4])
            second = int(date_vec[5])
            dt = datetime.datetime(year, month, day, hour, minute, second)
        except Exception:
            dt = None

        amb_temp = float(c["ambient_temperature"][0, 0]) if "ambient_temperature" in c.dtype.names else 24.0

        if ctype == "charge":
            ch_time = cdata["Time"][0, 0].flatten().astype(float)
            ch_v = cdata["Voltage_measured"][0, 0].flatten().astype(float)
            ch_i = cdata["Current_measured"][0, 0].flatten().astype(float)
            ch_temp = cdata["Temperature_measured"][0, 0].flatten().astype(float)
            last_charge_data = {
                "time": ch_time,
                "voltage": ch_v,
                "current": ch_i,
                "temperature": ch_temp,
                "duration_s": float(ch_time[-1] - ch_time[0]) if len(ch_time) > 1 else 0.0,
            }
            last_charge_time_s = last_charge_data["duration_s"]

        elif ctype == "discharge":
            discharge_cycle_idx += 1
            dis_time = cdata["Time"][0, 0].flatten().astype(float)
            dis_v = cdata["Voltage_measured"][0, 0].flatten().astype(float)
            dis_i = cdata["Current_measured"][0, 0].flatten().astype(float)
            dis_temp = cdata["Temperature_measured"][0, 0].flatten().astype(float)
            capacity = float(cdata["Capacity"][0, 0][0, 0])

            dis_duration = float(dis_time[-1] - dis_time[0]) if len(dis_time) > 1 else 0.0

            record = {
                "cell_id": cell_id,
                "dataset": "NASA",
                "cycle": discharge_cycle_idx,
                "date_time": dt,
                "capacity": capacity,
                "ambient_temp": amb_temp,
                "charge_duration_s": last_charge_time_s,
                "discharge_duration_s": dis_duration,
                "v_min": float(dis_v.min()) if len(dis_v) > 0 else np.nan,
                "v_max": float(dis_v.max()) if len(dis_v) > 0 else np.nan,
                "temp_mean": float(dis_temp.mean()) if len(dis_temp) > 0 else np.nan,
                "temp_max": float(dis_temp.max()) if len(dis_temp) > 0 else np.nan,
            }
            cycle_records.append(record)

            time_series[discharge_cycle_idx] = {
                "charge": last_charge_data,
                "discharge": {
                    "time": dis_time,
                    "voltage": dis_v,
                    "current": dis_i,
                    "temperature": dis_temp,
                    "duration_s": dis_duration,
                },
            }

    df = pd.DataFrame(cycle_records)
    c_ref = compute_reference_capacity(df["capacity"].values, n_first=3)
    df["C_ref"] = c_ref
    df["SOH"] = (df["capacity"] / c_ref) * 100.0

    result = {
        "cell_id": cell_id,
        "dataset": "NASA",
        "nominal_capacity": 2.0,
        "c_ref": c_ref,
        "summary_df": df,
        "time_series": time_series,
    }

    with open(cache_path, "wb") as f:
        pickle.dump(result, f)

    return result


def load_calce(folder: str, cache_dir: str = "data/cache", force_reload: bool = False):
    """
    Loads and parses CALCE battery aging Excel files (.xlsx).

    Parameters
    ----------
    folder : str
        Relative path to folder containing CALCE .xlsx files (e.g. 'CALCE-main/dataset/CS2_35').
    cache_dir : str
        Directory to store and load parsed pickle cache.
    force_reload : bool
        If True, ignores existing cache and re-parses from raw files.

    Returns
    -------
    dict
        {
            'cell_id': str,
            'dataset': 'CALCE',
            'nominal_capacity': 1.1,
            'c_ref': float,
            'summary_df': pd.DataFrame,
            'time_series': dict {cycle_num: {'charge': dict, 'discharge': dict}}
        }
    """
    os.makedirs(cache_dir, exist_ok=True)
    cell_id = os.path.basename(os.path.normpath(folder))
    cache_path = os.path.join(cache_dir, f"calce_{cell_id}.pkl")

    if not force_reload and os.path.exists(cache_path):
        with open(cache_path, "rb") as f:
            return pickle.load(f)

    if not os.path.exists(folder):
        raise FileNotFoundError(f"CALCE folder not found: {folder}")

    files = sorted(glob.glob(os.path.join(folder, "*.xlsx")))
    if not files:
        raise ValueError(f"No .xlsx files found in {folder}")

    # Step 1: Sort files chronologically using the first Date_Time in sheet 1
    file_chronology = []
    for f in files:
        df_head = pd.read_excel(f, sheet_name=1, nrows=2)
        start_dt = pd.to_datetime(df_head["Date_Time"].iloc[0])
        file_chronology.append((start_dt, f))
    file_chronology.sort(key=lambda x: x[0])

    cycle_records = []
    time_series = {}
    cumulative_cycle = 0

    for file_dt, fpath in file_chronology:
        df = pd.read_excel(fpath, sheet_name=1)
        unique_cycles = sorted(df["Cycle_Index"].unique())

        for c in unique_cycles:
            df_c = df[df["Cycle_Index"] == c]
            df_d = df_c[df_c["Step_Index"] == 7]  # CC Discharge step

            if len(df_d) == 0:
                continue

            # CC and CV charge steps
            df_cc = df_c[df_c["Step_Index"] == 2]
            df_cv = df_c[df_c["Step_Index"] == 4]

            cumulative_cycle += 1

            # Compute discharge capacity
            # Using maximum discharged Ah recorded directly by the Arbin cycler
            # Note: sub-cycle relative Ah = max - min
            d_cap_raw = float(df_d["Discharge_Capacity(Ah)"].max() - df_d["Discharge_Capacity(Ah)"].min())

            # Durations
            ccct = float(df_cc["Test_Time(s)"].max() - df_cc["Test_Time(s)"].min()) if len(df_cc) > 0 else 0.0
            cvct = float(df_cv["Test_Time(s)"].max() - df_cv["Test_Time(s)"].min()) if len(df_cv) > 0 else 0.0
            dis_dur = float(df_d["Test_Time(s)"].max() - df_d["Test_Time(s)"].min()) if len(df_d) > 0 else 0.0
            ch_dur = ccct + cvct

            # Timestamp
            dt_start = pd.to_datetime(df_d["Date_Time"].iloc[0])

            # Internal resistance if available
            ir = float(df_d["Internal_Resistance(Ohm)"].mean()) if "Internal_Resistance(Ohm)" in df_d.columns else np.nan

            dis_time = df_d["Test_Time(s)"].values.astype(float)
            dis_v = df_d["Voltage(V)"].values.astype(float)
            dis_i = df_d["Current(A)"].values.astype(float)

            df_ch = pd.concat([df_cc, df_cv]).sort_values("Test_Time(s)")
            ch_time = df_ch["Test_Time(s)"].values.astype(float) if len(df_ch) > 0 else np.array([])
            ch_v = df_ch["Voltage(V)"].values.astype(float) if len(df_ch) > 0 else np.array([])
            ch_i = df_ch["Current(A)"].values.astype(float) if len(df_ch) > 0 else np.array([])

            record = {
                "cell_id": cell_id,
                "dataset": "CALCE",
                "cycle": cumulative_cycle,
                "date_time": dt_start,
                "capacity": d_cap_raw,
                "ambient_temp": np.nan,  # Not measured in CALCE CS2
                "charge_duration_s": ch_dur,
                "discharge_duration_s": dis_dur,
                "ccct_s": ccct,
                "cvct_s": cvct,
                "internal_resistance_ohm": ir,
                "v_min": float(dis_v.min()) if len(dis_v) > 0 else np.nan,
                "v_max": float(dis_v.max()) if len(dis_v) > 0 else np.nan,
                "temp_mean": np.nan,
                "temp_max": np.nan,
            }
            cycle_records.append(record)

            time_series[cumulative_cycle] = {
                "charge": {
                    "time": ch_time,
                    "voltage": ch_v,
                    "current": ch_i,
                    "temperature": None,
                    "duration_s": ch_dur,
                },
                "discharge": {
                    "time": dis_time,
                    "voltage": dis_v,
                    "current": dis_i,
                    "temperature": None,
                    "duration_s": dis_dur,
                },
            }

    df = pd.DataFrame(cycle_records)
    c_ref = compute_reference_capacity(df["capacity"].values, n_first=3)
    df["C_ref"] = c_ref
    df["SOH"] = (df["capacity"] / c_ref) * 100.0

    result = {
        "cell_id": cell_id,
        "dataset": "CALCE",
        "nominal_capacity": 1.1,
        "c_ref": c_ref,
        "summary_df": df,
        "time_series": time_series,
    }

    with open(cache_path, "wb") as f:
        pickle.dump(result, f)

    return result

