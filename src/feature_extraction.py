"""
src/feature_extraction.py
=========================
Feature extraction module for lithium-ion battery cycling data (NASA & CALCE).

Implements:
1. robust_savgol_smooth: Odd window, capped at signal length, fallback for short
   signals, constant signal safe.
2. compute_ic_curve: Incremental Capacity (dQ/dV) peak height and voltage during charge,
   guarded against dV ≈ 0.
3. extract_cycle_features: Computes all required cycle-level health indicators using
   only physically measured signals, derived operational thresholds, and strict NaN
   recording for absent signals (never fabricated).
"""

import numpy as np
import pandas as pd
from scipy.signal import savgol_filter
from scipy.stats import skew


def robust_savgol_smooth(
    y: np.ndarray,
    window_length: int = 15,
    polyorder: int = 2,
    fallback_ma_window: int = 3,
) -> np.ndarray:
    """
    Applies Savitzky-Golay filter with robust guarantees:
    - Enforces window_length is odd.
    - Caps window_length to <= len(y).
    - Falls back to moving average or raw copy if signal is too short or constant.

    Parameters
    ----------
    y : np.ndarray
        1D input signal.
    window_length : int
        Desired smoothing window length (odd integer).
    polyorder : int
        Polynomial order for Savitzky-Golay.
    fallback_ma_window : int
        Window for moving average fallback if window_length <= polyorder.

    Returns
    -------
    np.ndarray
        Smoothed 1D signal with identical shape and length.
    """
    y = np.asarray(y, dtype=float)
    n = len(y)

    # Edge case: Empty or scalar/single-point signal
    if n <= 1:
        return y.copy()

    # Edge case: Constant signal (range ≈ 0)
    if np.allclose(y, y[0], atol=1e-9):
        return y.copy()

    # Force window to be odd
    win = window_length if window_length % 2 == 1 else window_length - 1

    # Cap window to signal length
    if win > n:
        win = n if n % 2 == 1 else n - 1

    # If win is still greater than polyorder, use Savitzky-Golay
    if win > polyorder and win >= 3:
        try:
            return savgol_filter(y, window_length=win, polyorder=polyorder)
        except Exception:
            pass  # Fall through to moving average

    # Fallback: Simple moving average
    ma_win = min(fallback_ma_window, n)
    if ma_win >= 2:
        kernel = np.ones(ma_win) / ma_win
        # mode='same' with edge padding to preserve length
        pad_size = ma_win // 2
        padded = np.pad(y, pad_size, mode="edge")
        smoothed = np.convolve(padded, kernel, mode="valid")
        return smoothed[:n]

    return y.copy()


def compute_ic_curve(
    time: np.ndarray,
    voltage: np.ndarray,
    current: np.ndarray,
    v_min: float = 3.60,
    v_max: float = 4.18,
    savgol_window: int = 21,
    min_dv: float = 1e-4,
) -> tuple[float, float]:
    """
    Computes Incremental Capacity (dQ/dV) peak height and voltage during charge.

    Parameters
    ----------
    time : np.ndarray
        Time in seconds.
    voltage : np.ndarray
        Charge voltage in Volts.
    current : np.ndarray
        Charge current in Amperes.
    v_min : float
        Lower voltage bound for searching IC peak.
    v_max : float
        Upper voltage bound for searching IC peak.
    savgol_window : int
        Smoothing window length.
    min_dv : float
        Minimum voltage difference threshold to guard against division by zero (dV ≈ 0).

    Returns
    -------
    tuple (ic_peak_height, ic_peak_voltage)
        Peak dQ/dV in Ah/V and the corresponding voltage in V.
        Returns (np.nan, np.nan) if signal is insufficient or invalid.
    """
    if len(voltage) < 5 or len(current) < 5 or len(time) < 5:
        return np.nan, np.nan

    # Filter to valid charge points within target voltage window
    # In charging, current must be positive (> 0.05 A)
    mask = (voltage >= v_min) & (voltage <= v_max) & (current > 0.05)
    if mask.sum() < 5:
        # Relax lower voltage bound if initial voltage was slightly above v_min
        mask = (voltage >= voltage.min()) & (voltage <= v_max) & (current > 0.05)
        if mask.sum() < 5:
            return np.nan, np.nan

    t_sub = time[mask]
    v_sub = voltage[mask]
    i_sub = current[mask]

    # Compute cumulative charge Q in Ah via trapezoidal rule
    dt = np.diff(t_sub)
    if len(dt) == 0 or np.all(dt <= 0):
        return np.nan, np.nan

    dq_elements = 0.5 * (i_sub[:-1] + i_sub[1:]) * dt / 3600.0
    q_sub = np.cumsum(np.insert(dq_elements, 0, 0.0))

    # Apply robust smoothing to voltage and charge
    v_smooth = robust_savgol_smooth(v_sub, window_length=savgol_window, polyorder=2)
    q_smooth = robust_savgol_smooth(q_sub, window_length=savgol_window, polyorder=2)

    # Compute numerical gradients
    dv = np.gradient(v_smooth)
    dq = np.gradient(q_smooth)

    # Guard strictly against dV ≈ 0
    valid_dv = dv >= min_dv
    if not np.any(valid_dv):
        return np.nan, np.nan

    # Calculate dQ/dV only on valid positive dV steps
    dq_dv = np.zeros_like(dv)
    dq_dv[valid_dv] = dq[valid_dv] / dv[valid_dv]

    # Mask to positive dQ/dV and non-extreme values (< 50 Ah/V)
    valid_ic = valid_dv & (dq_dv > 0) & (dq_dv < 50.0)
    if not np.any(valid_ic):
        return np.nan, np.nan

    valid_indices = np.where(valid_ic)[0]
    peak_sub_idx = np.argmax(dq_dv[valid_indices])
    peak_idx = valid_indices[peak_sub_idx]

    peak_height = float(dq_dv[peak_idx])
    peak_voltage = float(v_smooth[peak_idx])

    return peak_height, peak_voltage


def extract_cycle_features(cycle_record: dict, dataset_config: dict) -> dict:
    """
    Extracts all cycle-level features for a single battery cycle.

    Parameters
    ----------
    cycle_record : dict
        {
            'cycle': int,
            'charge': dict {'time', 'voltage', 'current', 'temperature', 'duration_s'},
            'discharge': dict {'time', 'voltage', 'current', 'temperature', 'duration_s'},
            'capacity': float,
            'SOH': float,
            'C_ref': float,
            'cell_id': str,
            'dataset': str,
        }
    dataset_config : dict
        Derived parameters (cc_current_nominal, cv_voltage_threshold,
        plateau_v_low, plateau_v_high, ic_v_min, ic_v_max, has_temperature).

    Returns
    -------
    dict
        Feature dictionary for the cycle.
    """
    ch_data = cycle_record.get("charge") or {}
    dis_data = cycle_record.get("discharge") or {}

    cycle_idx = cycle_record.get("cycle", 1)
    dataset = cycle_record.get("dataset", "NASA")

    # Configuration thresholds
    v_cv_thresh = dataset_config.get("cv_voltage_threshold", 4.18)
    v_plat_low = dataset_config.get("plateau_v_low", 3.46)
    v_plat_high = dataset_config.get("plateau_v_high", 3.70)
    ic_v_min = dataset_config.get("ic_v_min", 3.60)
    ic_v_max = dataset_config.get("ic_v_max", 4.18)
    has_temp = dataset_config.get("has_temperature", False)

    # -------------------------------------------------------------
    # 1. Charge Phase Features
    # -------------------------------------------------------------
    t_ch = ch_data.get("time", np.array([]))
    v_ch = ch_data.get("voltage", np.array([]))
    i_ch = ch_data.get("current", np.array([]))
    temp_ch = ch_data.get("temperature")

    if len(t_ch) > 1 and len(v_ch) > 1:
        # Determine transition from CC to CV:
        # CC phase: voltage is rising toward v_cv_thresh
        # CV phase: voltage reaches ~v_cv_thresh and current decays
        idx_cv = np.where(v_ch >= v_cv_thresh)[0]
        if len(idx_cv) > 0:
            t_trans = t_ch[idx_cv[0]]
            cc_charge_time = float(max(0.0, t_trans - t_ch[0]))
            cv_charge_time = float(max(0.0, t_ch[-1] - t_trans))
        else:
            # Did not reach 4.18V: entire phase treated as CC
            cc_charge_time = float(max(0.0, t_ch[-1] - t_ch[0]))
            cv_charge_time = 0.0

        # Incremental Capacity (IC) curve features
        ic_peak_height, ic_peak_voltage = compute_ic_curve(
            time=t_ch,
            voltage=v_ch,
            current=i_ch,
            v_min=ic_v_min,
            v_max=ic_v_max,
        )
    else:
        cc_charge_time = np.nan
        cv_charge_time = np.nan
        ic_peak_height = np.nan
        ic_peak_voltage = np.nan

    # Charge Temperature Features (NASA only, NaN for CALCE)
    if has_temp and temp_ch is not None and len(temp_ch) > 0:
        temp_charge_mean = float(np.mean(temp_ch))
        temp_charge_max = float(np.max(temp_ch))
        temp_charge_rise = float(np.max(temp_ch) - temp_ch[0])
    else:
        temp_charge_mean = np.nan
        temp_charge_max = np.nan
        temp_charge_rise = np.nan

    # -------------------------------------------------------------
    # 2. Discharge Phase Features
    # -------------------------------------------------------------
    t_dis = dis_data.get("time", np.array([]))
    v_dis = dis_data.get("voltage", np.array([]))
    i_dis = dis_data.get("current", np.array([]))
    temp_dis = dis_data.get("temperature")

    # Isolate active discharge under load (current < -0.1 A)
    if len(t_dis) > 1 and len(v_dis) > 1 and len(i_dis) > 1:
        active_mask = i_dis < -0.1
        if active_mask.sum() > 2:
            t_act = t_dis[active_mask]
            v_act = v_dis[active_mask]
            i_act = i_dis[active_mask]
        else:
            t_act, v_act, i_act = t_dis, v_dis, i_dis

        # Discharge time
        discharge_time = float(t_act[-1] - t_act[0]) if len(t_act) > 1 else 0.0

        # Voltage plateau duration between v_plat_low and v_plat_high
        plat_mask = (v_act >= v_plat_low) & (v_act <= v_plat_high)
        if plat_mask.sum() > 1:
            voltage_plateau_duration = float(t_act[plat_mask][-1] - t_act[plat_mask][0])
        else:
            voltage_plateau_duration = 0.0

        # Current statistics (magnitudes)
        i_mags = np.abs(i_act)
        discharge_current_mean = float(np.mean(i_mags))
        discharge_current_max = float(np.max(i_mags))

        # Voltage statistics
        voltage_mean = float(np.mean(v_act))
        voltage_std = float(np.std(v_act))
        voltage_skew = float(skew(v_act)) if len(v_act) > 2 else 0.0
        voltage_min = float(np.min(v_act))

        # Voltage slope dV/dt (linear regression fit)
        if len(t_act) > 2 and (t_act[-1] - t_act[0]) > 0:
            t_norm = t_act - t_act[0]
            slope, _ = np.polyfit(t_norm, v_act, deg=1)
            voltage_slope = float(slope)
        else:
            voltage_slope = 0.0
    else:
        discharge_time = np.nan
        voltage_plateau_duration = np.nan
        discharge_current_mean = np.nan
        discharge_current_max = np.nan
        voltage_mean = np.nan
        voltage_std = np.nan
        voltage_skew = np.nan
        voltage_min = np.nan
        voltage_slope = np.nan

    # Discharge Temperature Rise: STRICT REQUIREMENT:
    # "use DISCHARGE temperature, not charge"
    if has_temp and temp_dis is not None and len(temp_dis) > 0:
        temp_discharge_rise = float(np.max(temp_dis) - temp_dis[0])
    else:
        temp_discharge_rise = np.nan

    # -------------------------------------------------------------
    # 3. Assemble Feature Record
    # -------------------------------------------------------------
    features = {
        "cycle": int(cycle_idx),
        "cc_charge_time": cc_charge_time,
        "cv_charge_time": cv_charge_time,
        "discharge_time": discharge_time,
        "voltage_plateau_duration": voltage_plateau_duration,
        "temp_charge_mean": temp_charge_mean,
        "temp_charge_max": temp_charge_max,
        "temp_charge_rise": temp_charge_rise,
        "temp_discharge_rise": temp_discharge_rise,
        "discharge_current_mean": discharge_current_mean,
        "discharge_current_max": discharge_current_max,
        "voltage_mean": voltage_mean,
        "voltage_std": voltage_std,
        "voltage_skew": voltage_skew,
        "voltage_min": voltage_min,
        "voltage_slope": voltage_slope,
        "ic_peak_height": ic_peak_height,
        "ic_peak_voltage": ic_peak_voltage,
        # Targets and identifiers
        "capacity": float(cycle_record.get("capacity", np.nan)),
        "SOH": float(cycle_record.get("SOH", np.nan)),
        "C_ref": float(cycle_record.get("C_ref", np.nan)),
        "cell_id": str(cycle_record.get("cell_id", "")),
        "dataset": str(dataset),
        "is_outlier": bool(cycle_record.get("is_outlier", False)),
        "validation_flags": str(cycle_record.get("validation_flags", "VALID")),
    }

    return features


def extract_all_features_for_cell(
    cell_id: str,
    cell_data: dict,
    dataset_config: dict
) -> pd.DataFrame:
    """
    Extracts features across all cycles for a single battery cell.

    Parameters
    ----------
    cell_id : str
        Cell identifier (e.g., 'B0005', 'CS2_35').
    cell_data : dict
        Output from load_nasa or load_calce containing 'summary_df' and 'time_series'.
    dataset_config : dict
        Derived parameters for this cell/dataset.

    Returns
    -------
    pd.DataFrame
        Complete tabular DataFrame of extracted features.
    """
    summary_df = cell_data["summary_df"]
    time_series = cell_data["time_series"]

    feature_rows = []
    for _, row in summary_df.iterrows():
        cyc = int(row["cycle"])
        ts_record = time_series.get(cyc, {})

        cycle_record = {
            "cycle": cyc,
            "charge": ts_record.get("charge"),
            "discharge": ts_record.get("discharge"),
            "capacity": row["capacity"],
            "SOH": row["SOH"],
            "C_ref": row["C_ref"],
            "cell_id": cell_id,
            "dataset": row["dataset"],
            "is_outlier": row.get("is_outlier", False),
            "validation_flags": row.get("validation_flags", "VALID"),
        }

        feat = extract_cycle_features(cycle_record, dataset_config)
        feature_rows.append(feat)

    df_feats = pd.DataFrame(feature_rows)
    return df_feats

