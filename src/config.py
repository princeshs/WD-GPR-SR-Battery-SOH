"""
src/config.py
=============
Central configuration module for WD-GPR-SR battery health prediction.
Defines:
- Derived dataset thresholds per cell/dataset (CC current, plateau bounds, CV cutoff).
- Feature categories: Inference-valid vs. Post-discharge vs. Disallowed leakage.
- FEATURES_MAIN: Strictly inference-valid features for downstream models.
"""

# Derived per-cell and per-dataset operational parameters
DATASET_CONFIGS = {
    "NASA": {
        "nominal_capacity": 2.0,  # Ah (18650 cylindrical)
        "cc_current_nominal": 1.51,  # A (median early charge current)
        "cc_current_tolerance": 0.15,  # A window around nominal CC
        "cv_voltage_threshold": 4.18,  # V (CC-CV transition threshold)
        "plateau_v_low": 3.46,  # V (25th percentile fresh discharge under load)
        "plateau_v_high": 3.70,  # V (75th percentile fresh discharge under load)
        "ic_v_min": 3.60,  # V for IC curve peak search
        "ic_v_max": 4.18,  # V for IC curve peak search
        "has_temperature": True,
    },
    "CALCE": {
        "nominal_capacity": 1.1,  # Ah (prismatic LiCoO2)
        "cc_current_nominal": 0.55,  # A (0.5 C CC charge current)
        "cc_current_tolerance": 0.08,  # A window around nominal CC
        "cv_voltage_threshold": 4.18,  # V (CC-CV transition threshold)
        "plateau_v_low": 3.55,  # V (25th percentile fresh discharge under load)
        "plateau_v_high": 3.77,  # V (75th percentile fresh discharge under load)
        "ic_v_min": 3.60,  # V for IC curve peak search
        "ic_v_max": 4.18,  # V for IC curve peak search
        "has_temperature": False,  # CALCE CS2 does not log cell temperature
    },
    # Per-cell specific derived configurations
    "B0005": {
        "dataset": "NASA",
        "nominal_capacity": 2.0,
        "cc_current_nominal": 1.511,
        "cv_voltage_threshold": 4.18,
        "plateau_v_low": 3.460,
        "plateau_v_high": 3.701,
        "ic_v_min": 3.60,
        "ic_v_max": 4.18,
        "has_temperature": True,
    },
    "B0006": {
        "dataset": "NASA",
        "nominal_capacity": 2.0,
        "cc_current_nominal": 1.513,
        "cv_voltage_threshold": 4.18,
        "plateau_v_low": 3.459,
        "plateau_v_high": 3.699,
        "ic_v_min": 3.60,
        "ic_v_max": 4.18,
        "has_temperature": True,
    },
    "B0007": {
        "dataset": "NASA",
        "nominal_capacity": 2.0,
        "cc_current_nominal": 1.491,
        "cv_voltage_threshold": 4.18,
        "plateau_v_low": 3.459,
        "plateau_v_high": 3.708,
        "ic_v_min": 3.60,
        "ic_v_max": 4.18,
        "has_temperature": True,
    },
    "B0018": {
        "dataset": "NASA",
        "nominal_capacity": 2.0,
        "cc_current_nominal": 1.515,
        "cv_voltage_threshold": 4.18,
        "plateau_v_low": 3.443,
        "plateau_v_high": 3.689,
        "ic_v_min": 3.60,
        "ic_v_max": 4.18,
        "has_temperature": True,
    },
    "CS2_35": {
        "dataset": "CALCE",
        "nominal_capacity": 1.1,
        "cc_current_nominal": 0.550,
        "cv_voltage_threshold": 4.18,
        "plateau_v_low": 3.556,
        "plateau_v_high": 3.774,
        "ic_v_min": 3.60,
        "ic_v_max": 4.18,
        "has_temperature": False,
    },
    "CS2_36": {
        "dataset": "CALCE",
        "nominal_capacity": 1.1,
        "cc_current_nominal": 0.550,
        "cv_voltage_threshold": 4.18,
        "plateau_v_low": 3.557,
        "plateau_v_high": 3.775,
        "ic_v_min": 3.60,
        "ic_v_max": 4.18,
        "has_temperature": False,
    },
    "CS2_37": {
        "dataset": "CALCE",
        "nominal_capacity": 1.1,
        "cc_current_nominal": 0.550,
        "cv_voltage_threshold": 4.18,
        "plateau_v_low": 3.554,
        "plateau_v_high": 3.773,
        "ic_v_min": 3.60,
        "ic_v_max": 4.18,
        "has_temperature": False,
    },
    "CS2_38": {
        "dataset": "CALCE",
        "nominal_capacity": 1.1,
        "cc_current_nominal": 0.550,
        "cv_voltage_threshold": 4.18,
        "plateau_v_low": 3.540,
        "plateau_v_high": 3.760,
        "ic_v_min": 3.60,
        "ic_v_max": 4.18,
        "has_temperature": False,
    },
}

# Features that are strictly valid at inference time (available before discharge):
# Cycle count, charging characteristics (CCCT, CVCT), and charging IC peak metrics.
FEATURES_MAIN = [
    "cycle",
    "cc_charge_time",
    "cv_charge_time",
    "ic_peak_height",
    "ic_peak_voltage",
]

# Temperature features available only when thermal sensors are present (NASA Ames)
FEATURES_NASA_TEMP = [
    "temp_charge_mean",
    "temp_charge_max",
    "temp_charge_rise",
]

# Features derived from discharge (use with caution: diagnostic/post-cycle observation only)
FEATURES_DISCHARGE_DIAGNOSTIC = [
    "discharge_time",
    "voltage_plateau_duration",
    "discharge_current_mean",
    "discharge_current_max",
    "voltage_mean",
    "voltage_std",
    "voltage_skew",
    "voltage_min",
    "voltage_slope",
    "temp_discharge_rise",
]

# Full list of extracted features
ALL_EXTRACTED_FEATURES = (
    FEATURES_MAIN + FEATURES_NASA_TEMP + FEATURES_DISCHARGE_DIAGNOSTIC
)

# Target variable definition
TARGET_COLUMN = "SOH"
TARGET_RAW_CAPACITY = "capacity"

