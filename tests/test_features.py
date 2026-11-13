"""
tests/test_features.py
======================
Unit tests for feature extraction and smoothing functions:
- Savitzky-Golay helper: odd window, capped window, short signal fallback, constant signal.
- Incremental capacity (dQ/dV): guard against dV ≈ 0, short signals, synthetic peak detection.
- Feature extraction: discharge temperature rise using discharge temp, CALCE NaN temperatures.
"""

import numpy as np
import pytest
from src.feature_extraction import (
    robust_savgol_smooth,
    compute_ic_curve,
    extract_cycle_features,
)


def test_robust_savgol_smooth_odd_window():
    # Pass an even window (e.g. 10), verify smoothed output maintains shape
    y = np.linspace(3.0, 4.2, 50) + np.random.normal(0, 0.01, 50)
    smoothed = robust_savgol_smooth(y, window_length=10, polyorder=2)
    assert len(smoothed) == len(y)
    assert not np.isnan(smoothed).any()


def test_robust_savgol_smooth_caps_window():
    # Signal shorter than window (len=12, win=31)
    y = np.linspace(3.5, 4.2, 12)
    smoothed = robust_savgol_smooth(y, window_length=31, polyorder=2)
    assert len(smoothed) == len(y)
    assert not np.isnan(smoothed).any()


def test_robust_savgol_smooth_short_signals():
    # Lengths: 0, 1, 2, 3, 4
    for n in [0, 1, 2, 3, 4]:
        y = np.ones(n) * 3.7
        smoothed = robust_savgol_smooth(y, window_length=15, polyorder=2)
        assert len(smoothed) == n
        assert not np.isnan(smoothed).any()


def test_robust_savgol_smooth_constant_signal():
    # Constant signal (all 3.82 V)
    y = np.full(100, 3.82)
    smoothed = robust_savgol_smooth(y, window_length=15, polyorder=2)
    assert len(smoothed) == 100
    assert np.allclose(smoothed, 3.82, atol=1e-8)


def test_compute_ic_curve_guards_zero_dv():
    # Constant voltage (dV = 0 everywhere)
    time = np.linspace(0, 1000, 50)
    voltage = np.full(50, 3.85)
    current = np.full(50, 1.5)
    pk_h, pk_v = compute_ic_curve(time, voltage, current, min_dv=1e-4)
    # Must guard against division by zero and return np.nan
    assert np.isnan(pk_h)
    assert np.isnan(pk_v)


def test_compute_ic_curve_short_signal():
    # Very few points (< 5)
    time = np.array([0.0, 10.0, 20.0])
    voltage = np.array([3.6, 3.7, 3.8])
    current = np.array([1.5, 1.5, 1.5])
    pk_h, pk_v = compute_ic_curve(time, voltage, current)
    assert np.isnan(pk_h)
    assert np.isnan(pk_v)


def test_compute_ic_curve_synthetic_peak():
    # Synthetic charging signal with a voltage plateau around 3.9V
    # giving rise to an IC peak (dV/dt is smallest at t=1800 where V=3.9V)
    time = np.linspace(0, 3600, 300)
    current = np.full(300, 1.5)
    t_shift = time - 1800
    voltage = 3.9 + 5e-5 * t_shift + 3e-11 * (t_shift**3)
    pk_h, pk_v = compute_ic_curve(time, voltage, current, v_min=3.6, v_max=4.15)
    assert not np.isnan(pk_h)
    assert not np.isnan(pk_v)
    assert 3.85 <= pk_v <= 3.95
    assert pk_h > 0


def test_extract_cycle_features_temp_discharge_rise():
    # Verify that temp_discharge_rise specifically uses discharge temperature
    cycle_record = {
        "cycle": 1,
        "charge": {
            "time": np.linspace(0, 1000, 50),
            "voltage": np.linspace(3.5, 4.2, 50),
            "current": np.full(50, 1.5),
            "temperature": np.array([24.0, 25.0, 26.0]),  # rise = 2.0
        },
        "discharge": {
            "time": np.linspace(0, 1000, 50),
            "voltage": np.linspace(4.2, 2.7, 50),
            "current": np.full(50, -2.0),
            "temperature": np.array([24.0, 28.0, 34.0]),  # rise = 10.0
        },
        "capacity": 1.85,
        "SOH": 100.0,
        "C_ref": 1.85,
        "cell_id": "B0005",
        "dataset": "NASA",
    }
    cfg = {
        "cc_current_nominal": 1.5,
        "cv_voltage_threshold": 4.18,
        "plateau_v_low": 3.46,
        "plateau_v_high": 3.70,
        "has_temperature": True,
    }
    feats = extract_cycle_features(cycle_record, cfg)
    assert feats["temp_charge_rise"] == pytest.approx(2.0, abs=1e-3)
    assert feats["temp_discharge_rise"] == pytest.approx(10.0, abs=1e-3)


def test_extract_cycle_features_calce_nan_temperature():
    # Verify CALCE sets temperature features to NaN (never fabricated)
    cycle_record = {
        "cycle": 1,
        "charge": {
            "time": np.linspace(0, 1000, 50),
            "voltage": np.linspace(3.5, 4.2, 50),
            "current": np.full(50, 0.55),
            "temperature": None,
        },
        "discharge": {
            "time": np.linspace(0, 1000, 50),
            "voltage": np.linspace(4.2, 2.7, 50),
            "current": np.full(50, -1.1),
            "temperature": None,
        },
        "capacity": 1.12,
        "SOH": 100.0,
        "C_ref": 1.12,
        "cell_id": "CS2_35",
        "dataset": "CALCE",
    }
    cfg = {
        "cc_current_nominal": 0.55,
        "cv_voltage_threshold": 4.18,
        "plateau_v_low": 3.55,
        "plateau_v_high": 3.77,
        "has_temperature": False,
    }
    feats = extract_cycle_features(cycle_record, cfg)
    assert np.isnan(feats["temp_charge_mean"])
    assert np.isnan(feats["temp_charge_max"])
    assert np.isnan(feats["temp_charge_rise"])
    assert np.isnan(feats["temp_discharge_rise"])
