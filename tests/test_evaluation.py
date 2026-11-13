"""
tests/test_evaluation.py
========================
Unit tests for evaluation metrics and uncertainty evaluation:
- Deterministic synthetic targets and predictions
- Standard metrics (MAE, MSE, RMSE, R2, MAPE)
- Probabilistic metrics (NLL, 95% PICP, 95% MPIW)
- Variance floor numerical stability
"""

import numpy as np
import pytest
from src.evaluation import evaluate_uncertainty_metrics, PreprocessingPipeline


def test_evaluate_uncertainty_metrics_exact_prediction():
    # Perfect predictions with uniform uncertainty
    y_true = np.array([100.0, 95.0, 90.0, 85.0])
    y_pred = np.array([100.0, 95.0, 90.0, 85.0])
    y_std = np.array([1.0, 1.0, 1.0, 1.0])

    m = evaluate_uncertainty_metrics(y_true, y_pred, y_std)

    assert m["mae"] == pytest.approx(0.0, abs=1e-7)
    assert m["mse"] == pytest.approx(0.0, abs=1e-7)
    assert m["rmse"] == pytest.approx(0.0, abs=1e-7)
    assert m["r2"] == pytest.approx(1.0, abs=1e-7)
    assert m["mape"] == pytest.approx(0.0, abs=1e-7)

    # 100% of points are within [mu - 1.96*sigma, mu + 1.96*sigma]
    assert m["picp"] == pytest.approx(100.0)
    # MPIW = 2 * 1.96 * 1.0 = 3.92
    assert m["mpiw"] == pytest.approx(3.92, abs=1e-4)

    # NLL = 0.5 * ln(2*pi*1^2) + 0.5 * 0 = 0.5 * ln(2*pi) ≈ 0.9189
    expected_nll = 0.5 * np.log(2 * np.pi)
    assert m["nll"] == pytest.approx(expected_nll, abs=1e-4)


def test_evaluate_uncertainty_metrics_coverage():
    # Test PICP with known standard normal errors:
    # 95% of samples should fall within ±1.96 standard deviations
    np.random.seed(42)
    n = 10000
    y_true = np.zeros(n)
    sigma = 2.0
    # True samples with sigma=2.0
    errors = np.random.normal(0, sigma, n)
    y_pred = y_true + errors
    y_std = np.full(n, sigma)

    m = evaluate_uncertainty_metrics(y_true, y_pred, y_std)
    # PICP should be approximately 95% (±1.5%)
    assert 93.5 <= m["picp"] <= 96.5
    # MPIW = 2 * 1.96 * 2.0 = 7.84
    assert m["mpiw"] == pytest.approx(7.84, abs=1e-2)


def test_evaluate_uncertainty_metrics_variance_floor():
    # If predicted std is zero or tiny, variance floor 1e-6 guards against inf
    y_true = np.array([50.0, 60.0])
    y_pred = np.array([50.0, 60.0])
    y_std = np.array([0.0, 0.0])

    m = evaluate_uncertainty_metrics(y_true, y_pred, y_std, variance_floor=1e-6)
    assert np.isfinite(m["nll"])
    assert m["picp"] == 100.0


def test_evaluate_uncertainty_metrics_none_std():
    # For non-probabilistic models, y_std is None
    y_true = np.array([10.0, 20.0, 30.0])
    y_pred = np.array([11.0, 19.0, 31.0])

    m = evaluate_uncertainty_metrics(y_true, y_pred, y_std=None)
    assert m["mae"] == pytest.approx(1.0)
    assert np.isnan(m["nll"])
    assert np.isnan(m["picp"])
    assert np.isnan(m["mpiw"])


def test_preprocessing_pipeline_no_leakage():
    # Verify fit strictly computes stats on train, transform on test
    X_train = np.array([[1.0, 10.0], [2.0, 20.0], [3.0, 30.0], [np.nan, 40.0]])
    X_test = np.array([[2.0, 25.0]])

    pipe = PreprocessingPipeline()
    X_train_scaled = pipe.fit_transform(X_train)
    X_test_scaled = pipe.transform(X_test)

    # Train means must be approximately 0.0
    assert np.allclose(np.mean(X_train_scaled, axis=0), [0.0, 0.0], atol=1e-7)
    assert X_test_scaled.shape == (1, 2)


def test_mape_and_smape_robustness():
    from src.evaluation import evaluate_uncertainty_metrics
    # Low SOH tail (< 10%) should cause large standard MAPE, but smape is bounded <= 200%
    y_true = np.array([5.0, 50.0, 80.0, 100.0])
    y_pred = np.array([8.0, 52.0, 81.0, 99.0])
    m = evaluate_uncertainty_metrics(y_true, y_pred)

    assert m["mape"] > 10.0  # 60% on the first point
    assert 0.0 <= m["smape"] <= 200.0
    assert m["mape_soh50"] < m["mape"]  # restricted to >= 50% removes the 5% point


def test_shared_calibration():
    from src.evaluation import compute_recalibration_factor, apply_shared_calibration
    res_oof = np.array([2.0, -2.0, 2.0, -2.0])
    var_oof = np.array([1.0, 1.0, 1.0, 1.0])
    # Under-confident or over-confident: here residuals are twice the standard deviation
    s = compute_recalibration_factor(res_oof, var_oof)
    assert s == pytest.approx(2.0, abs=1e-3)

    y_test = np.array([80.0, 82.0])
    y_pred = np.array([80.0, 82.0])
    std_raw = np.array([1.0, 1.0])
    res = apply_shared_calibration(y_test, y_pred, std_raw, s)
    assert res["nominal_coverage_pct"] == 95.0
    assert res["mpiw_recal"] == pytest.approx(res["mpiw_raw"] * 2.0, abs=1e-3)


def test_mean_and_median_predictor_baselines():
    from src.baseline_models import MeanPredictorBaseline, MedianPredictorBaseline
    y_train = np.array([10.0, 20.0, 90.0])  # mean = 40, median = 20
    X_train = np.zeros((3, 2))
    X_test = np.zeros((5, 2))

    mean_b = MeanPredictorBaseline().fit(X_train, y_train)
    p_mean, _ = mean_b.predict(X_test)
    assert np.allclose(p_mean, 40.0)

    median_b = MedianPredictorBaseline().fit(X_train, y_train)
    p_median, _ = median_b.predict(X_test)
    assert np.allclose(p_median, 20.0)


def test_quantile_calibration():
    from src.evaluation import compute_quantile_calibration_factor, apply_shared_calibration
    # Construct out-of-fold residuals where std=1.0, but errors are distributed with 95th percentile at 3.0
    res_oof = np.linspace(-3.0, 3.0, 100)
    stds_oof = np.ones_like(res_oof)
    s_q = compute_quantile_calibration_factor(res_oof, stds_oof, confidence_level=0.95)
    # Target q_0.95 is approx 2.85, so s_q ≈ 2.85 / 1.95996 ≈ 1.45
    assert s_q > 1.0

    y_test = np.array([50.0, 60.0])
    y_pred = np.array([50.0, 60.0])
    std_raw = np.array([1.0, 1.0])
    res = apply_shared_calibration(y_test, y_pred, std_raw, s_factor=1.0, s_quantile=s_q)
    assert res["mpiw_recal_q"] > res["mpiw_raw"]


def test_recalibrated_metrics_differ_when_means_differ():
    from src.evaluation import apply_shared_calibration
    # Ground truth
    y_test = np.array([80.0, 85.0, 90.0, 75.0, 70.0])
    # Config C base mean
    mu_c = np.array([81.0, 84.0, 89.0, 76.0, 71.0])
    # Config D shifted mean (different residual pattern)
    mu_d = np.array([85.0, 80.0, 95.0, 70.0, 77.0])
    std_raw = np.array([1.5, 1.5, 1.5, 1.5, 1.5])
    s_factor = 1.25

    calib_c = apply_shared_calibration(y_test, mu_c, std_raw, s_factor)
    calib_d = apply_shared_calibration(y_test, mu_d, std_raw, s_factor)

    # Recalibrated NLL must differ because residuals differ
    assert calib_c["nll_recal"] != pytest.approx(calib_d["nll_recal"], abs=1e-3)
    # Recalibrated RMSE must differ
    assert calib_c["rmse"] != pytest.approx(calib_d["rmse"], abs=1e-3)


def test_per_cell_variants_differ_in_recal_metrics():
    """
    Integration test on per_cell.csv (Phase 3C Task 3):
    For any two model variants in the same fold whose RMSE differs,
    recalibrated NLL and PICP must differ unless residuals are identical.
    """
    import os
    import pandas as pd
    csv_path = "results/metrics/per_cell.csv"
    if not os.path.exists(csv_path):
        pytest.skip("per_cell.csv does not exist yet")
    df = pd.read_csv(csv_path)
    violations = []
    for (dset, cell), grp in df.groupby(["dataset", "cell"]):
        sr_models = grp[grp["model"].str.contains("SR", case=False, na=False)]
        for i, r1 in sr_models.iterrows():
            for j, r2 in sr_models.iterrows():
                if i < j:
                    if abs(r1["RMSE"] - r2["RMSE"]) > 0.05:
                        if abs(r1["NLL_recal"] - r2["NLL_recal"]) < 1e-4 and abs(r1["PICP_recal"] - r2["PICP_recal"]) < 1e-4:
                            violations.append(
                                f"{dset} {cell}: {r1['model']} (RMSE={r1['RMSE']:.2f}) vs {r2['model']} (RMSE={r2['RMSE']:.2f}) "
                                f"have identical NLL_recal={r1['NLL_recal']:.4f} and PICP_recal={r1['PICP_recal']:.2f}"
                            )
    assert len(violations) == 0, f"Found {len(violations)} metric export violations in per_cell.csv:\n" + "\n".join(violations[:5])




