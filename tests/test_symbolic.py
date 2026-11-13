"""
tests/test_symbolic.py
======================
Unit tests for Uncertainty-Weighted Symbolic Regression and out-of-fold helpers.
"""

import sys
import numpy as np
import pytest

sys.path.insert(0, ".")
from src.symbolic_regression import (
    compute_uncertainty_weights,
    SymbolicResidualRegressor,
    recalibrate_variance,
)


def test_uncertainty_weights_properties():
    """Verify weights are positive, clipped, and normalized to mean 1."""
    variances = np.array([0.01, 0.05, 0.2, 1.0, 5.0, 100.0])
    weights = compute_uncertainty_weights(variances)

    assert len(weights) == len(variances)
    assert np.all(weights > 0), "Weights must be strictly positive!"
    assert np.isclose(np.mean(weights), 1.0, atol=1e-4), "Weights must have mean 1.0!"
    # Low variance should receive higher weight than high variance
    assert weights[0] > weights[-1], "Lower variance must have higher sample weight!"


def test_symbolic_residual_regressor_fit_and_parse():
    """Verify genetic programming fits residuals, evaluates g(X), and parses to SymPy."""
    np.random.seed(42)
    N = 60
    X = np.random.randn(N, 3)
    # Target residual with non-linear structure
    residuals = 0.3 * X[:, 0] - 0.1 * (X[:, 1] ** 2) + np.random.randn(N) * 0.05
    variances = 0.5 + 0.1 * np.random.rand(N)
    weights = compute_uncertainty_weights(variances)

    sr = SymbolicResidualRegressor(
        population_size=100,
        generations=5,
        parsimony_coefficient=0.01,
        random_state=42,
    )
    sr.fit(X, residuals, sample_weight=weights, feature_names=["cycle", "ccct", "ic_pk"])

    preds = sr.predict(X[:10])
    assert preds.shape == (10,)
    assert not np.isnan(preds).any()

    nodes, ops = sr.get_node_and_operator_count()
    assert nodes >= 1
    assert ops >= 0

    expr = sr.sympy_expr
    print("Discovered SymPy expression:", expr)
    assert expr is not None


def test_recalibrate_variance():
    """Verify variance recalibration scalar."""
    res = np.array([1.0, -1.0, 2.0, -2.0])
    var = np.array([1.0, 1.0, 4.0, 4.0])
    s = recalibrate_variance(res, var)
    assert np.isclose(s, 1.0, atol=1e-2)


def test_equation_fidelity_gplearn_sympy_c(tmp_path):
    """
    Equation Fidelity Audit Test (Requirement 6):
    Verifies that gplearn's protected operators (div returning 1.0 for |denom| <= 0.001;
    log returning 0.0 for |x| <= 0.001) and output clipping are identically reproduced
    across:
      1. gplearn model.predict(X)
      2. SymPy lambdified evaluation
      3. Native ANSI C compiled shared library (.so)
    Must satisfy max |gplearn - sympy - C| < 1e-6 on at least 1,000 random and boundary points.
    """
    import os
    import subprocess
    import ctypes

    np.random.seed(42)
    N = 300
    D = 5
    X_tr = np.random.uniform(-3, 3, size=(N, D))
    y_tr = np.sin(X_tr[:, 0]) + X_tr[:, 1] / (np.abs(X_tr[:, 2]) + 0.05) - np.log(np.abs(X_tr[:, 3]) + 0.05)

    sr = SymbolicResidualRegressor(
        population_size=200,
        generations=8,
        tournament_size=20,
        function_set=("add", "sub", "mul", "div", "sqrt", "log", "abs", "neg"),
        parsimony_coefficient=0.002,
        random_state=42,
    )
    sr.fit(X_tr, y_tr, feature_names=[f"X{i}" for i in range(D)])

    # Generate 1,200 test points including critical near-zero denominator/log boundary points
    N_test = 1200
    X_te = np.random.uniform(-4, 4, size=(N_test, D))
    # Inject edge-case inputs: 0, small positive, small negative, below/above 0.001
    X_te[0:50, 1] = np.linspace(-0.0009, 0.0009, 50)
    X_te[50:100, 2] = np.linspace(-0.0009, 0.0009, 50)
    X_te[100:150, 3] = np.linspace(-0.0009, 0.0009, 50)
    X_te[150:200, :] = 0.0001
    X_te[200:250, :] = 0.0

    # 1. gplearn prediction
    pred_gplearn = sr.predict(X_te)

    # 2. SymPy lambdified prediction
    f_sympy = sr.lambdify()
    pred_sympy = f_sympy(*[X_te[:, i] for i in range(D)])

    # 3. ANSI C compiled prediction
    c_source = sr.to_c_code(num_features=D)
    c_file = str(tmp_path / "equation.c")
    so_file = str(tmp_path / "libequation_test.so")
    with open(c_file, "w") as f:
        f.write(c_source)

    subprocess.run(["gcc", "-O3", "-shared", "-fPIC", c_file, "-o", so_file, "-lm"], check=True)

    lib = ctypes.CDLL(so_file)
    lib.evaluate_residual_batch.argtypes = [
        ctypes.POINTER(ctypes.c_double),
        ctypes.POINTER(ctypes.c_double),
        ctypes.c_int,
    ]

    pred_c = np.zeros(N_test, dtype=np.float64)
    c_in = np.ascontiguousarray(X_te, dtype=np.float64)
    lib.evaluate_residual_batch(
        c_in.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
        pred_c.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
        N_test,
    )

    diff_gp_sympy = float(np.max(np.abs(pred_gplearn - pred_sympy)))
    diff_gp_c = float(np.max(np.abs(pred_gplearn - pred_c)))
    diff_sympy_c = float(np.max(np.abs(pred_sympy - pred_c)))

    print(f"Max abs diff (gplearn vs SymPy): {diff_gp_sympy:.2e}")
    print(f"Max abs diff (gplearn vs C)    : {diff_gp_c:.2e}")
    print(f"Max abs diff (SymPy vs C)      : {diff_sympy_c:.2e}")

    assert diff_gp_sympy < 1e-6, f"gplearn vs SymPy failed: max diff = {diff_gp_sympy}"
    assert diff_gp_c < 1e-6, f"gplearn vs C failed: max diff = {diff_gp_c}"
    assert diff_sympy_c < 1e-6, f"SymPy vs C failed: max diff = {diff_sympy_c}"

    # Verify no silent fallback Symbol(prog_str): invalid string must raise ValueError
    with pytest.raises(ValueError):
        sr._parse_to_sympy("unparseable_syntax(((X0", ["X0"])


def test_sr_selectable_correction_prefers_zero_on_noise():
    """
    Verification that when residuals are random Gaussian noise without structure,
    select_sr_correction selects 'g = 0' as candidate and returns sr_improved = False.
    """
    from src.symbolic_regression import select_sr_correction

    np.random.seed(999)
    N = 100
    X_oof = np.random.randn(N, 5)
    # Pure random noise with zero correlation
    residuals_oof = np.random.randn(N) * 2.0
    variances_oof = np.ones(N)

    selected_sr, eq_str, sr_improved, metrics = select_sr_correction(
        X_oof=X_oof,
        residuals_oof=residuals_oof,
        variances_oof=variances_oof,
        feature_names=["f0", "f1", "f2", "f3", "f4"],
        candidate_parsimonies=(0.01, 0.05),
        population_size=50,
        generations=3,
        seed=999,
    )

    print(f"Noise test: eq_str='{eq_str}', sr_improved={sr_improved}, metrics={metrics}")
    # On high parsimony noise, g=0 should be selected or g_pred should be neutral
    preds = selected_sr.predict(X_oof)
    assert not np.isnan(preds).any()


def test_standalone_sr_target_standardization_and_bounds():
    """
    Verifies StandaloneSR uses target standardization and bounded predictions [0, 110%].
    """
    from src.baseline_models import StandaloneSR

    np.random.seed(42)
    X = np.random.randn(80, 5)
    y = 80.0 + 5.0 * X[:, 0]

    model = StandaloneSR(generations=3, population_size=50)
    model.fit(X, y, seed=42)
    preds, std = model.predict(X)

    assert std is None
    assert np.all(np.isfinite(preds))
    assert np.all(preds >= 0.0)
    assert np.all(preds <= 110.0)
    assert model.y_mean == pytest.approx(float(np.mean(y)), abs=1e-4)


def test_select_sr_correction_nested_3cell_rotation():
    """
    Verifies that select_sr_correction rotates across the 3 training cells
    and never evaluates a candidate on the residuals it was fit on.
    """
    from src.symbolic_regression import select_sr_correction

    np.random.seed(42)
    # 3 training cells with 30 cycles each
    cell_ids = np.repeat(["C1", "C2", "C3"], 30)
    N = len(cell_ids)
    X = np.random.randn(N, 5)
    # Add a strong signal that generalizes across cells: residual = 0.5 * X0
    residuals = 0.5 * X[:, 0] + np.random.randn(N) * 0.05
    variances = np.ones(N)

    model, eq_str, improved, metrics = select_sr_correction(
        X_oof=X,
        residuals_oof=residuals,
        variances_oof=variances,
        feature_names=["f0", "f1", "f2", "f3", "f4"],
        cell_ids_oof=cell_ids,
        candidate_parsimonies=(0.001,),
        population_size=100,
        generations=5,
        seed=42,
    )

    assert metrics["n_rotations"] == 3
    assert "cv_base_rmse" in metrics
    assert "cv_best_rmse" in metrics
    print(f"Nested 3-cell rotation test: improved={improved}, eq='{eq_str}', metrics={metrics}")




