"""
src/evaluation.py
=================
Evaluation protocol and metrics for Leave-One-Cell-Out Cross-Validation (LOCO-CV).

Features:
- Deterministic random seeding (numpy, random, torch).
- Strict zero-leakage pipeline: scalers and imputers fit ONLY on training cells.
- Standard regression metrics: MAE, MSE, RMSE, R², MAPE.
- Probabilistic metrics: Gaussian NLL, 95% PICP, 95% MPIW with variance floor.
- Structured LOCO cross-validation runner within NASA and CALCE.
"""

import random
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer


def set_seed(seed: int = 42):
    """
    Sets deterministic random seeds across python, numpy, and torch.
    """
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass


def evaluate_uncertainty_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_std: np.ndarray | None = None,
    variance_floor: float = 1e-6,
    confidence_level: float = 0.95,
) -> dict:
    """
    Computes standard regression metrics and probabilistic uncertainty metrics.

    Parameters
    ----------
    y_true : np.ndarray
        Ground-truth target values.
    y_pred : np.ndarray
        Model predicted mean values.
    y_std : np.ndarray, optional
        Predicted predictive standard deviations (sigma). If None, uncertainty
        metrics return np.nan.
    variance_floor : float
        Minimum variance threshold to avoid numerical instability in log/division.

    Returns
    -------
    dict
        {
            'mae': float,
            'mse': float,
            'rmse': float,
            'r2': float,
            'mape': float,
            'nll': float or np.nan,
            'picp': float or np.nan,
            'mpiw': float or np.nan,
        }
    """
    y_true = np.asarray(y_true, dtype=float).flatten()
    y_pred = np.asarray(y_pred, dtype=float).flatten()

    n = len(y_true)
    if n == 0:
        return {k: np.nan for k in ["mae", "mse", "rmse", "r2", "mape", "nll", "picp", "mpiw"]}

    mae = float(mean_absolute_error(y_true, y_pred))
    mse = float(mean_squared_error(y_true, y_pred))
    rmse = float(np.sqrt(mse))

    # Guard r2 if constant y_true
    if np.var(y_true) < 1e-9:
        r2 = 0.0
    else:
        r2 = float(r2_score(y_true, y_pred))

    # 1. Standard MAPE: avoid division by zero
    nonzero_mask = np.abs(y_true) > 1e-4
    if nonzero_mask.sum() > 0:
        mape = float(np.mean(np.abs((y_true[nonzero_mask] - y_pred[nonzero_mask]) / np.abs(y_true[nonzero_mask]))) * 100.0)
    else:
        mape = np.nan

    # 2. Restricted MAPE for SOH >= 50% (guards against end-of-life tail division explosion on CALCE)
    mask_50 = y_true >= 50.0
    if mask_50.sum() > 0:
        mape_soh50 = float(np.mean(np.abs((y_true[mask_50] - y_pred[mask_50]) / np.abs(y_true[mask_50]))) * 100.0)
    else:
        mape_soh50 = np.nan

    # 3. Symmetric MAPE (sMAPE): bounded [0, 200%]
    denom_smape = (np.abs(y_true) + np.abs(y_pred)) / 2.0
    mask_smape = denom_smape > 1e-4
    if mask_smape.sum() > 0:
        smape = float(np.mean(np.abs(y_pred[mask_smape] - y_true[mask_smape]) / denom_smape[mask_smape]) * 100.0)
    else:
        smape = np.nan

    # Uncertainty Metrics
    if y_std is not None:
        y_std = np.asarray(y_std, dtype=float).flatten()
        var = np.maximum(y_std ** 2, variance_floor)
        std_clamped = np.sqrt(var)

        # Gaussian Negative Log-Likelihood:
        # 0.5 * ln(2*pi*sigma^2) + 0.5 * (y - mu)^2 / sigma^2
        nll_elements = 0.5 * np.log(2.0 * np.pi * var) + 0.5 * ((y_true - y_pred) ** 2) / var
        nll = float(np.mean(nll_elements))

        # Prediction Interval Coverage Probability (PICP) for nominal confidence level (default 95%)
        if abs(confidence_level - 0.95) < 1e-5:
            z_score = 1.95996
        else:
            from scipy.stats import norm
            z_score = float(norm.ppf((1.0 + confidence_level) / 2.0))
        lower_bound = y_pred - z_score * std_clamped
        upper_bound = y_pred + z_score * std_clamped
        covered = (y_true >= lower_bound) & (y_true <= upper_bound)
        picp = float(np.mean(covered) * 100.0)

        # Mean Prediction Interval Width (MPIW)
        mpiw = float(np.mean(upper_bound - lower_bound))
    else:
        nll = np.nan
        picp = np.nan
        mpiw = np.nan

    return {
        "mae": mae,
        "mse": mse,
        "rmse": rmse,
        "r2": r2,
        "mape": mape,
        "mape_soh50": mape_soh50,
        "smape": smape,
        "nll": nll,
        "picp": picp,
        "mpiw": mpiw,
    }


def compute_recalibration_factor(
    residuals_oof: np.ndarray,
    variances_oof: np.ndarray,
    clip_range: tuple[float, float] = (0.05, 20.0),
) -> float:
    """
    Fits an empirical variance scaling factor s on training out-of-fold residuals:
      s^2 = (1 / N) * sum( r_i^2 / sigma_i^2 )
    Returns scalar s (multiplier for predictive std).
    Computed strictly from inner out-of-fold training residuals.
    """
    res = np.asarray(residuals_oof, dtype=float).flatten()
    vars_ = np.asarray(variances_oof, dtype=float).flatten()
    ratio = (res ** 2) / np.maximum(vars_, 1e-6)
    s2 = float(np.mean(np.clip(ratio, clip_range[0], clip_range[1])))
    return float(np.sqrt(max(s2, 1e-6)))


def compute_quantile_calibration_factor(
    residuals_oof: np.ndarray,
    stds_oof: np.ndarray,
    confidence_level: float = 0.95,
) -> float:
    """
    Fits an empirical quantile scaling factor on training out-of-fold residuals:
      z_i = |r_i| / sigma_i
      q_target = quantile(|z|, confidence_level)
      s_quantile = q_target / z_crit
    where z_crit is the normal critical value (1.95996 for 95%).
    Guarantees empirical coverage >= confidence_level on inner out-of-fold residuals.
    """
    from scipy.stats import norm
    z_crit = float(norm.ppf(1.0 - (1.0 - confidence_level) / 2.0))
    res = np.abs(np.asarray(residuals_oof, dtype=float).flatten())
    stds = np.maximum(np.asarray(stds_oof, dtype=float).flatten(), 1e-6)
    z = res / stds
    q_val = float(np.quantile(z, confidence_level))
    return float(max(q_val / z_crit, 1e-4))


def apply_shared_calibration(
    y_test: np.ndarray,
    y_pred: np.ndarray,
    std_raw: np.ndarray | None,
    s_factor: float = 1.0,
    s_quantile: float | None = None,
    confidence_level: float = 0.95,
) -> dict[str, float]:
    """
    Applies identical calibration procedure to any probabilistic model:
    Reports raw, variance-scaled (s_factor), and quantile-scaled (s_quantile)
    PICP, MPIW, NLL alongside explicit nominal coverage (95%).
    """
    if std_raw is None:
        m_raw = evaluate_uncertainty_metrics(y_test, y_pred, None, confidence_level=confidence_level)
        return {
            "nominal_coverage_pct": confidence_level * 100.0,
            "s_factor": np.nan,
            "s_quantile": np.nan,
            "nll_raw": np.nan,
            "picp_raw": np.nan,
            "mpiw_raw": np.nan,
            "nll_recal": np.nan,
            "picp_recal": np.nan,
            "mpiw_recal": np.nan,
            "nll_recal_q": np.nan,
            "picp_recal_q": np.nan,
            "mpiw_recal_q": np.nan,
            "mae": m_raw["mae"],
            "mse": m_raw["mse"],
            "rmse": m_raw["rmse"],
            "r2": m_raw["r2"],
            "mape": m_raw["mape"],
            "mape_soh50": m_raw["mape_soh50"],
            "smape": m_raw["smape"],
        }

    std_raw = np.asarray(std_raw, dtype=float).flatten()
    std_recal = std_raw * s_factor
    std_q = std_raw * (s_quantile if s_quantile is not None else s_factor)

    m_raw = evaluate_uncertainty_metrics(y_test, y_pred, std_raw, confidence_level=confidence_level)
    m_recal = evaluate_uncertainty_metrics(y_test, y_pred, std_recal, confidence_level=confidence_level)
    m_q = evaluate_uncertainty_metrics(y_test, y_pred, std_q, confidence_level=confidence_level)

    return {
        "nominal_coverage_pct": confidence_level * 100.0,
        "s_factor": s_factor,
        "s_quantile": s_quantile if s_quantile is not None else s_factor,
        "nll_raw": m_raw["nll"],
        "picp_raw": m_raw["picp"],
        "mpiw_raw": m_raw["mpiw"],
        "nll_recal": m_recal["nll"],
        "picp_recal": m_recal["picp"],
        "mpiw_recal": m_recal["mpiw"],
        "nll_recal_q": m_q["nll"],
        "picp_recal_q": m_q["picp"],
        "mpiw_recal_q": m_q["mpiw"],
        "mae": m_raw["mae"],
        "mse": m_raw["mse"],
        "rmse": m_raw["rmse"],
        "r2": m_raw["r2"],
        "mape": m_raw["mape"],
        "mape_soh50": m_raw["mape_soh50"],
        "smape": m_raw["smape"],
    }


class PreprocessingPipeline:
    """
    Zero-leakage preprocessor that fits Imputer and StandardScaler
    strictly on training cells only.
    """

    def __init__(self):
        self.imputer = SimpleImputer(strategy="median")
        self.scaler = StandardScaler()

    def fit(self, X_train: np.ndarray):
        X_imp = self.imputer.fit_transform(X_train)
        self.scaler.fit(X_imp)
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        X_imp = self.imputer.transform(X)
        return self.scaler.transform(X_imp)

    def fit_transform(self, X_train: np.ndarray) -> np.ndarray:
        return self.fit(X_train).transform(X_train)

