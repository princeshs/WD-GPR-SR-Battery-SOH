"""
deployment/jetson_inference.py
==============================
Stage 7 Part B: Unified Standalone Inference Pipeline for Edge Deployment.
Executes on NVIDIA Jetson or development computer without PyTorch dependencies.
Pipeline flow:
 1. Preprocess: Impute missing values + feature standardization
 2. Deep Branch: ONNX Runtime (or TensorRT on Jetson) for phi_theta feature extractor
 3. Wide & Deep SVGP: Pure NumPy linear algebra for predictive mean and variance
 4. Symbolic Residual: Compiled C shared library (libequation.so) via ctypes
 5. Output: SOH point prediction y_hat and calibrated uncertainty sigma.
"""

import os
import sys
import numpy as np
import onnxruntime as ort

try:
    from deployment.equation_loader import evaluate_residual, evaluate_residual_batch
except ImportError:
    # If running from within deployment/
    from equation_loader import evaluate_residual, evaluate_residual_batch


class JetsonInferencePipeline:
    """
    Unified end-to-end edge inference engine for battery SOH prediction.
    Requires only NumPy and ONNX Runtime (or TensorRT).
    """
    def __init__(self, deployment_dir="deployment", use_tensorrt=False):
        self.deployment_dir = deployment_dir

        # 1. Load Scaler Parameters
        scaler_path = os.path.join(deployment_dir, "scaler_params.npz")
        if not os.path.exists(scaler_path):
            raise FileNotFoundError(f"Scaler parameters not found at {scaler_path}")
        scaler_data = np.load(scaler_path)
        self.scaler_mean = scaler_data["mean"]
        self.scaler_scale = scaler_data["scale"]
        self.imputer_stats = scaler_data["imputer_stats"]
        self.feature_names = list(scaler_data["features"])

        # 2. Load ONNX Session for phi_theta
        onnx_path = os.path.join(deployment_dir, "phi_theta.onnx")
        if not os.path.exists(onnx_path):
            raise FileNotFoundError(f"ONNX model not found at {onnx_path}")

        providers = ["CPUExecutionProvider"]
        if use_tensorrt:
            providers = ["TensorrtExecutionProvider", "CUDAExecutionProvider", "CPUExecutionProvider"]

        opts = ort.SessionOptions()
        opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        opts.intra_op_num_threads = 1
        self.session = ort.InferenceSession(onnx_path, sess_options=opts, providers=providers)

        # 3. Load SVGP Precomputed Inference Matrices
        svgp_path = os.path.join(deployment_dir, "svgp_arrays.npz")
        if not os.path.exists(svgp_path):
            raise FileNotFoundError(f"SVGP arrays not found at {svgp_path}")
        svgp_data = np.load(svgp_path)

        self.Z = svgp_data["Z"]                     # Inducing points (M x D)
        self.Z_phi = svgp_data["Z_phi"]             # Latent inducing points (M x d)
        self.alpha = svgp_data["alpha"]             # Linear mean weights (M,)
        self.A_cov = svgp_data["A_cov"]             # Variational cov update matrix (M x M)
        self.sigma_w2 = float(svgp_data["sigma_w2"])
        self.var_lin = float(svgp_data["var_lin"])
        self.ls_rbf = svgp_data["ls_rbf"]           # ARD lengthscales (D,)
        self.sigma_d2 = float(svgp_data["sigma_d2"])
        self.ls_matern = float(svgp_data["ls_matern"])
        self.mean_const = float(svgp_data["mean_const"])
        self.noise_var = float(svgp_data["noise_var"])
        self.y_mean = float(svgp_data["y_mean"])
        self.y_scale = float(svgp_data["y_scale"])
        self.recal_s = float(svgp_data["recal_s"])

    def preprocess(self, X_raw):
        """Imputes missing values and standardizes features."""
        arr = np.asarray(X_raw, dtype=np.float64)
        is_1d = (arr.ndim == 1)
        if is_1d:
            arr = arr.reshape(1, -1)

        # Impute missing values
        imputed = np.where(np.isnan(arr), self.imputer_stats, arr)
        # Standardize
        scaled = (imputed - self.scaler_mean) / self.scaler_scale
        return scaled, is_1d

    def extract_latent(self, X_scaled):
        """Passes scaled features through deep branch phi_theta."""
        inp = X_scaled.astype(np.float32)
        out = self.session.run(["latent"], {"input": inp})[0]
        return out

    def compute_kernel(self, X_scaled, z_lat):
        """Computes wide + deep kernel between input batch and inducing points Z."""
        # 1. Wide Linear
        k_lin = (X_scaled @ self.Z.T) * self.var_lin

        # 2. Wide ARD-RBF
        d2 = np.sum(((X_scaled[:, None, :] - self.Z[None, :, :]) / self.ls_rbf)**2, axis=-1)
        k_rbf = np.exp(-0.5 * d2)
        k_wide = self.sigma_w2 * (k_lin + k_rbf)

        # 3. Deep Matérn-5/2 on latent space
        d_lat = np.sqrt(np.sum(((z_lat[:, None, :] - self.Z_phi[None, :, :]) / self.ls_matern)**2, axis=-1) + 1e-12)
        sqrt5_d = np.sqrt(5.0) * d_lat
        k_matern = (1.0 + sqrt5_d + (5.0 / 3.0) * (d_lat**2)) * np.exp(-sqrt5_d)
        k_deep = self.sigma_d2 * k_matern

        return k_wide + k_deep

    def predict_gp(self, X_scaled, z_lat):
        """Computes SVGP predictive mean and variance via pure NumPy linear algebra."""
        k_XZ = self.compute_kernel(X_scaled, z_lat)

        # Mean calculation: O(B * M)
        mu_norm = self.mean_const + (k_XZ @ self.alpha)
        mu_gp = mu_norm * self.y_scale + self.y_mean

        # Variance calculation: O(B * M^2) row-wise without BxB matrix allocation
        k_xx_diag = self.sigma_w2 * (np.sum(X_scaled**2, axis=-1) * self.var_lin + 1.0) + self.sigma_d2
        var_update = np.sum((k_XZ @ self.A_cov) * k_XZ, axis=-1)
        var_latent = k_xx_diag + var_update
        var_obs = var_latent + self.noise_var
        var_orig = var_obs * (self.y_scale**2) * (self.recal_s**2)
        std_gp = np.sqrt(np.maximum(var_orig, 1e-6))

        return mu_gp, std_gp

    def predict(self, X_raw):
        """End-to-end SOH point estimation: scale -> phi_theta -> SVGP -> C equation."""
        X_scaled, is_1d = self.preprocess(X_raw)
        z_lat = self.extract_latent(X_scaled)
        mu_gp, _ = self.predict_gp(X_scaled, z_lat)
        g_c = evaluate_residual_batch(X_scaled)
        y_pred = mu_gp + g_c
        return y_pred[0] if is_1d else y_pred

    def predict_with_uncertainty(self, X_raw):
        """End-to-end SOH estimation with calibrated 95% uncertainty bounds."""
        X_scaled, is_1d = self.preprocess(X_raw)
        z_lat = self.extract_latent(X_scaled)
        mu_gp, std_gp = self.predict_gp(X_scaled, z_lat)
        g_c = evaluate_residual_batch(X_scaled)
        y_pred = mu_gp + g_c
        return (y_pred[0], std_gp[0]) if is_1d else (y_pred, std_gp)

    def predict_single(self, x_raw):
        """Optimized single-cycle inference for ultra-low latency edge monitoring."""
        arr = np.asarray(x_raw, dtype=np.float64).reshape(1, -1)
        imp = np.where(np.isnan(arr), self.imputer_stats, arr)
        x_s = (imp - self.scaler_mean) / self.scaler_scale
        z = self.session.run(["latent"], {"input": x_s.astype(np.float32)})[0]

        # Kernel
        k_lin = (x_s @ self.Z.T) * self.var_lin
        d2 = np.sum(((x_s[:, None, :] - self.Z[None, :, :]) / self.ls_rbf)**2, axis=-1)
        k_wide = self.sigma_w2 * (k_lin + np.exp(-0.5 * d2))

        d_lat = np.sqrt(np.sum(((z[:, None, :] - self.Z_phi[None, :, :]) / self.ls_matern)**2, axis=-1) + 1e-12)
        sqrt5_d = np.sqrt(5.0) * d_lat
        k_matern = (1.0 + sqrt5_d + (5.0 / 3.0) * (d_lat**2)) * np.exp(-sqrt5_d)
        k_deep = self.sigma_d2 * k_matern
        k_xz = k_wide + k_deep

        mu = float(self.mean_const + (k_xz @ self.alpha)[0]) * self.y_scale + self.y_mean
        k_xx = self.sigma_w2 * (np.sum(x_s**2) * self.var_lin + 1.0) + self.sigma_d2
        var_u = float(np.sum((k_xz @ self.A_cov) * k_xz))
        var = (k_xx + var_u + self.noise_var) * (self.y_scale**2) * (self.recal_s**2)
        std = float(np.sqrt(max(var, 1e-6)))

        g = evaluate_residual(x_s[0])
        return mu + g, std


if __name__ == "__main__":
    print("Testing JetsonInferencePipeline...")
    pipe = JetsonInferencePipeline()
    sample = [50.0, 3100.0, 1800.0, 1.25, 3.95]
    y_hat, sigma = pipe.predict_single(sample)
    print(f"Sample Input: {sample}")
    print(f"Prediction: SOH = {y_hat:.2f}% ± {1.96 * sigma:.2f}% (95% CI: [{y_hat - 1.96*sigma:.2f}%, {y_hat + 1.96*sigma:.2f}%])")

