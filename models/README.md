# Exported Models & Checkpoints
This directory stores production and standalone model artifacts for the WD-GPR-SR battery health prediction system.

## Artifacts:
- `phi_theta.onnx`: Deep feature extractor $\phi_\theta: \mathbb{R}^5 \to \mathbb{R}^4$ exported in ONNX format (opset 17, dynamic batch).
- `svgp_arrays.npz`: Precomputed Sparse Variational Gaussian Process inducing arrays and variational weights for BLAS-accelerated NumPy inference.
- `scaler_params.npz`: StandardScaler parameters (mean, scale) fitted exclusively on training cells.
- `libequation.so`: Compiled ANSI C shared library implementing the symbolic residual correction equation $g(x)$.
- PyTorch training checkpoints are saved in `results/checkpoints/`.

