"""
src/baseline_models.py
======================
Standardized baseline model suite for battery SOH prediction.
All models expose a common interface:
    fit(X_train, y_train, X_val=None, y_val=None, seed=42)
    predict(X) -> (y_pred, y_std or None)

Models implemented:
1. Linear Regression (Ridge)
2. Random Forest Regressor
3. Support Vector Regression (SVR - RBF)
4. XGBoost Regressor
5. Exact GPR (RBF kernel)
6. Exact GPR (Matérn-5/2 kernel)
7. Exact GPR (ARD RBF kernel)
8. Sparse GPR (GPyTorch SVGP with inducing points & ARD)
9. Standard Deep-Kernel GPR (MLP feature extractor -> RBF kernel)
10. WideNN (PyTorch wide feedforward network)
11. DeepNN (PyTorch deep feedforward network)
12. Wide&Deep NN (PyTorch Wide & Deep architecture)
13. Standalone Symbolic Regression (gplearn SymbolicRegressor)
14. BiLSTM on cycle sequences (PyTorch bidirectional LSTM on window of past cycles)
"""

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
import gpytorch
from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, Matern, ConstantKernel as C
import xgboost as xgb
from gplearn.genetic import SymbolicRegressor

from src.evaluation import set_seed


# =====================================================================
# 1. Classical ML Baselines
# =====================================================================

class LinearRegressionBaseline:
    def __init__(self, alpha: float = 1.0):
        self.alpha = alpha
        self.model = None

    def fit(self, X_train: np.ndarray, y_train: np.ndarray, X_val=None, y_val=None, seed: int = 42):
        set_seed(seed)
        self.model = Ridge(alpha=self.alpha, random_state=seed)
        self.model.fit(X_train, y_train)
        return self

    def predict(self, X: np.ndarray) -> tuple[np.ndarray, None]:
        return self.model.predict(X), None


class RandomForestBaseline:
    def __init__(self, n_estimators: int = 100, max_depth: int = 12):
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.model = None

    def fit(self, X_train: np.ndarray, y_train: np.ndarray, X_val=None, y_val=None, seed: int = 42):
        set_seed(seed)
        self.model = RandomForestRegressor(
            n_estimators=self.n_estimators,
            max_depth=self.max_depth,
            random_state=seed,
            n_jobs=-1,
        )
        self.model.fit(X_train, y_train)
        return self

    def predict(self, X: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        # Empirical ensemble standard deviation across decision trees
        preds = np.array([tree.predict(X) for tree in self.model.estimators_])
        mean = np.mean(preds, axis=0)
        std = np.std(preds, axis=0)
        return mean, std


class SVRBaseline:
    def __init__(self, C_val: float = 10.0, epsilon: float = 0.1):
        self.C_val = C_val
        self.epsilon = epsilon
        self.model = None

    def fit(self, X_train: np.ndarray, y_train: np.ndarray, X_val=None, y_val=None, seed: int = 42):
        set_seed(seed)
        self.model = SVR(kernel="rbf", C=self.C_val, epsilon=self.epsilon)
        self.model.fit(X_train, y_train)
        return self

    def predict(self, X: np.ndarray) -> tuple[np.ndarray, None]:
        return self.model.predict(X), None


class XGBoostBaseline:
    def __init__(self, n_estimators: int = 100, max_depth: int = 4, lr: float = 0.05):
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.lr = lr
        self.model = None

    def fit(self, X_train: np.ndarray, y_train: np.ndarray, X_val=None, y_val=None, seed: int = 42):
        set_seed(seed)
        self.model = xgb.XGBRegressor(
            n_estimators=self.n_estimators,
            max_depth=self.max_depth,
            learning_rate=self.lr,
            random_state=seed,
            n_jobs=2,
            verbosity=0,
        )
        self.model.fit(X_train, y_train)
        return self

    def predict(self, X: np.ndarray) -> tuple[np.ndarray, None]:
        return self.model.predict(X), None


# =====================================================================
# 2. Exact Gaussian Process Regressors (scikit-learn)
# =====================================================================

class ExactGPR_RBF:
    def __init__(self, alpha: float = 0.05):
        self.alpha = alpha
        self.model = None

    def fit(self, X_train: np.ndarray, y_train: np.ndarray, X_val=None, y_val=None, seed: int = 42):
        set_seed(seed)
        kernel = C(1.0, (1e-2, 1e2)) * RBF(length_scale=1.0, length_scale_bounds=(1e-2, 1e2))
        self.model = GaussianProcessRegressor(
            kernel=kernel,
            alpha=self.alpha,
            optimizer="fmin_l_bfgs_b",
            n_restarts_optimizer=0,
            random_state=seed,
            normalize_y=True,
        )
        # Subsample training data if CALCE is large to keep exact GPR fast
        if len(X_train) > 600:
            idx = np.linspace(0, len(X_train) - 1, 600, dtype=int)
            X_tr, y_tr = X_train[idx], y_train[idx]
        else:
            X_tr, y_tr = X_train, y_train
        self.model.fit(X_tr, y_tr)
        return self

    def predict(self, X: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        mean, std = self.model.predict(X, return_std=True)
        return mean, std


class ExactGPR_Matern52:
    def __init__(self, alpha: float = 0.05):
        self.alpha = alpha
        self.model = None

    def fit(self, X_train: np.ndarray, y_train: np.ndarray, X_val=None, y_val=None, seed: int = 42):
        set_seed(seed)
        kernel = C(1.0, (1e-2, 1e2)) * Matern(length_scale=1.0, length_scale_bounds=(1e-2, 1e2), nu=2.5)
        self.model = GaussianProcessRegressor(
            kernel=kernel,
            alpha=self.alpha,
            optimizer="fmin_l_bfgs_b",
            n_restarts_optimizer=0,
            random_state=seed,
            normalize_y=True,
        )
        if len(X_train) > 600:
            idx = np.linspace(0, len(X_train) - 1, 600, dtype=int)
            X_tr, y_tr = X_train[idx], y_train[idx]
        else:
            X_tr, y_tr = X_train, y_train
        self.model.fit(X_tr, y_tr)
        return self

    def predict(self, X: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        mean, std = self.model.predict(X, return_std=True)
        return mean, std


class ExactGPR_ARD:
    def __init__(self, alpha: float = 0.05):
        self.alpha = alpha
        self.model = None

    def fit(self, X_train: np.ndarray, y_train: np.ndarray, X_val=None, y_val=None, seed: int = 42):
        set_seed(seed)
        d = X_train.shape[1]
        kernel = C(1.0, (1e-2, 1e2)) * RBF(length_scale=np.ones(d), length_scale_bounds=(1e-2, 1e2))
        self.model = GaussianProcessRegressor(
            kernel=kernel,
            alpha=self.alpha,
            optimizer="fmin_l_bfgs_b",
            n_restarts_optimizer=0,
            random_state=seed,
            normalize_y=True,
        )
        if len(X_train) > 600:
            idx = np.linspace(0, len(X_train) - 1, 600, dtype=int)
            X_tr, y_tr = X_train[idx], y_train[idx]
        else:
            X_tr, y_tr = X_train, y_train
        self.model.fit(X_tr, y_tr)
        return self

    def predict(self, X: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        mean, std = self.model.predict(X, return_std=True)
        return mean, std


# =====================================================================
# 3. Sparse GPR & Deep-Kernel GPR (GPyTorch)
# =====================================================================

class _SVGPModel(gpytorch.models.ApproximateGP):
    def __init__(self, inducing_points: torch.Tensor, input_dim: int):
        variational_distribution = gpytorch.variational.CholeskyVariationalDistribution(
            inducing_points.size(0)
        )
        variational_strategy = gpytorch.variational.VariationalStrategy(
            self,
            inducing_points,
            variational_distribution,
            learn_inducing_locations=True,
        )
        super().__init__(variational_strategy)
        self.mean_module = gpytorch.means.ConstantMean()
        self.covar_module = gpytorch.kernels.ScaleKernel(
            gpytorch.kernels.RBFKernel(ard_num_dims=input_dim)
        )

    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)


class SparseGPR_SVGP:
    def __init__(self, n_inducing: int = 25, lr: float = 0.03, epochs: int = 40):
        self.n_inducing = n_inducing
        self.lr = lr
        self.epochs = epochs
        self.model = None
        self.likelihood = None
        self.y_mean = 0.0
        self.y_scale = 1.0

    def fit(self, X_train: np.ndarray, y_train: np.ndarray, X_val=None, y_val=None, seed: int = 42):
        set_seed(seed)
        self.y_mean = float(np.mean(y_train))
        self.y_scale = float(np.std(y_train) + 1e-6)
        y_norm = (y_train - self.y_mean) / self.y_scale

        X_t = torch.tensor(X_train, dtype=torch.float32)
        y_t = torch.tensor(y_norm, dtype=torch.float32)

        # Initialize inducing points via uniform random subsampling
        perm = torch.randperm(len(X_t))
        inducing = X_t[perm[:min(self.n_inducing, len(X_t))]].clone()

        self.model = _SVGPModel(inducing, input_dim=X_train.shape[1])
        self.likelihood = gpytorch.likelihoods.GaussianLikelihood()

        self.model.train()
        self.likelihood.train()

        optimizer = torch.optim.Adam([
            {"params": self.model.parameters()},
            {"params": self.likelihood.parameters()},
        ], lr=self.lr)

        mll = gpytorch.mlls.VariationalELBO(self.likelihood, self.model, num_data=len(X_t))
        dataset = TensorDataset(X_t, y_t)
        loader = DataLoader(dataset, batch_size=128, shuffle=True)

        for _ in range(self.epochs):
            for x_b, y_b in loader:
                optimizer.zero_grad()
                out = self.model(x_b)
                loss = -mll(out, y_b)
                loss.backward()
                optimizer.step()

        self.model.eval()
        self.likelihood.eval()
        return self

    def predict(self, X: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        X_t = torch.tensor(X, dtype=torch.float32)
        with torch.no_grad(), gpytorch.settings.fast_pred_var():
            dist = self.likelihood(self.model(X_t))
            mean = dist.mean.numpy() * self.y_scale + self.y_mean
            std = dist.stddev.numpy() * self.y_scale
        return mean, std


class _DKLFeatureExtractor(nn.Module):
    def __init__(self, in_features: int, latent_dim: int = 4):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_features, 32),
            nn.LayerNorm(32),
            nn.ReLU(),
            nn.Linear(32, latent_dim),
        )

    def forward(self, x):
        return self.net(x)


class _DKLGPModel(gpytorch.models.ApproximateGP):
    def __init__(self, inducing_points: torch.Tensor, latent_dim: int):
        variational_distribution = gpytorch.variational.CholeskyVariationalDistribution(
            inducing_points.size(0)
        )
        variational_strategy = gpytorch.variational.VariationalStrategy(
            self,
            inducing_points,
            variational_distribution,
            learn_inducing_locations=True,
        )
        super().__init__(variational_strategy)
        self.mean_module = gpytorch.means.ConstantMean()
        self.covar_module = gpytorch.kernels.ScaleKernel(
            gpytorch.kernels.RBFKernel(ard_num_dims=latent_dim)
        )

    def forward(self, x):
        return gpytorch.distributions.MultivariateNormal(self.mean_module(x), self.covar_module(x))


class _DKLCombined(nn.Module):
    def __init__(self, feat_extractor: nn.Module, gp: _DKLGPModel):
        super().__init__()
        self.feat_extractor = feat_extractor
        self.gp = gp

    def forward(self, x):
        z = self.feat_extractor(x)
        return self.gp(z)


class DeepKernelGPR:
    def __init__(self, latent_dim: int = 4, n_inducing: int = 25, lr: float = 0.02, epochs: int = 40):
        self.latent_dim = latent_dim
        self.n_inducing = n_inducing
        self.lr = lr
        self.epochs = epochs
        self.model = None
        self.gp = None
        self.likelihood = None
        self.y_mean = 0.0
        self.y_scale = 1.0

    def fit(self, X_train: np.ndarray, y_train: np.ndarray, X_val=None, y_val=None, seed: int = 42):
        set_seed(seed)
        self.y_mean = float(np.mean(y_train))
        self.y_scale = float(np.std(y_train) + 1e-6)
        y_norm = (y_train - self.y_mean) / self.y_scale

        X_t = torch.tensor(X_train, dtype=torch.float32)
        y_t = torch.tensor(y_norm, dtype=torch.float32)

        feat_net = _DKLFeatureExtractor(in_features=X_train.shape[1], latent_dim=self.latent_dim)
        with torch.no_grad():
            initial_projected = feat_net(X_t)
            perm = torch.randperm(len(initial_projected))
            inducing = initial_projected[perm[:min(self.n_inducing, len(X_t))]].clone()

        self.gp = _DKLGPModel(inducing, self.latent_dim)
        self.model = _DKLCombined(feat_net, self.gp)
        self.likelihood = gpytorch.likelihoods.GaussianLikelihood()

        self.model.train()
        self.likelihood.train()

        optimizer = torch.optim.Adam([
            {"params": self.model.parameters()},
            {"params": self.likelihood.parameters()},
        ], lr=self.lr)
        mll = gpytorch.mlls.VariationalELBO(self.likelihood, self.gp, num_data=len(X_t))

        dataset = TensorDataset(X_t, y_t)
        loader = DataLoader(dataset, batch_size=128, shuffle=True)

        for _ in range(self.epochs):
            for x_b, y_b in loader:
                optimizer.zero_grad()
                out = self.model(x_b)
                loss = -mll(out, y_b)
                loss.backward()
                optimizer.step()

        self.model.eval()
        self.likelihood.eval()
        return self

    def predict(self, X: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        X_t = torch.tensor(X, dtype=torch.float32)
        with torch.no_grad(), gpytorch.settings.fast_pred_var():
            dist = self.likelihood(self.model(X_t))
            mean = dist.mean.numpy() * self.y_scale + self.y_mean
            std = dist.stddev.numpy() * self.y_scale
        return mean, std


# =====================================================================
# 4. Neural Network Baselines (PyTorch)
# =====================================================================

class _WideNet(nn.Module):
    def __init__(self, in_features: int, width: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_features, width),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(width, 1),
        )

    def forward(self, x):
        return self.net(x).squeeze(-1)


class WideNN:
    def __init__(self, width: int = 128, lr: float = 0.01, epochs: int = 50):
        self.width = width
        self.lr = lr
        self.epochs = epochs
        self.model = None

    def fit(self, X_train: np.ndarray, y_train: np.ndarray, X_val=None, y_val=None, seed: int = 42):
        set_seed(seed)
        self.model = _WideNet(X_train.shape[1], width=self.width)
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)
        criterion = nn.MSELoss()

        X_t = torch.tensor(X_train, dtype=torch.float32)
        y_t = torch.tensor(y_train, dtype=torch.float32)
        loader = DataLoader(TensorDataset(X_t, y_t), batch_size=64, shuffle=True)

        self.model.train()
        for _ in range(self.epochs):
            for x_b, y_b in loader:
                optimizer.zero_grad()
                loss = criterion(self.model(x_b), y_b)
                loss.backward()
                optimizer.step()

        self.model.eval()
        return self

    def predict(self, X: np.ndarray) -> tuple[np.ndarray, None]:
        self.model.eval()
        with torch.no_grad():
            preds = self.model(torch.tensor(X, dtype=torch.float32)).numpy()
        return preds, None


class _DeepNet(nn.Module):
    def __init__(self, in_features: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_features, 64),
            nn.LayerNorm(64),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(64, 32),
            nn.LayerNorm(32),
            nn.ReLU(),
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Linear(16, 1),
        )

    def forward(self, x):
        return self.net(x).squeeze(-1)


class DeepNN:
    def __init__(self, lr: float = 0.01, epochs: int = 50):
        self.lr = lr
        self.epochs = epochs
        self.model = None

    def fit(self, X_train: np.ndarray, y_train: np.ndarray, X_val=None, y_val=None, seed: int = 42):
        set_seed(seed)
        self.model = _DeepNet(X_train.shape[1])
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)
        criterion = nn.MSELoss()

        X_t = torch.tensor(X_train, dtype=torch.float32)
        y_t = torch.tensor(y_train, dtype=torch.float32)
        loader = DataLoader(TensorDataset(X_t, y_t), batch_size=64, shuffle=True)

        self.model.train()
        for _ in range(self.epochs):
            for x_b, y_b in loader:
                optimizer.zero_grad()
                loss = criterion(self.model(x_b), y_b)
                loss.backward()
                optimizer.step()

        self.model.eval()
        return self

    def predict(self, X: np.ndarray) -> tuple[np.ndarray, None]:
        self.model.eval()
        with torch.no_grad():
            preds = self.model(torch.tensor(X, dtype=torch.float32)).numpy()
        return preds, None


class _WideAndDeepNet(nn.Module):
    def __init__(self, in_features: int):
        super().__init__()
        self.wide = nn.Linear(in_features, 1)
        self.deep = nn.Sequential(
            nn.Linear(in_features, 64),
            nn.LayerNorm(64),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
        )

    def forward(self, x):
        return (self.wide(x) + self.deep(x)).squeeze(-1)


class WideAndDeepNN:
    def __init__(self, lr: float = 0.01, epochs: int = 50):
        self.lr = lr
        self.epochs = epochs
        self.model = None

    def fit(self, X_train: np.ndarray, y_train: np.ndarray, X_val=None, y_val=None, seed: int = 42):
        set_seed(seed)
        self.model = _WideAndDeepNet(X_train.shape[1])
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)
        criterion = nn.MSELoss()

        X_t = torch.tensor(X_train, dtype=torch.float32)
        y_t = torch.tensor(y_train, dtype=torch.float32)
        loader = DataLoader(TensorDataset(X_t, y_t), batch_size=64, shuffle=True)

        self.model.train()
        for _ in range(self.epochs):
            for x_b, y_b in loader:
                optimizer.zero_grad()
                loss = criterion(self.model(x_b), y_b)
                loss.backward()
                optimizer.step()

        self.model.eval()
        return self

    def predict(self, X: np.ndarray) -> tuple[np.ndarray, None]:
        self.model.eval()
        with torch.no_grad():
            preds = self.model(torch.tensor(X, dtype=torch.float32)).numpy()
        return preds, None


# =====================================================================
# 5. Standalone Symbolic Regression (gplearn)
# =====================================================================

class StandaloneSR:
    """
    Standalone Symbolic Regression baseline:
    Trains directly on battery features to predict SOH without a Gaussian Process prior.
    Standardizes target SOH using training-only statistics to stabilize genetic evolution,
    safely inverts predictions, and caps predictions to physical SOH limits [0%, 110%].
    """
    def __init__(self, generations: int = 15, population_size: int = 300):
        self.generations = generations
        self.population_size = population_size
        self.model = None
        self.y_mean = 0.0
        self.y_std = 1.0
        self.has_nonfinite = False

    def fit(self, X_train: np.ndarray, y_train: np.ndarray, X_val=None, y_val=None, seed: int = 42):
        set_seed(seed)
        self.y_mean = float(np.mean(y_train))
        self.y_std = float(np.std(y_train)) if float(np.std(y_train)) > 1e-4 else 1.0
        y_norm = (y_train - self.y_mean) / self.y_std

        self.model = SymbolicRegressor(
            population_size=self.population_size,
            generations=self.generations,
            tournament_size=15,
            stopping_criteria=0.001,
            p_crossover=0.7,
            p_subtree_mutation=0.1,
            p_hoist_mutation=0.05,
            p_point_mutation=0.1,
            max_samples=0.9,
            parsimony_coefficient=0.005,
            function_set=("add", "sub", "mul", "div", "sqrt", "log", "abs", "neg"),
            metric="mean absolute error",
            random_state=seed,
            n_jobs=1,
            verbose=0,
        )
        # Subsample for speed if large
        if len(X_train) > 1000:
            idx = np.linspace(0, len(X_train) - 1, 1000, dtype=int)
            X_tr, y_tr = X_train[idx], y_norm[idx]
        else:
            X_tr, y_tr = X_train, y_norm

        self.model.fit(X_tr, y_tr)
        return self

    def predict(self, X: np.ndarray) -> tuple[np.ndarray, None]:
        if self.model is None:
            return np.full(len(X), self.y_mean), None
        raw_preds = self.model.predict(X)
        nonfinite_mask = ~np.isfinite(raw_preds)
        if np.any(nonfinite_mask):
            self.has_nonfinite = True
            raw_preds = np.nan_to_num(raw_preds, nan=0.0, posinf=5.0, neginf=-5.0)

        preds = raw_preds * self.y_std + self.y_mean
        preds_capped = np.clip(preds, 0.0, 110.0)
        return preds_capped, None


# =====================================================================
# Trivial Reference Baselines (IMPACT Guidelines Requirement)
# =====================================================================

class MeanPredictorBaseline:
    """Trivial reference baseline: always predicts constant training-set mean SOH."""
    def __init__(self):
        self.mean_val = 0.0

    def fit(self, X_train: np.ndarray, y_train: np.ndarray, **kwargs):
        self.mean_val = float(np.mean(y_train))
        return self

    def predict(self, X: np.ndarray) -> tuple[np.ndarray, None]:
        return np.full(len(X), self.mean_val, dtype=np.float64), None


class MedianPredictorBaseline:
    """Trivial reference baseline: always predicts constant training-set median SOH."""
    def __init__(self):
        self.median_val = 0.0

    def fit(self, X_train: np.ndarray, y_train: np.ndarray, **kwargs):
        self.median_val = float(np.median(y_train))
        return self

    def predict(self, X: np.ndarray) -> tuple[np.ndarray, None]:
        return np.full(len(X), self.median_val, dtype=np.float64), None


# =====================================================================
# 6. BiLSTM Sequence Baseline (Temporal history windows of length W)
# =====================================================================

class _BiLSTMModule(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int = 32):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=1,
            batch_first=True,
            bidirectional=True,
        )
        self.head = nn.Sequential(
            nn.Linear(hidden_dim * 2, 16),
            nn.ReLU(),
            nn.Linear(16, 1),
        )

    def forward(self, x):
        # x: (batch, seq_len, input_dim)
        out, _ = self.lstm(x)
        last_out = out[:, -1, :]  # (batch, hidden_dim * 2)
        return self.head(last_out).squeeze(-1)


class BiLSTMSeq:
    """
    BiLSTM trained on causal sequential windows of past cycles (window_size=5).
    Guaranteed zero future leakage (only cycles t-4, ..., t are fed).
    """
    def __init__(self, window_size: int = 5, hidden_dim: int = 32, lr: float = 0.01, epochs: int = 60):
        self.window_size = window_size
        self.hidden_dim = hidden_dim
        self.lr = lr
        self.epochs = epochs
        self.model = None

    def _create_windows(self, X: np.ndarray, y: np.ndarray | None = None):
        # Creates (N, window_size, D) sequences using edge replication for early cycles
        n, d = X.shape
        w = self.window_size
        pad_x = np.repeat(X[:1], w - 1, axis=0)
        X_padded = np.vstack([pad_x, X])
        windows = np.zeros((n, w, d), dtype=np.float32)
        for i in range(n):
            windows[i] = X_padded[i : i + w]
        return windows

    def fit(self, X_train: np.ndarray, y_train: np.ndarray, X_val=None, y_val=None, seed: int = 42):
        set_seed(seed)
        d = X_train.shape[1]
        self.model = _BiLSTMModule(input_dim=d, hidden_dim=self.hidden_dim)

        X_win = self._create_windows(X_train)
        X_t = torch.tensor(X_win, dtype=torch.float32)
        y_t = torch.tensor(y_train, dtype=torch.float32)

        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)
        criterion = nn.MSELoss()
        loader = DataLoader(TensorDataset(X_t, y_t), batch_size=64, shuffle=True)

        self.model.train()
        for _ in range(self.epochs):
            for x_b, y_b in loader:
                optimizer.zero_grad()
                loss = criterion(self.model(x_b), y_b)
                loss.backward()
                optimizer.step()

        self.model.eval()
        return self

    def predict(self, X: np.ndarray) -> tuple[np.ndarray, None]:
        self.model.eval()
        X_win = self._create_windows(X)
        with torch.no_grad():
            preds = self.model(torch.tensor(X_win, dtype=torch.float32)).numpy()
        return preds, None


# Factory dictionary mapping model name to constructor
BASELINE_MODELS = {
    "Mean_Predictor": MeanPredictorBaseline,
    "Median_Predictor": MedianPredictorBaseline,
    "Linear_Regression": LinearRegressionBaseline,
    "Random_Forest": RandomForestBaseline,
    "SVR": SVRBaseline,
    "XGBoost": XGBoostBaseline,
    "Exact_GPR_RBF": ExactGPR_RBF,
    "Exact_GPR_Matern52": ExactGPR_Matern52,
    "Exact_GPR_ARD": ExactGPR_ARD,
    "Sparse_GPR_SVGP": SparseGPR_SVGP,
    "Deep_Kernel_GPR": DeepKernelGPR,
    "WideNN": WideNN,
    "DeepNN": DeepNN,
    "Wide_and_Deep_NN": WideAndDeepNN,
    "Standalone_SR": StandaloneSR,
    "BiLSTM": BiLSTMSeq,
}
