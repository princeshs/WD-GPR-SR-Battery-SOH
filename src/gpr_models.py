"""
src/gpr_models.py
=================
Implementation of the Proposed Wide-and-Deep Variational Gaussian Process Regressor (WD-SVGP)
and its ablation variants in GPyTorch.

Ablations:
  - Variant A: Wide only (Linear + ARD-RBF kernel)
  - Variant B: Deep only (MLP phi_theta -> Matérn-5/2 kernel)
  - Variant C: Full Wide-and-Deep (Wide + Deep branches jointly optimized)

Prediction uses observation-noise-inclusive predictive variance:
  sigma_pred^2 = Var[f(x*)] + sigma_obs^2
  95% Confidence Interval: [mean - 1.96 * sigma_pred, mean + 1.96 * sigma_pred]
"""

import os
import copy
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from sklearn.cluster import KMeans
import gpytorch
from gpytorch.models import ApproximateGP
from gpytorch.variational import CholeskyVariationalDistribution, VariationalStrategy
from gpytorch.mlls import VariationalELBO
from src.evaluation import set_seed


class DeepFeatureExtractor(nn.Module):
    """
    Parameterised MLP mapping input features X in R^D to latent manifold z in R^p (p in [2, 8]).
    Includes LayerNorm and Dropout to prevent overfitting on small cell samples.
    """
    def __init__(self, input_dim: int, hidden_dim: int = 32, latent_dim: int = 4, dropout: float = 0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.LayerNorm(hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, latent_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class WideAndDeepGP(ApproximateGP):
    """
    GPyTorch ApproximateGP with Wide-and-Deep kernel:
    k_WD(x, x') = sigma_w^2 * [k_linear(x, x') + k_RBF-ARD(x, x')] + sigma_d^2 * k_Matern(phi_theta(x), phi_theta(x'))
    """
    def __init__(
        self,
        inducing_points: torch.Tensor,
        input_dim: int,
        latent_dim: int = 4,
        hidden_dim: int = 32,
        variant: str = "C",
    ):
        self.variant = variant.upper()
        if self.variant not in ("A", "B", "C"):
            raise ValueError(f"Unknown variant '{variant}'. Expected 'A', 'B', or 'C'.")

        variational_distribution = CholeskyVariationalDistribution(inducing_points.size(0))
        variational_strategy = VariationalStrategy(
            self, inducing_points, variational_distribution, learn_inducing_locations=True
        )
        super().__init__(variational_strategy)

        self.mean_module = gpytorch.means.ConstantMean()

        # Wide branch: ScaleKernel(LinearKernel() + RBFKernel(ard_num_dims=input_dim))
        if self.variant in ("A", "C"):
            self.wide_kernel = gpytorch.kernels.ScaleKernel(
                gpytorch.kernels.LinearKernel() + gpytorch.kernels.RBFKernel(ard_num_dims=input_dim)
            )
        else:
            self.wide_kernel = None

        # Deep branch: phi_theta -> ScaleKernel(MaternKernel(nu=2.5))
        if self.variant in ("B", "C"):
            self.deep_net = DeepFeatureExtractor(input_dim, hidden_dim=hidden_dim, latent_dim=latent_dim)
            self.deep_kernel = gpytorch.kernels.ScaleKernel(
                gpytorch.kernels.MaternKernel(nu=2.5)
            )
        else:
            self.deep_net = None
            self.deep_kernel = None

    def forward(self, x: torch.Tensor) -> gpytorch.distributions.MultivariateNormal:
        mean_x = self.mean_module(x)

        if self.variant == "A":
            covar_x = self.wide_kernel(x)
        elif self.variant == "B":
            z = self.deep_net(x)
            covar_x = self.deep_kernel(z)
        else:  # Variant C: Wide + Deep
            c_wide = self.wide_kernel(x)
            z = self.deep_net(x)
            c_deep = self.deep_kernel(z)
            covar_x = c_wide + c_deep

        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)


class WideAndDeepSVGPRegressor:
    """
    High-level scikit-learn compatible estimator wrapping WideAndDeepGP.
    Supports Variants A (Wide-only), B (Deep-only), and C (Full Wide-and-Deep).
    Automatically normalizes targets during training and scales back during prediction.
    Enforces training-only validation early stopping and neural-only weight decay.
    """
    def __init__(
        self,
        variant: str = "C",
        num_inducing: int = 64,
        latent_dim: int = 4,
        hidden_dim: int = 32,
        lr: float = 0.015,
        weight_decay: float = 1e-4,
        epochs: int = 80,
        batch_size: int = 128,
        patience: int = 15,
        val_fraction: float = 0.15,
        random_state: int = 42,
        training_regime: str = "refit_blocked",
    ):
        self.variant = variant.upper()
        self.num_inducing = num_inducing
        self.latent_dim = latent_dim
        self.hidden_dim = hidden_dim
        self.lr = lr
        self.weight_decay = weight_decay
        self.epochs = epochs
        self.batch_size = batch_size
        self.patience = patience
        self.val_fraction = val_fraction
        self.random_state = random_state
        self.training_regime = training_regime

        self.model = None
        self.likelihood = None
        self.y_mean = 0.0
        self.y_scale = 1.0
        self.stopping_epoch = epochs
        self.y_train_min = 0.0
        self.y_train_max = 100.0
        self.y_fit_min = 0.0
        self.y_fit_max = 100.0
        self.y_val_min = np.nan
        self.y_val_max = np.nan
        self.loss_history = []
        self.val_loss_history = []

    def _create_model_and_optimizer(self, X_data: np.ndarray, seed: int):
        n_samples, input_dim = X_data.shape
        n_ind = min(self.num_inducing, n_samples)
        kmeans = KMeans(n_clusters=n_ind, random_state=seed, n_init=5)
        kmeans.fit(X_data)
        inducing_points = torch.tensor(kmeans.cluster_centers_, dtype=torch.float32)

        model = WideAndDeepGP(
            inducing_points=inducing_points,
            input_dim=input_dim,
            latent_dim=self.latent_dim,
            hidden_dim=self.hidden_dim,
            variant=self.variant,
        )
        likelihood = gpytorch.likelihoods.GaussianLikelihood()

        decay_params = []
        no_decay_params = []
        if hasattr(model, "deep_net") and model.deep_net is not None:
            decay_params.extend(list(model.deep_net.parameters()))
            deep_param_ids = set(id(p) for p in model.deep_net.parameters())
        else:
            deep_param_ids = set()

        for p in model.parameters():
            if id(p) not in deep_param_ids:
                no_decay_params.append(p)
        no_decay_params.extend(list(likelihood.parameters()))

        param_groups = [{"params": no_decay_params, "weight_decay": 0.0}]
        if decay_params and self.weight_decay > 0.0:
            param_groups.append({"params": decay_params, "weight_decay": self.weight_decay})

        optimizer = torch.optim.Adam(param_groups, lr=self.lr)
        mll = VariationalELBO(likelihood, model, num_data=n_samples)
        return model, likelihood, optimizer, mll

    def fit(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray | None = None,
        y_val: np.ndarray | None = None,
        cell_ids: np.ndarray | list | None = None,
        seed: int = 42,
        log_file: str | None = None,
        training_regime: str | None = None,
    ):
        set_seed(seed)
        n_samples, input_dim = X_train.shape

        self.y_mean = float(np.mean(y_train))
        self.y_scale = float(np.std(y_train) + 1e-6)
        self.y_train_min = float(np.min(y_train))
        self.y_train_max = float(np.max(y_train))

        regime = (training_regime or self.training_regime).lower()

        # -------------------------------------------------------------
        # Regime C: Fixed epochs on 100% of training cycles (no early stopping)
        # -------------------------------------------------------------
        if regime == "fixed_80" or (self.val_fraction <= 0.0 and X_val is None):
            self.stopping_epoch = self.epochs
            self.y_fit_min = self.y_train_min
            self.y_fit_max = self.y_train_max
            self.y_val_min = np.nan
            self.y_val_max = np.nan

            self.model, self.likelihood, optimizer, mll = self._create_model_and_optimizer(X_train, seed)
            y_norm = (y_train - self.y_mean) / self.y_scale
            loader = DataLoader(
                TensorDataset(torch.tensor(X_train, dtype=torch.float32), torch.tensor(y_norm, dtype=torch.float32)),
                batch_size=min(self.batch_size, n_samples),
                shuffle=True,
            )
            self.model.train()
            self.likelihood.train()
            self.loss_history = []
            for _ in range(self.epochs):
                b_losses = []
                for x_b, y_b in loader:
                    optimizer.zero_grad()
                    loss = -mll(self.model(x_b), y_b)
                    loss.backward()
                    optimizer.step()
                    b_losses.append(loss.item())
                self.loss_history.append(float(np.mean(b_losses)))
            self.val_loss_history = list(self.loss_history)
            self.model.eval()
            self.likelihood.eval()
            return self

        # -------------------------------------------------------------
        # Construct Validation Split: Blocked vs. Random
        # -------------------------------------------------------------
        if X_val is not None and y_val is not None:
            X_fit, y_fit = X_train, y_train
            X_val_act, y_val_act = X_val, y_val
        elif regime == "random_no_refit":
            from sklearn.model_selection import train_test_split
            idx_all = np.arange(n_samples)
            idx_tr, idx_val = train_test_split(idx_all, test_size=self.val_fraction, random_state=seed)
            X_fit, y_fit = X_train[idx_tr], y_train[idx_tr]
            X_val_act, y_val_act = X_train[idx_val], y_train[idx_val]
        else:
            # Blocked split: last val_fraction of each training cell
            if cell_ids is not None:
                c_arr = np.asarray(cell_ids)
                unique_cells = np.unique(c_arr)
                idx_tr_list, idx_val_list = [], []
                for c in unique_cells:
                    c_indices = np.where(c_arr == c)[0]
                    n_c = len(c_indices)
                    n_val_c = max(1, int(self.val_fraction * n_c))
                    idx_tr_list.extend(c_indices[:-n_val_c])
                    idx_val_list.extend(c_indices[-n_val_c:])
                idx_tr = np.array(idx_tr_list, dtype=int)
                idx_val = np.array(idx_val_list, dtype=int)
            else:
                val_size = max(1, int(self.val_fraction * n_samples))
                idx_tr = np.arange(0, n_samples - val_size, dtype=int)
                idx_val = np.arange(n_samples - val_size, n_samples, dtype=int)

            X_fit, y_fit = X_train[idx_tr], y_train[idx_tr]
            X_val_act, y_val_act = X_train[idx_val], y_train[idx_val]

        self.y_fit_min = float(np.min(y_fit))
        self.y_fit_max = float(np.max(y_fit))
        self.y_val_min = float(np.min(y_val_act))
        self.y_val_max = float(np.max(y_val_act))

        # -------------------------------------------------------------
        # Step 1: Validation loop on X_fit to determine best stopping epoch E*
        # -------------------------------------------------------------
        set_seed(seed)
        model_val, lik_val, opt_val, mll_val = self._create_model_and_optimizer(X_fit, seed)
        y_fit_norm = (y_fit - self.y_mean) / self.y_scale
        y_val_norm = (y_val_act - self.y_mean) / self.y_scale

        loader_val = DataLoader(
            TensorDataset(torch.tensor(X_fit, dtype=torch.float32), torch.tensor(y_fit_norm, dtype=torch.float32)),
            batch_size=min(self.batch_size, len(X_fit)),
            shuffle=True,
        )
        X_val_t = torch.tensor(X_val_act, dtype=torch.float32)
        y_val_t = torch.tensor(y_val_norm, dtype=torch.float32)

        best_val_loss = float("inf")
        best_epoch = self.epochs
        best_state = None
        patience_counter = 0
        self.loss_history = []
        self.val_loss_history = []

        model_val.train()
        lik_val.train()

        for epoch in range(1, self.epochs + 1):
            b_losses = []
            for x_b, y_b in loader_val:
                opt_val.zero_grad()
                loss = -mll_val(model_val(x_b), y_b)
                loss.backward()
                opt_val.step()
                b_losses.append(loss.item())

            self.loss_history.append(float(np.mean(b_losses)))

            model_val.eval()
            lik_val.eval()
            with torch.no_grad():
                val_out = model_val(X_val_t)
                val_loss = float(-mll_val(val_out, y_val_t).item())
            self.val_loss_history.append(val_loss)
            model_val.train()
            lik_val.train()

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_epoch = epoch
                best_state = (copy.deepcopy(model_val.state_dict()), copy.deepcopy(lik_val.state_dict()))
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= self.patience:
                    break

        self.stopping_epoch = best_epoch

        # -------------------------------------------------------------
        # Regime A & B: Do not refit; restore weights fit only on 85%
        # -------------------------------------------------------------
        if regime in ("random_no_refit", "blocked_no_refit"):
            self.model = model_val
            self.likelihood = lik_val
            if best_state is not None:
                self.model.load_state_dict(best_state[0])
                self.likelihood.load_state_dict(best_state[1])
            self.model.eval()
            self.likelihood.eval()
            return self

        # -------------------------------------------------------------
        # Regime D (Phase 3C Default): REFIT on 100% of training cycles
        # for exactly best_epoch epochs
        # -------------------------------------------------------------
        set_seed(seed)
        self.model, self.likelihood, opt_refit, mll_refit = self._create_model_and_optimizer(X_train, seed)
        y_train_norm = (y_train - self.y_mean) / self.y_scale
        loader_full = DataLoader(
            TensorDataset(torch.tensor(X_train, dtype=torch.float32), torch.tensor(y_train_norm, dtype=torch.float32)),
            batch_size=min(self.batch_size, n_samples),
            shuffle=True,
        )

        self.model.train()
        self.likelihood.train()
        refit_loss_history = []
        for _ in range(best_epoch):
            b_losses = []
            for x_b, y_b in loader_full:
                opt_refit.zero_grad()
                loss = -mll_refit(self.model(x_b), y_b)
                loss.backward()
                opt_refit.step()
                b_losses.append(loss.item())
            refit_loss_history.append(float(np.mean(b_losses)))

        self.loss_history = refit_loss_history
        self.model.eval()
        self.likelihood.eval()

        if log_file:
            os.makedirs(os.path.dirname(log_file), exist_ok=True)
            with open(log_file, "w") as f:
                f.write("epoch,train_loss,val_loss\n")
                for ep, (t_l, v_l) in enumerate(zip(self.loss_history, self.val_loss_history), 1):
                    f.write(f"{ep},{t_l:.6f},{v_l:.6f}\n")

        return self

    def predict(self, X: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """
        Predict mean and observation-noise-inclusive standard deviation.
        sigma_pred^2 = Var[f(x*)] + sigma_obs^2
        Scales back from standardized space to physical SOH percentage units.
        """
        self.model.eval()
        self.likelihood.eval()
        X_t = torch.tensor(X, dtype=torch.float32)

        with torch.no_grad(), gpytorch.settings.fast_pred_var():
            f_dist = self.model(X_t)
            y_dist = self.likelihood(f_dist)
            mean_norm = y_dist.mean.cpu().numpy()
            var_norm = y_dist.variance.cpu().numpy()

        # Rescale mean and variance back to original units
        mean = mean_norm * self.y_scale + self.y_mean
        var = var_norm * (self.y_scale ** 2)
        std = np.sqrt(np.maximum(var, 1e-6))
        return mean, std

    def predict_with_interval(self, X: np.ndarray, coverage: float = 0.95) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Predict mean, std, lower bound, and upper bound for specified coverage.
        Default 95% interval: [mean - 1.96*std, mean + 1.96*std].
        """
        from scipy.stats import norm
        z = norm.ppf(1.0 - (1.0 - coverage) / 2.0)
        mean, std = self.predict(X)
        lower = mean - z * std
        upper = mean + z * std
        return mean, std, lower, upper

    def save_checkpoint(self, path: str):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        torch.save({
            "model_state": self.model.state_dict(),
            "likelihood_state": self.likelihood.state_dict(),
            "variant": self.variant,
            "num_inducing": self.num_inducing,
            "latent_dim": self.latent_dim,
            "hidden_dim": self.hidden_dim,
            "y_mean": self.y_mean,
            "y_scale": self.y_scale,
        }, path)

    def load_checkpoint(self, path: str, input_dim: int = 5):
        ckpt = torch.load(path, map_location="cpu")
        self.variant = ckpt["variant"]
        self.num_inducing = ckpt["num_inducing"]
        self.latent_dim = ckpt["latent_dim"]
        self.hidden_dim = ckpt["hidden_dim"]
        self.y_mean = ckpt.get("y_mean", 0.0)
        self.y_scale = ckpt.get("y_scale", 1.0)

        dummy_inducing = torch.zeros(self.num_inducing, input_dim)
        self.model = WideAndDeepGP(
            inducing_points=dummy_inducing,
            input_dim=input_dim,
            latent_dim=self.latent_dim,
            hidden_dim=self.hidden_dim,
            variant=self.variant,
        )
        self.likelihood = gpytorch.likelihoods.GaussianLikelihood()
        self.model.load_state_dict(ckpt["model_state"])
        self.likelihood.load_state_dict(ckpt["likelihood_state"])
        self.model.eval()
        self.likelihood.eval()
        return self


def train_gp_model(
    X_train: np.ndarray,
    y_train: np.ndarray,
    variant: str = "C",
    num_inducing: int = 64,
    latent_dim: int = 4,
    hidden_dim: int = 32,
    lr: float = 0.015,
    weight_decay: float = 1e-4,
    epochs: int = 80,
    batch_size: int = 128,
    patience: int = 15,
    val_fraction: float = 0.15,
    cell_ids: np.ndarray | list | None = None,
    seed: int = 42,
    log_file: str | None = None,
    training_regime: str = "refit_blocked",
) -> WideAndDeepSVGPRegressor:
    """
    Single canonical GP training function shared across Configs A, B, C,
    the base GP inside D and E, and all inner out-of-fold GPs.
    Guarantees identical hyperparameters, early stopping rule, training-only validation,
    and seed reproducibility.
    """
    set_seed(seed)
    model = WideAndDeepSVGPRegressor(
        variant=variant,
        num_inducing=num_inducing,
        latent_dim=latent_dim,
        hidden_dim=hidden_dim,
        lr=lr,
        weight_decay=weight_decay,
        epochs=epochs,
        batch_size=batch_size,
        patience=patience,
        val_fraction=val_fraction,
        random_state=seed,
        training_regime=training_regime,
    )
    model.fit(X_train, y_train, cell_ids=cell_ids, seed=seed, log_file=log_file, training_regime=training_regime)
    return model

