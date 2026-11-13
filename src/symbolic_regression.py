"""
src/symbolic_regression.py
==========================
Uncertainty-Weighted Symbolic Regression for Battery SOH Residual Discovery.
Implements inverse-variance sample weighting, parsimonious genetic programming,
SymPy expression extraction with exact gplearn protected operator parity,
ANSI C code generation, and validation-guided SR selection (including g=0).
"""

import re
import numpy as np
import sympy as sp
from gplearn.genetic import SymbolicRegressor
from gplearn.functions import _Function


def _program_to_full_precision_str(prog, feature_names=None) -> str:
    """
    Serializes a gplearn _Program to LISP string preserving full 64-bit float precision
    instead of gplearn's default '%.3f' string truncation.
    """
    if prog is None or not hasattr(prog, "program"):
        return "0"
    terminals = [0]
    output = ""
    for i, node in enumerate(prog.program):
        if isinstance(node, _Function):
            terminals.append(node.arity)
            output += node.name + "("
        else:
            if isinstance(node, int):
                output += f"X{node}"
            else:
                output += repr(float(node))
            terminals[-1] -= 1
            while terminals[-1] == 0:
                terminals.pop()
                terminals[-1] -= 1
                output += ")"
            if i != len(prog.program) - 1:
                output += ", "
    return output


def round_constants_to_sig_figs(expr: sp.Expr, sig: int = 3) -> sp.Expr:
    """
    Rounds all floating-point constants in a SymPy expression to `sig` significant figures.
    (Phase 3C Task 7 requirement).
    """
    def round_sig(val, s=3):
        try:
            f = float(val)
            if f == 0.0 or np.isnan(f) or np.isinf(f):
                return val
            return round(f, s - int(np.floor(np.log10(abs(f)))) - 1)
        except Exception:
            return val

    try:
        return expr.xreplace({n: sp.Float(round_sig(n, sig)) for n in expr.atoms(sp.Number)})
    except Exception:
        return expr


def compute_uncertainty_weights(
    variances: np.ndarray,
    epsilon: float = 1e-4,
    clip_quantiles: tuple[float, float] = (0.01, 0.99),
) -> np.ndarray:
    """
    Computes normalized sample weights inversely proportional to GP predictive variance:
      w_raw = 1 / (sigma_GP^2 + epsilon)
      w_clip = clip(w_raw, q_0.01, q_0.99)
      w = w_clip / mean(w_clip)  (normalized to mean 1)
    """
    var_clamped = np.maximum(variances, 0.0)
    w_raw = 1.0 / (var_clamped + epsilon)

    q_low = float(np.quantile(w_raw, clip_quantiles[0]))
    q_high = float(np.quantile(w_raw, clip_quantiles[1]))
    w_clip = np.clip(w_raw, q_low, q_high)

    mean_w = np.mean(w_clip)
    if mean_w > 0:
        w_norm = w_clip / mean_w
    else:
        w_norm = np.ones_like(w_clip)

    return w_norm.astype(np.float64)


class SymbolicResidualRegressor:
    """
    Parsimonious genetic programming regressor fitted on out-of-fold GP residuals.
    Wraps gplearn.genetic.SymbolicRegressor with safe prediction clipping,
    exact protected-operator SymPy parsing, and C code generation.
    """
    def __init__(
        self,
        population_size: int = 500,
        generations: int = 20,
        tournament_size: int = 20,
        parsimony_coefficient: float = 0.005,
        max_samples: float = 0.9,
        max_size: int = 15,
        function_set: tuple[str, ...] = ("add", "sub", "mul", "div"),
        random_state: int = 42,
        n_jobs: int = 1,
    ):
        self.population_size = population_size
        self.generations = generations
        self.tournament_size = tournament_size
        self.parsimony_coefficient = parsimony_coefficient
        self.max_samples = max_samples
        self.max_size = max_size
        self.function_set = function_set
        self.random_state = random_state
        self.n_jobs = n_jobs

        self.model = None
        self.feature_names = None
        self.raw_program_str = "0"
        self.sympy_expr = sp.Integer(0)
        self.display_expr = sp.Integer(0)
        self.display_equation_str = "0"

    def fit(
        self,
        X: np.ndarray,
        residuals: np.ndarray,
        sample_weight: np.ndarray | None = None,
        feature_names: list[str] | None = None,
    ):
        """
        Fit symbolic regressor on residuals r_i with optional uncertainty weights w_i.
        Inputs X are assumed standardized (zero mean, unit variance).
        Enforces compactness constraint: maxsize <= 15 nodes.
        """
        self.feature_names = feature_names or [f"X{i}" for i in range(X.shape[1])]

        self.model = SymbolicRegressor(
            population_size=self.population_size,
            generations=self.generations,
            tournament_size=self.tournament_size,
            init_depth=(2, 4),
            stopping_criteria=0.001,
            p_crossover=0.7,
            p_subtree_mutation=0.1,
            p_hoist_mutation=0.05,
            p_point_mutation=0.1,
            max_samples=self.max_samples,
            verbose=0,
            parsimony_coefficient=self.parsimony_coefficient,
            function_set=self.function_set,
            random_state=self.random_state,
            n_jobs=self.n_jobs,
        )

        self.model.fit(X, residuals, sample_weight=sample_weight)

        best_prog = None
        if hasattr(self.model, "_program") and self.model._program is not None:
            best_prog = self.model._program
            # Task 7: Enforce maxsize <= 15 nodes
            if getattr(best_prog, "length_", 999) > self.max_size and hasattr(self.model, "_programs"):
                valid_progs = []
                for gen in self.model._programs:
                    for p in gen:
                        if p is not None and getattr(p, "length_", 999) <= self.max_size:
                            valid_progs.append(p)
                if valid_progs:
                    best_prog = min(valid_progs, key=lambda p: p.fitness_)
                    self.model._program = best_prog

        if best_prog is not None:
            self.raw_program_str = _program_to_full_precision_str(best_prog)
            self.sympy_expr = self._parse_to_sympy(self.raw_program_str, self.feature_names)
            self.display_expr = round_constants_to_sig_figs(self.sympy_expr, sig=3)
            self.display_equation_str = str(self.display_expr)
        else:
            self.raw_program_str = "0"
            self.sympy_expr = sp.Integer(0)
            self.display_expr = sp.Integer(0)
            self.display_equation_str = "0"

        return self

    def get_pareto_front(self, X: np.ndarray, y: np.ndarray) -> list[dict]:
        """
        Extracts non-dominated Pareto front of (complexity vs RMSE) across all evaluated programs.
        """
        if self.model is None or not hasattr(self.model, "_programs"):
            return [{"complexity": 1, "rmse": float(np.sqrt(np.mean(y ** 2))), "equation": "0"}]

        best_by_len = {}
        for gen in self.model._programs:
            for p in gen:
                if p is not None:
                    l = getattr(p, "length_", None)
                    if l is not None:
                        try:
                            pred = p.execute(X)
                            pred = np.nan_to_num(pred, nan=0.0, posinf=30.0, neginf=-30.0)
                            rmse = float(np.sqrt(np.mean((y - pred) ** 2)))
                            if l not in best_by_len or rmse < best_by_len[l]["rmse"]:
                                best_by_len[l] = {"rmse": rmse, "prog": p}
                        except Exception:
                            continue

        sorted_lens = sorted(best_by_len.keys())
        pareto = []
        min_rmse = float("inf")
        for l in sorted_lens:
            item = best_by_len[l]
            if item["rmse"] < min_rmse:
                min_rmse = item["rmse"]
                p_str = _program_to_full_precision_str(item["prog"])
                try:
                    p_sym = self._parse_to_sympy(p_str, self.feature_names)
                    disp_p = str(round_constants_to_sig_figs(p_sym, 3))
                except Exception:
                    disp_p = p_str
                pareto.append({
                    "complexity": int(l),
                    "rmse": round(min_rmse, 4),
                    "equation": disp_p,
                })
        return pareto

        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict residual correction g(X).
        Clips predictions to [-30.0, 30.0] to safeguard against asymptotic poles on unseen test inputs.
        """
        if self.model is None or self.raw_program_str == "0":
            return np.zeros(len(X))
        try:
            preds = self.model.predict(X)
            preds = np.nan_to_num(preds, nan=0.0, posinf=30.0, neginf=-30.0)
            return np.clip(preds, -30.0, 30.0)
        except Exception:
            return np.zeros(len(X))

    def _parse_to_sympy(self, prog_str: str, feat_names: list[str]) -> sp.Expr:
        """
        Parses gplearn LISP-like program string into a SymPy expression reproducing
        gplearn's protected division (|x2| <= 0.001 -> 1.0) and protected log (|x1| <= 0.001 -> 0.0).
        Raises ValueError if parsing fails; never falls back to an atomic Symbol.
        """
        if prog_str == "0" or not prog_str:
            return sp.Integer(0)

        local_dict = {
            "add": lambda a, b: a + b,
            "sub": lambda a, b: a - b,
            "mul": lambda a, b: a * b,
            "div": lambda a, b: sp.Piecewise((a / b, sp.Abs(b) > 0.001), (1.0, True)),
            "neg": lambda a: -a,
            "abs": sp.Abs,
            "sqrt": lambda a: sp.sqrt(sp.Abs(a)),
            "log": lambda a: sp.Piecewise((sp.log(sp.Abs(a)), sp.Abs(a) > 0.001), (0.0, True)),
        }

        # Map X0, X1, ... to feature symbols
        for i in range(len(feat_names)):
            clean_name = re.sub(r"\W|^(?=\d)", "_", feat_names[i])
            local_dict[f"X{i}"] = sp.Symbol(clean_name)

        try:
            expr = sp.sympify(prog_str, locals=local_dict)
            return expr
        except Exception as e:
            raise ValueError(f"Failed to parse gplearn program string to SymPy: {prog_str}") from e

    def lambdify(self):
        """
        Returns a numpy-callable function evaluating the exact SymPy expression
        with gplearn protected operator semantics and output clipping.
        """
        if self.model is None or self.raw_program_str == "0":
            return lambda *args: np.zeros(len(args[0]) if len(args) > 0 else 1)

        clean_symbols = [sp.Symbol(re.sub(r"\W|^(?=\d)", "_", f)) for f in self.feature_names]
        f_raw = sp.lambdify(clean_symbols, self.sympy_expr, modules=["numpy"])

        def safe_eval(*args):
            with np.errstate(all="ignore"):
                out = f_raw(*args)
            out = np.asarray(out, dtype=np.float64)
            if out.ndim == 0:
                out = np.full(len(args[0]), float(out))
            out = np.nan_to_num(out, nan=0.0, posinf=30.0, neginf=-30.0)
            return np.clip(out, -30.0, 30.0)

        return safe_eval

    def to_c_code(self, num_features: int = 5) -> str:
        """
        Generates ANSI C code that exactly evaluates the discovered equation
        with gplearn protected operators and output clamping.
        """
        if self.raw_program_str == "0":
            body = "    double val = 0.0;"
        else:
            c_expr = self.raw_program_str
            c_expr = re.sub(r"\bdiv\(", "p_div(", c_expr)
            c_expr = re.sub(r"\blog\(", "p_log(", c_expr)
            c_expr = re.sub(r"\bsqrt\(", "p_sqrt(", c_expr)
            c_expr = re.sub(r"\babs\(", "fabs(", c_expr)
            c_expr = re.sub(r"\bneg\(", "p_neg(", c_expr)
            c_expr = re.sub(r"\badd\(", "p_add(", c_expr)
            c_expr = re.sub(r"\bsub\(", "p_sub(", c_expr)
            c_expr = re.sub(r"\bmul\(", "p_mul(", c_expr)
            for i in range(num_features):
                c_expr = re.sub(rf"\bX{i}\b", f"x[{i}]", c_expr)
            body = f"    double val = {c_expr};"

        c_src = f"""/* Auto-generated C residual evaluator with exact gplearn protected operators */
#include <math.h>

static inline double p_add(double a, double b) {{ return a + b; }}
static inline double p_sub(double a, double b) {{ return a - b; }}
static inline double p_mul(double a, double b) {{ return a * b; }}
static inline double p_neg(double a) {{ return -a; }}
static inline double p_div(double a, double b) {{ return (fabs(b) > 0.001) ? (a / b) : 1.0; }}
static inline double p_log(double a) {{ return (fabs(a) > 0.001) ? log(fabs(a)) : 0.0; }}
static inline double p_sqrt(double a) {{ return sqrt(fabs(a)); }}

double evaluate_residual(const double* x) {{
{body}
    if (val < -30.0) val = -30.0;
    if (val > 30.0) val = 30.0;
    return val;
}}

void evaluate_residual_batch(const double* X, double* out, int n) {{
    for (int i = 0; i < n; i++) {{
        out[i] = evaluate_residual(&X[i * {num_features}]);
    }}
}}
"""
        return c_src

    def get_node_and_operator_count(self) -> tuple[int, int]:
        """Returns (node_count, operator_count) of the best discovered equation."""
        if self.model is None or not hasattr(self.model, "_program"):
            return 1, 0
        prog = self.model._program
        node_count = len(prog.program)
        op_count = sum(1 for node in prog.program if hasattr(node, "arity"))
        return node_count, op_count


def recalibrate_variance(
    residuals_oof: np.ndarray,
    variances_oof: np.ndarray,
) -> float:
    """
    Fits an empirical variance scaling factor s on out-of-fold residuals:
      s^2 = (1 / N) * sum( r_i^2 / sigma_i^2 )
    Returns scalar s (scaling multiplier for predictive std).
    """
    ratio = (residuals_oof ** 2) / np.maximum(variances_oof, 1e-6)
    s2 = float(np.mean(np.clip(ratio, 0.05, 20.0)))
    return float(np.sqrt(s2))


def select_sr_correction(
    X_oof: np.ndarray,
    residuals_oof: np.ndarray,
    variances_oof: np.ndarray,
    feature_names: list[str],
    cell_ids_oof: np.ndarray | None = None,
    candidate_parsimonies: tuple[float, ...] = (0.001, 0.005, 0.01),
    population_size: int = 500,
    generations: int = 20,
    use_weights: bool = True,
    seed: int = 42,
    n_jobs: int = 2,
) -> tuple[SymbolicResidualRegressor, str, bool, dict[str, float]]:
    """
    Nested SR selection (P3 requirement):
    With 3 training cells, fit each candidate (g = 0 and candidate parsimonies)
    on the out-of-fold residuals of 2 cells and score it on the third cell's residuals,
    then average over the 3 rotations. Never score a candidate on residuals it was fit on.

    Supports both uncertainty-weighted (use_weights=True) and unweighted (use_weights=False).
    If a candidate parsimony strictly improves cross-validated RMSE over g = 0
    by at least 1e-4, fits final SR model on all OOF residuals with that parsimony.
    Otherwise, selects candidate g = 0 (no correction).
    """
    if cell_ids_oof is not None:
        unique_cells = np.unique(cell_ids_oof)
        folds = []
        for c in unique_cells:
            val_idx = np.where(cell_ids_oof == c)[0]
            tr_idx = np.where(cell_ids_oof != c)[0]
            if len(val_idx) > 0 and len(tr_idx) > 0:
                folds.append((tr_idx, val_idx))
    else:
        # Fallback to 3-fold split if cell_ids not provided
        n = len(residuals_oof)
        idx = np.arange(n)
        folds = []
        n_splits = min(3, n)
        chunk = n // n_splits
        for i in range(n_splits):
            val_idx = idx[i * chunk : (i + 1) * chunk if i < n_splits - 1 else n]
            tr_idx = np.setdiff1d(idx, val_idx)
            folds.append((tr_idx, val_idx))

    # 1. Candidate g = 0: score on each rotation's validation fold and average
    g0_fold_rmses = []
    for tr_idx, val_idx in folds:
        r_val = residuals_oof[val_idx]
        g0_fold_rmses.append(float(np.sqrt(np.mean(r_val ** 2))))
    cv_base_rmse = float(np.mean(g0_fold_rmses))

    # 2. Evaluate candidate parsimonies via nested rotations
    best_p = None
    best_cv_rmse = cv_base_rmse

    for p in candidate_parsimonies:
        p_fold_rmses = []
        for tr_idx, val_idx in folds:
            X_tr_f = X_oof[tr_idx]
            r_tr_f = residuals_oof[tr_idx]
            w_tr_f = compute_uncertainty_weights(variances_oof[tr_idx]) if use_weights else None

            X_val_f = X_oof[val_idx]
            r_val_f = residuals_oof[val_idx]

            sr_cand = SymbolicResidualRegressor(
                population_size=population_size,
                generations=generations,
                tournament_size=20,
                parsimony_coefficient=p,
                function_set=("add", "sub", "mul", "div"),
                random_state=seed,
                n_jobs=n_jobs,
            )
            sr_cand.fit(X_tr_f, r_tr_f, sample_weight=w_tr_f, feature_names=feature_names)
            g_val_pred = sr_cand.predict(X_val_f)
            fold_rmse = float(np.sqrt(np.mean((r_val_f - g_val_pred) ** 2)))
            p_fold_rmses.append(fold_rmse)

        mean_p_rmse = float(np.mean(p_fold_rmses))
        if mean_p_rmse < best_cv_rmse - 1e-4:
            best_cv_rmse = mean_p_rmse
            best_p = p

    metrics = {
        "cv_base_rmse": cv_base_rmse,
        "cv_best_rmse": best_cv_rmse,
        "selected_parsimony": best_p if best_p is not None else 0.0,
        "n_rotations": len(folds),
        "use_weights": use_weights,
    }

    if best_p is not None:
        # Fit final SR on all training cell OOF residuals using selected parsimony
        full_weights = compute_uncertainty_weights(variances_oof) if use_weights else None
        final_sr = SymbolicResidualRegressor(
            population_size=population_size,
            generations=generations,
            tournament_size=20,
            parsimony_coefficient=best_p,
            function_set=("add", "sub", "mul", "div"),
            random_state=seed,
            n_jobs=n_jobs,
        )
        final_sr.fit(X_oof, residuals_oof, sample_weight=full_weights, feature_names=feature_names)
        eq_str = str(final_sr.display_expr)
        return final_sr, eq_str, True, metrics
    else:
        # Candidate g = 0 wins: no correction selected
        zero_sr = SymbolicResidualRegressor(population_size=10, generations=1, random_state=seed)
        zero_sr.feature_names = feature_names
        zero_sr.raw_program_str = "0"
        zero_sr.sympy_expr = sp.Integer(0)
        zero_sr.display_expr = sp.Integer(0)
        zero_sr.display_equation_str = "0 (no correction)"
        return zero_sr, "0 (no correction)", False, metrics
