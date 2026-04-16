"""
Gaussian Process Regression
============================
Non-parametric Bayesian regression for degradation curves with:
  - Kernel comparison (RBF, Matern-3/2, Matern-5/2, Rational Quadratic)
  - Leave-one-out cross-validation [Stone, 1974]
  - Uncertainty calibration (empirical coverage of 68%/95% CI)

Kernel selection rationale:
  - RBF: assumes infinitely smooth functions (baseline)
  - Matern-3/2: once-differentiable — suitable for degradation curves that
    are continuous but not necessarily smooth [Rasmussen & Williams, 2006]
  - Matern-5/2: twice-differentiable — intermediate smoothness
  - Rational Quadratic: mixture of RBF at multiple length scales

See REFERENCES.md for full citations.
"""
import numpy as np
from dataclasses import dataclass
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import (
    RBF, Matern, RationalQuadratic, ConstantKernel, WhiteKernel)
from sklearn.model_selection import LeaveOneOut
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from ..utils import get_logger

log = get_logger(__name__)


@dataclass
class GPResult:
    kernel_name: str
    kernel_str: str
    log_marginal_likelihood: float
    r_squared: float
    rmse: float
    loo_rmse: float
    loo_mae: float
    loo_r2: float
    coverage_68: float
    coverage_95: float
    dose_grid: np.ndarray = None
    pred_mean: np.ndarray = None
    pred_std: np.ndarray = None
    model: object = None


KERNEL_REGISTRY = {
    "rbf": lambda: ConstantKernel() * RBF() + WhiteKernel(),
    "matern32": lambda: ConstantKernel() * Matern(nu=1.5) + WhiteKernel(),
    "matern52": lambda: ConstantKernel() * Matern(nu=2.5) + WhiteKernel(),
    "rational_quadratic": lambda: ConstantKernel() * RationalQuadratic() + WhiteKernel(),
}


def _coverage(y_true, y_pred, y_std):
    if len(y_true) == 0 or np.all(y_std == 0):
        return 0.0, 0.0
    return (float(np.mean(np.abs(y_true - y_pred) <= y_std)),
            float(np.mean(np.abs(y_true - y_pred) <= 2 * y_std)))


def fit_gp(dose, response, kernel_name="matern32", normalize_x=True, n_restarts=5):
    import warnings
    warnings.filterwarnings("ignore", category=UserWarning)

    X = dose.reshape(-1, 1).copy()
    y = response.copy()
    x_scale = X.max() if normalize_x and X.max() > 0 else 1.0
    X_n = X / x_scale

    gp = GaussianProcessRegressor(
        kernel=KERNEL_REGISTRY[kernel_name](),
        n_restarts_optimizer=n_restarts, alpha=1e-6, normalize_y=True)
    gp.fit(X_n, y)

    y_pred, y_std = gp.predict(X_n, return_std=True)
    r2 = r2_score(y, y_pred)
    rmse = np.sqrt(mean_squared_error(y, y_pred))

    # LOO-CV
    loo = LeaveOneOut()
    loo_p, loo_s = np.zeros_like(y), np.zeros_like(y)
    for tr, te in loo.split(X_n):
        g = GaussianProcessRegressor(
            kernel=KERNEL_REGISTRY[kernel_name](),
            n_restarts_optimizer=max(n_restarts-2, 1), alpha=1e-6, normalize_y=True)
        g.fit(X_n[tr], y[tr])
        p, s = g.predict(X_n[te], return_std=True)
        loo_p[te], loo_s[te] = p, s

    loo_rmse = np.sqrt(mean_squared_error(y, loo_p))
    loo_mae = mean_absolute_error(y, loo_p)
    loo_r2 = r2_score(y, loo_p)
    cov68, cov95 = _coverage(y, loo_p, loo_s)

    dose_grid = np.linspace(0, dose.max() * 1.15, 300)
    pm, ps = gp.predict(dose_grid.reshape(-1, 1) / x_scale, return_std=True)

    log.info(f"  {kernel_name}: R2={r2:.4f} LOO-RMSE={loo_rmse:.4f} "
             f"Cov95={cov95:.0%} LML={gp.log_marginal_likelihood_value_:.1f}")

    return GPResult(
        kernel_name=kernel_name, kernel_str=str(gp.kernel_),
        log_marginal_likelihood=gp.log_marginal_likelihood_value_,
        r_squared=r2, rmse=rmse, loo_rmse=loo_rmse, loo_mae=loo_mae,
        loo_r2=loo_r2, coverage_68=cov68, coverage_95=cov95,
        dose_grid=dose_grid, pred_mean=pm, pred_std=ps, model=gp)


def fit_all_gp(dose, response, kernel_names=None, normalize_x=True, n_restarts=5):
    if kernel_names is None:
        kernel_names = list(KERNEL_REGISTRY.keys())
    results = {}
    for kn in kernel_names:
        if kn in KERNEL_REGISTRY:
            results[kn] = fit_gp(dose, response, kn, normalize_x, n_restarts)
    ranked = sorted(results.values(), key=lambda r: r.loo_rmse)
    log.info("GP kernel ranking (LOO-RMSE):")
    for i, r in enumerate(ranked):
        log.info(f"  {i+1}. {r.kernel_name}: LOO-RMSE={r.loo_rmse:.4f} Cov95={r.coverage_95:.0%}")
    return results
