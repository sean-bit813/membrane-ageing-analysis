"""
Parametric Degradation Models
==============================
Nonlinear regression for membrane aging curves with information-theoretic
model selection.

Models:
  - Power law:   R/R0 = a * D^b + c           [Lei et al. 2018]
  - Exponential: R/R0 = a * (1 - exp(-bD)) + c
  - Weibull:     R/R0 = a * (1 - exp(-(D/l)^k)) + c  [Zhuang et al. 2024]

Selection:
  - AICc for small samples [Hurvich & Tsai, 1989]
  - BIC  [Schwarz, 1978]
  - Bootstrap 95% CI on parameters and predictions

See REFERENCES.md for full citations.
"""
import numpy as np
from dataclasses import dataclass
from scipy.optimize import least_squares, differential_evolution
from ..utils import get_logger

log = get_logger(__name__)


@dataclass
class FitResult:
    model_name: str
    params: np.ndarray
    param_names: list
    residuals: np.ndarray
    r_squared: float
    rmse: float
    aic: float
    aicc: float
    bic: float
    n_params: int
    n_data: int
    param_ci_low: np.ndarray = None
    param_ci_high: np.ndarray = None
    dose_grid: np.ndarray = None
    pred_grid: np.ndarray = None
    pred_ci_low: np.ndarray = None
    pred_ci_high: np.ndarray = None


# -- Model functions --
def power_law(dose, a, b, c):
    return a * np.power(np.maximum(dose, 1.0), b) + c

def exponential_saturation(dose, a, b, c):
    return a * (1.0 - np.exp(-b * dose)) + c

def weibull(dose, a, lam, k, c):
    return a * (1.0 - np.exp(-np.power(dose / max(lam, 1e-10), k))) + c


MODEL_REGISTRY = {
    "power_law": {
        "func": power_law, "param_names": ["a", "b", "c"],
        "p0": [1e-3, 0.5, 1.0],
        "bounds": ([0, 0, 0.5], [1e3, 3.0, 2.0]),
    },
    "exponential": {
        "func": exponential_saturation, "param_names": ["a", "b", "c"],
        "p0": [2.0, 1e-6, 1.0],
        "bounds": ([0, 0, 0], [20, 1e-2, 2.0]),
    },
    "weibull": {
        "func": weibull, "param_names": ["a", "lam", "k", "c"],
        "p0": [3.0, 5e5, 1.0, 1.0],
        "bounds": ([0, 1e3, 0.1, 0], [20, 1e8, 5.0, 2.0]),
    },
}


def _compute_info_criteria(n, k, rss):
    """AIC, AICc [Hurvich & Tsai 1989], BIC [Schwarz 1978]."""
    if n <= 0 or rss <= 0:
        return np.inf, np.inf, np.inf
    ll_term = n * np.log(rss / n)
    aic = ll_term + 2 * k
    # AICc: corrected for small samples — critical when n/k < ~40
    if n - k - 1 > 0:
        aicc = aic + (2 * k * (k + 1)) / (n - k - 1)
    else:
        aicc = np.inf
    bic = ll_term + k * np.log(n)
    return aic, aicc, bic


def fit_parametric(dose, response, model_name, n_bootstrap=200, seed=42):
    spec = MODEL_REGISTRY[model_name]
    func, p0 = spec["func"], spec["p0"]
    lo, hi = spec["bounds"]
    n, k = len(dose), len(p0)

    def res_fn(params):
        return func(dose, *params) - response

    try:
        result = least_squares(res_fn, p0, bounds=(lo, hi), method="trf",
                               max_nfev=10000, ftol=1e-12, xtol=1e-12)
        params, residuals = result.x, result.fun
    except Exception:
        def obj(p): return np.sum((func(dose, *p) - response) ** 2)
        result = differential_evolution(obj, list(zip(lo, hi)), seed=seed, maxiter=1000)
        params = result.x
        residuals = func(dose, *params) - response

    ss_res = np.sum(residuals ** 2)
    ss_tot = np.sum((response - np.mean(response)) ** 2)
    r2 = 1 - ss_res / ss_tot if ss_tot > 0 else 0
    rmse = np.sqrt(ss_res / n)
    aic, aicc, bic = _compute_info_criteria(n, k, ss_res)

    log.info(f"  {model_name}: R2={r2:.4f} RMSE={rmse:.4f} "
             f"AICc={aicc:.1f} BIC={bic:.1f}")

    # Bootstrap CI
    rng = np.random.default_rng(seed)
    boot_params = []
    for _ in range(n_bootstrap):
        idx = rng.choice(n, size=n, replace=True)
        try:
            r = least_squares(lambda p: func(dose[idx], *p) - response[idx],
                              params, bounds=(lo, hi), max_nfev=5000)
            boot_params.append(r.x)
        except Exception:
            continue

    dose_grid = np.linspace(0, dose.max() * 1.1, 200)
    pred_grid = func(dose_grid, *params)
    ci_lo = ci_hi = params
    pred_ci_lo = pred_ci_hi = pred_grid

    if len(boot_params) > 10:
        bp = np.array(boot_params)
        ci_lo, ci_hi = np.percentile(bp, 2.5, axis=0), np.percentile(bp, 97.5, axis=0)
        bpreds = np.array([func(dose_grid, *b) for b in bp])
        pred_ci_lo = np.percentile(bpreds, 2.5, axis=0)
        pred_ci_hi = np.percentile(bpreds, 97.5, axis=0)

    return FitResult(
        model_name=model_name, params=params, param_names=spec["param_names"],
        residuals=residuals, r_squared=r2, rmse=rmse,
        aic=aic, aicc=aicc, bic=bic, n_params=k, n_data=n,
        param_ci_low=ci_lo, param_ci_high=ci_hi,
        dose_grid=dose_grid, pred_grid=pred_grid,
        pred_ci_low=pred_ci_lo, pred_ci_high=pred_ci_hi)


def fit_all_parametric(dose, response, model_names=None, n_bootstrap=200):
    if model_names is None:
        model_names = list(MODEL_REGISTRY.keys())
    results = {}
    for name in model_names:
        if name in MODEL_REGISTRY:
            results[name] = fit_parametric(dose, response, name, n_bootstrap)
    ranked = sorted(results.values(), key=lambda r: r.aicc)
    log.info("Parametric ranking (AICc):")
    for i, r in enumerate(ranked):
        delta = r.aicc - ranked[0].aicc
        log.info(f"  {i+1}. {r.model_name}: AICc={r.aicc:.1f} (delta={delta:.1f})")
    return results
