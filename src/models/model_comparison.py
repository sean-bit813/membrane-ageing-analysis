"""
Model Comparison Framework
===========================
Unified comparison table across parametric and GP models.
Uses AICc for parametric [Hurvich & Tsai, 1989] and LOO-RMSE for GP.
"""
import numpy as np
import pandas as pd
from ..utils import get_logger

log = get_logger(__name__)


def build_comparison_table(parametric_results, gp_results):
    rows = []
    for name, r in parametric_results.items():
        rows.append({"model": name, "type": "parametric", "n_params": r.n_params,
                      "R2": r.r_squared, "RMSE": r.rmse,
                      "AICc": r.aicc, "BIC": r.bic,
                      "LOO_RMSE": np.nan, "LOO_R2": np.nan, "Coverage_95": np.nan})
    for name, r in gp_results.items():
        rows.append({"model": f"GP_{name}", "type": "gp", "n_params": np.nan,
                      "R2": r.r_squared, "RMSE": r.rmse,
                      "AICc": np.nan, "BIC": np.nan,
                      "LOO_RMSE": r.loo_rmse, "LOO_R2": r.loo_r2,
                      "Coverage_95": r.coverage_95})
    df = pd.DataFrame(rows)
    log.info("\n=== Model Comparison ===")
    log.info("\n" + df.to_string(index=False, float_format="%.4f"))
    return df


def select_best_models(table):
    par = table[table["type"] == "parametric"]
    gp = table[table["type"] == "gp"]
    result = {}
    if len(par) > 0:
        bp = par.sort_values("AICc").iloc[0]
        result["best_parametric"] = bp["model"]
        log.info(f"Best parametric: {bp['model']} (AICc={bp['AICc']:.1f})")
    if len(gp) > 0:
        bg = gp.sort_values("LOO_RMSE").iloc[0]
        result["best_gp"] = bg["model"]
        log.info(f"Best GP: {bg['model']} (LOO-RMSE={bg['LOO_RMSE']:.4f}, "
                 f"Cov95={bg['Coverage_95']:.0%})")
    return result
