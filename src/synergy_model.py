"""
Synergy model computation.

For each variable pair (var1, var2):
  - Dependent variable  : contrib1 + contrib2  (sum of contributions from Weekly)
  - Independent variables:
      X1 = transformed support for var1  (from WeeklyTransformSupport)
      X2 = transformed support for var2  (from WeeklyTransformSupport)
      X3 = synergy support stream        = (X1/mean(X1)) * (X2/mean(X2))
  - Constraints : all coefficients >= 0, no intercept, no seasonality
  - Estimation  : Non-Negative Least Squares (NNLS)
  - CIs         : Percentile bootstrap at user-specified level
"""

from typing import Tuple

import numpy as np
import pandas as pd
from scipy.optimize import nnls


# ---------------------------------------------------------------------------
# Synergy support
# ---------------------------------------------------------------------------

def create_synergy_support(s1: pd.Series, s2: pd.Series) -> pd.Series:
    """
    Centre each series on 1 by dividing by its mean, then multiply element-wise.
    Result is the synergy support stream.
    """
    m1 = s1.mean()
    m2 = s2.mean()
    if m1 == 0 or m2 == 0:
        return pd.Series(np.zeros(len(s1)), index=s1.index)
    return (s1 / m1) * (s2 / m2)


# ---------------------------------------------------------------------------
# Constrained estimation
# ---------------------------------------------------------------------------

def _fit(y: np.ndarray, X: np.ndarray) -> np.ndarray:
    """Non-negative least squares (no intercept)."""
    coeffs, _ = nnls(X, y)
    return coeffs


def _bootstrap_ci(
    y: np.ndarray,
    X: np.ndarray,
    ci_level: float,
    n_bootstrap: int,
    rng: np.random.Generator,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Percentile bootstrap confidence intervals for NNLS coefficients.
    Returns (lower, upper) arrays of shape (n_features,).
    """
    n, p = X.shape
    boot = np.full((n_bootstrap, p), np.nan)

    for i in range(n_bootstrap):
        idx = rng.integers(0, n, size=n)
        try:
            boot[i] = _fit(y[idx], X[idx])
        except Exception:
            pass

    # Drop failed iterations
    boot = boot[~np.isnan(boot).any(axis=1)]
    if len(boot) == 0:
        return np.zeros(p), np.zeros(p)

    alpha = (1.0 - ci_level) / 2.0
    lower = np.percentile(boot, alpha * 100, axis=0)
    upper = np.percentile(boot, (1.0 - alpha) * 100, axis=0)
    return lower, upper


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def compute_synergy_model(
    contrib1: pd.Series,
    contrib2: pd.Series,
    ts1: pd.Series,
    ts2: pd.Series,
    ci_level: float = 0.95,
    n_bootstrap: int = 1000,
    seed: int = 42,
) -> dict:
    """
    Run the full synergy model for one variable pair.

    Parameters
    ----------
    contrib1, contrib2 : contributions from Weekly (DatetimeIndex)
    ts1, ts2           : transformed support from WeeklyTransformSupport (DatetimeIndex)
    ci_level           : e.g. 0.95 for 95 % CIs
    n_bootstrap        : number of bootstrap iterations
    seed               : random seed for reproducibility

    Returns
    -------
    dict with keys:
        coefficients, ci_lower, ci_upper, r_squared,
        y, y_hat, support1, support2, synergy_support,
        index, residuals, n_obs, ci_level, error
    """
    # Align on common date index
    common = (
        contrib1.index
        .intersection(contrib2.index)
        .intersection(ts1.index)
        .intersection(ts2.index)
        .sort_values()
    )

    if len(common) < 4:
        return {"error": "Fewer than 4 overlapping observations — cannot fit model."}

    y = (contrib1.loc[common] + contrib2.loc[common]).values.astype(float)
    s1 = ts1.loc[common]
    s2 = ts2.loc[common]
    syn = create_synergy_support(s1, s2)

    X = np.column_stack([s1.values, s2.values, syn.values]).astype(float)

    # Remove rows that have NaN in y or X
    valid = ~(np.isnan(y) | np.isnan(X).any(axis=1))
    y, X = y[valid], X[valid]
    idx_valid = common[valid]

    if len(y) < 4:
        return {"error": "Fewer than 4 non-null observations after alignment."}

    coeffs = _fit(y, X)
    y_hat = X @ coeffs
    residuals = y - y_hat

    ss_res = float(np.sum(residuals ** 2))
    ss_tot = float(np.sum((y - np.mean(y)) ** 2))
    r_squared = 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0

    rng = np.random.default_rng(seed)
    ci_lower, ci_upper = _bootstrap_ci(y, X, ci_level, n_bootstrap, rng)

    return {
        "error":           None,
        "coefficients":    coeffs,          # [coeff_var1, coeff_var2, coeff_synergy]
        "ci_lower":        ci_lower,
        "ci_upper":        ci_upper,
        "r_squared":       r_squared,
        "y":               y,               # actual (sum of contributions)
        "y_hat":           y_hat,           # fitted values
        "support1":        X[:, 0],
        "support2":        X[:, 1],
        "synergy_support": X[:, 2],
        "index":           idx_valid,       # DatetimeIndex
        "residuals":       residuals,
        "n_obs":           int(len(y)),
        "ci_level":        ci_level,
    }
