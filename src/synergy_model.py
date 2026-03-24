"""
Synergy model computation — revised methodology.

Why the original approach didn't work
--------------------------------------
In a linear MMM, contribution_i = coefficient_i × support_i  (exactly).
So Y = contrib1 + contrib2 = k1·T1 + k2·T2 with R²_base = 1.0, leaving
zero residual for any synergy term to explain.

Revised approach
----------------
Dependent variable  : total model contributions (sum across ALL variables
                      for the chosen model, by week).  T1 and T2 each
                      explain only their own slice of this total, so the
                      base-model R² is well below 1 and there is residual
                      variance that a synergy interaction can explain.

Synergy support     : Three formulations are tried; the one with the
                      greatest ΔR² (R²_full − R²_base) is selected.

  A  Normalised product : (T1/μ1) × (T2/μ2)           -- original spec
  B  Deviation product  : (T1−μ1) × (T2−μ2)           -- co-elevation
  C  Z-score product    : z1 × z2                       -- standardised

Constraints         : All coefficients ≥ 0 (NNLS), no intercept,
                      no seasonality.

Synergy exists      : bootstrap CI lower bound > 0  AND  ΔR² > threshold.
"""

from typing import Tuple

import numpy as np
import pandas as pd
from scipy.optimize import nnls
from scipy.stats import f as f_dist


# ---------------------------------------------------------------------------
# Synergy support formulations
# ---------------------------------------------------------------------------

def _safe_scale(arr: np.ndarray) -> np.ndarray:
    """Divide by RMS so the array has unit scale — prevents ill-conditioning."""
    rms = np.sqrt(np.mean(arr ** 2))
    return arr / rms if rms > 1e-12 else arr


def _synergy_supports(T1: np.ndarray, T2: np.ndarray) -> dict:
    """Return three candidate synergy support streams, each unit-scaled."""
    m1, m2 = T1.mean(), T2.mean()
    s1 = T1.std() + 1e-12
    s2 = T2.std() + 1e-12

    raw = {
        "Normalised product": (T1 / (m1 + 1e-12)) * (T2 / (m2 + 1e-12)),
        "Deviation product":  (T1 - m1) * (T2 - m2),
        "Z-score product":    ((T1 - m1) / s1) * ((T2 - m2) / s2),
    }
    return {name: _safe_scale(v) for name, v in raw.items()}


# ---------------------------------------------------------------------------
# NNLS helpers
# ---------------------------------------------------------------------------

def _fit(y: np.ndarray, X: np.ndarray) -> np.ndarray:
    n = X.shape[0] * X.shape[1] * 10  # generous iteration budget
    try:
        coeffs, _ = nnls(X, y, maxiter=n)
    except RuntimeError:
        # Ill-conditioned matrix — fall back to zeros
        coeffs = np.zeros(X.shape[1])
    return coeffs


def _r2(y: np.ndarray, y_hat: np.ndarray) -> float:
    ss_tot = float(np.sum((y - y.mean()) ** 2))
    if ss_tot == 0:
        return 0.0
    return 1.0 - float(np.sum((y - y_hat) ** 2)) / ss_tot


def _f_test(y: np.ndarray, X_base: np.ndarray, X_full: np.ndarray,
            b_base: np.ndarray, b_full: np.ndarray) -> Tuple[float, float]:
    """F-test for the synergy term (one extra predictor)."""
    n = len(y)
    rss_base = float(np.sum((y - X_base @ b_base) ** 2))
    rss_full = float(np.sum((y - X_full @ b_full) ** 2))
    df_base = n - X_base.shape[1]
    df_full = n - X_full.shape[1]
    if rss_full == 0 or df_full <= 0:
        return 0.0, 1.0
    f_stat = ((rss_base - rss_full) / 1) / (rss_full / df_full)
    p_val = float(1.0 - f_dist.cdf(f_stat, 1, df_full))
    return float(f_stat), p_val


# ---------------------------------------------------------------------------
# Bootstrap CIs
# ---------------------------------------------------------------------------

def _bootstrap_ci(
    y: np.ndarray,
    X: np.ndarray,
    ci_level: float,
    n_bootstrap: int,
    rng: np.random.Generator,
) -> Tuple[np.ndarray, np.ndarray]:
    n, p = X.shape
    boot = np.full((n_bootstrap, p), np.nan)
    for i in range(n_bootstrap):
        idx = rng.integers(0, n, size=n)
        try:
            boot[i] = _fit(y[idx], X[idx])
        except Exception:
            pass
    boot = boot[~np.isnan(boot).any(axis=1)]
    if len(boot) == 0:
        return np.zeros(p), np.zeros(p)
    alpha = (1.0 - ci_level) / 2.0
    return (
        np.percentile(boot, alpha * 100, axis=0),
        np.percentile(boot, (1.0 - alpha) * 100, axis=0),
    )


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def compute_synergy_model(
    total_y: pd.Series,          # total model contributions (all variables summed)
    ts1: pd.Series,              # transformed support for var1
    ts2: pd.Series,              # transformed support for var2
    ci_level: float = 0.95,
    n_bootstrap: int = 1000,
    seed: int = 42,
) -> dict:
    """
    Run the synergy model for one variable pair.

    Parameters
    ----------
    total_y    : weekly total contributions for the model (all variables)
    ts1, ts2   : transformed support series from WeeklyTransformSupport
    ci_level   : bootstrap CI level
    n_bootstrap: number of bootstrap iterations
    seed       : RNG seed

    Returns
    -------
    dict with keys:
        error, coefficients, ci_lower, ci_upper, r2_base, r2_full,
        delta_r2, f_stat, p_value, synergy_formulation,
        y, y_hat, support1, support2, synergy_support,
        index, residuals, n_obs, ci_level, is_significant
    """
    # Align on common dates
    common = (
        total_y.index
        .intersection(ts1.index)
        .intersection(ts2.index)
        .sort_values()
    )

    if len(common) < 8:
        return {"error": "Fewer than 8 overlapping observations."}

    Y  = total_y.loc[common].values.astype(float)
    T1 = ts1.loc[common].values.astype(float)
    T2 = ts2.loc[common].values.astype(float)

    # Remove rows with NaN
    valid = ~(np.isnan(Y) | np.isnan(T1) | np.isnan(T2))
    Y, T1, T2 = Y[valid], T1[valid], T2[valid]
    idx_valid = common[valid]

    if len(Y) < 8:
        return {"error": "Fewer than 8 non-null observations after alignment."}

    # Base model (no synergy)
    X_base = np.column_stack([T1, T2])
    b_base = _fit(Y, X_base)
    r2_base = _r2(Y, X_base @ b_base)

    # Try each synergy formulation — pick the one with highest ΔR²
    best = {"delta_r2": -np.inf}
    for name, syn in _synergy_supports(T1, T2).items():
        if np.isnan(syn).any() or np.isinf(syn).any():
            continue
        X_full = np.column_stack([T1, T2, syn])
        b_full = _fit(Y, X_full)
        r2_full = _r2(Y, X_full @ b_full)
        dr2 = r2_full - r2_base
        if dr2 > best["delta_r2"]:
            best = {
                "delta_r2": dr2,
                "name": name,
                "syn": syn,
                "b_full": b_full,
                "r2_full": r2_full,
                "X_full": X_full,
            }

    if best.get("delta_r2", -np.inf) == -np.inf:
        return {"error": "Could not compute any synergy formulation."}

    b_full  = best["b_full"]
    r2_full = best["r2_full"]
    syn     = best["syn"]
    X_full  = best["X_full"]
    dr2     = best["delta_r2"]
    fname   = best["name"]

    y_hat     = X_full @ b_full
    residuals = Y - y_hat
    f_stat, p_val = _f_test(Y, X_base, X_full, b_base, b_full)

    rng = np.random.default_rng(seed)
    ci_lower, ci_upper = _bootstrap_ci(Y, X_full, ci_level, n_bootstrap, rng)

    # Synergy is considered significant when:
    #   • synergy coefficient bootstrap CI lower > 0  (statistically positive)
    #   • ΔR² > 0.001  (practically meaningful improvement)
    syn_coeff     = float(b_full[2])
    syn_ci_lower  = float(ci_lower[2])
    is_significant = syn_ci_lower > 0 and dr2 > 0.001

    return {
        "error":               None,
        "coefficients":        b_full,          # [var1, var2, synergy]
        "ci_lower":            ci_lower,
        "ci_upper":            ci_upper,
        "r2_base":             r2_base,
        "r2_full":             r2_full,
        "delta_r2":            dr2,
        "f_stat":              f_stat,
        "p_value":             p_val,
        "synergy_formulation": fname,
        "y":                   Y,
        "y_hat":               y_hat,
        "support1":            T1,
        "support2":            T2,
        "synergy_support":     syn,
        "index":               idx_valid,
        "residuals":           residuals,
        "n_obs":               int(len(Y)),
        "ci_level":            ci_level,
        "is_significant":      is_significant,
    }
