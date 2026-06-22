"""Parametric model fitting functions for growth curves.

This module provides functions to fit parametric growth models:

- Mechanistic models (ODE-based): mech_logistic, mech_gompertz, mech_richards,
  mech_baranyi
- Phenomenological models (ln-space): phenom_logistic, phenom_gompertz,
  phenom_gompertz_modified, phenom_richards

All models operate in linear OD space (not log-transformed).

"""

import numpy as np
from scipy.optimize import curve_fit

from .inference import validate_data
from .models import (
    get_all_models,
    mech_baranyi_model,
    mech_gompertz_model,
    mech_logistic_model,
    mech_richards_model,
    phenom_gompertz_model,
    phenom_gompertz_modified_model_ln,
    phenom_logistic_model,
    phenom_richards_model,
)

# -----------------------------------------------------------------------------
# Helper Functions
# -----------------------------------------------------------------------------


def _estimate_initial_params(t, N):
    """
    Estimate common initial parameters for all growth models.

    Parameters:
        t: Time array
        N: OD values (assumed baseline-corrected)

    Returns:
        Tuple of (K_init, dN) where:
            K_init: Initial carrying capacity (max OD)
            dN: First derivative (gradient) of OD with respect to t

    """
    K_init = np.max(N)
    dN = np.gradient(N, t)
    return K_init, dN


def _estimate_lag_time(t, dN, threshold_frac=0.1):
    """
    Estimate lag t from growth rate threshold.

    Parameters:
        t: Time array
        dy: First derivative of OD
        threshold_frac: Fraction of max derivative to use as threshold

    Returns:
        Estimated lag t (t when growth rate exceeds threshold)

    """
    threshold = threshold_frac * np.max(dN)
    lag_idx = np.where(dN > threshold)[0]
    return t[lag_idx[0]] if len(lag_idx) > 0 else t[0]


def _fit_model_generic(
    t, N, model_func, param_names, p0_func, bounds_func, model_type, log_space=False
):
    """
    Generic wrapper for fitting parametric growth models.

    This function encapsulates the common pattern used across all parametric
    model fitting functions, reducing code duplication.

    Parameters:
        t: Time array
        N: OD values
        model_func: Model function to fit (e.g., mech_logistic_model)
        param_names: List of parameter names in order
        p0_func: Function that takes (K_init, t, dN) and returns p0
        bounds_func: Function that takes (K_init, t) and returns bounds
        model_type: String identifier for the model

    Returns:
        Dict with 'params' and 'model_type', or None if fitting fails

    """
    t, N = validate_data(t, N)
    if t is None:
        return None

    # Estimate common initial parameters
    K_init, dN = _estimate_initial_params(t, N)

    # Generate initial guess and bounds
    p0 = p0_func(K_init, t, dN)
    bounds = bounds_func(K_init, t)

    # Fit the model
    if log_space:
        N_pos = np.maximum(N, 1e-8)

        def log_model(tt, *p):
            return np.log(np.maximum(model_func(tt, *p), 1e-8))

        params, _ = curve_fit(
            log_model, t, np.log(N_pos), p0=p0, bounds=bounds, maxfev=20000
        )
    else:
        params, _ = curve_fit(model_func, t, N, p0=p0, bounds=bounds, maxfev=20000)

    # Return structured result
    return {
        "params": dict(zip(param_names, params)),
        "model_type": model_type,
    }


# -----------------------------------------------------------------------------
# Mechanistic Model Fitting Functions (ODE-based)
# -----------------------------------------------------------------------------


def fit_mech_logistic(t, N):
    """
    Fit mechanistic logistic model (ODE) to growth N.

    ODE: dN/dt = μ * (1 - N/K) * N
    OD(t) = N(t)

    Assumes input data is baseline-corrected (no additive offset).

    Parameters:
        t: Time array (hours)
        N: OD values

    Returns:
        Dict with 'params' and 'model_type', or None if fitting fails.

    """
    return _fit_model_generic(
        t,
        N,
        model_func=mech_logistic_model,
        param_names=["mu", "K", "N0"],
        p0_func=lambda K, t, dy: [0.5, K, 0.001],
        bounds_func=lambda K, t: (
            [0.0001, 0.001, 1e-6],
            [10, np.inf, 10],
        ),
        model_type="mech_logistic",
        log_space=True,
    )


def fit_mech_gompertz(t, N):
    """
    Fit mechanistic Gompertz model (ODE) to growth data.

    ODE: dN/dt = μ * log(K/N) * N
    OD(t) = N(t)

    Assumes input data is baseline-corrected (no additive offset).

    Parameters:
        t: Time array (hours)
        N: OD values

    Returns:
        Dict with 'params' and 'model_type', or None if fitting fails.

    Note:
        The mechanistic Gompertz model can be numerically challenging to fit
        due to the logarithmic term in the ODE. If fitting fails or produces
        poor results, consider using mech_logistic, mech_richards, or
        phenom_gompertz instead.

    """
    return _fit_model_generic(
        t,
        N,
        model_func=mech_gompertz_model,
        param_names=["mu", "K", "N0"],
        p0_func=lambda K, t, dy: [0.05, K, 0.01],
        bounds_func=lambda K, t: (
            [0.0001, 0.01, 1e-4],
            [2, np.inf, 1],
        ),
        model_type="mech_gompertz",
        log_space=True,
    )


def fit_mech_richards(t, N):
    """
    Fit mechanistic Richards model (ODE) to growth N.

    ODE: dN/dt = μ * (1 - (N/K)^β) * N
    OD(t) = N(t)

    Assumes input data is baseline-corrected (no additive offset).

    Parameters:
        t: Time array (hours)
        N: OD values

    Returns:
        Dict with 'params' and 'model_type', or None if fitting fails.
    """
    # Multistart over β: a single seed often lands in the wrong basin for true
    # β at the extremes of the (0.1, 10) range. Three log-spaced seeds at ~3× cost.
    best, best_ssr = None, np.inf
    for beta_init in (0.1, 1.0, 10.0):
        try:
            result = _fit_model_generic(
                t,
                N,
                model_func=mech_richards_model,
                param_names=["mu", "K", "N0", "beta"],
                p0_func=lambda K, t, dy, b=beta_init: [0.5, K, 0.001, b],
                bounds_func=lambda K, t: (
                    [0.0001, 0.001, 1e-6, 0.01],
                    [10, np.inf, 10, 100],
                ),
                model_type="mech_richards",
                log_space=True,
            )
        except Exception:
            continue
        if result is None:
            continue
        p = result["params"]
        pred = mech_richards_model(t, p["mu"], p["K"], p["N0"], p["beta"])
        log_resid = np.log(np.maximum(pred, 1e-8)) - np.log(np.maximum(N, 1e-8))
        ssr = float(np.sum(log_resid**2))
        if ssr < best_ssr:
            best_ssr, best = ssr, result
    return best


def fit_mech_baranyi(t, N):
    """
    Fit mechanistic Baranyi-Roberts model (ODE) to growth N.

    ODE: dN/dt = μ * A(t) * (1 - N/K) * N
    where A(t) = exp(μ*t) / (exp(h0) - 1 + exp(μ*t))
    OD(t) = N(t)

    Assumes input data is baseline-corrected (no additive offset).

    Parameters:
        t: Time array (hours)
        N: OD values

    Returns:
        Dict with 'params' and 'model_type', or None if fitting fails.
    """

    def p0_baranyi(K, t, dy):
        mu_init = 0.5
        h0_init = 1.0
        N0_init = 0.001
        return [mu_init, K, N0_init, h0_init]

    def bounds_baranyi(K, t):
        return (
            [0.0001, 0.001, 1e-6, 0],
            [10, np.inf, 10, t.max() * 10],
        )

    return _fit_model_generic(
        t,
        N,
        model_func=mech_baranyi_model,
        param_names=["mu", "K", "N0", "h0"],
        p0_func=p0_baranyi,
        bounds_func=bounds_baranyi,
        model_type="mech_baranyi",
        log_space=True,
    )


# -----------------------------------------------------------------------------
# Phenomenological Model Fitting Functions (ln-space)
# -----------------------------------------------------------------------------


def fit_phenom_logistic(t, N):
    """
    Fit phenomenological logistic model to ln(OD/OD0) data.

    ln(Nt/N0) = A / (1 + exp(4 * μ_max * (λ - t) / A + 2))

    Parameters:
        t: Time array (hours)
        N: OD values

    Returns:
        Dict with 'params' and 'model_type', or None if fitting fails.
    """
    t, N = validate_data(t, N)
    if t is None:
        return None

    # Estimate initial parameters
    N0 = float(np.min(N))
    N_max = float(np.max(N))
    A_init = np.log(N_max / N0)
    mu_max_init = 0.5
    lam_init = _estimate_lag_time(t, np.gradient(N, t))

    p0 = [A_init, mu_max_init, lam_init, N0]
    bounds = ([0.01, 0.0001, 0, N0 * 0.1], [20, 10, t.max(), N0 * 5])

    # Fit the model
    params, _ = curve_fit(
        phenom_logistic_model, t, N, p0=p0, bounds=bounds, maxfev=20000
    )

    return {
        "params": dict(zip(["A", "mu_max", "lam", "N0"], params)),
        "model_type": "phenom_logistic",
    }


def fit_phenom_gompertz(t, N):
    """
    Fit phenomenological Gompertz model to ln(OD/OD0) N.

    ln(Nt/N0) = A * exp(-exp(μ_max * e * (λ - t) / A + 1))

    Parameters:
        t: Time array (hours)
        N: OD values

    Returns:
        Dict with 'params' and 'model_type', or None if fitting fails.
    """
    t, N = validate_data(t, N)
    if t is None:
        return None

    # Estimate initial parameters
    N0 = float(np.min(N))
    N_max = float(np.max(N))
    A_init = np.log(N_max / N0)
    mu_max_init = 0.5
    lam_init = _estimate_lag_time(t, np.gradient(N, t))

    p0 = [A_init, mu_max_init, lam_init, N0]
    bounds = ([0.01, 0.0001, 0, N0 * 0.1], [20, 10, t.max(), N0 * 5])

    # Fit the model
    params, _ = curve_fit(
        phenom_gompertz_model, t, N, p0=p0, bounds=bounds, maxfev=20000
    )

    return {
        "params": dict(zip(["A", "mu_max", "lam", "N0"], params)),
        "model_type": "phenom_gompertz",
    }


def fit_phenom_gompertz_modified(t, N):
    """
    Fit phenomenological modified Gompertz model with decay term.

    ln(Nt/N0) = A * exp(-exp(μ_max * e * (λ - t) / A + 1)) + A * exp(α * (t - t_shift))

    Parameters:
        t: Time array (hours)
        N: OD values

    Returns:
        Dict with 'params' and 'model_type', or None if fitting fails.
    """
    t, N = validate_data(t, N)
    if t is None:
        return None

    # Estimate initial parameters
    N0 = float(np.min(N))
    A_init = max(float(np.log(N[len(t) // 3] / N0)), 0.1)
    mu_max_init = 0.5
    lam_init = t.max() * 0.1
    alpha_init = 0.05
    t_shift_init = t.max() * 0.8

    p0 = [A_init, mu_max_init, lam_init, alpha_init, t_shift_init]
    bounds = (
        [0.01, 0.0001, 0, -1, 0],
        [20, 10, t.max(), 1, t.max()],
    )

    # Fit the model
    params, _ = curve_fit(
        phenom_gompertz_modified_model_ln, t, N, p0=p0, bounds=bounds, maxfev=20000
    )

    return {
        "params": dict(zip(["A", "mu_max", "lam", "alpha", "t_shift"], params)),
        "model_type": "phenom_gompertz_modified",
    }


def fit_phenom_richards(t, N):
    """
    Fit phenomenological Richards model to ln(OD/OD0) N.

    ln(Nt/N0)= A * (1 + ν * exp(1 + ν + μ_max * (1 + ν)^(1 + 1/ν) * (λ - t) / A))^(-1/ν)

    Parameters:
        t: Time array (hours)
        N: OD values

    Returns:
        Dict with 'params' and 'model_type', or None if fitting fails.
    """
    t, N = validate_data(t, N)
    if t is None:
        return None

    # Estimate initial parameters
    N0 = float(np.min(N))
    N_max = float(np.max(N))
    A_init = np.log(N_max / N0)
    mu_max_init = 0.5
    lam_init = _estimate_lag_time(t, np.gradient(N, t))
    nu_init = 1.0

    p0 = [A_init, mu_max_init, lam_init, nu_init, N0]
    bounds = ([0.01, 0.0001, 0, 0.01, N0 * 0.1], [20, 10, t.max(), 100, N0 * 5])

    # Fit the model
    params, _ = curve_fit(
        phenom_richards_model, t, N, p0=p0, bounds=bounds, maxfev=20000
    )

    return {
        "params": dict(zip(["A", "mu_max", "lam", "nu", "N0"], params)),
        "model_type": "phenom_richards",
    }


# -----------------------------------------------------------------------------
# Main Fitting Dispatcher
# -----------------------------------------------------------------------------


def fit_parametric(t, N, method="mech_logistic", **kwargs):
    """
    Fit a growth model to N.

    Parameters:
        t: Time array (hours)
        N: OD values
        method: Model type string. Options:
            Mechanistic (ODE):  "mech_logistic", "mech_gompertz",
                                "mech_richards", "mech_baranyi"
            Phenomenological (ln-space): "phenom_logistic", "phenom_gompertz",
                "phenom_gompertz_modified", "phenom_richards"

    Returns:
        Fit result dict or None if fitting fails.
    """
    fit_func = globals().get(f"fit_{method}")

    if fit_func is None:
        raise ValueError(
            f"Unknown method '{method}'. Must be one of {list(get_all_models())}."
        )

    result = fit_func(t, N)
    if result is not None:
        time_valid, _ = validate_data(t, N)
        if time_valid is None:
            return None
        result["params"]["fit_t_min"] = float(np.min(time_valid))
        result["params"]["fit_t_max"] = float(np.max(time_valid))
    return result
