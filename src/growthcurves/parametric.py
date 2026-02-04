"""Parametric model fitting functions for growth curves.

This module provides functions to fit parametric growth models:
- Mechanistic models (ODE-based):       mech_logistic,
                                        mech_gompertz,
                                        mech_richards,
                                        mech_baranyi
- Phenomenological models (ln-space):
                                        phenom_logistic,
                                        phenom_gompertz,
                                        phenom_gompertz_modified,
                                        phenom_richards

All models operate in linear OD space (not log-transformed).
"""

import numpy as np
from scipy.optimize import curve_fit

from .models import (
    mech_baranyi_model,
    mech_gompertz_model,
    mech_logistic_model,
    mech_richards_model,
    phenom_gompertz_model,
    phenom_gompertz_modified_model,
    phenom_logistic_model,
    phenom_richards_model,
)
from .utils import validate_data

# -----------------------------------------------------------------------------
# Helper Functions
# -----------------------------------------------------------------------------


def _estimate_initial_params(t, y):
    """
    Estimate common initial parameters for all growth models.

    Parameters:
        t: Time array
        y: OD values

    Returns:
        Tuple of (K_init, y0_init, dy) where:
            K_init: Initial carrying capacity (max OD)
            y0_init: Initial baseline OD (min OD)
            dy: First derivative (gradient) of OD with respect to time
    """
    K_init = np.max(y)
    y0_init = np.min(y)
    dy = np.gradient(y, t)
    return K_init, y0_init, dy


def _estimate_inflection_time(t, dy):
    """
    Estimate time at inflection point (maximum growth rate).

    Parameters:
        t: Time array
        dy: First derivative of OD

    Returns:
        Time at maximum derivative (inflection point)
    """
    return t[np.argmax(dy)]


def _estimate_lag_time(t, dy, threshold_frac=0.1):
    """
    Estimate lag time from growth rate threshold.

    Parameters:
        t: Time array
        dy: First derivative of OD
        threshold_frac: Fraction of max derivative to use as threshold

    Returns:
        Estimated lag time (time when growth rate exceeds threshold)
    """
    threshold = threshold_frac * np.max(dy)
    lag_idx = np.where(dy > threshold)[0]
    return t[lag_idx[0]] if len(lag_idx) > 0 else t[0]


def _fit_model_generic(t, y, model_func, param_names, p0_func, bounds_func, model_type):
    """
    Generic wrapper for fitting parametric growth models.

    This function encapsulates the common pattern used across all parametric
    model fitting functions, reducing code duplication.

    Parameters:
        t: Time array
        y: OD values
        model_func: Model function to fit (e.g., mech_logistic_model)
        param_names: List of parameter names in order
        p0_func: Function that takes (K_init, y0_init, t, dy) and returns p0
        bounds_func: Function that takes (K_init, y0_init, t) and returns bounds
        model_type: String identifier for the model

    Returns:
        Dict with 'params' and 'model_type', or None if fitting fails
    """
    t, y = validate_data(t, y)
    if t is None:
        return None

    # Estimate common initial parameters
    K_init, y0_init, dy = _estimate_initial_params(t, y)

    # Generate initial guess and bounds
    p0 = p0_func(K_init, y0_init, t, dy)
    bounds = bounds_func(K_init, y0_init, t)

    # Fit the model
    params, _ = curve_fit(model_func, t, y, p0=p0, bounds=bounds, maxfev=20000)

    # Return structured result
    return {
        "params": dict(zip(param_names, params)),
        "model_type": model_type,
    }


# -----------------------------------------------------------------------------
# Mechanistic Model Fitting Functions (ODE-based)
# -----------------------------------------------------------------------------


def fit_mech_logistic(t, y):
    """
    Fit mechanistic logistic model (ODE) to growth data.

    ODE: dN/dt = μ * (1 - N/K) * N
    OD(t) = y0 + N(t)

    Parameters:
        t: Time array (hours)
        y: OD values

    Returns:
        Dict with 'params' and 'model_type', or None if fitting fails.
    """
    return _fit_model_generic(
        t,
        y,
        model_func=mech_logistic_model,
        param_names=["mu", "K", "N0", "y0"],
        p0_func=lambda K, y0, t, dy: [0.5, K - y0, 0.001, y0],
        bounds_func=lambda K, y0, t: (
            [0.0001, 0.001, 1e-6, 0],
            [10, np.inf, 10, y0 * 2],
        ),
        model_type="mech_logistic",
    )


def fit_mech_gompertz(t, y):
    """
    Fit mechanistic Gompertz model (ODE) to growth data.

    ODE: dN/dt = μ * log(K/N) * N
    OD(t) = y0 + N(t)

    Parameters:
        t: Time array (hours)
        y: OD values

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
        y,
        model_func=mech_gompertz_model,
        param_names=["mu", "K", "N0", "y0"],
        p0_func=lambda K, y0, t, dy: [0.05, K - y0, 0.01, y0],
        bounds_func=lambda K, y0, t: (
            [0.0001, 0.01, 1e-4, y0 * 0.5],
            [2, np.inf, 1, y0 * 2],
        ),
        model_type="mech_gompertz",
    )


def fit_mech_richards(t, y):
    """
    Fit mechanistic Richards model (ODE) to growth data.

    ODE: dN/dt = μ * (1 - (N/K)^β) * N
    OD(t) = y0 + N(t)

    Parameters:
        t: Time array (hours)
        y: OD values

    Returns:
        Dict with 'params' and 'model_type', or None if fitting fails.
    """
    return _fit_model_generic(
        t,
        y,
        model_func=mech_richards_model,
        param_names=["mu", "K", "N0", "beta", "y0"],
        p0_func=lambda K, y0, t, dy: [0.5, K - y0, 0.001, 1.0, y0],
        bounds_func=lambda K, y0, t: (
            [0.0001, 0.001, 1e-6, 0.01, 0],
            [10, np.inf, 10, 100, y0 * 2],
        ),
        model_type="mech_richards",
    )


def fit_mech_baranyi(t, y):
    """
    Fit mechanistic Baranyi-Roberts model (ODE) to growth data.

    ODE: dN/dt = μ * A(t) * (1 - N/K) * N
    where A(t) = exp(μ*t) / (exp(h0) - 1 + exp(μ*t))
    OD(t) = y0 + N(t)

    Parameters:
        t: Time array (hours)
        y: OD values

    Returns:
        Dict with 'params' and 'model_type', or None if fitting fails.
    """

    def p0_baranyi(K, y0_init, t, dy):
        mu_init = 0.5
        h0_init = 1.0
        N0_init = 0.001
        return [mu_init, K - y0_init, N0_init, h0_init, y0_init]

    def bounds_baranyi(K, y0_init, t):
        return (
            [0.0001, 0.001, 1e-6, 0, 0],
            [10, np.inf, 10, t.max() * 10, y0_init * 2],
        )

    return _fit_model_generic(
        t,
        y,
        model_func=mech_baranyi_model,
        param_names=["mu", "K", "N0", "h0", "y0"],
        p0_func=p0_baranyi,
        bounds_func=bounds_baranyi,
        model_type="mech_baranyi",
    )


# -----------------------------------------------------------------------------
# Phenomenological Model Fitting Functions (ln-space)
# -----------------------------------------------------------------------------


def fit_phenom_logistic(t, y):
    """
    Fit phenomenological logistic model to ln(OD/OD0) data.

    ln(Nt/N0) = A / (1 + exp(4 * μ_max * (λ - t) / A + 2))

    Parameters:
        t: Time array (hours)
        y: OD values

    Returns:
        Dict with 'params' and 'model_type', or None if fitting fails.
    """
    t, y = validate_data(t, y)
    if t is None:
        return None

    # Estimate initial parameters
    N0 = float(np.min(y))
    N_max = float(np.max(y))
    A_init = np.log(N_max / N0)
    mu_max_init = 0.5
    lam_init = _estimate_lag_time(t, np.gradient(y, t))

    p0 = [A_init, mu_max_init, lam_init, N0]
    bounds = ([0.01, 0.0001, 0, N0 * 0.1], [20, 10, t.max(), N0 * 5])

    # Fit the model
    params, _ = curve_fit(
        phenom_logistic_model, t, y, p0=p0, bounds=bounds, maxfev=20000
    )

    return {
        "params": dict(zip(["A", "mu_max", "lam", "N0"], params)),
        "model_type": "phenom_logistic",
    }


def fit_phenom_gompertz(t, y):
    """
    Fit phenomenological Gompertz model to ln(OD/OD0) data.

    ln(Nt/N0) = A * exp(-exp(μ_max * e * (λ - t) / A + 1))

    Parameters:
        t: Time array (hours)
        y: OD values

    Returns:
        Dict with 'params' and 'model_type', or None if fitting fails.
    """
    t, y = validate_data(t, y)
    if t is None:
        return None

    # Estimate initial parameters
    N0 = float(np.min(y))
    N_max = float(np.max(y))
    A_init = np.log(N_max / N0)
    mu_max_init = 0.5
    lam_init = _estimate_lag_time(t, np.gradient(y, t))

    p0 = [A_init, mu_max_init, lam_init, N0]
    bounds = ([0.01, 0.0001, 0, N0 * 0.1], [20, 10, t.max(), N0 * 5])

    # Fit the model
    params, _ = curve_fit(
        phenom_gompertz_model, t, y, p0=p0, bounds=bounds, maxfev=20000
    )

    return {
        "params": dict(zip(["A", "mu_max", "lam", "N0"], params)),
        "model_type": "phenom_gompertz",
    }


def fit_phenom_gompertz_modified(t, y):
    """
    Fit phenomenological modified Gompertz model with decay term.

    ln(Nt/N0) = A * exp(-exp(μ_max * e * (λ - t) / A + 1)) + A * exp(α * (t - t_shift))

    Parameters:
        t: Time array (hours)
        y: OD values

    Returns:
        Dict with 'params' and 'model_type', or None if fitting fails.
    """
    t, y = validate_data(t, y)
    if t is None:
        return None

    # Estimate initial parameters
    N0 = float(np.min(y))
    N_max = float(np.max(y))
    A_init = np.log(N_max / N0)
    mu_max_init = 0.5
    lam_init = _estimate_lag_time(t, np.gradient(y, t))
    alpha_init = -0.01  # Small negative decay rate
    t_shift_init = t.max() * 0.5  # Midpoint

    p0 = [A_init, mu_max_init, lam_init, alpha_init, t_shift_init, N0]
    bounds = (
        [0.01, 0.0001, 0, -1, 0, N0 * 0.1],
        [20, 10, t.max(), 1, t.max(), N0 * 5],
    )

    # Fit the model
    params, _ = curve_fit(
        phenom_gompertz_modified_model, t, y, p0=p0, bounds=bounds, maxfev=20000
    )

    return {
        "params": dict(zip(["A", "mu_max", "lam", "alpha", "t_shift", "N0"], params)),
        "model_type": "phenom_gompertz_modified",
    }


def fit_phenom_richards(t, y):
    """
    Fit phenomenological Richards model to ln(OD/OD0) data.

    ln(Nt/N0)= A * (1 + ν * exp(1 + ν + μ_max * (1 + ν)^(1 + 1/ν) * (λ - t) / A))^(-1/ν)

    Parameters:
        t: Time array (hours)
        y: OD values

    Returns:
        Dict with 'params' and 'model_type', or None if fitting fails.
    """
    t, y = validate_data(t, y)
    if t is None:
        return None

    # Estimate initial parameters
    N0 = float(np.min(y))
    N_max = float(np.max(y))
    A_init = np.log(N_max / N0)
    mu_max_init = 0.5
    lam_init = _estimate_lag_time(t, np.gradient(y, t))
    nu_init = 1.0

    p0 = [A_init, mu_max_init, lam_init, nu_init, N0]
    bounds = ([0.01, 0.0001, 0, 0.01, N0 * 0.1], [20, 10, t.max(), 100, N0 * 5])

    # Fit the model
    params, _ = curve_fit(
        phenom_richards_model, t, y, p0=p0, bounds=bounds, maxfev=20000
    )

    return {
        "params": dict(zip(["A", "mu_max", "lam", "nu", "N0"], params)),
        "model_type": "phenom_richards",
    }


# -----------------------------------------------------------------------------
# Main Fitting Dispatcher
# -----------------------------------------------------------------------------


def fit_parametric(t, y, method="mech_logistic"):
    """
    Fit a growth model to data.

    Parameters:
        t: Time array (hours)
        y: OD values
        method: Model type string. Options:
            Mechanistic (ODE):  "mech_logistic", "mech_gompertz",
                                "mech_richards", "mech_baranyi"
            Phenomenological (ln-space): "phenom_logistic", "phenom_gompertz",
                "phenom_gompertz_modified", "phenom_richards"

    Returns:
        Fit result dict or None if fitting fails.
    """
    fit_funcs = {
        # Mechanistic models (ODE-based)
        "mech_logistic": fit_mech_logistic,
        "mech_gompertz": fit_mech_gompertz,
        "mech_richards": fit_mech_richards,
        "mech_baranyi": fit_mech_baranyi,
        # Phenomenological models (ln-space)
        "phenom_logistic": fit_phenom_logistic,
        "phenom_gompertz": fit_phenom_gompertz,
        "phenom_gompertz_modified": fit_phenom_gompertz_modified,
        "phenom_richards": fit_phenom_richards,
    }
    fit_func = fit_funcs.get(method)

    if fit_func is None:
        raise ValueError(
            f"Unknown method '{method}'. Must be one of {list(fit_funcs.keys())}"
        )

    result = fit_func(t, y)
    if result is not None:
        t_valid, _ = validate_data(t, y)
        if t_valid is None:
            return None
        result["params"]["fit_t_min"] = float(np.min(t_valid))
        result["params"]["fit_t_max"] = float(np.max(t_valid))
    return result


# -----------------------------------------------------------------------------
# Growth Statistics Extraction
# -----------------------------------------------------------------------------
