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
    phenom_gompertz_modified_model,
    phenom_logistic_model,
    phenom_richards_model,
)

# -----------------------------------------------------------------------------
# Helper Functions
# -----------------------------------------------------------------------------


def _estimate_initial_params(time, data):
    """
    Estimate common initial parameters for all growth models.

    Parameters:
        time: Time array
        data: OD values

    Returns:
        Tuple of (K_init, y0_init, dy) where:
            K_init: Initial carrying capacity (max OD)
            y0_init: Initial baseline OD (min OD)
            dy: First derivative (gradient) of OD with respect to time

    """
    K_init = np.max(data)
    y0_init = np.min(data)
    dy = np.gradient(data, time)
    return K_init, y0_init, dy


def _estimate_inflection_time(time, dy):
    """
    Estimate time at inflection point (maximum growth rate).

    Parameters:
        time: Time array
        dy: First derivative of OD

    Returns:
        Time at maximum derivative (inflection point)

    """
    return time[np.argmax(dy)]


def _estimate_lag_time(time, dy, threshold_frac=0.1):
    """
    Estimate lag time from growth rate threshold.

    Parameters:
        time: Time array
        dy: First derivative of OD
        threshold_frac: Fraction of max derivative to use as threshold

    Returns:
        Estimated lag time (time when growth rate exceeds threshold)

    """
    threshold = threshold_frac * np.max(dy)
    lag_idx = np.where(dy > threshold)[0]
    return time[lag_idx[0]] if len(lag_idx) > 0 else time[0]


def _fit_model_generic(time, data, model_func, param_names, p0_func, bounds_func, model_type):
    """
    Generic wrapper for fitting parametric growth models.

    This function encapsulates the common pattern used across all parametric
    model fitting functions, reducing code duplication.

    Parameters:
        time: Time array
        data: OD values
        model_func: Model function to fit (e.g., mech_logistic_model)
        param_names: List of parameter names in order
        p0_func: Function that takes (K_init, y0_init, time, dy) and returns p0
        bounds_func: Function that takes (K_init, y0_init, time) and returns bounds
        model_type: String identifier for the model

    Returns:
        Dict with 'params' and 'model_type', or None if fitting fails

    """
    time, data = validate_data(time, data)
    if time is None:
        return None

    # Estimate common initial parameters
    K_init, y0_init, dy = _estimate_initial_params(time, data)

    # Generate initial guess and bounds
    p0 = p0_func(K_init, y0_init, time, dy)
    bounds = bounds_func(K_init, y0_init, time)

    # Fit the model
    params, _ = curve_fit(model_func, time, data, p0=p0, bounds=bounds, maxfev=20000)

    # Return structured result
    return {
        "params": dict(zip(param_names, params)),
        "model_type": model_type,
    }


# -----------------------------------------------------------------------------
# Mechanistic Model Fitting Functions (ODE-based)
# -----------------------------------------------------------------------------


def fit_mech_logistic(time, data):
    """
    Fit mechanistic logistic model (ODE) to growth data.

    ODE: dN/dt = μ * (1 - N/K) * N
    OD(t) = y0 + N(t)

    Parameters:
        time: Time array (hours)
        data: OD values

    Returns:
        Dict with 'params' and 'model_type', or None if fitting fails.

    """
    return _fit_model_generic(
        time,
        data,
        model_func=mech_logistic_model,
        param_names=["mu", "K", "N0", "y0"],
        p0_func=lambda K, y0, time, dy: [0.5, K - y0, 0.001, y0],
        bounds_func=lambda K, y0, time: (
            [0.0001, 0.001, 1e-6, 0],
            [10, np.inf, 10, y0 * 2],
        ),
        model_type="mech_logistic",
    )


def fit_mech_gompertz(time, data):
    """
    Fit mechanistic Gompertz model (ODE) to growth data.

    ODE: dN/dt = μ * log(K/N) * N
    OD(t) = y0 + N(t)

    Parameters:
        time: Time array (hours)
        data: OD values

    Returns:
        Dict with 'params' and 'model_type', or None if fitting fails.

    Note:
        The mechanistic Gompertz model can be numerically challenging to fit
        due to the logarithmic term in the ODE. If fitting fails or produces
        poor results, consider using mech_logistic, mech_richards, or
        phenom_gompertz instead.

    """
    return _fit_model_generic(
        time,
        data,
        model_func=mech_gompertz_model,
        param_names=["mu", "K", "N0", "y0"],
        p0_func=lambda K, y0, time, dy: [0.05, K - y0, 0.01, y0],
        bounds_func=lambda K, y0, time: (
            [0.0001, 0.01, 1e-4, y0 * 0.5],
            [2, np.inf, 1, y0 * 2],
        ),
        model_type="mech_gompertz",
    )


def fit_mech_richards(time, data):
    """
    Fit mechanistic Richards model (ODE) to growth data.

    ODE: dN/dt = μ * (1 - (N/K)^β) * N
    OD(t) = y0 + N(t)

    Parameters:
        time: Time array (hours)
        data: OD values

    Returns:
        Dict with 'params' and 'model_type', or None if fitting fails.
    """
    return _fit_model_generic(
        time,
        data,
        model_func=mech_richards_model,
        param_names=["mu", "K", "N0", "beta", "y0"],
        p0_func=lambda K, y0, time, dy: [0.5, K - y0, 0.001, 1.0, y0],
        bounds_func=lambda K, y0, time: (
            [0.0001, 0.001, 1e-6, 0.01, 0],
            [10, np.inf, 10, 100, y0 * 2],
        ),
        model_type="mech_richards",
    )


def fit_mech_baranyi(time, data):
    """
    Fit mechanistic Baranyi-Roberts model (ODE) to growth data.

    ODE: dN/dt = μ * A(t) * (1 - N/K) * N
    where A(t) = exp(μ*t) / (exp(h0) - 1 + exp(μ*t))
    OD(t) = y0 + N(t)

    Parameters:
        time: Time array (hours)
        data: OD values

    Returns:
        Dict with 'params' and 'model_type', or None if fitting fails.
    """

    def p0_baranyi(K, y0_init, time, dy):
        mu_init = 0.5
        h0_init = 1.0
        N0_init = 0.001
        return [mu_init, K - y0_init, N0_init, h0_init, y0_init]

    def bounds_baranyi(K, y0_init, time):
        return (
            [0.0001, 0.001, 1e-6, 0, 0],
            [10, np.inf, 10, time.max() * 10, y0_init * 2],
        )

    return _fit_model_generic(
        time,
        data,
        model_func=mech_baranyi_model,
        param_names=["mu", "K", "N0", "h0", "y0"],
        p0_func=p0_baranyi,
        bounds_func=bounds_baranyi,
        model_type="mech_baranyi",
    )


# -----------------------------------------------------------------------------
# Phenomenological Model Fitting Functions (ln-space)
# -----------------------------------------------------------------------------


def fit_phenom_logistic(time, data):
    """
    Fit phenomenological logistic model to ln(OD/OD0) data.

    ln(Nt/N0) = A / (1 + exp(4 * μ_max * (λ - t) / A + 2))

    Parameters:
        time: Time array (hours)
        data: OD values

    Returns:
        Dict with 'params' and 'model_type', or None if fitting fails.
    """
    time, data = validate_data(time, data)
    if time is None:
        return None

    # Estimate initial parameters
    N0 = float(np.min(data))
    N_max = float(np.max(data))
    A_init = np.log(N_max / N0)
    mu_max_init = 0.5
    lam_init = _estimate_lag_time(time, np.gradient(data, time))

    p0 = [A_init, mu_max_init, lam_init, N0]
    bounds = ([0.01, 0.0001, 0, N0 * 0.1], [20, 10, time.max(), N0 * 5])

    # Fit the model
    params, _ = curve_fit(
        phenom_logistic_model, time, data, p0=p0, bounds=bounds, maxfev=20000
    )

    return {
        "params": dict(zip(["A", "mu_max", "lam", "N0"], params)),
        "model_type": "phenom_logistic",
    }


def fit_phenom_gompertz(time, data):
    """
    Fit phenomenological Gompertz model to ln(OD/OD0) data.

    ln(Nt/N0) = A * exp(-exp(μ_max * e * (λ - t) / A + 1))

    Parameters:
        time: Time array (hours)
        data: OD values

    Returns:
        Dict with 'params' and 'model_type', or None if fitting fails.
    """
    time, data = validate_data(time, data)
    if time is None:
        return None

    # Estimate initial parameters
    N0 = float(np.min(data))
    N_max = float(np.max(data))
    A_init = np.log(N_max / N0)
    mu_max_init = 0.5
    lam_init = _estimate_lag_time(time, np.gradient(data, time))

    p0 = [A_init, mu_max_init, lam_init, N0]
    bounds = ([0.01, 0.0001, 0, N0 * 0.1], [20, 10, time.max(), N0 * 5])

    # Fit the model
    params, _ = curve_fit(
        phenom_gompertz_model, time, data, p0=p0, bounds=bounds, maxfev=20000
    )

    return {
        "params": dict(zip(["A", "mu_max", "lam", "N0"], params)),
        "model_type": "phenom_gompertz",
    }


def fit_phenom_gompertz_modified(time, data):
    """
    Fit phenomenological modified Gompertz model with decay term.

    ln(Nt/N0) = A * exp(-exp(μ_max * e * (λ - t) / A + 1)) + A * exp(α * (t - t_shift))

    Parameters:
        time: Time array (hours)
        data: OD values

    Returns:
        Dict with 'params' and 'model_type', or None if fitting fails.
    """
    time, data = validate_data(time, data)
    if time is None:
        return None

    # Estimate initial parameters
    N0 = float(np.min(data))
    N_max = float(np.max(data))
    A_init = np.log(N_max / N0)
    mu_max_init = 0.5
    lam_init = _estimate_lag_time(time, np.gradient(data, time))
    alpha_init = -0.01  # Small negative decay rate
    t_shift_init = time.max() * 0.5  # Midpoint

    p0 = [A_init, mu_max_init, lam_init, alpha_init, t_shift_init, N0]
    bounds = (
        [0.01, 0.0001, 0, -1, 0, N0 * 0.1],
        [20, 10, time.max(), 1, time.max(), N0 * 5],
    )

    # Fit the model
    params, _ = curve_fit(
        phenom_gompertz_modified_model, time, data, p0=p0, bounds=bounds, maxfev=20000
    )

    return {
        "params": dict(zip(["A", "mu_max", "lam", "alpha", "t_shift", "N0"], params)),
        "model_type": "phenom_gompertz_modified",
    }


def fit_phenom_richards(time, data):
    """
    Fit phenomenological Richards model to ln(OD/OD0) data.

    ln(Nt/N0)= A * (1 + ν * exp(1 + ν + μ_max * (1 + ν)^(1 + 1/ν) * (λ - t) / A))^(-1/ν)

    Parameters:
        time: Time array (hours)
        data: OD values

    Returns:
        Dict with 'params' and 'model_type', or None if fitting fails.
    """
    time, data = validate_data(time, data)
    if time is None:
        return None

    # Estimate initial parameters
    N0 = float(np.min(data))
    N_max = float(np.max(data))
    A_init = np.log(N_max / N0)
    mu_max_init = 0.5
    lam_init = _estimate_lag_time(time, np.gradient(data, time))
    nu_init = 1.0

    p0 = [A_init, mu_max_init, lam_init, nu_init, N0]
    bounds = ([0.01, 0.0001, 0, 0.01, N0 * 0.1], [20, 10, time.max(), 100, N0 * 5])

    # Fit the model
    params, _ = curve_fit(
        phenom_richards_model, time, data, p0=p0, bounds=bounds, maxfev=20000
    )

    return {
        "params": dict(zip(["A", "mu_max", "lam", "nu", "N0"], params)),
        "model_type": "phenom_richards",
    }


# -----------------------------------------------------------------------------
# Main Fitting Dispatcher
# -----------------------------------------------------------------------------


def fit_parametric(time, data, method="mech_logistic", **kwargs):
    """
    Fit a growth model to data.

    Parameters:
        time: Time array (hours)
        data: OD values
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

    result = fit_func(time, data)
    if result is not None:
        time_valid, _ = validate_data(time, data)
        if time_valid is None:
            return None
        result["params"]["fit_t_min"] = float(np.min(time_valid))
        result["params"]["fit_t_max"] = float(np.max(time_valid))
    return result
