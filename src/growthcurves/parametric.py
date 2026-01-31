"""Parametric model fitting functions for growth curves.

This module provides functions to fit parametric growth models (Richards, Logistic,
Gompertz, Baranyi) and extract growth statistics from the fitted models.

All models operate in linear OD space (not log-transformed).
"""

import numpy as np
from scipy.optimize import curve_fit

from .models import baranyi_model, gompertz_model, logistic_model, richards_model
from .utils import validate_data, extract_stats_from_fit

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
        model_func: Model function to fit (e.g., logistic_model)
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
# Model Fitting Functions
# -----------------------------------------------------------------------------


def fit_logistic(t, y):
    """
    Fit logistic model to growth data.

    Parameters:
        t: Time array (hours)
        y: OD values

    Returns:
        Dict with 'params' and 'model_type', or None if fitting fails.
    """
    return _fit_model_generic(
        t,
        y,
        model_func=logistic_model,
        param_names=["K", "y0", "r", "t0"],
        p0_func=lambda K, y0, t, dy: [K, y0, 0.01, _estimate_inflection_time(t, dy)],
        bounds_func=lambda K, y0, t: (
            [y0 * 0.5, 0, 0.0001, t.min()],
            [np.inf, y0 * 2, 10, t.max()],
        ),
        model_type="logistic",
    )


def fit_gompertz(t, y):
    """
    Fit modified Gompertz model to growth data.

    Parameters:
        t: Time array (hours)
        y: OD values

    Returns:
        Dict with 'params' and 'model_type', or None if fitting fails.
    """
    return _fit_model_generic(
        t,
        y,
        model_func=gompertz_model,
        param_names=["K", "y0", "mu_max", "lam"],
        p0_func=lambda K, y0, t, dy: [K, y0, 0.01, _estimate_lag_time(t, dy)],
        bounds_func=lambda K, y0, t: (
            [y0 * 0.5, 0, 0.0001, 0],
            [np.inf, y0 * 2, 10, t.max()],
        ),
        model_type="gompertz",
    )


def fit_richards(t, y):
    """
    Fit Richards model to growth data.

    Parameters:
        t: Time array (hours)
        y: OD values

    Returns:
        Dict with 'params' and 'model_type', or None if fitting fails.
    """
    return _fit_model_generic(
        t,
        y,
        model_func=richards_model,
        param_names=["K", "y0", "r", "t0", "nu"],
        p0_func=lambda K, y0, t, dy: [
            K,
            y0,
            0.01,
            _estimate_inflection_time(t, dy),
            1.0,
        ],
        bounds_func=lambda K, y0, t: (
            [y0 * 0.5, 0, 0.0001, t.min(), 0.01],
            [np.inf, y0 * 2, 10, t.max(), 100],
        ),
        model_type="richards",
    )


def fit_baranyi(t, y):
    """
    Fit Baranyi-Roberts model to growth data.

    Parameters:
        t: Time array (hours)
        y: OD values

    Returns:
        Dict with 'params' and 'model_type', or None if fitting fails.
    """

    def p0_baranyi(K, y0, t, dy):
        lag_time = _estimate_lag_time(t, dy)
        mu_max_init = 0.01
        h0_init = lag_time * mu_max_init if lag_time > 0 else 0.1
        y0_floor = max(y0 * 0.5, 1e-6)
        return [K, max(y0, y0_floor), mu_max_init, h0_init]

    def bounds_baranyi(K, y0, t):
        y0_floor = max(y0 * 0.5, 1e-6)
        y0_ceil = max(y0 * 2, y0_floor * 10)
        return ([y0_floor, 0, 0.0001, 0], [np.inf, y0_ceil, 10, t.max() * 10])

    return _fit_model_generic(
        t,
        y,
        model_func=baranyi_model,
        param_names=["K", "y0", "mu_max", "h0"],
        p0_func=p0_baranyi,
        bounds_func=bounds_baranyi,
        model_type="baranyi",
    )


def fit_parametric(t, y, method="logistic"):
    """
    Fit a growth model to data.

    Parameters:
        t: Time array (hours)
        y: OD values
        model_type: One of "logistic", "gompertz", "richards", "baranyi"

    Returns:
        Fit result dict or None if fitting fails.
    """
    fit_funcs = {
        "logistic": fit_logistic,
        "gompertz": fit_gompertz,
        "richards": fit_richards,
        "baranyi": fit_baranyi,
    }
    fit_func = fit_funcs.get(method)

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
