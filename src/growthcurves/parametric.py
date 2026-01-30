"""Parametric model fitting functions for growth curves.

This module provides functions to fit parametric growth models (Richards, Logistic, Gompertz)
and extract growth statistics from the fitted models.

All models operate in linear OD space (not log-transformed).
"""

import numpy as np
from scipy.optimize import curve_fit

from .models import gompertz_model, logistic_model, richards_model
from .utils import validate_data, extract_stats_from_fit

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
    t, y = validate_data(t, y)
    if t is None:
        return None

    # Initial estimates
    K_init = np.max(y)
    y0_init = np.min(y)  # Baseline OD
    # Estimate t0 as time of maximum growth rate (inflection point)
    dy = np.gradient(y, t)
    t0_init = t[np.argmax(dy)]
    r_init = 0.01  # Initial growth rate guess

    p0 = [K_init, y0_init, r_init, t0_init]
    bounds = ([y0_init * 0.5, 0, 0.0001, t.min()], [np.inf, y0_init * 2, 10, t.max()])

    params, _ = curve_fit(logistic_model, t, y, p0=p0, bounds=bounds, maxfev=20000)

    return {
        "params": {
            "K": params[0],
            "y0": params[1],
            "r": params[2],
            "t0": params[3],
        },
        "model_type": "logistic",
    }


def fit_gompertz(t, y):
    """
    Fit modified Gompertz model to growth data.

    Parameters:
        t: Time array (hours)
        y: OD values

    Returns:
        Dict with 'params' and 'model_type', or None if fitting fails.
    """
    t, y = validate_data(t, y)
    if t is None:
        return None

    # Initial estimates
    K_init = np.max(y)
    y0_init = np.min(y)  # Baseline OD
    # Estimate lag time as time when growth first accelerates
    dy = np.gradient(y, t)
    threshold = 0.1 * np.max(dy)
    lag_idx = np.where(dy > threshold)[0]
    lam_init = t[lag_idx[0]] if len(lag_idx) > 0 else t[0]
    mu_max_init = 0.01  # Initial growth rate guess

    p0 = [K_init, y0_init, mu_max_init, lam_init]
    bounds = ([y0_init * 0.5, 0, 0.0001, 0], [np.inf, y0_init * 2, 10, t.max()])

    params, _ = curve_fit(gompertz_model, t, y, p0=p0, bounds=bounds, maxfev=20000)

    return {
        "params": {
            "K": params[0],
            "y0": params[1],
            "mu_max_param": params[2],
            "lam": params[3],
        },
        "model_type": "gompertz",
    }


def fit_richards(t, y):
    """
    Fit Richards model to growth data.

    Parameters:
        t: Time array (hours)
        y: OD values

    Returns:
        Dict with 'params' and 'model_type', or None if fitting fails.
    """
    t, y = validate_data(t, y)
    if t is None:
        return None

    # Initial estimates
    K_init = np.max(y)
    y0_init = np.min(y)  # Baseline OD
    dy = np.gradient(y, t)
    t0_init = t[np.argmax(dy)]
    r_init = 0.01
    nu_init = 1.0  # Start with logistic-like shape

    p0 = [K_init, y0_init, r_init, t0_init, nu_init]
    bounds = (
        [y0_init * 0.5, 0, 0.0001, t.min(), 0.01],
        [np.inf, y0_init * 2, 10, t.max(), 100],
    )

    params, _ = curve_fit(richards_model, t, y, p0=p0, bounds=bounds, maxfev=20000)

    return {
        "params": {
            "K": params[0],
            "y0": params[1],
            "r": params[2],
            "t0": params[3],
            "nu": params[4],
        },
        "model_type": "richards",
    }


def fit_parametric(t, y, model="logistic"):
    """
    Fit a growth model to data.

    Parameters:
        t: Time array (hours)
        y: OD values
        model_type: One of "logistic", "gompertz", "richards"

    Returns:
        Fit result dict or None if fitting fails.
    """
    fit_funcs = {
        "logistic": fit_logistic,
        "gompertz": fit_gompertz,
        "richards": fit_richards,
    }
    fit_func = fit_funcs.get(model)

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
