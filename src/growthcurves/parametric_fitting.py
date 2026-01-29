"""Parametric model fitting functions for growth curves.

This module provides functions to fit parametric growth models (Richards, Logistic, Gompertz)
and extract growth statistics from the fitted models.

All models operate in linear OD space (not log-transformed).
"""

import numpy as np
from scipy.optimize import curve_fit

from .models import gompertz_model, logistic_model, richards_model
from .utils import (
    validate_data,
    compute_rmse,
    calculate_specific_growth_rate,
    bad_fit_stats,
)


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
        Dict with 'params', 'y_fit', 't' or None if fitting fails.
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
    y_fit = logistic_model(t, *params)

    return {
        "params": {
            "K": params[0],
            "y0": params[1],
            "r": params[2],
            "t0": params[3],
        },
        "y_fit": y_fit,
        "y": y,
        "t": t,
        "model_type": "logistic",
    }


def fit_gompertz(t, y):
    """
    Fit modified Gompertz model to growth data.

    Parameters:
        t: Time array (hours)
        y: OD values

    Returns:
        Dict with 'params', 'y_fit', 't' or None if fitting fails.
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
    y_fit = gompertz_model(t, *params)
    return {
        "params": {
            "K": params[0],
            "y0": params[1],
            "mu_max_param": params[2],
            "lam": params[3],
        },
        "y_fit": y_fit,
        "y": y,
        "t": t,
        "model_type": "gompertz",
    }


def fit_richards(t, y):
    """
    Fit Richards model to growth data.

    Parameters:
        t: Time array (hours)
        y: OD values

    Returns:
        Dict with 'params', 'y_fit', 't' or None if fitting fails.
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
    y_fit = richards_model(t, *params)

    return {
        "params": {
            "K": params[0],
            "y0": params[1],
            "r": params[2],
            "t0": params[3],
            "nu": params[4],
        },
        "y_fit": y_fit,
        "y": y,
        "t": t,
        "model_type": "richards",
    }


def fit_model(t, y, model_type="logistic"):
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
    fit_func = fit_funcs.get(model_type)

    return fit_func(t, y)


# -----------------------------------------------------------------------------
# Growth Statistics Extraction
# -----------------------------------------------------------------------------


def extract_stats_from_fit(fit_result, lag_frac=0.15, exp_frac=0.15):
    """
    Extract growth statistics from a model fit result.

    Parameters:
        fit_result: Dict from fit_* functions
        lag_frac: Fraction of peak growth rate for lag phase detection
        exp_frac: Fraction of peak growth rate for exponential phase end detection

    Returns:
        Growth statistics dictionary.
    """
    if fit_result is None:
        return bad_fit_stats()

    t = fit_result["t"]
    y = fit_result["y"]
    y_fit = fit_result["y_fit"]
    model_type = fit_result["model_type"]
    params = fit_result["params"]
    # Calculate true specific growth rate from fitted curve
    mu_max = calculate_specific_growth_rate(t, y_fit)

    # Generate dense predictions for derivative calculation
    t_dense = np.linspace(t.min(), t.max(), 500)

    if model_type == "logistic":
        y_dense = logistic_model(
            t_dense, params["K"], params["y0"], params["r"], params["t0"]
        )
    elif model_type == "gompertz":
        y_dense = gompertz_model(
            t_dense, params["K"], params["y0"], params["mu_max_param"], params["lam"]
        )
    elif model_type == "richards":
        y_dense = richards_model(
            t_dense, params["K"], params["y0"], params["r"], params["t0"], params["nu"]
        )
    else:
        return bad_fit_stats()

    # Maximum OD (carrying capacity from fit)
    max_od = float(params["K"])

    # Calculate dN/dt for phase boundary detection (linear space)
    dN_dt = np.gradient(y_dense, t_dense)

    # Specific growth rate: mu = d(ln N)/dt
    y_safe = np.maximum(y_dense, 1e-10)
    mu_dense = np.gradient(np.log(y_safe), t_dense)

    # Find time of maximum specific growth rate
    max_mu_idx = int(np.argmax(mu_dense))
    time_at_umax = float(t_dense[max_mu_idx])
    od_at_umax = float(y_dense[max_mu_idx])

    if mu_max <= 0:
        stats = bad_fit_stats()
        stats["max_od"] = max_od
        stats["fit_method"] = f"model_fitting_{model_type}"
        return stats

    # Phase boundaries based on derivative thresholds (linear space)
    max_dN = np.max(dN_dt)
    lag_threshold = lag_frac * max_dN
    exp_threshold = exp_frac * max_dN

    lag_idx = np.where(dN_dt >= lag_threshold)[0]
    exp_phase_start = (
        float(t_dense[lag_idx[0]]) if len(lag_idx) > 0 else float(t_dense[0])
    )

    exp_idx = np.where(
        (dN_dt <= exp_threshold) & (np.arange(len(t_dense)) > max_mu_idx)
    )[0]
    exp_phase_end = (
        float(t_dense[exp_idx[0]]) if len(exp_idx) > 0 else float(t_dense[-1])
    )

    # Doubling time based on mu_max (specific growth rate)
    doubling_time = np.log(2) / mu_max if mu_max > 0 else np.nan

    # RMSE in linear space
    rmse = compute_rmse(y, y_fit)

    return {
        "max_od": max_od,
        "specific_growth_rate": float(mu_max),
        "doubling_time": float(doubling_time),
        "exp_phase_start": exp_phase_start,
        "exp_phase_end": max(exp_phase_end, exp_phase_start),
        "time_at_umax": time_at_umax,
        "od_at_umax": od_at_umax,
        "t_window_start": float(t.min()),
        "t_window_end": float(t.max()),
        "fit_method": f"model_fitting_{model_type}",
        "model_rmse": rmse,
    }
