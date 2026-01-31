"""Non-parametric fitting methods for growth curves.

This module provides non-parametric methods for growth curve analysis,
including sliding window fitting and no-growth detection.

All methods operate in linear OD space (not log-transformed).
"""

import numpy as np

from .models import spline_model
from .utils import (
    bad_fit_stats,
    calculate_phase_ends,
    smooth,
)

# -----------------------------------------------------------------------------
# Sliding Window Helpers
# -----------------------------------------------------------------------------


# -----------------------------------------------------------------------------
# Umax Calculation Methods
# -----------------------------------------------------------------------------


def fit_sliding_window(t, y_raw, window_points=15):
    """
    Calculate maximum specific growth rate using the sliding window method.

    Finds the maximum specific growth rate by fitting a line to log-transformed
    OD data in consecutive windows, selecting the window with the steepest slope.

    Parameters:
        t: Time array (hours)
        y_raw: OD values (baseline-corrected, must be positive)
        window_points: Number of points in each sliding window

    Returns:
        Dict with model parameters:
            - slope: Slope of the linear fit in log space
                     (equals specific growth rate, h⁻¹)
            - intercept: Intercept of the linear fit in log space
            - time_at_umax: Time at maximum growth rate (hours)
            - model_type: "sliding_window"
        Returns None if calculation fails.
    """
    if len(t) < window_points or np.ptp(t) <= 0:
        return None

    # Log-transform for growth rate calculation
    y_log = np.log(y_raw)

    # Find window with maximum slope on log-transformed data
    w = window_points
    best_slope = -np.inf
    best_intercept = np.nan
    best_time = np.nan
    best_window_start = np.nan
    best_window_end = np.nan

    for i in range(len(t) - w + 1):
        t_win = t[i : i + w]
        y_log_win = y_log[i : i + w]

        if np.ptp(t_win) <= 0:
            continue

        slope, intercept = np.polyfit(t_win, y_log_win, 1)

        if slope > best_slope:
            best_slope = slope
            best_intercept = intercept
            best_time = float(np.mean(t_win))
            best_window_start = float(t_win.min())
            best_window_end = float(t_win.max())

    if not np.isfinite(best_slope) or best_slope <= 0:
        return None

    return {
        "params": {
            "slope": float(best_slope),
            "intercept": float(best_intercept),
            "time_at_umax": best_time,
            "fit_t_min": best_window_start,
            "fit_t_max": best_window_end,
        },
        "model_type": "sliding_window",
    }


def fit_spline(t_exp, y_exp, spline_s=None):
    """
    Calculate maximum specific growth rate using spline fitting.

    Fits a smoothing spline to log-transformed OD data and calculates
    the maximum specific growth rate from the spline's derivative.

    Parameters:
        t_exp: Time array for exponential phase (hours)
        y_exp: OD values for exponential phase
        spline_s: Smoothing factor for spline (None = automatic)

    Returns:
        Dict with model parameters:
            - t_knots: Spline knot points (time values)
            - spline_coeffs: Spline coefficients
            - spline_k: Spline degree (3)
            - time_at_umax: Time at maximum growth rate (hours)
            - model_type: "spline"
        Returns None if calculation fails.
    """
    if len(t_exp) < 5:
        return None

    # Fit spline to log-transformed data
    y_log_exp = np.log(y_exp)

    try:
        # Fit spline with automatic or specified smoothing
        if spline_s is None:
            # Automatic smoothing based on number of points
            spline_s = len(t_exp) * 0.1

        spline, spline_s = spline_model(t_exp, y_log_exp, spline_s, k=3)

        # Evaluate spline on dense grid for accurate derivative calculation
        t_eval = np.linspace(t_exp.min(), t_exp.max(), 200)

        # Calculate specific growth rate: μ = d(ln(N))/dt
        mu_eval = spline.derivative()(t_eval)

        # Find maximum specific growth rate
        max_mu_idx = int(np.argmax(mu_eval))
        mu_max = float(mu_eval[max_mu_idx])
        time_at_umax = float(t_eval[max_mu_idx])

        if mu_max <= 0 or not np.isfinite(mu_max):
            return None

        # Extract spline parameters for later reconstruction
        tck_t, tck_c, tck_k = spline._eval_args

        return {
            "params": {
                "tck_t": tck_t.tolist(),
                "tck_c": tck_c.tolist(),
                "tck_k": int(tck_k),
                "spline_s": spline_s,
                "time_at_umax": time_at_umax,
            },
            "model_type": "spline",
        }

    except Exception:
        return None


# -----------------------------------------------------------------------------
# Main API Functions
# -----------------------------------------------------------------------------


def fit_non_parametric(
    t,
    y,
    method="sliding_window",
    exp_start=0.15,
    exp_end=0.15,
    sg_window=11,
    sg_poly=1,
    window_points=15,
    spline_s=None,
):
    """
    Calculate growth statistics using non-parametric methods.

    This unified function supports multiple methods for calculating the maximum
    specific growth rate (Umax):
    - "sliding_window": Finds maximum slope in log-transformed OD across windows
    - "spline": Fits spline to exponential phase and calculates from derivative

    Parameters:
        t: Time array (hours)
        y: OD values (baseline-corrected, must be positive)
        umax_method: Method for calculating Umax ("sliding_window" or "spline")
        lag_frac: Fraction of peak growth rate for lag phase detection (default: 0.15)
        exp_frac: Fraction of peak growth rate for exponential phase end detection
                  (default: 0.15)
        sg_window: Savitzky-Golay filter window size for smoothing (default: 11)
        sg_poly: Polynomial order for Savitzky-Golay filter (default: 1)
        window_points: Number of points in sliding window (for sliding_window method)
        spline_s: Smoothing factor for spline (for spline method, None = automatic)

    Returns:
        Dict containing:
            - params: Model parameters (includes fit_t_min, fit_t_max, and other
                      method-specific values)
            - model_type: Method used for fitting
    """
    t = np.asarray(t, dtype=float)
    y = np.asarray(y, dtype=float)

    # Filter valid data (y must be positive for log transform)
    mask = np.isfinite(t) & np.isfinite(y) & (y > 0)
    t, y_raw = t[mask], y[mask]

    # Check minimum data requirements
    min_points = window_points if method == "sliding_window" else 10
    if len(t) < min_points or np.ptp(t) <= 0:
        return bad_fit_stats()

    # Maximum OD from raw data
    # max_od = float(np.max(y_raw))

    # Smooth the data for phase detection
    y_smooth = smooth(y_raw, sg_window, sg_poly)

    # Interpolate smoothed data to dense grid for phase boundary detection
    t_dense = np.linspace(t.min(), t.max(), 500)
    y_dense = np.interp(t_dense, t, y_smooth)

    # Calculate phase boundaries based on specific growth rate thresholds
    lag_end, exp_end = calculate_phase_ends(t_dense, y_dense, exp_start, exp_end)

    # Calculate Umax using specified method
    if method == "sliding_window":
        umax_result = fit_sliding_window(t, y_raw, window_points)

        if umax_result is None:
            return None

        # Extract parameters
        # params = umax_result["params"]
        # mu_max = params["slope"]
        # time_at_umax = params["time_at_umax"]

        return {
            "params": {
                **umax_result["params"],
                "window_points": window_points,
            },
            "model_type": "sliding_window",
        }

    elif method == "spline":
        # Extract exponential phase data
        exp_mask = (t >= lag_end) & (t <= exp_end)
        if np.sum(exp_mask) < 5:
            return None

        t_exp = t[exp_mask]
        y_exp = y_raw[exp_mask]
        umax_result = fit_spline(t_exp, y_exp, spline_s)

        if umax_result is None:
            return None

        # Extract parameters
        # params = umax_result["params"]
        # For spline, calculate mu_max from the derivative at time_at_umax
        # time_at_umax = params["time_at_umax"]

        # Reconstruct spline to get mu_max
        y_log_exp = np.log(y_exp)
        s = spline_s if spline_s is not None else len(t_exp) * 0.1
        try:
            spline, _ = spline_model(t_exp, y_log_exp, s, k=3)
            # mu_max = float(spline.derivative()(time_at_umax))
        except Exception:
            return None

        return {
            "params": {
                **umax_result["params"],
                "fit_t_min": float(t_exp.min()),
                "fit_t_max": float(t_exp.max()),
            },
            "model_type": "spline",
        }

    else:
        raise ValueError(f"Unknown umax_method: {method}")
