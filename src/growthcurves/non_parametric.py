"""Non-parametric fitting methods for growth curves.

This module provides non-parametric methods for growth curve analysis,
including sliding window fitting and no-growth detection.

All methods operate in linear OD space (not log-transformed).
"""

import numpy as np
from scipy.stats import theilslopes

from .inference import bad_fit_stats, calculate_phase_ends, smooth
from .models import spline_model

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
    OD data in consecutive windows using the Theil-Sen estimator, selecting the
    window with the steepest slope.

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

        # Use Theil-Sen estimator for robust line fitting
        result = theilslopes(y_log_win, t_win)
        slope, intercept = result.slope, result.intercept

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


def fit_spline(t, y, spline_s=None):
    """
    Calculate maximum specific growth rate using spline fitting.

    Fits a smoothing spline to log-transformed OD data and calculates
    the maximum specific growth rate from the spline's derivative.

    Parameters:
        t: Time array (hours)
        y: OD values
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
    if len(t) < 5:
        return None

    # Fit spline to log-transformed data
    y_log = np.log(y)

    try:
        # Fit spline with automatic or specified smoothing
        if spline_s is None:
            # Low smoothing for tight fit to data
            spline_s = 0.01

        spline, spline_s = spline_model(t, y_log, spline_s, k=3)

        # Evaluate spline on dense grid for accurate derivative calculation
        t_eval = np.linspace(t.min(), t.max(), 200)

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
                "mu_max": mu_max,  # Store calculated mu_max for consistency
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
    window_points=15,
    spline_s=None,
):
    """
    Calculate growth statistics using non-parametric methods.

    This unified function supports multiple methods for calculating the maximum
    specific growth rate (Umax):
    - "sliding_window": Finds maximum slope in log-transformed OD across windows
    - "spline": Fits spline to entire curve and calculates from derivative

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

    # Calculate Umax using specified method
    if method == "sliding_window":
        umax_result = fit_sliding_window(t, y_raw, window_points)

        if umax_result is None:
            return None

        return {
            "params": {
                **umax_result["params"],
                "window_points": window_points,
            },
            "model_type": "sliding_window",
        }

    elif method == "spline":
        # Fit spline to the entire dataset
        umax_result = fit_spline(t, y_raw, spline_s)

        if umax_result is None:
            return None

        # Store fit_t_min and fit_t_max as the full data range
        return {
            "params": {
                **umax_result["params"],
                "fit_t_min": float(t.min()),
                "fit_t_max": float(t.max()),
            },
            "model_type": "spline",
        }

    else:
        raise ValueError(f"Unknown umax_method: {method}")
