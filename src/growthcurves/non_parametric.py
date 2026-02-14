"""Non-parametric fitting methods for growth curves.

This module provides non-parametric methods for growth curve analysis,
including sliding window fitting and no-growth detection.

All methods operate in linear OD space (not log-transformed).
"""

import numpy as np
from scipy.stats import theilslopes

from .inference import bad_fit_stats
from .models import spline_model

# -----------------------------------------------------------------------------
# Sliding Window Helpers
# -----------------------------------------------------------------------------


# -----------------------------------------------------------------------------
# Umax Calculation Methods
# -----------------------------------------------------------------------------


def fit_sliding_window(time, data, window_points=15, step=None, n_fits=None):
    """
    Calculate maximum specific growth rate using the sliding window method.

    Finds the maximum specific growth rate by fitting a line to log-transformed
    OD data in consecutive windows using the Theil-Sen estimator, selecting the
    window with the steepest slope.

    Parameters:
        time: Time array (hours)
        data: OD values (baseline-corrected, must be positive)
        window_points: Number of points in each sliding window
        step: Step size for sliding window (default: 1 if step is None and
              `n_fits` is None)
        n_fits: Approximate number of fits to perform (default: None). Ignored
                if `step` is provided.

    Returns:
        Dict with model parameters:
            - slope: Slope of the linear fit in log space
                     (equals specific growth rate, h⁻¹)
            - intercept: Intercept of the linear fit in log space
            - time_at_umax: Time at maximum growth rate (hours)
            - model_type: "sliding_window"
        Returns None if calculation fails.

    """
    if len(time) < window_points or np.ptp(time) <= 0:
        return None

    # Log-transform for growth rate calculation
    y_log = np.log(data)

    # Find window with maximum slope on log-transformed data
    w = window_points
    best_slope = -np.inf
    best_intercept = np.nan
    best_time = np.nan
    best_window_start = np.nan
    best_window_end = np.nan

    if step is None:
        if n_fits is None:
            step = 1
        else:
            step = max(1, int(len(time) / n_fits))

    # limit number of fits to avoid excessive computation using step parameter
    for i in range(0, len(time) - w + 1, step):
        t_win = time[i : i + w]
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


def fit_spline(time, data, spline_s=None):
    """
    Calculate maximum specific growth rate using spline fitting.

    Fits a smoothing spline to log-transformed OD data and calculates
    the maximum specific growth rate from the spline's derivative.

    Parameters:
        time: Time array (hours)
        data: OD values
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
    if len(time) < 5:
        return None

    # Fit spline to log-transformed data
    y_log = np.log(data)

    try:
        # Fit spline with automatic or specified smoothing
        if spline_s is None:
            # Low smoothing for tight fit to data
            spline_s = 0.01

        spline, spline_s = spline_model(time, y_log, spline_s, k=3)

        # Evaluate spline on dense grid for accurate derivative calculation
        t_eval = np.linspace(time.min(), time.max(), 200)

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
    time,
    data,
    method="sliding_window",
    window_points=15,
    spline_s=None,
    **kwargs,
):
    """
    Calculate growth statistics using non-parametric methods.

    This unified function supports multiple methods for calculating the maximum
    specific growth rate (Umax):
    - "sliding_window": Finds maximum slope in log-transformed OD across windows
    - "spline": Fits spline to entire curve and calculates from derivative

    Parameters:
        time: Time array (hours)
        data: OD values (baseline-corrected, must be positive)
        method: Method for calculating Umax ("sliding_window" or "spline")
        window_points: Number of points in sliding window (for sliding_window method)
        spline_s: Smoothing factor for spline (for spline method, None = automatic)
        **kwargs: Additional arguments (for compatibility)

    Returns:
        Dict containing:
            - params: Model parameters (includes fit_t_min, fit_t_max, and other
                      method-specific values)
            - model_type: Method used for fitting
    """
    time = np.asarray(time, dtype=float)
    data = np.asarray(data, dtype=float)

    # Filter valid data (data must be positive for log transform)
    mask = np.isfinite(time) & np.isfinite(data) & (data > 0)
    time, y_raw = time[mask], data[mask]

    # Check minimum data requirements
    min_points = window_points if method == "sliding_window" else 10
    if len(time) < min_points or np.ptp(time) <= 0:
        return bad_fit_stats()

    # Calculate Umax using specified method
    if method == "sliding_window":
        umax_result = fit_sliding_window(time, y_raw, window_points)

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
        umax_result = fit_spline(time, y_raw, spline_s)

        if umax_result is None:
            return None

        # Store fit_t_min and fit_t_max as the full data range
        return {
            "params": {
                **umax_result["params"],
                "fit_t_min": float(time.min()),
                "fit_t_max": float(time.max()),
            },
            "model_type": "spline",
        }

    else:
        raise ValueError(f"Unknown umax_method: {method}")
