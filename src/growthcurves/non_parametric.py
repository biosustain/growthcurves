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
            # Low smoothing for tight fit to data
            spline_s = 0.01

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
        # For spline fitting, identify approximate region of Umax using 30% threshold
        # on instantaneous mu. This defines fit_t_min and fit_t_max.
        # Use dense grid to find boundaries, then expand to include sufficient data points

        # Calculate instantaneous mu on dense grid
        from .utils import compute_mu_max as calc_mu

        _, mu_dense = calc_mu(t_dense, y_dense)
        mu_dense = np.nan_to_num(mu_dense, nan=0.0)
        mu_dense = np.maximum(mu_dense, 0)

        if np.max(mu_dense) <= 0:
            return None

        # Find time of maximum mu
        max_mu_idx = np.argmax(mu_dense)
        t_at_max_mu = t_dense[max_mu_idx]
        max_mu = mu_dense[max_mu_idx]

        # Find region where mu > 30% of max_mu
        threshold_30pct = 0.30 * max_mu
        above_threshold = mu_dense >= threshold_30pct

        if not np.any(above_threshold):
            return None

        # Find contiguous region containing the maximum
        indices_above = np.where(above_threshold)[0]

        # Find the contiguous block that contains max_mu_idx
        # Split into contiguous segments
        segments = np.split(indices_above, np.where(np.diff(indices_above) > 1)[0] + 1)

        # Find which segment contains the maximum
        target_segment = None
        for seg in segments:
            if max_mu_idx in seg:
                target_segment = seg
                break

        if target_segment is None or len(target_segment) == 0:
            return None

        # Get time boundaries from the segment
        fit_t_min_initial = float(t_dense[target_segment[0]])
        fit_t_max_initial = float(t_dense[target_segment[-1]])

        # Expand window to ensure we have at least 10 data points for robust spline fitting
        exp_mask = (t >= fit_t_min_initial) & (t <= fit_t_max_initial)
        n_points = np.sum(exp_mask)

        # If fewer than 10 points, expand window symmetrically around t_at_max_mu
        if n_points < 10:
            # Calculate required window width to get ~15 points
            dt_avg = np.mean(np.diff(t))  # Average time spacing
            half_width = 7.5 * dt_avg  # Width for ~15 points
            fit_t_min_initial = max(t.min(), t_at_max_mu - half_width)
            fit_t_max_initial = min(t.max(), t_at_max_mu + half_width)
            exp_mask = (t >= fit_t_min_initial) & (t <= fit_t_max_initial)

        if np.sum(exp_mask) < 5:
            return None

        t_exp = t[exp_mask]
        y_exp = y_raw[exp_mask]
        umax_result = fit_spline(t_exp, y_exp, spline_s)

        if umax_result is None:
            return None

        # Store fit_t_min and fit_t_max (the region where spline was fitted)
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
