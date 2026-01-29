"""Non-parametric fitting methods for growth curves.

This module provides non-parametric methods for growth curve analysis,
including sliding window fitting and no-growth detection.

All methods operate in linear OD space (not log-transformed).
"""

import numpy as np
from scipy.optimize import curve_fit
from scipy.interpolate import UnivariateSpline

from .models import gaussian
from .utils import compute_rmse, smooth, bad_fit_stats, detect_no_growth, is_no_growth


# -----------------------------------------------------------------------------
# Sliding Window Helpers
# -----------------------------------------------------------------------------


def fit_gaussian_to_derivative(t, dy, t_dense):
    """
    Fit a symmetric Gaussian to first-derivative data and evaluate on t_dense.

    Returns:
        Array of fitted Gaussian values evaluated at t_dense.
    """
    t = np.asarray(t, dtype=float)
    dy = np.asarray(dy, dtype=float)
    t_dense = np.asarray(t_dense, dtype=float)

    mask = np.isfinite(t) & np.isfinite(dy)
    t_fit = t[mask]
    dy_fit = dy[mask]

    if len(t_fit) < 3 or np.ptp(t_fit) <= 0 or np.max(dy_fit) <= 0:
        if len(t_fit) == 0:
            return np.zeros_like(t_dense)
        return np.interp(t_dense, t_fit, dy_fit, left=0.0, right=0.0)

    amplitude_init = float(np.max(dy_fit))
    center_init = float(t_fit[np.argmax(dy_fit)])
    weights = np.maximum(dy_fit, 0)
    if np.sum(weights) > 0:
        sigma_init = float(
            np.sqrt(np.sum(weights * (t_fit - center_init) ** 2) / np.sum(weights))
        )
    else:
        sigma_init = float(np.ptp(t_fit) / 6.0)
    sigma_init = max(sigma_init, np.ptp(t_fit) / 20.0)

    p0 = [amplitude_init, center_init, sigma_init]
    bounds = (
        [0.0, float(t_fit.min()), 1e-6],
        [np.inf, float(t_fit.max()), np.ptp(t_fit)],
    )

    try:
        params, _ = curve_fit(
            gaussian, t_fit, dy_fit, p0=p0, bounds=bounds, maxfev=20000
        )
        return gaussian(t_dense, *params)
    except Exception:
        return np.interp(t_dense, t_fit, dy_fit, left=0.0, right=0.0)


def calculate_phase_ends(t, dy, lag_frac=0.15, exp_frac=0.15):
    """
    Calculate lag and exponential phase end times from a first derivative curve.

    Parameters:
        t: Time array
        dy: First derivative values (should be from fitted/idealized curve)
        lag_frac: Fraction of peak derivative for lag phase end detection
        exp_frac: Fraction of peak derivative for exponential phase end detection

    Returns:
        Tuple of (lag_end, exp_end) times.
    """
    if len(t) < 5 or np.ptp(t) <= 0:
        return float(t[0]) if len(t) > 0 else np.nan, (
            float(t[-1]) if len(t) > 0 else np.nan
        )

    dy = np.maximum(dy, 0)  # Only consider positive growth

    peak_idx = np.argmax(dy)
    peak_val = dy[peak_idx]

    if peak_val <= 0:
        return float(t[0]), float(t[-1])

    lag_threshold = lag_frac * peak_val
    exp_threshold = exp_frac * peak_val

    lag_end = float(t[0])
    above_lag = dy >= lag_threshold
    if np.any(above_lag):
        first_idx = int(np.argmax(above_lag))
        if first_idx > 0:
            t0, t1 = t[first_idx - 1], t[first_idx]
            y0, y1 = dy[first_idx - 1], dy[first_idx]
            frac = 0.0 if y1 == y0 else (lag_threshold - y0) / (y1 - y0)
            lag_end = float(t0 + frac * (t1 - t0))
        else:
            lag_end = float(t[first_idx])

    exp_end = float(t[-1])
    after_peak = np.arange(len(t)) > peak_idx
    below_exp = dy <= exp_threshold
    exp_candidates = np.where(after_peak & below_exp)[0]
    if len(exp_candidates) > 0:
        idx = int(exp_candidates[0])
        if idx > 0:
            t0, t1 = t[idx - 1], t[idx]
            y0, y1 = dy[idx - 1], dy[idx]
            frac = 0.0 if y1 == y0 else (exp_threshold - y0) / (y1 - y0)
            exp_end = float(t0 + frac * (t1 - t0))
        else:
            exp_end = float(t[idx])

    return lag_end, max(exp_end, lag_end)


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
        Dict with:
            - specific_growth_rate: Maximum specific growth rate (h^-1)
            - time_at_umax: Time at maximum growth rate (hours)
        Returns None if calculation fails.
    """
    if len(t) < window_points or np.ptp(t) <= 0:
        return None

    # Log-transform for growth rate calculation
    y_log = np.log(y_raw)

    # Find window with maximum slope on log-transformed data
    w = window_points
    best_slope = -np.inf
    best_time = np.nan

    for i in range(len(t) - w + 1):
        t_win = t[i : i + w]
        y_log_win = y_log[i : i + w]

        if np.ptp(t_win) <= 0:
            continue

        slope, _ = np.polyfit(t_win, y_log_win, 1)

        if slope > best_slope:
            best_slope = slope
            best_time = float(np.mean(t_win))

    if not np.isfinite(best_slope) or best_slope <= 0:
        return None

    return {
        "specific_growth_rate": float(best_slope),
        "time_at_umax": best_time,
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
        Dict with:
            - specific_growth_rate: Maximum specific growth rate (h^-1)
            - time_at_umax: Time at maximum growth rate (hours)
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

        spline = UnivariateSpline(t_exp, y_log_exp, s=spline_s, k=3)

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

        return {
            "specific_growth_rate": mu_max,
            "time_at_umax": time_at_umax,
        }

    except Exception:
        return None


# -----------------------------------------------------------------------------
# Main API Functions
# -----------------------------------------------------------------------------


def fit_parametric(
    t,
    y,
    umax_method="sliding_window",
    lag_frac=0.15,
    exp_frac=0.15,
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
        exp_frac: Fraction of peak growth rate for exponential phase end detection (default: 0.15)
        sg_window: Savitzky-Golay filter window size for smoothing (default: 11)
        sg_poly: Polynomial order for Savitzky-Golay filter (default: 1)
        window_points: Number of points in sliding window (for sliding_window method)
        spline_s: Smoothing factor for spline (for spline method, None = automatic)

    Returns:
        Dict containing:
            - max_od: Maximum OD value
            - specific_growth_rate: Maximum specific growth rate (h^-1)
            - doubling_time: Doubling time (hours)
            - exp_phase_start: Time when lag phase ends (hours)
            - exp_phase_end: Time when exponential phase ends (hours)
            - time_at_umax: Time at maximum growth rate (hours)
            - od_at_umax: OD at maximum growth rate
            - t_window_start: Start time of analysis window
            - t_window_end: End time of analysis window
            - fit_method: Method used for fitting
            - model_rmse: RMSE of the fit
    """
    t = np.asarray(t, dtype=float)
    y = np.asarray(y, dtype=float)

    # Filter valid data (y must be positive for log transform)
    mask = np.isfinite(t) & np.isfinite(y) & (y > 0)
    t, y_raw = t[mask], y[mask]

    # Check minimum data requirements
    min_points = window_points if umax_method == "sliding_window" else 10
    if len(t) < min_points or np.ptp(t) <= 0:
        return bad_fit_stats()

    # Maximum OD from raw data
    max_od = float(np.max(y_raw))

    # Smooth the data and calculate first derivative for phase detection
    y_smooth = smooth(y_raw, sg_window, sg_poly)
    dy_data = np.gradient(y_smooth, t)
    dy_data = np.maximum(dy_data, 0)  # Only consider positive growth

    # Fit idealized symmetric Gaussian to the derivative data
    t_dense = np.linspace(t.min(), t.max(), 500)
    dy_idealized = fit_gaussian_to_derivative(t, dy_data, t_dense)

    # Calculate phase boundaries
    lag_end, exp_end = calculate_phase_ends(t_dense, dy_idealized, lag_frac, exp_frac)

    # Calculate Umax using specified method
    if umax_method == "sliding_window":
        umax_result = fit_sliding_window(t, y_raw, window_points)
        fit_method = "sliding_window"

        if umax_result is None:
            stats = bad_fit_stats()
            stats["max_od"] = max_od
            return stats

        # For sliding window, the analysis window is the full data range
        t_window_start = float(t.min())
        t_window_end = float(t.max())

    elif umax_method == "spline":
        # Extract exponential phase data
        exp_mask = (t >= lag_end) & (t <= exp_end)
        if np.sum(exp_mask) < 5:
            return bad_fit_stats()

        t_exp = t[exp_mask]
        y_exp = y_raw[exp_mask]
        umax_result = fit_spline(t_exp, y_exp, spline_s)
        fit_method = "spline"

        if umax_result is None:
            stats = bad_fit_stats()
            stats["max_od"] = max_od
            return stats

        # For spline, the analysis window is the exponential phase
        t_window_start = float(t_exp.min())
        t_window_end = float(t_exp.max())

    else:
        raise ValueError(
            f"Unknown umax_method: {umax_method}. Use 'sliding_window' or 'spline'"
        )

    # Extract results from helper functions
    mu_max = umax_result["specific_growth_rate"]
    time_at_umax = umax_result["time_at_umax"]

    # Calculate OD at Umax by interpolating from raw data
    od_at_umax = float(np.interp(time_at_umax, t, y_raw))

    # Calculate RMSE based on method
    if umax_method == "sliding_window":
        # For sliding window, fit a line in log space around time_at_umax
        w = window_points
        y_log = np.log(y_raw)

        # Find closest window to time_at_umax
        best_dist = np.inf
        best_idx = 0
        for i in range(len(t) - w + 1):
            t_win = t[i : i + w]
            t_mid = np.mean(t_win)
            dist = abs(t_mid - time_at_umax)
            if dist < best_dist:
                best_dist = dist
                best_idx = i

        # Calculate RMSE for this window
        t_win = t[best_idx : best_idx + w]
        y_log_win = y_log[best_idx : best_idx + w]
        if len(t_win) >= 2 and np.ptp(t_win) > 0:
            slope, intercept = np.polyfit(t_win, y_log_win, 1)
            y_log_pred = slope * t_win + intercept
            rmse = compute_rmse(y_log_win, y_log_pred)
        else:
            rmse = np.nan

    elif umax_method == "spline":
        # For spline, refit and calculate RMSE
        y_log_exp = np.log(y_exp)
        s = spline_s if spline_s is not None else len(t_exp) * 0.1
        try:
            spline = UnivariateSpline(t_exp, y_log_exp, s=s, k=3)
            y_log_pred = spline(t_exp)
            rmse = compute_rmse(y_log_exp, y_log_pred)
        except Exception:
            rmse = np.nan

    else:
        rmse = np.nan

    # Doubling time: t_d = ln(2) / mu
    doubling_time = np.log(2) / mu_max if mu_max > 0 else np.nan

    return {
        "max_od": max_od,
        "specific_growth_rate": mu_max,
        "doubling_time": float(doubling_time),
        "exp_phase_start": float(lag_end),
        "exp_phase_end": float(exp_end),
        "time_at_umax": time_at_umax,
        "od_at_umax": od_at_umax,
        "t_window_start": t_window_start,
        "t_window_end": t_window_end,
        "fit_method": fit_method,
        "model_rmse": rmse,
    }
