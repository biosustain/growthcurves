"""Utility functions for growth curve analysis.

This module provides utility functions for data validation, smoothing,
derivative calculations, and RMSE computation.
"""

import numpy as np
from scipy.signal import savgol_filter

from .models import gaussian

# -----------------------------------------------------------------------------
# Constants
# -----------------------------------------------------------------------------

no_fit_dictionary = {
    "max_od": 0.0,
    "specific_growth_rate": 0.0,
    "doubling_time": np.nan,
    "exp_phase_start": np.nan,
    "exp_phase_end": np.nan,
    "time_at_umax": np.nan,
    "od_at_umax": np.nan,
    "fit_t_min": np.nan,
    "fit_t_max": np.nan,
    "fit_method": None,
    "model_rmse": np.nan,
}


# -----------------------------------------------------------------------------
# Utility Functions
# -----------------------------------------------------------------------------


def validate_data(t, y, min_points=10):
    """
    Validate and clean input data.

    Returns:
        Tuple of (t, y) arrays with finite values only, or (None, None) if invalid.
    """
    t = np.asarray(t, dtype=float)
    y = np.asarray(y, dtype=float)

    mask = np.isfinite(t) & np.isfinite(y) & (y > 0)
    t, y = t[mask], y[mask]

    if len(t) < min_points or np.ptp(t) <= 0:
        return None, None

    return t, y


def compute_rmse(y_observed, y_predicted):
    """Calculate root mean square error between observed and predicted values."""
    mask = np.isfinite(y_observed) & np.isfinite(y_predicted)
    if mask.sum() == 0:
        return np.nan
    residuals = y_observed[mask] - y_predicted[mask]
    return float(np.sqrt(np.mean(residuals**2)))


def calculate_specific_growth_rate(t, y_fit):
    """
    Calculate the maximum specific growth rate from a fitted curve.

    The specific growth rate is μ = (1/N) * dN/dt = d(ln(N))/dt

    Parameters:
        t: Time array
        y_fit: Fitted OD values

    Returns:
        Maximum specific growth rate (time^-1)
    """
    # Use dense time points for accurate derivative
    t_dense = np.linspace(t.min(), t.max(), 1000)
    y_interp = np.interp(t_dense, t, y_fit)

    # Calculate specific growth rate: μ = (1/N) * dN/dt
    dN_dt = np.gradient(y_interp, t_dense)

    # Avoid division by very small values
    y_safe = np.maximum(y_interp, 1e-10)
    mu = dN_dt / y_safe

    # Return maximum specific growth rate
    return float(np.max(mu))


def smooth(y, window=11, poly=1, passes=2):
    """Apply Savitzky-Golay smoothing filter."""
    n = len(y)
    if n < 7:
        return y
    w = int(window) | 1  # Ensure odd
    w = min(w, n if n % 2 else n - 1)
    p = min(int(poly), w - 1)
    for _ in range(passes):
        y = savgol_filter(y, w, p, mode="interp")
    return y


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
        from scipy.optimize import curve_fit

        params, _ = curve_fit(
            gaussian, t_fit, dy_fit, p0=p0, bounds=bounds, maxfev=20000
        )
        return gaussian(t_dense, *params)
    except Exception:
        return np.interp(t_dense, t_fit, dy_fit, left=0.0, right=0.0)


def calculate_phase_ends(t, y, lag_frac=0.15, exp_frac=0.15):
    """
    Calculate lag and exponential phase end times from specific growth rate.

    Parameters:
        t: Time array
        y: OD values (should be from fitted/idealized curve)
        lag_frac: Fraction of peak specific growth rate (μ_max) for lag phase end detection
        exp_frac: Fraction of peak specific growth rate (μ_max) for exponential phase end detection

    Returns:
        Tuple of (lag_end, exp_end) times.
    """
    if len(t) < 5 or np.ptp(t) <= 0:
        return float(t[0]) if len(t) > 0 else np.nan, (
            float(t[-1]) if len(t) > 0 else np.nan
        )

    # Calculate specific growth rate using existing function
    _, mu = compute_specific_growth_rate(t, y)
    mu = np.nan_to_num(mu, nan=0.0)  # Replace NaN with 0
    mu = np.maximum(mu, 0)  # Only consider positive growth

    peak_idx = np.argmax(mu)
    peak_val = mu[peak_idx]

    if peak_val <= 0:
        return float(t[0]), float(t[-1])

    lag_threshold = lag_frac * peak_val
    exp_threshold = exp_frac * peak_val

    lag_end = float(t[0])
    above_lag = mu >= lag_threshold
    if np.any(above_lag):
        first_idx = int(np.argmax(above_lag))
        if first_idx > 0:
            t0, t1 = t[first_idx - 1], t[first_idx]
            y0, y1 = mu[first_idx - 1], mu[first_idx]
            frac = 0.0 if y1 == y0 else (lag_threshold - y0) / (y1 - y0)
            lag_end = float(t0 + frac * (t1 - t0))
        else:
            lag_end = float(t[first_idx])

    exp_end = float(t[-1])
    after_peak = np.arange(len(t)) > peak_idx
    below_exp = mu <= exp_threshold
    exp_candidates = np.where(after_peak & below_exp)[0]
    if len(exp_candidates) > 0:
        idx = int(exp_candidates[0])
        if idx > 0:
            t0, t1 = t[idx - 1], t[idx]
            y0, y1 = mu[idx - 1], mu[idx]
            frac = 0.0 if y1 == y0 else (exp_threshold - y0) / (y1 - y0)
            exp_end = float(t0 + frac * (t1 - t0))
        else:
            exp_end = float(t[idx])

    return lag_end, max(exp_end, lag_end)


# -----------------------------------------------------------------------------
# Growth Statistics Extraction
# -----------------------------------------------------------------------------


def extract_stats_from_fit(fit_result, t, y, lag_frac=0.15, exp_frac=0.15):
    """
    Extract growth statistics from parametric or non-parametric fit results.

    Parameters:
        fit_result: Dict from fit_* functions (contains 'params' and 'model_type')
        t: Time array (hours) used for fitting
        y: OD values used for fitting
        lag_frac: Fraction of peak growth rate for lag phase detection
        exp_frac: Fraction of peak growth rate for exponential phase end detection

    Returns:
        Growth statistics dictionary.
    """
    if fit_result is None:
        return bad_fit_stats()

    t = np.asarray(t, dtype=float)
    y = np.asarray(y, dtype=float)

    model_type = fit_result.get("model_type")
    params = fit_result.get("params", {})

    if model_type in {"logistic", "gompertz", "richards", "baranyi"}:
        from .models import (
            baranyi_model,
            gompertz_model,
            logistic_model,
            richards_model,
        )

        if model_type == "logistic":
            y_fit = logistic_model(
                t, params["K"], params["y0"], params["r"], params["t0"]
            )
        elif model_type == "gompertz":
            y_fit = gompertz_model(
                t, params["K"], params["y0"], params["mu_max_param"], params["lam"]
            )
        elif model_type == "baranyi":
            y_fit = baranyi_model(
                t, params["K"], params["y0"], params["mu_max_param"], params["h0"]
            )
        elif model_type == "richards":
            y_fit = richards_model(
                t, params["K"], params["y0"], params["r"], params["t0"], params["nu"]
            )
        else:
            return bad_fit_stats()

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
                t_dense,
                params["K"],
                params["y0"],
                params["mu_max_param"],
                params["lam"],
            )
        elif model_type == "baranyi":
            y_dense = baranyi_model(
                t_dense,
                params["K"],
                params["y0"],
                params["mu_max_param"],
                params["h0"],
            )
        else:
            y_dense = richards_model(
                t_dense,
                params["K"],
                params["y0"],
                params["r"],
                params["t0"],
                params["nu"],
            )

        # Maximum OD (carrying capacity from fit)
        max_od = float(params["K"])

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

        # Phase boundaries based on specific growth rate thresholds
        exp_phase_start, exp_phase_end = calculate_phase_ends(
            t_dense, y_dense, lag_frac, exp_frac
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
            "fit_t_min": float(t.min()),
            "fit_t_max": float(t.max()),
            "fit_method": f"model_fitting_{model_type}",
            "model_rmse": rmse,
        }

    if model_type in {"sliding_window", "spline"}:
        from .models import spline_model

        valid_mask = np.isfinite(t) & np.isfinite(y) & (y > 0)
        t_clean = t[valid_mask]
        y_clean = y[valid_mask]

        if len(t_clean) < 5 or np.ptp(t_clean) <= 0:
            return bad_fit_stats()

        y_smooth = smooth(y_clean, 11, 1)

        t_dense = np.linspace(t_clean.min(), t_clean.max(), 500)
        y_dense = np.interp(t_dense, t_clean, y_smooth)
        exp_start, exp_end = calculate_phase_ends(
            t_dense, y_dense, lag_frac, exp_frac
        )

        if model_type == "sliding_window":
            mu_max = params["slope"]
        else:
            exp_mask = (t_clean >= exp_start) & (t_clean <= exp_end)
            t_exp = t_clean[exp_mask]
            y_exp = y_clean[exp_mask]

            if len(t_exp) < 5:
                return bad_fit_stats()

            y_log_exp = np.log(y_exp)
            spline_s = params.get("spline_s", len(t_exp) * 0.1)

            spline, _ = spline_model(t_exp, y_log_exp, spline_s, k=3)
            mu_max = float(spline.derivative()(params["time_at_umax"]))

        doubling_time = np.log(2) / mu_max if mu_max > 0 else np.nan
        max_od = float(np.max(y_clean))
        time_at_umax = params["time_at_umax"]
        od_at_umax = float(np.interp(time_at_umax, t, y))

        if model_type == "sliding_window":
            window_points = params.get("window_points", 15)
            t_window_start = params.get("fit_t_min", np.nan)
            t_window_end = params.get("fit_t_max", np.nan)

            window_centers = []
            window_indices = []
            for i in range(len(t_clean) - window_points + 1):
                window_centers.append(float(t_clean[i : i + window_points].mean()))
                window_indices.append(i)

            if len(window_centers) > 0:
                best_idx = min(
                    range(len(window_centers)),
                    key=lambda i: abs(window_centers[i] - time_at_umax),
                )

                start_idx = window_indices[best_idx]
                t_window = t_clean[start_idx : start_idx + window_points]
                y_window = y_clean[start_idx : start_idx + window_points]
                if len(t_window) > 0:
                    t_window_start = float(t_window.min())
                    t_window_end = float(t_window.max())

                slope = params["slope"]
                intercept = params["intercept"]
                y_fit_window = np.exp(slope * t_window + intercept)

                mask = (
                    (y_window > 0)
                    & np.isfinite(y_window)
                    & np.isfinite(y_fit_window)
                    & (y_fit_window > 0)
                )
                if mask.sum() > 0:
                    # RMSE in log space (sliding window fits linear model in log space)
                    y_log = np.log(y_window[mask])
                    y_fit_log = np.log(y_fit_window[mask])
                    rmse = float(np.sqrt(np.mean((y_log - y_fit_log) ** 2)))
                else:
                    rmse = np.nan
            else:
                rmse = np.nan

        else:
            exp_mask = (t_clean >= exp_start) & (t_clean <= exp_end)
            t_exp = t_clean[exp_mask]
            y_exp = y_clean[exp_mask]

            y_log_exp = np.log(y_exp)
            spline_s = params.get("spline_s", len(t_exp) * 0.1)
            t_window_start = float(np.min(t_exp)) if len(t_exp) > 0 else np.nan
            t_window_end = float(np.max(t_exp)) if len(t_exp) > 0 else np.nan

            try:
                spline, _ = spline_model(t_exp, y_log_exp, spline_s, k=3)
                y_log_fit = spline(t_exp)

                # RMSE in log space (spline fits in log space)
                mask = np.isfinite(y_log_exp) & np.isfinite(y_log_fit)
                if mask.sum() > 0:
                    rmse = float(np.sqrt(np.mean((y_log_exp[mask] - y_log_fit[mask]) ** 2)))
                else:
                    rmse = np.nan
            except Exception:
                rmse = np.nan

        return {
            "max_od": max_od,
            "specific_growth_rate": mu_max,
            "doubling_time": doubling_time,
            "exp_phase_start": exp_start,
            "exp_phase_end": exp_end,
            "time_at_umax": time_at_umax,
            "od_at_umax": od_at_umax,
            "fit_t_min": t_window_start,
            "fit_t_max": t_window_end,
            "fit_method": f"model_fitting_{model_type}",
            "model_rmse": rmse,
        }

    return bad_fit_stats()


# -----------------------------------------------------------------------------
# Derivative Functions
# -----------------------------------------------------------------------------


def first_derivative(t, y):
    """
    Calculate the first derivative (dN/dt) of a curve.

    Parameters:
        t: Time array
        y: Values array (e.g., OD or fitted values)

    Returns:
        Array of first derivative values (same length as input)
    """
    t = np.asarray(t, dtype=float)
    y = np.asarray(y, dtype=float)
    return np.gradient(y, t)


def second_derivative(t, y):
    """
    Calculate the second derivative (d²N/dt²) of a curve.

    Parameters:
        t: Time array
        y: Values array (e.g., OD or fitted values)

    Returns:
        Array of second derivative values (same length as input)
    """
    t = np.asarray(t, dtype=float)
    y = np.asarray(y, dtype=float)
    dy_dt = np.gradient(y, t)
    return np.gradient(dy_dt, t)


def bad_fit_stats():
    """Return default stats for failed fits."""
    return no_fit_dictionary.copy()


# -----------------------------------------------------------------------------
# No-Growth Detection
# -----------------------------------------------------------------------------


def detect_no_growth(
    t,
    y,
    growth_stats=None,
    min_data_points=5,
    min_signal_to_noise=5.0,
    min_od_increase=0.05,
    min_growth_rate=1e-6,
):
    """
    Detect whether a growth curve shows no significant growth.

    Performs multiple checks to determine if a well should be marked as "no growth":
    1. Insufficient data points
    2. Low signal-to-noise ratio (max/min OD ratio)
    3. Insufficient OD increase (flat curve)
    4. Zero or near-zero growth rate (from fitted stats)

    Parameters:
        t: Time array
        y: OD values (baseline-corrected)
        growth_stats: Optional dict of fitted growth statistics
            (from extract_stats_from_fit or sliding_window_fit).
            If provided, growth rate is checked.
        min_data_points: Minimum number of valid data points required (default: 5)
        min_signal_to_noise: Minimum ratio of max/min OD values (default: 5.0)
        min_od_increase: Minimum absolute OD increase required (default: 0.05)
        min_growth_rate: Minimum specific growth rate to be considered growth
                         (default: 1e-6)

    Returns:
        Dict with:
            - is_no_growth: bool, True if no growth detected
            - reason: str, description of why it was flagged (or "growth detected")
            - checks: dict with individual check results
    """
    t = np.asarray(t, dtype=float)
    y = np.asarray(y, dtype=float)

    # Filter to finite positive values
    mask = np.isfinite(t) & np.isfinite(y) & (y > 0)
    t_valid = t[mask]
    y_valid = y[mask]

    checks = {
        "has_sufficient_data": True,
        "has_sufficient_snr": True,
        "has_sufficient_od_increase": True,
        "has_positive_growth_rate": True,
    }

    # Check 1: Minimum data points
    if len(t_valid) < min_data_points:
        checks["has_sufficient_data"] = False
        return {
            "is_no_growth": True,
            "reason": f"Insufficient data points ({len(t_valid)} < {min_data_points})",
            "checks": checks,
        }

    # Check 2: Signal-to-noise ratio (max/min OD)
    y_min = np.min(y_valid)
    y_max = np.max(y_valid)

    if y_min > 0:
        snr = y_max / y_min
    else:
        snr = np.inf if y_max > 0 else 0.0

    if snr < min_signal_to_noise:
        checks["has_sufficient_snr"] = False
        return {
            "is_no_growth": True,
            "reason": f"Low signal-to-noise ratio ({snr:.2f} < {min_signal_to_noise})",
            "checks": checks,
        }

    # Check 3: Minimum OD increase (detects flat curves)
    od_increase = y_max - y_min
    if od_increase < min_od_increase:
        checks["has_sufficient_od_increase"] = False
        return {
            "is_no_growth": True,
            "reason": (
                f"Insufficient OD increase ({od_increase:.4f} < {min_od_increase})"
            ),
            "checks": checks,
        }

    # Check 4: Growth rate from fitted statistics (if provided)
    if growth_stats is not None:
        mu = growth_stats.get("specific_growth_rate")
        if mu is None or not np.isfinite(mu) or mu < min_growth_rate:
            checks["has_positive_growth_rate"] = False
            mu_str = f"{mu:.6f}" if mu is not None and np.isfinite(mu) else "N/A"
            return {
                "is_no_growth": True,
                "reason": f"Zero or negative growth rate (μ = {mu_str})",
                "checks": checks,
            }

    return {
        "is_no_growth": False,
        "reason": "growth detected",
        "checks": checks,
    }


def is_no_growth(growth_stats):
    """
    Simple check if growth stats indicate no growth (failed or missing fit).

    This is a convenience function for quick checks on growth_stats dicts.
    For more comprehensive checks including raw data analysis, use detect_no_growth().

    Parameters:
        growth_stats: Dict from extract_stats_from_fit or sliding_window_fit

    Returns:
        bool: True if no growth detected (empty stats or zero growth rate)
    """
    if not growth_stats:
        return True
    mu = growth_stats.get("specific_growth_rate", 0.0)
    return mu is None or mu == 0.0


# -----------------------------------------------------------------------------
# Derivative and Growth Rate Calculation Functions
# -----------------------------------------------------------------------------


def compute_first_derivative(t, y):
    """
    Compute the first derivative of a growth curve.

    Parameters
    ----------
    t : array_like
        Time array
    y : array_like
        OD600 values (baseline-corrected)

    Returns
    -------
    tuple of (np.ndarray, np.ndarray)
        Tuple of (t, dy) where dy is the first derivative dy/dt
    """
    t = np.asarray(t, dtype=float)
    y = np.asarray(y, dtype=float)
    dy = np.gradient(y, t)
    return t, dy


def compute_second_derivative(t, y):
    """
    Compute the second derivative of a growth curve.

    Parameters
    ----------
    t : array_like
        Time array
    y : array_like
        OD600 values (baseline-corrected)

    Returns
    -------
    tuple of (np.ndarray, np.ndarray)
        Tuple of (t, d2y) where d2y is the second derivative d²y/dt²
    """
    t = np.asarray(t, dtype=float)
    y = np.asarray(y, dtype=float)
    dy = np.gradient(y, t)
    d2y = np.gradient(dy, t)
    return t, d2y


def compute_specific_growth_rate(t, y):
    """
    Compute the instantaneous specific growth rate (μ = 1/N × dN/dt).

    The specific growth rate μ represents the rate of population growth per unit
    of population. It is calculated as μ = (1/y) × dy/dt.

    Parameters
    ----------
    t : array_like
        Time array
    y : array_like
        OD600 values (baseline-corrected)

    Returns
    -------
    tuple of (np.ndarray, np.ndarray)
        Tuple of (t, mu) where mu is the specific growth rate μ = (1/y) × dy/dt
    """
    t = np.asarray(t, dtype=float)
    y = np.asarray(y, dtype=float)
    dy = np.gradient(y, t)

    # Avoid division by zero - set mu to nan where y is too small
    mu = np.where(np.abs(y) > 1e-10, dy / y, np.nan)

    return t, mu


def compute_sliding_window_growth_rate(t, y, window_points=15):
    """
    Compute instantaneous specific growth rate using a sliding window approach.

    For each time point, fits a linear regression to log(y) vs t in a window
    centered at that point. The slope of the regression is the instantaneous
    specific growth rate μ at that time.

    This method is more robust to noise than direct differentiation but requires
    more data points.

    Parameters
    ----------
    t : array_like
        Time array
    y : array_like
        OD600 values (baseline-corrected, must be positive)
    window_points : int, optional
        Number of points in each sliding window (default: 15)

    Returns
    -------
    tuple of (np.ndarray, np.ndarray)
        Tuple of (t_out, mu_out) where mu_out is the sliding window growth rate.
        Returns arrays with NaN for points where window fitting failed.
    """
    t = np.asarray(t, dtype=float)
    y = np.asarray(y, dtype=float)

    if len(t) < window_points:
        return t, np.full_like(t, np.nan)

    # Compute log(y), handling non-positive values
    y_log = np.where(y > 0, np.log(y), np.nan)

    mu = np.full_like(t, np.nan)
    half_window = window_points // 2

    for i in range(len(t)):
        # Define window boundaries
        start = max(0, i - half_window)
        end = min(len(t), i + half_window + 1)

        # Ensure window has enough points
        if end - start < max(3, window_points // 2):
            continue

        t_win = t[start:end]
        y_log_win = y_log[start:end]

        # Skip if we have NaN values in the window
        valid_mask = np.isfinite(y_log_win)
        if valid_mask.sum() < 3:
            continue

        t_win_valid = t_win[valid_mask]
        y_log_win_valid = y_log_win[valid_mask]

        # Fit linear regression: log(y) = slope * t + intercept
        # slope = μ (specific growth rate)
        try:
            if np.ptp(t_win_valid) > 0:
                slope, _ = np.polyfit(t_win_valid, y_log_win_valid, 1)
                if np.isfinite(slope):
                    mu[i] = slope
        except (np.linalg.LinAlgError, ValueError):
            continue

    return t, mu
