"""Utility functions for growth curve analysis.

This module provides utility functions for data validation, smoothing,
derivative calculations, and RMSE computation.
"""

import numpy as np
from scipy.signal import savgol_filter


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
    "t_window_start": np.nan,
    "t_window_end": np.nan,
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
        growth_stats: Optional dict of fitted growth statistics (from extract_stats_from_fit
            or sliding_window_fit). If provided, growth rate is checked.
        min_data_points: Minimum number of valid data points required (default: 5)
        min_signal_to_noise: Minimum ratio of max/min OD values (default: 5.0)
        min_od_increase: Minimum absolute OD increase required (default: 0.05)
        min_growth_rate: Minimum specific growth rate to be considered growth (default: 1e-6)

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
            "reason": f"Insufficient OD increase ({od_increase:.4f} < {min_od_increase})",
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
