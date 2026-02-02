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
    "intrinsic_growth_rate": None,
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


def compute_rmse(y_observed, y_predicted, in_log_space=False):
    """
    Calculate root mean square error between observed and predicted values.

    Parameters:
        y_observed: Observed values
        y_predicted: Predicted values
        in_log_space: If True, compute RMSE in log space (default: False)

    Returns:
        RMSE value (float), or np.nan if no valid data points

    Note:
        - Parametric models use in_log_space=False (linear space)
        - Non-parametric models (spline, sliding window) use in_log_space=False
          when data is already log-transformed
    """
    mask = np.isfinite(y_observed) & np.isfinite(y_predicted)
    if mask.sum() == 0:
        return np.nan

    obs = y_observed[mask]
    pred = y_predicted[mask]

    # Apply log transformation if requested and values are positive
    if in_log_space:
        if np.any(obs <= 0) or np.any(pred <= 0):
            return np.nan
        obs = np.log(obs)
        pred = np.log(pred)

    residuals = obs - pred
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


def _linear_interpolate_crossing(t, values, threshold, search_condition):
    """
    Find time at which values cross a threshold using linear interpolation.

    This helper function eliminates duplication in phase boundary calculation.

    Parameters:
        t: Time array
        values: Value array (e.g., specific growth rate)
        threshold: Threshold value to find crossing for
        search_condition: Boolean array indicating where to search for crossing

    Returns:
        Float time at threshold crossing, or None if no crossing found
    """
    if not np.any(search_condition):
        return None

    crossing_idx = int(np.argmax(search_condition))

    if crossing_idx == 0:
        return float(t[crossing_idx])

    # Linear interpolation between points
    t0, t1 = t[crossing_idx - 1], t[crossing_idx]
    v0, v1 = values[crossing_idx - 1], values[crossing_idx]

    if v1 == v0:
        frac = 0.0
    else:
        frac = (threshold - v0) / (v1 - v0)

    return float(t0 + frac * (t1 - t0))


def calculate_phase_ends(t, y, lag_frac=0.15, exp_frac=0.15):
    """
    Calculate lag and exponential phase end times from specific growth rate.

    Parameters:
        t: Time array
        y: OD values (should be from fitted/idealized curve)
        lag_frac: Fraction of μ_max for lag phase end detection
        exp_frac: Fraction of μ_max for exponential phase end detection

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

    # Find lag phase end (first crossing of lag threshold)
    lag_end = _linear_interpolate_crossing(t, mu, lag_threshold, mu >= lag_threshold)
    if lag_end is None:
        lag_end = float(t[0])

    # Find exp phase end (first crossing below exp threshold after peak)
    after_peak = np.arange(len(t)) > peak_idx
    exp_end = _linear_interpolate_crossing(
        t, mu, exp_threshold, after_peak & (mu <= exp_threshold)
    )
    if exp_end is None:
        exp_end = float(t[-1])

    return lag_end, max(exp_end, lag_end)


def calculate_phase_boundaries_tangent(
    t, y, time_at_umax, od_at_umax, mu_max, baseline_od=None, plateau_od=None
):
    """
    Calculate exponential phase boundaries using tangent line method.

    This method extends the tangent line at the point of maximum growth rate
    (μ_max) down to the baseline (lag phase) and up to the plateau (stationary
    phase). The intersection points define the start and end of exponential phase.

    At the point of maximum growth rate, the tangent line in linear OD space is:
        OD(t) = od_at_umax * (1 + μ_max * (t - time_at_umax))

    Parameters:
        t: Time array
        y: OD values (should be from fitted/idealized curve)
        time_at_umax: Time at which μ_max occurs
        od_at_umax: OD value at time_at_umax
        mu_max: Maximum specific growth rate (μ_max)
        baseline_od: Baseline OD (lag phase level). If None, uses min(y)
        plateau_od: Plateau OD (stationary phase level). If None, uses max(y)

    Returns:
        Tuple of (exp_phase_start, exp_phase_end) times.

    Note:
        This method is more appropriate for non-parametric fits where the
        exponential phase is well-defined by the tangent at μ_max.
    """
    if mu_max <= 0 or od_at_umax <= 0:
        # Invalid parameters, return full range
        return float(t[0]) if len(t) > 0 else np.nan, (
            float(t[-1]) if len(t) > 0 else np.nan
        )

    # Determine baseline and plateau if not provided
    if baseline_od is None:
        baseline_od = float(np.min(y))
    if plateau_od is None:
        plateau_od = float(np.max(y))

    # Ensure baseline < od_at_umax < plateau
    baseline_od = min(baseline_od, od_at_umax * 0.9)
    plateau_od = max(plateau_od, od_at_umax * 1.1)

    # Tangent line at point of max growth: OD(t) = od_at_umax * (1 + μ_max * (t - time_at_umax))
    # Solving for t when OD = baseline_od:
    #   baseline_od = od_at_umax * (1 + μ_max * (t_start - time_at_umax))
    #   t_start = time_at_umax + (baseline_od/od_at_umax - 1) / μ_max
    t_start = time_at_umax + ((baseline_od / od_at_umax) - 1) / mu_max

    # Solving for t when OD = plateau_od:
    #   plateau_od = od_at_umax * (1 + μ_max * (t_end - time_at_umax))
    #   t_end = time_at_umax + (plateau_od/od_at_umax - 1) / μ_max
    t_end = time_at_umax + ((plateau_od / od_at_umax) - 1) / mu_max

    # Constrain to data range
    t_start = max(float(t[0]), t_start)
    t_end = min(float(t[-1]), t_end)

    # Ensure start < end
    if t_start >= t_end:
        # Fallback: use time_at_umax as midpoint
        t_range = float(t[-1] - t[0])
        t_start = max(float(t[0]), time_at_umax - 0.3 * t_range)
        t_end = min(float(t[-1]), time_at_umax + 0.3 * t_range)

    return float(t_start), float(t_end)


def calculate_phase_boundaries(
    t,
    y,
    method="tangent",
    time_at_umax=None,
    od_at_umax=None,
    mu_max=None,
    baseline_od=None,
    plateau_od=None,
    lag_frac=0.15,
    exp_frac=0.15,
):
    """
    Calculate exponential phase boundaries using specified method.

    This unified function allows choosing between different methods for
    determining the start and end of the exponential growth phase.

    Parameters:
        t: Time array
        y: OD values (should be from fitted/idealized curve)
        method: Method to use for phase boundary calculation:
            - "tangent": Tangent line method (requires time_at_umax, od_at_umax, mu_max)
            - "threshold": Threshold-based method using fractions of μ_max
        time_at_umax: Time at which μ_max occurs (required for "tangent" method)
        od_at_umax: OD value at time_at_umax (required for "tangent" method)
        mu_max: Maximum specific growth rate (required for "tangent" method)
        baseline_od: Baseline OD for tangent method (defaults to min(y))
        plateau_od: Plateau OD for tangent method (defaults to max(y))
        lag_frac: Fraction of μ_max for lag phase end (threshold method, default: 0.15)
        exp_frac: Fraction of μ_max for exp phase end (threshold method, default: 0.15)

    Returns:
        Tuple of (exp_phase_start, exp_phase_end) times.

    Examples:
        >>> # Tangent method (for non-parametric fits)
        >>> exp_start, exp_end = calculate_phase_boundaries(
        ...     t, y, method="tangent",
        ...     time_at_umax=5.0, od_at_umax=0.5, mu_max=0.8
        ... )
        >>> # Threshold method (for parametric fits)
        >>> exp_start, exp_end = calculate_phase_boundaries(
        ...     t, y, method="threshold", lag_frac=0.15, exp_frac=0.15
        ... )
    """
    if method == "tangent":
        if time_at_umax is None or od_at_umax is None or mu_max is None:
            raise ValueError(
                "Tangent method requires time_at_umax, od_at_umax, and mu_max"
            )
        return calculate_phase_boundaries_tangent(
            t, y, time_at_umax, od_at_umax, mu_max, baseline_od, plateau_od
        )
    elif method == "threshold":
        return calculate_phase_ends(t, y, lag_frac, exp_frac)
    else:
        raise ValueError(
            f"Unknown method '{method}'. Choose 'tangent' or 'threshold'."
        )


# -----------------------------------------------------------------------------
# Growth Statistics Extraction
# -----------------------------------------------------------------------------


def _extract_stats_logistic(
    fit_result, t, y, lag_frac=0.15, exp_frac=0.15, phase_boundary_method="threshold"
):
    """
    Extract growth statistics from logistic model fit.

    Logistic model: N(t) = y0 + (K - y0) / [1 + exp(-r * (t - t0))]
    - r: intrinsic growth rate (equals mu_max for logistic)
    - t0: inflection time
    - K: carrying capacity

    Parameters:
        fit_result: Dict containing 'params' with K, y0, r, t0
        t: Time array
        y: OD values
        lag_frac, exp_frac: Phase detection thresholds
        phase_boundary_method: "threshold" or "tangent"

    Returns:
        Growth statistics dictionary.
    """
    from .models import evaluate_parametric_model

    params = fit_result.get("params", {})

    # Extract model parameters
    K = float(params["K"])  # Carrying capacity
    r = float(params["r"])  # Intrinsic growth rate (equals mu_max for logistic)
    t0 = float(params.get("t0", np.nan))  # Inflection time

    # Evaluate model
    y_fit = evaluate_parametric_model(t, "logistic", params)
    mu_max = calculate_specific_growth_rate(t, y_fit)

    # Dense grid for accurate calculations
    t_dense = np.linspace(t.min(), t.max(), 500)
    y_dense = evaluate_parametric_model(t_dense, "logistic", params)

    # Calculate specific growth rate curve
    y_safe = np.maximum(y_dense, 1e-10)
    mu_dense = np.gradient(np.log(y_safe), t_dense)

    # Find time of maximum specific growth rate
    max_mu_idx = int(np.argmax(mu_dense))
    time_at_umax = float(t_dense[max_mu_idx])
    od_at_umax = float(y_dense[max_mu_idx])

    if mu_max <= 0:
        stats = bad_fit_stats()
        stats["max_od"] = K
        stats["intrinsic_growth_rate"] = r
        stats["fit_method"] = "model_fitting_logistic"
        return stats

    # Phase boundaries using specified method
    exp_phase_start, exp_phase_end = calculate_phase_boundaries(
        t_dense,
        y_dense,
        method=phase_boundary_method,
        time_at_umax=time_at_umax,
        od_at_umax=od_at_umax,
        mu_max=mu_max,
        baseline_od=float(np.min(y_dense)),
        plateau_od=K,
        lag_frac=lag_frac,
        exp_frac=exp_frac,
    )

    # Doubling time from mu_max
    doubling_time = np.log(2) / mu_max if mu_max > 0 else np.nan

    # RMSE
    rmse = compute_rmse(y, y_fit)

    return {
        "max_od": K,
        "specific_growth_rate": float(mu_max),
        "intrinsic_growth_rate": r,
        "doubling_time": float(doubling_time),
        "exp_phase_start": exp_phase_start,
        "exp_phase_end": max(exp_phase_end, exp_phase_start),
        "time_at_umax": time_at_umax,
        "od_at_umax": od_at_umax,
        "fit_t_min": float(t.min()),
        "fit_t_max": float(t.max()),
        "fit_method": "model_fitting_logistic",
        "model_rmse": rmse,
    }


def _extract_stats_gompertz(
    fit_result, t, y, lag_frac=0.15, exp_frac=0.15, phase_boundary_method="threshold"
):
    """
    Extract growth statistics from Gompertz model fit.

    Gompertz model: N(t) = y0 + (K - y0) * exp{-exp[(mu_max * e / (K - y0)) * (lam - t) + 1]}
    - mu_max: maximum specific growth rate (intrinsic parameter)
    - lam: lag time
    - K: carrying capacity

    Parameters:
        fit_result: Dict containing 'params' with K, y0, mu_max, lam
        t: Time array
        y: OD values
        lag_frac, exp_frac: Phase detection thresholds
        phase_boundary_method: "threshold" or "tangent"

    Returns:
        Growth statistics dictionary.
    """
    from .models import evaluate_parametric_model

    params = fit_result.get("params", {})

    # Extract model parameters
    K = float(params["K"])  # Carrying capacity
    mu_max_intrinsic = float(params["mu_max"])  # Intrinsic growth rate parameter
    lam = float(params.get("lam", np.nan))  # Lag time

    # Evaluate model
    y_fit = evaluate_parametric_model(t, "gompertz", params)
    mu_max_observed = calculate_specific_growth_rate(t, y_fit)

    # Dense grid for accurate calculations
    t_dense = np.linspace(t.min(), t.max(), 500)
    y_dense = evaluate_parametric_model(t_dense, "gompertz", params)

    # Calculate specific growth rate curve
    y_safe = np.maximum(y_dense, 1e-10)
    mu_dense = np.gradient(np.log(y_safe), t_dense)

    # Find time of maximum specific growth rate
    max_mu_idx = int(np.argmax(mu_dense))
    time_at_umax = float(t_dense[max_mu_idx])
    od_at_umax = float(y_dense[max_mu_idx])

    if mu_max_observed <= 0:
        stats = bad_fit_stats()
        stats["max_od"] = K
        stats["intrinsic_growth_rate"] = mu_max_intrinsic
        stats["fit_method"] = "model_fitting_gompertz"
        return stats

    # Phase boundaries using specified method
    exp_phase_start, exp_phase_end = calculate_phase_boundaries(
        t_dense,
        y_dense,
        method=phase_boundary_method,
        time_at_umax=time_at_umax,
        od_at_umax=od_at_umax,
        mu_max=mu_max_observed,
        baseline_od=float(np.min(y_dense)),
        plateau_od=K,
        lag_frac=lag_frac,
        exp_frac=exp_frac,
    )

    # Use model's lag parameter if available and reasonable (only for threshold method)
    if phase_boundary_method == "threshold" and np.isfinite(lam) and lam > 0:
        exp_phase_start = float(lam)

    # Doubling time from observed mu_max
    doubling_time = np.log(2) / mu_max_observed if mu_max_observed > 0 else np.nan

    # RMSE
    rmse = compute_rmse(y, y_fit)

    return {
        "max_od": K,
        "specific_growth_rate": float(mu_max_observed),
        "intrinsic_growth_rate": mu_max_intrinsic,
        "doubling_time": float(doubling_time),
        "exp_phase_start": exp_phase_start,
        "exp_phase_end": max(exp_phase_end, exp_phase_start),
        "time_at_umax": time_at_umax,
        "od_at_umax": od_at_umax,
        "fit_t_min": float(t.min()),
        "fit_t_max": float(t.max()),
        "fit_method": "model_fitting_gompertz",
        "model_rmse": rmse,
    }


def _extract_stats_richards(
    fit_result, t, y, lag_frac=0.15, exp_frac=0.15, phase_boundary_method="threshold"
):
    """
    Extract growth statistics from Richards model fit.

    Richards model: N(t) = y0 + (K - y0) * [1 + nu * exp(-r * (t - t0))]^(-1/nu)
    - r: growth rate constant (intrinsic parameter)
    - t0: inflection time
    - nu: shape parameter
    - K: carrying capacity
    - Note: mu_max = r / (nu + 1)^(1/nu)

    Parameters:
        fit_result: Dict containing 'params' with K, y0, r, t0, nu
        t: Time array
        y: OD values
        lag_frac, exp_frac: Phase detection thresholds

    Returns:
        Growth statistics dictionary.
    """
    from .models import evaluate_parametric_model

    params = fit_result.get("params", {})

    # Extract model parameters
    K = float(params["K"])  # Carrying capacity
    r = float(params["r"])  # Intrinsic growth rate constant
    nu = float(params.get("nu", 1.0))  # Shape parameter
    t0 = float(params.get("t0", np.nan))  # Inflection time

    # Evaluate model
    y_fit = evaluate_parametric_model(t, "richards", params)
    mu_max_observed = calculate_specific_growth_rate(t, y_fit)

    # Dense grid for accurate calculations
    t_dense = np.linspace(t.min(), t.max(), 500)
    y_dense = evaluate_parametric_model(t_dense, "richards", params)

    # Calculate specific growth rate curve
    y_safe = np.maximum(y_dense, 1e-10)
    mu_dense = np.gradient(np.log(y_safe), t_dense)

    # Find time of maximum specific growth rate
    max_mu_idx = int(np.argmax(mu_dense))
    time_at_umax = float(t_dense[max_mu_idx])
    od_at_umax = float(y_dense[max_mu_idx])

    if mu_max_observed <= 0:
        stats = bad_fit_stats()
        stats["max_od"] = K
        stats["intrinsic_growth_rate"] = r
        stats["fit_method"] = "model_fitting_richards"
        return stats

    # Phase boundaries using specified method
    exp_phase_start, exp_phase_end = calculate_phase_boundaries(
        t_dense,
        y_dense,
        method=phase_boundary_method,
        time_at_umax=time_at_umax,
        od_at_umax=od_at_umax,
        mu_max=mu_max_observed,
        baseline_od=float(np.min(y_dense)),
        plateau_od=K,
        lag_frac=lag_frac,
        exp_frac=exp_frac,
    )

    # Doubling time from observed mu_max
    doubling_time = np.log(2) / mu_max_observed if mu_max_observed > 0 else np.nan

    # RMSE
    rmse = compute_rmse(y, y_fit)

    return {
        "max_od": K,
        "specific_growth_rate": float(mu_max_observed),
        "intrinsic_growth_rate": r,
        "doubling_time": float(doubling_time),
        "exp_phase_start": exp_phase_start,
        "exp_phase_end": max(exp_phase_end, exp_phase_start),
        "time_at_umax": time_at_umax,
        "od_at_umax": od_at_umax,
        "fit_t_min": float(t.min()),
        "fit_t_max": float(t.max()),
        "fit_method": "model_fitting_richards",
        "model_rmse": rmse,
    }


def _extract_stats_baranyi(
    fit_result, t, y, lag_frac=0.15, exp_frac=0.15, phase_boundary_method="threshold"
):
    """
    Extract growth statistics from Baranyi-Roberts model fit.

    Baranyi model: N(t) = K / (1 + ((K - y0) / y0) * exp(-mu_max * A(t)))
    - mu_max: maximum specific growth rate (intrinsic parameter)
    - h0: dimensionless lag parameter
    - K: carrying capacity
    - lag time ≈ h0 / mu_max

    Parameters:
        fit_result: Dict containing 'params' with K, y0, mu_max, h0
        t: Time array
        y: OD values
        lag_frac, exp_frac: Phase detection thresholds

    Returns:
        Growth statistics dictionary.
    """
    from .models import evaluate_parametric_model

    params = fit_result.get("params", {})

    # Extract model parameters
    K = float(params["K"])  # Carrying capacity
    mu_max_intrinsic = float(params["mu_max"])  # Intrinsic growth rate parameter
    h0 = float(params.get("h0", 0.0))  # Dimensionless lag parameter

    # Calculate lag time from h0
    lag_time = h0 / mu_max_intrinsic if mu_max_intrinsic > 0 and h0 > 0 else np.nan

    # Evaluate model
    y_fit = evaluate_parametric_model(t, "baranyi", params)
    mu_max_observed = calculate_specific_growth_rate(t, y_fit)

    # Dense grid for accurate calculations
    t_dense = np.linspace(t.min(), t.max(), 500)
    y_dense = evaluate_parametric_model(t_dense, "baranyi", params)

    # Calculate specific growth rate curve
    y_safe = np.maximum(y_dense, 1e-10)
    mu_dense = np.gradient(np.log(y_safe), t_dense)

    # Find time of maximum specific growth rate
    max_mu_idx = int(np.argmax(mu_dense))
    time_at_umax = float(t_dense[max_mu_idx])
    od_at_umax = float(y_dense[max_mu_idx])

    if mu_max_observed <= 0:
        stats = bad_fit_stats()
        stats["max_od"] = K
        stats["intrinsic_growth_rate"] = mu_max_intrinsic
        stats["fit_method"] = "model_fitting_baranyi"
        return stats

    # Phase boundaries using specified method
    exp_phase_start, exp_phase_end = calculate_phase_boundaries(
        t_dense,
        y_dense,
        method=phase_boundary_method,
        time_at_umax=time_at_umax,
        od_at_umax=od_at_umax,
        mu_max=mu_max_observed,
        baseline_od=float(np.min(y_dense)),
        plateau_od=K,
        lag_frac=lag_frac,
        exp_frac=exp_frac,
    )

    # Use model's lag parameter if available and reasonable (only for threshold method)
    if phase_boundary_method == "threshold" and np.isfinite(lag_time) and lag_time > 0:
        exp_phase_start = float(lag_time)

    # Doubling time from observed mu_max
    doubling_time = np.log(2) / mu_max_observed if mu_max_observed > 0 else np.nan

    # RMSE
    rmse = compute_rmse(y, y_fit)

    return {
        "max_od": K,
        "specific_growth_rate": float(mu_max_observed),
        "intrinsic_growth_rate": mu_max_intrinsic,
        "doubling_time": float(doubling_time),
        "exp_phase_start": exp_phase_start,
        "exp_phase_end": max(exp_phase_end, exp_phase_start),
        "time_at_umax": time_at_umax,
        "od_at_umax": od_at_umax,
        "fit_t_min": float(t.min()),
        "fit_t_max": float(t.max()),
        "fit_method": "model_fitting_baranyi",
        "model_rmse": rmse,
    }


def _extract_stats_parametric(
    fit_result, t, y, lag_frac=0.15, exp_frac=0.15, phase_boundary_method="threshold"
):
    """
    Dispatcher for parametric model statistics extraction.

    This function reads the model type and dispatches to the appropriate
    model-specific extraction function.

    Handles: logistic, gompertz, richards, baranyi models.

    Parameters:
        fit_result: Dict from fit_* functions (contains 'params' and 'model_type')
        t: Time array (hours) used for fitting
        y: OD values used for fitting
        lag_frac: Fraction of peak growth rate for lag phase detection
        exp_frac: Fraction of peak growth rate for exponential phase end detection
        phase_boundary_method: Method for phase boundary calculation ("threshold" or "tangent")

    Returns:
        Growth statistics dictionary.
    """
    model_type = fit_result.get("model_type")

    # Dispatch to model-specific extraction function
    if model_type == "logistic":
        return _extract_stats_logistic(
            fit_result, t, y, lag_frac, exp_frac, phase_boundary_method
        )
    elif model_type == "gompertz":
        return _extract_stats_gompertz(
            fit_result, t, y, lag_frac, exp_frac, phase_boundary_method
        )
    elif model_type == "richards":
        return _extract_stats_richards(
            fit_result, t, y, lag_frac, exp_frac, phase_boundary_method
        )
    elif model_type == "baranyi":
        return _extract_stats_baranyi(
            fit_result, t, y, lag_frac, exp_frac, phase_boundary_method
        )
    else:
        # Unknown parametric model
        return bad_fit_stats()


def _extract_stats_sliding_window(
    fit_result, t, y, lag_frac=0.15, exp_frac=0.15, phase_boundary_method="tangent"
):
    """
    Extract growth statistics from sliding window fits.

    Parameters:
        fit_result: Dict from fit_* functions (contains 'params' and 'model_type')
        t: Time array (hours) used for fitting
        y: OD values used for fitting
        lag_frac: Fraction of peak growth rate for lag phase detection
        exp_frac: Fraction of peak growth rate for exponential phase end detection
        phase_boundary_method: "threshold" or "tangent" (default: "tangent")

    Returns:
        Growth statistics dictionary.
    """
    params = fit_result.get("params", {})

    valid_mask = np.isfinite(t) & np.isfinite(y) & (y > 0)
    t_clean = t[valid_mask]
    y_clean = y[valid_mask]

    if len(t_clean) < 5 or np.ptp(t_clean) <= 0:
        return bad_fit_stats()

    y_smooth = smooth(y_clean, 11, 1)

    mu_max = params["slope"]
    doubling_time = np.log(2) / mu_max if mu_max > 0 else np.nan
    max_od = float(np.max(y_clean))
    time_at_umax = params["time_at_umax"]
    od_at_umax = float(np.interp(time_at_umax, t_clean, y_smooth))

    # Calculate phase boundaries using specified method
    baseline_od = float(np.min(y_clean))
    plateau_od = max_od
    exp_start, exp_end = calculate_phase_boundaries(
        t_clean,
        y_smooth,
        method=phase_boundary_method,
        time_at_umax=time_at_umax,
        od_at_umax=od_at_umax,
        mu_max=mu_max,
        baseline_od=baseline_od,
        plateau_od=plateau_od,
        lag_frac=lag_frac,
        exp_frac=exp_frac,
    )

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

        # RMSE in log space (sliding window fits linear model in log space)
        rmse = compute_rmse(y_window, y_fit_window, in_log_space=True)
    else:
        rmse = np.nan

    return {
        "max_od": max_od,
        "specific_growth_rate": mu_max,
        "intrinsic_growth_rate": None,  # Non-parametric: no intrinsic parameter
        "doubling_time": doubling_time,
        "exp_phase_start": exp_start,
        "exp_phase_end": exp_end,
        "time_at_umax": time_at_umax,
        "od_at_umax": od_at_umax,
        "fit_t_min": t_window_start,
        "fit_t_max": t_window_end,
        "fit_method": "model_fitting_sliding_window",
        "model_rmse": rmse,
    }


def _extract_stats_spline(
    fit_result, t, y, lag_frac=0.15, exp_frac=0.15, phase_boundary_method="tangent"
):
    """
    Extract growth statistics from spline fits.

    Parameters:
        fit_result: Dict from fit_* functions (contains 'params' and 'model_type')
        t: Time array (hours) used for fitting
        y: OD values used for fitting
        lag_frac: Fraction of peak growth rate for lag phase detection
        exp_frac: Fraction of peak growth rate for exponential phase end detection
        phase_boundary_method: "threshold" or "tangent" (default: "tangent")

    Returns:
        Growth statistics dictionary.
    """
    from .models import spline_model

    params = fit_result.get("params", {})

    valid_mask = np.isfinite(t) & np.isfinite(y) & (y > 0)
    t_clean = t[valid_mask]
    y_clean = y[valid_mask]

    if len(t_clean) < 5 or np.ptp(t_clean) <= 0:
        return bad_fit_stats()

    y_smooth = smooth(y_clean, 11, 1)

    # Use threshold method to get initial phase boundaries for spline fitting
    t_dense = np.linspace(t_clean.min(), t_clean.max(), 500)
    y_dense = np.interp(t_dense, t_clean, y_smooth)
    exp_start_initial, exp_end_initial = calculate_phase_ends(
        t_dense, y_dense, lag_frac, exp_frac
    )

    exp_mask = (t_clean >= exp_start_initial) & (t_clean <= exp_end_initial)
    t_exp = t_clean[exp_mask]
    y_exp = y_clean[exp_mask]

    if len(t_exp) < 5:
        return bad_fit_stats()

    y_log_exp = np.log(y_exp)
    spline_s = params.get("spline_s", len(t_exp) * 0.1)

    spline, _ = spline_model(t_exp, y_log_exp, spline_s, k=3)
    time_at_umax = params["time_at_umax"]
    mu_max = float(spline.derivative()(time_at_umax))

    doubling_time = np.log(2) / mu_max if mu_max > 0 else np.nan
    max_od = float(np.max(y_clean))
    od_at_umax = float(np.interp(time_at_umax, t_clean, y_smooth))

    # Calculate refined phase boundaries using specified method
    baseline_od = float(np.min(y_clean))
    plateau_od = max_od
    exp_start, exp_end = calculate_phase_boundaries(
        t_clean,
        y_smooth,
        method=phase_boundary_method,
        time_at_umax=time_at_umax,
        od_at_umax=od_at_umax,
        mu_max=mu_max,
        baseline_od=baseline_od,
        plateau_od=plateau_od,
        lag_frac=lag_frac,
        exp_frac=exp_frac,
    )

    t_window_start = float(np.min(t_exp)) if len(t_exp) > 0 else np.nan
    t_window_end = float(np.max(t_exp)) if len(t_exp) > 0 else np.nan

    try:
        spline, _ = spline_model(t_exp, y_log_exp, spline_s, k=3)
        y_log_fit = spline(t_exp)

        # RMSE in log space (spline fits in log space)
        # Data is already log-transformed, so use in_log_space=False
        rmse = compute_rmse(y_log_exp, y_log_fit, in_log_space=False)
    except Exception:
        rmse = np.nan

    return {
        "max_od": max_od,
        "specific_growth_rate": mu_max,
        "intrinsic_growth_rate": None,  # Non-parametric: no intrinsic parameter
        "doubling_time": doubling_time,
        "exp_phase_start": exp_start,
        "exp_phase_end": exp_end,
        "time_at_umax": time_at_umax,
        "od_at_umax": od_at_umax,
        "fit_t_min": t_window_start,
        "fit_t_max": t_window_end,
        "fit_method": "model_fitting_spline",
        "model_rmse": rmse,
    }


def extract_stats_from_fit(
    fit_result, t, y, lag_frac=0.15, exp_frac=0.15, phase_boundary_method="auto"
):
    """
    Extract growth statistics from parametric or non-parametric fit results.

    This function acts as a dispatcher that reads the model type from the fit_result
    and calls the appropriate model-specific extraction function.

    Parameters:
        fit_result: Dict from fit_* functions (contains 'params' and 'model_type')
        t: Time array (hours) used for fitting
        y: OD values used for fitting
        lag_frac: Fraction of peak growth rate for lag phase detection (threshold method)
        exp_frac: Fraction of peak growth rate for exponential phase end (threshold method)
        phase_boundary_method: Method for calculating phase boundaries:
            - "auto" (default): Use "threshold" for parametric, "tangent" for non-parametric
            - "threshold": Threshold-based method using fractions of μ_max
            - "tangent": Tangent line method at point of maximum growth rate

    Returns:
        Growth statistics dictionary.
    """
    if fit_result is None:
        return bad_fit_stats()

    t = np.asarray(t, dtype=float)
    y = np.asarray(y, dtype=float)

    model_type = fit_result.get("model_type")

    # Determine method based on model type if auto
    if phase_boundary_method == "auto":
        if model_type in {"logistic", "gompertz", "richards", "baranyi"}:
            method = "threshold"
        else:
            method = "tangent"
    else:
        method = phase_boundary_method

    # Dispatch to appropriate extraction function based on model type
    if model_type in {"logistic", "gompertz", "richards", "baranyi"}:
        return _extract_stats_parametric(fit_result, t, y, lag_frac, exp_frac, method)
    elif model_type == "sliding_window":
        return _extract_stats_sliding_window(fit_result, t, y, lag_frac, exp_frac, method)
    elif model_type == "spline":
        return _extract_stats_spline(fit_result, t, y, lag_frac, exp_frac, method)
    else:
        # Unknown or unsupported model type
        return bad_fit_stats()


# -----------------------------------------------------------------------------
# Derivative Functions
# -----------------------------------------------------------------------------


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
