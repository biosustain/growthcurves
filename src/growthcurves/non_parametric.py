"""Non-parametric fitting methods for growth curves.

This module provides non-parametric methods for growth curve analysis,
including sliding window fitting and no-growth detection.

All methods operate in linear OD space (not log-transformed).
"""

import numpy as np
import sklearn.linear_model
from scipy.interpolate import make_smoothing_spline

from .inference import bad_fit_stats

# from scipy.stats import theilslopes


# Default settings for the auto-spline (sigma-based smoothing + OD weights).
_SPLINE_SMOOTH_MULT = 5.0
_SPLINE_GCV_WEIGHT_FLOOR_Q = 0.15
_SPLINE_GCV_WEIGHT_POWER = 1.0

# -----------------------------------------------------------------------------
# Sliding Window Helpers
# -----------------------------------------------------------------------------


# -----------------------------------------------------------------------------
# Umax Calculation Methods
# -----------------------------------------------------------------------------


def fit_sliding_window(t, N, window_points=15, step=None, n_fits=None):
    """
    Calculate maximum specific growth rate using the sliding window method.

    Finds the maximum specific growth rate by fitting a line to log-transformed
    OD N in consecutive windows using the Theil-Sen estimator, selecting the
    window with the steepest slope.

    Parameters:
        t: Time array (hours)
        N: OD values (baseline-corrected, must be positive)
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
    if len(t) < window_points or np.ptp(t) <= 0:
        return None

    # Log-transform for growth rate calculation
    y_log = np.log(N)

    # Find window with maximum slope on log-transformed N
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
            step = max(1, int(len(t) / n_fits))
    huber_regressor = sklearn.linear_model.HuberRegressor()
    # limit number of fits to avoid excessive computation using step parameter
    for i in range(0, len(t) - w + 1, step):
        t_win = t[i : i + w]
        y_log_win = y_log[i : i + w]

        if np.ptp(t_win) <= 0:
            continue

        # # Use Theil-Sen estimator for robust line fitting
        # result = theilslopes(y_log_win, t_win)
        # slope, intercept = result.slope, result.intercept
        # # Use HuberRegressor which uses L2 regularization and is twice as fast as
        # # Theil-Sen.
        result = huber_regressor.fit(t_win.reshape(-1, 1), y_log_win)
        slope, intercept = result.coef_[0], result.intercept_

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


def _second_diff_series(t, y):
    """Compute second divided differences of y w.r.t. t, scaled by median(dt)^2."""
    t = np.asarray(t, dtype=float)
    y = np.asarray(y, dtype=float)
    if len(t) < 4 or np.ptp(t) <= 0:
        return np.array([], dtype=float)
    order = np.argsort(t, kind="mergesort")
    t = t[order]
    y = y[order]
    dt = np.diff(t)
    if len(dt) < 2 or np.any(dt <= 0):
        return np.array([], dtype=float)
    d1 = np.diff(y) / dt
    dt2 = 0.5 * (dt[:-1] + dt[1:])
    if len(dt2) < 1 or np.any(dt2 <= 0):
        return np.array([], dtype=float)
    d2 = np.diff(d1) / dt2
    if len(d2) < 3:
        return np.array([], dtype=float)
    h = float(np.median(dt))
    return np.asarray(d2 * h**2, dtype=float)


def _trim_by_abs_deviation(vals, trim_q=0.8):
    """Trim values with absolute deviation from median above the trim_q quantile."""
    vals = np.asarray(vals, dtype=float)
    vals = vals[np.isfinite(vals)]
    if len(vals) < 3:
        return vals
    q = float(trim_q)
    if not np.isfinite(q) or q <= 0.0 or q >= 1.0:
        return vals
    abs_dev = np.abs(vals - np.median(vals))
    if len(abs_dev) < 5:
        return vals
    cutoff = float(np.quantile(abs_dev, q))
    core = vals[abs_dev <= cutoff]
    if len(core) >= 3:
        return core
    return vals


def _sigma_second_diff_mad(t, y, trim_q=0.8):
    """Estimate log-space noise sigma from second divided differences using MAD."""
    d2y = _second_diff_series(t, y)
    if len(d2y) < 3:
        return np.nan
    d2y = _trim_by_abs_deviation(d2y, trim_q=trim_q)
    if len(d2y) < 3:
        return np.nan
    mad = float(np.median(np.abs(d2y - np.median(d2y))))
    return float(1.4826 * mad / np.sqrt(6.0))


def _prepare_spline_inputs(t, N):
    """Sort, clip, and deduplicate inputs for stable spline fitting in log-space."""
    t = np.asarray(t, dtype=float)
    N = np.asarray(N, dtype=float)

    order = np.argsort(t, kind="mergesort")
    t = t[order]
    N = np.clip(N[order], 1e-12, None)

    uniq_t, uniq_idx = np.unique(t, return_index=True)
    t = uniq_t
    N = N[uniq_idx]
    y_log = np.log(N)

    if len(t) < 5 or np.ptp(t) <= 0:
        raise ValueError("Need >=5 unique time points with non-zero span.")

    return t, N, y_log


def _od_weight_vector(
    N, floor_q=_SPLINE_GCV_WEIGHT_FLOOR_Q, power=_SPLINE_GCV_WEIGHT_POWER
):
    """Build OD-dependent weights to downweight low-OD log-noise leverage."""
    N = np.asarray(N, dtype=float)
    if len(N) == 0:
        return np.asarray([], dtype=float)

    q = float(np.clip(float(floor_q), 0.0, 0.49))
    od_floor = max(float(np.quantile(N, q)), 1e-6)
    N_eff = np.clip(N, od_floor, None)

    p = float(np.clip(float(power), 0.0, 3.0))
    med = float(np.median(N_eff))
    if (not np.isfinite(med)) or (med <= 0):
        med = od_floor

    w = (N_eff / med) ** p
    w = np.clip(w, 0.2, 5.0)

    rms = float(np.sqrt(np.mean(w**2)))
    if (not np.isfinite(rms)) or (rms <= 0):
        return np.ones_like(N_eff, dtype=float)

    return np.asarray(w / rms, dtype=float)


def _fast_auto_lam(t, y_log):
    """Compute notebook-style fast auto smoothing value for make_smoothing_spline."""
    sigma = _sigma_second_diff_mad(t, y_log, trim_q=0.8)
    if not np.isfinite(sigma) or sigma < 1e-8:
        sigma = 1e-4

    n = len(t)
    smooth_mult_eff = float(np.clip(_SPLINE_SMOOTH_MULT, 0.25, 30.0))
    raw_lam = smooth_mult_eff * n * sigma**2
    lam_max = max(1e-8, 0.8 * n * float(np.var(y_log)))
    return float(np.clip(raw_lam, 0.0, lam_max))


def fit_spline(
    t,
    N,
    smooth="fast",
    use_weights=True,
):
    """
    Calculate maximum specific growth rate using spline fitting.

    Fits a smoothing spline to log-transformed OD N and calculates
    the maximum specific growth rate from the spline's derivative.

    Parameters:
        t: Time array (hours)
        N: OD values
        smooth: Smoothing mode/value.
                - "fast": notebook auto-default rule mapped to lam
                - "slow": GCV-selected smoothing (lam=None)
                - float: manual lam value
        use_weights: Whether to apply OD-dependent weighting (default: True)

    Returns:
        Dict with model parameters:
            - t_knots: Spline knot points (t values)
            - spline_coeffs: Spline coefficients
            - spline_k: Spline degree (3)
            - time_at_umax: Time at maximum growth rate (hours)
            - model_type: "spline"
        Returns None if calculation fails.

    """
    if len(t) < 5:
        return None

    try:
        t_fit, N_fit, y_log = _prepare_spline_inputs(t, N)
        if bool(use_weights):
            w = _od_weight_vector(
                N_fit,
                floor_q=_SPLINE_GCV_WEIGHT_FLOOR_Q,
                power=_SPLINE_GCV_WEIGHT_POWER,
            )
        else:
            w = np.ones_like(y_log, dtype=float)

        # Fit spline with selected smoothing mode/value.
        if isinstance(smooth, str):
            smooth_mode = smooth.strip().lower()
            if smooth_mode == "fast":
                lam = _fast_auto_lam(t_fit, y_log)
            elif smooth_mode == "slow":
                lam = None
            else:
                raise ValueError("smooth must be 'fast', 'slow', or numeric.")
            smooth_out = smooth_mode
        else:
            lam = float(smooth)
            if (not np.isfinite(lam)) or (lam < 0):
                raise ValueError(
                    "Manual smooth value must be a finite non-negative float."
                )
            smooth_out = lam

        spline = make_smoothing_spline(t_fit, y_log, w=w, lam=lam)
        if lam is None:
            resid = y_log - np.asarray(spline(t_fit), dtype=float)
            spline_s = float(np.sum((w * resid) ** 2))
        else:
            spline_s = float(lam)

        # Evaluate spline on dense grid for accurate derivative calculation
        t_eval = np.linspace(t_fit.min(), t_fit.max(), 200)

        # Calculate specific growth rate: μ = d(ln(N))/dt
        mu_eval = spline.derivative()(t_eval)

        # Find maximum specific growth rate
        max_mu_idx = int(np.argmax(mu_eval))
        mu_max = float(mu_eval[max_mu_idx])
        time_at_umax = float(t_eval[max_mu_idx])

        if mu_max <= 0 or not np.isfinite(mu_max):
            return None

        # Extract spline parameters for later reconstruction.
        if hasattr(spline, "_eval_args"):
            tck_t, tck_c, tck_k = spline._eval_args
        elif hasattr(spline, "t") and hasattr(spline, "c") and hasattr(spline, "k"):
            tck_t = np.asarray(spline.t, dtype=float)
            tck_c = np.asarray(spline.c, dtype=float)
            tck_k = int(spline.k)
        else:
            return None

        return {
            "params": {
                "tck_t": tck_t.tolist(),
                "tck_c": tck_c.tolist(),
                "tck_k": int(tck_k),
                "smooth": smooth_out,
                "spline_s": float(spline_s),
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
    N,
    method="sliding_window",
    window_points=15,
    smooth="fast",
    use_weights=True,
    **kwargs,
):
    """
    Calculate growth statistics using non-parametric methods.

    This unified function supports multiple methods for calculating the maximum
    specific growth rate (Umax):
    - "sliding_window": Finds maximum slope in log-transformed OD across windows
    - "spline": Fits spline to entire curve and calculates from derivative

    Parameters:
        t: Time array (hours)
        N: OD values (baseline-corrected, must be positive)
        method: Method for calculating Umax ("sliding_window" or "spline")
        window_points: Number of points in sliding window (for sliding_window method)
        smooth: Smoothing mode/value for spline method.
                - "fast": notebook auto-default rule mapped to lam
                - "slow": GCV-selected smoothing (auto GCV)
                - float: manual lam value
        use_weights: Whether to apply OD-dependent weighting for spline method
        **kwargs: Additional arguments (for compatibility)

    Returns:
        Dict containing:
            - params: Model parameters (includes fit_t_min, fit_t_max, and other
                      method-specific values)
            - model_type: Method used for fitting
    """
    t = np.asarray(t, dtype=float)
    N = np.asarray(N, dtype=float)

    # Filter valid N (N must be positive for log transform)
    mask = np.isfinite(t) & np.isfinite(N) & (N > 0)
    t, y_raw = t[mask], N[mask]

    # Check minimum N requirements
    min_points = window_points if method == "sliding_window" else 10
    if len(t) < min_points or np.ptp(t) <= 0:
        return bad_fit_stats()

    # Calculate Umax using specified method
    if method == "sliding_window":
        umax_result = fit_sliding_window(t, y_raw, window_points, **kwargs)

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
        umax_result = fit_spline(
            t,
            y_raw,
            smooth=smooth,
            use_weights=use_weights,
        )

        if umax_result is None:
            return None

        # Store fit_t_min and fit_t_max as the full N range
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
