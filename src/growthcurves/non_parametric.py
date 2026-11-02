"""Non-parametric fitting methods for growth curves.

This module provides the sliding window and spline methods for estimating the
maximum specific growth rate.

Both methods take OD on a linear scale and fit ln(OD).
"""

from logging import getLogger
from typing import Union

import numpy as np
from scipy.interpolate import make_smoothing_spline

from .inference import smooth as smooth_od
from .inference import validate_data

logger = getLogger(__name__)

# Default settings for the auto-spline (sigma-based smoothing + OD weights).
_SPLINE_SMOOTH_MULT = 5.0
_SPLINE_GCV_WEIGHT_FLOOR_Q = 0.15
_SPLINE_GCV_WEIGHT_POWER = 1.0

# -----------------------------------------------------------------------------
# Sliding Window
# -----------------------------------------------------------------------------


# region: sliding window helpers
def _window_lengths(n, w_min, ratio=1.25):
    """Window lengths from w_min to n // 2, each ~ratio x the last."""
    lengths, w = [], w_min
    while w <= max(w_min, n // 2):
        lengths.append(w)
        w = max(w + 1, round(w * ratio))
    return lengths


def _ln_noise_sd(t, N):
    """Per-reading sd of ln N: proportional noise plus an additive OD floor.

    Both levels come from second differences (MAD): additive from N in the lowest
    third of readings, proportional from ln N in the highest. The additive level is
    at least the rounding noise of the data. Each reading uses the smaller of its
    observed and smoothed OD.
    """
    N_ref = smooth_od(N)
    mid = N_ref[1:-1]
    lo, hi = mid <= np.quantile(mid, 1 / 3), mid >= np.quantile(mid, 2 / 3)
    mad_to_sd = 1.4826 / np.sqrt(6)
    sd_add = mad_to_sd * np.median(np.abs(_second_diff_series(t, N))[lo])
    sd_add = max(sd_add, np.diff(np.unique(N)).min() / np.sqrt(12))
    sd_prop = mad_to_sd * np.median(np.abs(_second_diff_series(t, np.log(N)))[hi])
    return np.hypot(sd_prop, sd_add / np.maximum(np.minimum(N, N_ref), 1e-12))


def _wls_windows(t, y, sd, lengths):
    """Weighted least squares of every window of each length, via prefix sums.

    Returns slope, SE of the slope (from sd), reduced chi-square, start, length.
    """
    t, y, w = t - t.mean(), y - y.mean(), 1 / sd**2
    c = [
        np.concatenate([[0], np.cumsum(v)])
        for v in (w, w * t, w * y, w * t * t, w * t * y, w * y * y)
    ]
    out = []
    for L in lengths:
        i = np.arange(len(t) - L + 1)
        sw, st, sy, stt, sty, syy = (v[i + L] - v[i] for v in c)
        sxx, sxy = stt - st**2 / sw, sty - st * sy / sw
        slope = sxy / sxx
        chi2 = np.maximum(syy - sy**2 / sw - slope * sxy, 0) / (L - 2)
        out.append((slope, 1 / np.sqrt(sxx), chi2, i, np.full_like(i, L)))
    return (np.concatenate(a) for a in zip(*out))


# endregion


# region: sliding window method
def fit_sliding_window(t, N, window_points=5):
    """
    Calculate maximum specific growth rate using the sliding window method.

    Fits ln(N) by weighted least squares in windows of every length from
    `window_points` readings to half the curve, weighting readings by a noise model
    estimated from the curve. The window with the highest lower bound, slope - z*SE,
    wins (z = sqrt(2 ln M) for M windows; SE inflated by sqrt(reduced chi-square)
    where a window fits worse than the noise). So the window length adapts: short on
    clean or sparse data, long on noisy data.

    Parameters:
        t: Time array (hours)
        N: OD values (linear-scale, baseline-corrected, must be positive)
        window_points: Minimum readings per window (default: 5, at least 3)

    Returns:
        Dict with params (slope = μmax in h⁻¹, intercept in log space, time_at_umax
        = window midpoint, fit_t_min, fit_t_max) and model_type "sliding_window".
        None if no window's lower bound is above zero (no growth) or too few points.
    """
    if int(window_points) != window_points or window_points < 3:
        raise ValueError("window_points must be an integer of at least 3.")

    t, N = validate_data(t, N)
    if t is None or len(t) < window_points or np.ptp(N) == 0:
        return None

    y, sd = np.log(N), _ln_noise_sd(t, N)
    slope, se, chi2, start, length = _wls_windows(
        t, y, sd, _window_lengths(len(t), int(window_points))
    )
    bound = slope - np.sqrt(2 * np.log(len(slope))) * se * np.sqrt(np.maximum(chi2, 1))
    k = np.nanargmax(bound)
    if not bound[k] > 0:
        return None

    win = slice(start[k], start[k] + length[k])
    w = sd[win] ** -2
    intercept = np.average(y[win], weights=w) - slope[k] * np.average(t[win], weights=w)
    return {
        "params": {
            "slope": float(slope[k]),
            "intercept": float(intercept),
            "time_at_umax": float(t[win].mean()),
            "fit_t_min": float(t[win][0]),
            "fit_t_max": float(t[win][-1]),
        },
        "model_type": "sliding_window",
    }


# endregion

# -----------------------------------------------------------------------------
# Spline
# -----------------------------------------------------------------------------


# region: spline fitting helpers
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


# endregion


# region: fit_spline
def fit_spline(
    t,
    N,
    smooth="fast",
    use_weights=False,
):
    """
    Calculate maximum specific growth rate using spline fitting.

    Fits a smoothing spline to log-transformed OD N and calculates
    the maximum specific growth rate from the spline's derivative.

    Parameters:
        t: Time array (hours)
        N: OD values (linear space, must be positive)
        smooth: Smoothing mode/value.
                - "fast": notebook auto-default rule mapped to lam
                - "slow": GCV-selected smoothing (lam=None)
                - float: manual lam value

        use_weights: Whether to apply OD-dependent weighting (default: False)

    Returns:
        Dict with model parameters:
            - t_knots: Spline knot points (t values)
            - spline_coeffs: Spline coefficients
            - spline_k: Spline degree (3)
            - time_at_umax: Time at maximum growth rate (hours)
            - model_type: "spline"

        Returns None if calculation fails.
    """
    # Sorted, de-duplicated, finite and N > 0 (raises on negative times)
    t_fit, N_fit = validate_data(t, N)
    if t_fit is None:
        return None

    try:
        y_log = np.log(N_fit)
        if bool(use_weights):
            w = _od_weight_vector(
                N_fit,
                floor_q=_SPLINE_GCV_WEIGHT_FLOOR_Q,
                power=_SPLINE_GCV_WEIGHT_POWER,
            )
        else:
            # Default in scipy.interpolate.make_smoothing_spline are uniform weights.
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
            # ? is this correct? it gives the residual sum of squares?
            # ! lam is not returned if not provided by make_smoothing_spline
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


# endregion

# -----------------------------------------------------------------------------
# Main API Functions
# -----------------------------------------------------------------------------


def fit_non_parametric(
    t,
    N,
    method="sliding_window",
    window_points=5,
    smooth: Union[str, float] = "fast",
    use_weights=False,
    **kwargs,
):
    """
    Calculate growth statistics using non-parametric methods.

    This unified function supports multiple methods for calculating the maximum
    specific growth rate (Umax):

    - "sliding_window": Best-supported straight line on log-transformed OD over
      windows of every length (see `fit_sliding_window`)
    - "spline": Fits spline to entire curve and calculates from derivative


    Parameters
    ----------
    t : Iterable[float]
        Time array (hours)
    N : Iterable[float]
        OD values
    method : str, optional
        Method for calculating Umax ("sliding_window" or "spline"),
        by default "sliding_window"
    window_points : int, optional
        Minimum number of readings in a window (sliding_window method), by default 5
    smooth : str or float, optional
        Smoothing mode/value for spline method, by default "fast"
            - "fast": notebook auto-default rule mapped to lam
            - "slow": GCV-selected smoothing (auto GCV)
            - float: manual lam value

    use_weights : bool, optional
        Whether to apply OD-dependent weighting for spline method, by default False
    **kwargs
        Ignored. Accepted so that `fit_model` can pass the same keyword arguments
        to the fit and to `extract_stats`.

    Returns
    -------
    dict
        Dictionary with entries for
        - params: Model parameters (includes fit_t_min, fit_t_max, and other \
method-specific values)
        - model_type: Method used for fitting

    Raises
    ------
    ValueError
        Method must be "sliding_window" or "spline". Smooth must be "fast", "slow",
        or a non-negative float.
    """
    # Sorted, de-duplicated, finite and N > 0 (raises on negative times)
    t, y_raw = validate_data(t, N)
    if t is None:
        return None

    # Calculate Umax using specified method
    if method == "sliding_window":
        return fit_sliding_window(t, y_raw, window_points)

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
