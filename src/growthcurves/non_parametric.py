"""Non-parametric fitting methods for growth curves.

This module provides the sliding window and spline methods for estimating the
maximum specific growth rate.

Both methods take OD on a linear scale and fit ln(OD).
"""

from logging import getLogger

import numpy as np
from scipy.interpolate import BSpline
from scipy.linalg import cholesky, eigh, solve_triangular

from .inference import smooth as smooth_od
from .inference import validate_data

logger = getLogger(__name__)

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


def _noise_levels(t, N):
    """Additive OD and proportional noise sds (second differences, MAD) and the smoothed OD."""
    N_ref = smooth_od(N)
    mid = N_ref[1:-1]
    lo, hi = mid <= np.quantile(mid, 1 / 3), mid >= np.quantile(mid, 2 / 3)
    mad_to_sd = 1.4826 / np.sqrt(6)
    sd_add = mad_to_sd * np.median(np.abs(_second_diff_series(t, N))[lo])
    sd_add = max(sd_add, np.diff(np.unique(N)).min() / np.sqrt(12))
    sd_prop = mad_to_sd * np.median(np.abs(_second_diff_series(t, np.log(N)))[hi])
    return sd_add, sd_prop, N_ref


def _ln_noise_sd(N, sd_add, sd_prop, N_ref):
    """Per-reading sd of ln N, using the smaller of the observed and smoothed OD."""
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
def fit_sliding_window(t, N, window_points=5, min_rise=0.10, min_fold=1.5):
    """
    Calculate maximum specific growth rate using the sliding window method.

    Fits ln(N) by weighted least squares in windows of every length from
    `window_points` readings to half the curve, weighting readings by a noise model
    estimated from the curve. The window with the highest lower bound, slope - z*SE,
    wins (z = sqrt(2 ln M) for M windows; SE inflated by sqrt(reduced chi-square)
    where a window fits worse than the noise). So the window length adapts: short on
    clean or sparse data, long on noisy data.

    To reject correlated baseline noise, windows must end by the smoothed OD peak
    and cover at least `min_rise` of its rise from the preceding minimum.
    The peak must reach `min_fold` times that minimum.

    Parameters:
        t: Time array (hours)
        N: OD values (linear-scale, baseline-corrected, must be positive)
        window_points: Minimum readings per window (default: 5, at least 3)
        min_rise: Minimum share of the OD rise a window must cover (default: 0.10)
        min_fold: Minimum fold increase of the smoothed OD for growth (default: 1.5)

    Returns:
        Dict with params (slope = μmax in h⁻¹, intercept in log space, time_at_umax
        = window midpoint, fit_t_min, fit_t_max) and model_type "sliding_window".
        None for too few points, insufficient fold increase, or no qualifying
        window with a positive lower bound.
    """
    if int(window_points) != window_points or window_points < 3:
        raise ValueError("window_points must be an integer of at least 3.")
    if not 0 <= min_rise < 1:
        raise ValueError("min_rise must be in [0, 1).")
    if min_fold < 1:
        raise ValueError("min_fold must be at least 1.")

    t, N = validate_data(t, N)
    if t is None or len(t) < window_points or np.ptp(N) == 0:
        return None

    sd_add, sd_prop, N_ref = _noise_levels(t, N)
    peak = int(np.argmax(N_ref))
    base = N_ref[: peak + 1].min()
    if N_ref[peak] < min_fold * base:
        return None

    y, sd = np.log(N), _ln_noise_sd(N, sd_add, sd_prop, N_ref)
    slope, se, chi2, start, length = _wls_windows(
        t, y, sd, _window_lengths(len(t), int(window_points))
    )
    end = start + length - 1
    on_growth = (end <= peak) & (N_ref[end] - N_ref[start] >= min_rise * (N_ref[peak] - base))
    bound = slope - np.sqrt(2 * np.log(len(slope))) * se * np.sqrt(np.maximum(chi2, 1))
    bound = np.where(on_growth, bound, -np.inf)
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


# region: spline helpers
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


_LAM_GRID = np.logspace(-6, 4, 41)  # in units of tr(BᵀWB)/tr(P)


def _bisquare(r, c=4.685):
    """Tukey bisquare weights of standardised residuals."""
    r = r / max(1.0, 1.4826 * np.median(np.abs(r)))
    return np.clip(1 - (r / c) ** 2, 0, 1) ** 2


# endregion


# region: fit_spline
def fit_spline(t, N, smooth="auto"):
    """
    Maximum specific growth rate from a penalised spline of ln(N).

    Noise-weighted cubic P-spline, smoothing chosen by REML, with robust reweighting
    and IRLS in OD; long stationary phases are cropped. μmax is the derivative where
    derivative - 2·SE is highest.

    Parameters:
        t: Time array (hours)
        N: OD values (linear scale, baseline-corrected)
        smooth: "auto" (chosen by REML) or a fixed value > 0, in units of tr(BᵀWB)/tr(P)

    Returns:
        Dict with params (tck_t, tck_c, tck_k, mu_max, time_at_umax, fit_t_min,
        fit_t_max) and model_type "spline". None if no growth (derivative - 3.5·SE
        not above zero) or too few points.
    """
    # Sorted, de-duplicated, finite and N > 0 (raises on negative times)
    auto = isinstance(smooth, str) and smooth == "auto"
    if not auto and not (isinstance(smooth, (int, float)) and smooth > 0):
        raise ValueError('smooth must be "auto" or a positive number.')
    t, N = validate_data(t, N)
    if t is None or np.ptp(N) == 0:
        return None

    # Crop: end half the 10-95 % rise duration after the curve reaches 95 % of its rise
    sd_add, sd_prop, N_ref = _noise_levels(t, N)
    y_ref = np.log(np.maximum(N_ref, max(3 * sd_add, 1e-12)))
    lo, hi = y_ref.min(), y_ref.max()
    t10, t95 = (t[np.argmax(y_ref >= lo + q * (hi - lo))] for q in (0.1, 0.95))
    keep = t <= max(t95 + 0.5 * (t95 - t10), t[min(len(t), 20) - 1])
    if not keep.all():
        t, N = t[keep], N[keep]
        sd_add, sd_prop, N_ref = _noise_levels(t, N)
    n = len(t)

    n_seg = max(4, min(round(n / 2), 25))
    knots = t[0] + (t[-1] - t[0]) / n_seg * np.arange(-3, n_seg + 4)
    knots[3 + n_seg] = t[-1]  # floating point could leave t[-1] outside
    B = BSpline.design_matrix(t, knots, 3).toarray()
    K = B.shape[1]
    D = np.diff(np.eye(K), 2, axis=0)
    P = D.T @ D

    def solve(w, y):
        # Eigen-decomposition: (A + λP)⁻¹ = T diag(1 / (1 + λe)) Tᵀ for any λ
        A = (B.T * w) @ B
        A[np.diag_indices(K)] += 1e-10 * np.trace(A) / K
        Li = solve_triangular(cholesky(A, lower=True), np.eye(K), lower=True)
        e, U = eigh(Li @ P @ Li.T)
        T = Li.T @ U
        return T, np.maximum(e, 0), T.T @ (B.T @ (w * y)), np.sum(w * y * y)

    def reml(S):
        _, e, beta, yWy = S
        Q = yWy - np.sum(beta**2 / (1 + lams[:, None] * e), axis=1)
        logdet = np.sum(np.log1p(lams[:, None] * e), axis=1)
        logdet -= (e > 1e-10 * e.max()).sum() * np.log(lams)
        return lams[np.argmin((n - 2) * np.log(np.maximum(Q, 1e-300)) + logdet)]

    def fitted(S, lam):
        T, e, beta, _ = S
        return B @ (T @ (beta / (1 + lam * e)))

    y = np.log(N)
    w0 = _ln_noise_sd(N, sd_add, sd_prop, N_ref) ** -2
    scale = np.trace((B.T * w0) @ B) / np.trace(P)
    lams = _LAM_GRID * scale
    S = solve(w0, y)
    lam = reml(S) if auto else smooth * scale
    for _ in range(2):  # robust weights, smoothing re-chosen
        S = solve(w0 * _bisquare((y - fitted(S, lam)) * np.sqrt(w0)), y)
        lam = reml(S) if auto else lam
    for _ in range(2):  # IRLS with the noise in OD
        f = fitted(S, lam)
        m = np.exp(f)
        sd_N = np.hypot(sd_prop * m, sd_add)
        S = solve((m / sd_N) ** 2 * _bisquare((N - m) / sd_N), f + (N - m) / m)

    T, e, beta, yWy = S
    s = 1 / (1 + lam * e)
    chi2 = (yWy - np.sum(beta**2 * (2 * s - s**2))) / max(n - s.sum(), 1)
    t_eval = np.linspace(t[0], t[-1], 400)
    G = BSpline(knots, np.eye(K), 3).derivative()(t_eval) @ T
    mu = G @ (beta * s)
    se = np.sqrt(np.sum((G * s) ** 2, axis=1) * max(chi2, 1.0))
    i = int(np.argmax(mu - 2 * se))
    if not mu[i] - 3.5 * se[i] > 0:
        return None

    return {
        "params": {
            "tck_t": knots.tolist(),
            "tck_c": (T @ (beta * s)).tolist(),
            "tck_k": 3,
            "mu_max": float(mu[i]),
            "time_at_umax": float(t_eval[i]),
            "fit_t_min": float(t[0]),
            "fit_t_max": float(t[-1]),
        },
        "model_type": "spline",
    }


# endregion

# -----------------------------------------------------------------------------
# Main API Functions
# -----------------------------------------------------------------------------


def fit_non_parametric(t, N, method="sliding_window", window_points=5, smooth="auto", **kwargs):
    """
    Fit a non-parametric model: "sliding_window" (`fit_sliding_window`) or "spline"
    (`fit_spline`).

    Parameters
    ----------
    t : Iterable[float]
        Time array (hours)
    N : Iterable[float]
        OD values
    method : str, optional
        "sliding_window" or "spline", by default "sliding_window"
    window_points : int, optional
        Minimum number of readings in a window (sliding_window method), by default 5
    smooth : "auto" or float, optional
        Spline smoothing: "auto" (default, chosen by REML) or a fixed positive value
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
        Method must be "sliding_window" or "spline".
    """
    if method == "sliding_window":
        return fit_sliding_window(t, N, window_points)
    if method == "spline":
        return fit_spline(t, N, smooth)
    raise ValueError(f"Unknown umax_method: {method}")
