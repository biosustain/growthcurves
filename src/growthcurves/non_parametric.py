"""Non-parametric fitting methods for growth curves.

This module provides the sliding window and spline methods for estimating the
maximum specific growth rate.

Both methods take OD on a linear scale and fit ln(OD).
"""

from logging import getLogger

import numpy as np
from scipy.interpolate import BSpline

from .inference import _noise_levels
from .preprocessing import prepare_curve

logger = getLogger(__name__)

# -----------------------------------------------------------------------------
# Sliding Window
# -----------------------------------------------------------------------------


# region: sliding window helpers
def _window_lengths(n: int, w_min: int, ratio: float = 1.25):
    """Window lengths from w_min to n // 2, each ~ratio x the last."""
    lengths, w = [], w_min
    while w <= max(w_min, n // 2):
        lengths.append(w)
        w = max(w + 1, round(w * ratio))
    return lengths


def _ln_noise_sd(N: np.ndarray, sd_add: float, sd_prop: float, N_ref: np.ndarray):
    """Per-reading sd of ln N, using the smaller of the observed and smoothed OD."""
    return np.hypot(sd_prop, sd_add / np.maximum(np.minimum(N, N_ref), 1e-12))


def _wls_windows(t: np.ndarray, y: np.ndarray, sd: np.ndarray, lengths: list):
    """Weighted least squares of every window of each length, via prefix sums.

    Yields, per length, slope, SE of the slope (from sd), reduced chi-square, start
    and length of every window, so only one length is held in memory at a time.
    """
    t, y, w = t - t.mean(), y - y.mean(), 1 / sd**2
    c = [
        np.concatenate([[0], np.cumsum(v)])
        for v in (w, w * t, w * y, w * t * t, w * t * y, w * y * y)
    ]
    for L in lengths:
        i = np.arange(len(t) - L + 1)
        sw, st, sy, stt, sty, syy = (v[i + L] - v[i] for v in c)
        sxx, sxy = stt - st**2 / sw, sty - st * sy / sw
        slope = sxy / sxx
        chi2 = np.maximum(syy - sy**2 / sw - slope * sxy, 0) / (L - 2)
        yield slope, 1 / np.sqrt(sxx), chi2, i, L


# endregion


# region: sliding window method
def fit_sliding_window(
    t: np.ndarray,
    N: np.ndarray,
    window_points: int = 5,
    min_rise: float = 0.10,
    min_fold: float = 1.5,
):
    """
    Calculate maximum specific growth rate using the sliding window method.

    Fits ln(N) by weighted least squares in windows of every length from
    `window_points` readings to half the curve, weighting readings by a noise model
    estimated from the curve. The window with the highest lower bound, slope - z*SE,
    wins (z = sqrt(2 ln M) for M windows; SE inflated by sqrt(reduced chi-square)
    where a window fits worse than the noise). So the window length adapts: short on
    clean or sparse data, long on noisy data.

    The data are first prepared by `preprocessing.prepare_curve` (growth check with
    `min_fold`, truncation at the smoothed OD peak and after long stationary phases),
    as for every growth model. To reject correlated baseline noise, windows must
    cover at least `min_rise` of the smoothed OD rise from its minimum to its peak.

    Parameters:
        t: Time array (hours)
        N: OD values (linear-scale, baseline-corrected, must be positive)
        window_points: Minimum readings per window (default: 5, at least 3)
        min_rise: Minimum share of the OD rise a window must cover (default: 0.10)
        min_fold: Minimum fold increase of the smoothed OD for growth (default: 1.5;
            see `prepare_curve`)

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

    t, N = prepare_curve(t, N, min_fold=min_fold)
    if t is None or len(t) < window_points:
        return None

    sd_add, sd_prop, N_ref = _noise_levels(t, N)
    peak = int(np.argmax(N_ref))
    base = N_ref[: peak + 1].min()

    y, sd = np.log(N), _ln_noise_sd(N, sd_add, sd_prop, N_ref)
    lengths = _window_lengths(len(t), int(window_points))
    z = np.sqrt(2 * np.log(sum(len(t) - L + 1 for L in lengths)))
    # Keep only the best window so far: memory stays O(n) on long curves
    best_bound, best = -np.inf, None
    for slope, se, chi2, start, L in _wls_windows(t, y, sd, lengths):
        end = start + L - 1
        on_growth = N_ref[end] - N_ref[start] >= min_rise * (N_ref[peak] - base)
        bound = slope - z * se * np.sqrt(np.maximum(chi2, 1))
        bound = np.where(on_growth & ~np.isnan(bound), bound, -np.inf)
        k = int(np.argmax(bound))
        if bound[k] > best_bound:
            best_bound, best = bound[k], (slope[k], start[k], L)
    if not best_bound > 0:
        return None

    slope_k, s0, L = best
    win = slice(s0, s0 + L)
    w = sd[win] ** -2
    intercept = np.average(y[win], weights=w) - slope_k * np.average(t[win], weights=w)
    return {
        "params": {
            "slope": float(slope_k),
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
_LAM_GRID = np.logspace(-6, 4, 41)  # in units of tr(BᵀWB)/tr(P)


def _bisquare(r: np.ndarray, c: float = 4.685):
    """Tukey bisquare weights of standardised residuals."""
    r = r / max(1.0, 1.4826 * np.median(np.abs(r)))
    return np.clip(1 - (r / c) ** 2, 0, 1) ** 2


# endregion


# region: fit_spline
def fit_spline(t: np.ndarray, N: np.ndarray, smooth="auto", min_fold: float = 1.5):
    """
    Maximum specific growth rate from a penalised spline of ln(N).

    Noise-weighted cubic P-spline, smoothing chosen by REML, with robust reweighting
    and IRLS in OD. μmax is the derivative where derivative - 2·SE is highest.
    The data are first prepared by `preprocessing.prepare_curve` (growth check,
    truncation after the growth phase), as for every growth model.

    Parameters:
        t: Time array (hours)
        N: OD values (linear scale, baseline-corrected)
        smooth: "auto" (chosen by REML) or a fixed value > 0, in units of tr(BᵀWB)/tr(P)
        min_fold: Minimum fold increase of the smoothed OD, from its lowest value before
            the peak to the peak, for growth (default: 1.5; see `prepare_curve`)

    Returns:
        Dict with params (tck_t, tck_c, tck_k, mu_max, time_at_umax, fit_t_min,
        fit_t_max) and model_type "spline". None if no growth (fold increase below
        `min_fold`, or derivative - 3.5·SE not above zero) or too few points.
    """
    auto = isinstance(smooth, str) and smooth == "auto"
    if not auto and not (isinstance(smooth, (int, float)) and smooth > 0):
        raise ValueError('smooth must be "auto" or a positive number.')
    t, N = prepare_curve(t, N, min_fold=min_fold)
    if t is None:
        return None

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
        Li = np.linalg.inv(
            np.linalg.cholesky(A)
        )  # numpy: less overhead than scipy at K ~ 30
        e, U = np.linalg.eigh(Li @ P @ Li.T)
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


def fit_non_parametric(
    t: np.ndarray,
    N: np.ndarray,
    method: str = "sliding_window",
    window_points: int = 5,
    smooth="auto",
    min_rise: float = 0.10,
    min_fold: float = 1.5,
    **kwargs,
):
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
    min_rise : float, optional
        Minimum rise of ln(N) for a window (sliding_window method), by default 0.10
    min_fold : float, optional
        Minimum fold increase of the curve for growth (both methods), by default 1.5
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
        return fit_sliding_window(t, N, window_points, min_rise, min_fold)
    if method == "spline":
        return fit_spline(t, N, smooth, min_fold)
    raise ValueError(f"Unknown umax_method: {method}")
