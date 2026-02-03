"""Growth curve models.

This module defines parametric growth model functions (Richards, Logistic,
Gompertz, Baranyi, Gaussian) that operate in linear OD space (not log-transformed).
"""

import numpy as np
from scipy.interpolate import UnivariateSpline


def logistic_model(t, K, y0, r, t0):
    """
    Logistic growth model in linear OD space with baseline offset.

    N(t) = y0 + (K - y0) / [1 + exp(-r * (t - t0))]

    Parameters:
        t: Time array
        K: Carrying capacity (maximum OD)
        y0: Baseline OD (minimum value at t=0)
        r: Growth rate constant (h^-1), equals mu_max
        t0: Inflection time (time at which N = (K + y0)/2)

    Returns:
        OD values at each time point

    Note:
        mu_max = r for the logistic model
    """
    return y0 + (K - y0) / (1 + np.exp(-r * (t - t0)))


def gompertz_model(t, K, y0, mu_max, lam):
    """
    Modified Gompertz growth model in linear OD space with baseline offset.

    N(t) = y0 + (K - y0) * exp{-exp[(mu_max * e / (K - y0)) * (lam - t) + 1]}

    Parameters:
        t: Time array
        K: Carrying capacity (maximum OD)
        y0: Baseline OD (minimum value)
        mu_max: Maximum specific growth rate (h^-1)
        lam: Lag time (hours)

    Returns:
        OD values at each time point

    Note:
        mu_max is directly a fitted parameter
    """
    e = np.e
    A = K - y0  # Amplitude
    return y0 + A * np.exp(-np.exp((mu_max * e / A) * (lam - t) + 1))


def richards_model(t, K, y0, r, t0, nu):
    """
    Richards growth model in linear OD space with baseline offset.

    N(t) = y0 + (K - y0) * [1 + nu * exp(-r * (t - t0))]^(-1/nu)

    Parameters:
        t: Time array
        K: Carrying capacity (maximum OD)
        y0: Baseline OD (minimum value)
        r: Growth rate constant (h^-1)
        t0: Inflection time
        nu: Shape parameter (nu=1 gives logistic, nu->0 gives Gompertz)

    Returns:
        OD values at each time point

    Note:
        mu_max = r / (nu + 1)^(1/nu) for the Richards model
    """
    return y0 + (K - y0) * (1 + nu * np.exp(-r * (t - t0))) ** (-1 / nu)


def baranyi_model(t, K, y0, mu_max, h0):
    """
    Baranyi-Roberts growth model in linear OD space with baseline offset.

    This model includes a lag phase parameter h0 that represents the initial
    physiological state of the cells.

    N(t) = K / (1 + ((K - y0) / y0) * exp(-mu_max * A(t)))
    and A(t) = t + (1/mu_max) * ln(exp(-mu_max*t) + exp(-h0) - exp(-mu_max*t - h0))

    Parameters:
        t: Time array
        K: Carrying capacity (maximum OD)
        y0: Baseline OD (minimum value)
        mu_max: Maximum specific growth rate (h^-1)
        h0: Dimensionless lag parameter (higher values = longer lag)

    Returns:
        OD values at each time point

    Note:
        - h0 > 0 indicates lag phase
        - The lag time can be approximated as lambda ≈ h0/mu_max
        - mu_max is directly a fitted parameter
    """
    # Adjustment function A(t) accounting for lag phase
    # A(t) = t + (1/mu_max) * ln(exp(-mu_max*t) + exp(-h0) - exp(-mu_max*t - h0))
    # Use stable computation to avoid overflow
    exp_neg_mu_t = np.exp(-mu_max * t)
    exp_neg_h0 = np.exp(-h0)
    exp_neg_both = np.exp(-mu_max * t - h0)

    # A(t) adjustment function
    A_t = t + (1.0 / mu_max) * np.log(exp_neg_mu_t + exp_neg_h0 - exp_neg_both)

    # Baranyi model (linear OD), ensuring N(0) = y0 when A(0) -> 0
    y0_safe = np.maximum(y0, 1e-9)
    denom = 1.0 + ((K - y0_safe) / y0_safe) * np.exp(-mu_max * A_t)
    return K / denom


def gaussian(t, amplitude, center, sigma):
    """Symmetric Gaussian bell-shaped curve."""
    sigma = np.maximum(sigma, 1e-12)
    return amplitude * np.exp(-((t - center) ** 2) / (2 * sigma**2))


def evaluate_parametric_model(t, model_type, params):
    """
    Evaluate a fitted parametric model at given time points.

    This function provides a unified interface for evaluating any parametric
    growth model, eliminating the need for repeated if-elif chains.

    Parameters:
        t: Time array or scalar
        model_type: One of 'logistic', 'gompertz', 'richards', 'baranyi'
        params: Parameter dictionary containing model-specific parameters

    Returns:
        OD values predicted by the model at time points t

    Raises:
        ValueError: If model_type is not recognized

    Example:
        >>> params = {"K": 0.5, "y0": 0.05, "r": 0.1, "t0": 10}
        >>> y_fit = evaluate_parametric_model(t, "logistic", params)
    """
    # Model registry: maps model_type to (function, required_param_names)
    MODEL_REGISTRY = {
        "logistic": (logistic_model, ["K", "y0", "r", "t0"]),
        "gompertz": (gompertz_model, ["K", "y0", "mu_max", "lam"]),
        "richards": (richards_model, ["K", "y0", "r", "t0", "nu"]),
        "baranyi": (baranyi_model, ["K", "y0", "mu_max", "h0"]),
    }

    if model_type not in MODEL_REGISTRY:
        raise ValueError(
            f"Unknown model type: {model_type}. "
            f"Must be one of {list(MODEL_REGISTRY.keys())}"
        )

    model_func, param_names = MODEL_REGISTRY[model_type]
    model_args = [params[name] for name in param_names]

    return model_func(t, *model_args)


def spline_model(t, y, spline_s=None, k=3):
    """
    Fit a smoothing spline to data.

    Parameters:
        t: Time array
        y: Values array (e.g., log-transformed OD)
        spline_s: Smoothing factor (None = automatic)
        k: Spline degree (default: 3)

    Returns:
        Tuple of (spline, spline_s) where spline is a UnivariateSpline instance.
    """
    t = np.asarray(t, dtype=float)
    y = np.asarray(y, dtype=float)

    if spline_s is None:
        spline_s = 0.01

    spline = UnivariateSpline(t, y, s=spline_s, k=k)
    return spline, spline_s


def spline_from_params(params):
    """
    Reconstruct a spline from stored parameters.

    Parameters:
        params: Dict containing 't_knots', 'spline_coeffs', and 'spline_k'

    Returns:
        UnivariateSpline or BSpline instance.
    """
    if "tck_t" in params and "tck_c" in params:
        t_knots = np.asarray(params["tck_t"], dtype=float)
        coeffs = np.asarray(params["tck_c"], dtype=float)
        k = int(params.get("tck_k", params.get("spline_k", 3)))
    else:
        t_knots = np.asarray(params["t_knots"], dtype=float)
        coeffs = np.asarray(params["spline_coeffs"], dtype=float)
        k = int(params.get("spline_k", 3))

    try:
        return UnivariateSpline._from_tck((t_knots, coeffs, k))
    except Exception:
        from scipy.interpolate import BSpline

        return BSpline(t_knots, coeffs, k)
