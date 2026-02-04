"""Growth curve models.

This module defines parametric growth model functions for growth curve analysis.
Models are categorized into two classes:

1. **Mechanistic models**: Defined as ODEs (dN/dt), fitted using ODE integration
   - mech_logistic, mech_gompertz, mech_richards, mech_baranyi

2. **Phenomenological models**: Fitted directly to ln(OD/OD0)
   - phenom_logistic, phenom_gompertz, phenom_gompertz_modified, phenom_richards

Legacy models (logistic_model, gompertz_model, etc.) are retained for compatibility.
"""

import numpy as np
from scipy.interpolate import UnivariateSpline
from scipy.integrate import solve_ivp


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


# =============================================================================
# MECHANISTIC MODELS (ODE-based)
# =============================================================================


def mech_logistic_ode(t, N, mu, K):
    """
    Logistic growth ODE: dN/dt = μ * (1 - N/K) * N

    Parameters:
        t: Time (scalar)
        N: Population (OD) at time t
        mu: Intrinsic growth rate (h^-1)
        K: Carrying capacity (maximum OD)

    Returns:
        dN/dt: Rate of change
    """
    return mu * (1 - N / K) * N


def mech_gompertz_ode(t, N, mu, K):
    """
    Gompertz growth ODE: dN/dt = μ * log(K/N) * N

    Parameters:
        t: Time (scalar)
        N: Population (OD) at time t
        mu: Intrinsic growth rate (h^-1)
        K: Carrying capacity (maximum OD)

    Returns:
        dN/dt: Rate of change
    """
    if N <= 0:
        return 0.0
    # Ensure N is at least 0.1% of K to prevent log from exploding
    # This provides numerical stability without affecting realistic growth dynamics
    N_safe = np.maximum(N, K * 0.001)
    return mu * np.log(K / N_safe) * N


def mech_richards_ode(t, N, mu, K, beta):
    """
    Richards growth ODE: dN/dt = μ * (1 - (N/K)^β) * N

    Parameters:
        t: Time (scalar)
        N: Population (OD) at time t
        mu: Intrinsic growth rate (h^-1)
        K: Carrying capacity (maximum OD)
        beta: Shape parameter

    Returns:
        dN/dt: Rate of change
    """
    if N <= 0:
        return 0.0
    ratio = N / K
    if ratio >= 1:
        return 0.0
    return mu * (1 - ratio**beta) * N


def mech_baranyi_ode(t, N, mu, K, h0):
    """
    Baranyi-Roberts growth ODE: dN/dt = μ * A(t) * (1 - N/K) * N
    where A(t) = exp(μ*t) / (exp(h0) - 1 + exp(μ*t))

    Parameters:
        t: Time (scalar)
        N: Population (OD) at time t
        mu: Maximum specific growth rate (h^-1)
        K: Carrying capacity (maximum OD)
        h0: Dimensionless lag parameter

    Returns:
        dN/dt: Rate of change
    """
    # Adjustment function A(t)
    exp_mu_t = np.exp(mu * t)
    exp_h0 = np.exp(h0)
    A_t = exp_mu_t / (exp_h0 - 1 + exp_mu_t)

    return mu * A_t * (1 - N / K) * N


def mech_logistic_model(t, mu, K, N0, y0):
    """
    Solve logistic ODE and return OD values at time points t.

    ODE: dN/dt = μ * (1 - N/K) * N
    OD(t) = y0 + N(t)

    Parameters:
        t: Time array
        mu: Intrinsic growth rate (h^-1)
        K: Carrying capacity above baseline (maximum ΔOD)
        N0: Initial population above baseline at t=0
        y0: Baseline OD (offset parameter)

    Returns:
        OD values at each time point
    """
    t = np.asarray(t, dtype=float)
    if np.isscalar(t):
        t = np.array([t])

    # Solve ODE
    sol = solve_ivp(
        lambda t_val, N: mech_logistic_ode(t_val, N[0], mu, K),
        [t.min(), t.max()],
        [N0],
        t_eval=t,
        method="RK45",
    )

    return y0 + sol.y[0]


def mech_gompertz_model(t, mu, K, N0, y0):
    """
    Solve Gompertz ODE and return OD values at time points t.

    ODE: dN/dt = μ * log(K/N) * N
    OD(t) = y0 + N(t)

    Parameters:
        t: Time array
        mu: Intrinsic growth rate (h^-1)
        K: Carrying capacity above baseline (maximum ΔOD)
        N0: Initial population above baseline at t=0
        y0: Baseline OD (offset parameter)

    Returns:
        OD values at each time point
    """
    t = np.asarray(t, dtype=float)
    if np.isscalar(t):
        t = np.array([t])

    # Solve ODE
    sol = solve_ivp(
        lambda t_val, N: mech_gompertz_ode(t_val, N[0], mu, K),
        [t.min(), t.max()],
        [N0],
        t_eval=t,
        method="RK45",
    )

    return y0 + sol.y[0]


def mech_richards_model(t, mu, K, N0, beta, y0):
    """
    Solve Richards ODE and return OD values at time points t.

    ODE: dN/dt = μ * (1 - (N/K)^β) * N
    OD(t) = y0 + N(t)

    Parameters:
        t: Time array
        mu: Intrinsic growth rate (h^-1)
        K: Carrying capacity above baseline (maximum ΔOD)
        N0: Initial population above baseline at t=0
        beta: Shape parameter
        y0: Baseline OD (offset parameter)

    Returns:
        OD values at each time point
    """
    t = np.asarray(t, dtype=float)
    if np.isscalar(t):
        t = np.array([t])

    # Solve ODE
    sol = solve_ivp(
        lambda t_val, N: mech_richards_ode(t_val, N[0], mu, K, beta),
        [t.min(), t.max()],
        [N0],
        t_eval=t,
        method="RK45",
    )

    return y0 + sol.y[0]


def mech_baranyi_model(t, mu, K, N0, h0, y0):
    """
    Solve Baranyi-Roberts ODE and return OD values at time points t.

    ODE: dN/dt = μ * A(t) * (1 - N/K) * N
    where A(t) = exp(μ*t) / (exp(h0) - 1 + exp(μ*t))
    OD(t) = y0 + N(t)

    Parameters:
        t: Time array
        mu: Maximum specific growth rate (h^-1)
        K: Carrying capacity above baseline (maximum ΔOD)
        N0: Initial population above baseline at t=0
        h0: Dimensionless lag parameter
        y0: Baseline OD (offset parameter)

    Returns:
        OD values at each time point
    """
    t = np.asarray(t, dtype=float)
    if np.isscalar(t):
        t = np.array([t])

    # Solve ODE
    sol = solve_ivp(
        lambda t_val, N: mech_baranyi_ode(t_val, N[0], mu, K, h0),
        [t.min(), t.max()],
        [N0],
        t_eval=t,
        method="RK45",
    )

    return y0 + sol.y[0]


# =============================================================================
# PHENOMENOLOGICAL MODELS (ln-space)
# =============================================================================


def phenom_logistic_model(t, A, mu_max, lam, N0):
    """
    Phenomenological logistic model in ln-space.

    ln(Nt/N0) = A / (1 + exp(4 * μ_max * (λ - t) / A + 2))

    Parameters:
        t: Time array
        A: Maximum ln(OD/OD0) (amplitude)
        mu_max: Maximum specific growth rate (h^-1)
        lam: Lag time (hours)
        N0: Initial OD at t=0

    Returns:
        OD values at each time point
    """
    t = np.asarray(t, dtype=float)
    ln_ratio = A / (1 + np.exp(4 * mu_max * (lam - t) / A + 2))
    return N0 * np.exp(ln_ratio)


def phenom_gompertz_model(t, A, mu_max, lam, N0):
    """
    Phenomenological Gompertz model in ln-space.

    ln(Nt/N0) = A * exp(-exp(μ_max * e * (λ - t) / A + 1))

    Parameters:
        t: Time array
        A: Maximum ln(OD/OD0) (amplitude)
        mu_max: Maximum specific growth rate (h^-1)
        lam: Lag time (hours)
        N0: Initial OD at t=0

    Returns:
        OD values at each time point
    """
    t = np.asarray(t, dtype=float)
    e = np.e
    ln_ratio = A * np.exp(-np.exp(mu_max * e * (lam - t) / A + 1))
    return N0 * np.exp(ln_ratio)


def phenom_gompertz_modified_model(t, A, mu_max, lam, alpha, t_shift, N0):
    """
    Phenomenological modified Gompertz model with decay term.

    ln(Nt/N0) = A * exp(-exp(μ_max * e * (λ - t) / A + 1)) + A * exp(α * (t - t_shift))

    Parameters:
        t: Time array
        A: Maximum ln(OD/OD0) (amplitude)
        mu_max: Maximum specific growth rate (h^-1)
        lam: Lag time (hours)
        alpha: Decay rate (h^-1)
        t_shift: Time shift for decay (hours)
        N0: Initial OD at t=0

    Returns:
        OD values at each time point
    """
    t = np.asarray(t, dtype=float)
    e = np.e
    ln_ratio = A * np.exp(-np.exp(mu_max * e * (lam - t) / A + 1)) + A * np.exp(
        alpha * (t - t_shift)
    )
    return N0 * np.exp(ln_ratio)


def phenom_richards_model(t, A, mu_max, lam, nu, N0):
    """
    Phenomenological Richards model in ln-space.

    ln(Nt/N0) = A * (1 + ν * exp(1 + ν + μ_max * (1 + ν)^(1/ν) * (λ - t) / A))^(-1/ν)

    Parameters:
        t: Time array
        A: Maximum ln(OD/OD0) (amplitude)
        mu_max: Maximum specific growth rate (h^-1)
        lam: Lag time (hours)
        nu: Shape parameter
        N0: Initial OD at t=0

    Returns:
        OD values at each time point
    """
    t = np.asarray(t, dtype=float)
    # Avoid division by very small nu
    nu = np.maximum(nu, 0.01)
    exponent = 1 + nu + mu_max * (1 + nu) ** (1 / nu) * (lam - t) / A
    ln_ratio = A * (1 + nu * np.exp(exponent)) ** (-1 / nu)
    return N0 * np.exp(ln_ratio)


# =============================================================================
# UNIFIED MODEL EVALUATION
# =============================================================================


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
        # Legacy models (for backward compatibility)
        "logistic": (logistic_model, ["K", "y0", "r", "t0"]),
        "gompertz": (gompertz_model, ["K", "y0", "mu_max", "lam"]),
        "richards": (richards_model, ["K", "y0", "r", "t0", "nu"]),
        "baranyi": (baranyi_model, ["K", "y0", "mu_max", "h0"]),
        # Mechanistic models (ODE-based)
        "mech_logistic": (mech_logistic_model, ["mu", "K", "N0", "y0"]),
        "mech_gompertz": (mech_gompertz_model, ["mu", "K", "N0", "y0"]),
        "mech_richards": (mech_richards_model, ["mu", "K", "N0", "beta", "y0"]),
        "mech_baranyi": (mech_baranyi_model, ["mu", "K", "N0", "h0", "y0"]),
        # Phenomenological models (ln-space)
        "phenom_logistic": (phenom_logistic_model, ["A", "mu_max", "lam", "N0"]),
        "phenom_gompertz": (phenom_gompertz_model, ["A", "mu_max", "lam", "N0"]),
        "phenom_gompertz_modified": (
            phenom_gompertz_modified_model,
            ["A", "mu_max", "lam", "alpha", "t_shift", "N0"],
        ),
        "phenom_richards": (phenom_richards_model, ["A", "mu_max", "lam", "nu", "N0"]),
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
