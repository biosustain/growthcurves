"""Growth curve models.

This module defines parametric growth model functions for growth curve analysis.
Models are categorized into two classes:

1. **Mechanistic models**: Defined as ODEs (dN/dt), fitted using ODE integration
   - mech_logistic, mech_gompertz, mech_richards, mech_baranyi

2. **Phenomenological models**: Fitted directly to ln(OD/OD0)
   - phenom_logistic, phenom_gompertz, phenom_gompertz_modified, phenom_richards
"""

import numpy as np
from scipy.integrate import solve_ivp
from scipy.interpolate import UnivariateSpline

# =============================================================================
# MODEL REGISTRY
# =============================================================================

MODEL_REGISTRY = {
    "mechanistic": [
        "mech_logistic",
        "mech_gompertz",
        "mech_richards",
        "mech_baranyi",
    ],
    "phenomenological": [
        "phenom_logistic",
        "phenom_gompertz",
        "phenom_gompertz_modified",
        "phenom_richards",
    ],
    "non_parametric": [
        "spline",
        "sliding_window",
    ],
}


def get_all_parametric_models():
    """Return a set of all parametric model names (mechanistic + phenomenological)."""
    return set(MODEL_REGISTRY["mechanistic"] + MODEL_REGISTRY["phenomenological"])


def get_all_models():
    """Return a set of all model names (parametric + non-parametric)."""
    return list(
        MODEL_REGISTRY["mechanistic"]
        + MODEL_REGISTRY["phenomenological"]
        + MODEL_REGISTRY["non_parametric"]
    )


def get_model_category(model_type):
    """
    Return the category of a model type.

    Parameters:
        model_type: Model type string

    Returns:
        Category string: "mechanistic", "phenomenological", or "non_parametric"

    Raises:
        ValueError: If model_type is not recognized
    """
    for category, models in MODEL_REGISTRY.items():
        if model_type in models:
            return category
    raise ValueError(
        f"Unknown model type: {model_type}. " f"Must be one of {get_all_models()}"
    )


# =============================================================================
# MECHANISTIC MODELS (ODE-based)
# =============================================================================


def mech_logistic_ode(t, N, mu, K):
    """
    Logistic growth ODE: dN/dt = μ * (1 - N/K) * N

    Parameters:
        t: Time (scalar)
        N: Population (OD) at t
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
        N: Population (OD) at t
        mu: Intrinsic growth rate (h^-1)
        K: Carrying capacity (maximum OD)

    Returns:
        dN/dt: Rate of change
    """
    if N <= 0:
        return 0.0
    # Ensure N is at least 0.1% of K to prevent log from exploding
    # This provides numerical stability without affecting realistic growth dynamics
    data_safe = np.maximum(N, K * 0.001)
    return mu * np.log(K / data_safe) * N


def mech_richards_ode(t, N, mu, K, beta):
    """
    Richards growth ODE: dN/dt = μ * (1 - (N/K)^β) * N

    Parameters:
        t: Time (scalar)
        N: Population (OD) at t
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
        N: Population (OD) at t
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


def mech_logistic_model(t, mu, K, N0):
    """
    Solve logistic ODE and return OD values at t points.

    ODE: dN/dt = μ * (1 - N/K) * N
    OD(t) = N(t)

    Assumes input data is baseline-corrected (no additive offset).

    Parameters:
        t: Time array
        mu: Intrinsic growth rate (h^-1)
        K: Carrying capacity (maximum OD)
        N0: Initial population at t=0

    Returns:
        OD values at each t point
    """
    t = np.asarray(t, dtype=float)
    if np.isscalar(t):
        t = np.array([t])

    # Solve ODE
    sol = solve_ivp(
        lambda time_val, N: mech_logistic_ode(time_val, N[0], mu, K),
        [t.min(), t.max()],
        [N0],
        t_eval=t,
        method="RK45",
    )

    return sol.y[0]


def mech_gompertz_model(t, mu, K, N0):
    """
    Solve Gompertz ODE and return OD values at time points.

    ODE: dN/dt = μ * log(K/N) * N
    OD(t) = N(t)

    Assumes input data is baseline-corrected (no additive offset).

    Parameters:
        t: Time array
        mu: Intrinsic growth rate (h^-1)
        K: Carrying capacity (maximum OD)
        N0: Initial population at t=0

    Returns:
        OD values at each t point
    """
    t = np.asarray(t, dtype=float)
    if np.isscalar(t):
        t = np.array([t])

    # Solve ODE
    sol = solve_ivp(
        lambda time_val, N: mech_gompertz_ode(time_val, N[0], mu, K),
        [t.min(), t.max()],
        [N0],
        t_eval=t,
        method="RK45",
    )

    return sol.y[0]


def mech_richards_model(t, mu, K, N0, beta):
    """
    Solve Richards ODE and return OD values at time points.

    ODE: dN/dt = μ * (1 - (N/K)^β) * N
    OD(t) = N(t)

    Assumes input data is baseline-corrected (no additive offset).

    Parameters:
        t: Time array
        mu: Intrinsic growth rate (h^-1)
        K: Carrying capacity (maximum OD)
        N0: Initial population at t=0
        beta: Shape parameter

    Returns:
        OD values at each time point
    """
    t = np.asarray(t, dtype=float)
    if np.isscalar(t):
        t = np.array([t])

    # Solve ODE
    sol = solve_ivp(
        lambda time_val, N: mech_richards_ode(time_val, N[0], mu, K, beta),
        [t.min(), t.max()],
        [N0],
        t_eval=t,
        method="RK45",
    )

    return sol.y[0]


def mech_baranyi_model(t, mu, K, N0, h0):
    """
    Solve Baranyi-Roberts ODE and return OD values at t points.

    ODE: dN/dt = μ * A(t) * (1 - N/K) * N
    where A(t) = exp(μ*t) / (exp(h0) - 1 + exp(μ*t))
    OD(t) = N(t)

    Assumes input data is baseline-corrected (no additive offset).

    Parameters:
        t: Time array
        mu: Maximum specific growth rate (h^-1)
        K: Carrying capacity (maximum OD)
        N0: Initial population at t=0
        h0: Dimensionless lag parameter

    Returns:
        OD values at each time point
    """
    t = np.asarray(t, dtype=float)
    if np.isscalar(t):
        t = np.array([t])

    # Solve ODE
    sol = solve_ivp(
        lambda time_val, N: mech_baranyi_ode(time_val, N[0], mu, K, h0),
        [t.min(), t.max()],
        [N0],
        t_eval=t,
        method="RK45",
    )

    return sol.y[0]


# =============================================================================
# PHENOMENOLOGICAL MODELS (ln-space)
# =============================================================================


def phenom_logistic_model_ln(t, A, mu_max, lam):
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
    return ln_ratio


def phenom_gompertz_model_ln(t, A, mu_max, lam):
    """
    Phenomenological Gompertz model in ln-space.

    ln(Nt/N0) = A * exp(-exp(μ_max * e * (λ - t) / A + 1))

    Parameters:
        t: Time array
        A: Maximum ln(OD/OD0) (amplitude)
        mu_max: Maximum specific growth rate (h^-1)
        lam: Lag t (hours)

    Returns:
        OD values at each t point
    """
    t = np.asarray(t, dtype=float)
    e = np.e
    ln_ratio = A * np.exp(-np.exp(mu_max * e * (lam - t) / A + 1))
    return ln_ratio


def phenom_gompertz_modified_model_ln(t, A, mu_max, lam, alpha, t_shift):
    """
    Phenomenological modified Gompertz model with decay term. Models
    biphasic growth where the second exponential growth phase starts taking effect
    at time_shift.

    ln(Nt/N0) = A * exp(-exp(μ_max * e * (λ - t) / A + 1)) + A * exp(α * (t - t_shift))

    Parameters:
        t: Time array
        A: Maximum ln(OD/OD0) (amplitude)
        mu_max: Maximum specific growth rate (h^-1)
        lam: Lag t (hours)
        alpha: Decay rate (h^-1)
        t_shift: Time shift for decay (hours)

    Returns:
        OD values at each t point
    """
    t = np.asarray(t, dtype=float)
    e = np.e
    ln_ratio = A * np.exp(-np.exp(mu_max * e * (lam - t) / A + 1)) + A * np.exp(
        alpha * (t - t_shift)
    )
    return ln_ratio


def phenom_richards_model_ln(t, A, mu_max, lam, nu):
    """
    Phenomenological Richards model in ln-space.

    ln(Nt/N0)= A * (1 + ν * exp(1 + ν + μ_max * (1 + ν)^(1 + 1/ν) * (λ - t) / A))^(-1/ν)

    Parameters:
        t: Time array
        A: Maximum ln(OD/OD0) (amplitude)
        mu_max: Maximum specific growth rate (h^-1)
        lam: Lag t (hours)
        nu: Shape parameter

    Returns:
        OD values at each t point
    """
    t = np.asarray(t, dtype=float)
    # Avoid division by very small nu
    nu = np.maximum(nu, 0.01)
    exponent = 1 + nu + mu_max * (1 + nu) ** (1 + 1 / nu) * (lam - t) / A
    ln_ratio = A * (1 + nu * np.exp(exponent)) ** (-1 / nu)
    return ln_ratio


# Phenomological models are fitted to ln(OD/OD0) values, so the output of these
# functions is in ln-space. To convert to linear space, use the log_to_linear function
# below.
def log_to_linear(ln_ratio, N0):
    """
    Convert log-space values to linear-space values.

    Parameters:
        ln_ratio: Log-space values
        N0: Initial OD at t=0

    Returns:
        Linear-space values
    """
    return N0 * np.exp(ln_ratio)


# =============================================================================
# SPLINE MODELS
# =============================================================================


def spline_model(t, N, spline_s=None, k=3):
    """
    Fit a smoothing spline to N.

    Parameters:
        t: Time array
        N: Values array (e.g., log-transformed OD)
        spline_s: Smoothing factor (None = automatic)
        k: Spline degree (default: 3)

    Returns:
        Tuple of (spline, spline_s) where spline is a UnivariateSpline instance.
    """
    t = np.asarray(t, dtype=float)
    N = np.asarray(N, dtype=float)

    if spline_s is None:
        spline_s = 0.01

    spline = UnivariateSpline(t, N, s=spline_s, k=k)
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


# =============================================================================
# UNIFIED MODEL EVALUATION
# =============================================================================


def evaluate_parametric_model(t, model_type, params):
    """
    Evaluate a fitted parametric model at given t points.

    This function provides a unified interface for evaluating any parametric
    growth model, eliminating the need for repeated if-elif chains.

    Parameters:
        t: Time array or scalar
        model_type: Model type string (e.g., 'mech_logistic', 'phenom_gompertz')
        params: Parameter dictionary containing model-specific parameters

    Returns:
        OD values predicted by the model at t points

    Raises:
        ValueError: If model_type is not recognized

    Example:
        >>> params = {"mu": 0.5, "K": 0.5, "N0": 0.001}
        >>> y_fit = evaluate_parametric_model(t, "mech_logistic", params)
    """
    # Model function registry: maps model_type to (function, required_param_names)
    PARAMETRIC_MODEL_FUNCTIONS = {
        # Mechanistic models (ODE-based)
        "mech_logistic": (mech_logistic_model, ["mu", "K", "N0"]),
        "mech_gompertz": (mech_gompertz_model, ["mu", "K", "N0"]),
        "mech_richards": (mech_richards_model, ["mu", "K", "N0", "beta"]),
        "mech_baranyi": (mech_baranyi_model, ["mu", "K", "N0", "h0"]),
        # Phenomenological models (ln-space)
        "phenom_logistic": (phenom_logistic_model_ln, ["A", "mu_max", "lam"]),
        "phenom_gompertz": (phenom_gompertz_model_ln, ["A", "mu_max", "lam"]),
        "phenom_gompertz_modified": (
            phenom_gompertz_modified_model_ln,
            ["A", "mu_max", "lam", "alpha", "t_shift"],
        ),
        "phenom_richards": (phenom_richards_model_ln, ["A", "mu_max", "lam", "nu"]),
    }

    if model_type not in PARAMETRIC_MODEL_FUNCTIONS:
        raise ValueError(
            f"Unknown model type: {model_type}. "
            f"Must be one of {list(PARAMETRIC_MODEL_FUNCTIONS.keys())}"
        )

    model_func, param_names = PARAMETRIC_MODEL_FUNCTIONS[model_type]
    model_args = [params[name] for name in param_names]

    return model_func(t, *model_args)
