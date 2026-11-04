"""Parametric model fitting functions for growth curves.

This module provides functions to fit parametric growth models:

- Mechanistic models (ODE-based): mech_logistic, mech_gompertz, mech_richards,
  mech_baranyi
- Phenomenological models (ln-space): phenom_logistic, phenom_gompertz,
  phenom_gompertz_modified, phenom_richards

Mechanistic Models operate in linear OD space while Phenomenological Models
operate in the log space.

Abbreviations:
- OD: Optical Density
- ODE: Ordinary Differential Equation
"""

import inspect

import numpy as np
from scipy.optimize import curve_fit

from .inference import validate_data
from .models import (
    get_all_models,
    mech_baranyi_model,
    mech_gompertz_model,
    mech_logistic_model,
    mech_richards_model,
    phenom_gompertz_model_ln,
    phenom_gompertz_modified_model_ln,
    phenom_logistic_model_ln,
    phenom_richards_model_ln,
)

# -----------------------------------------------------------------------------
# Helper Functions for mechanistic model fitting
# -----------------------------------------------------------------------------


def _fit_model_generic(
    t: np.ndarray,
    N: np.ndarray,
    model_func,
    param_names: list,
    p0_func,
    bounds_func,
    model_type: str,
    log_space: bool = False,
):
    """
    Generic wrapper for fitting parametric growth models.

    This function encapsulates the common pattern used across all parametric
    model fitting functions, reducing code duplication.

    Parameters:
        t: Time array
        N: OD values
        model_func: Model function to fit (e.g., mech_logistic_model)
        param_names: List of parameter names in order
        p0_func: Function that takes K_init and returns p0
        bounds_func: Function that takes (K_init, t) and returns bounds
        model_type: String identifier for the model

    Returns:
        Dict with 'params' and 'model_type', or None if fitting fails

    """
    t, N = validate_data(t, N)
    if t is None:
        return None

    # Initial guess and bounds; K starts at the max OD
    K_init = np.max(N)
    p0 = p0_func(K_init)
    bounds = bounds_func(K_init, t)

    # Fit the model
    if log_space:
        N_pos = np.maximum(N, 1e-8)

        def log_model(tt, *p):
            return np.log(np.maximum(model_func(tt, *p), 1e-8))

        params, _ = curve_fit(
            log_model, t, np.log(N_pos), p0=p0, bounds=bounds, maxfev=20000
        )
    else:
        params, _ = curve_fit(model_func, t, N, p0=p0, bounds=bounds, maxfev=20000)

    # Return structured result
    return {
        "params": dict(zip(param_names, params)),
        "model_type": model_type,
    }


# -----------------------------------------------------------------------------
# region :Mechanistic Model Fitting Functions (ODE-based)
# -----------------------------------------------------------------------------


def fit_mech_logistic(t: np.ndarray, N: np.ndarray):
    """
    Fit mechanistic logistic model (ODE) to growth N(t). N(t) could be Optical
    Density (OD) or any other measurement depending on variable t (which is time).

    ODE: dN/dt = μ * (1 - N/K) * N

    Assumes input data is baseline-corrected (no additive offset).

    Parameters:
        t: Time array (hours)
        N: OD values

    Returns:
        Dict with 'params' and 'model_type', or None if fitting fails.

    """

    return _fit_model_generic(
        t,
        N,
        model_func=mech_logistic_model,
        param_names=["mu", "K", "N0"],
        p0_func=lambda K: [0.5, K, 0.001],
        bounds_func=lambda K, t: (
            [0.0001, 0.001, 1e-6],
            [10, np.inf, 10],
        ),
        model_type="mech_logistic",
        log_space=True,
    )


def fit_mech_gompertz(t: np.ndarray, N: np.ndarray):
    """
    Fit mechanistic Gompertz model (ODE) to growth data N(t). N(t) could be Optical
    Density (OD) or any other measurement depending on variable t (which is time).

    ODE: dN/dt = μ * log(K/N) * N

    Assumes input data is baseline-corrected (no additive offset).

    Parameters:
        t: Time array (hours)
        N: OD values

    Returns:
        Dict with 'params' and 'model_type', or None if fitting fails.

    Note:
        The mechanistic Gompertz model can be numerically challenging to fit
        due to the logarithmic term in the ODE. If fitting fails or produces
        poor results, consider using mech_logistic, mech_richards, or
        phenom_gompertz instead.

    """
    return _fit_model_generic(
        t,
        N,
        model_func=mech_gompertz_model,
        param_names=["mu", "K", "N0"],
        p0_func=lambda K: [0.05, K, 0.01],
        # ToDo: This should be a function to avoid hardcoding K bounds.
        bounds_func=lambda K, t: (
            [0.0001, 0.01, 1e-4],
            [2, np.inf, 1],
        ),
        model_type="mech_gompertz",
        log_space=True,
    )


def fit_mech_richards(t: np.ndarray, N: np.ndarray):
    """
    Fit mechanistic Richards model (ODE) to growth N(t). N(t) could be Optical Density
    (OD) or any other measurement depending on variable t (which is time).

    ODE: dN/dt = μ * (1 - (N/K)^β) * N

    Assumes input data is baseline-corrected (no additive offset).

    Parameters:
        t: Time array (hours)
        N: OD values

    Returns:
        Dict with 'params' and 'model_type', or None if fitting fails.
    """
    return _fit_model_generic(
        t,
        N,
        model_func=mech_richards_model,
        param_names=["mu", "K", "N0", "beta"],
        p0_func=lambda K: [0.5, K, 0.001, 1.0],
        bounds_func=lambda K, t: (
            [0.0001, 0.001, 1e-6, 0.01],
            [10, np.inf, 10, 100],
        ),
        model_type="mech_richards",
        log_space=True,
    )


def fit_mech_baranyi(t: np.ndarray, N: np.ndarray):
    """
    Fit mechanistic Baranyi-Roberts model (ODE) to growth N(t). N(t) could be Optical
    Density (OD) or any other measurement depending on variable t (which is time).

    ODE: dN/dt = μ * A(t) * (1 - N/K) * N

    where A(t) = exp(μ*t) / (exp(h0) - 1 + exp(μ*t))


    Assumes input data is baseline-corrected (no additive offset).

    Parameters:
        t: Time array (hours)
        N: OD values

    Returns:
        Dict with 'params' and 'model_type', or None if fitting fails.
    """

    def p0_baranyi(K):
        mu_init = 0.5
        h0_init = 1.0
        N0_init = 0.001
        return [mu_init, K, N0_init, h0_init]

    def bounds_baranyi(K, t):
        return (
            [0.0001, 0.001, 1e-6, 0],
            [10, np.inf, 10, t.max() * 10],
        )

    return _fit_model_generic(
        t,
        N,
        model_func=mech_baranyi_model,
        param_names=["mu", "K", "N0", "h0"],
        p0_func=p0_baranyi,
        bounds_func=bounds_baranyi,
        model_type="mech_baranyi",
        log_space=True,
    )


# endregion
# -----------------------------------------------------------------------------
# Helper Functions for phenomenological model fitting
# -----------------------------------------------------------------------------


def _estimate_lag_time(t: np.ndarray, dN: np.ndarray, threshold_frac: float = 0.1):
    """
    Estimate lag t from growth rate threshold.

    Parameters:
        t: Time array
        dN: First derivative of OD
        threshold_frac: Fraction of max derivative to use as threshold

    Returns:
        Estimated lag t (t when growth rate exceeds threshold)

    """
    threshold = threshold_frac * np.max(dN)
    lag_idx = np.where(dN > threshold)[0]
    return t[lag_idx[0]] if len(lag_idx) > 0 else t[0]


def _fit_phenom_model(
    t: np.ndarray,
    N: np.ndarray,
    model_func,
    model_type: str,
    extra_params=None,
    p0_overrides=None,
):
    """
    Generic wrapper for fitting phenomenological growth models.

    Fits ln(N) = model_func(t, A, mu_max, lam, *extra) + ln(N0), with ln(N0) as a
    free parameter. N0 is fit jointly rather than estimated as min(N): the
    ln-ratio term is not zero at the smallest sampled time, so using min(N)
    systematically overestimates the true baseline, biasing the other fitted
    parameters.

    Parameters:
        t: Time array (hours)
        N: OD values
        model_func: Phenomenological model returning ln(Nt/N0), with signature
            (t, A, mu_max, lam, *extra)
        model_type: String identifier for the model
        extra_params: Optional function of the validated t returning a dict of
            {name: (initial, lower, upper)} for parameters after lam
        p0_overrides: Optional function of the validated (t, N) returning a dict
            of {name: initial} replacing the shared initial guesses of A, mu_max
            or lam for models where they are a poor starting point

    Returns:
        Dict with 'params' (A, mu_max, lam, *extra, N0) and 'model_type', or None
        if fitting fails.
    """
    t, N = validate_data(t, N)
    if t is None:
        return None

    # Estimate initial parameters
    N0_init = max(float(np.min(N)), 0.01)  # ensure that N0 is positive
    N_max = max(float(np.max(N)), N0_init + 0.02)  # ensure that N_max > N0
    A_init = max(np.log(N_max / N0_init), 0.01)  # ensure that the log-ratio > 0
    gradient_ln_N = np.gradient(np.log(N), t)
    mu_max_init = np.max(gradient_ln_N)
    lam_init = _estimate_lag_time(t, gradient_ln_N)

    # Initial parameter guess and bounds for A, mu_max, lam, then extra parameters
    param_names = ["A", "mu_max", "lam"]
    p0 = [A_init, mu_max_init, lam_init]
    if p0_overrides:
        overrides = p0_overrides(t, N)
        p0 = [overrides.get(name, init) for name, init in zip(param_names, p0)]
    lower = [0.01, 0.0001, 0]
    upper = [20, 10, t.max()]
    for name, (init, lo, hi) in (extra_params(t) if extra_params else {}).items():
        param_names.append(name)
        p0.append(init)
        lower.append(lo)
        upper.append(hi)
    p0 = np.clip(p0, lower, upper)

    def ln_model(tt, *p):
        return model_func(tt, *p[:-1]) + p[-1]

    # Fit the model directly on ln(N), with ln_N0 as the last free parameter
    params, _ = curve_fit(
        ln_model,
        t,
        np.log(N),
        p0=[*p0, np.log(N0_init)],
        bounds=([*lower, -np.inf], [*upper, np.inf]),
        maxfev=20000,
    )

    return {
        "params": {
            **dict(zip(param_names, (float(p) for p in params[:-1]))),
            "N0": float(np.exp(params[-1])),
        },
        "model_type": model_type,
    }


# -----------------------------------------------------------------------------
# Phenomenological Model Fitting Functions (ln-space)
# -----------------------------------------------------------------------------
def fit_fct(t: np.ndarray, N: np.ndarray, fct):
    """
    Fit an arbitrary model function fct(t, *params) directly to N(t).

    No transformation is applied: fct must return values on the same scale as N.
    A fitted parameter named 'ln_N0' is returned as 'N0' = exp(ln_N0).

    Parameters:
        t: Time array (hours)
        N: Observed values
        fct: Model function whose first argument is t

    Returns:
        Dict with 'params' (named after fct's arguments) and 'model_type', or None
        if fitting fails.
    """

    t, N = validate_data(t, N)
    if t is None:
        return None

    # Fit the model directly on N, using scipy's default initial guess
    params, _ = curve_fit(fct, t, N)

    # skip the first parameter (t) to get the fitted parameter names in order
    param_names = list(inspect.signature(fct).parameters)[1:]
    fitted_params = dict(zip(param_names, (float(p) for p in params)))
    if "ln_N0" in fitted_params:
        fitted_params["N0"] = float(np.exp(fitted_params.pop("ln_N0")))

    fct_name = getattr(fct, "__name__", None)
    model_type = f"fit:{fct_name}" if fct_name else "phenom_unnamed"

    return {
        "params": fitted_params,
        "model_type": model_type,
    }


def fit_phenom_logistic(t: np.ndarray, N: np.ndarray):
    """
    Fit non-standard phenomenological logistic model to OD data
    N(t) = N0 * exp(ln_ratio(t)).

    ln_ratio(t) = ln(Nt/N0) = A / (1 + exp(4 * μ_max * (λ - t) / A + 2))

    Parameters:
        t: Time array (hours)
        N: OD values

    Returns:
        Dict with 'params' (A, mu_max, lam, N0) and 'model_type', or None if
        fitting fails.
    """
    return _fit_phenom_model(t, N, phenom_logistic_model_ln, "phenom_logistic")


def fit_phenom_gompertz(t: np.ndarray, N: np.ndarray):
    """
    Fit phenomenological Gompertz model to OD data N(t) = N0 * exp(ln_ratio(t)).

    ln(Nt/N0) = A * exp(-exp(μ_max * e * (λ - t) / A + 1))

    Parameters:
        t: Time array (hours)
        N: OD values

    Returns:
        Dict with 'params' (A, mu_max, lam, N0) and 'model_type', or None if
        fitting fails.
    """
    return _fit_phenom_model(t, N, phenom_gompertz_model_ln, "phenom_gompertz")


def fit_phenom_gompertz_modified(t: np.ndarray, N: np.ndarray):
    """
    Fit phenomenological modified Gompertz model with decay term to OD data
    N(t) = N0 * exp(ln_ratio(t)).

    ln(Nt/N0) = A * exp(-exp(μ_max * e * (λ - t) / A + 1)) + A * exp(α * (t - t_shift))

    Parameters:
        t: Time array (hours)
        N: OD values

    Returns:
        Dict with 'params' (A, mu_max, lam, alpha, t_shift, N0) and 'model_type',
        or None if fitting fails.
    """
    return _fit_phenom_model(
        t,
        N,
        phenom_gompertz_modified_model_ln,
        "phenom_gompertz_modified",
        extra_params=lambda t: {
            "alpha": (0.05, -1, 1),
            "t_shift": (t.max() * 0.8, 0, t.max()),
        },
        # The shared A guess, ln(max(N) / N0), includes the second phase, which
        # traps the fit in a local minimum. The second term equals A at t_shift,
        # so ln(N/N0) ~ 2A there: halve the observed amplitude instead.
        p0_overrides=lambda _, N: {"A": np.log(np.max(N) / np.min(N)) / 2},
    )


def fit_phenom_richards(t: np.ndarray, N: np.ndarray):
    """
    Fit phenomenological Richards model to OD data N(t) = N0 * exp(ln_ratio(t)).

    ln(Nt/N0)= A * (1 + ν * exp(1 + ν + μ_max * (1 + ν)^(1 + 1/ν) * (λ - t) / A))^(-1/ν)

    Parameters:
        t: Time array (hours)
        N: OD values

    Returns:
        Dict with 'params' (A, mu_max, lam, nu, N0) and 'model_type', or None if
        fitting fails.
    """
    return _fit_phenom_model(
        t,
        N,
        phenom_richards_model_ln,
        "phenom_richards",
        extra_params=lambda _: {"nu": (1.0, 0.01, 100)},
    )


# -----------------------------------------------------------------------------
# Main Fitting Dispatcher
# -----------------------------------------------------------------------------

FITTING_FUNCTIONS = {
    "mech_logistic": fit_mech_logistic,
    "mech_gompertz": fit_mech_gompertz,
    "mech_richards": fit_mech_richards,
    "mech_baranyi": fit_mech_baranyi,
    "phenom_logistic": fit_phenom_logistic,
    "phenom_gompertz": fit_phenom_gompertz,
    "phenom_gompertz_modified": fit_phenom_gompertz_modified,
    "phenom_richards": fit_phenom_richards,
}


def fit_parametric(
    t: np.ndarray, N: np.ndarray, method: str = "mech_logistic", **kwargs
):
    """Fit a growth model to N (mechanistic) or ln(N/N0) (phenomenological).


    Parameters
    ----------
    t : Iterable[float]
        Time array (hours)
    N : Iterable[float]
        OD values (in linear space)
    method : str, optional
        Model type string. Options:

        Mechanistic (ODE):
            - "mech_logistic", "mech_gompertz", "mech_richards", "mech_baranyi"

        Phenomenological (ln-space):
            - "phenom_logistic", "phenom_gompertz", "phenom_gompertz_modified",\
 "phenom_richards"

    Returns
    -------
    Optional[dict]
        Fit result dict or None if fitting fails.

    Raises
    ------
    ValueError
        Unknown method string.
    """

    fit_func = FITTING_FUNCTIONS.get(method)

    if fit_func is None:
        raise ValueError(
            f"Unknown method '{method}'. Must be one of {list(get_all_models())}."
        )

    result = fit_func(t, N)
    if result is not None:
        # ? validate_data is called in fitting functions. recomputation could be
        # ? avoided by returning time_valid from validate_data in fitting functions.
        time_valid, _ = validate_data(t, N)
        if time_valid is None:
            return None
        result["params"]["fit_t_min"] = float(np.min(time_valid))
        result["params"]["fit_t_max"] = float(np.max(time_valid))
    return result
