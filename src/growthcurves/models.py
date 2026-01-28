"""Growth curve models.

This module defines parametric growth model functions (Richards, Logistic,
Gompertz, Gaussian) that operate in linear OD space (not log-transformed).
"""

import numpy as np


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


def gaussian(t, amplitude, center, sigma):
    """Symmetric Gaussian bell-shaped curve."""
    sigma = np.maximum(sigma, 1e-12)
    return amplitude * np.exp(-((t - center) ** 2) / (2 * sigma**2))
