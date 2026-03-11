import numpy as np

import growthcurves as gc


def logistic_growth(t, baseline, N0, K, mu, lag):
    """Logistic growth model with smooth transition through lag phase"""
    # Standard logistic formula centered at lag time
    # This creates a smooth S-curve with inflection point at t = lag + (K - N0) / N0
    factor = (K - N0) / N0
    growth = K / (1 + factor * np.exp(-mu * (t - lag)))
    return baseline + growth, lag + np.log(factor) / mu


def test_fit_parametric():
    np.random.seed(42)
    expected = {
        "params": {
            "mu": 0.1499259523503361,
            "K": 0.4499768424238933,
            "N0": 0.0006250371640236538,
            "y0": 0.05000966836757205,
            "fit_t_min": 0.0,
            "fit_t_max": 87.8,
        },
        "model_type": "mech_logistic",
    }
    expected = expected["params"]
    # Test that the function returns expected values for known parameters
    n_points = 440
    measurement_interval_minutes = 12
    t = np.array([(measurement_interval_minutes * n) / 60 for n in range(n_points)])
    K = 0.45
    mu = 0.15
    N0 = 0.05
    baseline = N0
    lag = 30.0
    N, _ = logistic_growth(t, baseline=baseline, N0=N0, K=K, mu=mu, lag=lag)
    actual = gc.parametric.fit_parametric(t, N, method="mech_logistic")
    actual = actual["params"]
    for k, v in expected.items():
        assert k in actual, f"Parameter {k} not found in actual output"
        assert np.isclose(actual[k], v), f"Parameter {k} does not match expected value"
