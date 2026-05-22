import numpy as np

import growthcurves as gc
from growthcurves.models import mech_logistic_model, phenom_logistic_model


def test_fit_parametric():
    n_points = 440
    measurement_interval_minutes = 12
    t = np.array([(measurement_interval_minutes * n) / 60 for n in range(n_points)])

    mu = 0.15
    K = 0.45
    N0 = 0.05
    y0 = 0.05
    expected = {"mu": mu, "K": K, "N0": N0, "y0": y0}

    # test mechanistic logistic model fitting

    N = mech_logistic_model(t=t, mu=mu, K=K, N0=N0, y0=y0)
    actual = gc.parametric.fit_parametric(t, N, method="mech_logistic")
    actual = actual["params"]
    for k, v in expected.items():
        assert k in actual, f"Parameter {k} not found in actual output"
        assert np.isclose(
            actual[k], v, rtol=2e-2
        ), f"Parameter {k} does not match expected value"

    # test phenomenological logistic model fitting

    A = 2.5
    mu_max = 0.3
    lam = 5.0
    N0_phenom = 0.05
    expected_phenom = {"A": A, "mu_max": mu_max, "lam": lam, "N0": N0_phenom}

    N = phenom_logistic_model(t, A=A, mu_max=mu_max, lam=lam, N0=N0_phenom)
    actual = gc.parametric.fit_parametric(t, N, method="phenom_logistic")
    actual = actual["params"]
    for k, v in expected_phenom.items():
        assert k in actual, f"Parameter {k} not found in actual output"
        assert np.isclose(
            actual[k], v, rtol=1e-3
        ), f"Parameter {k} does not match expected value"
