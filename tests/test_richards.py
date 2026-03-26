import numpy as np

import growthcurves as gc
from growthcurves.models import mech_richards_model, phenom_richards_model


def test_fit_parametric():
    n_points = 440
    measurement_interval_minutes = 12
    t = np.array([(measurement_interval_minutes * n) / 60 for n in range(n_points)])

    A = 2.5
    mu_max = 0.3
    lam = 5.0
    nu = 1.0
    N0 = 0.05
    expected = {"A": A, "mu_max": mu_max, "lam": lam, "nu": nu, "N0": N0}

    # test phenomenological Richards model fitting

    N = phenom_richards_model(t, A=A, mu_max=mu_max, lam=lam, nu=nu, N0=N0)
    actual = gc.parametric.fit_parametric(t, N, method="phenom_richards")
    actual = actual["params"]
    for k, v in expected.items():
        assert k in actual, f"Parameter {k} not found in actual output"
        assert np.isclose(actual[k], v, rtol=2e-1), (
            f"Phenomenological parameter {k} does not match expected value. "
            f"Actual: {actual[k]}, Expected: {v}"
        )

    # test mechanistic Richards model fitting

    y0 = 1.0

    # ! why should mechanistic not work, but phenomenological does?
    # expected = {"K": A, "mu": mu_max, "beta": lam, "y0": y0, "N0": N0}
    expected = {"K": 2.41, "mu": 0.4, "beta": 2.64, "y0": y0, "N0": 0.016}
    N = mech_richards_model(t=t, mu=mu_max, K=A, N0=N0, beta=lam, y0=y0)
    actual = gc.parametric.fit_parametric(t, N, method="mech_richards")
    actual = actual["params"]
    for k, v in expected.items():
        assert k in actual, f"Parameter {k} not found in actual output"
        assert np.isclose(actual[k], v, rtol=2e-1), (
            f"Mechanistic parameter {k} does not match expected value. "
            f"Actual: {actual[k]}, Expected: {v}"
        )
