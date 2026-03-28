import numpy as np

import growthcurves as gc
from growthcurves.models import mech_baranyi_model


def test_fit_parametric():
    n_points = 440
    measurement_interval_minutes = 12
    t = np.array([(measurement_interval_minutes * n) / 60 for n in range(n_points)])

    mu = 0.3
    K = 2.5
    N0 = 0.05
    h0 = 1.0
    y0 = 0.05
    expected = {"mu": mu, "K": K, "N0": N0, "h0": h0, "y0": y0}

    # test mechanistic Baranyi-Roberts model fitting

    N = mech_baranyi_model(t=t, mu=mu, K=K, N0=N0, h0=h0, y0=y0)
    actual = gc.parametric.fit_parametric(t, N, method="mech_baranyi")
    actual = actual["params"]
    for k, v in expected.items():
        assert k in actual, f"Parameter {k} not found in actual output"
        assert np.isclose(
            actual[k], v, rtol=2e-1
        ), f"Parameter {k} does not match expected value"
