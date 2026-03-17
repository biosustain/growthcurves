import numpy as np
import growthcurves as gc
from growthcurves.models import phenom_gompertz_model, mech_gompertz_model


def test_fit_parametric():
    n_points = 440
    measurement_interval_minutes = 12
    t = np.array([(measurement_interval_minutes * n) / 60 for n in range(n_points)])

    A = 2.5
    mu_max = 0.3
    lam = 5.0
    N0 = 0.05
    mu = 0.3
    K = 2.5
    y0 = 0.05

    # test phenomenological Gompertz model fitting

    expected_phenom = {"A": A, "mu_max": mu_max, "lam": lam, "N0": N0}
    N = phenom_gompertz_model(t, A=A, mu_max=mu_max, lam=lam, N0=N0)
    actual = gc.parametric.fit_parametric(t, N, method="phenom_gompertz")
    actual = actual["params"]
    for k, v in expected_phenom.items():
        assert k in actual, f"Parameter {k} not found in actual output"
        assert np.isclose(
            actual[k], v, rtol=2e-1
        ), f"Parameter {k} does not match expected value"

    # test mechanistic Gompertz model fitting

    expected_mech = {"mu": mu, "K": K, "N0": N0, "y0": y0}
    N = mech_gompertz_model(t=t, mu=mu, K=K, N0=N0, y0=y0)
    actual = gc.parametric.fit_parametric(t, N, method="mech_gompertz")
    actual = actual["params"]
    for k, v in expected_mech.items():
        assert k in actual, f"Parameter {k} not found in actual output"
        assert np.isclose(
            actual[k], v, rtol=2e-1
        ), f"Parameter {k} does not match expected value"
