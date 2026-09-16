import numpy as np

import growthcurves as gc
from growthcurves.models import (
    log_to_linear,
    mech_logistic_model,
    phenom_logistic_model_ln,
)


def test_fit_parametric_mech_logistic():
    n_points = 440
    measurement_interval_minutes = 12
    t = np.array([(measurement_interval_minutes * n) / 60 for n in range(n_points)])

    mu = 0.15
    K = 0.45
    N_init = 0.05
    expected = {"mu": mu, "K": K, "N_init": N_init}

    # test mechanistic logistic model fitting

    N = mech_logistic_model(t=t, mu=mu, K=K, N_init=N_init)
    actual = gc.parametric.fit_parametric(t, N, method="mech_logistic")
    actual = actual["params"]
    for k, v in expected.items():
        assert k in actual, f"Parameter {k} not found in actual output"
        assert np.isclose(
            actual[k], v, rtol=2e-2
        ), f"Parameter {k} does not match expected value"


def test_fit_parametric_mech_phenom():
    """Test phenomenological logistic model fitting."""
    n_points = 440
    measurement_interval_minutes = 12
    t = np.array([(measurement_interval_minutes * n) / 60 for n in range(n_points)])

    A = 3.0
    mu_max = 0.3
    lam = 5.0
    N_init = 0.05
    expected_phenom = {"A": A, "mu_max": mu_max, "lam": lam, "N_init": N_init}

    ln_ratio = phenom_logistic_model_ln(t, A=A, mu_max=mu_max, lam=lam)
    N = log_to_linear(ln_ratio, N_init=N_init)
    actual = gc.parametric.fit_parametric(t, N, method="phenom_logistic")
    actual = actual["params"]
    print(actual)
    for k, v in expected_phenom.items():
        assert k in actual, f"Parameter {k} not found in actual output"
        assert np.isclose(
            actual[k], v, rtol=1e-3
        ), f"Parameter {k} does not match expected value"
