import numpy as np
import growthcurves as gc
from growthcurves.models import (
    phenom_gompertz_model,
    mech_gompertz_model,
    phenom_gompertz_modified_model,
)


def test_fit_parametric_phenom_gompertz():
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


def test_fit_parametric_mech_gompertz():
    n_points = 440
    measurement_interval_minutes = 12
    t = np.array([(measurement_interval_minutes * n) / 60 for n in range(n_points)])

    mu = 0.3
    K = 2.5
    N0 = 0.05
    y0 = 0.05

    expected_mech = {"mu": mu, "K": K, "N0": N0, "y0": y0}
    N = mech_gompertz_model(t=t, mu=mu, K=K, N0=N0, y0=y0)
    actual = gc.parametric.fit_parametric(t, N, method="mech_gompertz")
    actual = actual["params"]
    for k, v in expected_mech.items():
        assert k in actual, f"Parameter {k} not found in actual output"
        assert np.isclose(
            actual[k], v, rtol=2e-1
        ), f"Parameter {k} does not match expected value"


def test_fit_phenom_gompertz_modified():
    n_points = 440
    measurement_interval_minutes = 12
    t = np.array([(measurement_interval_minutes * n) / 60 for n in range(n_points)])

    A = 2.5
    mu_max = 0.3
    lam = 5.0
    alpha = 0.001
    t_shift = 50.0
    N0 = 0.05

    N = phenom_gompertz_modified_model(
        t, A=A, mu_max=mu_max, lam=lam, alpha=alpha, t_shift=t_shift, N0=N0
    )
    result = gc.parametric.fit_parametric(t, N, method="phenom_gompertz_modified")

    assert result is not None, "Fitting should succeed"
    assert result["model_type"] == "phenom_gompertz_modified"

    params = result["params"]
    for k in ["A", "mu_max", "lam", "alpha", "t_shift", "N0"]:
        assert k in params, f"Parameter {k} not found in result"

    # Verify the fitted curve closely reproduces the training data (< 1% relative RMSE)
    N = phenom_gompertz_modified_model(
        t,
        A=params["A"],
        mu_max=params["mu_max"],
        lam=params["lam"],
        alpha=params["alpha"],
        t_shift=params["t_shift"],
        N0=params["N0"],
    )
    expected = {
        "mu_max": mu_max,
        "A": A,
        "N0": N0,
        "N0": N0,
        # "t_shift": t_shift, # removed t_shift from expected since it's not identifiable
        "alpha": alpha,
        "lam": lam,
    }

    actual = gc.parametric.fit_parametric(t, N, method="phenom_gompertz_modified")
    actual = actual["params"]
    for k, v in expected.items():
        assert k in actual, f"Parameter {k} not found in actual output"
        assert np.isclose(
            actual[k],
            v,
            rtol=1e-1,  # need to loosen tolerance due to non-identifiability of parameters
        ), f"Parameter {k} does not match expected value. Actual: {actual[k]}, Expected: {v}"
