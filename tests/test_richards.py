import numpy as np

import growthcurves as gc
from growthcurves.models import (
    evaluate_parametric_model,
    log_to_linear,
    mech_richards_model,
    phenom_richards_model_ln,
)


def test_fit_parametric_phenom_richards():
    n_points = 440
    measurement_interval_minutes = 12
    t = np.array([(measurement_interval_minutes * n) / 60 for n in range(n_points)])

    A = 2.5
    mu_max = 0.3
    lam = 5.0
    nu = 1.0
    expected = {"A": A, "mu_max": mu_max, "lam": lam, "nu": nu}

    # test phenomenological Richards model fitting

    N = log_to_linear(
        phenom_richards_model_ln(t, A=A, mu_max=mu_max, lam=lam, nu=nu), 0.05
    )
    actual = gc.parametric.fit_parametric(t, N, method="phenom_richards")
    actual = actual["params"]
    for k, v in expected.items():
        assert k in actual, f"Parameter {k} not found in actual output"
        assert np.isclose(actual[k], v, rtol=2e-1), (
            f"Phenomenological parameter {k} does not match expected value. "
            f"Actual: {actual[k]}, Expected: {v}"
        )


def test_fit_parametric_mech_richards():
    n_points = 440
    measurement_interval_minutes = 12
    t = np.array([(measurement_interval_minutes * n) / 60 for n in range(n_points)])

    expected = {"K": 2.41, "mu": 0.4, "beta": 2.64, "N0": 0.016}
    N = mech_richards_model(
        t=t,
        mu=expected["mu"],
        K=expected["K"],
        N0=expected["N0"],
        beta=expected["beta"],
    )
    actual = gc.parametric.fit_parametric(t, N, method="mech_richards")
    actual = actual["params"]
    for k, v in expected.items():
        assert k in actual, f"Parameter {k} not found in actual output"
        assert np.isclose(actual[k], v, rtol=2e-1), (
            f"Mechanistic parameter {k} does not match expected value. "
            f"Actual: {actual[k]}, Expected: {v}"
        )


def test_extract_stats_mu_max_mech_richards():
    """Reported mu_max matches the true max d ln(N)/dt even on a sparse time grid."""
    mu = 0.5
    K = 1.0
    N0 = 0.05
    beta = 1.5
    params = {"mu": mu, "K": K, "N0": N0, "beta": beta}

    # Ground truth: numerical derivative of the true curve on a very fine grid
    t_fine = np.linspace(0, 24, 20001)
    N_fine = evaluate_parametric_model(t_fine, "mech_richards", params)
    true_mu_max = np.max(np.gradient(np.log(N_fine), t_fine))

    # Sparse observations (~2 h apart) of the same curve
    t = np.linspace(0, 24, 12)
    N = evaluate_parametric_model(t, "mech_richards", params)

    fit_result = gc.parametric.fit_parametric(t, N, method="mech_richards")
    stats = gc.inference.extract_stats(fit_result, t, N)
    assert np.isclose(
        stats["mu_max"], true_mu_max, rtol=2e-2
    ), f"mu_max {stats['mu_max']:.3f} does not match true value {true_mu_max:.3f}"


def test_compute_analytical_mu_max_mech_richards():
    """Analytic mu_max on the true parameters matches the fine-grid derivative."""
    mu = 0.5
    K = 1.0
    N0 = 0.05
    beta = 1.5
    params = {"mu": mu, "K": K, "N0": N0, "beta": beta}

    # Ground truth: numerical derivative of the true curve on a very fine grid
    t_fine = np.linspace(0, 24, 20001)
    N_fine = evaluate_parametric_model(t_fine, "mech_richards", params)
    true_mu_max = np.max(np.gradient(np.log(N_fine), t_fine))

    mu_max = gc.inference.compute_analytical_mu_max("mech_richards", params, 0.0, 24.0)
    assert np.isclose(
        mu_max, true_mu_max, rtol=1e-2
    ), f"mu_max {mu_max:.3f} does not match true value {true_mu_max:.3f}"
