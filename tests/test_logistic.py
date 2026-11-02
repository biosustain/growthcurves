import numpy as np

import growthcurves as gc
from growthcurves.models import (
    evaluate_parametric_model,
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
    N0 = 0.05
    expected = {"mu": mu, "K": K, "N0": N0}

    # test mechanistic logistic model fitting

    N = mech_logistic_model(t=t, mu=mu, K=K, N0=N0)
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
    N0 = 0.05
    expected_phenom = {"A": A, "mu_max": mu_max, "lam": lam, "N0": N0}

    ln_ratio = phenom_logistic_model_ln(t, A=A, mu_max=mu_max, lam=lam)
    N = log_to_linear(ln_ratio, N0=N0)
    actual = gc.parametric.fit_parametric(t, N, method="phenom_logistic")
    actual = actual["params"]
    print(actual)
    for k, v in expected_phenom.items():
        assert k in actual, f"Parameter {k} not found in actual output"
        assert np.isclose(
            actual[k], v, rtol=1e-3
        ), f"Parameter {k} does not match expected value"


def test_extract_stats_mu_max_mech_logistic():
    """Reported mu_max matches the true max d ln(N)/dt even on a sparse time grid."""
    mu = 0.5
    K = 1.0
    N0 = 0.05
    params = {"mu": mu, "K": K, "N0": N0}

    # Ground truth: numerical derivative of the true curve on a very fine grid
    t_fine = np.linspace(0, 24, 20001)
    N_fine = evaluate_parametric_model(t_fine, "mech_logistic", params)
    true_mu_max = np.max(np.gradient(np.log(N_fine), t_fine))

    # Sparse observations (~2 h apart) of the same curve
    t = np.linspace(0, 24, 12)
    N = evaluate_parametric_model(t, "mech_logistic", params)

    fit_result = gc.parametric.fit_parametric(t, N, method="mech_logistic")
    stats = gc.inference.extract_stats(fit_result, t, N)
    assert np.isclose(
        stats["mu_max"], true_mu_max, rtol=2e-2
    ), f"mu_max {stats['mu_max']:.3f} does not match true value {true_mu_max:.3f}"


def test_compute_analytical_mu_max_mech_logistic():
    """Analytic mu_max on the true parameters matches the fine-grid derivative."""
    mu = 0.5
    K = 1.0
    N0 = 0.05
    params = {"mu": mu, "K": K, "N0": N0}

    # Ground truth: numerical derivative of the true curve on a very fine grid
    t_fine = np.linspace(0, 24, 20001)
    N_fine = evaluate_parametric_model(t_fine, "mech_logistic", params)
    true_mu_max = np.max(np.gradient(np.log(N_fine), t_fine))

    mu_max = gc.inference.compute_analytical_mu_max("mech_logistic", params, 0.0, 24.0)
    assert np.isclose(
        mu_max, true_mu_max, rtol=1e-2
    ), f"mu_max {mu_max:.3f} does not match true value {true_mu_max:.3f}"
