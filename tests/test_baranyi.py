import numpy as np

import growthcurves as gc
from growthcurves.models import evaluate_parametric_model, mech_baranyi_model


def test_fit_parametric():
    n_points = 440
    measurement_interval_minutes = 12
    t = np.array([(measurement_interval_minutes * n) / 60 for n in range(n_points)])

    mu = 0.3
    K = 2.5
    N0 = 0.05
    h0 = 1.0
    expected = {"mu": mu, "K": K, "N0": N0, "h0": h0}

    # test mechanistic Baranyi-Roberts model fitting

    N = mech_baranyi_model(t=t, mu=mu, K=K, N0=N0, h0=h0)
    fit_result = gc.parametric.fit_parametric(t, N, method="mech_baranyi")
    actual = fit_result["params"]
    for k, v in expected.items():
        assert k in actual, f"Parameter {k} not found in actual output"
        assert np.isclose(
            actual[k], v, rtol=2e-1
        ), f"Parameter {k} does not match expected value"

    # Lag time is h0 / mu (Baranyi & Roberts 1994), not h0
    stats = gc.inference.extract_stats(fit_result, t, N)
    assert np.isclose(
        stats["exp_phase_start"], h0 / mu, rtol=2e-1
    ), "Lag time does not match h0 / mu"


def test_extract_stats_mu_max_mech_baranyi():
    """Reported mu_max matches the true max d ln(N)/dt even on a sparse time grid."""
    mu = 0.5
    K = 1.0
    N0 = 0.05
    h0 = 2.0
    params = {"mu": mu, "K": K, "N0": N0, "h0": h0}

    # Ground truth: numerical derivative of the true curve on a very fine grid
    t_fine = np.linspace(0, 24, 20001)
    N_fine = evaluate_parametric_model(t_fine, "mech_baranyi", params)
    true_mu_max = np.max(np.gradient(np.log(N_fine), t_fine))

    # Sparse observations (~2 h apart) of the same curve
    t = np.linspace(0, 24, 12)
    N = evaluate_parametric_model(t, "mech_baranyi", params)

    fit_result = gc.parametric.fit_parametric(t, N, method="mech_baranyi")
    stats = gc.inference.extract_stats(fit_result, t, N)
    assert np.isclose(
        stats["mu_max"], true_mu_max, rtol=2e-2
    ), f"mu_max {stats['mu_max']:.3f} does not match true value {true_mu_max:.3f}"


def test_compute_analytical_mu_max_mech_baranyi():
    """Analytic mu_max on the true parameters matches the fine-grid derivative."""
    mu = 0.5
    K = 1.0
    N0 = 0.05
    h0 = 2.0
    params = {"mu": mu, "K": K, "N0": N0, "h0": h0}

    # Ground truth: numerical derivative of the true curve on a very fine grid
    t_fine = np.linspace(0, 24, 20001)
    N_fine = evaluate_parametric_model(t_fine, "mech_baranyi", params)
    true_mu_max = np.max(np.gradient(np.log(N_fine), t_fine))

    mu_max = gc.inference.compute_analytical_mu_max("mech_baranyi", params, 24.0)
    assert np.isclose(
        mu_max, true_mu_max, rtol=1e-2
    ), f"mu_max {mu_max:.3f} does not match true value {true_mu_max:.3f}"
