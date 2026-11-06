"""Mechanistic mu_max, its time and its OD are taken within the observed span."""

import numpy as np
import pytest

import growthcurves as gc
from growthcurves.models import evaluate_parametric_model

PARAMS = {
    "mech_logistic": {"mu": 0.5, "K": 1.0, "N0": 0.05},
    "mech_gompertz": {"mu": 0.5, "K": 1.0, "N0": 0.05},
    "mech_richards": {"mu": 0.5, "K": 1.0, "N0": 0.05, "beta": 0.5},
    # Small h0: the rate peaks before the first observation
    "mech_baranyi": {"mu": 0.5, "K": 1.0, "N0": 0.05, "h0": 0.1},
}


@pytest.mark.parametrize("model", PARAMS)
def test_mu_max_within_observed_span(model):
    params = PARAMS[model]
    t = np.linspace(5, 24, 50)  # first observation after t = 0
    N = evaluate_parametric_model(t, model, params)
    stats = gc.inference.extract_stats({"params": params, "model_type": model}, t, N)

    # Ground truth: numerical derivative over the observed span on a very fine grid
    t_fine = np.linspace(5, 24, 20001)
    mu_fine = np.gradient(
        np.log(evaluate_parametric_model(t_fine, model, params)), t_fine
    )
    i = np.argmax(mu_fine)
    assert np.isclose(stats["mu_max"], mu_fine[i], rtol=1e-3)
    assert np.isclose(stats["time_at_umax"], t_fine[i], atol=1e-2)
    assert np.isclose(
        stats["od_at_umax"],
        evaluate_parametric_model(t_fine[i], model, params),
        rtol=1e-3,
    )
