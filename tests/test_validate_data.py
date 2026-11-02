"""Every method must clean its input the same way, through validate_data."""

import numpy as np
import pytest

from growthcurves import fit_model
from growthcurves.models import get_all_models

# =============================================================================
# Helpers
# =============================================================================


def _clean_curve(n_points=60):
    """Noisy logistic curve with a lag, sorted by time."""
    rng = np.random.default_rng(0)
    t = np.linspace(0, 24, n_points)
    K, N0, mu, lam = 1.5, 0.05, 0.6, 4.0
    N = K / (1 + ((K - N0) / N0) * np.exp(-mu * np.clip(t - lam, 0, None)))
    N = N * np.exp(rng.normal(0, 0.02, n_points))
    return t, N


def _messy_curve(t, N):
    """The same readings shuffled, plus rows validate_data must drop."""
    rng = np.random.default_rng(1)
    order = rng.permutation(len(t))
    t_messy = np.concatenate([t[order], [5.0, 6.0, np.inf, t[10]]])
    # NaN OD, zero OD, infinite time, and a later reading at a repeated time
    N_messy = np.concatenate([N[order], [np.nan, 0.0, 0.3, N[10] * 2]])
    return t_messy, N_messy


def _assert_same(a, b):
    assert a.keys() == b.keys()
    for key in a:
        if isinstance(a[key], (int, float, np.floating)) and a[key] is not None:
            np.testing.assert_allclose(a[key], b[key], rtol=1e-9, err_msg=key)
        else:
            assert a[key] == b[key], key


# =============================================================================
# Same cleaning for every method
# =============================================================================


@pytest.mark.parametrize("model_name", get_all_models())
def test_messy_input_matches_clean_input(model_name):
    t, N = _clean_curve()
    fit_clean, stats_clean = fit_model(t, N, model_name)
    fit_messy, stats_messy = fit_model(*_messy_curve(t, N), model_name)

    assert fit_clean is not None
    _assert_same(fit_clean["params"], fit_messy["params"])
    _assert_same(stats_clean, stats_messy)


@pytest.mark.parametrize("model_name", get_all_models())
def test_negative_times_raise(model_name):
    t, N = _clean_curve()
    with pytest.raises(ValueError, match="Negative times"):
        fit_model(t - 1.0, N, model_name)


@pytest.mark.parametrize("model_name", get_all_models())
def test_too_few_valid_points_fails(model_name):
    # 9 valid readings: below validate_data's minimum of 10 for every method
    t, N = _clean_curve(n_points=12)
    N[:3] = np.nan
    with pytest.warns(UserWarning, match="Model fitting failed"):
        fit_res, _ = fit_model(t, N, model_name)
    assert fit_res is None
