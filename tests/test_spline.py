import numpy as np
import pytest

from growthcurves.non_parametric import fit_spline, fit_non_parametric
from growthcurves.models import phenom_gompertz_model


# =============================================================================
# Helpers
# =============================================================================


def _gompertz_curve(mu_max=0.4, A=2.5, lam=5.0, N0=0.05, n_points=200):
    """Phenomenological Gompertz curve with known mu_max."""
    t = np.linspace(0, 88, n_points)
    N = phenom_gompertz_model(t, A=A, mu_max=mu_max, lam=lam, N0=N0)
    return t, N, mu_max


# =============================================================================
# fit_spline — output structure
# =============================================================================


def test_output_model_type():
    t, N, _ = _gompertz_curve()
    result = fit_spline(t, N)
    assert result["model_type"] == "spline"


def test_returns_dict():
    t, N, _ = _gompertz_curve()
    result = fit_spline(t, N)
    assert isinstance(result, dict)
    assert "params" in result


# =============================================================================
# fit_spline — mu_max recovery
# =============================================================================


def test_mu_max_positive():
    t, N, _ = _gompertz_curve()
    result = fit_spline(t, N)
    assert result["params"]["mu_max"] > 0


def test_mu_max_close_to_known_value_fast():
    # The spline derivative of ln(N) should approximate the true mu_max
    t, N, _ = _gompertz_curve(mu_max=0.4)
    result = fit_spline(t, N, smooth="fast")
    assert np.isclose(result["params"]["mu_max"], 0.4, rtol=0.01)

    t1, N1, _ = _gompertz_curve(mu_max=0.6)
    result1 = fit_spline(t1, N1, smooth="fast")
    assert np.isclose(result1["params"]["mu_max"], 0.6, rtol=0.01)

    t2, N2, _ = _gompertz_curve(mu_max=0.2)
    result2 = fit_spline(t2, N2, smooth="fast")
    assert np.isclose(result2["params"]["mu_max"], 0.2, rtol=0.01)


def test_mu_max_close_to_known_value_slow():

    # The spline derivative of ln(N) should approximate the true mu_max
    t, N, _ = _gompertz_curve(mu_max=0.4)
    result = fit_spline(t, N, smooth="slow")
    assert np.isclose(result["params"]["mu_max"], 0.4, rtol=0.01)

    t1, N1, _ = _gompertz_curve(mu_max=0.6)
    result1 = fit_spline(t1, N1)
    assert np.isclose(result1["params"]["mu_max"], 0.6, rtol=0.01)

    t2, N2, _ = _gompertz_curve(mu_max=0.2)
    result2 = fit_spline(t2, N2, smooth="slow")
    assert np.isclose(result2["params"]["mu_max"], 0.2, rtol=0.01)


def test_time_at_umax_within_range():
    t, N, _ = _gompertz_curve()
    result = fit_spline(t, N)
    assert t.min() <= result["params"]["time_at_umax"] <= t.max()


# =============================================================================
# fit_spline — smoothing modes
# =============================================================================


def test_smooth_fast_returns_result():
    t, N, _ = _gompertz_curve()
    result = fit_spline(t, N, smooth="fast")
    assert result is not None
    assert result["params"]["smooth"] == "fast"


def test_smooth_slow_returns_result():
    t, N, _ = _gompertz_curve()
    result = fit_spline(t, N, smooth="slow")
    assert result is not None
    assert result["params"]["smooth"] == "slow"


def test_smooth_manual_float_returns_result():
    t, N, _ = _gompertz_curve()
    lam = 0.01
    result = fit_spline(t, N, smooth=lam)
    assert result is not None
    assert result["params"]["smooth"] == lam


def test_smooth_invalid_string_returns_none():
    # fit_spline catches all exceptions and returns None
    t, N, _ = _gompertz_curve()
    result = fit_spline(t, N, smooth="invalid")
    assert result is None


def test_smooth_negative_float_returns_none():
    t, N, _ = _gompertz_curve()
    result = fit_spline(t, N, smooth=-1.0)
    assert result is None


# =============================================================================
# fit_spline — use_weights parameter
# =============================================================================


def test_use_weights_false_returns_result():
    t, N, _ = _gompertz_curve()
    result = fit_spline(t, N, use_weights=False)
    assert result is not None
    assert result["params"]["mu_max"] > 0


def test_use_weights_true_and_false_both_recover_mu_max():
    mu_max = 0.4
    t, N, _ = _gompertz_curve(mu_max=mu_max)
    r_weighted = fit_spline(t, N, use_weights=True)
    r_unweighted = fit_spline(t, N, use_weights=False)
    assert np.isclose(r_weighted["params"]["mu_max"], mu_max, rtol=0.15)
    assert np.isclose(r_unweighted["params"]["mu_max"], mu_max, rtol=0.15)


# =============================================================================
# fit_spline — edge cases returning None
# =============================================================================


def test_too_few_points_returns_none():
    t = np.linspace(0, 5, 4)  # fewer than 5
    N = 0.05 * np.exp(0.3 * t)
    result = fit_spline(t, N)
    assert result is None


def test_flat_curve_mu_max_negligible():
    # Constant OD — the spline fits fine but its derivative is numerical noise (~1e-15)
    t = np.linspace(0, 10, 50)
    N = np.ones(50) * 0.5
    result = fit_spline(t, N)
    # result may be None or may return near-zero mu_max due to floating-point noise
    if result is not None:
        assert result["params"]["mu_max"] < 1e-10


# =============================================================================
# fit_non_parametric dispatcher — spline
# =============================================================================


def test_fit_non_parametric_returns_spline_model_type():
    t, N, _ = _gompertz_curve()
    result = fit_non_parametric(t, N, method="spline")
    assert result["model_type"] == "spline"


def test_fit_non_parametric_spline_includes_fit_bounds():
    t, N, _ = _gompertz_curve()
    result = fit_non_parametric(t, N, method="spline")
    assert "fit_t_min" in result["params"]
    assert "fit_t_max" in result["params"]
    assert result["params"]["fit_t_min"] == t.min()
    assert result["params"]["fit_t_max"] == t.max()


def test_fit_non_parametric_spline_mu_max_close_to_known():
    mu_max = 0.4
    t, N, _ = _gompertz_curve(mu_max=mu_max)
    result = fit_non_parametric(t, N, method="spline")
    assert np.isclose(result["params"]["mu_max"], mu_max, rtol=0.15)
