import numpy as np
import pytest

from growthcurves.non_parametric import fit_non_parametric, fit_sliding_window

# =============================================================================
# Helpers
# =============================================================================


def _exponential_curve(mu=0.4, t_max=10.0, n_points=200):
    """Pure exponential growth: ln(N) is linear with slope = mu."""
    t = np.linspace(0, t_max, n_points)
    N = 0.05 * np.exp(mu * t)
    return t, N, mu


def _logistic_curve(mu=0.4, K=2.5, N0=0.05, t_max=40.0, n_points=200):
    """Logistic growth curve."""
    t = np.linspace(0, t_max, n_points)
    N = K / (1 + ((K - N0) / N0) * np.exp(-mu * t))
    return t, N, mu


def _decreasing_curve(n_points=100):
    t = np.linspace(0, 10, n_points)
    N = 2.5 * np.exp(-0.3 * t)
    return t, N


# =============================================================================
# fit_sliding_window — growth rate recovery
# =============================================================================


def test_slope_matches_exponential_growth_rate():
    # Pure exponential: d(ln N)/dt = mu exactly, so the best window slope = mu
    mu = 0.4
    t, N, _ = _exponential_curve(mu=mu)
    result = fit_sliding_window(t, N, window_points=15)
    assert np.isclose(result["params"]["slope"], mu, rtol=0.05)


def test_slope_positive_for_growth_curve():
    t, N, _ = _logistic_curve()
    result = fit_sliding_window(t, N)
    assert result["params"]["slope"] > 0


def test_time_at_umax_within_range():
    t, N, _ = _logistic_curve()
    result = fit_sliding_window(t, N)
    assert t.min() <= result["params"]["time_at_umax"] <= t.max()


def test_fit_window_bounds_within_time_range():
    t, N, _ = _logistic_curve()
    result = fit_sliding_window(t, N)
    assert result["params"]["fit_t_min"] >= t.min()
    assert result["params"]["fit_t_max"] <= t.max()


# =============================================================================
# fit_sliding_window — edge cases returning None
# =============================================================================


def test_too_few_points_returns_none():
    t = np.linspace(0, 5, 10)
    N = 0.05 * np.exp(0.3 * t)
    result = fit_sliding_window(t, N, window_points=15)
    assert result is None


def test_zero_time_span_returns_none():
    t = np.ones(20)  # all same time point
    N = np.ones(20) * 0.1
    result = fit_sliding_window(t, N)
    assert result is None


def test_decreasing_data_returns_none():
    t, N = _decreasing_curve()
    result = fit_sliding_window(t, N)
    assert result is None


# =============================================================================
# fit_sliding_window — step parameters
# =============================================================================


def test_n_fits_slope_close_to_step_one():
    # Using n_fits (larger step) should still recover a slope close to step=1
    mu = 0.4
    t, N, _ = _exponential_curve(mu=mu)
    result_dense = fit_sliding_window(t, N, step=1)
    result_sparse = fit_sliding_window(t, N, n_fits=10)
    assert np.isclose(
        result_dense["params"]["slope"], result_sparse["params"]["slope"], rtol=0.1
    )


# =============================================================================
# fit_non_parametric dispatcher — sliding_window
# =============================================================================


def test_fit_non_parametric_returns_sliding_window_model_type():
    t, N, _ = _logistic_curve()
    result = fit_non_parametric(t, N, method="sliding_window")
    assert result["model_type"] == "sliding_window"


def test_fit_non_parametric_includes_window_points_in_params():
    t, N, _ = _logistic_curve()
    window_points = 20
    result = fit_non_parametric(
        t, N, method="sliding_window", window_points=window_points
    )
    assert result["params"]["window_points"] == window_points


def test_fit_non_parametric_unknown_method_raises():
    t, N, _ = _logistic_curve()
    with pytest.raises(ValueError):
        fit_non_parametric(t, N, method="not_a_method")
