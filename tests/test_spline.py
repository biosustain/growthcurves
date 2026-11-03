import pytest
import numpy as np

from growthcurves.models import log_to_linear, phenom_gompertz_model_ln, spline_from_params
from growthcurves.non_parametric import fit_non_parametric, fit_spline

# =============================================================================
# Helpers
# =============================================================================


def _gompertz_curve(mu_max=0.4, A=2.5, lam=5.0, N0=0.05, n_points=200):
    """Phenomenological Gompertz curve with known mu_max."""
    t = np.linspace(0, 88, n_points)
    ln_ratio = phenom_gompertz_model_ln(t, A=A, mu_max=mu_max, lam=lam)
    N = log_to_linear(ln_ratio, N0=N0)
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


def test_mu_max_close_to_known_value():
    # The spline derivative of ln(N) should approximate the true mu_max
    for mu_max in (0.2, 0.4, 0.6):
        t, N, _ = _gompertz_curve(mu_max=mu_max)
        result = fit_spline(t, N)
        assert np.isclose(result["params"]["mu_max"], mu_max, rtol=0.01)


def test_time_at_umax_within_range():
    t, N, _ = _gompertz_curve()
    result = fit_spline(t, N)
    assert t.min() <= result["params"]["time_at_umax"] <= t.max()


def test_spline_from_params_reproduces_fit():
    t, N, _ = _gompertz_curve()
    params = fit_spline(t, N)["params"]
    spline = spline_from_params(params)
    assert np.isclose(spline.derivative()(params["time_at_umax"]), params["mu_max"], rtol=1e-3)


def test_fixed_smooth():
    t, N, mu_max = _gompertz_curve(mu_max=0.4)
    result = fit_spline(t, N, smooth=0.1)
    assert np.isclose(result["params"]["mu_max"], mu_max, rtol=0.05)
    assert fit_spline(t, N, smooth=1e4)["params"]["mu_max"] < result["params"]["mu_max"]


def test_invalid_smooth_raises():
    t, N, _ = _gompertz_curve()
    for bad in (-1.0, 0, "fast"):
        with pytest.raises(ValueError):
            fit_spline(t, N, smooth=bad)


# =============================================================================
# fit_spline — noise, outliers, long runs
# =============================================================================


def test_noisy_curve_recovers_mu_max():
    t, N, mu_max = _gompertz_curve(mu_max=0.4)
    rng = np.random.default_rng(0)
    N = N * np.exp(0.03 * rng.standard_normal(len(N))) + 0.002 * rng.standard_normal(len(N))
    result = fit_spline(t, N)
    assert np.isclose(result["params"]["mu_max"], mu_max, rtol=0.1)


def test_spike_does_not_move_mu_max():
    t, N, mu_max = _gompertz_curve(mu_max=0.4)
    N = N.copy()
    N[len(N) // 3] *= 3
    result = fit_spline(t, N)
    assert np.isclose(result["params"]["mu_max"], mu_max, rtol=0.05)


def test_long_stationary_phase_is_cropped():
    # Growth over the first ~25 h, then days of plateau
    t = np.linspace(0, 200, 1000)
    N = log_to_linear(phenom_gompertz_model_ln(t, A=2.5, mu_max=0.6, lam=5.0), N0=0.05)
    result = fit_spline(t, N)
    assert result["params"]["fit_t_max"] < 50
    assert np.isclose(result["params"]["mu_max"], 0.6, rtol=0.02)


def test_rounded_data_recovers_mu_max():
    t, N, mu_max = _gompertz_curve(mu_max=0.4)
    result = fit_spline(t, np.round(N, 3))
    assert np.isclose(result["params"]["mu_max"], mu_max, rtol=0.05)


def test_flat_noise_returns_none():
    rng = np.random.default_rng(1)
    t = np.linspace(0, 48, 150)
    N = 0.05 * np.exp(0.02 * rng.standard_normal(150))
    assert fit_spline(t, N) is None


@pytest.mark.parametrize("seed", range(5))
def test_flat_correlated_noise_returns_none(seed):
    # AR(1) noise wanders smoothly, so a REML spline can see a significant slope in it
    rng = np.random.default_rng(seed)
    z = np.zeros(200)
    for i in range(1, 200):
        z[i] = 0.8 * z[i - 1] + 0.6 * rng.standard_normal()
    N = 0.05 + 0.004 * z
    assert fit_spline(np.linspace(0, 48, 200), N) is None


def test_small_rise_returns_none():
    # OD rises 1.3-fold: below min_fold
    t = np.linspace(0, 48, 200)
    N = log_to_linear(phenom_gompertz_model_ln(t, A=np.log(1.3), mu_max=0.1, lam=5.0), N0=0.05)
    assert fit_spline(t, N) is None
    assert fit_spline(t, N, min_fold=1.2) is not None


def test_invalid_min_fold_raises():
    t, N, _ = _gompertz_curve()
    with pytest.raises(ValueError):
        fit_spline(t, N, min_fold=0.5)


# =============================================================================
# fit_spline — edge cases returning None
# =============================================================================


def test_too_few_points_returns_none():
    t = np.linspace(0, 5, 4)  # fewer than validate_data's minimum
    N = 0.05 * np.exp(0.3 * t)
    result = fit_spline(t, N)
    assert result is None


def test_constant_curve_returns_none():
    t = np.linspace(0, 10, 50)
    N = np.ones(50) * 0.5
    assert fit_spline(t, N) is None


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
    assert t.min() < result["params"]["fit_t_max"] <= t.max()


def test_fit_non_parametric_spline_mu_max_close_to_known():
    mu_max = 0.4
    t, N, _ = _gompertz_curve(mu_max=mu_max)
    result = fit_non_parametric(t, N, method="spline")
    assert np.isclose(result["params"]["mu_max"], mu_max, rtol=0.15)
