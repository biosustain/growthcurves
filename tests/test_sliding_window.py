import numpy as np
import pytest

from growthcurves.inference import extract_stats
from growthcurves.non_parametric import fit_non_parametric, fit_sliding_window

MU = 0.6


def _logistic(n, sd_mult=0.0, sd_add=0.0, seed=0):
    """Zwietering logistic in ln N (max slope = MU), lag 5 h, over 24 h."""
    rng = np.random.default_rng(seed)
    t = np.linspace(0, 24, n)
    N = 0.02 * np.exp(4 / (1 + np.exp(MU * (5 - t) + 2)))
    return t, N * np.exp(
        sd_mult * rng.standard_normal(n)
    ) + sd_add * rng.standard_normal(n)


def _median_error(n, seeds=20, **kw):
    slopes = [
        fit_sliding_window(*_logistic(n, seed=s, **kw))["params"]["slope"]
        for s in range(seeds)
    ]
    return np.median(np.abs(np.array(slopes) / MU - 1))


# =============================================================================
# mu_max recovery
# =============================================================================


def test_recovers_mu_max_clean():
    r = fit_sliding_window(*_logistic(200))
    assert np.isclose(r["params"]["slope"], MU, rtol=0.01)


@pytest.mark.parametrize(
    "n, kw, tol",
    [
        (200, dict(sd_mult=0.03), 0.05),  # dense, noisy
        (25, dict(sd_mult=0.01), 0.12),  # sparse: window spans curvature
        (300, dict(sd_mult=0.01, sd_add=0.005), 0.05),  # additive noise floor
    ],
)
def test_recovers_mu_max_noisy(n, kw, tol):
    assert _median_error(n, **kw) < tol


def test_window_longer_on_noisy_data():
    span = lambda r: r["params"]["fit_t_max"] - r["params"]["fit_t_min"]  # noqa: E731
    clean = fit_sliding_window(*_logistic(200, seed=1))
    noisy = fit_sliding_window(*_logistic(200, sd_mult=0.03, seed=1))
    assert span(noisy) > 2 * span(clean)


def test_window_bounds_and_time_at_umax():
    t, N = _logistic(200)
    p = fit_sliding_window(t, N)["params"]
    assert t.min() <= p["fit_t_min"] < p["time_at_umax"] < p["fit_t_max"] <= t.max()


# =============================================================================
# Robustness
# =============================================================================


def test_spike_does_not_move_estimate():
    t, N = _logistic(200, sd_mult=0.01, seed=2)
    base = fit_sliding_window(t, N)["params"]["slope"]
    N[60] *= 3
    assert np.isclose(fit_sliding_window(t, N)["params"]["slope"], base, rtol=0.02)


def test_rounded_od():
    t, N = _logistic(200, sd_mult=0.01, seed=3)
    r = fit_sliding_window(t, np.round(N, 3))
    assert np.isclose(r["params"]["slope"], MU, rtol=0.05)


# =============================================================================
# No growth and invalid input
# =============================================================================


@pytest.mark.parametrize("seed", range(5))
def test_flat_noise_returns_none(seed):
    rng = np.random.default_rng(seed)
    N = 0.2 * np.exp(0.03 * rng.standard_normal(100))
    assert fit_sliding_window(np.linspace(0, 24, 100), N) is None


def test_declining_returns_none():
    t = np.linspace(0, 24, 100)
    assert fit_sliding_window(t, 0.5 * np.exp(-0.05 * t)) is None


def test_constant_returns_none():
    assert fit_sliding_window(np.linspace(0, 24, 50), np.full(50, 0.1)) is None


def test_too_few_points_returns_none():
    t, N = _logistic(8)
    assert fit_sliding_window(t, N) is None


@pytest.mark.parametrize("window_points", [2, 4.5])
def test_invalid_window_points_raises(window_points):
    with pytest.raises(ValueError):
        fit_sliding_window(*_logistic(50), window_points=window_points)


# =============================================================================
# fit_non_parametric and extract_stats
# =============================================================================


def test_fit_non_parametric_matches_fit_sliding_window():
    t, N = _logistic(100, sd_mult=0.01)
    assert fit_non_parametric(t, N, method="sliding_window") == fit_sliding_window(t, N)


def test_fit_non_parametric_unknown_method_raises():
    with pytest.raises(ValueError):
        fit_non_parametric(*_logistic(50), method="not_a_method")


def test_extract_stats_uses_fitted_window():
    t, N = _logistic(200, sd_mult=0.01)
    fit = fit_sliding_window(t, N)
    stats = extract_stats(fit, t, N)
    assert stats["mu_max"] == fit["params"]["slope"]
    assert stats["fit_t_min"] == fit["params"]["fit_t_min"]
    assert stats["fit_t_max"] == fit["params"]["fit_t_max"]
    assert np.isfinite(stats["model_rmse"])
