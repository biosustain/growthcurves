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


def _ar1(n, rho, rng):
    """Unit-variance AR(1) noise."""
    z, e = rng.standard_normal(n), np.zeros(n)
    e[0] = z[0]
    for i in range(1, n):
        e[i] = rho * e[i - 1] + np.sqrt(1 - rho**2) * z[i]
    return e


def _lagged_correlated(seed, n=500, rho=0.7, sd=0.002):
    """Logistic in ln N (max slope MU), 20 h lag, rounded additive AR(1) noise."""
    rng = np.random.default_rng(seed)
    t = np.linspace(0, 48, n)
    N = 0.005 * np.exp(5 / (1 + np.exp(4 * MU / 5 * (20 - t) + 2)))
    return t, np.round(N + sd * _ar1(n, rho, rng), 4)


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
    # The minimum rise forces a wider window, biasing the slope ~1 % low.
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


def test_correlated_noise_baseline_not_picked():
    # Correlated baseline noise can otherwise win over true growth.
    errors = [
        fit_sliding_window(*_lagged_correlated(seed))["params"]["slope"] / MU - 1
        for seed in range(20)
    ]
    assert np.max(np.abs(errors)) < 0.25
    assert np.median(np.abs(errors)) < 0.05


def test_window_ends_at_or_before_peak():
    t = np.linspace(0, 48, 300)
    N = 0.01 * np.exp(4 / (1 + np.exp(4 * MU / 4 * (10 - t) + 2)))
    N = N * np.exp(-0.05 * np.clip(t - 30, 0, None))  # death phase after 30 h
    p = fit_sliding_window(t, N)["params"]
    assert p["fit_t_max"] <= t[np.argmax(N)] + 1e-9


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


@pytest.mark.parametrize("seed", range(5))
def test_flat_correlated_noise_returns_none(seed):
    t = np.linspace(0, 48, 400)
    N = np.round(0.05 + 0.002 * _ar1(400, 0.7, np.random.default_rng(seed)), 4)
    assert fit_sliding_window(t, N) is None


def test_small_rise_returns_none():
    # OD rises 1.3-fold: below min_fold
    t = np.linspace(0, 24, 100)
    assert fit_sliding_window(t, 0.1 * (1 + 0.3 / (1 + np.exp(-(t - 12))))) is None


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


@pytest.mark.parametrize(
    "kw", [dict(min_rise=-0.1), dict(min_rise=1), dict(min_fold=0.5)]
)
def test_invalid_window_rules_raise(kw):
    with pytest.raises(ValueError):
        fit_sliding_window(*_logistic(50), **kw)


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
