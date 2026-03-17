import numpy as np
import pytest

from growthcurves.preprocessing import (
    blank_subtraction,
    detect_outliers,
    detect_outliers_ecod,
    detect_outliers_hampel,
    detect_outliers_iqr,
    out_of_iqr_window,
    path_correct,
)


# =============================================================================
# Helpers
# =============================================================================

RNG = np.random.default_rng(42)


def _smooth_sigmoid(n=100):
    """Synthetic smooth growth curve with no outliers."""
    t = np.linspace(0, 50, n)
    N = 0.05 + 2.0 / (1 + np.exp(-0.3 * (t - 25)))
    return t, N


def _sigmoid_with_spike(n=100, spike_idx=50, spike_magnitude=5.0):
    """Smooth growth curve with a single large spike at spike_idx."""
    t, N = _smooth_sigmoid(n)
    N_spike = N.copy()
    N_spike[spike_idx] += spike_magnitude
    return t, N_spike, spike_idx


# =============================================================================
# blank_subtraction
# =============================================================================


def test_blank_subtraction_basic():
    N = np.array([0.5, 0.6, 0.7, 0.8])
    blank = np.array([0.05, 0.05, 0.05, 0.05])
    result = blank_subtraction(N, blank)
    np.testing.assert_allclose(result, [0.45, 0.55, 0.65, 0.75])


def test_blank_subtraction_zero_blank():
    N = np.array([0.1, 0.2, 0.3])
    result = blank_subtraction(N, [0.0, 0.0, 0.0])
    np.testing.assert_allclose(result, N)


# =============================================================================
# path_correct
# =============================================================================


def test_path_correct_half_cm():
    N = np.array([0.25, 0.30, 0.35])
    result = path_correct(N, path_length_cm=0.5)
    np.testing.assert_allclose(result, [0.5, 0.6, 0.7])


def test_path_correct_two_cm():
    N = np.array([1.0, 1.2, 1.4])
    result = path_correct(N, path_length_cm=2.0)
    np.testing.assert_allclose(result, [0.5, 0.6, 0.7])


def test_path_correct_one_cm_is_identity():
    N = np.array([0.5, 1.0, 1.5])
    result = path_correct(N, path_length_cm=1.0)
    np.testing.assert_allclose(result, N)


def test_path_correct_zero_raises():
    with pytest.raises(ValueError):
        path_correct(np.array([0.5]), path_length_cm=0.0)


def test_path_correct_negative_raises():
    with pytest.raises(ValueError):
        path_correct(np.array([0.5]), path_length_cm=-1.0)


# =============================================================================
# out_of_iqr_window
# =============================================================================


def test_out_of_iqr_window_detects_center_outlier():
    # Outlier at center of window
    values = np.array([1.0, 1.0, 100.0, 1.0, 1.0])
    assert out_of_iqr_window(values, position="center") == True


def test_out_of_iqr_window_no_outlier():
    values = np.array([1.0, 1.1, 1.0, 1.1, 1.0])
    assert out_of_iqr_window(values, position="center") == False


def test_out_of_iqr_window_even_length_center_raises():
    values = np.array([1.0, 2.0, 3.0, 4.0])
    with pytest.raises(ValueError):
        out_of_iqr_window(values, position="center")


def test_out_of_iqr_window_invalid_position_raises():
    values = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    with pytest.raises(ValueError):
        out_of_iqr_window(values, position="middle")


def test_out_of_iqr_window_first_position():
    # Outlier at first element
    values = np.array([100.0, 1.0, 1.0, 1.0, 1.0])
    assert out_of_iqr_window(values, position="first") == True


def test_out_of_iqr_window_last_position():
    values = np.array([1.0, 1.0, 1.0, 1.0, 100.0])
    assert out_of_iqr_window(values, position="last") == True


def test_out_of_iqr_window_nan_center_returns_false():
    values = np.array([1.0, 1.0, np.nan, 1.0, 1.0])
    assert out_of_iqr_window(values, position="center") is False


# =============================================================================
# detect_outliers_iqr
# =============================================================================


def test_detect_outliers_iqr_flags_spike():
    _, N_spike, spike_idx = _sigmoid_with_spike(spike_idx=50, spike_magnitude=5.0)
    mask = detect_outliers_iqr(N_spike, window_size=11)
    assert mask[spike_idx], "Spike should be flagged"


def test_detect_outliers_iqr_no_false_positives_on_smooth():
    _, N = _smooth_sigmoid()
    mask = detect_outliers_iqr(N, window_size=11)
    assert mask.sum() == 0, "No outliers expected in smooth curve"


def test_detect_outliers_iqr_output_shape():
    _, N = _smooth_sigmoid(n=80)
    mask = detect_outliers_iqr(N, window_size=11)
    assert mask.shape == N.shape


def test_detect_outliers_iqr_output_dtype():
    _, N = _smooth_sigmoid()
    mask = detect_outliers_iqr(N, window_size=11)
    assert mask.dtype == bool


def test_detect_outliers_iqr_higher_factor_flags_fewer():
    _, N_spike, _ = _sigmoid_with_spike(spike_magnitude=2.0)
    mask_strict = detect_outliers_iqr(N_spike, window_size=11, factor=3.0)
    mask_loose = detect_outliers_iqr(N_spike, window_size=11, factor=0.5)
    assert mask_strict.sum() <= mask_loose.sum()


# =============================================================================
# detect_outliers_hampel
# =============================================================================


def test_detect_outliers_hampel_flags_spike():
    _, N_spike, spike_idx = _sigmoid_with_spike(spike_idx=50, spike_magnitude=5.0)
    mask = detect_outliers_hampel(N_spike, window=15, factor=3.0)
    assert mask[spike_idx], "Spike should be flagged by Hampel"


def test_detect_outliers_hampel_no_false_positives_on_smooth():
    _, N = _smooth_sigmoid()
    mask = detect_outliers_hampel(N, window=15, factor=3.0)
    assert mask.sum() == 0, "No outliers expected in smooth curve"


def test_detect_outliers_hampel_output_shape():
    _, N = _smooth_sigmoid(n=80)
    mask = detect_outliers_hampel(N)
    assert mask.shape == N.shape


# =============================================================================
# detect_outliers_ecod
# =============================================================================


def test_detect_outliers_ecod_flags_spike():
    _, N_spike, spike_idx = _sigmoid_with_spike(spike_idx=50, spike_magnitude=5.0)
    mask = detect_outliers_ecod(N_spike)
    assert mask[spike_idx], "Spike should be flagged by ECOD"


def test_detect_outliers_ecod_output_shape():
    _, N = _smooth_sigmoid(n=80)
    mask = detect_outliers_ecod(N)
    assert mask.shape == N.shape


def test_detect_outliers_ecod_no_false_positives_on_smooth():
    _, N = _smooth_sigmoid()
    mask = detect_outliers_ecod(
        N, factor=5.0
    )  # need to increase factor to avoid false positives on smooth curve
    assert mask.sum() == 0, "No outliers expected in smooth curve"


# =============================================================================
# detect_outliers dispatcher
# =============================================================================


def test_detect_outliers_dispatches_iqr():
    _, N_spike, spike_idx = _sigmoid_with_spike()
    mask = detect_outliers(N_spike, method="iqr", window_size=11)
    assert mask[spike_idx]


def test_detect_outliers_dispatches_hampel():
    _, N_spike, spike_idx = _sigmoid_with_spike()
    mask = detect_outliers(N_spike, method="hampel")
    assert mask[spike_idx]


def test_detect_outliers_dispatches_ecod():
    _, N_spike, spike_idx = _sigmoid_with_spike()
    mask = detect_outliers(N_spike, method="ecod")
    assert mask[spike_idx]


def test_detect_outliers_unknown_method_raises():
    _, N = _smooth_sigmoid()
    with pytest.raises(ValueError):
        detect_outliers(N, method="unknown_method")
