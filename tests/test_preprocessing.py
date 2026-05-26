import numpy as np
import pandas as pd
import pytest

from growthcurves.preprocessing import detect_outliers_ecod


def _sample_series():
    """Return a simple OD-like growth series."""
    return [0.1, 0.12, 0.15, 0.2, 0.3, 0.5, 0.8, 1.2, 1.5, 1.6, 1.61, 1.62]


def test_ecod_pandas_series_default_index():
    """detect_outliers_ecod should accept a pandas Series with the default (0-based) index."""
    s = pd.Series(_sample_series())
    result = detect_outliers_ecod(s)
    assert isinstance(result, np.ndarray)
    assert result.dtype == bool
    assert len(result) == len(s)


def test_ecod_pandas_series_non_default_index():
    """detect_outliers_ecod should accept a pandas Series with a non-standard index."""
    values = _sample_series()
    # Use an index that starts at 5 so label-based access N[0] would raise KeyError
    s = pd.Series(values, index=range(5, 5 + len(values)))
    result = detect_outliers_ecod(s)
    assert isinstance(result, np.ndarray)
    assert result.dtype == bool
    assert len(result) == len(s)


def test_ecod_numpy_array():
    """detect_outliers_ecod should work with a plain numpy array."""
    arr = np.array(_sample_series())
    result = detect_outliers_ecod(arr)
    assert isinstance(result, np.ndarray)
    assert result.dtype == bool
    assert len(result) == len(arr)


def test_ecod_missing_values_raises():
    """detect_outliers_ecod should raise ValueError when the input contains NaN."""
    arr = np.array([0.1, 0.2, np.nan, 0.4, 0.5] * 3, dtype=float)
    with pytest.raises(ValueError, match="missing values"):
        detect_outliers_ecod(arr)


def test_ecod_missing_values_in_series_raises():
    """detect_outliers_ecod should raise ValueError when a pandas Series contains NaN."""
    s = pd.Series([0.1, 0.2, np.nan, 0.4, 0.5] * 3)
    with pytest.raises(ValueError, match="missing values"):
        detect_outliers_ecod(s)
