"""Data preprocessing utilities for growth curve analysis.

This module provides functions for common preprocessing steps such as blank
subtraction and path length correction.
"""

from functools import partial
from pyod.models.ecod import ECOD

import numpy as np


def blank_subtraction(N: np.ndarray, blank: np.ndarray) -> np.ndarray:
    """
    Subtract blank values from time data series of growth measurements.

    Performs element-wise subtraction of blank measurements from measurements.
    This is commonly used for baseline correction in optical density measurements.

    Parameters
    ----------
    N : numpy.ndarray
        Data series to be corrected (e.g., OD measurements)
    blank : numpy.ndarray
        Blank/background measurements to subtract. Must be the same length as N,
        or a scalar value to subtract from all N points.

    Returns
    -------
    numpy.ndarray
        Blank-subtracted N

    Examples
    --------
    >>> import numpy as np
    >>> N = np.array([0.5, 0.6, 0.7, 0.8])
    >>> blank = np.array([0.05, 0.05, 0.05, 0.05])
    >>> corrected = blank_subtraction(N, blank)
    >>> corrected
    array([0.45, 0.55, 0.65, 0.75])
    """

    # Handle numpy arrays
    data_array = np.asarray(N, dtype=float)
    blank_array = np.asarray(blank, dtype=float)

    return data_array - blank_array


def path_correct(N: np.ndarray, path_length_cm: float) -> np.ndarray:
    """
    Correct optical density measurements to a standard 1 cm path length.

    Normalizes OD measurements taken at a specific path length to what they would
    be at a 1 cm path length using Beer-Lambert law (OD is proportional to path length).

    Parameters
    ----------
    N : numpy.ndarray
        Optical density measurements to correct
    path_length_cm : float
        Actual path length of the measurement in centimeters (must be > 0)

    Returns
    -------
    numpy.ndarray
        Path-corrected N normalized to 1 cm path length

    Raises
    ------
    ValueError
        If path_length_cm is not positive

    Examples
    --------
    >>> import numpy as np
    >>> # Measurement taken with 0.5 cm path length
    >>> N = np.array([0.25, 0.30, 0.35])
    >>> corrected = path_correct(N, path_length_cm=0.5)
    >>> corrected  # OD values as if measured with 1 cm path
    array([0.5, 0.6, 0.7])

    >>> # Measurement taken with 2 cm path length
    >>> N = np.array([1.0, 1.2, 1.4])
    >>> corrected = path_correct(N, path_length_cm=2.0)
    >>> corrected
    array([0.5, 0.6, 0.7])

    Notes
    -----
    The correction uses the relationship: OD_1cm = OD_measured / path_length

    This assumes the Beer-Lambert law holds (linear relationship between
    absorbance and path length), which is typically valid for OD < 1.0-1.5.
    """
    if path_length_cm <= 0:
        raise ValueError(f"Path length must be positive, got {path_length_cm} cm")

    # Handle numpy arrays
    data_array = np.asarray(N, dtype=float)

    return data_array / float(path_length_cm)


def out_of_iqr_window(
    values: np.ndarray, factor: float = 1.5, position: str = "center"
) -> bool:
    """Return True if the selected value is an outlier based on the IQR method.

    Parameters
    ----------
    values : numpy.ndarray
        Input window of values.
    factor : float, default=1.5
        IQR multiplier used to define outlier bounds.
    position : {"center", "first", "last"}, default="center"
        Which value in the window to test as the target point.

    Raises
    ------
    ValueError
        If `position` is invalid.
        If `position="center"` and the input array does not have an odd
        number of elements.
    """
    if position == "center":
        if len(values) % 2 == 0:
            raise ValueError(
                "Input array must have an odd number of elements when "
                "position='center'."
            )
        center = values[len(values) // 2]
    elif position == "first":
        center = values[0]
    elif position == "last":
        center = values[-1]
    else:
        raise ValueError("position must be one of: 'center', 'first', 'last'.")

    if np.isnan(center):
        return False

    q1 = np.nanquantile(values, 0.25)
    q3 = np.nanquantile(values, 0.75)
    iqr = q3 - q1

    lower_bound = q1 - factor * iqr
    upper_bound = q3 + factor * iqr

    return (center < lower_bound) or (center > upper_bound)


def out_of_iqr(N: np.array, window_size: int, factor: float = 1.5) -> np.array:
    """Return a boolean array indicating whether each value is an outlier
    based on the IQR method.

    The sliding window size gives for the middle values the central window points
    IQR status. For the first and last points in data series N, the first and last
    point in window is respectively used instead of the center to label a value
    as an outlier.
    """
    edge_idx = window_size // 2
    windows = np.lib.stride_tricks.sliding_window_view(N, window_size)
    # use partial to fix factor argument for out_of_iqr_window
    _out_of_iqr_window = partial(out_of_iqr_window, factor=factor)
    window_flags = np.apply_along_axis(_out_of_iqr_window, 1, windows)

    edge_window_size = window_size + window_size // 2 - 1
    start_windows = np.lib.stride_tricks.sliding_window_view(
        N[:edge_window_size], window_size
    )
    start_window_mask = np.apply_along_axis(
        _out_of_iqr_window,
        1,
        start_windows,
        position="first",  # passed to out_of_iqr_window
    )
    end_windows = np.lib.stride_tricks.sliding_window_view(
        N[-(edge_window_size):], window_size
    )
    end_window_mask = np.apply_along_axis(
        _out_of_iqr_window,
        1,
        end_windows,
        position="last",  # passed to out_of_iqr_window
    )
    # create mask of same length as array N
    mask = np.full(N.shape, False)

    mask[:edge_idx] = start_window_mask  # [:edge_idx]
    mask[edge_idx:-edge_idx] = window_flags
    mask[-edge_idx:] = end_window_mask

    return mask


def detect_outliers(N: np.ndarray, factor: float = 3.5) -> np.ndarray:
    """Return a boolean array indicating whether each value is an outlier
    using ECOD (Empirical Cumulative Distribution-based Outlier Detection).

    Builds a 3-feature matrix per point — absolute rolling-mean residual,
    raw OD value, and first difference — then fits ECOD and flags points
    whose MAD z-score of the decision score exceeds `factor`.

    Parameters
    ----------
    N : numpy.ndarray
        Input time series of OD values.
    factor : float, default=3.5
        MAD z-score threshold for flagging outliers. Higher values flag
        fewer, more extreme points.

    Returns
    -------
    numpy.ndarray
        Boolean mask of the same length as N where True indicates an outlier.
    """

    n = len(N)
    half = 15 // 2
    residual = np.zeros(n)
    for i in range(n):
        win = N[max(0, i - half) : min(n, i + half + 1)]
        residual[i] = N[i] - win.mean()
    d = np.diff(N, prepend=N[0])
    X = np.column_stack([np.abs(residual), N, d])

    clf = ECOD()
    clf.fit(X)
    scores = clf.decision_scores_

    med = np.median(scores)
    mad = np.median(np.abs(scores - med))
    if mad < 1e-12:
        return np.zeros(n, dtype=bool)
    mad_z = np.abs(scores - med) / (1.4826 * mad)
    return mad_z > factor


if __name__ == "__main__":
    # Example usage
    data = np.array([20, 1, 2, 3, 4, 5, 20, 6, 7, 8, 9, 10, 25])
    print(out_of_iqr_window(data))

    # rolling windows of size 5: shape -> (len(data)-4, 5)
    window_size = 5
    edge_idx = window_size // 2
    windows = np.lib.stride_tricks.sliding_window_view(data, window_size)

    # outlier flag for each 5-value window (checks center element of each window)
    # center points (indices 2..-3)
    window_flags = np.apply_along_axis(out_of_iqr_window, 1, windows)
    edge_window_size = window_size + window_size // 2 - 1
    start_windows = np.lib.stride_tricks.sliding_window_view(
        data[:edge_window_size], window_size
    )
    start_window_flags = np.apply_along_axis(
        out_of_iqr_window,
        1,
        start_windows,
        position="first",  # passed to out_of_iqr_window
    )
    end_windows = np.lib.stride_tricks.sliding_window_view(
        data[-(edge_window_size):], window_size
    )
    end_window_flags = np.apply_along_axis(
        out_of_iqr_window,
        1,
        end_windows,
        position="last",  # passed to out_of_iqr_window
    )
    print("center windows flags:", window_flags)
    print("start edge windows:", start_windows)
    print("start edge window flags:", start_window_flags)
    print("end edge windows:", end_windows)
    print("end edge window flags:", end_window_flags)
    # optional: align back to original array length (center indices only)
    flags = np.full(data.shape, False)

    flags[:edge_idx] = start_window_flags  # [:edge_idx]
    flags[edge_idx:-edge_idx] = window_flags
    flags[-edge_idx:] = end_window_flags

    print("windows:\n", windows)
    print("window_flags:", window_flags)
    print("aligned flags:", flags)
    mask = out_of_iqr(data, window_size=5)

    print("final mask:", mask)
    assert (mask == flags).all()
