"""Data preprocessing utilities for growth curve analysis.

This module provides functions for common preprocessing steps such as blank
subtraction and path length correction.
"""

import numpy as np


def blank_subtraction(data: np.ndarray, blank: np.ndarray) -> np.ndarray:
    """
    Subtract blank values from data series.

    Performs element-wise subtraction of blank measurements from data measurements.
    This is commonly used for baseline correction in optical density measurements.

    Parameters
    ----------
    data : numpy.ndarray
        Data series to be corrected (e.g., OD measurements)
    blank : numpy.ndarray
        Blank/background measurements to subtract. Must be the same length as data,
        or a scalar value to subtract from all data points.

    Returns
    -------
    numpy.ndarray
        Blank-subtracted data

    Examples
    --------
    >>> import numpy as np
    >>> data = np.array([0.5, 0.6, 0.7, 0.8])
    >>> blank = np.array([0.05, 0.05, 0.05, 0.05])
    >>> corrected = blank_subtraction(data, blank)
    >>> corrected
    array([0.45, 0.55, 0.65, 0.75])
    """

    # Handle numpy arrays
    data_array = np.asarray(data, dtype=float)
    blank_array = np.asarray(blank, dtype=float)

    return data_array - blank_array


def path_correct(data: np.ndarray, path_length_cm: float) -> np.ndarray:
    """
    Correct optical density measurements to a standard 1 cm path length.

    Normalizes OD measurements taken at a specific path length to what they would
    be at a 1 cm path length using Beer-Lambert law (OD is proportional to path length).

    Parameters
    ----------
    data : numpy.ndarray
        Optical density measurements to correct
    path_length_cm : float
        Actual path length of the measurement in centimeters (must be > 0)

    Returns
    -------
    numpy.ndarray
        Path-corrected data normalized to 1 cm path length

    Raises
    ------
    ValueError
        If path_length_cm is not positive

    Examples
    --------
    >>> import numpy as np
    >>> # Measurement taken with 0.5 cm path length
    >>> data = np.array([0.25, 0.30, 0.35])
    >>> corrected = path_correct(data, path_length_cm=0.5)
    >>> corrected  # OD values as if measured with 1 cm path
    array([0.5, 0.6, 0.7])

    >>> # Measurement taken with 2 cm path length
    >>> data = np.array([1.0, 1.2, 1.4])
    >>> corrected = path_correct(data, path_length_cm=2.0)
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
    data_array = np.asarray(data, dtype=float)

    return data_array / float(path_length_cm)
