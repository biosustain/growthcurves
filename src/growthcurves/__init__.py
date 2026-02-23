# The __init__.py file is loaded when the package is loaded.
# It is used to indicate that the directory in which it resides is a Python package
import warnings
from importlib import metadata

import numpy as np

__version__ = metadata.version("growthcurves")

from . import inference, models, non_parametric, parametric, plot, preprocessing
from .inference import compare_methods
from .models import (
    MODEL_REGISTRY,
    get_all_models,
    get_all_parametric_models,
    get_model_category,
)
from .preprocessing import blank_subtraction, path_correct

# The __all__ variable is a list of variables which are imported
# when a user does "from example import *"
__all__ = [
    "models",
    "inference",
    "parametric",
    "non_parametric",
    "plot",
    "preprocessing",
    "MODEL_REGISTRY",
    "get_all_models",
    "get_all_parametric_models",
    "get_model_category",
    "blank_subtraction",
    "path_correct",
    "compare_methods",
]


def fit_model(
    t: np.ndarray,
    N: np.ndarray,
    model_name: str,
    lag_threshold: float = 0.15,
    exp_threshold: float = 0.15,
    phase_boundary_method=None,
    **kwargs,
) -> tuple[dict, dict]:
    if model_name in models.MODEL_REGISTRY["non_parametric"]:
        fit_res = non_parametric.fit_non_parametric(t, N, method=model_name, **kwargs)
    else:
        fit_res = parametric.fit_parametric(t, N, method=model_name, **kwargs)
    # return None if fit fails, along with bad fit stats
    if fit_res is None:
        warnings.warn(
            f"Model fitting failed for model {model_name}. Returning None.",
            stacklevel=2,
        )
        return None, inference.bad_fit_stats()

    stats_res = inference.extract_stats(
        fit_res,
        t=t,
        N=N,
        lag_threshold=lag_threshold,
        exp_threshold=exp_threshold,
        phase_boundary_method=phase_boundary_method,
        **kwargs,
    )

    stats_res["model_name"] = model_name
    return fit_res, stats_res


# ! not so good for dynamic inspection tools...
fit_model.__doc__ = f"""Fit a growth model to the provided t and N.

    Parameters
    ----------
    t : np.ndarray
        Time points corresponding to N (in hours).
    N : np.ndarray
        Growth data points corresponding to t.
    model_name : str
        One of the models in {', '.join(get_all_models())}.
    lag_threshold : float, optional
        Fraction of μ_max to define end of lag phase (threshold method, default: 0.15).
    exp_threshold : float, optional
        Fraction of μ_max to define end of exponential phase (threshold method,
        default: 0.15).
    phase_boundary_method : str, optional
        Method to determine phase boundaries ("tangent", "threshold",
        or None for default for model class).
    **kwargs
        Additiona keyword arguments to be passed to fitting and inference functions.

    Returns
    -------
    tuple[dict, dict]
        Return tuple of two dictionaries: (fit_res, stats_res)
        - fit_res: Dictionary containing fitted model parameters.
        - stats_res: Dictionary containing goodness-of-fit statistics and growth metrics

    """
