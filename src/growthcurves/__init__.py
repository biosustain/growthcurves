# The __init__.py file is loaded when the package is loaded.
# It is used to indicate that the directory in which it resides is a Python package
import warnings
from importlib import metadata

import numpy as np

__version__ = metadata.version("growthcurves")

from . import inference, models, non_parametric, parametric, plot, preprocessing
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
]


def fit_model(
    t: np.ndarray,
    N: np.ndarray,
    model_name: str,
    lag_frac: float = 0.15,
    exp_frac: float = 0.15,
    **fit_kwargs,
) -> tuple[dict, dict]:

    if model_name in models.MODEL_REGISTRY["non_parametric"]:
        fit_res = non_parametric.fit_non_parametric(
            t, N, method=model_name, **fit_kwargs
        )
    else:
        fit_res = parametric.fit_parametric(t, N, method=model_name, **fit_kwargs)
    # return None if fit fails, along with bad fit stats
    if fit_res is None:
        warnings.warn(
            f"Model fitting failed for model {model_name}. Returning None.",
            stacklevel=2,
        )
        return None, inference.bad_fit_stats()
    # Extract only statistics-related keyword arguments to avoid mixing
    # fitting parameters with stats extraction parameters.
    stats_kwargs = {}
    if "phase_boundary_method" in fit_kwargs:
        stats_kwargs["phase_boundary_method"] = fit_kwargs["phase_boundary_method"]
    stats_res = inference.extract_stats(
        fit_res, t=t, N=N, lag_frac=lag_frac, exp_frac=exp_frac, **stats_kwargs
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

    Returns
    -------
    tuple[dict, dict]
        Return tuple of two dictionaries: (fit_res, stats_res)
        - fit_res: Dictionary containing fitted model parameters.
        - stats_res: Dictionary containing goodness-of-fit statistics and growth metrics

    """
