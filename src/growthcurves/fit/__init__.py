"""Entry point of growthcurves package"""

import inspect
import warnings

import numpy as np

from growthcurves import inference, models
from growthcurves.models import get_all_models
from growthcurves.non_parametric import fit_non_parametric
from growthcurves.parametric import fit_parametric

__all__ = [
    "fit_model",
    "fit_non_parametric",
    "fit_parametric",
]


def _named_params(func) -> set[str]:
    """Names of the parameters `func` explicitly declares (no *args/**kwargs)."""
    return {
        name
        for name, p in inspect.signature(func).parameters.items()
        if p.kind in (p.POSITIONAL_OR_KEYWORD, p.KEYWORD_ONLY)
    }


def _accepted_kwargs(func, kwargs: dict) -> dict:
    """Keep only the keyword arguments that `func` explicitly names."""
    names = _named_params(func)
    return {k: v for k, v in kwargs.items() if k in names}


def _check_known_kwargs(kwargs: dict) -> None:
    """Raise if a keyword is not used by any fit function or by `extract_stats`."""
    known = (
        _named_params(fit_parametric)
        | _named_params(fit_non_parametric)
        | _named_params(inference.extract_stats)
        | {"lag_frac", "exp_frac"}  # aliases read from extract_stats' **kwargs
    )
    unknown = sorted(set(kwargs) - known)
    if unknown:
        raise TypeError(
            f"fit_model() got unknown keyword argument(s) {unknown}; "
            f"no model accepts them. Known options: {sorted(known)}"
        )


def fit_model(
    t: np.ndarray,
    N: np.ndarray,
    model_name: str,
    lag_threshold: float = 0.15,
    exp_threshold: float = 0.15,
    phase_boundary_method=None,
    **kwargs,
) -> tuple[dict, dict]:
    if kwargs:
        _check_known_kwargs(kwargs)
    if model_name in models.MODEL_REGISTRY["non_parametric"]:
        fit_func = fit_non_parametric
    else:
        fit_func = fit_parametric
    # model-specific options (e.g. smooth, window_points) are only passed on to
    # the fit functions that accept them, so one set of kwargs works for all models
    fit_res = fit_func(
        t, N, method=model_name, **_accepted_kwargs(fit_func, kwargs)
    )
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
        Additional keyword arguments (e.g. ``smooth``, ``window_points``,
        ``min_fold``). Each fit function only receives the ones it accepts, so
        options meant for other model families are ignored.

    Returns
    -------
    tuple[dict, dict]
        Return tuple of two dictionaries: (fit_res, stats_res)
        - fit_res: Dictionary containing fitted model parameters.
        - stats_res: Dictionary containing goodness-of-fit statistics and growth metrics

    """
