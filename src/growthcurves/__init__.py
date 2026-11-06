# The __init__.py file is loaded when the package is loaded.
# It is used to indicate that the directory in which it resides is a Python package

from importlib import metadata

__version__ = metadata.version("growthcurves")

from . import inference, models, non_parametric, parametric, plot, preprocessing
from .fit import fit_model
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
    "MODEL_REGISTRY",
    "blank_subtraction",
    "compare_methods",
    "fit_model",
    "get_all_models",
    "get_all_parametric_models",
    "get_model_category",
    "inference",
    "models",
    "non_parametric",
    "parametric",
    "path_correct",
    "plot",
    "preprocessing",
]
