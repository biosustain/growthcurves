# The __init__.py file is loaded when the package is loaded.
# It is used to indicate that the directory in which it resides is a Python package
from importlib import metadata

__version__ = metadata.version("growthcurves")

from . import models, non_parametric, parametric, plot, utils
from .models import (
    MODEL_REGISTRY,
    get_all_models,
    get_all_parametric_models,
    get_model_category,
)

# The __all__ variable is a list of variables which are imported
# when a user does "from example import *"
__all__ = [
    "models",
    "utils",
    "parametric",
    "non_parametric",
    "plot",
    "MODEL_REGISTRY",
    "get_all_models",
    "get_all_parametric_models",
    "get_model_category",
]
