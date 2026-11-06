import numpy as np
import pytest

from growthcurves.parametric import fit_parametric


def _curve(fold):
    t = np.linspace(0, 20, 80)
    N = 0.05 + (0.05 * (fold - 1)) / (1 + np.exp(-(t - 10)))
    return t, N


@pytest.mark.parametrize("method", ["mech_logistic", "phenom_logistic"])
def test_min_fold_gates_growth(method):
    t, N = _curve(1.3)
    assert fit_parametric(t, N, method=method) is None
    assert fit_parametric(t, N, method=method, min_fold=1.2) is not None


def test_invalid_min_fold_raises():
    t, N = _curve(10)
    with pytest.raises(ValueError):
        fit_parametric(t, N, method="mech_logistic", min_fold=0.5)
