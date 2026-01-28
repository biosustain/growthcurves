# growthcurves

A Python package for fitting and analyzing microbial growth curves.

Supports logistic, Gompertz, and Richards parametric models with automatic
growth statistics extraction (specific growth rate, doubling time, phase
boundaries) and a non-parametric sliding-window method.

## Installation

```bash
pip install growthcurves
```

For development:

```bash
pip install -e ".[dev]"
```

## Quick start

```python
import growthcurves as gc
import numpy as np

# Example time series (hours) and OD measurements
time = np.linspace(0, 24, 100)
od = 0.01 + 1.5 / (1 + np.exp(-0.5 * (time - 10)))  # synthetic logistic data

# Fit a model and extract growth statistics
stats = gc.fitting_functions.fit_growth_model(time, od,model_type="logistic")

print(f"Max OD:              {stats['max_od']:.3f}")
print(f"Specific growth rate: {stats['specific_growth_rate']:.4f} h⁻¹")
print(f"Doubling time:        {stats['doubling_time']:.2f} h")
```

## Available models

| Model    | Function                | Parameters         |
| -------- | ----------------------- | ------------------ |
| Logistic | `models.logistic_model` | K, y0, r, t0       |
| Gompertz | `models.gompertz_model` | K, y0, mu_max, lam |
| Richards | `models.richards_model` | K, y0, r, t0, nu   |

The Richards model generalizes both logistic (nu = 1) and Gompertz (nu → 0)
growth curves via its shape parameter `nu`.

### Logistic

$$
N(t) = y_0 + \frac{K - y_0}{1 + \exp\!\bigl[-r\,(t - t_0)\bigr]}
$$

| Parameter | Meaning                                         |
| --------- | ----------------------------------------------- |
| $K$       | Carrying capacity (maximum OD)                  |
| $y_0$     | Baseline OD at $t=0$                            |
| $r$       | Growth rate constant (h⁻¹); equals $\mu_{\max}$ |
| $t_0$     | Inflection time                                 |

### Gompertz (modified)

$$
N(t) = y_0 + (K - y_0)\,\exp\!\left[-\exp\!\left(\frac{\mu_{\max}\,e}{K - y_0}\,(\lambda - t) + 1\right)\right]
$$

| Parameter    | Meaning                            |
| ------------ | ---------------------------------- |
| $K$          | Carrying capacity (maximum OD)     |
| $y_0$        | Baseline OD                        |
| $\mu_{\max}$ | Maximum specific growth rate (h⁻¹) |
| $\lambda$    | Lag time (h)                       |

### Richards (generalized logistic)

$$
N(t) = y_0 + (K - y_0)\,\bigl[1 + \nu\,\exp\!\bigl(-r\,(t - t_0)\bigr)\bigr]^{-1/\nu}
$$

| Parameter | Meaning                                                                         |
| --------- | ------------------------------------------------------------------------------- |
| $K$       | Carrying capacity (maximum OD)                                                  |
| $y_0$     | Baseline OD                                                                     |
| $r$       | Growth rate constant (h⁻¹)                                                      |
| $t_0$     | Inflection time                                                                 |
| $\nu$     | Shape parameter ($\nu=1 \Rightarrow$ logistic; $\nu\to 0 \Rightarrow$ Gompertz) |

The maximum specific growth rate for the Richards model is $\mu_{\max} = r\,/\,(1+\nu)^{1/\nu}$.

### Derived growth statistics

| Statistic            | Formula                            |
| -------------------- | ---------------------------------- |
| Specific growth rate | $\mu = \dfrac{1}{N}\dfrac{dN}{dt}$ |
| Doubling time        | $t_d = \dfrac{\ln 2}{\mu_{\max}}$  |

## Key features

- **Parametric fitting** — fit logistic, Gompertz, or Richards models with automatic parameter estimation
- **Sliding-window method** — non-parametric growth rate estimation via sliding window fits to log-tranformed data
- **Growth statistics** — automatic extraction of max OD, specific growth rate (µ_max), doubling time, and exponential-phase boundaries
- **Derivative analysis** — first and second derivatives with Savitzky-Golay smoothing
- **No-growth detection** — automatic identification of non-growing samples
- **Model comparison** — RMSE fit-quality metric for comparing fits

## Documentation and tutorial

An interactive tutorial notebook is available at
[docs/tutorial/tutorial.ipynb](docs/tutorial/tutorial.ipynb). It covers model
fitting, derivative analysis, parameter extraction, and cross-model comparison
using a realistic microbial growth dataset.

## Citation

If you use this package, please cite it as described in [CITATION.cff](CITATION.cff).

## License

GPL-3.0-or-later. See [LICENSE](LICENSE).
