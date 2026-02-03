# growthcurves

A Python package for fitting and analyzing microbial growth curves.

Supports logistic, Gompertz, and Richards parametric models with automatic
growth statistics extraction (specific growth rate, doubling time, phase
boundaries) and non-parametric methods (spline fitting and sliding window).

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

# Fit a parametric model and extract growth statistics
fit_result = gc.parametric.fit_parametric(time, od, model="logistic")
stats = gc.utils.extract_stats(fit_result, time, od)

print(f"Max OD:               {stats['max_od']:.3f}")
print(f"Specific growth rate: {stats['specific_growth_rate']:.4f} h⁻¹")
print(f"Doubling time:        {stats['doubling_time']:.2f} h")

# Or use a non-parametric spline fit
spline_fit = gc.non_parametric.fit_non_parametric(time, od, umax_method="spline")
spline_stats = gc.utils.extract_stats(spline_fit, time, od)

print(f"\nSpline fit results:")
print(f"Specific growth rate: {spline_stats['specific_growth_rate']:.4f} h⁻¹")
print(f"Doubling time:        {spline_stats['doubling_time']:.2f} h")
```

## Available models

### Parametric models

| Model    | Function                | Parameters         |
| -------- | ----------------------- | ------------------ |
| Logistic | `models.logistic_model` | K, y0, r, t0       |
| Gompertz | `models.gompertz_model` | K, y0, mu_max, lam |
| Richards | `models.richards_model` | K, y0, r, t0, nu   |

The Richards model generalizes both logistic (nu = 1) and Gompertz (nu → 0)
growth curves via its shape parameter `nu`.

### Non-parametric methods

| Method         | Function                            | Key parameters       |
| -------------- | ----------------------------------- | -------------------- |
| Spline         | `non_parametric.fit_non_parametric` | spline_s (smoothing) |
| Sliding window | `non_parametric.fit_non_parametric` | window_points        |

The **spline method** fits a smoothing spline to log-transformed OD data and calculates growth rate from the spline's derivative. This provides a flexible, model-free approach that adapts to the data shape. The smoothing parameter `spline_s` controls the balance between fit quality and smoothness (default: `0.01` for tight fit to data).

The **sliding window method** estimates growth rate by fitting a linear regression to log-transformed data within a moving window, identifying the window with maximum slope.

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

### Spline fitting (non-parametric)

The spline method provides a model-free approach to growth curve analysis by fitting a smoothing spline to log-transformed OD data:

1. Transform OD data: $y_{\text{log}} = \ln(N)$
2. Fit a cubic smoothing spline $s(t)$ to $(t, y_{\text{log}})$ using `scipy.interpolate.UnivariateSpline`
3. Calculate specific growth rate: $\mu(t) = \frac{d\,s(t)}{dt}$
4. Find maximum growth rate: $\mu_{\max} = \max_{t} \mu(t)$

| Parameter  | Meaning                           |
| ---------- | --------------------------------- |
| `spline_s` | Smoothing factor (default: 0.01)  |
| `k`        | Spline degree (default: 3, cubic) |

The smoothing parameter `spline_s` controls the tradeoff between fit quality and smoothness. Lower values produce tighter fits to data; higher values produce smoother curves. The default of 0.01 produces a tight fit that closely follows the data, which can be increased for noisier datasets.

### Derived growth statistics

| Statistic            | Formula                            |
| -------------------- | ---------------------------------- |
| Specific growth rate | $\mu = \dfrac{1}{N}\dfrac{dN}{dt}$ |
| Doubling time        | $t_d = \dfrac{\ln 2}{\mu_{\max}}$  |

## Key features

- **Parametric fitting** — fit logistic, Gompertz, or Richards models with automatic parameter estimation
- **Non-parametric methods** — model-free growth rate estimation using:
  - **Spline fitting** — smoothing splines on log-transformed data with derivative-based growth rate calculation
  - **Sliding window** — moving window linear fits to log-transformed data
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
