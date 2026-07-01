# growthcurves

A Python package for fitting and analyzing microbial growth curves.

Supports logistic, Gompertz, Richards, and Baranyi parametric models with
automatic growth statistics extraction (specific growth rate, doubling time,
phase boundaries) and non-parametric methods (spline fitting and sliding window).

## Web apps

This package powers two browser-based apps for human-in-the-loop growth curve
analysis, hosted at <https://biosustain.github.io/growthcurves_app/>:

- **[MicroGrowth](https://microgrowth.streamlit.app/)** - analysis of
  microtiter plate reader experiments, with support for multi-condition layouts
  and interactive quality control.
- **[AutoGrowth](https://autogrowth.streamlit.app/)** - analysis of mini-bioreactor
  data (e.g. Pioreactor, Chi.Bio), built for continuous culture and real-time
  monitoring.

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
t = np.linspace(0, 24, 100)
N = 0.01 + 1.5 / (1 + np.exp(-0.5 * (t - 10)))  # synthetic logistic data

# Fit a model and extract growth statistics in one call.
# fit_model returns (fit_result, stats); it dispatches on the model name, so the
# same entry point works for parametric and non-parametric methods alike.
fit_result, stats = gc.fit_model(t, N, "mech_logistic")

print(f"Max OD:               {stats['max_od']:.3f}")
print(f"Specific growth rate: {stats['mu_max']:.4f} h⁻¹")
print(f"Doubling time:        {stats['doubling_time']:.2f} h")

# Or use a non-parametric spline fit
# smooth: "fast" (auto-default), "slow" (GCV), or a float (manual lambda)
spline_fit, spline_stats = gc.fit_model(t, N, "spline", smooth="fast")

print(f"\nSpline fit results:")
print(f"Specific growth rate: {spline_stats['mu_max']:.4f} h⁻¹")
print(f"Doubling time:        {spline_stats['doubling_time']:.2f} h")
```

giving output like:

```
Max OD:               1.554
Specific growth rate: 0.4610 h⁻¹
Doubling time:        1.50 h

Spline fit results:
Specific growth rate: 0.4247 h⁻¹
Doubling time:        1.63 h
```

## Available models

We use the formulations as stated in

> Ghenu A-H, Marrec L and Bank C (2024) Challenges and pitfalls of inferring microbial
> growth rates from lab cultures. Front. Ecol. Evol. 11:1313500.
> https://doi.org/10.3389/fevo.2023.1313500

### Parametric models

#### Mechanistic models (ODE-based)

| Model          | Function                     | Parameters      |
| -------------- | ---------------------------- | --------------- |
| Mech. Logistic | `models.mech_logistic_model` | mu, K, N0       |
| Mech. Gompertz | `models.mech_gompertz_model` | mu, K, N0       |
| Mech. Richards | `models.mech_richards_model` | mu, K, N0, beta |
| Mech. Baranyi  | `models.mech_baranyi_model`  | mu, K, N0, h0   |

Mechanistic models are defined as ordinary differential equations (ODEs) and fitted using numerical integration.

#### Phenomenological models (ln-space)

> No `N0` parameter is present in the phenomenological models. To apply a N0 constant use
> `log_to_linear` function to convert the log-space model output to linear space.

| Model              | Function                                   | Parameters                     |
| ------------------ | ------------------------------------------ | ------------------------------ |
| Phenom. Logistic   | `models.phenom_logistic_model`             | A, mu_max, lam                 |
| Phenom. Gompertz   | `models.phenom_gompertz_model_ln`          | A, mu_max, lam                 |
| Phenom. Gompertz\* | `models.phenom_gompertz_modified_model_ln` | A, mu_max, lam, alpha, t_shift |
| Phenom. Richards   | `models.phenom_richards_model`             | A, mu_max, lam, nu             |

Phenomenological models are fitted directly to ln(OD/OD0) data.

### Non-parametric methods

| Method         | Function                            | Key parameters          |
| -------------- | ----------------------------------- | ----------------------- |
| Spline         | `non_parametric.fit_non_parametric` | `smooth`, `use_weights` |
| Sliding window | `non_parametric.fit_non_parametric` | `window_points`         |

The **spline method** fits a smoothing spline to log-transformed OD data and calculates growth rate from the spline's derivative. Smoothing is controlled by `smooth`:

- `"fast"`: automatic default lambda rule (fast)
- `"slow"`: weighted GCV selection (slower)
- `float`: manual lambda value

The **sliding window method** estimates growth rate by fitting a linear regression to log-transformed data within a moving window, identifying the window with maximum slope.

### Spline fitting (non-parametric)

The spline method provides a model-free approach to growth curve analysis by fitting a smoothing spline to log-transformed OD data:

1. Transform OD data: $y_{\text{log}} = \ln(N)$
2. Fit a cubic smoothing spline $s(t)$ to $(t, y_{\text{log}})$ using `scipy.interpolate.make_smoothing_spline`
3. Calculate specific growth rate: $\mu(t) = \frac{d\,s(t)}{dt}$
4. Find maximum growth rate: $\mu_{\max} = \max_{t} \mu(t)$

| Parameter     | Meaning                                          |
| ------------- | ------------------------------------------------ |
| `smooth`      | `"fast"`, `"slow"`, or manual float lambda value |
| `use_weights` | Apply OD-dependent weighting (default: `True`)   |

When `smooth` is a float, higher values produce smoother curves and lower values follow the data more tightly.

### Derived growth statistics

| Statistic            | Formula                            |
| -------------------- | ---------------------------------- |
| Specific growth rate | $\mu = \dfrac{1}{N}\dfrac{dN}{dt}$ |
| Doubling time        | $t_d = \dfrac{\ln 2}{\mu_{\max}}$  |

## Key features

- **Parametric fitting** - fit logistic, Gompertz, Richards, or Baranyi models with automatic parameter estimation
- **Non-parametric methods** - model-free growth rate estimation using:
  - **Spline fitting** - smoothing splines on log-transformed data with derivative-based growth rate calculation
  - **Sliding window** - moving window linear fits to log-transformed data
- **Growth statistics** - automatic extraction of max OD, specific growth rate (µ_max), doubling time, and exponential-phase boundaries
- **Derivative analysis** - first and second derivatives with Savitzky-Golay smoothing
- **No-growth detection** - automatic identification of non-growing samples
- **Model comparison** - RMSE fit-quality metric for comparing fits

## Documentation and tutorial

An interactive tutorial notebook is available at
[docs/tutorial/analysis.ipynb](docs/tutorial/analysis.ipynb). It covers model
fitting, derivative analysis, parameter extraction, and cross-model comparison
using a realistic microbial growth dataset.

## Citation

If you use this package, please cite it as described in [CITATION.cff](CITATION.cff).

## License

GPL-3.0-or-later. See [LICENSE](LICENSE).
