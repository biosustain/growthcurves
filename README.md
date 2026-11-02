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
spline_fit, spline_stats = gc.fit_model(t, N, "spline")

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

| Method         | Function                            | Key parameters  |
| -------------- | ----------------------------------- | --------------- |
| Spline         | `non_parametric.fit_non_parametric` | `smooth`        |
| Sliding window | `non_parametric.fit_non_parametric` | `window_points` |

The **spline method** fits a penalised spline to log-transformed OD data and calculates growth rate from the spline's derivative. By default (`smooth="auto"`) the smoothing is chosen per curve by REML; pass a positive number to fix it instead. It returns `None` when the curve shows no significant growth.

The **sliding window method** fits weighted linear regressions to log-transformed data in windows of every length (at least `window_points`, default 5) and reports the window with the highest lower confidence bound on its slope, so the window length adapts to each curve's noise and sampling. It returns `None` when no window shows significant growth.

### Spline fitting (non-parametric)

The spline method fits $s(t) \approx \ln N$ and takes $\mu(t) = \frac{d\,s(t)}{dt}$:

1. Each reading is weighted by a noise model (proportional noise plus an additive OD floor, estimated from the curve).
2. Long stationary phases are cropped.
3. A cubic P-spline (up to 25 equal segments, second-difference penalty) is fitted, with the smoothing chosen by REML. Two robust reweighting passes downweight outliers, and two IRLS passes treat the noise in OD rather than ln OD.
4. $\mu_{\max}$ is $\mu(t)$ where $\mu(t) - 2\,\mathrm{SE}$ is highest. If $\mu - 3.5\,\mathrm{SE}$ is not above zero there, the method returns `None` (no growth).

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
