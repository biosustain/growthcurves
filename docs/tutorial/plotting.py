# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.19.2
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# %% [markdown]
# # Visualize fitted growth curves, derivatives, and growth statistics
# This tutorial demonstrates how to visualize fitted growth curves, derivatives,
# and growth statistics.
#
# The workflow includes:
# 1. Generating growth data and fitting models
# 2. Plotting mechanistic model fits
# 3. Plotting phenomenological model fits
# 4. Visualizing phase boundary methods
# 5. Plotting derivatives (μ and dOD/dt)
# 6. Comparing growth statistics across models
#
# For a notebook focused on the analysis workflow only (without plotting),
# see [`analysis.ipynb`](analysis.ipynb) (Fit growth models and extract
# growth statistics)

# %%
import numpy as np
import pandas as pd

import growthcurves as gc
from growthcurves.inference import compare_methods
from growthcurves.plot import plot_growth_stats_comparison

# %% [markdown]
# ## Generate synthetic data
#
# This cell generates synthetic growth data from a clean logistic function.

# %% tags=["hide-input"]
# Generate synthetic growth N from logistic function
np.random.seed(42)

# Parameters for synthetic growth curve
n_points = 440
measurement_interval_minutes = 12
t = np.array([(measurement_interval_minutes * n) / 60 for n in range(n_points)])


def logistic_growth(t, baseline, N0, K, mu, lag):
    """Logistic growth model with smooth transition through lag phase"""
    # Standard logistic formula centered at lag time
    # This creates a smooth S-curve with inflection point at t = lag
    growth = K / (1 + ((K - N0) / N0) * np.exp(-mu * (t - lag)))
    return baseline + growth


# Generate clean logistic curve
N = logistic_growth(t, 0.05, 0.05, 0.45, 0.15, 30.0)
N = N.tolist()

_ = pd.Series(N, index=t).plot(
    title="Synthetic Growth Curve", xlabel="Time (hours)", ylabel="OD"
)

# %% [markdown]
# Fit and compare all methods using the
# [`compare_methods`](growthcurves.inference.compare_methods) function,
# then filter to only
# phenomenological and non-parametric models for comparison.

# %% tags=["hide-input"]
# Fit and extract stats for all phenomenological models (parametric and non-parametric)
phenom_fits, phenom_stats = compare_methods(
    t,
    N,
    model_family="all",  # Include mechanistic, phenomenological, and non-parametric
    phase_boundary_method="tangent",  # tangent or threshold
    spline=0.2,
    window_points=7,
)

# Filter to only phenomenological and non-parametric models for comparison
phenom_model_names = [
    "phenom_logistic",
    "phenom_gompertz",
    "phenom_gompertz_modified",
    "phenom_richards",
    "spline",
    "sliding_window",
]
phenom_fits = {k: v for k, v in phenom_fits.items() if k in phenom_model_names}
phenom_stats = {k: v for k, v in phenom_stats.items() if k in phenom_model_names}

# Phase boundary comparison on spline fit
fit_spline = phenom_fits.get("spline")
if fit_spline is None:
    raise RuntimeError(
        f"No spline fit produced; available fits: {list(phenom_fits.keys())}"
    )

phase_boundary_rows = []

# Tangent method
stats_tangent = gc.inference.extract_stats(
    fit_spline, t, N, phase_boundary_method="tangent"
)
phase_boundary_rows.append(
    {
        "label": "tangent",
        "method": "tangent",
        "lag_threshold": np.nan,
        "exp_threshold": np.nan,
        "stats": stats_tangent,
    }
)

# Threshold methods
for frac, label in [(0.10, "threshold_low"), (0.30, "threshold_high")]:
    stats_threshold = gc.inference.extract_stats(
        fit_spline,
        t,
        N,
        phase_boundary_method="threshold",
        lag_threshold=frac,
        exp_threshold=frac,
    )
    phase_boundary_rows.append(
        {
            "label": label,
            "method": "threshold",
            "lag_threshold": frac,
            "exp_threshold": frac,
            "stats": stats_threshold,
        }
    )

print(f"Generated {len(N)} data points over {t[-1]:.1f} hours")
print(f"OD range: {min(N):.3f} to {max(N):.3f}")
print(f"Fitted {len(phenom_fits)} phenomenological models")

# %% [markdown]
# ## Mechanistic Models - Fit Visualization
# Example: Plot phenomenological Richards model

# %% tags=["hide-input"]
# Example: Plot phenomenological Richards model
# Fit phenomenological parametric models
fit_phenom_richards = gc.parametric.fit_parametric(t, N, method="phenom_richards")
stats_phenom_richards = gc.inference.extract_stats(
    fit_phenom_richards, t, N, phase_boundary_method="tangent"
)

# Create base plot with data
scale = "log"
fig = gc.plot.create_base_plot(t, N, scale=scale)

# Annotate with fit and growth statistics (all annotations shown by default)
fig = gc.plot.annotate_plot(
    fig,
    fit_result=fit_phenom_richards,
    stats=stats_phenom_richards,
    scale=scale,
)

# Add title and display
fig.update_layout(
    title="Phenomenological Richards Model",
    height=500,
    width=800,
    template="plotly_white",
)
fig.show()

# %% [markdown]
# ## Phenomenological Models - Growth Statistics Comparison
#
# Compare growth statistics across all phenomenological methods (parametric and
# non-parametric).

# %% tags=["hide-input"]
# Fit and extract stats for all phenomenological models (parametric and non-parametric)
phenom_fits, phenom_stats = compare_methods(
    t,
    N,
    model_family="all",  # Include mechanistic, phenomenological, and non-parametric
    phase_boundary_method="tangent",  # tangent or threshold
    spline=0.2,
    window_points=7,
)

# Filter to only phenomenological and non-parametric models for comparison
phenom_model_names = [
    "phenom_logistic",
    "phenom_gompertz",
    "phenom_gompertz_modified",
    "phenom_richards",
    "spline",
    "sliding_window",
]
phenom_fits = {k: v for k, v in phenom_fits.items() if k in phenom_model_names}
phenom_stats = {k: v for k, v in phenom_stats.items() if k in phenom_model_names}

# Plot growth statistics comparison for phenomenological models
fig_phenom_stats = plot_growth_stats_comparison(
    phenom_stats,
    title="Phenomenological models: growth statistics comparison",
)

fig_phenom_stats.show()

# Display as table
phenom_df = pd.DataFrame(phenom_stats).T[
    [
        "mu_max",
        "doubling_time",
        "time_at_umax",
        "exp_phase_start",
        "exp_phase_end",
        "model_rmse",
        "fit_method",
    ]
]
phenom_df

# %% [markdown]
# ## Phase Boundary Detection Methods
#
# Visualize how different phase boundary detection methods affect exponential phase
# identification.
#
# Two methods are available:
#
# #### 1. **Threshold Method**
# - Tracks instantaneous specific growth rate μ(t)
# - Phase starts when μ exceeds a fraction of μ_max (default: 15%)
# - Phase ends when μ drops below the threshold
#
# #### 2. **Tangent Method**
# - Constructs a tangent line in log space at maximum growth rate
# - Extends tangent to intersect baseline and plateau

# %% [markdown]
# ## Generate the phase boundary comparison
# - see plots

# %% tags=["hide-input"]
# Phase boundary comparison on spline fit
fit_spline = gc.non_parametric.fit_non_parametric(
    t, N, method="spline", spline=0.2, window_points=7
)

phase_boundary_rows = []

# Tangent method
stats_tangent = gc.inference.extract_stats(
    fit_spline, t, N, phase_boundary_method="tangent"
)
phase_boundary_rows.append(
    {
        "label": "tangent",
        "method": "tangent",
        "lag_threshold": np.nan,
        "exp_threshold": np.nan,
        "stats": stats_tangent,
    }
)

# Threshold methods
for frac, label in [(0.10, "threshold_low"), (0.30, "threshold_high")]:
    stats_threshold = gc.inference.extract_stats(
        fit_spline,
        t,
        N,
        phase_boundary_method="threshold",
        lag_threshold=frac,
        exp_threshold=frac,
    )
    phase_boundary_rows.append(
        {
            "label": label,
            "method": "threshold",
            "lag_threshold": frac,
            "exp_threshold": frac,
            "stats": stats_threshold,
        }
    )


# %% tags=["hide-input"]
def build_phase_plot(label, stats, fitted_model):
    fig = gc.plot.create_base_plot(t, N, scale="log")
    # All annotations shown by default, including tangent line
    fig = gc.plot.annotate_plot(
        fig,
        fit_result=fitted_model,
        stats=stats,
        scale="log",
    )
    fig.update_layout(title=label, height=500, width=800, template="plotly_white")
    return fig


# Create plots for each phase boundary method
fig_tangent = build_phase_plot(
    "Spline fit + tangent phase boundaries",
    phase_boundary_rows[0]["stats"],
    fit_spline,
)
fig_threshold_low = build_phase_plot(
    "Spline fit + threshold phase boundaries (low=0.10)",
    phase_boundary_rows[1]["stats"],
    fit_spline,
)
fig_threshold_high = build_phase_plot(
    "Spline fit + threshold phase boundaries (high=0.30)",
    phase_boundary_rows[2]["stats"],
    fit_spline,
)

fig_tangent.show()
fig_threshold_low.show()
fig_threshold_high.show()

# %% [markdown]
# ## Derivative Visualizations
#
# Visualize growth curves and their derivatives:
# - **Specific growth rate (μ)**: d(ln N)/dt - the per capita growth rate
# - **First derivative (dOD/dt)**: The rate of change of OD

# %% tags=["hide-input"]
# Use spline fit for derivative plots
stats_for_derivative = phenom_stats.get("spline")
if stats_for_derivative is None:
    raise RuntimeError(
        f"No spline stats available; available models: {list(phenom_stats.keys())}"
    )

phase_bounds = (
    stats_for_derivative["exp_phase_start"],
    stats_for_derivative["exp_phase_end"],
)

# Plot specific growth rate (mu)
fig_mu = gc.plot.plot_derivative_metric(
    t,
    N,
    metric="mu",
    fit_result=fit_spline,
    phase_boundaries=phase_bounds,
    title="Specific growth rate (mu)",
)

# Plot first derivative (dOD/dt)
fig_doddt = gc.plot.plot_derivative_metric(
    t,
    N,
    metric="dndt",
    fit_result=fit_spline,
    phase_boundaries=phase_bounds,
    title="First derivative (dOD/dt)",
)

fig_mu.show()
fig_doddt.show()
