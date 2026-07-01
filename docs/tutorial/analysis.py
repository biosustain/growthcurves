# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.19.2
#   kernelspec:
#     display_name: growthcurves_env
#     language: python
#     name: python3
# ---

# %% [markdown]
# # Fit growth models and extract growth statistics
#
# This tutorial demonstrates how to fit growth models and extract growth statistics
# using the growthcurves package.
#
# The analysis workflow includes:
# 1. Generating or loading growth data
# 2. Fitting **mechanistic** models (ODE-based, parametric)
# 3. Fitting **phenomenological** models (parametric and non-parametric)
# 4. Extracting growth statistics from all fits
# 5. Saving results for visualization
#
#
# For preprocessing examples (blank subtraction, outlier detection, path length correction), see the companion notebook:
# [`preprocessing.ipynb`](preprocessing.ipynb).
#
# For visualization of the results, see the companion notebook:
# [`plotting.ipynb`](plotting.ipynb) (Visualize fitted growth curves, derivatives,
#  and growth statistics)

# %%
from pprint import pprint

import numpy as np
import pandas as pd

import growthcurves as gc

# %% [markdown]
# ## Generate synthetic data
#
# This cell generates synthetic growth data from a clean logistic function.
# - time is modeled in hours, with measurements every 12 minutes (0.2 hours) for
#   a total of 440 points (88 hours).
# - We assume a lag of 30 hours, an intrinsic growth rate of 0.15 hour⁻¹,
#   and a carrying capacity of 0.45 OD.

# %% tags=["hide-input"]
# Generate synthetic growth data from logistic function
np.random.seed(42)

# Parameters for synthetic growth curve
n_points = 440
measurement_interval_minutes = 12
t = np.array([(measurement_interval_minutes * n) / 60 for n in range(n_points)])


def logistic_growth(t, N0, K, mu, lag):
    """Logistic growth model with smooth transition through lag phase"""
    # Standard logistic formula centered at lag time
    # This creates a smooth S-curve with inflection point at t = lag + (K - N0) / N0
    factor = (K - N0) / N0
    growth = K / (1 + factor * np.exp(-mu * (t - lag)))
    return growth, lag + np.log(factor) / mu


def get_logistic_growth_and_rate(t, K, N0, mu, lag):
    """
    Returns (Population, Growth_Rate) at time t.
    """
    p_t, _ = logistic_growth(t, N0, K, mu, lag)
    derivative = mu * p_t * (1 - (p_t / K))
    return p_t, derivative


# Example: At the inflection point (where P = K/2)
# log_der = mu * (1 - 0.5) = 0.5 * mu

# Generate clean logistic curve
K = 2.45
mu = 0.15
N0 = 0.05
lag = 10.0
ln_N, t_inflec = logistic_growth(t, N0=N0, K=K, mu=mu, lag=lag)

ax = pd.Series(ln_N, index=t).plot(
    title="Synthetic Growth Curve", xlabel="Time (hours)", ylabel="$ln(OD/N_0)$"
)
_ = ax.vlines(
    t_inflec,
    ymin=N0,
    ymax=K,
    color="red",
    linestyle="--",
)
der_inflec, p_inflec = mu * K / 4, (K / 2 + N0)
_ = ax.annotate(
    f"Inflection Point\nt={t_inflec:.2f}\n"
    f"$\\frac{{dP}}{{dt}}_{{inflection}}$={der_inflec:.5f}",
    xy=(t_inflec, p_inflec),
    xytext=(t_inflec + 2, 0.15),
    arrowprops=dict(arrowstyle="->", connectionstyle="arc3,rad=.2"),
)
delta_t = np.log(2 + np.sqrt(3)) / mu
t_accel, p_accel = t_inflec - delta_t, K * (0.5 - np.sqrt(3) / 6)
_ = ax.vlines(
    t_accel,
    ymin=N0,
    ymax=K,
    color="green",
    linestyle="--",
)
der_max = 1 / 6 * mu * K
_ = ax.annotate(
    f"Mu max\nt={t_accel:.2f}\n$\\frac{{dP}}{{dt}}_{{max}}$={der_max:.5f}",
    xy=(t_accel, p_accel),
    xytext=(t_accel - 22, 0.25),
    arrowprops=dict(arrowstyle="->", connectionstyle="arc3,rad=.2"),
)
doubling_time_at_inflection = np.log(2) / (mu * (1 - (p_accel) / K))
print(
    "Doubling time at maximum acceleration point "
    f"(t={t_accel:.2f}): {doubling_time_at_inflection:.2f} hours"
)

max_mu = mu * (1 - ((p_accel) / K))
print(
    "Maximum specific growth rate (mu_max of log curve)"
    f" at t={t_accel:.2f}: {max_mu:.5f} hour^-1"
)

# %% [markdown]
# If we go from logarithmic to linear space, the curve looks very similar, but the
# interpretation of the y-axis changed. Starting from linear space is recommended in
# practice.

# %%
N = N0 * np.exp(ln_N)

ax = pd.Series(N, index=t).plot(
    title="Synthetic Growth Curve", xlabel="Time (hours)", ylabel="OD"
)
_ = ax.vlines(
    t_inflec,
    ymin=N0,
    ymax=max(N),
    color="red",
    linestyle="--",
)
der_inflec, p_inflec = mu * K / 4, (K / 2 + N0)
_ = ax.annotate(
    (
        f"Inflection Point\nt={t_inflec:.2f}\n"
        # f"$\\frac{{dP}}{{dt}}_{{inflection}}$={der_inflec:.5f}"
    ),
    xy=(t_inflec, 0.063),
    xytext=(t_inflec + 2, 0.055),
    arrowprops=dict(arrowstyle="->", connectionstyle="arc3,rad=.2"),
)
delta_t = np.log(2 + np.sqrt(3)) / mu
t_accel, p_accel = t_inflec - delta_t, K * (0.5 - np.sqrt(3) / 6)
_ = ax.vlines(
    t_accel,
    ymin=N0,
    ymax=max(N),
    color="green",
    linestyle="--",
)
der_max = 1 / 6 * mu * K
_ = ax.annotate(
    (
        f"Mu max\nt={t_accel:.2f}"
        # f"\n$\\frac{{dP}}{{dt}}_{{max}}$={der_max:.5f}"
    ),
    xy=(t_accel, 0.055),
    xytext=(t_accel - 22, 0.06),
    arrowprops=dict(arrowstyle="->", connectionstyle="arc3,rad=.2"),
)
doubling_time_at_inflection = np.log(2) / (mu * (1 - (p_accel) / K))
print(
    "Doubling time at maximum acceleration point "
    f"(t={t_accel:.2f}): {doubling_time_at_inflection:.2f} hours"
)

max_mu = mu * (1 - ((p_accel) / K))
print(
    "Maximum specific growth rate (mu_max of log curve)"
    f" at t={t_accel:.2f}: {max_mu:.5f} hour^-1"
)

# %% [markdown]
# ## How Growth Parameters Are Calculated
#
# The table below summarizes how the main reported growth statistics are calculated
# across model classes.
#
# | Output key | Meaning | How it is calculated |
# |---|---|---|
# | `max_od` | Maximum observed/fitted OD | Maximum OD over the valid data range |
# | `mu_max` | Maximum specific growth rate (μ_max) | Maximum of `d(ln N)/dt` from the fitted model (or local fit for non-parametric) |
# | `intrinsic_growth_rate` | Intrinsic model rate parameter | For mechanistic models: fitted intrinsic `μ`; for phenomenological/non-parametric: `None` |
# | `doubling_time` | Doubling time in hours | `ln(2) / mu_max` |
# | `time_at_umax` | Time at maximum specific growth | Time where `mu_max` reaches its maximum |
# | `od_at_umax` | OD at time of μ_max | Model-predicted OD at `time_at_umax` |
# | `exp_phase_start`, `exp_phase_end` | Exponential phase boundaries | From threshold or tangent phase-boundary method in `extract_stats()` |
# | `model_rmse` | Fit error | RMSE between observed OD and model-predicted OD over the model fit window |
#
# For this tutorial:
# - Mechanistic comparisons use mechanistic parametric fits.
# - Phenomenological comparisons include both phenomenological parametric and non-parametric fits.
#

# %% [markdown]
# ## Extract growth stats from the dataset
#
# The `extract_stats_from_fit()` function calculates these key metrics:
#
# - `max_od`: Maximum OD value within the fitted window
# - `mu_max`: **Observed** maximum specific growth rate μ_max (hour⁻¹) - calculated
#   from the fitted curve
# - `intrinsic_growth_rate`: **Model parameter** for intrinsic growth rate
#   (parametric models only, `None` for non-parametric)
# - `doubling_time`: Time to double the population at peak growth (hours)
# - `exp_phase_start`: When exponential phase begins (hours)
# - `exp_phase_end`: When exponential phase ends (hours)
# - `time_at_umax`: Time when μ reaches its maximum (hours)
# - `od_at_umax`: OD value at time of maximum μ
# - `fit_t_min`: Start of fitting window (hours)
# - `fit_t_max`: End of fitting window (hours)
# - `fit_method`: Identifier for the method used
# - `model_rmse`: Root mean squared error
#
# Descriptive parameters are extracted from the fits. Where parameters are not extracted
# directly from the fitted model, they are calculated. The table below shows how
# different stats are calculated according to the different approaches:
#
# ### MECHANISTIC MODELS
#
# > Use linear space (`N(t)`) to fit the mechanistic models.
#
# | Name | Model | Equation | Exp Start | Exp End | Intrinsic μ | μ max | Carrying Capacity | Fit |
# |------|-------|----------|-----------|---------|-------------|-------|-------------------|-----|
# | Logistic | parametric | `dN/dt = μ * (1 - N(t) / K) * N(t)` | threshold/<br>tangent | threshold/<br>tangent | μ | max dln(N)/dt | K | entire curve |
# | Gompertz | parametric | `dN/dt = μ * math.log(K / N(t)) * N(t)` | threshold/<br>tangent | threshold/<br>tangent | μ | max dln(N)/dt | K | entire curve |
# | Richards | parametric | `dN/dt = μ * (1 - (N(t) / K)**beta) * N(t)` | threshold/<br>tangent | threshold/<br>tangent | μ | max dln(N)/dt | A | entire curve |
# | Baranyi  | parametric | `dN/dt= μ * math.exp(μ * t) / (math.exp(h0) - 1 + math.exp(μ * t)) * (1 - N(t) / K) * N(t)` | threshold/<br>tangent | threshold/<br>tangent | μ | max dln(N)/dt | K | entire curve |
#
# try transposed display
#
# | Name | Logistic | Gompertz | Richards | Baranyi |
# |------|----------|---------|----------|---------|
# | type | parametric | parametric | parametric | parametric |
# | Equation | `dN/dt = μ * (1 - N(t) / K) * N(t)` | `dN/dt = μ * math.log(K / N(t)) * N(t)` | `dN/dt = μ * (1 - (N(t) / K)**beta) * N(t)` | `dN/dt= μ * math.exp(μ * t) / (math.exp(h0) - 1 + math.exp(μ * t)) * (1 - N(t) / K) * N(t)` |
# | Exp Start | threshold/<br>tangent | threshold/<br>tangent | threshold/<br>tangent | threshold/<br>tangent |
# | Exp End | threshold/<br>tangent | threshold/<br>tangent | threshold/<br>tangent | threshold/<br>tangent |
# | Intrinsic μ | μ | μ | μ | μ |
# | μ max | max dln(N)/dt | max dln(N)/dt | max dln(N)/dt | max dln(N)/dt |
# | Carrying Capacity | K | K | A | K |
# | Fit | entire curve | entire curve | entire curve | entire curve |
#
# ### PHENOMENOLOGICAL MODELS
#
# > Use log space (`ln(N(t)/N0)`) to fit the phenomenological models.
#
# | Name | Model | Equation | Exp Start | Exp End | Intrinsic μ | μ max | Max OD | Fit |
# |------|-------|----------|-----------|---------|-------------|-------|--------|-----|
# | Linear | non-parametric | `ln(N(t)) = N0 + b * t` | threshold/<br>tangent | threshold/<br>tangent | n.a. | b | max OD raw | only window |
# | Spline | non-parametric | `ln(N(t)) = spline(t)` | threshold/<br>tangent | threshold/<br>tangent | n.a. | max of derivative of spline | max OD raw | entire curve |
# | Logistic (phenom) | parametric | `ln(N(t)/N0) = A / (1 + exp(4 * μ_max * (λ - t) / A + 2))` | λ | threshold/<br>tangent | n.a. | μ_max | K | entire curve |
# | Gompertz (phenom) | parametric | `ln(N(t)/N0) = A * exp(-exp(μ_max * exp(1) * (λ - t) / A + 1))` | λ | threshold/<br>tangent | n.a. | μ_max | K | entire curve |
# | Gompertz (modified) | parametric | `ln(N(t)/N0) = A * exp(-exp(μ_max * exp(1) * (λ - t) / A + 1)) + A * exp(α * (t - t_shift))` | λ | threshold/<br>tangent | n.a. | μ_max | K | entire curve |
# | Richards (phenom) | parametric | `ln(N(t)/N0) = A * (1 + ν * exp(1 + ν + μ_max * (1 + ν)**(1/ν) * (λ - t) / A))**(-1/ν)` | λ | threshold/<br>tangent | n.a. | μ_max | K | entire curve |
#
# ### Understanding Growth Rates: Intrinsic vs. Observed
#
# **Important distinction:**
#
# - **`mu_max`** (μ_max): The **observed** maximum specific growth rate calculated
#   from the fitted curve as max(d(ln N)/dt). This is what you measure from the data.
#
# - **`intrinsic_growth_rate`**: The **model parameter** representing intrinsic growth
#   capacity:
#   - **Parametric models**: This is a fitted parameter (e.g., `r` in Logistic,
#     `mu_max` in Gompertz)
#   - **Non-parametric methods**: Returns `None` (no model parameter exists)

# %% [markdown]
# ## Mechanistic Models
#
# Mechanistic models are ODE-based parametric models that encode growth dynamics as
# differential equations.
#
# ### Fit Models

# %%
# Fit mechanistic models
fit_mech_logistic = gc.parametric.fit_parametric(t, ln_N, method="mech_logistic")
fit_mech_gompertz = gc.parametric.fit_parametric(t, ln_N, method="mech_gompertz")
fit_mech_richards = gc.parametric.fit_parametric(t, ln_N, method="mech_richards")
fit_mech_baranyi = gc.parametric.fit_parametric(t, ln_N, method="mech_baranyi")

# Combine fits into a dictionary
mechanistic_fits = {
    "mech_logistic": fit_mech_logistic,
    "mech_gompertz": fit_mech_gompertz,
    "mech_richards": fit_mech_richards,
    "mech_baranyi": fit_mech_baranyi,
}

# Display example fit result
print("=== Logistic Fit Result ===")
pprint(fit_mech_logistic, indent=2)

# %% [markdown]
# ### Extract Growth Statistics

# %%
# Extract stats from each mechanistic fit
stats_mech_logistic = gc.inference.extract_stats(fit_mech_logistic, t, ln_N)
stats_mech_gompertz = gc.inference.extract_stats(fit_mech_gompertz, t, ln_N)
stats_mech_richards = gc.inference.extract_stats(fit_mech_richards, t, ln_N)
stats_mech_baranyi = gc.inference.extract_stats(fit_mech_baranyi, t, ln_N)

# Combine stats into a dictionary
mechanistic_stats = {
    "mech_logistic": stats_mech_logistic,
    "mech_gompertz": stats_mech_gompertz,
    "mech_richards": stats_mech_richards,
    "mech_baranyi": stats_mech_baranyi,
}

# Display growth statistics for logistic fit
print("=== Logistic Growth Statistics ===")
pprint(stats_mech_logistic, indent=2)

# Create comparison dataframe
print("\n=== Mechanistic Models Comparison ===")
mechanistic_df = pd.DataFrame(mechanistic_stats).T[
    [
        "mu_max",
        "intrinsic_growth_rate",
        "doubling_time",
        "time_at_umax",
        "exp_phase_start",
        "exp_phase_end",
        "model_rmse",
    ]
]
mechanistic_df.T

# %% [markdown]
# ## Phenomenological Models - Parametric
#
# These are phenomenological parametric models fit in ln-space.
#
# ### Fit Models

# %%
# Fit phenomenological parametric models
fit_phenom_logistic = gc.parametric.fit_parametric(t, ln_N, method="phenom_logistic")
fit_phenom_gompertz = gc.parametric.fit_parametric(t, ln_N, method="phenom_gompertz")
fit_phenom_gompertz_modified = gc.parametric.fit_parametric(
    t, ln_N, method="phenom_gompertz_modified"
)
fit_phenom_richards = gc.parametric.fit_parametric(t, ln_N, method="phenom_richards")

# Combine fits into a dictionary
phenom_param_fits = {
    "phenom_logistic": fit_phenom_logistic,
    "phenom_gompertz": fit_phenom_gompertz,
    "phenom_gompertz_modified": fit_phenom_gompertz_modified,
    "phenom_richards": fit_phenom_richards,
}

# Display example fit
print("=== Phenomenological Logistic Fit ===")
pprint(fit_phenom_logistic, indent=2)
pprint("\n=== Phenomenological Richards Fit ===")
pprint(fit_phenom_richards, indent=2)
pprint("\n=== Phenomenological Gompertz Fit ===")
pprint(fit_phenom_gompertz, indent=2)
pprint("\n=== Phenomenological Modified Gompertz Fit ===")
pprint(fit_phenom_gompertz_modified, indent=2)

# %% [markdown]
# ### Extract Growth Statistics

# %%
# Extract stats from each phenomenological parametric fit
stats_phenom_logistic = gc.inference.extract_stats(
    fit_phenom_logistic, t, ln_N, phase_boundary_method="tangent"
)
stats_phenom_gompertz = gc.inference.extract_stats(
    fit_phenom_gompertz, t, ln_N, phase_boundary_method="tangent"
)
stats_phenom_gompertz_modified = gc.inference.extract_stats(
    fit_phenom_gompertz_modified, t, ln_N, phase_boundary_method="tangent"
)
stats_phenom_richards = gc.inference.extract_stats(
    fit_phenom_richards, t, ln_N, phase_boundary_method="tangent"
)

# Combine stats into a dictionary
phenom_param_stats = {
    "phenom_logistic": stats_phenom_logistic,
    "phenom_gompertz": stats_phenom_gompertz,
    "phenom_gompertz_modified": stats_phenom_gompertz_modified,
    "phenom_richards": stats_phenom_richards,
}

# Display example stats
print("=== Phenomenological Logistic Stats ===")
pprint(stats_phenom_logistic, indent=2)

# Create comparison dataframe
print("\n=== Phenomenological Parametric Models Comparison ===")
phenom_param_df = pd.DataFrame(phenom_param_stats).T[
    [
        "mu_max",
        "intrinsic_growth_rate",
        "doubling_time",
        "time_at_umax",
        "exp_phase_start",
        "exp_phase_end",
        "model_rmse",
    ]
]
phenom_param_df.T

# %% [markdown]
# ## Phenomenological Models - Non-Parametric
#
# These are phenomenological non-parametric fits that estimate growth features directly
# from local trends and smoothing.
#
# For spline fitting, use `smooth` to choose smoothing behavior:
# - `smooth="fast"` (default): auto-default lambda rule
# - `smooth="slow"`: weighted GCV smoothing
# - `smooth=<float>`: manual lambda value
#
# ### Fit Models
#

# %%
# Fit non-parametric models

# Spline supports smooth="fast" (default), smooth="slow", or a manual float.
fit_spline = gc.non_parametric.fit_non_parametric(
    t,
    ln_N,
    method="spline",
    smooth="fast",
)

# Example manual smoothing value:
# fit_manual = gc.non_parametric.fit_non_parametric(t, ln_N, method="spline", smooth=0.5)

fit_sliding_window = gc.non_parametric.fit_non_parametric(
    t,
    ln_N,
    method="sliding_window",
    window_points=7,
)

# Combine fits into a dictionary
phenom_nonparam_fits = {
    "spline": fit_spline,
    "sliding_window": fit_sliding_window,
}

# Display non-parametric fit results
pprint(phenom_nonparam_fits, indent=2)

# %% [markdown]
# ### Extract Growth Statistics

# %%
# Extract stats from each non-parametric fit
stats_spline = gc.inference.extract_stats(
    fit_spline,
    t,
    ln_N,
    phase_boundary_method="tangent",
)

stats_sliding_window = gc.inference.extract_stats(
    fit_sliding_window,
    t,
    ln_N,
    phase_boundary_method="tangent",
)

# Combine stats into a dictionary
phenom_nonparam_stats = {
    "spline": stats_spline,
    "sliding_window": stats_sliding_window,
}

# Create comparison dataframe
print("=== Phenomenological Non-Parametric Models Comparison ===")
phenom_nonparam_df = pd.DataFrame(phenom_nonparam_stats).T[
    [
        "mu_max",
        "intrinsic_growth_rate",
        "doubling_time",
        "time_at_umax",
        "exp_phase_start",
        "exp_phase_end",
        "model_rmse",
    ]
]
phenom_nonparam_df

# %% [markdown]
# ## Customizing Phase Boundary Detection
#
# Two methods are available for determining exponential phase boundaries:
#
# ### 1. **Threshold Method**
# - Tracks the instantaneous specific growth rate μ(t)
# - `exp_phase_start`: First time when μ exceeds a fraction of μ_max (default: 15%)
# - `exp_phase_end`: First time after peak when μ drops below the threshold
#
# ### 2. **Tangent Method**
# - Constructs a tangent line in log space at the point of maximum growth rate
# - Extends this tangent to intersect baseline (exp_phase_start) and plateau
#   (exp_phase_end)

# %%
# Compare phase-boundary methods on the same fit
phase_boundary_rows = []

# Tangent method
stats_tangent = gc.inference.extract_stats(
    fit_spline,
    t,
    ln_N,
    phase_boundary_method="tangent",
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

# Threshold method at different cutoffs
for frac, label in [
    (0.10, "threshold_low"),
    (0.15, "threshold_default"),
    (0.30, "threshold_high"),
]:
    stats_threshold = gc.inference.extract_stats(
        fit_spline,
        t,
        ln_N,
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

# Create comparison dataframe
print("=== Phase Boundary Method Comparison ===")
phase_boundary_df = pd.DataFrame(
    [
        {
            "label": row["label"],
            "method": row["method"],
            "lag_threshold": row["lag_threshold"],
            "exp_threshold": row["exp_threshold"],
            "exp_phase_start": row["stats"]["exp_phase_start"],
            "exp_phase_end": row["stats"]["exp_phase_end"],
        }
        for row in phase_boundary_rows
    ]
)
phase_boundary_df

# %% [markdown]
# See more details on the phase boundary methods in the next tutorial notebook:
# [`plotting.ipynb`](plotting.ipynb) (Visualize fitted growth curves, derivatives,
# and growth statistics)
