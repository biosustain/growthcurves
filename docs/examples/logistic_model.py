# %% [markdown]
# # Logistic Growth Model Simulation and Fitting
# - N0 should be close to zero (as recommended in the review paper)
# - K is the carrying capacity (maximum OD) in linear space
# - A is the log ratio of K to N0, used in the phenomenological model
# - mu_max and mu are the growth rate constants for log-scale and linear-scale models,
#   respectively. mu is only defined for mechanistic models.
#
# The ODE given in the review and the phenomological model are not equivalent, but
# closely related.

# %% tags=["hide-input"]
from pprint import pprint

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import growthcurves as gc
from growthcurves.models import (  # mech_logistic_ode,;
    log_to_linear,
    mech_logistic_model,
    phenom_logistic_model_ln,
)


# from scipy.integrate import solve_ivp
def logistic_growth(t, N0, K, mu, lag):
    """Logistic growth model with smooth transition through lag phase"""
    # Standard logistic formula centered at lag time
    # This creates a smooth S-curve with inflection point at t = lag + (K - N0) / N0
    factor = (K - N0) / N0
    N = K / (1 + factor * np.exp(-mu * (t - lag)))
    if lag > 0:
        # For t < lag, set N to N0 to model the lag phase
        N[t < lag] = N0
    return N

# %% [markdown]
# # 1. Set your simulation parameters

# %% tags=["parameters"]
mu_max = 0.3  # Growth rate constant
K = 5  # Carrying capacity for logistic growth
N0 = 0.3  # Initial condition (must be a list/array)
A = float(np.log((K - N0) / N0))
t_start = 0.0  # Start time
t_end = 60.0  # End time
lag = 12.3  # Lag time
num_points = int(t_end * 12)  # Number of data points to generate

mu = mu_max / (1 - N0 / K)
print(f"mu_max: {mu_max}, mu: {mu}, K: {K}, N0: {N0}")

ground_truth_params = {
    "mu_max": mu_max,
    "mu": mu,
    "K": K,
    "N0": N0,
    "A": A,
    "lag": lag,
}

# %% [markdown]
# ## 2. Create the time grid where you want data points

# %%
t_eval = np.linspace(t_start, t_end, num_points)


# %% [markdown]
# ## 3. Solve the ODE
# args passes extra constants (like k) to the model function
# ```python
# ln_ratio = solve_ivp(
#     fun=mech_logistic_ode,
#     t_span=(t_start, t_end),
#     y0=[y0],
#     t_eval=t_eval,
#     args=(mu, K),
# ).y[0]
# ```
# Extract the solution arrays
# and use the analytical solution for comparison
#
# To model the lag phase, we will shift the solution to the right by the lag time,
# filling in the initial values with N0.

# %%
N = mech_logistic_model(t_eval, mu, K, N0)
if lag > 0:
    idx_lag = int(lag * 12)
    # shift the data to the right by lag time, filling in the initial values with N0
    N = np.concatenate((np.full(idx_lag, N0), N[:-idx_lag]))

# %% [markdown]
# ## 4. Structure the generated data into a clean DataFrame
# And plot.

# %%
data = pd.DataFrame(
    {
        "Time": t_eval,
        "OD_mech": N,
        "ln_OD_mech": np.log(N / N0),
    }
)
ax = data.plot.scatter(x="Time", y="OD_mech", s=1)
ax.vlines(x=lag, ymin=N0, ymax=K, color="red", linestyle="--", label="Lag time")


# %% [markdown]
# ## 5. Generate the phenomenological model for comparison
#
# As in review paper we have a slightly modified logistical model:
#
# ```
# A = (K - N0) / N0
# ln(Nt/N0) =          A / (1 + exp((4 * μ_max / A) * (λ - t) + 2))
# Nt        = N0 * exp(A / (1 + exp((4 * μ_max / A) * (λ - t) + 2)))
# ```
#
# > Note that you can model the lag phase with a time shift in this formulation, which
# > had to be manually added using the mechanistic model upon data generation.
# > Buy contrast N0 is not modeled using the phenomological model(s) operating in log
# > space, so the initial condition has to be inferred from the data.

# %%
A = np.log((K - N0) / N0)
data["OD_phenom_paper_ln"] = phenom_logistic_model_ln(
    t=data["Time"],
    mu_max=mu_max,
    A=A,
    lam=lag,
)
data["OD_phenom_paper"] = log_to_linear(data["OD_phenom_paper_ln"], N0)
data

# %% [markdown]
# ## 6. Generate the classic phenomenological model for comparison
#
# ```
# N(t) = K / (1 + ((K - N0)/N0) * exp(-μ * (t - lag)))
# ```
#
# One subtle but important point:
# If you use
# factor = (K - N0)/N0
# N(t) = K / (1 + factor * exp(-μ * (t - lag)))
# then N0 is not equal to N(0) unless lag = 0. So you usually choose either:
# N(0)=N0 and no extra lag shift, or
# a shifted curve parameterization where the shift replaces the initial-condition
# constant.
# - use only time shift and let A be fitted?

# %%
data["OD_phenom_classic"] = logistic_growth(
    t=data["Time"],
    N0=N0,
    K=K,
    mu=mu_max,
    lag=lag,
)

data["OD_phenom_classic_ln"] = np.log(data["OD_phenom_classic"] / N0)

# %% tags=["hide-input"]
fig, axes = plt.subplots(2, 1, figsize=(7, 6), sharex=True)
ax = data.plot(
    x="Time",
    y="OD_mech",
    title="Logistic Growth Simulation (linear scale)",
    xlabel="Time",
    ylabel="OD",
    label="OD (mechanistic)",
    color="C0",
    alpha=0.7,
    ax=axes[0],
)
ax = data.plot(
    x="Time",
    ax=ax,
    y="OD_phenom_paper",
    label="OD (phenomenological)",
    color="C1",
    alpha=0.7,
)
ax = data.plot(
    x="Time",
    ax=ax,
    y="OD_phenom_classic",
    label="OD (classic phenomenological)",
    color="C2",
    alpha=0.3,
)
ax.legend()
ax2 = axes[1]
ax2 = data.plot(
    x="Time",
    y="ln_OD_mech",
    title="Logistic Growth Simulation (log scale)",
    xlabel="Time",
    ylabel="ln(OD)",
    label="ln(OD) (mechanistic)",
    color="C4",
    alpha=0.7,
    ax=ax2,
)
data.plot(
    x="Time",
    y="OD_phenom_paper_ln",
    xlabel="Time",
    ylabel="ln(OD)",
    label="ln OD Curve (phenomenological)",
    ax=ax2,
    color="C3",
    alpha=0.7,
)

data.plot(
    x="Time",
    y="OD_phenom_classic_ln",
    xlabel="Time",
    ylabel="ln(OD)",
    label="ln(OD) (classic phenomenological)",
    ax=ax2,
    color="C5",
    alpha=0.3,
)
ax2.set_ylabel("ln(OD)")
_ = ax2.legend(title="ln(OD) curves")


# %% [markdown]
# ## 8. Fit the mechanistic model to the synthetic data

# %% [markdown]
# ### 8.1 Helper function to fit the model and extract statistics

# %% tags=["hide-input"]
def fit_model_and_extract_stats(time_in_hours, observations, model):
    fit_mech_logistic = gc.parametric.fit_parametric(
        time_in_hours, observations, method=model
    )
    stats_mech_logistic = gc.inference.extract_stats(
        fit_mech_logistic, time_in_hours, observations
    )
    stats_mech_logistic = {
        k: float(v)
        for k, v in stats_mech_logistic.items()
        if isinstance(v, (int, float, np.number))
    }
    fit_mech_logistic["params"]["model_type"] = fit_mech_logistic["model_type"]
    fit_mech_logistic = fit_mech_logistic["params"]
    # Combine fits into a dictionary
    # Display example fit result
    print("=== Ground Truth Parameters ===")
    pprint(ground_truth_params)
    print(f"=== Fit Result for {model} ===")
    pprint(fit_mech_logistic, indent=2)
    pprint(f"=== Fit Stats for {model} ===")
    pprint(stats_mech_logistic, indent=2)
    return fit_mech_logistic, stats_mech_logistic


# %% [markdown]
# - fit data from the classic logistic model using the methods implemented in
#   growthcurves (based on the review paper)

# %%
model = "phenom_logistic"
col = "OD_mech"
N = data[col]
t = data["Time"]

fit_mech_logistic, stats_mech_logistic = fit_model_and_extract_stats(t, N, model)


# %% [markdown]
# ## 7. Fit the mechanistic model to the synthetic data
# - fit data from the classic logistic model using the methods implemented in
#   growthcurves (based on the review paper)

# %%
model = "phenom_logistic"
col = "OD_phenom_paper"
N = data[col]
t = data["Time"]

fit_mech_logistic, stats_mech_logistic = fit_model_and_extract_stats(t, N, model)

# %% [markdown]
# ## 9. Compare differences between models
#
# Compare regions where models differ:
# - changes maximum capacity K for phenomenological model according to review
# - OD phenomenological and OD mechanistic should match

# %% tags=["hide-input"]
fig, axes = plt.subplots(1, 2, figsize=(7, 3), sharex=True)
ax = data.plot.scatter(x="OD_mech", y="OD_phenom_paper", s=1, color="C1", ax=axes[0])
_ = ax.plot([0, 5], [0, 5], color="black", linestyle="--", label="y=x", alpha=0.5)
ax = data.plot.scatter(x="OD_mech", y="OD_phenom_classic", s=1, color="C1", ax=axes[1])
_ = ax.plot([0, 5], [0, 5], color="black", linestyle="--", label="y=x", alpha=0.5)

# %%
