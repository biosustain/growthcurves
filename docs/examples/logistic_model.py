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
import sympy as sp
from IPython.display import display

import growthcurves as gc
from growthcurves.models import (  # mech_logistic_ode,;
    log_to_linear,
    mech_logistic_model,
    phenom_logistic_model_ln,
)

# classic model


# from scipy.integrate import solve_ivp
def logistic_growth(t, N0, K, mu, lag):
    """Logistic growth model with smooth transition through lag phase"""
    # Standard logistic formula centered at lag time
    # This creates a smooth S-curve with inflection point at t = lag + (K - N0) / N0
    factor = (K - N0) / N0
    N = K / (1 + factor * np.exp(-mu * (t - lag)))
    # if lag > 0:
    # For t < lag, set N to N0 to model the lag phase
    # N[t < lag] = N0
    return N


def get_acceleration(t, K, N0, mu, lag):
    """
    Returns the acceleration (second derivative) at t t.
    """
    N = logistic_growth(t, K, N0, mu, lag)
    accel = mu**2 * N * (1 - (N / K)) * (1 - (2 * N / K))
    accel[t < lag] = 0
    return accel


def get_doubling_time(t, K, N0, mu, lag):
    """
    Returns the instantaneous doubling time at time t.
    """
    N = logistic_growth(t, K, N0, mu, lag)
    with np.errstate(divide="ignore"):
        doubling_time = np.log(2) / (mu * (1 - (N / K)))
    doubling_time[t < lag] = np.nan  # Undefined during lag phase
    return doubling_time


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
# # 2. Create the time grid where you want data points

# %%
t_eval = np.linspace(t_start, t_end, num_points)


# %% [markdown]
# # 3. Solve the ODE
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
# Shift the solution to the right by the lag time (in continuous time, not by a
# fixed number of grid points, so it lines up exactly with the closed-form
# comparison below regardless of the time grid spacing).
post_lag = t_eval >= lag
N = np.full_like(t_eval, N0)
N[post_lag] = mech_logistic_model(t_eval[post_lag] - lag, mu, K, N0)

# %% [markdown]
# # 4. Structure the generated data into a clean DataFrame
# And plot.

# %%
data = pd.DataFrame(
    {
        "Time": t_eval,
        "OD_mech": N,
        "ln_OD_mech": np.log(N / N0),
    }
)
ax = data.plot.scatter(
    x="Time",
    y="OD_mech",
    s=1,
    color="C0",
    title="Mechanistic Logistic Growth Simulation",
    xlabel="Time (hours)",
    ylabel="OD",
)
ax.vlines(
    x=lag,
    ymin=N0,
    ymax=K,
    color="red",
    linestyle="--",
    label="Lag time ends",
)
_ = ax.legend()


# %% [markdown]
# # 5. Generate the phenomenological model for comparison
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
    t=data["Time"], mu_max=mu_max, A=A, lam=lag, ln_N0=np.log(N0)
)
data["OD_phenom_paper"] = np.exp(data["OD_phenom_paper_ln"])
data

# %% tags = ["hide-input"]
ax = data.plot.scatter(
    x="Time",
    y="OD_mech",
    s=1,
    color="C0",
    title="Mechanistic Logistic Growth Simulation",
    xlabel="Time (hours)",
    ylabel="OD",
)
_ = data.plot.scatter(
    x="Time",
    y="OD_phenom_paper",
    s=1,
    color="C1",
    ax=ax,
    label="Phenomenological Logistic Growth",
)
ax.vlines(
    x=lag,
    ymin=N0,
    ymax=K,
    color="red",
    linestyle="--",
    label="Lag time ends",
)
_ = ax.legend()

# %% [markdown]
# # 6. Generate the classic phenomenological model for comparison
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
    mu=mu,  # the ODE rate constant, not mu_max — see 1. Set your simulation parameters
    lag=lag,
)
data["OD_phenom_classic_1der"] = logistic_growth(
    t=data["Time"], K=K, N0=N0, mu=mu, lag=lag
)
data["OD_phenom_classic_2der"] = get_acceleration(
    t=data["Time"], K=K, N0=N0, mu=mu, lag=lag
)
data["OD_phenom_classic_doubling_time"] = get_doubling_time(
    t=data["Time"], K=K, N0=N0, mu=mu, lag=lag
)
data["OD_phenom_classic_ln"] = np.log(data["OD_phenom_classic"] / N0)
data.set_index("Time").filter(like="OD_phenom_classic").plot(
    subplots=True, layout=(3, 2), figsize=(7, 6), sharex=True
)

# %%

# %%
ax = data.plot.scatter(
    x="Time",
    y="OD_mech",
    s=1,
    color="C0",
    label="Mechanistic Logistic Growth Simulation (ODE)",
    xlabel="Time (hours)",
    alpha=0.5,
    ylabel="OD",
)
_ = data.plot.scatter(
    x="Time",
    y="OD_phenom_classic",
    s=1,
    color="C1",
    ax=ax,
    alpha=0.5,
    label="Classic Logistic Growth (phenomenological)",
)
_ = ax.vlines(
    x=lag,
    ymin=N0,
    ymax=K,
    color="red",
    linestyle="--",
    label="Lag time ends",
    alpha=0.5
)
_ = ax.legend()

# %% tags=["hide-input"]
ax = pd.Series(N, index=data["Time"]).plot(
    title="Synthetic Growth Curve", xlabel="Time (hours)", ylabel="OD"
)

# Inflection point of the OD curve: N = K/2, where the absolute growth rate
# dN/dt = mu * N * (1 - N/K) is maximal. Solve N(t) = K/2 analytically instead
# of taking idxmax of the (numerically noisy) acceleration column, which finds
# the peak of d²N/dt² rather than its zero-crossing at the true inflection.
_ = ax.hlines(
    K / 2,
    # ((K-N0) / 2) + N0,
    xmin=1.0,
    xmax=t_end,
    alpha=0.2,
    color="grey",
    linestyle="--",
    label="Inflection Point",
)
t_inflec = lag + np.log((K - N0) / N0) / mu
p_inflec = K / 2
der_inflec = mu * K / 4
_ = ax.vlines(
    t_inflec,
    ymin=N0,
    ymax=K,
    color="red",
    linestyle="--",
)
_ = ax.annotate(
    f"Inflection Point\nt={t_inflec:.2f}\n"
    f"$\\frac{{dP}}{{dt}}_{{inflection}}$={der_inflec:.5f}",
    xy=(t_inflec, p_inflec),
    xytext=(t_inflec + 10, K / 2),
    arrowprops=dict(arrowstyle="->", connectionstyle="arc3,rad=.2"),
)

# Maximum specific growth rate: mu(t) = (1/N) * dN/dt = mu * (1 - N/K) decreases
# monotonically as N grows, so it peaks at the smallest N, i.e. right when the
# lag phase ends (t=lag, N=N0) — exactly where it equals mu_max, since
# mu = mu_max / (1 - N0/K) by construction (see simulation parameters above).
t_mumax, p_mumax = lag, N0
_ = ax.vlines(
    t_mumax,
    ymin=N0,
    ymax=K,
    color="green",
    linestyle="--",
)
_ = ax.annotate(
    f"Max Specific Growth Rate\nt={t_mumax:.2f}\n$\\mu_{{max}}$={mu_max:.5f}",
    xy=(t_mumax, p_mumax),
    xytext=(t_mumax + 15, 1.0),
    arrowprops=dict(arrowstyle="->", connectionstyle="arc3,rad=.2"),
)
doubling_time_at_inflection = np.log(2) / (mu * (1 - p_inflec / K))

# %%
# %%
print(f"Time of mu_max: {lag + np.log((K - N0) / N0) / mu}")
data.set_index("Time").filter(like="OD_phenom_classic").idxmax()

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
# # 7. Fit the mechanistic model to the synthetic data

# %% [markdown]
# ## 7.1 Helper function to fit the model and extract statistics


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
# ## 7.2 Fit the mechanistic model to the synthetic data
# - fit data from the classic logistic model using the methods implemented in
#   growthcurves (based on the review paper)

# %%
model = "phenom_logistic"
col = "OD_phenom_paper"
N = data[col]
t = data["Time"]

fit_mech_logistic, stats_mech_logistic = fit_model_and_extract_stats(t, N, model)

# %% [markdown]
# # 8. Compare differences between models
#
# Compare regions where models differ:
# - changes maximum capacity K for phenomenological model according to review
# - OD phenomenological and OD mechanistic should match

# %% tags=["hide-input"]
fig, axes = plt.subplots(1, 2, figsize=(7, 3), sharex=True)
ax = data.plot.scatter(x="OD_mech", y="OD_phenom_paper", s=1, color="C1", ax=axes[0])
_ = ax.plot([0, K], [0, K], color="black", linestyle="--", label="y=x", alpha=0.5)
ax = data.plot.scatter(x="OD_mech", y="OD_phenom_classic", s=1, color="C1", ax=axes[1])
_ = ax.plot([0, K], [0, K], color="black", linestyle="--", label="y=x", alpha=0.5)

# %% [markdown]
# # 9. Symbolic derivatives of the models


# %% tags=["hide-input"]
t, s_K, s_N0, s_mu, s_lag = sp.symbols("t K N0 mu lag", positive=True)
factor = (s_K - s_N0) / s_N0
N = s_K / (1 + factor * sp.exp(s_mu * (s_lag - t)))
print("Logistic growth model N(t):")
display(N)
dN = sp.diff(N, t)
print("First derivative of N(t):")
display(dN)
d2N = sp.diff(N, t, 2)
print("Second derivative of N(t):")
display(d2N)
# create a function
f_N = sp.lambdify((t, s_N0, s_mu, s_K, s_lag), N, modules="numpy")
f_dN = sp.lambdify((t, s_N0, s_mu, s_K, s_lag), dN, modules="numpy")
f_d2N = sp.lambdify((t, s_N0, s_mu, s_K, s_lag), d2N, modules="numpy")

dN_eval = f_dN(t=t_eval, N0=N0, mu=mu, K=K, lag=lag)
d2N_eval = f_d2N(t=t_eval, N0=N0, mu=mu, K=K, lag=lag)

t_logistic_classic_max_od_increase = t_eval[np.argmax(dN_eval)]
v_logistic_classic_max_od_increase = np.max(dN_eval)
t_logistic_classic_max_acceleration = t_eval[np.argmax(d2N_eval)]
v_logistic_classic_max_acceleration = np.max(d2N_eval)

print(
    "Max growth rate (first derivative) at "
    f"t={t_logistic_classic_max_od_increase:.2f}: {v_logistic_classic_max_od_increase:.5f}"
)
print(
    "Max acceleration (second derivative) at "
    f"t={t_logistic_classic_max_acceleration:.2f}: {v_logistic_classic_max_acceleration :.5f}"
    "\n\t with growth rate at that time: "
    f"{dN_eval[np.argmax(d2N_eval)]:.5f}"
)

# %% tags=["hide-input"]
s_mu_max, s_A, s_lam = sp.symbols("mu_max A lam", positive=True)
N_pheno = s_N0 * sp.exp(s_A) / (1 + sp.exp((4 * s_mu_max / s_A * (s_lam - t)) + 2))
dN = sp.diff(N_pheno, t)
print("First derivative of N(t):")
display(dN)
d2N = sp.diff(N_pheno, t, 2)
print("Second derivative of N(t):")
display(d2N)
# create a function
f_dN = sp.lambdify((t, s_N0, s_mu_max, s_A, s_lam), dN, modules="numpy")
f_d2N = sp.lambdify((t, s_N0, s_mu_max, s_A, s_lam), d2N, modules="numpy")

dN_eval = f_dN(t=t_eval, N0=N0, mu_max=mu_max, A=A, lam=lag)
d2N_eval = f_d2N(t=t_eval, N0=N0, mu_max=mu_max, A=A, lam=lag)

print(
    f"Max growth rate (first derivative) at "
    f"t={t_eval[np.argmax(dN_eval)]:.2f}: {np.max(dN_eval):.5f}"
)
print(
    f"Max acceleration (second derivative) at "
    f"t={t_eval[np.argmax(d2N_eval)]:.2f}: {np.max(d2N_eval):.5f}"
)

# %% [markdown]
# evaluates
#
# ```python
#
# def f_dN(t, N0, mu_max, A, lam):
#     return (
#         4
#         * N0
#         * mu_max
#         * np.exp(A)
#         * np.exp(2 + 4 * mu_max * (lam - t) / A)
#         / (A * (np.exp(2 + 4 * mu_max * (lam - t) / A) + 1) ** 2)
#     )
#
#
# def f_d2N(t, N0, mu_max, A, lam):
#     return (
#         -16
#         * N0
#         * mu_max**2
#         * (
#             np.exp(2 * (1 + 2 * mu_max * (lam - t) / A))
#             - 2
#             * np.exp(4 + 8 * mu_max * (lam - t) / A)
#             / (np.exp(2 * (1 + 2 * mu_max * (lam - t) / A)) + 1)
#         )
#         * np.exp(A)
#         / (A**2 * (np.exp(2 * (1 + 2 * mu_max * (lam - t) / A)) + 1) ** 2)
#     )
# ```


# %% [markdown]
# Done.
