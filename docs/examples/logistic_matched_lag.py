# %% [markdown]
# # Matching the mechanistic and classic logistic curves with a lag time
#
# Goal: produce a single **smooth** logistic growth curve that
#
# - starts from a specified initial OD `N0` in **linear** space, and
# - is shifted to the right by a lag time `lag = 12.3`,
#
# and show that `OD_phenom_classic` (closed-form logistic) and `OD_mech` (solved
# from the mechanistic ODE) are the *same curve* — no flat lag-phase plateau, just
# one continuous S-curve translated along the time axis.
#
# ## The key idea
#
# The lag is applied as a pure **horizontal time shift** of the logistic sigmoid:
#
# ```
# N(t) = K / (1 + ((K - N0) / N0) * exp(-mu * (t - lag)))
# ```
#
# With this parameterization `N(lag) = N0` — i.e. `N0` is the OD the curve passes
# through *at the end of the lag*, not at `t = 0`. For `t < lag` the smooth curve
# sits just below `N0` (it does not clamp to a flat line).
#
# The mechanistic solver `mech_logistic_model` instead fixes the initial condition
# at `t = 0`. To reproduce the *identical* shifted curve we back-extrapolate the
# initial OD to `t = 0`:
#
# ```
# N0_eff = N(0) = K / (1 + ((K - N0) / N0) * exp(mu * lag))
# ```
#
# Integrating the ODE from `t = 0` with `N0_eff` then yields exactly the classic
# curve above (up to ODE solver tolerance).
#
# > This example is not identical to the logistic phenomenological model from the review.

# %% tags=["hide-input"]
import numpy as np
import pandas as pd

from growthcurves.models import mech_logistic_model

# %% [markdown]
# ## 1. Parameters

# %% tags=["parameters"]
mu_max = 0.3  # maximum specific growth rate (log-scale)
K = 5.0  # carrying capacity (maximum OD) in linear space
N0 = 0.3  # OD at the end of the lag phase (t = lag), in linear space
lag = 12.3  # lag time (hours)
t_start = 0.0
t_end = 60.0
num_points = int(t_end * 12)

# ODE rate constant. mu_max is the *specific* growth rate at the start of growth
# (t = lag, N = N0); the ODE rate constant mu is larger by 1 / (1 - N0/K).
mu = mu_max / (1 - N0 / K)
factor = (K - N0) / N0
print(f"mu_max: {mu_max}, mu: {mu}, K: {K}, N0: {N0}, lag: {lag}")

# %% [markdown]
# ## 2. Time grid

# %%
t = np.linspace(t_start, t_end, num_points)

# %% [markdown]
# ## 3. Classic phenomenological curve (closed form)
#
# Smooth logistic sigmoid shifted by `lag`; passes through `N0` at `t = lag`.

# %%
OD_phenom_classic = K / (1 + factor * np.exp(-mu * (t - lag)))

# %% [markdown]
# ## 4. Mechanistic curve (ODE solver)
#
# Back-extrapolate the initial OD to `t = 0` so the solved curve is the same
# sigmoid, only expressed through its `t = 0` initial condition.

# %%
N0_eff = K / (1 + factor * np.exp(mu * lag))
print(f"Back-extrapolated initial OD at t=0: N0_eff = {N0_eff:.6f}")

OD_mech = mech_logistic_model(t, mu=mu, K=K, N0=N0_eff)

# %% [markdown]
# ## 5. They match exactly
#
# The two curves are analytically identical; the tiny residual is only the RK45
# integration tolerance.

# %%
data = pd.DataFrame(
    {
        "Time": t,
        "OD_phenom_classic": OD_phenom_classic,
        "OD_mech": OD_mech,
    }
)

max_abs_diff = np.max(np.abs(data["OD_mech"] - data["OD_phenom_classic"]))
od_at_lag = np.interp(lag, t, OD_phenom_classic)
print(f"max |OD_mech - OD_phenom_classic| = {max_abs_diff:.5f}")
print(f"OD at t=lag = {od_at_lag:.6f} (should equal N0 = {N0})")

# %% tags=["hide-input"]
ax = data.plot(
    x="Time",
    y="OD_phenom_classic",
    color="C1",
    linewidth=3,
    alpha=0.6,
    label="OD_phenom_classic (closed form)",
    xlabel="Time (hours)",
    ylabel="OD",
    title="Smoothed logistic growth, shifted by a lag time",
)
data.plot(
    x="Time",
    y="OD_mech",
    ax=ax,
    color="C0",
    linestyle="--",
    label="OD_mech (ODE solver)",
)
ax.axvline(lag, color="red", linestyle=":", label=f"lag = {lag}")
ax.axhline(N0, color="grey", linestyle=":", alpha=0.6, label=f"N0 = {N0}")
_ = ax.legend()

# %% [markdown]
# Identity plot: every point lies on `y = x`.

# %% tags=["hide-input"]
ax = data.plot.scatter(
    x="OD_mech",
    y="OD_phenom_classic",
    s=3,
    color="C1",
    title="OD_mech vs OD_phenom_classic",
)
_ = ax.plot([0, K], [0, K], color="black", linestyle="--", alpha=0.5, label="y = x")
_ = ax.legend()

# %% [markdown]
# Done — a single smooth curve, starting from `N0` in linear space and shifted by
# the lag time, reproduced identically by both the closed-form (phenomenological)
# and the ODE-solved (mechanistic) logistic model.
