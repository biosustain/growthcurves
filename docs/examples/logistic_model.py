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
#
# \begin{gather*}
# N(t) = \frac{K}{1 + \mathrm{factor}\,\exp(-\mu (t - \mathrm{lag}))},
# \qquad
# \mathrm{factor} = \frac{K - N_0}{N_0}.
# \end{gather*}
#
# It matches the "classic" logistic shape from Wikipedia, but writes the usual
# integration constant in a biologically meaningful way through `K` and `N0`.
# > Note: `N0` is not `N(t=0)`unless `lag=0`. The shifted form is used to model the
# > lag phase.
#
# ## What the parameters do
#
# - `K` sets the upper plateau (carrying capacity).
# - `mu` sets how quickly the transition happens.
# - `lag` shifts the whole S-curve left or right.
# - `factor = (K - N0) / N0` is not an extra free parameter once `K` and `N0`
#   are chosen. It measures how much capacity is still empty compared with what is
#   already present at `t = lag`.
#
# ## What the factor changes
#
# The factor is marked directly on the plot through two special points:
#
# - At `t = lag`, the curve passes through `N(lag) = N0 = K / (1 + factor)`.
# - The inflection point is at
#   `t* = lag + ln(factor) / mu`, where `factor * exp(-mu * (t - lag)) = 1`.
#   There the curve reaches `K / 2` and the growth rate is maximal.
#
# For fixed `K` and `mu`, a larger factor means a smaller `N0 / K`, so the curve
# starts lower and the inflection happens later. The maximum slope itself stays
# the same (`mu * K / 4`); the factor mainly changes *where* the midpoint happens.
#
# One subtle point is worth watching in the app: with this shifted form, `N0` is
# the value at `t = lag`, not necessarily the value at `t = 0`. The app marks both.


# %% tags=["hide-input"]
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.subplots
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
def logistic_growth(t, N_lag, K, mu, lag):
    """Logistic growth model with smooth transition through lag phase"""
    # Standard logistic formula centered at lag time
    # This creates a smooth S-curve with inflection point at t = lag + (K - N0) / N0
    factor = (K - N_lag) / N_lag
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
    # accel[t < lag] = 0
    return accel


def logistic_derivative(t, K, N_lag, mu, lag):
    N = logistic_growth(t, N_lag, K, mu, lag)
    return mu * N * (1 - N / K)


def get_doubling_time(t, K, N0, mu, lag):
    """
    Returns the instantaneous doubling time at time t.
    """
    N = logistic_growth(t, K, N0, mu, lag)
    with np.errstate(divide="ignore"):
        doubling_time = np.log(2) / (mu * (1 - (N / K)))
    # doubling_time[t < lag] = np.nan  # Undefined during lag phase
    return doubling_time


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
    return fit_mech_logistic, stats_mech_logistic


def format_summary(K, N_lag, mu, lag, factor, N_at_zero, t_inflect):
    max_slope = mu * K / 4
    inflection_note = (
        "The inflection lies before `t = 0`, so the observed window already starts "
        "after the midpoint."
        if t_inflect < 0
        else "The inflection lies inside the plotted time window."
    )
    return f"""
### Read the curve

- `factor = (K - N_lag) / N_lag = {factor:.3f}`
- `lag = {lag:.3f}`
- `N(lag) = N_lag = {N_lag:.3f} ('N0')`
- `N(0) = {N_at_zero:.3f} (not 'N0'!)`
- `t_inflect = lag + ln(factor) / mu = {t_inflect:.3f}`
- `max dN/dt = mu * K / 4 = {max_slope:.3f}`

`factor` controls how far the midpoint sits from `lag`.
A larger factor pushes the inflection to the right and lowers the starting part of
the curve relative to `K`.

{inflection_note}
"""


def make_figure(K, N_lag, mu, lag):
    N0 = N_lag  # N0 is where t = lag
    factor = (K - N0) / N0
    t_inflect = lag + np.log(factor) / mu
    N_at_zero = logistic_growth(np.array([0.0]), N0, K, mu, lag)[0]

    t_start = min(0.0, t_inflect - 2.0 / mu)
    t_end = max(24.0, lag + 6.0 / mu, t_inflect + 6.0 / mu)
    t = np.linspace(t_start, t_end, 500)
    N = logistic_growth(t, N0, K, mu, lag)
    dNdt = logistic_derivative(t, K, N0, mu, lag)

    fig = plotly.subplots.make_subplots(
        rows=2,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.08,
        row_heights=[0.72, 0.28],
        subplot_titles=("Population size N(t)", "Growth rate dN/dt"),
    )

    fig.add_trace(
        go.Scatter(
            x=t,
            y=N,
            mode="lines",
            line={"color": "#1565c0", "width": 3},
            name="N(t)",
        ),
        row=1,
        col=1,
    )
    fig.add_trace(
        go.Scatter(
            x=t,
            y=dNdt,
            mode="lines",
            line={"color": "#c62828", "width": 3},
            name="dN/dt",
        ),
        row=2,
        col=1,
    )

    fig.add_trace(
        go.Scatter(
            x=[0.0, lag, t_inflect],
            y=[N_at_zero, N0, K / 2],
            mode="markers+text",
            text=["N(0)", "N(lag)=N0=N_lag", "Inflection"],
            textposition="top center",
            marker={"size": 10, "color": ["#455a64", "#2e7d32", "#ef6c00"]},
            name="Key points",
        ),
        row=1,
        col=1,
    )
    fig.add_trace(
        go.Scatter(
            x=[t_inflect],
            y=[mu * K / 4],
            mode="markers+text",
            text=["Peak slope"],
            textposition="top center",
            marker={"size": 10, "color": "#ef6c00"},
            name="Peak slope",
        ),
        row=2,
        col=1,
    )

    for row in (1, 2):
        fig.add_vline(
            x=lag,
            line_dash="dash",
            line_color="#2e7d32",
            annotation_text="lag",
            annotation_position="top left",
            row=row,
            col=1,
        )
        fig.add_vline(
            x=t_inflect,
            line_dash="dot",
            line_color="#ef6c00",
            annotation_text="t_inflect",
            annotation_position="top right",
            row=row,
            col=1,
        )

    fig.add_hline(
        y=K,
        line_dash="dash",
        line_color="#1565c0",
        annotation_text="K",
        annotation_position="top left",
        row=1,
        col=1,
    )
    fig.add_hline(
        y=N0,
        line_dash="dot",
        line_color="#2e7d32",
        annotation_text="N0",
        annotation_position="bottom left",
        row=1,
        col=1,
    )
    fig.add_hline(
        y=K / 2,
        line_dash="dot",
        line_color="#ef6c00",
        annotation_text="K/2",
        annotation_position="bottom left",
        row=1,
        col=1,
    )

    fig.add_annotation(
        x=lag,
        y=N0,
        xref="x",
        yref="y",
        text=(f"factor = {factor:.2f}<br>N0 = K / (1 + factor)"),
        showarrow=True,
        arrowhead=2,
        ax=120,
        ay=-70,
        bgcolor="rgba(255,255,255,0.9)",
        bordercolor="#2e7d32",
    )
    fig.add_annotation(
        x=t_inflect,
        y=K / 2,
        xref="x",
        yref="y",
        text=("factor * exp(-mu * (t - lag)) = 1<br><=> t = lag + ln(factor) / mu"),
        showarrow=True,
        arrowhead=2,
        ax=130,
        ay=20,
        bgcolor="rgba(255,255,255,0.9)",
        bordercolor="#ef6c00",
    )

    fig.update_xaxes(title_text="Time", row=2, col=1)
    fig.update_yaxes(title_text="N(t)", row=1, col=1)
    fig.update_yaxes(title_text="dN/dt", row=2, col=1)
    fig.update_layout(
        height=720,
        template="plotly_white",
        showlegend=False,
        title=(
            "Shifted logistic curve with explicit factor "
            f"(N0/K = {N0/K:.3f}, factor = {factor:.3f})"
        ),
        margin={"l": 60, "r": 30, "t": 90, "b": 60},
    )

    return fig, format_summary(K, N0, mu, lag, factor, N_at_zero, t_inflect)


# %% [markdown]
# # 1. Set your simulation parameters

# %% tags=["parameters"]
mu_max = 0.3  # Growth rate constant
K = 5  # Carrying capacity for logistic growth
N0 = 0.3  # condition at lag
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
    "N0": N0,  # N_lag
    "A": A,
    "lag": lag,
}

fig, summary = make_figure(K, N0, mu, lag)
print(summary)
fig

# %% [markdown]
# # 2. Create the time grid where you want data points

# %%
t_eval = np.linspace(t_start, t_end, num_points)


# %% [markdown]
# # 3. Solve the ODE for classic logistic growth
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
# filling in the initial values with N0. N0 is the value at t = lag, which can be
# remodeled using the classic logistic growth formula. The lag phase is not explicitly
# modeled in the ODE.
#
# > Shift the solution to the right by the lag time (in continuous time, not by a
# > fixed number of grid points, so it lines up exactly with the closed-form
# > comparison below regardless of the time grid spacing).
#

# %%
post_lag = t_eval >= lag
idx_post_lag = np.where(post_lag)[0][0]
# N = np.full_like(t_eval, N0)
N = mech_logistic_model(t_eval, mu, K, N0)
N[idx_post_lag:] = N[:-idx_post_lag]  # Shift the solution to the right by lag time
N[:idx_post_lag] = N0  # Fill in the initial values with N0
_ = plt.plot(t_eval, N)
_ = plt.title("Mechanistic Logistic Growth Simulation (ODE)")

# %% tags=["hide-input"]
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
# # 4. Generate the classic phenomenological model (closed form)
#
# ```
# N(t) = K / (1 + ((K - N0)/N0) * exp(-μ * (t - lag)))
# ```
#
# One subtle but important point:
# If you use
# factor = (K - N0)/N0
# N(t) = K / (1 + factor * exp(-μ * (t - lag)))
# then N0 is N(lag). This is not equal to N(0) unless lag = 0. This way our curve will
# overlap to the mechanistic model from after the lag phase.
#
# > N(0) is here not N0!

# %% tags=["hide-input"]
data["OD_phenom_classic"] = logistic_growth(
    t=data["Time"],
    N_lag=N0,
    K=K,
    mu=mu,  # the ODE rate constant, not mu_max — see 1. Set your simulation parameters
    lag=lag,
)
data["OD_phenom_classic_1der"] = logistic_derivative(
    t=data["Time"], K=K, N_lag=N0, mu=mu, lag=lag
)
data["OD_phenom_classic_2der"] = get_acceleration(
    t=data["Time"], K=K, N0=N0, mu=mu, lag=lag
)
data["OD_phenom_classic_doubling_time"] = get_doubling_time(
    t=data["Time"], K=K, N0=N0, mu=mu, lag=lag
)
# N(0) is the initial condition at t=0
data["OD_phenom_classic_ln"] = np.log(
    data["OD_phenom_classic"] / N0  #  data["OD_phenom_classic"].min()
)

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
    alpha=0.5,
)
_ = ax.legend()

# %% [markdown]
# If we check the derived quantities, we see that the maximum observed growth rate will
# be by construction at t=lag

# %% tags=["hide-input"]
_ = (
    data.set_index("Time")
    .filter(like="OD_phenom_classic")
    .plot(
        subplots=True,
        layout=(3, 2),
        figsize=(7, 6),
        sharex=True,
        style=".",
        markersize=1,
        ylim=(-5, 8),
    )
)
pd.concat(
    [
        data.set_index("Time").filter(like="OD_phenom_classic").idxmax(),
        data.set_index("Time").filter(like="OD_phenom_classic").max(),
    ],
    axis=1,
    keys=["Time", "maximum"],
)

# %%
pd.concat(
    [
        data.set_index("Time").filter(like="OD_phenom_classic").idxmin(),
        data.set_index("Time").filter(like="OD_phenom_classic").min(),
    ],
    axis=1,
    keys=["Time", "minimum"],
)

# %% [markdown]
# # Recover parameters using mechanistic logistic model
# - mechanistic model do not fit a lag phase, so we need to start fitting after the lag
#   phase where N(t=0) will now indeed be N(t_lag) = N(0) = N0. The mechanistic model
#  will then be able to recover the growth rate and carrying capacity K.


# %%
model = "mech_logistic"
# mask_timepoints_after_lag = data["Time"] > lag
data_mech = data.query(f"Time >= {lag - 0.0001}")

data_mech["Time"] = data_mech["Time"] - lag  # Shift time to start at lag
data_mech[["Time", "OD_mech", "OD_phenom_classic"]]


# %% [markdown]
# Fit to `OD_mech`

# %% tags=["hide-input"]
fit_mech_logistic, stats_mech_logistic = fit_model_and_extract_stats(
    data_mech["Time"], data_mech["OD_mech"], model
)
# Combine fits into a  DataFrame for display
pd.concat(
    [
        pd.Series(ground_truth_params),
        pd.Series(fit_mech_logistic),
        pd.Series(stats_mech_logistic),
    ],
    axis=1,
    keys=["Ground Truth", "Fit", "Stats"],
)

# %%
fit_mech_logistic, stats_mech_logistic = fit_model_and_extract_stats(
    data_mech["Time"], data_mech["OD_phenom_classic"], model
)
# Combine fits into a  DataFrame for display
pd.concat(
    [
        pd.Series(ground_truth_params),
        pd.Series(fit_mech_logistic),
        pd.Series(stats_mech_logistic),
    ],
    axis=1,
    keys=["Ground Truth", "Fit", "Stats"],
)

# %% [markdown]
# # 6. Generate the phenomenological model for comparison (paper version)
#
# As in review paper we have a slightly modified logistical model:
#
# ```
# A = K / N0 # Carrying capacity in log-space
# ln(Nt/N0) =          A / (1 + exp((4 * μ_max / A) * (λ - t) + 2))
# Nt        = N0 * exp(A / (1 + exp((4 * μ_max / A) * (λ - t) + 2)))
# ```
#
# > Note that you can model the lag phase with a time shift in this formulation, which
# > had to be manually added using the mechanistic model upon data generation.
# > Buy contrast N0 is not modeled using the phenomological model(s) operating in log
# > space, so the initial condition has to be inferred from the data.
#
# We see that the models are not the same. Both are S-curve shaped.


# %% tags=["hide-input"]
N_0 = 0.06  # one decimal of from classic logistic model
A = np.log(K / N_0)

ground_truth_params_phenom_paper = {
    "mu_max": mu_max,
    "A": A,
    "lam": lag - 5,
    "N0": N_0,
}
data["OD_phenom_paper_ln"] = phenom_logistic_model_ln(
    t=data["Time"],
    mu_max=ground_truth_params_phenom_paper["mu_max"],
    A=A,
    lam=ground_truth_params_phenom_paper["lam"],
)
# data["OD_phenom_paper"] = np.exp(data["OD_phenom_paper_ln"]) * N_0
data["OD_phenom_paper"] = log_to_linear(data["OD_phenom_paper_ln"], N_0)
ax = data.plot.scatter(
    x="Time",
    y="OD_phenom_classic",
    s=1,
    alpha=0.5,
    color="C0",
    title="Logistic Growth Simulation (classic vs paper version)",
    label="Classic Logistic Growth",
    xlabel="Time (hours)",
    ylabel="OD",
)
_ = data.plot.scatter(
    x="Time",
    y="OD_phenom_paper",
    s=1,
    alpha=0.5,
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

# %%
# print(f"N of first timepoint after lag: {data.loc[148, 'OD_phenom_paper']:.5f}")
# data["OD_phenom_paper"].describe()

# %% [markdown]
# We see that the model with different lag-time and similar N0 look similar in linear
# space. Let's compare these in log-space:

# %%
data["OD_phenom_classic_ln"] = np.log(
    data["OD_phenom_classic"] / N_0  # data["OD_phenom_classic"].min()
)

ax = data.plot.scatter(
    x="Time",
    y="OD_phenom_classic_ln",
    s=1,
    color="C0",
    title="Logistic Growth in log-space (classic vs paper version)",
    label="Classic Logistic Growth",
    xlabel="Time (hours)",
    ylabel="OD",
)
_ = data.plot.scatter(
    x="Time",
    y="OD_phenom_paper_ln",
    s=1,
    color="C1",
    ax=ax,
    label="Phenomenological Logistic Growth",
)

# %% [markdown]
# Fit synthetic data from classic logistic regression model (`OD_phenom_classic`)
# with the phenomenological
# model (paper version). The assumption is here that the initial condition is observed
# at t=0. this is different from N0 = N(lag) in the classic logistic model formulation.
# - find a good fit for the phenomenological model
# - compare the parameters
# - plot the fit

# %% tags=["hide-input"]
model = "phenom_logistic"
col = "OD_phenom_classic"
fit_, stats_ = fit_model_and_extract_stats(data["Time"], data[col], model)
pd.concat(
    [
        pd.Series(ground_truth_params),
        pd.Series(ground_truth_params_phenom_paper),
        pd.Series(fit_),
        pd.Series(stats_),
    ],
    axis=1,
    keys=["Ground Truth (Classic)", "Ground Truth (Phenom Paper)", "Fit", "Stats"],
)


# %% [markdown]
# We see that the phenomenological logistic model and the classic logistic model
# are similar in shape after fitting, but disagree in the `lag` phase parameter
# estimated.

# %% tags=["hide-input"]
data["OD_phenom_classic_fit"] = log_to_linear(
    phenom_logistic_model_ln(
        t=data["Time"],
        mu_max=fit_["mu_max"],
        A=fit_["A"],
        lam=fit_["lam"],
    ),
    fit_["N0"],
)
ax = data.plot.scatter(
    x="Time",
    y="OD_phenom_classic",
    s=1,
    color="C0",
    alpha=0.5,
    title="Logistic Growth Simulation (classic vs paper version)",
    label="Classic Logistic Growth",
    xlabel="Time (hours)",
    ylabel="OD",
)
_ = data.plot.scatter(
    x="Time",
    y="OD_phenom_paper",
    s=1,
    alpha=0.5,
    color="C1",
    ax=ax,
    label="Phenomenological Logistic Growth",
)
_ = data.plot.scatter(
    x="Time",
    y="OD_phenom_classic_fit",
    s=1,
    alpha=0.5,
    color="C2",
    ax=ax,
    label="Phenomenological Logistic Growth (Fit)",
)
ax.vlines(
    x=lag,
    ymin=N0,
    ymax=K,
    color="red",
    linestyle="--",
    label="Lag time ends (phenom classic)",
)
ax.vlines(
    x=fit_["lam"],
    ymin=N0,
    ymax=K,
    color="red",
    linestyle="-.",
    label="Lag time ends (phenom fit)",
)
_ = ax.legend()

# %% [markdown]
# If we use instead the phenomological model to generate synthetic data on the linear
# scale with N(t=0) = 0.06 (without lag phase) we can get back the exact parameters.
#
# - not that N0 is added to the equation using the log_to_linear function as the
#   `phenom_logistic_model_ln` function returns the log of the ratio of N(t)/N0 with an
# .  estimated offset as the log of the ratio N(t)/N0 itself never zero.
#
# > The fit masks the original data (as expected)

# %%
model = "phenom_logistic"
col = "OD_phenom_paper"

fit_, stats_ = fit_model_and_extract_stats(data["Time"], data["OD_phenom_paper"], model)
display(pd.concat(
    [
        pd.Series(ground_truth_params_phenom_paper),
        pd.Series(fit_),
        pd.Series(stats_),
    ],
    axis=1,
    keys=["Ground Truth", "Fit", "Stats"],
))

data["OD_phenom_paper_fit"] = log_to_linear(
    phenom_logistic_model_ln(
        t=data["Time"],
        mu_max=fit_["mu_max"],
        A=fit_["A"],
        lam=fit_["lam"],
    ),
    fit_["N0"],
)
ax = data.plot.scatter(
    x="Time",
    y="OD_phenom_paper_fit",
    s=1,
    alpha=0.8,
    color="C1",
    label="Phenomenological Logistic Growth (Original)",
)
ax = data.plot.scatter(
    x="Time",
    y="OD_phenom_paper",
    s=1,
    alpha=0.8,
    color="C2",
    label="Phenomenological Logistic Growth (Fit)",
    ax=ax
)

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
