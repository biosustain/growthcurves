"""Based on the grpredict package.

The functionality is also included in piogrowth as of now (see update method of class):
https://github.com/Pioreactor/pioreactor/blob/bf30d09646a8e38b79c81507f78afb75b8a22497/core/pioreactor/utils/streaming_calculations.py

Articles describing the Kalman filter approach to growth rate estimation:
- good for continous updates. Need to check how scalable the approach is to long
  time series.
- blog article highlights the three states that are modeled: log-level measurement,
  growth rate, and growth rate drift.
- preprocessing aligns the intial values to one, i.e. zero in non-log space
  (we need to check how we do this elsewhere in the methods)
https://pioreactor.com/en-dk/blogs/pioreactor/estimating-growth-rates-with-kalman-filters
"""

import grpredict
import numpy as np

MINIMUM_OBSERVATION_VALUE = 1e-9


def run_ekf(
    observations: np.ndarray,
    dt_hours: float,
    *,
    warmup_length: int = 50,
    outlier_std_threshold: float = 5.0,
    min_growth_rate: float = -1.0,
    max_growth_rate: float = 3.0,
    recent_dilution: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Run the EKF over an entire observation array in one call.

    The first `warmup_length` observations are used to estimate noise
    parameters and a normalization factor; the filter is then run from
    timestep 0 through the end of the array.

    Parameters
    ----------
    observations
        OD timeseries with shape ``(n_time,)`` for a single sensor or
        ``(n_time, n_sensors)`` for multiple sensors.
    dt_hours
        Constant timestep between consecutive observations, in hours.
        For example, 5-second sampling is ``5 / 3600``.
    warmup_length
        Number of initial observations used only to initialise filter
        parameters (noise, normalization).  Must be >= 2 and < n_time.
    outlier_std_threshold
        Standardised-innovation threshold above which a measurement is
        treated as an outlier and its variance is inflated.
    min_growth_rate, max_growth_rate
        Hard clipping bounds applied to the growth-rate state after each
        update, in units of 1/hour.
    recent_dilution
        Boolean array of length ``n_time``.  Set ``True`` at timesteps
        immediately following a dilution event (turbidostat / chemostat)
        to temporarily inflate process noise.  ``None`` means no dilutions.

    Returns
    -------
    states : ndarray, shape (n_time, 3)
        Estimated ``[log_od, growth_rate, growth_rate_drift]`` at every
        timestep.  ``np.exp(states[:, 0])`` recovers estimated OD;
        ``states[:, 1]`` is the growth rate in 1/hour.
    covariances : ndarray, shape (n_time, 3, 3)
        State covariance matrix at every timestep.
    """
    import numpy as np

    observation_matrix, _ = grpredict._as_positive_observation_matrix(observations)
    n_time = observation_matrix.shape[0]

    if warmup_length < 2:
        raise ValueError("warmup_length must be at least 2")
    if warmup_length >= n_time:
        raise ValueError(
            "warmup_length must be less than the total number of observations"
        )

    warmup_obs = observation_matrix[:warmup_length]
    summary = grpredict.summarize_warmup_observations(warmup_obs, dt_hours)
    ekf = grpredict.build_filter_from_observation_summary(
        summary,
        outlier_std_threshold=outlier_std_threshold,
        min_growth_rate=min_growth_rate,
        max_growth_rate=max_growth_rate,
    )

    norm_obs = grpredict.normalize_observations_by_factor(
        observation_matrix, summary.normalization_factors
    )

    if recent_dilution is None:
        dilution_flags = np.zeros(n_time, dtype=bool)
    else:
        dilution_flags = np.asarray(recent_dilution, dtype=bool)
        if dilution_flags.shape[0] != n_time:
            raise ValueError(
                "recent_dilution must have the same length as observations"
            )

    states = np.empty((n_time, 3), dtype=float)
    covariances = np.empty((n_time, 3, 3), dtype=float)

    for t in range(n_time):
        state, cov = ekf.update(
            norm_obs[t], dt_hours, recent_dilution=bool(dilution_flags[t])
        )
        states[t] = state
        covariances[t] = cov

    return states, covariances


if __name__ == "__main__":
    from matplotlib import pyplot as plt

    rng = np.random.default_rng(42)
    dt = 5 / 3600
    T = 1000
    time = np.arange(T) * dt
    true_gr = 0.9
    od = np.array([0.5 * np.exp(true_gr * t) + 0.01 * rng.normal() for t in time])

    states_filter, _ = run_ekf(od, dt)
    print(
        "run_ekf   growth rate (mean of last 20):",
        round(states_filter[-20:, 1].mean(), 4),
    )
    fig, ax = plt.subplots(figsize=(8, 4), nrows=2, sharex=True)
    ax[0].plot(time, states_filter[:, 1], label="Filter growth rate")
    ax[0].set_xlabel("Time in hours.")
    ax[0].set_ylabel("Growth rate (1/hour)")
    ax[0].axhline(true_gr, color="gray", linestyle="--", label="True growth rate")
    ax[0].legend()
    ax[1].plot(time, np.exp(states_filter[:, 0]), label="Filter OD estimate")
    ax[1].plot(
        time, od, label="Noisy observations", alpha=0.7, marker="o", linestyle=""
    )
    ax[1].set_xlabel("Time in hours.")
    ax[1].set_ylabel("OD")
    ax[1].legend()
    plt.show()
