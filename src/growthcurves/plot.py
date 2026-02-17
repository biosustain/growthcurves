"""
Plotting functions for growth curve analysis using Plotly.

This module provides modular functions for creating and annotating growth curve plots.
"""

from typing import Any, Dict, Optional, Tuple

import numpy as np
import plotly.graph_objects as go

from .models import (
    evaluate_parametric_model,
    get_all_parametric_models,
    spline_from_params,
)


def create_base_plot(
    t: np.ndarray,
    N: np.ndarray,
    scale: str = "linear",
    xlabel: str = "Time (hours)",
    ylabel: Optional[str] = None,
    marker_size: int = 5,
    marker_opacity: float = 0.3,
    marker_color: str = "gray",
) -> go.Figure:
    """
    Create a base plot with raw N points.

    Parameters
    ----------
    t : numpy.ndarray
        Time points
    N : numpy.ndarray
        OD measurements or other growth N
    scale : str, optional
        'linear' or 'log' scale for y-axis (default: 'linear')
    xlabel : str, optional
        X-axis label (default: 'Time (hours)')
    ylabel : str, optional
        Y-axis label. If None, automatically set based on scale
    marker_size : int, optional
        Size of N point markers (default: 5)
    marker_opacity : float, optional
        Opacity of N point markers (default: 0.3)
    marker_color : str, optional
        Color of N point markers (default: 'gray')

    Returns
    -------
    plotly.graph_objects.Figure
        Plotly figure object with raw N
    """
    # Convert to numpy arrays
    t = np.asarray(t, dtype=float)
    N = np.asarray(N, dtype=float)

    # Filter out non-positive and non-finite values for valid plotting
    mask = np.isfinite(t) & np.isfinite(N) & (N > 0)
    t = t[mask]
    N = N[mask]

    # Determine y-axis N based on scale
    if scale == "log":
        y_data = np.log(N)
        if ylabel is None:
            ylabel = "ln(OD)"
    else:
        y_data = N
        if ylabel is None:
            ylabel = "OD"

    # Create figure
    fig = go.Figure()

    # Add raw N trace
    fig.add_trace(
        go.Scatter(
            x=t,
            y=y_data,
            mode="markers",
            name="Data",
            marker=dict(size=marker_size, opacity=marker_opacity, color=marker_color),
            showlegend=False,
        )
    )

    # Update layout
    # For linear scale, set y-axis to start at 0; for log scale, auto-range
    yaxis_config = dict(visible=True, showline=True)
    if scale == "linear":
        yaxis_config["range"] = [0, None]

    fig.update_layout(
        xaxis_title=xlabel,
        yaxis_title=ylabel,
        hovermode="closest",
        template="plotly_white",
        showlegend=False,
        xaxis=dict(range=[0, None]),  # Start x-axis at 0 to remove gap
        yaxis=yaxis_config,
    )

    return fig


def add_exponential_phase(
    fig: go.Figure,
    exp_start: float,
    exp_end: float,
    color: str = "lightgreen",
    opacity: float = 0.25,
    name: str = "Exponential phase",
    row: Optional[int] = None,
    col: Optional[int] = None,
) -> go.Figure:
    """
    Add shaded region for exponential growth phase.

    Parameters
    ----------
    fig : plotly.graph_objects.Figure
        Plotly figure to annotate
    exp_start : float
        Start time of exponential phase
    exp_end : float
        End time of exponential phase
    color : str, optional
        Color for shaded region (default: 'lightgreen')
    opacity : float, optional
        Opacity of shaded region (default: 0.25)
    name : str, optional
        Legend name (default: 'Exponential phase')
    row : int, optional
        Subplot row (for subplots)
    col : int, optional
        Subplot column (for subplots)

    Returns
    -------
    plotly.graph_objects.Figure
        Updated figure with exponential phase shading
    """
    if exp_start is None or exp_end is None:
        return fig

    if not np.isfinite(exp_start) or not np.isfinite(exp_end):
        return fig

    # Add vertical rectangle for exponential phase
    fig.add_vrect(
        x0=exp_start,
        x1=exp_end,
        fillcolor=color,
        opacity=opacity,
        layer="below",
        line_width=0,
        row=row,
        col=col,
    )

    return fig


def add_fitted_curve(
    fig: go.Figure,
    time_fit: np.ndarray,
    N_fit: np.ndarray,
    name: str = "Fitted curve",
    color: str = "blue",
    line_width: int = 5,
    window_start: Optional[float] = None,
    window_end: Optional[float] = None,
    scale: Optional[str] = "linear",
    row: Optional[int] = None,
    col: Optional[int] = None,
) -> go.Figure:
    """
    Add fitted curve to the plot, optionally constrained to a window.

    Parameters
    ----------
    fig : plotly.graph_objects.Figure
        Plotly figure to annotate
    time_fit : numpy.ndarray
        Time points for fitted curve
    N_fit : numpy.ndarray
        Fitted y values
    name : str, optional
        Legend name for fitted curve (default: 'Fitted curve')
    color : str, optional
        Color of fitted curve (default: 'blue')
    line_width : int, optional
        Width of fitted curve line (default: 5)
    window_start : float, optional
        Start of fitting window (if specified, only show curve in this range)
    window_end : float, optional
        End of fitting window (if specified, only show curve in this range)
    scale : str, optional
        'linear' or 'log' - determines y-axis transformation (default: 'linear')
    row : int, optional
        Subplot row (for subplots)
    col : int, optional
        Subplot column (for subplots)

    Returns
    -------
    plotly.graph_objects.Figure
        Updated figure with fitted curve
    """
    if time_fit is None or N_fit is None:
        return fig

    # Convert to numpy arrays
    time_fit = np.asarray(time_fit, dtype=float)
    N_fit = np.asarray(N_fit, dtype=float)

    # Filter to window if specified
    if window_start is not None and window_end is not None:
        mask = (time_fit >= window_start) & (time_fit <= window_end)
        time_fit = time_fit[mask]
        N_fit = N_fit[mask]

    # Transform y-values based on scale
    if scale == "log":
        N_fit = np.log(N_fit)

    # Add fitted curve
    fig.add_trace(
        go.Scatter(
            x=time_fit,
            y=N_fit,
            mode="lines",
            name=name,
            line=dict(color=color, width=line_width),
            showlegend=False,
        ),
        row=row,
        col=col,
    )

    return fig


def add_od_max_line(
    fig: go.Figure,
    od_max: float,
    scale: str = "linear",
    line_color: str = "black",
    line_dash: str = "dot",
    line_width: float = 2,
    line_opacity: float = 0.5,
    name: str = "ODmax",
    row: Optional[int] = None,
    col: Optional[int] = None,
) -> go.Figure:
    """
    Add horizontal line at maximum OD value.

    Parameters
    ----------
    fig : plotly.graph_objects.Figure
        Plotly figure to annotate
    od_max : float
        Maximum OD value
    scale : str, optional
        'linear' or 'log' - determines y-axis transformation (default: 'linear')
    line_color : str, optional
        Color of horizontal line (default: 'red')
    line_dash : str, optional
        Dash style for horizontal line (default: 'dash')
    line_width : float, optional
        Width of horizontal line (default: 1)
    line_opacity : float, optional
        Opacity of horizontal line (default: 0.5)
    name : str, optional
        Legend name (default: 'ODmax')
    row : int, optional
        Subplot row (for subplots)
    col : int, optional
        Subplot column (for subplots)

    Returns
    -------
    plotly.graph_objects.Figure
        Updated figure with od_max horizontal line
    """
    if od_max is None:
        return fig

    if not np.isfinite(od_max):
        return fig

    # Transform y-value based on scale
    y_val = np.log(od_max) if scale == "log" else od_max

    # Add horizontal line at od_max
    fig.add_hline(
        y=y_val,
        line_color=line_color,
        line_dash=line_dash,
        line_width=line_width,
        opacity=line_opacity,
        row=row,
        col=col,
    )

    return fig


def add_N0_line(
    fig: go.Figure,
    N0: float,
    scale: str = "linear",
    line_color: str = "gray",
    line_dash: str = "dot",
    line_width: float = 2,
    line_opacity: float = 0.5,
    name: str = "N0",
    row: Optional[int] = None,
    col: Optional[int] = None,
) -> go.Figure:
    """
    Add horizontal line at initial OD value (N0).

    Parameters
    ----------
    fig : gplotly.graph_objectso.Figure
        Plotly figure to annotate
    N0 : float
        Initial OD value
    scale : str, optional
        'linear' or 'log' - determines y-axis transformation (default: 'linear')
    line_color : str, optional
        Color of horizontal line (default: 'gray')
    line_dash : str, optional
        Dash style for horizontal line (default: 'dot')
    line_width : float, optional
        Width of horizontal line (default: 2)
    line_opacity : float, optional
        Opacity of horizontal line (default: 0.5)
    name : str, optional
        Legend name (default: 'N0')
    row : int, optional
        Subplot row (for subplots)
    col : int, optional
        Subplot column (for subplots)

    Returns
    -------
    plotly.graph_objects.Figure
        Updated figure with N0 horizontal line
    """
    if N0 is None:
        return fig

    if not np.isfinite(N0):
        return fig

    # Transform y-value based on scale
    y_val = np.log(N0) if scale == "log" else N0

    # Add horizontal line at N0
    fig.add_hline(
        y=y_val,
        line_color=line_color,
        line_dash=line_dash,
        line_width=line_width,
        opacity=line_opacity,
        row=row,
        col=col,
    )

    return fig


def prepare_fitted_curve(
    fitted_model: Dict[str, Any], n_points: int = 200
) -> Optional[Tuple[np.ndarray, np.ndarray]]:
    """
    Convert a fitted model dictionary to curve data for plotting.

    Parameters
    ----------
    fitted_model : dict
        Fit result dictionary from fit_parametric() or fit_non_parametric()
    n_points : int, optional
        Number of points to generate for the curve (default: 200)

    Returns
    -------
    tuple of (np.ndarray, np.ndarray) or None
        (time_points, od_values) ready for plotting, or None if invalid
    """
    if fitted_model is None:
        return None

    model_type = fitted_model.get("model_type")
    params = fitted_model.get("params")

    if model_type is None or params is None:
        return None

    # Extract window boundaries
    if "fit_t_min" in params and "fit_t_max" in params:
        window_start = params["fit_t_min"]
        window_end = params["fit_t_max"]
    else:
        window_start = fitted_model.get("window_start")
        window_end = fitted_model.get("window_end")

    if window_start is None or window_end is None:
        return None

    # Generate time points
    time_fit = np.linspace(window_start, window_end, n_points)

    # Evaluate model
    if model_type in get_all_parametric_models():
        od_fit = evaluate_parametric_model(time_fit, model_type, params)
    elif model_type == "spline":
        spline = spline_from_params(params)
        od_fit = np.exp(spline(time_fit))
    elif model_type == "sliding_window":
        slope = params["slope"]
        intercept = params["intercept"]
        od_fit = np.exp(slope * time_fit + intercept)
    else:
        return None

    return (time_fit, od_fit)


def prepare_tangent_line(
    umax: float,
    time_umax: float,
    od_umax: float,
    fig: go.Figure,
    scale: Optional[str] = "linear",
    n_points: Optional[int] = 100,
) -> Optional[Tuple[np.ndarray, np.ndarray]]:
    """
    Calculate tangent line at maximum growth rate point.

    Parameters
    ----------
    umax : float
        Maximum growth rate (μ_max)
    time_umax : float
        Time at maximum growth rate
    od_umax : float
        OD value at maximum growth rate
    fig : plotly.graph_objects.Figure
        Figure to extract data range from
    scale : str, optional
        'linear' or 'log' for determining data range (default: 'linear')
    n_points : int, optional
        Number of points to generate for tangent line (default: 100)

    Returns
    -------
    tuple of (numpy.ndarray, numpy.ndarray) or None
        (time_points, od_values) for tangent line, or None if invalid
    """
    if not np.isfinite(umax) or not np.isfinite(time_umax) or not np.isfinite(od_umax):
        return None

    # Extract y-values from figure to determine baseline and plateau OD
    all_y_values = []
    for trace in fig.data:
        if trace.y is not None and len(trace.y) > 0:
            valid_y = [y for y in trace.y if y is not None and np.isfinite(y)]
            if valid_y:
                all_y_values.extend(valid_y)

    if len(all_y_values) == 0:
        return None

    if scale == "log":
        baseline_od = np.exp(min(all_y_values))
        plateau_od = np.exp(max(all_y_values))
    else:
        baseline_od = min(all_y_values)
        plateau_od = max(all_y_values)

    # Ensure baseline < od_umax < plateau (with safety margins)
    baseline_od = min(baseline_od, od_umax * 0.95)
    plateau_od = max(plateau_od, od_umax * 1.05)

    if baseline_od <= 0 or plateau_od <= 0 or od_umax <= 0:
        return None

    # Calculate tangent intersections
    # Tangent equation: OD(t) = od_umax * exp(umax * (t - time_umax))
    t_start = time_umax + np.log(baseline_od / od_umax) / umax
    t_end = time_umax + np.log(plateau_od / od_umax) / umax

    # Generate tangent line points
    t_tangent = np.linspace(t_start, t_end, n_points)
    od_tangent = od_umax * np.exp(umax * (t_tangent - time_umax))

    return (t_tangent, od_tangent)


def annotate_plot(
    fig: go.Figure,
    fit_result: Optional[Dict[str, Any]] = None,
    stats: Optional[Dict[str, Any]] = None,
    show_fitted_curve: bool = True,
    show_phase_boundaries: bool = True,
    show_crosshairs: bool = True,
    show_od_max_line: bool = True,
    show_n0_line: bool = True,
    show_umax_marker: bool = True,
    show_tangent: bool = True,
    scale: str = "linear",
    fitted_curve_color: str = "#8dcde0",
    fitted_curve_width: int = 5,
    row: Optional[int] = None,
    col: Optional[int] = None,
) -> go.Figure:
    """
    Add annotations to a growth curve plot.

    Parameters
    ----------
    fig : plotly.graph_objects.Figure
        Plotly figure to annotate
    fit_result : dict, optional
        Fit result dictionary from fit_parametric() or fit_non_parametric()
    stats : dict, optional
        Statistics dictionary from extract_stats()
    show_fitted_curve : bool, optional
        Whether to show the fitted curve (default: True)
    show_phase_boundaries : bool, optional
        Whether to show exponential phase boundaries (default: True)
    show_crosshairs : bool, optional
        Whether to show crosshairs to umax point (default: True)
    show_od_max_line : bool, optional
        Whether to show horizontal line at maximum OD (default: True)
    show_n0_line : bool, optional
        Whether to show horizontal line at initial OD (default: True)
    show_umax_marker : bool, optional
        Whether to show marker at umax point (default: True)
    show_tangent : bool, optional
        Whether to show tangent line at umax (default: True)
    scale : str, optional
        'linear' or 'log' for y-axis scale (default: 'linear')
    fitted_curve_color : str, optional
        Color of the fitted model curve (default: '#8dcde0')
    fitted_curve_width : int, optional
        Line width of the fitted model curve (default: 5)
    row : int, optional
        Subplot row for subplots
    col : int, optional
        Subplot column for subplots

    Returns
    -------
    plotly.graph_objects.Figure
        Updated figure with annotations
    """
    # Add fitted curve
    if show_fitted_curve and fit_result is not None:
        fitted_curve = prepare_fitted_curve(fit_result)
        if fitted_curve is not None:
            time_fit, od_fit = fitted_curve
            fig = add_fitted_curve(
                fig,
                time_fit,
                od_fit,
                name="Fitted curve",
                color=fitted_curve_color,
                line_width=fitted_curve_width,
                scale=scale,
                row=row,
                col=col,
            )

    # Add exponential phase shading
    if show_phase_boundaries and stats is not None:
        exp_start = stats.get("exp_phase_start")
        exp_end = stats.get("exp_phase_end")
        if exp_start is not None and exp_end is not None:
            fig = add_exponential_phase(fig, exp_start, exp_end, row=row, col=col)

    # Add crosshairs to umax point
    if show_crosshairs and stats is not None:
        time_umax = stats.get("time_at_umax")
        od_umax = stats.get("od_at_umax")
        if time_umax is not None and od_umax is not None:
            if np.isfinite(time_umax) and np.isfinite(od_umax):
                y_val = np.log(od_umax) if scale == "log" else od_umax

                # Determine bottom of vertical line
                if scale == "log":
                    y_min_vals = []
                    for trace in fig.data:
                        if trace.y is not None and len(trace.y) > 0:
                            valid_y = [y for y in trace.y if np.isfinite(y)]
                            if valid_y:
                                y_min_vals.append(min(valid_y))
                    y_bottom = min(y_min_vals) if y_min_vals else y_val
                else:
                    y_bottom = 0

                # Vertical line
                fig.add_shape(
                    type="line",
                    x0=time_umax,
                    y0=y_bottom,
                    x1=time_umax,
                    y1=y_val,
                    line=dict(color="black", dash="dot", width=2),
                    opacity=0.5,
                    row=row,
                    col=col,
                )

                # Horizontal line
                fig.add_shape(
                    type="line",
                    x0=0,
                    y0=y_val,
                    x1=time_umax,
                    y1=y_val,
                    line=dict(color="black", dash="dot", width=2),
                    opacity=0.5,
                    row=row,
                    col=col,
                )

    # Add od_max horizontal line
    if show_od_max_line and stats is not None:
        od_max = stats.get("max_od")
        if od_max is not None:
            fig = add_od_max_line(fig, od_max, scale=scale, row=row, col=col)

    # Add N0 horizontal line
    if show_n0_line and stats is not None:
        n0 = stats.get("N0")
        if n0 is not None:
            fig = add_N0_line(fig, n0, scale=scale, row=row, col=col)

    # Add umax marker point
    if show_umax_marker and stats is not None:
        time_umax = stats.get("time_at_umax")
        od_umax = stats.get("od_at_umax")
        if time_umax is not None and od_umax is not None:
            if np.isfinite(time_umax) and np.isfinite(od_umax):
                y_val = np.log(od_umax) if scale == "log" else od_umax
                fig.add_trace(
                    go.Scatter(
                        x=[time_umax],
                        y=[y_val],
                        mode="markers",
                        marker=dict(size=15, color="#66BB6A", symbol="circle"),
                        showlegend=False,
                    ),
                    row=row,
                    col=col,
                )

    # Add tangent line at umax
    if show_tangent and stats is not None:
        umax = stats.get("mu_max")
        time_umax = stats.get("time_at_umax")
        od_umax = stats.get("od_at_umax")
        if umax is not None and time_umax is not None and od_umax is not None:
            tangent_data = prepare_tangent_line(umax, time_umax, od_umax, fig, scale)
            if tangent_data is not None:
                time_vals, od_vals = tangent_data
                y_vals = np.log(od_vals) if scale == "log" else od_vals
                fig.add_trace(
                    go.Scatter(
                        x=time_vals,
                        y=y_vals,
                        mode="lines",
                        line=dict(color="green", width=2, dash="dash"),
                        name="umax tangent",
                        showlegend=False,
                        hovertemplate="Tangent line at μmax<extra></extra>",
                    ),
                    row=row,
                    col=col,
                )

    return fig


def plot_derivative_metric(
    t: np.ndarray,
    N: np.ndarray,
    metric: str = "mu",
    fit_result: Optional[Dict[str, Any]] = None,
    sg_window: int = 11,
    sg_poly: int = 2,
    phase_boundaries: Optional[Tuple[float, float]] = None,
    title: Optional[str] = None,
    raw_line_width: int = 1,
    smoothed_line_width: int = 2,
    fitted_line_width: int = 2,
) -> go.Figure:
    """
    Plot either dN/dt or μ (specific growth rate) for a growth curve.

    This function generates up to three traces:
    1. Raw N metric (light grey)
    2. Smoothed N metric (main trace, pink/red)
    3. Model fit metric (dashed blue line, if fit_result provided)

    Parameters
    ----------
    t : numpy.ndarray
        Time array
    N : numpy.ndarray
        OD600 values (baseline-corrected)
    metric : str, optional
        Either "dndt" for dN/dt or "mu" for μ (default: "mu")
    fit_result : dict, optional
        Fit result dictionary from fit_parametric() or fit_non_parametric().
        If provided, the fitted model's derivative will be shown.
        Should contain 'model_type' and 'params' keys.
    sg_window : int, optional
        Savitzky-Golay window size for smoothing (default: 11)
    sg_poly : int, optional
        Savitzky-Golay polynomial order for smoothing (default: 2)
    phase_boundaries : tuple of (float, float), optional
        Tuple of (exp_start, exp_end) for exponential phase boundaries.
        If provided, adds shading for the phase.
    title : str, optional
        Plot title. If None, automatically generated based on metric.
    raw_line_width : int, optional
        Line width of the raw metric trace (default: 1)
    smoothed_line_width : int, optional
        Line width of the smoothed metric trace (default: 2)
    fitted_line_width : int, optional
        Line width of the fitted model metric trace (default: 2)

    Returns
    -------
    plotly.graph_objects.Figure
        Plotly figure with derivative metric plot

    Examples
    --------
    >>> import numpy as np
    >>> from growthcurves import plot_derivative_metric, fit_non_parametric
    >>>
    >>> # Generate some example N
    >>> t = np.linspace(0, 24, 100)
    >>> N = 0.05 * np.exp(0.5 * t) / (1 + (0.05/2.0) * (np.exp(0.5 * t) - 1))
    >>>
    >>> # Plot specific growth rate without fit
    >>> fig = plot_derivative_metric(t, N, metric="mu")
    >>>
    >>> # Plot with fitted model
    >>> fit_result = fit_non_parametric(t, N, umax_method="spline")
    >>> fig = plot_derivative_metric(
    ...     t, N,
    ...     metric="mu",
    ...     fit_result=fit_result,
    ...     phase_boundaries=(5, 15)
    ... )
    """
    from .inference import (
        compute_first_derivative,
        compute_instantaneous_mu,
        compute_sliding_window_growth_rate,
        smooth,
    )

    # Validate metric
    if metric not in ["dndt", "mu"]:
        raise ValueError(f"metric must be 'dndt' or 'mu', got '{metric}'")

    # Convert to numpy arrays
    t = np.asarray(t, dtype=float)
    N = np.asarray(N, dtype=float)

    # Remove non-finite and non-positive values (needed for mu calculation)
    mask = np.isfinite(t) & np.isfinite(N) & (N > 0)
    t = t[mask]
    N = N[mask]

    if len(t) < 3:
        return go.Figure()

    # Store full t range for x-axis
    x_range = [float(t.min()), float(t.max())]

    # Step 1: Calculate metric on raw N
    if metric == "dndt":
        t_metric_raw, metric_raw = compute_first_derivative(t, N)
        metric_label = "dN/dt"
        y_axis_title = "dN/dt"
        plot_title = title or "First Derivative (dN/dt)"
    else:  # mu
        t_metric_raw, metric_raw = compute_instantaneous_mu(t, N)
        metric_label = "μ"
        y_axis_title = "μ (h⁻¹)"
        plot_title = title or "Specific Growth Rate (μ)"

    # Step 2: Smooth the N
    y_smooth = smooth(N, sg_window, sg_poly)

    # Step 3: Calculate metric on smoothed N
    if metric == "dndt":
        t_metric_smooth, metric_smooth = compute_first_derivative(t, y_smooth)
    else:  # mu
        t_metric_smooth, metric_smooth = compute_instantaneous_mu(t, y_smooth)

    # Create figure
    fig = go.Figure()

    template = f"Time=%{{x:.2f}}<br>{metric_label} (raw)=%{{y:.4f}}<extra></extra>"
    # Plot raw metric (light grey)
    fig.add_trace(
        go.Scatter(
            x=t_metric_raw,
            y=metric_raw,
            mode="lines",
            line=dict(width=raw_line_width, color="lightgrey"),
            hovertemplate=template,
            showlegend=False,
            name="Raw",
        )
    )

    template = (
        f"Time=%{{x:.2f}}<br>{metric_label} (smoothed)=%{{y:.4f}}<extra></extra>",
    )
    # Plot smoothed metric (pink/red)
    fig.add_trace(
        go.Scatter(
            x=t_metric_smooth,
            y=metric_smooth,
            mode="lines",
            line=dict(width=smoothed_line_width, color="#FF6692"),
            hovertemplate=template,
            showlegend=False,
            name="Smoothed",
        )
    )

    # Step 4 & 5: Generate model metric and plot (if fit_result provided)
    if fit_result is not None:
        model_type = fit_result.get("model_type", "")
        params = fit_result.get("params", {})
        metric_model = None
        t_model = None

        # Get the fitted N range
        fit_t_min = params.get("fit_t_min")
        fit_t_max = params.get("fit_t_max")

        # Filter to fitted range if available
        if fit_t_min is not None and fit_t_max is not None:
            fit_mask = (t >= fit_t_min) & (t <= fit_t_max)
            t_model = t[fit_mask]
            y_model_raw = N[fit_mask]
            y_model_smooth = y_smooth[fit_mask]
        else:
            # Use full range if fit bounds not available
            t_model = t
            y_model_raw = N
            y_model_smooth = y_smooth

        if len(t_model) >= 2:
            if model_type == "sliding_window":
                # For sliding window, calculate from raw N (as growthcurves does)
                window_points = params.get("window_points", 15)
                if metric == "dndt":
                    # For dN/dt, we need to smooth first then compute derivative
                    _, metric_model = compute_first_derivative(t_model, y_model_smooth)
                else:  # mu
                    # For μ, use sliding window on raw N
                    _, metric_model = compute_sliding_window_growth_rate(
                        t_model, y_model_raw, window_points=window_points
                    )

            elif model_type in get_all_parametric_models():
                # For parametric models, compute metric from the model
                # Evaluate the model on fitted range
                y_model = evaluate_parametric_model(t_model, model_type, params)

                # Compute metric from model
                if metric == "dndt":
                    _, metric_model = compute_first_derivative(t_model, y_model)
                else:  # mu
                    _, metric_model = compute_instantaneous_mu(t_model, y_model)

            elif model_type == "spline":
                # For spline model, reconstruct the spline and evaluate
                try:
                    spline = spline_from_params(params)

                    if metric == "dndt":
                        # Spline is fitted to log(y), so exp(spline(t)) gives y
                        y_log_model = spline(t_model)
                        y_model = np.exp(y_log_model)
                        _, metric_model = compute_first_derivative(t_model, y_model)
                    else:  # mu
                        # μ = d(ln(y))/dt, which is the derivative of the spline
                        metric_model = spline.derivative()(t_model)
                except Exception:
                    # If spline reconstruction fails, skip model trace
                    pass

        # Plot model metric if available
        if (
            metric_model is not None
            and t_model is not None
            and np.isfinite(metric_model).any()
        ):
            template = (
                f"Time=%{{x:.2f}}<br>{metric_label} (fitted)=%{{y:.4f}}<extra></extra>"
            )
            fig.add_trace(
                go.Scatter(
                    x=t_model,
                    y=metric_model,
                    mode="lines",
                    line=dict(width=fitted_line_width, color="#8dcde0"),
                    hovertemplate=template,
                    showlegend=False,
                    name="Fitted",
                )
            )

    # Add phase boundary annotations if provided
    if phase_boundaries is not None and len(phase_boundaries) == 2:
        exp_start, exp_end = phase_boundaries
        if exp_start is not None and exp_end is not None:
            if np.isfinite(exp_start) and np.isfinite(exp_end):
                fig = add_exponential_phase(fig, exp_start, exp_end)

    # Update layout
    fig.update_layout(
        title=plot_title,
        height=400,
        showlegend=False,
        plot_bgcolor="white",
        paper_bgcolor="white",
        margin=dict(l=40, r=20, t=60, b=40),
        template="plotly_white",
    )
    fig.update_xaxes(showgrid=False, title="Time (hours)", range=x_range)
    fig.update_yaxes(showgrid=False, title=y_axis_title)

    return fig


def plot_growth_stats_comparison(
    stats_dict: Dict[str, Dict[str, Any]],
    title: str = "Growth Statistics Comparison",
    metric_order: Optional[list] = None,
) -> go.Figure:
    """
    Create a multi-panel bar chart comparing growth statistics across methods.

    Parameters
    ----------
    stats_dict : dict
        Dictionary mapping method names to their growth statistics dictionaries.
        Each stats dict should contain keys like 'mu_max', 'doubling_time', etc.
    title : str, optional
        Overall title for the figure (default: "Growth Statistics Comparison")
    metric_order : list, optional
        List of metric keys to plot in specific order. If None, uses default order.

    Returns
    -------
    plotly.graph_objects.Figure
        Plotly figure with subplots showing each metric comparison

    Examples
    --------
    >>> # Compare multiple fitting methods
    >>> stats_dict = {
    ...     'logistic': stats_logistic,
    ...     'gompertz': stats_gompertz,
    ...     'spline': stats_spline
    ... }
    >>> fig = plot_growth_stats_comparison(
    ...     stats_dict,
    ...     title="Model Comparison"
    ... )
    >>> fig.show()
    """
    import pandas as pd
    from plotly.subplots import make_subplots

    df = pd.DataFrame(stats_dict).T

    default_metrics = [
        "mu_max",
        "intrinsic_growth_rate",
        "doubling_time",
        "time_at_umax",
        "exp_phase_start",
        "exp_phase_end",
        "model_rmse",
    ]

    metrics = metric_order or [m for m in default_metrics if m in df.columns]
    numeric_df = df.copy()
    for m in metrics:
        numeric_df[m] = pd.to_numeric(numeric_df[m], errors="coerce")

    n_metrics = len(metrics)
    n_cols = 3
    n_rows = int(np.ceil(n_metrics / n_cols))

    fig = make_subplots(
        rows=n_rows,
        cols=n_cols,
        subplot_titles=[m.replace("_", " ").title() for m in metrics],
        horizontal_spacing=0.08,
        vertical_spacing=0.15,
    )

    method_names = list(numeric_df.index)
    for i, metric in enumerate(metrics):
        row = i // n_cols + 1
        col = i % n_cols + 1
        fig.add_trace(
            go.Bar(
                x=method_names,
                y=numeric_df[metric].tolist(),
                showlegend=False,
                marker=dict(line=dict(color="black", width=1)),
            ),
            row=row,
            col=col,
        )

    fig.update_layout(
        title=title,
        height=max(420, 320 * n_rows),
        width=1200,
        bargap=0.25,
        template="plotly_white",
    )
    return fig
