"""
Plotting functions for growth curve analysis using Plotly.

This module provides modular functions for creating and annotating growth curve plots.
"""

import numpy as np
import plotly.graph_objects as go
from typing import Optional, Dict, Any, Tuple

from .models import (
    logistic_model,
    gompertz_model,
    richards_model,
    spline_from_params,
)


def create_base_plot(
    time: np.ndarray,
    data: np.ndarray,
    scale: str = "linear",
    xlabel: str = "Time (hours)",
    ylabel: Optional[str] = None,
    marker_size: int = 5,
    marker_opacity: float = 0.3,
    marker_color: str = "gray",
) -> go.Figure:
    """
    Create a base plot with raw data points.

    Parameters
    ----------
    time : np.ndarray
        Time points
    data : np.ndarray
        OD measurements or other growth data
    scale : str, optional
        'linear' or 'log' scale for y-axis (default: 'linear')
    xlabel : str, optional
        X-axis label (default: 'Time (hours)')
    ylabel : str, optional
        Y-axis label. If None, automatically set based on scale
    marker_size : int, optional
        Size of data point markers (default: 5)
    marker_opacity : float, optional
        Opacity of data point markers (default: 0.3)
    marker_color : str, optional
        Color of data point markers (default: 'gray')

    Returns
    -------
    go.Figure
        Plotly figure object with raw data
    """
    # Convert to numpy arrays
    time = np.asarray(time, dtype=float)
    data = np.asarray(data, dtype=float)

    # Determine y-axis data based on scale
    if scale == "log":
        y_data = np.log(data)
        if ylabel is None:
            ylabel = "ln(OD)"
    else:
        y_data = data
        if ylabel is None:
            ylabel = "OD"

    # Create figure
    fig = go.Figure()

    # Add raw data trace
    fig.add_trace(
        go.Scatter(
            x=time,
            y=y_data,
            mode="markers",
            name="Data",
            marker=dict(size=marker_size, opacity=marker_opacity, color=marker_color),
            showlegend=False,
        )
    )

    # Update layout
    fig.update_layout(
        xaxis_title=xlabel,
        yaxis_title=ylabel,
        hovermode="closest",
        template="plotly_white",
        showlegend=False,
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
    fig : go.Figure
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
    go.Figure
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


def add_umax_marker(
    fig: go.Figure,
    time_umax: float,
    od_umax: float,
    mu_max: Optional[float] = None,
    scale: str = "linear",
    marker_symbol: str = "circle",
    marker_size: int = 8,
    marker_color: str = "green",
    vline_color: str = "red",
    vline_dash: str = "dash",
    vline_width: float = 1,
    vline_opacity: float = 0.5,
    row: Optional[int] = None,
    col: Optional[int] = None,
) -> go.Figure:
    """
    Add marker for maximum specific growth rate (Umax).

    Parameters
    ----------
    fig : go.Figure
        Plotly figure to annotate
    time_umax : float
        Time at maximum growth rate
    od_umax : float
        OD value at maximum growth rate
    mu_max : float, optional
        Maximum specific growth rate value (for label)
    scale : str, optional
        'linear' or 'log' - determines y-axis transformation (default: 'linear')
    marker_symbol : str, optional
        Marker symbol (default: 'circle')
    marker_size : int, optional
        Size of marker (default: 8)
    marker_color : str, optional
        Color of marker (default: 'green')
    vline_color : str, optional
        Color of vertical line (default: 'red')
    vline_dash : str, optional
        Dash style for vertical line (default: 'dash')
    vline_width : float, optional
        Width of vertical line (default: 1)
    vline_opacity : float, optional
        Opacity of vertical line (default: 0.5)
    row : int, optional
        Subplot row (for subplots)
    col : int, optional
        Subplot column (for subplots)

    Returns
    -------
    go.Figure
        Updated figure with Umax marker
    """
    if time_umax is None or od_umax is None:
        return fig

    if not np.isfinite(time_umax) or not np.isfinite(od_umax):
        return fig

    # Transform y-value based on scale
    y_val = np.log(od_umax) if scale == "log" else od_umax

    # Create label
    if mu_max is not None:
        label = f"μmax = {mu_max:.3f} h⁻¹"
    else:
        label = "μmax"

    # Add marker
    fig.add_trace(
        go.Scatter(
            x=[time_umax],
            y=[y_val],
            mode="markers",
            name=label,
            marker=dict(symbol=marker_symbol, size=marker_size, color=marker_color),
            showlegend=True,
        ),
        row=row,
        col=col,
    )

    # Add vertical line at Umax
    fig.add_vline(
        x=time_umax,
        line_color=vline_color,
        line_dash=vline_dash,
        line_width=vline_width,
        opacity=vline_opacity,
        row=row,
        col=col,
    )

    return fig


def add_fitted_curve(
    fig: go.Figure,
    time_fit: np.ndarray,
    y_fit: np.ndarray,
    name: str = "Fitted curve",
    color: str = "blue",
    line_width: int = 2,
    window_start: Optional[float] = None,
    window_end: Optional[float] = None,
    scale: str = "linear",
    row: Optional[int] = None,
    col: Optional[int] = None,
) -> go.Figure:
    """
    Add fitted curve to the plot, optionally constrained to a window.

    Parameters
    ----------
    fig : go.Figure
        Plotly figure to annotate
    time_fit : np.ndarray
        Time points for fitted curve
    y_fit : np.ndarray
        Fitted y values
    name : str, optional
        Legend name for fitted curve (default: 'Fitted curve')
    color : str, optional
        Color of fitted curve (default: 'blue')
    line_width : int, optional
        Width of fitted curve line (default: 2)
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
    go.Figure
        Updated figure with fitted curve
    """
    if time_fit is None or y_fit is None:
        return fig

    # Convert to numpy arrays
    time_fit = np.asarray(time_fit, dtype=float)
    y_fit = np.asarray(y_fit, dtype=float)

    # Filter to window if specified
    if window_start is not None and window_end is not None:
        mask = (time_fit >= window_start) & (time_fit <= window_end)
        time_fit = time_fit[mask]
        y_fit = y_fit[mask]

    # Transform y-values based on scale
    if scale == "log":
        y_fit = np.log(y_fit)

    # Add fitted curve
    fig.add_trace(
        go.Scatter(
            x=time_fit,
            y=y_fit,
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
    line_color: str = "red",
    line_dash: str = "dash",
    line_width: float = 1,
    line_opacity: float = 0.5,
    name: str = "ODmax",
    row: Optional[int] = None,
    col: Optional[int] = None,
) -> go.Figure:
    """
    Add horizontal line at maximum OD value.

    Parameters
    ----------
    fig : go.Figure
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
    go.Figure
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


def annotate_plot(
    fig: go.Figure,
    phase_boundaries: Optional[Tuple[float, float]] = None,
    time_umax: Optional[float] = None,
    od_umax: Optional[float] = None,
    od_max: Optional[float] = None,
    umax_point: Optional[Tuple[float, float]] = None,
    fitted_model: Optional[Dict[str, Any]] = None,
    row: Optional[int] = None,
    col: Optional[int] = None,
) -> go.Figure:
    """
    Unified function to add multiple annotations to a plot.

    Annotations are automatically added if the corresponding data is provided (not None).
    To hide an annotation, simply don't pass that parameter or pass None.

    Parameters
    ----------
    fig : go.Figure
        Plotly figure to annotate
    phase_boundaries : tuple of (float, float), optional
        Tuple of (exp_start, exp_end) defining the exponential phase boundaries.
        If provided, adds shading for the exponential growth phase.
    time_umax : float, optional
        Time at maximum growth rate. If provided, adds vertical line.
    od_umax : float, optional
        OD value at maximum growth rate. If provided, adds horizontal line.
    od_max : float, optional
        Maximum OD value. If provided, adds horizontal line at this value.
    umax_point : tuple of (float, float), optional
        Tuple of (time_umax, od_umax) to draw a small green dot at the intersection.
        Format: (time, od_value)
    fitted_model : dict, optional
        Fit result dictionary from fit_model() or fit_non_parametric().
        Can be passed directly (recommended) or as a custom dictionary with:
        - 'model_type': one of 'logistic', 'gompertz', 'richards', 'spline', 'sliding_window'
        - 'params': model parameter dictionary (must include 'fit_t_min' and 'fit_t_max')
        - 'name': legend name (optional, default based on model type)

        For backward compatibility, also accepts:
        - 'window_start': start of fitting window (if fit_t_min not in params)
        - 'window_end': end of fitting window (if fit_t_max not in params)
    row : int, optional
        Subplot row (for subplots)
    col : int, optional
        Subplot column (for subplots)

    Returns
    -------
    go.Figure
        Updated figure with annotations

    Examples
    --------
    >>> # Pass fit result directly (recommended)
    >>> spline_result = gc.non_parametric.fit_non_parametric(time, data, umax_method="spline")
    >>> fig = create_base_plot(time, data, scale="linear")
    >>> fig = annotate_plot(fig, fitted_model=spline_result)  # Scale auto-detected

    >>> # Add all annotations including od_max line and umax point
    >>> fig = create_base_plot(time, data, scale="log")
    >>> fig = annotate_plot(
    ...     fig,
    ...     phase_boundaries=(30, 60),
    ...     time_umax=45,
    ...     od_umax=0.25,
    ...     od_max=0.8,
    ...     umax_point=(45, 0.25),
    ...     fitted_model=spline_result
    ... )  # Scale auto-detected from figure

    >>> # Add only exponential phase, lines, and umax point (no fitted curve)
    >>> fig = annotate_plot(
    ...     fig,
    ...     phase_boundaries=(30, 60),
    ...     time_umax=45,
    ...     od_umax=0.25,
    ...     od_max=0.8,
    ...     umax_point=(45, 0.25)
    ... )
    """
    # Auto-detect scale from existing figure data
    # This ensures annotations match the existing plot scale
    scale = "linear"  # default
    if len(fig.data) > 0:
        # Check the first trace to detect scale
        first_trace_y = fig.data[0].y
        if first_trace_y is not None and len(first_trace_y) > 0:
            # If y-values are mostly negative and small (like log-transformed OD),
            # it's likely in log scale
            y_arr = np.asarray(first_trace_y)
            valid_y = y_arr[np.isfinite(y_arr)]
            if len(valid_y) > 0:
                # Heuristic: log(OD) typically ranges from -5 to 1
                # Linear OD typically ranges from 0 to 10
                if np.max(valid_y) < 5 and np.min(valid_y) < 0:
                    # Likely log scale
                    scale = "log"
                else:
                    # Likely linear scale
                    scale = "linear"

    # Extract exp_start and exp_end from phase_boundaries tuple
    exp_start = None
    exp_end = None
    if phase_boundaries is not None and len(phase_boundaries) == 2:
        exp_start, exp_end = phase_boundaries

    # Add exponential phase shading (if both start and end are provided)
    if exp_start is not None and exp_end is not None:
        fig = add_exponential_phase(fig, exp_start, exp_end, row=row, col=col)

    # Add fitted curve (if data is provided)
    # Added before marker so marker is on top
    if fitted_model is not None:
        model_type = fitted_model.get("model_type")
        params = fitted_model.get("params")

        # Extract window boundaries from params (new format) or top-level (old format)
        if params is not None and "fit_t_min" in params and "fit_t_max" in params:
            window_start = params["fit_t_min"]
            window_end = params["fit_t_max"]
        else:
            # Backward compatibility: check for old format
            window_start = fitted_model.get("window_start")
            window_end = fitted_model.get("window_end")

        # Generate default name based on model type
        default_names = {
            "logistic": "Logistic fit",
            "gompertz": "Gompertz fit",
            "richards": "Richards fit",
            "spline": "Spline fit",
            "sliding_window": "Sliding window fit",
        }
        name = fitted_model.get("name", default_names.get(model_type, "Fitted curve"))

        if (
            model_type is not None
            and params is not None
            and window_start is not None
            and window_end is not None
        ):
            time_fit = np.linspace(window_start, window_end, 200)
            y_fit = None

            if model_type == "logistic":
                y_fit = logistic_model(
                    time_fit, params["K"], params["y0"], params["r"], params["t0"]
                )
            elif model_type == "gompertz":
                y_fit = gompertz_model(
                    time_fit,
                    params["K"],
                    params["y0"],
                    params["mu_max_param"],
                    params["lam"],
                )
            elif model_type == "richards":
                y_fit = richards_model(
                    time_fit,
                    params["K"],
                    params["y0"],
                    params["r"],
                    params["t0"],
                    params["nu"],
                )
            elif model_type == "spline":
                spline = spline_from_params(params)
                y_fit = np.exp(spline(time_fit))
            elif model_type == "sliding_window":
                # For sliding window, we can show the linear fit in log space
                slope = params["slope"]
                intercept = params["intercept"]
                y_fit = np.exp(slope * time_fit + intercept)

            if y_fit is not None:
                fig = add_fitted_curve(
                    fig,
                    time_fit,
                    y_fit,
                    name=name,
                    color="blue",
                    window_start=window_start,
                    window_end=window_end,
                    scale=scale,
                    row=row,
                    col=col,
                )

    # Add vertical and horizontal lines from axes to umax point (if both coordinates provided)
    if (
        time_umax is not None
        and od_umax is not None
        and np.isfinite(time_umax)
        and np.isfinite(od_umax)
    ):
        y_val = np.log(od_umax) if scale == "log" else od_umax

        # Determine the y-axis minimum from existing data to properly anchor the vertical line
        y_min = 0  # default for linear scale
        if len(fig.data) > 0:
            y_values = []
            for trace in fig.data:
                if trace.y is not None:
                    y_values.extend([y for y in trace.y if y is not None and np.isfinite(y)])
            if y_values:
                y_min = min(y_values)

        # Add vertical line from bottom of plot to umax point
        fig.add_shape(
            type="line",
            x0=time_umax,
            y0=y_min,
            x1=time_umax,
            y1=y_val,
            line=dict(color="black", dash="dot", width=1),
            opacity=0.5,
            row=row,
            col=col,
        )

        # Add horizontal line from y-axis to umax point
        fig.add_shape(
            type="line",
            x0=0,
            y0=y_val,
            x1=time_umax,
            y1=y_val,
            line=dict(color="black", dash="dot", width=1),
            opacity=0.5,
            row=row,
            col=col,
        )

    # Add od_max horizontal line (if provided)
    if od_max is not None:
        fig = add_od_max_line(fig, od_max, scale=scale, row=row, col=col)

    # Add umax point (if provided)
    if umax_point is not None and len(umax_point) == 2:
        t_val, od_val = umax_point
        if np.isfinite(t_val) and np.isfinite(od_val):
            # Transform y-value based on scale
            y_val = np.log(od_val) if scale == "log" else od_val

            # Add small green dot at the intersection
            fig.add_trace(
                go.Scatter(
                    x=[t_val],
                    y=[y_val],
                    mode="markers",
                    marker=dict(size=8, color="green", symbol="circle"),
                    showlegend=False,
                ),
                row=row,
                col=col,
            )

    return fig
