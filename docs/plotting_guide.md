# Plotting Module Guide

The `growthcurves.plot` module provides modular, interactive plotting functions using Plotly for growth curve analysis.

## Overview

The plotting module is designed with a modular architecture:

1. **Base plot creation**: Start with raw data visualization
2. **Individual annotations**: Add specific features (exponential phase, Umax markers, fitted curves)
3. **Unified annotation**: Apply multiple annotations with boolean flags
4. **Convenience functions**: Complete plots with minimal code

## Module Functions

### 1. `create_base_plot()`

Creates a base plot with raw data points.

```python
import growthcurves as gc
import numpy as np

time = np.array([...])  # Your time data
data = np.array([...])  # Your OD data

# Create base plot (linear scale)
fig = gc.plot.create_base_plot(time, data, scale='linear', title='Growth Curve')
fig.show()

# Create base plot (log scale)
fig = gc.plot.create_base_plot(time, data, scale='log', title='Growth Curve (Log)')
fig.show()
```

**Parameters:**
- `time`: Time points (numpy array)
- `data`: OD measurements (numpy array)
- `scale`: 'linear' or 'log' (default: 'linear')
- `title`: Plot title (optional)
- `xlabel`, `ylabel`: Axis labels (optional)
- `marker_size`, `marker_opacity`, `marker_color`: Marker styling (optional)

### 2. `add_exponential_phase()`

Adds shaded region showing the exponential growth phase.

```python
# Start with base plot
fig = gc.plot.create_base_plot(time, data)

# Add exponential phase shading
fig = gc.plot.add_exponential_phase(
    fig,
    exp_start=30.0,  # Start of exponential phase
    exp_end=70.0     # End of exponential phase
)
fig.show()
```

**Parameters:**
- `fig`: Plotly figure object
- `exp_start`: Start time of exponential phase
- `exp_end`: End time of exponential phase
- `color`: Shade color (default: 'lightgreen')
- `opacity`: Shade opacity (default: 0.25)

### 3. `add_umax_marker()`

Adds a marker indicating the point of maximum specific growth rate.

```python
fig = gc.plot.create_base_plot(time, data, scale='log')

# Add Umax marker
fig = gc.plot.add_umax_marker(
    fig,
    time_umax=50.0,     # Time at max growth rate
    od_umax=0.25,       # OD at max growth rate
    mu_max=0.12,        # Max specific growth rate (for label)
    scale='log'         # Must match figure scale
)
fig.show()
```

**Parameters:**
- `fig`: Plotly figure object
- `time_umax`: Time at maximum growth rate
- `od_umax`: OD value at maximum growth rate
- `mu_max`: Maximum specific growth rate value (optional, for label)
- `scale`: 'linear' or 'log' (must match figure scale)
- Marker styling options: `marker_symbol`, `marker_size`, `marker_color`, etc.

### 4. `add_fitted_curve()`

Adds a fitted curve to the plot, optionally constrained to a specific window.

```python
# Prepare fitted curve data
time_fit = np.linspace(30, 70, 200)
y_fit = your_fitted_values  # From spline, model, etc.

fig = gc.plot.create_base_plot(time, data)

# Add fitted curve (constrained to fitting window)
fig = gc.plot.add_fitted_curve(
    fig,
    time_fit=time_fit,
    y_fit=y_fit,
    name='Spline fit',
    color='blue',
    window_start=30.0,  # Only show curve in this range
    window_end=70.0
)
fig.show()
```

**Parameters:**
- `fig`: Plotly figure object
- `time_fit`: Time points for fitted curve
- `y_fit`: Fitted y values
- `name`: Legend name (default: 'Fitted curve')
- `color`: Line color (default: 'blue')
- `window_start`, `window_end`: Constrain curve to this window (optional)
- `scale`: 'linear' or 'log' (must match figure scale)

### 5. `annotate_plot()` - Unified Annotation Function

Apply multiple annotations based on provided inputs.

```python
# Create base plot
fig = gc.plot.create_base_plot(time, data, scale='linear')

# Prepare fitted model config
fitted_model = {
    'model_type': 'spline',
    'params': result['params'],
    'name': 'Spline fit',
    'window_start': 30.0,
    'window_end': 70.0
}

# Add all annotations
fig = gc.plot.annotate_plot(
    fig,
    exp_start=30.0,
    exp_end=70.0,
    time_umax=50.0,
    od_umax=0.25,
    mu_max=0.12,
    fitted_model=fitted_model,
    scale='linear'
)
fig.show()
```

**Parameters:**
- `fig`: Plotly figure object
- `exp_start`, `exp_end`: Exponential phase boundaries
- `time_umax`, `od_umax`, `mu_max`: Umax marker data
- `fitted_model`: Dictionary with fitted model params
- `scale`: 'linear' or 'log'

### 6. `plot_growth_curve()` - Convenience Function

Create a complete annotated plot with minimal code.

```python
# Fit data using non-parametric method
result = gc.non_parametric.fit_non_parametric(time, data, umax_method="spline")
stats = extract_stats_from_fit(result, time, data)

# Prepare fitted curve (from spline)
exp_start = stats['exp_phase_start']
exp_end = stats['exp_phase_end']
exp_mask = (time >= exp_start) & (time <= exp_end) & np.isfinite(data) & (data > 0)
t_exp = np.array(time)[exp_mask]
y_exp = np.array(data)[exp_mask]
spline, _ = gc.models.spline_model(
    t_exp, np.log(y_exp), result['params']['spline_s'], k=3
)
t_fit = np.linspace(t_exp.min(), t_exp.max(), 200)
y_fit = np.exp(spline(t_fit))

fitted_model = {
    'model_type': 'spline',
    'params': result['params'],
    'name': 'Spline fit',
    'window_start': exp_start,
    'window_end': exp_end
}

# Create complete plot in one call
fig = gc.plot.plot_growth_curve(
    time,
    data,
    stats=stats,
    fitted_model=fitted_model,
    scale='log',
    title='Spline Method - Growth Curve',
    show_exp_phase=True,
    show_umax=True,
    show_fitted=True
)
fig.show()
```

**Parameters:**
- `time`, `data`: Raw data arrays
- `stats`: Dictionary of growth statistics (from `extract_stats_from_fit()`)
- `fitted_model`: Dictionary with fitted model params (optional)
- `scale`: 'linear' or 'log'
- `title`: Plot title (optional)
- `show_exp_phase`, `show_umax`, `show_fitted`: Boolean flags

## Usage Examples

### Example 1: Manual Step-by-Step Annotation

```python
import growthcurves as gc
import numpy as np

# Create base plot
fig = gc.plot.create_base_plot(time, data, scale='log', title='Growth Analysis')

# Add features one by one
fig = gc.plot.add_exponential_phase(fig, 30, 70)
fig = gc.plot.add_umax_marker(fig, 50, 0.25, mu_max=0.12, scale='log')
fig = gc.plot.add_fitted_curve(
    fig, t_fit, y_fit,
    name='Model',
    window_start=30,
    window_end=70,
    scale='log'
)

fig.show()
```

### Example 2: Selective Annotations with Boolean Flags

```python
# Show only fitted curve, hide other annotations
fig = gc.plot.plot_growth_curve(
    time, data,
    stats=stats,
    fitted_model=fitted_model,
    show_exp_phase=False,  # Hide
    show_umax=False,       # Hide
    show_fitted=True       # Show
)
fig.show()
```

### Example 3: Compare Linear and Log Scales

```python
from plotly.subplots import make_subplots

# This could be extended to create side-by-side comparisons
fig_linear = gc.plot.plot_growth_curve(
    time, data, stats=stats, fitted_model=fitted_model,
    scale='linear', title='Linear Scale'
)

fig_log = gc.plot.plot_growth_curve(
    time, data, stats=stats, fitted_model=fitted_model,
    scale='log', title='Log Scale'
)

fig_linear.show()
fig_log.show()
```

## Key Features

### 1. Modular Design
- Build plots incrementally
- Add only the annotations you need
- Full control over each component

### 2. Boolean Control
- Use `show_*` flags to toggle annotations
- Same function call, different outputs
- Easy to compare different visualizations

### 3. Window-Constrained Fitted Curves
- Show fitted curves only where they were fitted
- Avoid extrapolation artifacts
- Clearly communicate the scope of the fit

### 4. Interactive Plotly Figures
- Zoom, pan, and explore
- Hover for data values
- Export to PNG, SVG, etc.

### 5. Scale Flexibility
- Switch between linear and log scales
- Automatic y-axis transformation
- Consistent annotation behavior

## Integration with Growth Curve Analysis

The plotting module integrates seamlessly with the fitting functions:

```python
# 1. Fit your data
result = gc.non_parametric.fit_non_parametric(time, data, umax_method="spline")
stats = extract_stats_from_fit(result, time, data)

# 2. Extract fitted curve from result
exp_start = stats['exp_phase_start']
exp_end = stats['exp_phase_end']
exp_mask = (time >= exp_start) & (time <= exp_end) & np.isfinite(data) & (data > 0)
t_exp = np.array(time)[exp_mask]
y_exp = np.array(data)[exp_mask]
# ... reconstruct spline or use model parameters

# 3. Plot with annotations
fig = gc.plot.plot_growth_curve(time, data, stats=stats, fitted_model=fitted_model)
fig.show()
```

## Tips and Best Practices

1. **Match scales**: Always pass the correct `scale` parameter to annotation functions
2. **Window constraints**: Use `window_start`/`window_end` to show curves only where fitted
3. **Boolean flags**: Leverage `show_*` flags to create multiple plot variants efficiently
4. **Interactive exploration**: Use Plotly's built-in tools to explore your data
5. **Export options**: Save figures using Plotly's export functionality

## Requirements

- `plotly>=5.0.0` (added to requirements.txt)
- `numpy>=1.20.0`

## See Also

- [Tutorial notebook](tutorial/tutorial.ipynb) for complete examples
- Main fitting modules: `growthcurves.parametric`, `growthcurves.non_parametric`
- Utility functions: `growthcurves.utils`
