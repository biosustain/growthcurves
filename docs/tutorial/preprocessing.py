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
# # Preprocess growth data
#
# This tutorial demonstrates the preprocessing functions in
# `growthcurves.preprocessing`:
#
# - **`path_correct(N, path_length_cm)`**
# - **`blank_subtraction(N, blank)`**
# - **`out_of_iqr_window(values, factor, position)`** — single-window helper
# - **`detect_outliers(N, method, **kwargs)`** — main outlier detection entry point
#   - `method="iqr"` — sliding-window IQR (kwargs: `window_size`, `factor`)
#   - `method="ecod"` — ECOD anomaly detection (kwargs: `factor`)
#
# Use this workflow before model fitting when measurements require optical corrections
# or outlier screening.

# %%
import numpy as np

import growthcurves as gc
from growthcurves import preprocessing as prep

# %% [markdown]
# ## Path length correction

# %%
# Measurements taken at 0.5 cm path length
raw_od = np.array([0.25, 0.30, 0.35, 0.40])
od_1cm = gc.path_correct(raw_od, path_length_cm=0.5)

print(f"Raw OD (0.5 cm): {raw_od}")
print(f"Corrected OD (1.0 cm): {od_1cm}")

# %% [markdown]
# ## Blank subtraction

# %%
sample_od = np.array([0.50, 0.60, 0.70, 0.80])
blank_od = np.array([0.05, 0.052, 0.048, 0.051])
corrected_od = gc.blank_subtraction(sample_od, blank_od)

print(f"Sample OD:   {sample_od}")
print(f"Blank OD:    {blank_od}")
print(f"Corrected OD:{corrected_od}")

# %% [markdown]
# ## Outlier detection in a single window

# %%
window = np.array([0.10, 0.12, 0.65, 0.11, 0.13])
center_is_outlier = prep.out_of_iqr_window(window, factor=1.5, position="center")
first_is_outlier = prep.out_of_iqr_window(window, factor=1.5, position="first")
last_is_outlier = prep.out_of_iqr_window(window, factor=1.5, position="last")

print(f"Window: {window}")
print(f"Center value outlier? {center_is_outlier}")
print(f"First value outlier?  {first_is_outlier}")
print(f"Last value outlier?   {last_is_outlier}")

# %% [markdown]
# ## Outlier detection across a full time series with `detect_outliers`
#
# `detect_outliers(N, method=..., **kwargs)` is the main entry point. Pass
# `method="iqr"` for the sliding-window IQR approach:
#
# - For values in the centre of a window the IQR status is calculated for that window.
# - For the first and last values (which cannot be centred in a window) the IQR status
#   is calculated using the first and last positions of their respective windows.
#   This is especially useful for catching outliers at the start of a series.
#
# Example with a centre outlier:

# %%
od_series = np.array([0.08, 0.11, 0.14, 0.19, 0.23, 0.25, 0.95, 0.31, 0.36, 0.41])
mask = prep.detect_outliers(od_series, method="iqr", window_size=5, factor=1.5)

print(f"OD series: {od_series}")
print(f"Outlier mask: {mask}")
print(f"Outlier indices: {np.where(mask)[0]}")
print(f"Outlier values: {od_series[mask]}")

# %% [markdown]
# Example with a center outlier, and an outlier at the beginning of the series:

# %%
od_series = np.array([0.08, 0.99, 0.14, 0.19, 0.23, 0.25, 0.95, 0.31, 0.36, 0.41])
mask = prep.detect_outliers(od_series, method="iqr", window_size=5, factor=1.5)

print(f"OD series: {od_series}")
print(f"Outlier mask: {mask}")
print(f"Outlier indices: {np.where(mask)[0]}")
print(f"Outlier values: {od_series[mask]}")

# %% [markdown]
# If several outliers are present at the start of a time series, IQR values need to be
# calculated with a sufficiently large window, and maybe iteratively, to detect all
# outliers (here the first value is not detected as an outlier as the second value
# is included in the window and increases the IQR range).

# %%
od_series = np.array([0.99, 0.99, 0.14, 0.19, 0.23, 0.25, 0.95, 0.31, 0.36, 0.41])
mask = prep.detect_outliers(od_series, method="iqr", window_size=5, factor=1.5)

print(f"OD series: {od_series}")
print(f"Outlier mask: {mask}")
print(f"Outlier indices: {np.where(mask)[0]}")
print(f"Outlier values: {od_series[mask]}")

# %% [markdown]
# ## Combined preprocessing pipeline

# %%
raw = np.array([0.10, 0.12, 0.14, 0.16, 0.48, 0.20, 0.22])
blank = np.full_like(raw, 0.02)
path_length_cm = 0.5

raw_1cm = gc.path_correct(raw, path_length_cm=path_length_cm)
blank_1cm = gc.path_correct(blank, path_length_cm=path_length_cm)
baseline_corrected = gc.blank_subtraction(raw_1cm, blank_1cm)
outlier_mask = prep.detect_outliers(
    baseline_corrected, method="iqr", window_size=5, factor=1.5
)

print(f"Raw OD ({path_length_cm} cm): {raw}")
print(f"Path-corrected OD (1 cm): {raw_1cm}")
print(f"Blank-subtracted OD: {baseline_corrected}")
print(f"Outlier mask: {outlier_mask}")
