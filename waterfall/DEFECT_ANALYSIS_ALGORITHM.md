# KZ Defect Analysis Algorithm

This document explains how `KZ_defect_analysis` works in `waterfall_lib.py` and what each parameter in `waterfall_config.py` does.

## Overview

For each shot, the algorithm:
1. Smooths the 1D magnetization profile.
2. Computes the spatial derivative $\partial m / \partial x$.
3. Detects positive and negative derivative peaks above fixed thresholds.
4. Rejects weak/false peaks using a local magnetization-step test.
5. Merges nearby same-sign peaks.
6. Adds missed defects using the **opposite-peak** method (only).
7. Counts defects and builds summary histograms.

Threshold-scan logic is removed from `KZ_defect_analysis`.

---

## Step-by-step pipeline

## 1) Build ROI and preprocess signal
- Uses integration bounds from shared params:
  - `X_MIN_INTEGRATION`
  - `X_MAX_INTEGRATION`
- Converts x-axis to microns using camera pixel size.
- Applies Gaussian smoothing to each shot magnetization profile.

Parameter:
- `gaussian_sigma`  
  Gaussian smoothing width (pixels). Larger values suppress noise but can blur sharp features.

## 2) Detect derivative peaks
- Compute $d(x) = \nabla m(x)$ in the ROI.
- Positive defects candidates: local maxima with $d(x) \ge \text{derivative_threshold_pos}$.
- Negative defects candidates: local minima with $d(x) \le \text{derivative_threshold_neg}$.

Parameters:
- `derivative_threshold`  
  Backward-compatibility fallback magnitude.
- `derivative_threshold_pos`  
  Positive derivative threshold.
- `derivative_threshold_neg`  
  Negative derivative threshold.

## 3) Filter false positives by magnetization step
For each candidate peak index $i$:
- Compute left/right local averages on the smoothed magnetization:
  - left: `[i-window, i)`
  - right: `(i, i+1+window]`
- Keep peak only if
$$
\left|\langle m\rangle_{\text{right}} - \langle m\rangle_{\text{left}}\right| \ge \text{peak_step_filter_min_abs_delta_m}
$$

Parameters:
- `peak_step_filter_enabled`
- `peak_step_filter_window_px`
- `peak_step_filter_min_abs_delta_m`

## 4) Merge nearby same-sign peaks
After filtering, same-sign peaks closer than a minimum distance are merged into one midpoint defect.

Parameter:
- `min_same_sign_peak_distance_um`

## 5) Correct missed defects (opposite-peak only)
For each run of adjacent same-sign detected peaks:
- Search only between neighboring same-sign peaks.
- Find opposite-sign extremum in derivative (local min for positive run, local max for negative run).
- Accept candidate only if its distance from existing kept threshold peaks is at least:
$$
\Delta x_{\min} = \text{zero_crossing_min_distance_um}
$$

Parameter:
- `zero_crossing_min_distance_um`

Note: correction method is fixed to **opposite_peak** in code.

## 6) Count defects
Per shot defect count:
- all kept threshold peaks (positive + negative)
- plus accepted corrected defects.

Summary metrics printed:
- total defects
- corrected missed defects
- rejected peaks
- average defects/shot with SEM.

## 7) Histograms produced
1. Defects per shot.
2. Defect position accumulation (counts within $\pm$ `defect_position_hist_half_window_um` around each 1-um center).
3. Defect size distribution (nearest pos-to-neg distance estimate).

Parameter:
- `defect_position_hist_half_window_um`

---

## Parameter list (current `defect_analysis` block)

- `gaussian_sigma`
- `derivative_threshold`
- `derivative_threshold_pos`
- `derivative_threshold_neg`
- `peak_step_filter_enabled`
- `peak_step_filter_window_px`
- `peak_step_filter_min_abs_delta_m`
- `min_same_sign_peak_distance_um`
- `zero_crossing_min_distance_um`
- `defect_position_hist_half_window_um`

These are the only parameters needed now for `KZ_defect_analysis`.
