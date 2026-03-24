# Single-shot defect finding algorithm (waterfall)

This note documents how one shot is converted into a defect count in the waterfall analysis.

## Scope

- Scan mode: `KZ_det_scan`
- Main implementation: [analysislib_v2/waterfall/waterfall_lib.py](analysislib_v2/waterfall/waterfall_lib.py#L1468-L1735)
- Config source for thresholds and options: [analysislib_v2/waterfall/waterfall_config.py](analysislib_v2/waterfall/waterfall_config.py#L182-L204)

## Inputs (per shot)

For one shot, the algorithm uses:

- Magnetization profile `m(x)` from `M_full[i]`
- Pixel-to-micron conversion from camera settings
- Region of interest (ROI) set by:
  - `X_MIN_INTEGRATION`
  - `X_MAX_INTEGRATION`

## Step-by-step algorithm

1. **Smooth magnetization**
   - Apply Gaussian smoothing to `m(x)` with `gaussian_sigma`.
   - If `gaussian_sigma <= 0`, use the raw signal.

2. **Compute derivative**
   - Compute `d(x) = \partial m / \partial x` using `np.gradient`.

3. **Restrict to ROI**
   - Keep only samples between `x_min` and `x_max` (converted to nearest indices).
   - Work with `m_roi(x)` and `d_roi(x)`.

4. **Pick thresholds for this shot**
   - Positive threshold: `thr_pos_i`
   - Negative threshold: `thr_neg_i`
   - These are either:
     - fixed from config, or
     - selected by threshold-scan plateau logic (per detuning group), then assigned to each shot in that group.

5. **Find threshold peaks**
   - Positive defects are local maxima in `d_roi` above `thr_pos_i`:
     - `d[k] >= d[k-1]` and `d[k] > d[k+1]` and `d[k] >= thr_pos_i`
   - Negative defects are local minima in `d_roi` below `thr_neg_i`:
     - `d[k] <= d[k-1]` and `d[k] < d[k+1]` and `d[k] <= thr_neg_i`

6. **Zero-crossing correction for missed defects**
   - Merge positive and negative events, sorted by position.
   - For each consecutive run of same-sign events with length `>= 2`:
     - Search for a zero crossing of `m_roi(x)` between first and last event in that run.
     - If found, add as an extra defect **only if** it is at least `zero_crossing_min_distance_um` from all threshold-peak positions.

    **Alternative way to recover missed defects (derivative-only):**
    - For each same-sign run (`+,+,...` or `-, -,...`) with length `>= 2`, take the interval between the first and last peak of that run.
    - Inside that interval, look at the derivative and find the strongest opposite-sign extremum:
       - if the run is positive, find the most negative value of `d_roi`
       - if the run is negative, find the most positive value of `d_roi`
    - Accept this candidate as an additional missed defect only if:
       - it is a local extremum of the opposite sign,
      - its derivative value has the opposite sign (negative inside a `+` run, positive inside a `-` run),
       - and it is at least `zero_crossing_min_distance_um` from already counted defect positions.
    - This alternative does not require an explicit zero crossing in `m_roi(x)` and can be more robust when the magnetization baseline is offset.

7. **Compute single-shot defect count**

$$
N_{\mathrm{defects}} = N_{+\,\mathrm{peaks}} + N_{-\,\mathrm{peaks}} + N_{\mathrm{zero\_cross\_corrections}}
$$

This is the final defect count for that shot.

## Parameters that control behavior

From `MODE_CONFIGS['KZ_defect_analysis']['defect_analysis']`:

- `gaussian_sigma`
- `derivative_threshold_pos`
- `derivative_threshold_neg`
- `threshold_scan` (+ min/max/points controls)
- `plateau_delta_defects`
- `plateau_min_points`
- `zero_crossing_min_distance_um`
- `missed_defect_correction_method` (`zero_crossing`, `opposite_peak`, `both`)

ROI limits come from top-level `PARAMS`:

- `X_MIN_INTEGRATION`
- `X_MAX_INTEGRATION`

## Practical interpretation

- Threshold peaks catch sharp domain-wall-like transitions.
- Zero-crossing correction recovers likely missing defects when derivative thresholding undercounts in clustered same-sign runs.
- ROI limits ensure defects are counted only in the physically relevant central region.
