# Waterfall v2 Fixes - Summary

## Issues Fixed

### 1. Data Loading Logic (get_day_data)
**Problem**: When `today=True`, the function was loading BOTH HDF files AND live Lyse data, which could cause duplicate/conflicting data.

**Solution**: 
- **When `today=True`**: Only load live Lyse data (shots currently in memory, not yet saved to disk)
- **When `today=False`**: Only load HDF files from the specified day folder on NAS or data_reanalyzed

**File**: `/home/rick/labscript-suite/userlib/analysislib/analysislib_v2/multishot_lib/main.py`

**Changes**:
- Restructured `get_day_data()` to have two distinct paths
- If today=True: attempts to load from `lyse.data()` with error handling
- If today=False: finds and loads all `.hdf` files from the day folder
- Added better logging to distinguish between the two modes

### 2. Missing Imports in waterfall_v2_example.py
**Problem**: 
- Import path was incorrect: `from waterfall_lib import run_mode` should be `from waterfall_v2.waterfall_lib import run_mode`
- Missing configs: `SHOT_FILTER_CONFIG` and `DEFECT_ANALYSIS_PARAMS` were not defined in `common_config.py`

**Solution**:
- Fixed import path to use absolute import from `waterfall_v2.waterfall_lib`
- Added return None on error for better error handling

**File**: `/home/rick/labscript-suite/userlib/analysislib/analysislib_v2/waterfall_v2/waterfall_v2_example.py`

**Changes**:
- Line 139: `from waterfall_lib import run_mode` → `from waterfall_v2.waterfall_lib import run_mode`
- Added return statement on exception

### 3. Missing Configuration Definitions
**Problem**: `SHOT_FILTER_CONFIG` and `DEFECT_ANALYSIS_PARAMS` were being imported in `waterfall_v2_example.py` but not defined in `common_config.py`

**Solution**: Added both configs to `common_config.py` with sensible defaults

**File**: `/home/rick/labscript-suite/userlib/analysislib/analysislib_v2/waterfall_v2/common_config.py`

**Added**:
```python
SHOT_FILTER_CONFIG = {
    'enabled': True,
    'apply_back_check': True,
    'apply_sat_check': False,
    'apply_probe_norm_err_check': False,
    'back_threshold': 0.05,
    'sat_threshold': 0.001,
    'probe_norm_err_threshold': 1.0,
    'ntot_threshold': 7e5,
}

DEFECT_ANALYSIS_PARAMS = {
    # Defect-finding algorithm parameters
    'gaussian_sigma': 1.5,
    'derivative_threshold_pos': 0.1,
    'derivative_threshold_neg': -0.1,
    'peak_step_filter_enabled': True,
    'peak_step_filter_window_px': 4,
    'peak_step_filter_min_abs_delta_m': 0.4,
    'min_same_sign_peak_distance_um': 10.0,
    'zero_crossing_min_distance_um': 3.0,
    'defect_position_hist_half_window_um': 4.0,
    'defect_size_hist_bin_size': 5.,
}
```

## Testing

A test script `waterfall_v2/test_fixes.py` was created to verify:
1. ✓ All imports work correctly
2. ✓ `get_day_data` with `today=False` loads HDF files
3. ✓ Mode config loads successfully

All tests passed.

## Expected Behavior After Fixes

### When ACTIVE_MODE = 'ARP_Backward' and today = True:
1. Script loads live Lyse data (new shots not yet saved)
2. Filters by sequence 11
3. Extracts magnetization profiles from cam_vert1
4. Averages profiles at each ARP field value
5. Generates waterfall plots (if plotting is working)

### When ACTIVE_MODE = 'ARP_Backward' and today = False:
1. Script loads all HDF files from the specified date folder
2. Same downstream processing as above

## Next Steps to Debug Plotting

If plotting still doesn't work, check:
1. Run the waterfall_v2_example.py with logging enabled
2. Check if `waterfall_plot()` is raising exceptions (now logged)
3. Verify that magnetization profile data is being extracted correctly
4. Check matplotlib display settings if figures are generated but not shown
