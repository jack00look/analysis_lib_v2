"""
MULTISHOT_LIB - README

New library for uniform multishot analysis with parameter consistency checking
and automatic shot quality filtering.

Created: April 16, 2026

IMPORTANT NOTE ON DATA FORMAT
=============================
The HDF files in /home/{user}/NAS542_dataBEC2/ are **Lyse database result files**
(pandas DataFrames stored with pd.to_hdf), NOT raw shot HDF5 files.

These files contain:
  ✓ Pre-analyzed results (already processed by show_ODs or other analysis scripts)
  ✓ Tabular data (rows = shots, columns = analysis results)
  ✓ Parameter consistency already assumed (same analysis config for all shots in file)
  ✗ NOT raw image data
  ✗ NOT individual shot files

Therefore, multishot_lib is designed to work with these Lyse result files, NOT raw shots.
"""

The multishot_lib provides a unified system for loading and filtering shots across
multiple experiments. It ensures data consistency and quality by:

1. **Parameter Consistency Check**: Verifies all processing parameters from the
   raw image analysis (show_ODs_v2) are identical across all shots for a given camera.
   If parameters differ, the script stops and reports which shots have different values.

2. **Quality Filtering**: Rejects individual shots based on:
   - Background contamination (cnt_rel_atoms_back)
   - Saturation levels (cnt_rel_atoms_sat)
   - Probe normalization errors (probe_norm_err)
   - DMD profile update status (dmd_profile_updated) - optional, can be disabled

3. **Reporting**: Prints detailed rejection statistics for each failure mode


Structure
=========

multishot_lib/
├── __init__.py           # Module exports
├── config.py             # Configurable thresholds and parameters
└── main.py               # Core functions: get_day_data(), get_and_filter_shots()

waterfall_v2/
├── waterfall_config_v2.py   # Configuration file (mirrors original waterfall_config.py)
└── waterfall_v2_example.py  # Example script showing how to use multishot_lib


Key Functions
=============

get_day_data(hdf_config, hdf_root='...')
  ├─ Loads all HDF5 files from a specific day
  ├─ hdf_config: dict with 'today', 'year', 'month', 'day'
  └─ Returns: sorted list of HDF5 file paths

get_and_filter_shots(hdf_config, camera_name, seqs=None)
  ├─ Main function: Load and filter shots
  ├─ Steps:
  │  1. Load all files for the day
  │  2. Filter by sequence indices (if provided)
  │  3. CHECK PARAMETER CONSISTENCY for camera across all shots
  │     → If params differ, prints difference report and returns empty DataFrame
  │  4. For each shot, extract image names for camera
  │  5. For each image, check quality metrics
  │  6. Filter by DMD update status (if ENABLE_DMD_CHECK = True)
  │  7. Collect accepted shots into DataFrame
  ├─ Parameters:
  │  - hdf_config: dict (see get_day_data)
  │  - camera_name: str (e.g., 'cam_vert1')
  │  - seqs: list or None (sequence indices to include)
  └─ Returns: pd.DataFrame with accepted shots and all HDF5 results


Configuration (config.py)
=========================

ENABLE_DMD_CHECK
  → Set to False to skip DMD profile update validation
  → Default: True

DMD_LOAD_TIME_WINDOW_S
  → Time window (seconds) for DMD load to be considered fresh
  → If (load_time - shot_run_time) > this value, shot rejected
  → Default: 20.0

BACKGROUND_CONTAMINATION_THRESHOLD
  → Maximum allowed cnt_rel_atoms_back fraction
  → Default: 0.001 (0.1%)

SATURATION_THRESHOLD
  → Maximum allowed cnt_rel_atoms_sat fraction
  → Default: 0.05 (5%)

PROBE_NORM_ERROR_THRESHOLD
  → Maximum allowed probe_norm_err
  → Default: 0.1 (10%)


Usage Example
=============

from multishot_lib import get_and_filter_shots

# Define what day and camera to load
hdf_config = {
    'today': True,
    'year': 2026,
    'month': 4,
    'day': 16,
}

# Get filtered DataFrame
df = get_and_filter_shots(
    hdf_config=hdf_config,
    camera_name='cam_vert1',
    seqs=[49, 50]
)

# Now df contains only shots that:
# ✓ All have identical processing parameters for cam_vert1
# ✓ All have acceptable background, saturation, and norm_err
# ✓ All have dmd_profile_updated=True (if enabled)

# Use df for further analysis...
for idx, row in df.iterrows():
    shot_file = row['shot_file']
    # Extract results from row, e.g.:
    # n1D_x = row['cam_vert1_PTAI_m1_n1D_x']
    # etc.


Parameter Consistency Report Example
=====================================

When parameter inconsistency is detected, multishot_lib prints:

================================================================================
PARAMETER INCONSISTENCY DETECTED for camera cam_vert1
================================================================================

Parameter 'alpha' differs in 2 shot(s):
  - 2026_04_16_Do_BEC_0039_rep00616
  - 2026_04_16_Do_BEC_0039_rep00632

Parameter 'roi' differs in 1 shot(s):
  - 2026_04_16_Do_BEC_0039_rep00650

CANNOT PROCEED: All processing parameters must be identical.
================================================================================

→ This indicates which parameters differ and which shots have the different values
→ Script stops here to prevent inconsistent analysis


Rejection Statistics Example
=============================

SHOT FILTERING SUMMARY:
  Total shots: 50
  Accepted: 42
  Rejected: 8

REJECTION BREAKDOWN:
  saturation (PTAI_m1): 3
  background (PTAI_m1): 2
  norm_err (PTAI_m1): 2
  dmd_not_updated: 1


Integration with show_ODs_v2
=============================

For multishot_lib to work, show_ODs_v2.py must save:

1. Processing parameters:
   results/cam_vert1_param_roi
   results/cam_vert1_param_alpha
   results/cam_vert1_param_chi_sat
   ... (all parameters from get_processing_params)

2. Quality metrics:
   results/cam_vert1_PTAI_m1_cnt_rel_atoms_back
   results/cam_vert1_PTAI_m1_cnt_rel_atoms_sat
   results/cam_vert1_PTAI_m1_probe_norm_err
   ... (for each image)

3. DMD update status (NEW):
   results/cam_vert1_dmd_profile_updated

✓ show_ODs_v2.py already saves all of these!


Common Workflows
================

Workflow 1: Load all shots from today, filter for one camera
────────────────────────────────────────────────────────────
    df = get_and_filter_shots(
        hdf_config={'today': True},
        camera_name='cam_vert1'
    )

Workflow 2: Load specific sequences, check parameters, filter
──────────────────────────────────────────────────────────────
    df = get_and_filter_shots(
        hdf_config={'today': True},
        camera_name='cam_vert1',
        seqs=[49, 50, 51]
    )

Workflow 3: Load from a previous day
───────────────────────────────────────
    df = get_and_filter_shots(
        hdf_config={
            'today': False,
            'year': 2026,
            'month': 4,
            'day': 14
        },
        camera_name='cam_vert1'
    )

Workflow 4: Disable DMD check
──────────────────────────────
    # In config.py:
    ENABLE_DMD_CHECK = False
    
    # Then:
    df = get_and_filter_shots(...)  # No DMD rejection


Troubleshooting
===============

Q: "Parameter inconsistency detected" - what should I do?
A: Review which parameters differ and which shots have the different values.
   Likely causes:
   - Different raw image analysis parameters were used
   - Different camera calibrations
   - Solution: Re-run show_ODs_v2 with consistent parameters, or use separate analyses

Q: Too many shots rejected for saturation/background?
A: Adjust thresholds in multishot_lib/config.py:
   - SATURATION_THRESHOLD
   - BACKGROUND_CONTAMINATION_THRESHOLD
   - PROBE_NORM_ERROR_THRESHOLD

Q: DMD check failing unexpectedly?
A: Set ENABLE_DMD_CHECK = False in config.py if not relevant for your experiment
   Or adjust DMD_LOAD_TIME_WINDOW_S if timing window is too strict

Q: No results appearing in DataFrame?
A: Check logs for:
   - Parameter inconsistency (stops early)
   - "No images found" (HDF5 structure issue)
   - All shots rejected (too strict thresholds?)


Future Enhancements
===================

- Per-image quality metrics summary (show which images failed most often)
- Automatic threshold optimization based on image statistics
- Integration with Lyse database queries
- Support for filtering by custom HDF5 paths (not just camera name)
- Cached parameter consistency checks for faster repeated runs
"""
