"""
MULTISHOT ANALYSIS SYSTEM - IMPLEMENTATION SUMMARY
April 16, 2026

Created for uniform shot acceptance/rejection across multishot experiments.
"""

COMPLETED TASKS
===============

1. ✅ Enhanced show_ODs_v2.py with DMD Profile Update Check
   ─────────────────────────────────────────────────────────
   Location: /home/rick/labscript-suite/userlib/analysislib/analysislib_v2/
            imaging_analysis/show_ODs_v2.py

   Changes:
   • Added zerorpc import with graceful fallback if unavailable
   • Implemented _to_unix_timestamp() function for timestamp parsing
   • Implemented _get_dmd_profile_update_status() function that:
     - Connects to DMD server
     - Compares load completion time vs shot run time
     - Returns True if profile was updated within LOAD_TIME_WINDOW_S
   • Added DMD check after each camera processing
   • Saves cam_{name}_dmd_profile_updated flag to HDF5 results

   Behavior:
   - Gracefully handles missing zerorpc/feedback_config
   - Logs debug info if DMD check fails
   - Sets dmd_profile_updated = False if any step fails


2. ✅ Created multishot_lib - Core Library
   ─────────────────────────────────────────
   Location: /home/rick/labscript-suite/userlib/analysislib/analysislib_v2/
            multishot_lib/

   Files:
   • __init__.py - Module exports
   • config.py - All configurable thresholds:
     - ENABLE_DMD_CHECK (default: True) - Can be disabled manually
     - DMD_LOAD_TIME_WINDOW_S (default: 20.0)
     - DMD_SERVER_IP, DMD_SERVER_PORT
     - BACKGROUND_CONTAMINATION_THRESHOLD (default: 0.001)
     - SATURATION_THRESHOLD (default: 0.05)
     - PROBE_NORM_ERROR_THRESHOLD (default: 0.1)
   • main.py - Core functions:
     - get_day_data(hdf_config) → List of HDF5 files for day
     - get_and_filter_shots(hdf_config, camera_name, seqs) → DataFrame

   Key Features:
   ✓ get_and_filter_shots() workflow:
     1. Load all files from specified day
     2. Filter by sequence indices (if provided)
     3. CHECK PARAMETER CONSISTENCY for camera across all shots
        → If ANY parameter differs, prints detailed report showing which
          parameters differ and which shots have different values
        → Returns empty DataFrame, stopping analysis
     4. For each shot, get image names for camera
     5. Check quality metrics (background, saturation, norm_err)
     6. Check DMD profile update status (if ENABLE_DMD_CHECK=True)
     7. Collect passing shots into DataFrame
     8. Print rejection statistics breakdown

   ✓ Rejection Statistics Include:
     - Total shots processed
     - Shots accepted
     - Shots rejected
     - Breakdown by reason (saturation, background, norm_err, dmd_not_updated, etc.)


3. ✅ Created waterfall_v2 - New Waterfall Analysis Scripts
   ───────────────────────────────────────────────────────────
   Location: /home/rick/labscript-suite/userlib/analysislib/analysislib_v2/
            waterfall_v2/

   Files:
   • waterfall_config_v2.py - Configuration (mirrors original waterfall_config)
     - Uses v2 library paths
     - All original settings preserved
   • waterfall_v2_example.py - Example script showing:
     - How to import multishot_lib
     - How to load configuration
     - How to call get_and_filter_shots()
     - Interpretation of returned DataFrame
     - Ready for further analysis (waterfall plots, fitting, etc.)

   Usage:
     python waterfall_v2_example.py
       → Loads shots based on ACTIVE_MODE and HDF_CONFIG
       → Filters by camera from magnetization modality
       → Returns DataFrame with accepted shots


WORKFLOW EXAMPLE
================

    from multishot_lib import get_and_filter_shots

    # Configuration
    hdf_config = {'today': True}
    camera_name = 'cam_vert1'
    seqs = [49]

    # Load and filter
    df = get_and_filter_shots(hdf_config, camera_name, seqs)

    # df now contains only shots where:
    ✓ All cam_vert1 processing parameters are identical
    ✓ All cam_vert1 images pass background/saturation/norm_err checks
    ✓ All shots have dmd_profile_updated=True

    # Use df for analysis
    for idx, row in df.iterrows():
        n1D_x = row['cam_vert1_PTAI_m1_n1D_x']
        n1D_y = row['cam_vert1_PTAI_m1_n1D_y']
        # ... further processing


PARAMETER CONSISTENCY CHECK OUTPUT
===================================

When parameters differ, the script stops and prints:

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

→ User must resolve parameter differences before proceeding
→ Can re-run show_ODs_v2 with consistent parameters


REJECTION STATISTICS OUTPUT
============================

    SHOT FILTERING SUMMARY:
      Total shots: 50
      Accepted: 42
      Rejected: 8

    REJECTION BREAKDOWN:
      saturation (PTAI_m1): 3
      background (PTAI_m1): 2
      norm_err (PTAI_m1): 2
      dmd_not_updated: 1

→ Detailed breakdown of rejection reasons
→ Helps identify systematic issues (e.g., many saturation rejections)


CONFIGURATION ADJUSTMENT
=========================

To disable specific checks or adjust thresholds:

    # In multishot_lib/config.py:

    # Option 1: Disable DMD check entirely
    ENABLE_DMD_CHECK = False

    # Option 2: Make saturation threshold less strict
    SATURATION_THRESHOLD = 0.10  # Allow up to 10%

    # Option 3: Make background check more strict
    BACKGROUND_CONTAMINATION_THRESHOLD = 0.0005  # Only 0.05%


BACKWARD COMPATIBILITY
======================

✓ Original libraries unchanged:
  - imaging_analysis_lib/main.py (unchanged)
  - imaging_analysis_lib/camera_settings.py (unchanged)
  - waterfall/ (original scripts unchanged)
  - show_ODs.py (original still works)

✓ show_ODs_v2.py enhancements:
  - Added DMD check (gracefully fails if zerorpc not available)
  - No breaking changes to existing functionality
  - New field in results: cam_{name}_dmd_profile_updated

✓ New infrastructure:
  - multishot_lib/ (completely new)
  - waterfall_v2/ (new, separate from original waterfall/)


INTEGRATION WITH SHOW_ODS_V2
=============================

show_ODs_v2.py now saves for each camera:

Results group structure:
    results/
    ├── cam_vert1_param_roi              ← Processing parameters (for consistency check)
    ├── cam_vert1_param_alpha
    ├── cam_vert1_param_chi_sat
    ├── ... (all get_processing_params)
    ├── cam_vert1_PTAI_m1_cnt_rel_atoms_back    ← Quality metrics (for rejection)
    ├── cam_vert1_PTAI_m1_cnt_rel_atoms_sat
    ├── cam_vert1_PTAI_m1_probe_norm_err
    ├── cam_vert1_dmd_profile_updated   ← DMD status (NEW)
    ├── cam_vert1_PTAI_m1_n1D_x         ← Analysis results
    ├── cam_vert1_PTAI_m1_n1D_y
    └── ... (other results)

✓ All required fields already saved by show_ODs_v2


TESTING RECOMMENDATIONS
========================

1. Test parameter consistency check:
   - Create 2-3 test shots with different alpha values
   - Run show_ODs_v2 on each
   - Call get_and_filter_shots() → Should detect inconsistency

2. Test quality rejection:
   - Manually modify HDF5 to set saturation=0.1 for one shot
   - Call get_and_filter_shots() → Should reject that shot
   - Verify rejection statistics

3. Test DMD check:
   - Set ENABLE_DMD_CHECK = False and re-run → Different results
   - Verify output differences


TROUBLESHOOTING
===============

Issue: "No images found for cam_vert1"
→ show_ODs_v2 didn't process that camera
→ Check camera_settings for typos in camera_name

Issue: "Could not extract parameters"
→ show_ODs_v2 didn't save results
→ Verify show_ODs_v2 ran successfully and saved results

Issue: "Parameter inconsistency detected" on first run
→ Likely each shot was processed with different parameters
→ Re-run show_ODs_v2 with consistent camera_settings
→ Or update parameters manually in HDF5 (not recommended)

Issue: Too many shots rejected
→ Check config.py thresholds - may be too strict
→ Adjust SATURATION_THRESHOLD, BACKGROUND_CONTAMINATION_THRESHOLD, etc.


FUTURE ENHANCEMENTS
===================

Potential improvements:
- Auto-generate parameter consistency reports (which params vary most)
- Per-image quality statistics (show which images fail most often)
- Automatic threshold optimization based on statistics
- Caching of parameter values for faster repeated runs
- Integration with Lyse database queries
- Support for filtering by custom HDF5 paths
- Export rejection details to CSV for further analysis
"""
