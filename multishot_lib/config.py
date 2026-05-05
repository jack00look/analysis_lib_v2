"""
Configuration for multishot analysis rejection thresholds and parameters.

This file defines the quality criteria used to reject shots during multishot analysis.
"""

# ============================================================================
# DMD Profile Update Check
# ============================================================================
# Set to False to disable DMD profile update validation
# NOTE: show_ODs_v2 does not currently save dmd_profile_updated flag,
# so this check will not be performed even if enabled
ENABLE_DMD_CHECK = False

# Time window (seconds) for DMD load to be considered valid relative to shot
# If (load_completion_time - shot_run_time) > this value, shot is rejected
DMD_LOAD_TIME_WINDOW_S = 20.0

# DMD Server connection settings (used if ENABLE_DMD_CHECK is True)
DMD_SERVER_IP = 'localhost'
DMD_SERVER_PORT = 4242

# ============================================================================
# Image Quality Rejection Thresholds
# ============================================================================
# These thresholds are applied to per-image metrics from show_ODs results.
# Format: For image "PTAI_m1" on camera "cam_vert1":
#   - Background contamination: cam_vert1_PTAI_m1_cnt_rel_atoms_back
#   - Saturation: cam_vert1_PTAI_m1_cnt_rel_atoms_sat
#   - Probe normalization error: cam_vert1_PTAI_m1_probe_norm_err

# Maximum allowed background contamination (fraction of atoms in background)
# Typical: 0.001 (0.1%) to 0.01 (1%)
BACKGROUND_CONTAMINATION_THRESHOLD = 0.001

# Maximum allowed saturation in atom image (fraction of atoms in saturated region)
# Typical: 0.05 (5%) to 0.1 (10%)
SATURATION_THRESHOLD = 0.05

# Maximum allowed probe normalization error
# Typical: 0.05 (5%) to 0.15 (15%)
PROBE_NORM_ERROR_THRESHOLD = 0.1

# ============================================================================
# Logging and Reporting
# ============================================================================
# Print detailed rejection statistics
VERBOSE_REJECTION_STATS = True
