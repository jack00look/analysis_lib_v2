"""
Configuration for multishot analysis rejection thresholds and parameters.

This file defines the quality criteria used to reject shots during multishot analysis.
"""

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
SATURATION_THRESHOLD = 0.02

# Maximum allowed probe normalization error
# Typical: 0.05 (5%) to 0.15 (15%)
PROBE_NORM_ERROR_THRESHOLD = 0.03

# ============================================================================
# DMD Validation
# ============================================================================
# DMD validation is now performed in show_ODs_v2 and saved as 'show_ODs_v2_dmd_validation_valid'
# Shots are accepted if dmd_validation_valid is NaN (older shots) or True
# Shots are rejected if dmd_validation_valid is False (DMD was loading or load finished too recently)

# ============================================================================
# Logging and Reporting
# ============================================================================
# Print detailed rejection statistics
VERBOSE_REJECTION_STATS = True
