"""
Main waterfall_v2 configuration - User settings

Set ACTIVE_MODE, SEQS, and HDF_CONFIG here.
Mode-specific configurations are loaded automatically from mode_configs/ subfolder.
"""

# Set this to pick which mode to run
ACTIVE_MODE = 'bubbles_evolution'
# Available modes:
#   - ARP_Forward, ARP_Forward_Feedback, ARP_Backward
#   - KZ_det_scan, KZ_reps, KZ_defect_analysis
#   - ARPF_reps
#   - DMD_density_feedback
#   - bubbles, bubbles_evolution, bubbles_repeat

# Optional override for sequence indices (set to None to use mode default)
SEQS = [40]

# -------------------------
# HDF Data Configuration
# -------------------------
HDF_CONFIG = {
    'today': False,  # Set to False to load HDF from a previous day
    'year': 2026,
    'month': 5,
    'day': 4,
    'reanalyzed': False,  # Load from data_reanalyzed/ instead of NAS
}
