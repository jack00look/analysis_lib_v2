"""
Main waterfall_v2 configuration - User settings

Set ACTIVE_MODE, SEQS, and HDF_CONFIG here.
Mode-specific configurations are loaded automatically from mode_configs/ subfolder.
"""
#test

# Set this to pick which mode to run
ACTIVE_MODE = 'KZ_moving_window'  # <-- Set your active mode here. Available: 'ARP_Forward', 'ARP_Forward_Feedback', 'ARP_Backward', 'KZ_det_scan', 'KZ_det_scan_window', 'KZ_moving_window', 'KZ_window_moving', 'KZ_det_scan_PHC', 'KZ_moving_window_PHC', 'KZ_window_moving_PHC', 'KZ_reps', 'ARPF_reps', 'DMD_density_feedback', 'bubbles', 'bubbles_evolution', 'bubbles_repeat'
# Available modes:
#   - ARP_Forward, ARP_Forward_Feedback, ARP_Backward
#   - KZ_det_scan, KZ_det_scan_window, KZ_moving_window, KZ_window_moving, KZ_det_scan_PHC, KZ_moving_window_PHC, KZ_window_moving_PHC, KZ_reps, KZ_defect_analysis
#   - ARPF_reps
#   - DMD_density_feedback
#   - bubbles, bubbles_evolution, bubbles_repeat

# Optional override for sequence indices (set to None to use mode default)
SEQS=[51, 52]
#SEQS = [15,16,17,18,19,20,21]  # <-- Set your sequence indices here, or set to None to use defaults from MODE_CONFIGS

# -------------------------
# HDF Data Configuration
# -------------------------
HDF_CONFIG = {
    'today': True,  # Set to False to load HDF from a previous day
    'year': 2026,
    'month': 5,
    'day': 27,
    'reanalyzed': False,  # Load from data_reanalyzed/ instead of NAS
}
