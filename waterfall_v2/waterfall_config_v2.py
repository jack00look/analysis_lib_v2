"""
Waterfall v2 configuration - Mirror of original with updated paths for v2 libraries

Set parameters here to control which analysis to run and what sequences to process.
"""

import numpy as np
import sys
import os

# Add analysislib_v2 to path
sys.path.insert(0, '/home/rick/labscript-suite/userlib/analysislib/analysislib_v2')

# Set this to pick what to run in waterfall_v2 scripts
ACTIVE_MODE = 'bubbles_evolution'  # <-- Set your active mode here

# Optional override for sequence indices used by ACTIVE_MODE (set to None to use mode default)
SEQS = [15,16,17,18,19,20,21]  # <-- Set your sequence indices here, or set to None to use defaults from MODE_CONFIGS

# -------------------------
# HDF Data Configuration
# -------------------------
HDF_CONFIG = {
    'today': False,  # Set to False to load HDF from a previous day
    'year': 2026,   # Set if today=False (e.g., 2026)
    'month': 5,  # Set if today=False (e.g., 5 for May)
    'day': 4,    # Set if today=False (e.g., 4)
    'reanalyzed': True,  # NEW: Set to True to load from data_reanalyzed/ instead of NAS
}
# 
# reanalyzed options:
#   False (default) → Load from NAS: /home/rick/NAS542_dataBEC2/year/month/day/
#   True  → Load from: analysislib_v2/data_reanalyzed/year/month/day/
#
# Use reanalyzed=True to load consolidated HDF files created by run_show_ODs_batch.py
# Shared numeric parameters
# -------------------------
PARAMS = {
    'SIGMA_Z_LOCAL_AVG': 1.,
    'SIGMA_Z_LOCAL_FLUCT': 1.,
    # Integration window limits are in micrometers (um)
    'X_MIN_INTEGRATION': 900,
    'X_MAX_INTEGRATION': 1200,
    'NUM_SECTIONS': 300,
    # Set to None for autoscale
    'WATERFALL_MAG_CLIM': (-.1, 1.),
    'WATERFALL_DENSITY_CLIM': None,
    # Domain wall velocity analysis (bubbles_evolution mode)
    # Set X range for linear plot of sigmoid center time vs position
    'DOMAIN_WALL_X_MIN': 910,  # Set to um value to enable (e.g., 950)
    'DOMAIN_WALL_X_MAX': 1150,  # Set to um value to enable (e.g., 1150)
}

# -------------------------
# Magnetization extraction modalities
# -------------------------
MAGNETIZATION_MODALITIES = {
    # Modality 1: standard two-component magnetization from cam_vert1.
    # Uses m1/m2 1D profiles and applies affine correction coefficients.
    'vert1_two_component': {
        'camera_name': 'cam_vert1',
        'type': 'two_component',
        'data_origin': 'show_ODs',
        'atoms_images': ['PTAI_m1', 'PTAI_m2'],
        'apply_ntot_check': True,
        # Optional affine recomputation of line densities:
        # n1D_m1' = a1*n1D_m1 + b1*n1D_m2 + c1
        # n1D_m2' = a2*n1D_m2 + b2*n1D_m1 + c2
        'affine_correction': {
            'a1': 1.,
            'b1': 0.0,
            'c1': 0.,
            'a2': 1.,
            'b2': 0.0,
            'c2': 0.
        },
        # Optional explicit labels (if None, library auto-detects columns)
        'm1_column': None,
        'm2_column': None,
    },
    # Modality 2: PHC multiple magnetization profiles from cam_vert2_PHC.
    'vert2_phc_multicomponent': {
        'camera_name': 'cam_vert2_PHC',
        'type': 'phc_profiles',
        'data_origin': 'show_ODs',
        'atoms_images': ['PHC_1', 'PHC_2', 'PHC_3', 'PHC_4'],
        'apply_ntot_check': False,
    },
}

# -------------------------
# Global defect-finding algorithm parameters
# -------------------------
DEFECT_ANALYSIS_PARAMS = {
    # 1) Signal preprocessing
    'gaussian_sigma': 1.5,                 # Gaussian smoothing sigma (pixels)

    # 2) Peak detection on dM/dx
    'derivative_threshold': 0.0,           # Backward-compatible base magnitude
    'derivative_threshold_pos': 0.1,       # +dM/dx threshold for positive peaks
    'derivative_threshold_neg': -0.1,      # -dM/dx threshold for negative peaks

    # 3) Post-threshold peak quality filter (on magnetization step)
    'peak_step_filter_enabled': True,
    'peak_step_filter_window_px': 4,
    'peak_step_filter_min_abs_delta_m': 0.4,

    # 4) Geometric cleanup and correction
    'min_same_sign_peak_distance_um': 10.0,
    'zero_crossing_min_distance_um': 3.0,

    # 5) Histogram control
    'defect_position_hist_half_window_um': 4.0,
    'defect_size_hist_bin_size': 5.,
}

# -------------------------
# Plot toggles per mode
# -------------------------
DEFAULT_PLOTS = {
    'main_waterfall': True,
    'fluctuations_waterfall': True,
    'sectioned_sigmoid': True,
    'avg_density_profile': False,
    'avg_magnetization_profile': False,
    'corr_sigmoid_density': False,
    'corr_mag_density': False,
    'evolution_plots': False,
    'used_region_density_fluctuations': False,
}


MODE_CONFIGS = {
    'bubbles_evolution': {
        'scan': 'bubbles_evolution',
        'seqs': [1],
        'data_origin': 'show_ODs',
        'magnetization_modality': 'vert1_two_component',
        'constraints': None,
        'average': True,
        'plots': {
            **DEFAULT_PLOTS,
            'sectioned_sigmoid': True,
            'corr_sigmoid_density': False,
            'evolution_plots': True,
        },
        'globals_in_title': [],
    },
    # ... (other modes can be added here)
}
