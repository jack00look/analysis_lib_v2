"""
Common waterfall_v2 configuration - Shared parameters across all modes
"""

# -------------------------
# Shared numeric parameters
# -------------------------
PARAMS = {
    'SIGMA_Z_LOCAL_AVG': 1.,
    'SIGMA_Z_LOCAL_FLUCT': 1.,
    'NUM_SECTIONS': 300,
    'WATERFALL_DENSITY_CLIM': None,
    # Domain wall velocity analysis
    'DOMAIN_WALL_X_MIN': 910,
    'DOMAIN_WALL_X_MAX': 1350,
    # Sectioned sigmoid fit region (in micrometers)
    'SIGMOID_FIT_X_MIN': 1018,
    'SIGMOID_FIT_X_MAX': 1037,
    # Error profile parameters for DMD feedback visualization
    'KP_SIGMOID': 0.5,              # Gain for sigmoid-based error profile
    'SMOOTHING_SIGMA_SIGMOID': 3.0, # Smoothing for sigmoid error profile
    'KP_DENSITY': 0.5e-10,              # Gain for density-based error profile
    'SMOOTHING_SIGMA_DENSITY': 3.0, # Smoothing for density error profile
}

# -------------------------
# Magnetization extraction modalities
# -------------------------
MAGNETIZATION_MODALITIES = {
    # Modality 1: standard two-component magnetization from cam_vert1
    'vert1_two_component': {
        'camera_name': 'cam_vert1',
        'type': 'two_component',
        'data_origin': 'show_ODs_v2',
        'atoms_images': ['PTAI_m1', 'PTAI_m2'],
        'apply_ntot_check': False,
        'affine_correction': {
            'a1': 1./1.74,
            'b1': 0.0,
            'c1': -1.31e9/1.74,
            'a2': 1./1.55,
            'b2': 0.0,
            'c2': -1.17e9/1.55,
        },
        'm1_column': None,
        'm2_column': None,
        # Integration window for vert1 (in micrometers)
        'x_min_integration': 950,
        'x_max_integration': 1150,
        # Waterfall magnetization color limits (set to None for autoscale)
        'waterfall_mag_clim': (-1.0, 1.0),
    },
    # Modality 2: PHC multiple magnetization profiles from cam_vert2_PHC
    'vert2_phc_multicomponent': {
        'camera_name': 'cam_vert2_PHC',
        'type': 'phc_profiles',
        'data_origin': 'show_ODs_v2',
        'atoms_images': ['PHC_m1_1'],
        'apply_ntot_check': False,
        # Integration window for vert2_PHC (in micrometers)
        'x_min_integration': 1000,
        'x_max_integration': 1250,
        # Waterfall magnetization color limits (set to None for autoscale)
        'waterfall_mag_clim': (3.64e-6, 8.91e-6) #(6.5e-6, 1.324e-5)#(5.15e-6, 1.024e-5)#(.4e-5, 1.6e-5),
    },
}

# -------------------------
# Shot quality filtering configuration
# -------------------------
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

# -------------------------
# Global defect-finding algorithm parameters
# -------------------------
DEFECT_ANALYSIS_PARAMS = {
    # 1) Signal preprocessing
    'gaussian_sigma': 2,                 # Gaussian smoothing sigma (pixels)

    # 2) Peak detection on dM/dx
    'derivative_threshold': 0.4e-6,           # Backward-compatible base magnitude
    'derivative_threshold_pos': 0.4e-6,       # +dM/dx threshold for positive peaks
    'derivative_threshold_neg': -0.4e-6,      # -dM/dx threshold for negative peaks

    # 3) Post-threshold peak quality filter (on magnetization step)
    'peak_step_filter_enabled': True,
    'peak_step_filter_window_px': 4,
    'peak_step_filter_min_abs_delta_m': 0.5,

    # 4) Geometric cleanup and correction
    'min_same_sign_peak_distance_um': 10.0,
    'zero_crossing_min_distance_um': 3.0,

    # 5) Histogram control
    'defect_position_hist_half_window_um': 4.0,
    'defect_size_hist_bin_size': 5.,
}
