"""
Common waterfall_v2 configuration - Shared parameters across all modes
"""

# -------------------------
# Shared numeric parameters
# -------------------------
PARAMS = {
    'SIGMA_Z_LOCAL_AVG': 1.,
    'SIGMA_Z_LOCAL_FLUCT': 1.,
    # Integration window limits are in micrometers (um)
    'X_MIN_INTEGRATION': 950,
    'X_MAX_INTEGRATION': 1150,
    'NUM_SECTIONS': 100,
    # Set to None for autoscale
    'WATERFALL_MAG_CLIM': (-1., 1.),
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
            'a1': 1.,
            'b1': 0.0,
            'c1': 0.,
            'a2': 1.,
            'b2': 0.0,
            'c2': 0.,
        },
        'm1_column': None,
        'm2_column': None,
    },
    # Modality 2: PHC multiple magnetization profiles from cam_vert2_PHC
    'vert2_phc_multicomponent': {
        'camera_name': 'cam_vert2_PHC',
        'type': 'phc_profiles',
        'data_origin': 'show_ODs_v2',
        'atoms_images': ['PHC_1', 'PHC_2', 'PHC_3', 'PHC_4'],
        'apply_ntot_check': False,
    },
}
