import numpy as np

# Set this to pick what to run in waterfall_FLAT_v4.py
ACTIVE_MODE = 'ARP_Backward'  # <-- Set your active mode here. Available: 'ARP_Forward', 'ARP_Forward_Feedback', 'ARP_Backward', 'KZ_det_scan', 'KZ_reps', 'ARPF_reps', 'DMD_density_feedback', 'bubbles', 'bubbles_evolution', 'bubbles_repeat'
# Optional override for sequence indices used by ACTIVE_MODE (set to None to use mode default)

# If True, use recommended_center from the latest
# results/kz_param_stability_safe_box_*.json.
RECOMMENDED_CENTER = False
#SEQS = [28]
SEQS = [23]  # <-- Set your sequence indices here, or set to None to use defaults from MODE_CONFIGS
#EQS = [41]
# -------------------------
# HDF Data Configuration
# -------------------------
HDF_CONFIG = {
    'today': True,  # Set to False to load HDF from a previous day
    'year': 2026,   # Set if today=False (e.g., 2026)
    'month': 5,  # Set if today=False (e.g., 3 for March)
    'day': 12,    # Set if today=False (e.g., 17)
}

# -------------------------
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
    'WATERFALL_MAG_CLIM': (-1., 1.),
    'WATERFALL_DENSITY_CLIM': None,
    # Domain wall velocity analysis (bubbles_evolution mode)
    # Set X range for linear plot of sigmoid center time vs position
    'DOMAIN_WALL_X_MIN': 910,  # Set to um value to enable (e.g., 950)
    'DOMAIN_WALL_X_MAX': 1150,  # Set to um value to enable (e.g., 1150)
    # Sectioned sigmoid fit region (in micrometers)
    'SIGMOID_FIT_X_MIN': 1018,
    'SIGMOID_FIT_X_MAX': 1037,
    # Error profile parameters for DMD feedback visualization
    'KP_SIGMOID': 0.5,              # Gain for sigmoid-based error profile
    'SMOOTHING_SIGMA_SIGMOID': 2.0, # Smoothing for sigmoid error profile
    'KP_DENSITY': 0.3,              # Gain for density-based error profile
    'SMOOTHING_SIGMA_DENSITY': 3.0, # Smoothing for density error profile
}

# -------------------------
# Shot quality filtering (bad-shot rejection)
# -------------------------
# All thresholds are applied on the show_ODs metrics:
#   *_cnt_rel_atoms_back, *_cnt_rel_probe_back,
#   *_cnt_rel_atoms_sat,  *_cnt_rel_probe_sat,
#   *_probe_norm_err,     *_N_atoms
SHOT_FILTER_CONFIG = {
    'enabled': True,
    'apply_back_check': True,
    'apply_sat_check': True,
    'apply_probe_norm_err_check': True,
    'back_threshold': 0.001,
    'sat_threshold': 1.,
    'probe_norm_err_threshold': 0.1,
    # Used only when modality enables Ntot filtering
    'ntot_threshold': 2e6,
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
            'a1': 1.,#1./2.8,#1./2.4,#1.43/1.07,
            'b1': 0.0,
            'c1': 0.,#-1.6/2.8*1e9,#-1.7*(1./2.4)*1e9,#-0.3E8*1.43/1.07,#-0.47e8,
            'a2': 1.,#1./1.5,#1./0.9,#0.65/0.38,
            'b2': 0.0,
            'c2': 0.,#-2.5/1.5*1e9#-2.5*(1./0.9)*1e9,#-0.7E8,#-0.76e8*0.65/0.38,
        },
        # Optional explicit labels (if None, library auto-detects columns)
        'm1_column': None,
        'm2_column': None,
    },
    # Modality 2: PHC multiple magnetization profiles from cam_vert2_PHC.
    # All atoms_images correspond to the same state basis; multiple images mean
    # multiple magnetization profiles (no mF weights are applied).
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
    # Require |mean(m right N px) - mean(m left N px)| >= min_abs_delta_m
    # Rejected peaks are marked with black crosses and not counted.
    'peak_step_filter_enabled': True,
    'peak_step_filter_window_px': 4,
    'peak_step_filter_min_abs_delta_m': 0.4,

    # 4) Geometric cleanup and correction
    'min_same_sign_peak_distance_um': 10.0,  # merge same-sign peaks closer than this
    'zero_crossing_min_distance_um': 3.0,   # min distance from kept threshold peaks for corrected defects

    # 5) Histogram control
    'defect_position_hist_half_window_um': 4.0,  # counts within ±window around 1-um centers
    'defect_size_hist_bin_size': 5.,  # 'sqrt', 'fd', 'sturges', int, or float for bin width
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
    'all_shots_waterfall': True,  # Shows all individual shots with separator lines when scan variable changes
}


MODE_CONFIGS = {
    'ARP_Forward': {
        'scan': 'ARP_Forward',
        'seqs': [18,19, 20, 22],
        'data_origin': 'show_ODs_v2',
        'magnetization_modality': 'vert1_two_component',
        'constraints': None,
        'average': True,
        'plots': {**DEFAULT_PLOTS},
        'globals_in_title': [
            ('ARP_omega_rabi', 'Hz'),
            ('ARP_ramp_speed', 'mG/ms'),
            ('ARPF_initial_set_field', 'mG'),
        ],
    },
    'ARP_Forward_Feedback': {
        'scan': 'ARPF_feedback',
        'seqs': [22],
        'data_origin': 'show_ODs',
        'magnetization_modality': 'vert1_two_component',
        'constraints': None,
        'average': False,
        'plots': {**DEFAULT_PLOTS},
        'globals_in_title': [
            ('ARP_omega_rabi', 'Hz'),
            ('ARP_ramp_speed', 'mG/ms'),
            ('ARPF_initial_set_field', 'mG'),
            ('ARP_final_set_field', 'mG'),
        ],
    },
    'ARP_Backward': {
        'scan': 'ARP_Backward',
        'seqs': [48],
        'data_origin': 'show_ODs_v2',
        'magnetization_modality': 'vert1_two_component',
        'constraints': None,
        'average': True,
        'plots': {**DEFAULT_PLOTS},
        'globals_in_title': [
            ('ARP_omega_rabi', 'Hz'),
            ('ARP_ramp_speed', 'mG/ms'),
            ('ARPB_initial_set_field', 'mG'),
        ],
    },
    'KZ_det_scan': {
        'scan': 'KZ_det_scan',
        'seqs': [17, 18, 19],
        'data_origin': 'show_ODs',
        'magnetization_modality': 'vert1_two_component',
        'constraints': None,
        'average': False,
        'plots': {
            **DEFAULT_PLOTS,
            'fluctuations_waterfall': False,
            'sectioned_sigmoid': False,
            'corr_sigmoid_density': False,
            'corr_mag_density': False,
        },
        'globals_in_title': [],
        'domain_balance_fit_window': 0.5,  # Domain balance range for constrained fit: points with |balance| <= this value
    },
    'KZ_reps': {
        'scan': 'ARP_KZ_reps',
        'seqs': [50],
        'data_origin': 'show_ODs',
        'magnetization_modality': 'vert1_two_component',
        'constraints': {'ARPKZ_final_set_field': 130.429},
        'average': False,
        'plots': {
            **DEFAULT_PLOTS,
            'sectioned_sigmoid': False,
            'corr_sigmoid_density': False,
        },
        'globals_in_title': [],
    },
    'ARPF_reps': {
        'scan': 'ARPF_reps',
        'seqs': [23],
        'data_origin': 'show_ODs',
        'magnetization_modality': 'vert1_two_component',
        'constraints': None,
        'average': False,
        'plots': {
            **DEFAULT_PLOTS,
            'sectioned_sigmoid': False,
            'corr_sigmoid_density': False,
        },
        'globals_in_title': [
            ('ARP_omega_rabi', 'Hz'),
            ('ARP_ramp_speed', 'mG/ms'),
            ('ARPF_initial_set_field', 'mG'),
        ],
    },
    'DMD_density_feedback': {
        'scan': 'DMD_density_feedback',
        'seqs': [26],
        'data_origin': 'show_ODs',
        'magnetization_modality': 'vert1_two_component',
        'constraints': None,
        'average': False,
        'plots': {
            **DEFAULT_PLOTS,
            'main_waterfall': False,
            'sectioned_sigmoid': False,
            'corr_sigmoid_density': False,
            'used_region_density_fluctuations': True,
        },
        'globals_in_title': [],
        # Optional: Set to filter shots by name (e.g., 'Do_BEC_DMD_feedback')
        'shot_name_filter': None,
        # Optional: Custom y-axis label for density fluctuations plot (default uses scan variable)
        'density_fluct_y_label': None,
        # Optional: Override the scan variable column (e.g., use 'run repeat' instead of default)
        'scan_variable_override': 'rep',
        # Skip feedback filtering if only using density fluctuations plot
        'skip_feedback_filter': True,
    },
    'bubbles': {
        'scan': 'bubbles',
        'seqs': [1],
        'data_origin': 'show_ODs',
        'magnetization_modality': 'vert1_two_component',
        'constraints': None,
        'average': False,
        'plots': {
            **DEFAULT_PLOTS,
            'sectioned_sigmoid': True,
            'corr_sigmoid_density': False,
            'evolution_plots': False,
        },
        'globals_in_title': [],
    },
    'bubbles_evolution': {
        'scan': 'bubbles_evolution',
        'seqs': [1],
        'data_origin': 'show_ODs_v2',
        'magnetization_modality': 'vert1_two_component',
        'constraints': None,
        'average': True,
        'plots': {
            **DEFAULT_PLOTS,
            'sectioned_sigmoid': False,
            'corr_sigmoid_density': False,
            'evolution_plots': True,
            'all_shots_waterfall': True,  # Show all individual shots with separator lines for each bubbles_time
        },
        'globals_in_title': [],
        'scan_variable_override': 'bubbles_time',  # Column name for the x-axis variable in evolution plots
    },
    'bubbles_repeat': {
        'scan': 'bubbles_repeat',
        'seqs': [1.5],
        'data_origin': 'show_ODs',
        'magnetization_modality': 'vert1_two_component',
        'constraints': {'bubbles_time': [1.5]},
        'average': False,
        'plots': {
            **DEFAULT_PLOTS,
            'sectioned_sigmoid': False,
            'corr_sigmoid_density': False,
            'evolution_plots': False,
        },
        'globals_in_title': [
            ('ARP_omega_rabi', 'Hz'),
            ('ARP_ramp_speed', 'mG/ms'),
            ('ARPB_initial_set_field', 'mG'),
            ('bubbles_time', 'Hz'),
        ],
    },
    'KZ_defect_analysis': {
        'scan': 'KZ_defect_analysis',
        'seqs': [],  # Set your sequence indices
        'data_origin': 'show_ODs',
        'magnetization_modality': 'vert1_two_component',
        'constraints': {
            'ARPKZ_final_set_field': 130.40,  # Set to desired value (e.g., 130.429)
            'ARPKZ_omega_ramp_time': 16.,  # Set to desired value
            'ARPKZ_waiting_time': 10.,  # Set to desired value
        },
        'average': False,
        'plots': {
            **DEFAULT_PLOTS,
            'sectioned_sigmoid': False,
            'corr_sigmoid_density': False,
        },
        'globals_in_title': [
            ('ARPKZ_final_set_field', 'mG'),
            ('ARPKZ_omega_ramp_time', 'ms'),
            ('ARPKZ_waiting_time', 'ms'),
        ],
    },
}