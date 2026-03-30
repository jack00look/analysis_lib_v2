import numpy as np

# Set this to pick what to run in waterfall_FLAT_v4.py
ACTIVE_MODE = 'ARP_Backward'  # <-- Set your active mode here. Available: 'ARP_Forward', 'ARP_Forward_Feedback', 'ARP_Backward', 'KZ_det_scan', 'KZ_reps', 'ARPF_reps', 'DMD_density_feedback', 'bubbles', 'bubbles_evolution', 'bubbles_repeat'
# Optional override for sequence indices used by ACTIVE_MODE (set to None to use mode default)

# If True, use recommended_center from the latest
# results/kz_param_stability_safe_box_*.json.
RECOMMENDED_CENTER = False

SEQS = [3,4,5,6,7]  # <-- Set your sequence indices here, or set to None to use defaults from MODE_CONFIGS
#SEQS=[33,34,35,36,37,38]
#47,48,49,50,51,52
# -------------------------
# HDF Data Configuration
# -------------------------
HDF_CONFIG = {
    'today': True,  # Set to False to load HDF from a previous day
    'year': 2026,   # Set if today=False (e.g., 2026)
    'month': 3,  # Set if today=False (e.g., 3 for March)
    'day': 24,    # Set if today=False (e.g., 17)
}

# -------------------------
# Shared numeric parameters
# -------------------------
PARAMS = {
    # Camera used to convert x axis from pixels to microns using camera_settings.cameras[CAMERA_NAME]['px_size']
    'CAMERA_NAME': 'cam_vert1',
    'SIGMA_Z_LOCAL_AVG': 3,
    'SIGMA_Z_LOCAL_FLUCT': 2,
    'AUTOCORR_CENTER': 1085,
    'AUTOCORR_WINDOW': 20,
    # Integration window limits are in micrometers (um)
    'X_MIN_INTEGRATION': 905,
    'X_MAX_INTEGRATION': 1205,
    'NUM_SECTIONS': 300,
    'BACK_CHECK_THRESHOLD': 0.05,
    'NTOT_CHECK_THRESHOLD': 7e5,
    # Set to None for autoscale
    'WATERFALL_MAG_CLIM': (-.2, .4),
    'WATERFALL_DENSITY_CLIM': None,
}

# -------------------------
# Plot toggles per mode
# -------------------------
DEFAULT_PLOTS = {
    'main_waterfall': True,
    'fluctuations_waterfall': True,
    'sectioned_sigmoid': True,
    'avg_density_profile': False,
    'avg_magnetization_profile': True,
    'corr_sigmoid_density': False,
    'corr_mag_density': True,
    'evolution_plots': False,
}


MODE_CONFIGS = {
    'ARP_Forward': {
        'scan': 'ARP_Forward',
        'seqs': [18,19, 20, 22],
        'data_origin': 'show_ODs',
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
        'data_origin': 'show_ODs',
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
        'constraints': None,
        'average': True,
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
        'constraints': None,
        'average': False,
        'plots': {
            **DEFAULT_PLOTS,
            'sectioned_sigmoid': False,
            'corr_sigmoid_density': False,
        },
        'globals_in_title': [],
    },
    'bubbles': {
        'scan': 'bubbles',
        'seqs': [1],
        'data_origin': 'show_ODs',
        'constraints': None,
        'average': True,
        'plots': {
            **DEFAULT_PLOTS,
            'sectioned_sigmoid': False,
            'corr_sigmoid_density': False,
            'evolution_plots': False,
        },
        'globals_in_title': [],
    },
    'bubbles_evolution': {
        'scan': 'bubbles_evolution',
        'seqs': [1],
        'data_origin': 'show_ODs',
        'constraints': None,
        'average': True,
        'plots': {
            **DEFAULT_PLOTS,
            'sectioned_sigmoid': False,
            'corr_sigmoid_density': False,
            'evolution_plots': True,
        },
        'globals_in_title': [],
    },
    'bubbles_repeat': {
        'scan': 'bubbles_repeat',
        'seqs': [1],
        'data_origin': 'show_ODs',
        'constraints': {'bubbles_time': [10]},
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
        'defect_analysis': {
            # 1) Signal preprocessing
            'gaussian_sigma': 0.3,                 # Gaussian smoothing sigma (pixels)

            # 2) Peak detection on dM/dx
            'derivative_threshold': 0.0,           # Backward-compatible base magnitude
            'derivative_threshold_pos': 0.055,     # +dM/dx threshold for positive peaks
            'derivative_threshold_neg': -0.035,    # -dM/dx threshold for negative peaks

            # 3) Post-threshold peak quality filter (on magnetization step)
            # Require |mean(m right N px) - mean(m left N px)| >= min_abs_delta_m
            # Rejected peaks are marked with black crosses and not counted.
            'peak_step_filter_enabled': True,
            'peak_step_filter_window_px': 4,
            'peak_step_filter_min_abs_delta_m': 0.1,

            # 4) Geometric cleanup and correction
            'min_same_sign_peak_distance_um': 7.0,  # merge same-sign peaks closer than this
            'zero_crossing_min_distance_um': 3.0,   # min distance from kept threshold peaks for corrected defects

            # 5) Histogram control
            'defect_position_hist_half_window_um': 4.0,  # counts within ±window around 1-um centers
        },
        'constraints': {
            'ARPKZ_final_set_field': 130.395,  # Set to desired value (e.g., 130.429)
            'ARPKZ_omega_ramp_time': 4.,  # Set to desired value
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


