import numpy as np

# Set this to pick what to run in waterfall_FLAT_v4.py
ACTIVE_MODE = 'bubbles_repeat'  # <-- Set your active mode here. Available: 'ARP_Forward', 'ARP_Forward_Feedback', 'ARP_Backward', 'KZ_det_scan', 'KZ_reps', 'ARPF_reps', 'DMD_density_feedback', 'bubbles', 'bubbles_evolution', 'bubbles_repeat'
# Optional override for sequence indices used by ACTIVE_MODE (set to None to use mode default)
SEQS = [41]

# -------------------------
# Shared numeric parameters
# -------------------------
PARAMS = {
    'SIGMA_Z_LOCAL_AVG': 3,
    'SIGMA_Z_LOCAL_FLUCT': 2,
    'AUTOCORR_CENTER': 1085,
    'AUTOCORR_WINDOW': 20,
    'X_MIN_INTEGRATION': 890,
    'X_MAX_INTEGRATION': 1150,
    'NUM_SECTIONS': 250,
    'BACK_CHECK_THRESHOLD': 0.05,
    'NTOT_CHECK_THRESHOLD': 7e5,
    # Set to None for autoscale
    'WATERFALL_MAG_CLIM': (-1., 1.),
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
    },
    'KZ_reps': {
        'scan': 'ARP_KZ_reps',
        'seqs': [50],
        'data_origin': 'show_ODs',
        'constraints': {'ARPKZ_final_set_field': 130.43},
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
}


