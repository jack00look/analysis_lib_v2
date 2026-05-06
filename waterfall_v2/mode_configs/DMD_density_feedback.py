"""Configuration for DMD_density_feedback mode"""

MODE_CONFIG = {
    'scan': 'DMD_density_feedback',
    'data_origin': 'show_ODs_v2',
    'magnetization_modality': 'vert1_two_component',
    'constraints': None,
    'average': False,
    'plots': {
        'main_waterfall': False,
        'fluctuations_waterfall': True,
        'sectioned_sigmoid': False,
        'avg_density_profile': False,
        'avg_magnetization_profile': False,
        'corr_sigmoid_density': False,
        'corr_mag_density': False,
        'evolution_plots': False,
        'used_region_density_fluctuations': True,
    },
    'globals_in_title': [],
    'shot_name_filter': None,
    'density_fluct_y_label': None,
    'scan_variable_override': 'rep',
    'skip_feedback_filter': True,
}
