"""Configuration for bubbles_evolution mode"""

MODE_CONFIG = {
    'scan': 'bubbles_evolution',
    'data_origin': 'show_ODs_v2',
    'magnetization_modality': 'vert1_two_component',
    'constraints': None,
    'average': True,
    'plots': {
        'main_waterfall': True,
        'fluctuations_waterfall': False,
        'sectioned_sigmoid': False,
        'avg_density_profile': False,
        'avg_magnetization_profile': False,
        'corr_sigmoid_density': False,
        'corr_mag_density': False,
        'evolution_plots': False,
        'used_region_density_fluctuations': False,
        'all_shots_waterfall': True,  # Set to True to show all individual shots with domain wall detection
    },
    'globals_in_title': [],
    # Domain wall velocity fit range (starting and ending points for linear fit)
    'dw_fit_x_start': None,  # Will be set by analysis, starting x position (μm)
    'dw_fit_x_end': None,    # Will be set by analysis, ending x position (μm)
    'dw_fit_t_start': 1.,  # Will be set by analysis, starting time (ms)
    'dw_fit_t_end': 81.,    # Will be set by analysis, ending time (ms)
}
