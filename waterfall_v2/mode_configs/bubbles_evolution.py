"""Configuration for bubbles_evolution mode"""

MODE_CONFIG = {
    'scan': 'bubbles_evolution',
    'data_origin': 'show_ODs_v2',
    'magnetization_modality': 'vert1_two_component',
    'constraints': None,
    'average': True,
    'plots': {
        'main_waterfall': True,
        'fluctuations_waterfall': True,
        'sectioned_sigmoid': False,
        'avg_density_profile': False,
        'avg_magnetization_profile': False,
        'corr_sigmoid_density': False,
        'corr_mag_density': False,
        'evolution_plots': True,
        'used_region_density_fluctuations': False,
    },
    'globals_in_title': [],
}
