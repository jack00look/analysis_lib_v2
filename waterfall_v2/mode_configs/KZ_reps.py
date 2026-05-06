"""Configuration for KZ_reps mode"""

MODE_CONFIG = {
    'scan': 'ARP_KZ_reps',
    'data_origin': 'show_ODs_v2',
    'magnetization_modality': 'vert1_two_component',
    'constraints': {'ARPKZ_final_set_field': 130.429},
    'average': False,
    'plots': {
        'main_waterfall': True,
        'fluctuations_waterfall': True,
        'sectioned_sigmoid': False,
        'avg_density_profile': False,
        'avg_magnetization_profile': False,
        'corr_sigmoid_density': False,
        'corr_mag_density': False,
        'evolution_plots': False,
        'used_region_density_fluctuations': False,
    },
    'globals_in_title': [],
}
