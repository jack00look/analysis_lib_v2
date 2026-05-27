"""Configuration for ARP_Backward mode"""

MODE_CONFIG = {
    'scan': 'ARP_Backward',
    'data_origin': 'show_ODs_v2',
    'magnetization_modality': 'vert1_two_component',
    'constraints': None,  # Constrain to Rabi frequency of 800 Hz
    'average': True,
    'plots': {
        'main_waterfall': True,
        'fluctuations_waterfall': False,
        'sectioned_sigmoid': True,
        'avg_density_profile': False,
        'avg_magnetization_profile': False,
        'corr_sigmoid_density': False,
        'corr_mag_density': False,
        'evolution_plots': False,
        'used_region_density_fluctuations': False,
        'all_shots_waterfall': False,
    },
    'globals_in_title': [
        ('ARP_omega_rabi', 'Hz'),
        ('ARP_ramp_speed', 'mG/ms'),
        ('ARPB_initial_set_field', 'mG'),
    ],
}
