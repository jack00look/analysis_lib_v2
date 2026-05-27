"""Configuration for KZ_det_scan mode"""

MODE_CONFIG = {
    'scan': 'KZ_det_scan',
    'data_origin': 'show_ODs_v2',
    'magnetization_modality': 'vert1_two_component',
    'constraints': None,
    'average': False,
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
        'all_shots_waterfall': False,
    },
    'globals_in_title': [],
    'domain_balance_fit_window': 0.6,
    
    # Defect analysis configuration
    'defect_analysis': {
        'gaussian_sigma': 0.3,
        'derivative_threshold': 0.0,
        'derivative_threshold_pos': 0.05,
        'derivative_threshold_neg': -0.05,
        'threshold_scan': False,
        'zero_crossing_min_distance_um': 5.0,
        'peak_step_filter_enabled': True,
        'peak_step_filter_window_px': 4,
        'peak_step_filter_min_abs_delta_m': 0.25,
        'min_same_sign_peak_distance_um': 6.0,
        'missed_defect_correction_method': 'opposite_peak',
    },
}
