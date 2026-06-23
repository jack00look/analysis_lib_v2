"""Configuration for the laptop-style KZ moving-window analysis."""

from waterfall_v2.mode_configs.KZ_det_scan import MODE_CONFIG as _BASE_MODE_CONFIG


MODE_CONFIG = {
    **_BASE_MODE_CONFIG,
    'mode_name': 'KZ_window_moving',
    'apply_ntot_check': False,
    'moving_window_enabled': True,
    'moving_window_width_um': 50.0,
    'moving_window_step_um': 5.0,
    'moving_window_mag_abs_threshold': 0.1,
    'moving_window_guide_linestyle': '-',
    'moving_window_guide_linewidth': 2.8,
    'moving_window_guide_alpha': 1.0,
}
