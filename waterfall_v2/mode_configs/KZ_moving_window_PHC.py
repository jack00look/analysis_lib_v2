"""Compatibility spelling for the PHC KZ moving-window analysis."""

from waterfall_v2.mode_configs.KZ_window_moving_PHC import MODE_CONFIG as _BASE_MODE_CONFIG


MODE_CONFIG = {
    **_BASE_MODE_CONFIG,
    'mode_name': 'KZ_moving_window_PHC',
}
