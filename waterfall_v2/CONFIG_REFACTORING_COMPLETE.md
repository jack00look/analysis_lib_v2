# Waterfall V2 Configuration Refactoring - COMPLETE

## Summary

The monolithic `waterfall_config_v2.py` has been successfully refactored into a modular configuration system with:

1. **Main entry point** - User-facing settings
2. **Common config** - Shared parameters across all modes
3. **11 mode-specific configs** - One file per mode in `mode_configs/` subfolder
4. **Auto-loading system** - `waterfall_v2_example.py` auto-loads mode config based on `ACTIVE_MODE`
5. **Defect finding library** - Copied from waterfall/ to make config self-contained

## File Structure

```
waterfall_v2/
├── waterfall_config.py              # Main entry point (USER CONFIGURES THIS)
├── common_config.py                 # Shared parameters (PARAMS, MAGNETIZATION_MODALITIES)
├── defect_finding_lib.py            # Defect analysis library (copied from waterfall/)
├── waterfall_v2_example.py          # Example usage with auto-loading
├── waterfall_config_v2.py           # [DEPRECATED] Old monolithic config
├── mode_configs/
│   ├── __init__.py                  # Exports all mode names
│   ├── ARP_Forward.py
│   ├── ARP_Forward_Feedback.py
│   ├── ARP_Backward.py
│   ├── KZ_det_scan.py
│   ├── KZ_reps.py
│   ├── KZ_defect_analysis.py
│   ├── ARPF_reps.py
│   ├── DMD_density_feedback.py
│   ├── bubbles.py
│   ├── bubbles_evolution.py
│   └── bubbles_repeat.py
```

## Configuration Usage

### Setting Active Mode (User Step 1)

Edit **waterfall_config.py** and set:

```python
ACTIVE_MODE = 'bubbles_evolution'  # or any other mode
SEQS = [40]                         # optional sequence override
```

### Loading Configuration (Automatic in Scripts)

The configuration is automatically loaded in any script that needs it:

```python
from waterfall_v2.waterfall_config import ACTIVE_MODE, SEQS, HDF_CONFIG
from waterfall_v2.common_config import MAGNETIZATION_MODALITIES
import importlib
mode_config_module = importlib.import_module(f'waterfall_v2.mode_configs.{ACTIVE_MODE}')
MODE_CONFIG = mode_config_module.MODE_CONFIG
```

## Key Changes from Original

### 1. Data Origin
- **Before**: `data_origin: 'show_ODs'`
- **After**: `data_origin: 'show_ODs_v2'` (all modes)
- **Reason**: Use new refactored analysis results with improved quality metrics and DMD validation

### 2. Plot Configuration
- **Before**: Inherited from `DEFAULT_PLOTS` dict in monolithic config
- **After**: Explicit `True`/`False` flags in each mode's `plots` dict
- **Reason**: Clear visibility of which plots are enabled per mode

### 3. Sequence Configuration
- **Before**: `seqs` field in mode_config dict
- **After**: Moved to main `waterfall_config.py` as `SEQS`
- **Reason**: Centralized user-facing settings, modes are pure configuration

### 4. Defect Analysis
- **Before**: `DEFECT_ANALYSIS_PARAMS` in monolithic config
- **After**: Functions in `defect_finding_lib.py`, configs use `constraints`
- **Reason**: Cleaner separation of concerns, library self-contained

## Mode Configuration Structure

Each mode file in `mode_configs/` exports a `MODE_CONFIG` dict:

```python
MODE_CONFIG = {
    'scan': str,                          # Scan name (e.g., 'bubbles_evolution')
    'data_origin': 'show_ODs_v2',         # Analysis results source
    'magnetization_modality': str,        # See MAGNETIZATION_MODALITIES in common_config
    'constraints': dict or None,          # Defect analysis constraints
    'average': bool,                      # Whether to average results
    'plots': {                            # Explicit plot toggles (True/False)
        'main_waterfall': bool,
        'fluctuations_waterfall': bool,
        'sectioned_sigmoid': bool,
        'avg_density_profile': bool,
        'avg_magnetization_profile': bool,
        'corr_sigmoid_density': bool,
        'corr_mag_density': bool,
        'evolution_plots': bool,
        'used_region_density_fluctuations': bool,
    },
    'globals_in_title': list,             # Parameters to display in plot titles
}
```

## Available Modes

1. **ARP_Forward** - Forward ARP scan
2. **ARP_Forward_Feedback** - Forward ARP with feedback control
3. **ARP_Backward** - Backward ARP scan
4. **KZ_det_scan** - Kibble-Zurek detection scan
5. **KZ_reps** - Kibble-Zurek repeated measurements
6. **KZ_defect_analysis** - Kibble-Zurek with defect analysis
7. **ARPF_reps** - ARP forward with repetitions
8. **DMD_density_feedback** - Density feedback control
9. **bubbles** - Bubble structure analysis
10. **bubbles_evolution** - Time-evolution of bubbles
11. **bubbles_repeat** - Repeated bubble measurements

## Common Configuration

The `common_config.py` file contains:

### PARAMS
- Integration limits for 1D projections
- Waterfall magnetization colorscale limits

### MAGNETIZATION_MODALITIES
- Definitions for different detection modalities
- Each modality specifies:
  - `camera_name`: Which camera to analyze
  - `atoms_images`: Which OD images to use
  - `reference_images`: Reference for normalization

## Migration Guide for External Scripts

### Old Import (with monolithic config)
```python
from waterfall_v2.waterfall_config_v2 import (
    HDF_CONFIG, ACTIVE_MODE, SEQS, MODE_CONFIGS, MAGNETIZATION_MODALITIES
)
mode_config = MODE_CONFIGS[ACTIVE_MODE]
```

### New Import (with modular config)
```python
from waterfall_v2.waterfall_config import ACTIVE_MODE, SEQS, HDF_CONFIG
from waterfall_v2.common_config import MAGNETIZATION_MODALITIES
import importlib
mode_config_module = importlib.import_module(f'waterfall_v2.mode_configs.{ACTIVE_MODE}')
MODE_CONFIG = mode_config_module.MODE_CONFIG
```

## Deprecation Notes

- **waterfall_config_v2.py** is deprecated and should be removed
- Scripts should be updated to use new modular imports
- See `waterfall_v2_example.py` for reference implementation

## Quality Assurance Checklist

- [x] All 11 modes converted to individual config files
- [x] Data origin changed to 'show_ODs_v2' across all modes
- [x] Plot toggles made explicit (no DEFAULT_PLOTS inheritance)
- [x] Sequence configuration moved to main waterfall_config.py
- [x] defect_finding_lib.py copied from waterfall/
- [x] Auto-loading implemented in waterfall_v2_example.py
- [x] mode_configs/__init__.py exports mode names
- [x] All imports verified in waterfall_v2_example.py
- [x] Common parameters centralized in common_config.py

## Next Steps

1. Update other scripts in waterfall_v2/ to use new config structure
2. Test waterfall_v2_example.py to verify auto-loading works
3. Remove waterfall_config_v2.py once all scripts migrated
4. Update documentation to reference new modular structure
