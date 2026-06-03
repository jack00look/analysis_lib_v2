# Domain Extraction for KZ_det_scan_window Mode

## Overview

The domain extraction system extracts and saves domain boundary information for each shot in the KZ_det_scan_window analysis mode. This data includes:

- **Domain boundaries**: x_start and x_end positions (in μm) for each domain
- **Domain signs**: +1 (blue) or -1 (red) indicating magnetization direction
- **Field value**: ARPKZ_final_set_field for each shot
- **Defect counts**: Number of positive and negative domain walls detected

## Components

### 1. `domain_extraction_lib.py`
Main library with functions:

#### `extract_domain_boundaries_and_signs(m_profile, x_axis_um, x_min_um, x_max_um, ...)`
Extracts domain boundaries and normalized signs from a magnetization profile.

**Parameters:**
- `m_profile`: 1D magnetization profile
- `x_axis_um`: x positions in μm
- `x_min_um`, `x_max_um`: integration region bounds
- `gaussian_sigma`: smoothing parameter (default: 0.3)
- `pos_threshold`: positive derivative threshold (default: 0.08)
- `neg_threshold`: negative derivative threshold (default: -0.08)

**Returns:**
- `domain_boundaries`: list of (x_start, x_end) tuples
- `domain_signs`: list of +1/-1 signs (normalized)
- `peaks_pos`: list of positive peak positions
- `peaks_neg`: list of negative peak positions

#### `create_domain_info_per_shot(df, seqs, scan, data_origin, mode_cfg, params)`
Extracts domain info for all shots in a DataFrame.

**Returns:** Dictionary with structure:
```python
{
    shot_index: {
        'field': field_value,
        'domains': [
            {'x_start_um': x1, 'x_end_um': x2, 'sign': +1, 'color': 'blue'},
            ...
        ],
        'n_domains': number_of_domains,
        'defect_count': total_defects,
        'n_pos_defects': positive_defects,
        'n_neg_defects': negative_defects
    },
    ...
}
```

#### `save_domain_info_to_csv(domain_info_dict, output_dir, filename)`
Saves domain info to CSV file (one row per shot).

#### `save_domain_info_to_json(domain_info_dict, output_dir, filename)`
Saves complete domain info to JSON file with full details.

### 2. `extract_domain_info.py`
Example script showing how to use the domain extraction system.

**Usage:**
```bash
cd /home/rick/labscript-suite/userlib/analysislib/analysislib_v2/waterfall_v2
python extract_domain_info.py
```

## Algorithm

For each shot:

1. **Smooth magnetization** using Gaussian filter (sigma from config)
2. **Compute derivative** to find domain walls
3. **Detect peaks** in derivative above/below thresholds
4. **Create domain boundaries** between peaks
5. **Assign raw signs** based on peak locations
6. **Normalize signs** so +1 = group with higher average magnetization
7. **Store with field value** for analysis

## Output Files

### CSV Format (`domain_info.csv`)
```
shot_index,field,n_domains,defect_count,n_pos_defects,n_neg_defects,domain_json
0,0.001234,3,5,2,3,"[{""x_start_um"": 0, ""x_end_um"": 20, ...}, ...]"
1,0.001456,2,4,2,2,"[{""x_start_um"": 0, ""x_end_um"": 25, ...}, ...]"
```

### JSON Format (`domain_info.json`)
```json
{
  "0": {
    "field": 0.001234,
    "domains": [
      {
        "x_start_um": 0.0,
        "x_end_um": 20.5,
        "sign": 1,
        "color": "blue"
      },
      {
        "x_start_um": 20.5,
        "x_end_um": 45.3,
        "sign": -1,
        "color": "red"
      }
    ],
    "n_domains": 3,
    "defect_count": 5,
    "n_pos_defects": 2,
    "n_neg_defects": 3
  },
  "1": { ... }
}
```

## Configuration

Domain extraction parameters come from:

1. **Mode config** (`KZ_det_scan_window.py`):
   ```python
   'defect_analysis': {
       'gaussian_sigma': 0.3,
       'derivative_threshold_pos': 0.08,
       'derivative_threshold_neg': -0.08,
       ...
   }
   ```

2. **Common params** (`PARAMS`):
   ```python
   'DOMAIN_WALL_X_MIN': x_min_um
   'DOMAIN_WALL_X_MAX': x_max_um
   'UM_PER_PX': um_per_px
   ```

## Usage Examples

### Extract and save domain info:
```python
from waterfall.domain_extraction_lib import (
    create_domain_info_per_shot,
    save_domain_info_to_csv,
    save_domain_info_to_json
)

domain_info = create_domain_info_per_shot(df, seqs, 'KZ_det_scan_window', 'show_ODs_v2', MODE_CONFIG, PARAMS)
save_domain_info_to_csv(domain_info, output_dir='./results')
save_domain_info_to_json(domain_info, output_dir='./results')
```

### Extract domain info for single shot:
```python
from waterfall.domain_extraction_lib import extract_domain_boundaries_and_signs

domains, signs, peaks_pos, peaks_neg = extract_domain_boundaries_and_signs(
    m_profile, 
    x_axis_um, 
    x_min_um=0, 
    x_max_um=100,
    gaussian_sigma=0.3,
    pos_threshold=0.08,
    neg_threshold=-0.08
)
```

### Load saved domain info:
```python
from waterfall.domain_extraction_lib import load_domain_info_from_json

domain_info = load_domain_info_from_json('./results/domain_info.json')
```

## Data Structure

Each shot's domain info contains:

| Field | Type | Description |
|-------|------|-------------|
| `field` | float | ARPKZ_final_set_field value |
| `domains` | list | List of domain dicts (see below) |
| `n_domains` | int | Number of domains detected |
| `defect_count` | int | Total defects (pos + neg) |
| `n_pos_defects` | int | Positive domain walls detected |
| `n_neg_defects` | int | Negative domain walls detected |

Each domain dict contains:

| Field | Type | Description |
|-------|------|-------------|
| `x_start_um` | float | Left boundary in μm |
| `x_end_um` | float | Right boundary in μm |
| `sign` | int | +1 (blue) or -1 (red) |
| `color` | str | Color name for visualization |

## Further Analysis

The saved domain information can be used for:

1. **Domain dynamics analysis**: Track how domains evolve with field
2. **Defect statistics**: Analyze defect density and location per field value
3. **Domain size distribution**: Extract domain widths and their statistics
4. **Nucleation/annihilation events**: Detect changes in domain structure
5. **Windowed analysis**: Apply window_size and window_step for local statistics

## Output Directory

Domain info is saved to:
```
./results/domain_info/
├── domain_info.csv
└── domain_info.json
```

Change `output_dir` parameter to save elsewhere.
