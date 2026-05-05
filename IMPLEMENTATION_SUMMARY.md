# Implementation Summary: Multishot Analysis System with Reanalysis Support

## What Was Built

A complete multishot analysis pipeline with:
1. **Batch processing** of show_ODs_v2 on multiple sequences
2. **Data consolidation** into focused reanalyzed HDF files
3. **Quality filtering** with parameter consistency checking
4. **Dual data source support** (NAS or reanalyzed)

## Components Created

### 1. run_show_ODs_batch.py
**Purpose**: Batch process raw shot files through show_ODs_v2 analysis

**Features**:
- Interactive date/sequence selection OR scripted configuration
- Processes 100+ files per sequence
- Saves results to each file's HDF5 group
- **NEW**: Also consolidates into single HDF file per day
- Saves to: `analysislib_v2/data_reanalyzed/YYYY/MM/DD/seq_26_seq_30.hdf`

**Usage**:
```bash
python run_show_ODs_batch.py                 # Interactive
python run_show_ODs_for_waterfall.py         # Use waterfall_config_v2
# Or configure SEQS/HDF_CONFIG in the script
```

### 2. run_show_ODs_for_waterfall.py
**Purpose**: Wrapper that automatically uses waterfall_config_v2 settings

**Simplifies**: `python run_show_ODs_for_waterfall.py` (no prompts)

### 3. multishot_lib/main.py - Enhanced
**New Feature**: `reanalyzed` flag in hdf_config

**Before**:
```python
hdf_config = {'today': False, 'year': 2026, 'month': 4, 'day': 15}
df = get_day_data(hdf_config)  # Only NAS
```

**After**:
```python
# Option A: NAS (Lyse results)
hdf_config = {'today': False, 'year': 2026, 'month': 4, 'day': 15, 'reanalyzed': False}

# Option B: Reanalyzed consolidation
hdf_config = {'today': False, 'year': 2026, 'month': 4, 'day': 15, 'reanalyzed': True}

df = get_day_data(hdf_config)  # Loads from correct location
```

### 4. data_reanalyzed/ Directory
**Structure**:
```
analysislib_v2/data_reanalyzed/
├── 2026/04/15/
│   ├── seq_26_seq_30.hdf          # Consolidated from 210 files
│   └── seq_0001_seq_0002.hdf
├── 2026/04/16/
│   └── ...
└── README.md
```

**Contents**: Device params + timing + RF + show_ODs_v2 results only

## Data Flow

### Raw Processing Flow
```
Raw .h5 files (210 per day per sequences)
    ↓ [show_ODs_v2 extracts OD, parameters, metrics]
    ↓
.h5 with results/show_ODs_v2 group
    ↓ [Lyse consolidates all results]
    ↓
NAS: seq_26.hdf (701 rows × 170 cols) ← Lyse output
    ↓ [run_show_ODs_batch consolidates]
    ↓
Reanalyzed: seq_26_seq_30.hdf ← Selection of key columns
```

### Load and Filter
```
Both data sources supported by multishot_lib:
    ↓
get_day_data(hdf_config)
    ├─ Load from NAS (reanalyzed=False)
    └─ Load from data_reanalyzed (reanalyzed=True)
    ↓
get_and_filter_shots() 
    ├─ Check parameter consistency (11 parameters)
    ├─ Filter by quality metrics (saturation, background, norm error)
    └─ Return filtered DataFrame
```

## Column Availability

### After Processing Complete

**NAS Location** (seq_26.hdf):
- All 170 columns available
- Device params (1-114), show_ODs (115-130), show_ODs_v2 (131-138), show_raws (139-147), timing/RF (148-170)

**Reanalyzed Location** (seq_26_seq_30.hdf):
- Selected columns:
  - All 114 device/timing/RF parameters
  - 8 show_ODs_v2 quality metrics
  - Total: ~122 columns
- Optimized for typical multishot workflow

## Quality Filtering

**Automatic rejection criteria** (configurable in multishot_lib/config.py):
- Saturation contamination > 5% (default: 0.05)
- Background contamination > 0.1% (default: 0.001)
- Probe norm error > 10% (default: 0.1)

**Automatic parameter check**:
- All 11 processing parameters must be identical across shots
- If not: returns error report showing which parameters differ

## Configuration Files

### run_show_ODs_batch.py
```python
SEQS = [26, 30]                              # Sequences to process
HDF_CONFIG = {                               # Date configuration
    'today': False,
    'year': 2026, 'month': 4, 'day': 15
}
```

### waterfall_config_v2.py
```python
SEQS = [26, 30]                              # Already set
HDF_CONFIG = {                               # Already set
    'today': False,
    'year': 2026, 'month': 4, 'day': 15
}
```

### multishot_lib/config.py
```python
BACKGROUND_CONTAMINATION_THRESHOLD = 0.001   # 0.1%
SATURATION_THRESHOLD = 0.05                  # 5%
PROBE_NORM_ERROR_THRESHOLD = 0.1             # 10%
ENABLE_DMD_CHECK = False                     # DMD disabled per user request
```

## Quick Start

1. **Process sequences 26 & 30** (day 2026-04-15):
   ```bash
   python run_show_ODs_for_waterfall.py
   ```

2. **Load and filter**:
   ```python
   hdf_config = {'today': False, 'year': 2026, 'month': 4, 'day': 15, 'reanalyzed': True}
   df_filtered = get_and_filter_shots(hdf_config, 'PTAI_m1')
   ```

3. **Proceed to waterfall analysis** with filtered, consistent shots

## Key Files

| File | Purpose |
|------|---------|
| `run_show_ODs_batch.py` | Main batch processor |
| `run_show_ODs_for_waterfall.py` | Waterfall-integrated wrapper |
| `multishot_lib/__init__.py` | Exports public functions |
| `multishot_lib/main.py` | Core filtering logic (with reanalyzed support) |
| `multishot_lib/config.py` | Quality thresholds |
| `data_reanalyzed/` | Consolidated HDF storage |
| `QUICKSTART.md` | Quick reference |
| `REANALYSIS_WORKFLOW.md` | Full workflow documentation |

## Benefits

1. ✓ **Unified quality control** - All shots must meet same thresholds
2. ✓ **Parameter consistency** - Automatic detection of parameter drift
3. ✓ **Flexible data source** - Switch between NAS and reanalyzed with one flag
4. ✓ **Batch processing** - No interactive UI needed if configured
5. ✓ **Consolidated view** - Single HDF file per day for focused analysis
6. ✓ **Reproducible** - All parameters/thresholds in config files
