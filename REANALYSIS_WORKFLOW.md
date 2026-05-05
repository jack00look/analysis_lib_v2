# Multishot Analysis Workflow with Reanalyzed Data

## Overview

You now have a complete system for batch processing shot data and analyzing it with quality filtering:

```
Raw shot files (individual .h5)
    ↓
run_show_ODs_batch.py (extracts OD, parameters, metrics)
    ↓
Lyse framework (consolidates into per-day DataFrames)
    ↓
save_dataframe.py (saves to NAS: /home/rick/NAS542_dataBEC2/...)
    ↓
data_reanalyzed consolidation (saves to: analysislib_v2/data_reanalyzed/...)
    ↓
multishot_lib (loads + filters by quality + checks parameter consistency)
    ↓
Analysis results (filtered shots ready for interpretation)
```

## Step 1: Run show_ODs_v2 Analysis

Process raw shots for a specific date and sequences:

```bash
cd /home/rick/labscript-suite/userlib/analysislib/analysislib_v2

# Option A: Interactive selection
python run_show_ODs_batch.py

# Option B: Using waterfall_config_v2 settings
python run_show_ODs_for_waterfall.py

# Option C: Direct configuration in script
# Edit run_show_ODs_batch.py and set:
# SEQS = [26, 30]
# HDF_CONFIG = {'today': False, 'year': 2026, 'month': 4, 'day': 15}
python run_show_ODs_batch.py
```

**What this does:**
1. Finds all raw .h5 files in: `/home/rick/NAS542_dataBEC2/2026/04/15/0026/`, etc.
2. Runs `show_ODs()` on each file (no plots)
3. Extracts OD images, 1D projections, parameters, quality metrics
4. Saves to each file's `results/show_ODs_v2` group
5. Lyse's `save_dataframe.py` consolidates into: `/home/rick/NAS542_dataBEC2/2026/04/15/seq_26.hdf`
6. **NEW**: Also saves consolidated HDF to: `analysislib_v2/data_reanalyzed/2026/04/15/seq_26_seq_30.hdf`

## Step 2: Load and Filter with multishot_lib

Option A: Load from NAS (original Lyse results):
```python
from multishot_lib import get_day_data, get_and_filter_shots

hdf_config = {
    'today': False,
    'year': 2026,
    'month': 4,
    'day': 15,
    'reanalyzed': False  # Use NAS
}

df = get_day_data(hdf_config)  # 701 rows × 170 cols

df_filtered = get_and_filter_shots(
    hdf_config, 
    camera_name='PTAI_m1'
)
```

Option B: Load from reanalyzed consolidated data:
```python
hdf_config = {
    'today': False,
    'year': 2026,
    'month': 4,
    'day': 15,
    'reanalyzed': True  # Use data_reanalyzed
}

df = get_day_data(hdf_config)  # Consolidated data

df_filtered = get_and_filter_shots(
    hdf_config, 
    camera_name='PTAI_m1'
)
```

## Key Features

### Column Structure
Today's data (701 shots):
- **Columns 1-114**: Device parameters (B-fields, timings, etc.)
- **Columns 115-130**: `show_ODs` results
- **Columns 131-138**: `show_ODs_v2` results (quality metrics!)
  - `PTAI_m1_cnt_rel_atoms_sat` - saturation (reject if > 0.05)
  - `PTAI_m1_cnt_rel_atoms_back` - background (reject if > 0.001)
  - `PTAI_m1_probe_norm_err` - probe error (reject if > 0.1)
- **Columns 139-147**: `show_raws` results
- **Columns 148-170**: Timing and RF parameters

### Data Sources

**NAS Location** (original Lyse results):
```
/home/rick/NAS542_dataBEC2/2026/04/15/seq_26.hdf
```
- Contains: 701 rows, all 170 columns
- Source: `save_dataframe.py` consolidation

**Reanalyzed Location** (batch processing consolidation):
```
analysislib_v2/data_reanalyzed/2026/04/15/seq_26_seq_30.hdf
```
- Contains: All 701+189=890 rows (if processing sequences 26+30)
- Selected columns: Device params + timing + RF + show_ODs_v2 only
- Source: `run_show_ODs_batch.py` consolidation

### Configuration Options

In any script using `multishot_lib`:

```python
# Load from NAS (default)
hdf_config = {'today': False, 'year': 2026, 'month': 4, 'day': 15}

# Load from reanalyzed (new)
hdf_config = {
    'today': False, 
    'year': 2026, 
    'month': 4, 
    'day': 15,
    'reanalyzed': True
}

# Use today
hdf_config = {'today': True, 'reanalyzed': False}
```

## Quality Filtering

`get_and_filter_shots()` automatically:
1. ✓ Loads all shots for the day
2. ✓ Checks all 11 processing parameters are identical across shots
3. ✓ Rejects shots by quality metrics (configurable in `multishot_lib/config.py`):
   - Saturation contamination > 5%
   - Background contamination > 0.1%
   - Probe normalization error > 10%
4. ✓ Returns filtered DataFrame with stats

## Next Steps

1. Run `python run_show_ODs_for_waterfall.py` to process sequences 26 & 30
2. Check `data_reanalyzed/2026/04/15/seq_26_seq_30.hdf` was created
3. Use `multishot_lib` with `reanalyzed: True` to load filtered data
4. Integrate with `waterfall_v2` for full analysis pipeline
