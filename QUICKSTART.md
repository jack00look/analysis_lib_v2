# Quick Start: Batch Processing and Multishot Filtering

## Current Configuration

`run_show_ODs_batch.py` is pre-configured to process:
- **Date**: 2026-04-15
- **Sequences**: 26, 30
- **Files**: 21 + 189 = 210 total .h5 files

## To Process Sequences 26 & 30:

```bash
cd /home/rick/labscript-suite/userlib/analysislib/analysislib_v2

# Method 1: Direct run (uses embedded config)
python run_show_ODs_batch.py

# Method 2: Use waterfall_config_v2
python run_show_ODs_for_waterfall.py

# Method 3: Interactive (to change date/sequences)
# Edit run_show_ODs_batch.py:
#   SEQS = None
#   HDF_CONFIG = None
# Then run and follow prompts
```

**Estimated time**: ~5-10 minutes to process 210 files

**Output**:
1. Each .h5 file gets `results/show_ODs_v2` group with analysis
2. NAS: `/home/rick/NAS542_dataBEC2/2026/04/15/seq_26.hdf` (701 rows × 170 cols)
3. Reanalyzed: `analysislib_v2/data_reanalyzed/2026/04/15/seq_26_seq_30.hdf` (890 rows, selected cols)

## To Analyze with Filtering:

```python
import sys
sys.path.insert(0, '/home/rick/labscript-suite/userlib/analysislib/analysislib_v2')

from multishot_lib import get_day_data, get_and_filter_shots

# Load from reanalyzed consolidated data
hdf_config = {
    'today': False,
    'year': 2026,
    'month': 4,
    'day': 15,
    'reanalyzed': True
}

# Get filtered data
df_filtered = get_and_filter_shots(
    hdf_config,
    camera_name='PTAI_m1'
)

print(f"Shots after filtering: {len(df_filtered)}")
```

## Column Reference

Available after running show_ODs_batch:

**Quality Metrics** (for filtering):
- `('show_ODs_v2', 'PTAI_m1_cnt_rel_atoms_sat')` - saturation (reject > 0.05)
- `('show_ODs_v2', 'PTAI_m1_cnt_rel_atoms_back')` - background (reject > 0.001)  
- `('show_ODs_v2', 'PTAI_m1_probe_norm_err')` - probe error (reject > 0.1)

**Analysis Results**:
- `('show_ODs_v2', 'PTAI_m1_OD_lin_sum')` - linear OD sum
- `('show_ODs_v2', 'PTAI_m1_OD_log_sum')` - log OD sum

**Device Parameters** (device parameters):
- All 114 device/timing/RF parameters remain available

## Configuration

To change sequences/date, edit `run_show_ODs_batch.py`:

```python
SEQS = [26, 30]  # or None to prompt
HDF_CONFIG = {'today': False, 'year': 2026, 'month': 4, 'day': 15}  # or None
```

Or edit `waterfall_config_v2.py` to change waterfall settings and it will be picked up automatically by `run_show_ODs_for_waterfall.py`.

## File Locations

- Script: `analysislib_v2/run_show_ODs_batch.py`
- Wrapper: `analysislib_v2/run_show_ODs_for_waterfall.py`
- Lib: `analysislib_v2/multishot_lib/`
- Reanalyzed data: `analysislib_v2/data_reanalyzed/2026/04/15/seq_26_seq_30.hdf`
- NAS data: `/home/rick/NAS542_dataBEC2/2026/04/15/seq_26.hdf`
