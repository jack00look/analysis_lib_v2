# Data Reanalyzed Directory

This directory stores consolidated HDF5 files from batch reanalysis runs using `run_show_ODs_batch.py`.

## Structure

```
data_reanalyzed/
├── 2026/
│   ├── 04/
│   │   ├── 15/
│   │   │   ├── seq_0026_seq_0030.hdf      # Consolidated results from sequences 26 and 30
│   │   │   └── seq_0001_seq_0002_seq_0003.hdf
│   │   └── 16/
│   │       └── seq_0001.hdf
│   └── 05/
│       └── ...
```

## Contents

Each HDF file contains:
- All device/timing/RF parameters (columns 1-170 excluding analysis results)
- All `show_ODs_v2` analysis results
- MultiIndex DataFrame format matching Lyse output

## Usage with multishot_lib

Set `reanalyzed: True` in `hdf_config`:

```python
hdf_config = {
    'reanalyzed': True,
    'year': 2026,
    'month': 4,
    'day': 15
}

df = get_day_data(hdf_config)  # Loads from data_reanalyzed
```

Or leave `reanalyzed: False` (default) to load from NAS:

```python
hdf_config = {
    'reanalyzed': False,
    'year': 2026,
    'month': 4,
    'day': 15
}

df = get_day_data(hdf_config)  # Loads from /home/rick/NAS542_dataBEC2
```

## Generation

Files are created by running:

```bash
python run_show_ODs_batch.py
# or
python run_show_ODs_for_waterfall.py
```

The script automatically:
1. Processes raw shot files through `show_ODs_v2`
2. Extracts device parameters and analysis results
3. Consolidates into per-day HDF files
4. Saves to `data_reanalyzed/YYYY/MM/DD/seq_*.hdf`
