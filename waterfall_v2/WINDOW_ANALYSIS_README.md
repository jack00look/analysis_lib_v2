# Sliding Window Analysis for KZ_det_scan_window Mode

## Overview

The sliding window analysis performs a comprehensive spatial analysis of domain magnetization and structure across the sample. For each spatial window, it:

1. Computes average magnetization in that window for each shot
2. Groups shots by field value
3. Finds the field where average magnetization is closest to zero
4. Reports average number of domains at that field
5. Slides the window across the spatial range

## Workflow

### Step 1: Window Definition
```
Window 1: [x_min, x_min + window_size]
Window 2: [x_min + window_step, x_min + window_step + window_size]
Window 3: [x_min + 2*window_step, x_min + 2*window_step + window_size]
...
```

Window center for each: `(x_start + x_end) / 2`

### Step 2: Per-Shot Analysis in Window

For each shot in each window:
- Extract magnetization values: `M[x_start <= x < x_end]`
- Compute average: `<M> = mean(M[window])`
- Count domain walls in window

### Step 3: Group by Field

For each unique field value (ARPKZ_final_set_field):
- Collect all average magnetizations from shots at that field
- Compute statistics:
  - Mean magnetization: `mag_mean = mean(<M>)`
  - Std deviation: `mag_std = std(<M>)`
  - Standard error: `mag_sem = std(<M>) / sqrt(n)`
  - Mean domains: `domain_mean = mean(n_domains)`
  - Std domains: `domain_std = std(n_domains)`

### Step 4: Find Zero-Crossing Field

Find field value where `|mag_mean|` is minimized:
- This is the field where domain magnetization crosses zero
- Extract corresponding domain count statistics

### Step 5: Move Window

Repeat for next window at `x_min + window_step`, continue until `x_max`

## Output Structure

### JSON Format (`window_analysis.json`)

```json
[
  {
    "window_num": 0,
    "window_center_um": 40.0,
    "window_start_um": 0.0,
    "window_end_um": 80.0,
    "field_at_zero": 0.0012456,
    "mag_mean_at_zero": 0.00023,
    "mag_std_at_zero": 0.15234,
    "mag_sem_at_zero": 0.04521,
    "domain_mean_at_zero": 3.45,
    "domain_std_at_zero": 0.82,
    "n_shots_at_zero": 12,
    "field_summary": {
      "0.0010000": {
        "mag_mean": -0.23456,
        "mag_std": 0.15234,
        "mag_sem": 0.04521,
        "defect_mean": 5.2,
        "defect_std": 1.3,
        "domain_mean": 3.1,
        "domain_std": 0.8,
        "n_shots": 12
      },
      ...
    }
  },
  {
    "window_num": 1,
    "window_center_um": 45.0,
    "window_start_um": 5.0,
    "window_end_um": 85.0,
    ...
  },
  ...
]
```

### CSV Format (`window_analysis.csv`)

Concise summary with one row per window:

```csv
window_num,window_center_um,window_start_um,window_end_um,field_at_zero,mag_mean_at_zero,mag_sem_at_zero,domain_mean_at_zero,domain_std_at_zero,n_shots_at_zero
0,40.0,0.0,80.0,0.0012456,0.00023,0.04521,3.45,0.82,12
1,45.0,5.0,85.0,0.0012389,0.00145,0.04123,3.52,0.79,12
2,50.0,10.0,90.0,0.0012567,−0.00089,0.04678,3.38,0.85,12
...
```

## Configuration

Configure windows in `KZ_det_scan_window.py`:

```python
MODE_CONFIG = {
    ...
    'window_size': 80.0,      # μm, width of each window
    'window_step': 5.0,       # μm, step size between windows
    ...
}
```

Integration region in `PARAMS`:

```python
'DOMAIN_WALL_X_MIN': 0.0,      # μm, leftmost x
'DOMAIN_WALL_X_MAX': 200.0,    # μm, rightmost x
```

## Usage

### Run analysis:
```bash
cd /home/rick/labscript-suite/userlib/analysislib/analysislib_v2/waterfall_v2
python sliding_window_analysis.py
```

### In Python code:
```python
from waterfall.window_analysis_lib import sliding_window_analysis, results_to_dataframe

results = sliding_window_analysis(
    df,
    x_min_um=0.0,
    x_max_um=200.0,
    window_size_um=80.0,
    window_step_um=5.0,
    mode_cfg=MODE_CONFIG,
    params=PARAMS,
    domain_info_dict=domain_info_dict
)

# Convert to DataFrame for plotting
df_results = results_to_dataframe(results)
```

## Output Files

Results are saved to `./results/window_analysis/`:

- **`window_analysis.json`**: Complete results with full field_summary for each window
- **`window_analysis.csv`**: Summary table with key metrics

## Data Fields

Each window result includes:

| Field | Type | Description |
|-------|------|-------------|
| `window_num` | int | Window index (0, 1, 2, ...) |
| `window_center_um` | float | Center position of window in μm |
| `window_start_um` | float | Left edge of window in μm |
| `window_end_um` | float | Right edge of window in μm |
| `field_at_zero` | float | ARPKZ field value where mag ≈ 0 |
| `mag_mean_at_zero` | float | Average magnetization at zero field |
| `mag_sem_at_zero` | float | Standard error of magnetization |
| `domain_mean_at_zero` | float | Average number of domains at zero field |
| `domain_std_at_zero` | float | Std dev of domain count at zero field |
| `n_shots_at_zero` | int | Number of shots at the zero field |
| `field_summary` | dict | Full statistics for all field values |

## Interpretation

### Key Findings:

1. **`window_center_um`**: Spatial location being analyzed
2. **`field_at_zero`**: Field value showing balanced/zero magnetization in that region
3. **`domain_mean_at_zero`**: How many domains are present at field balance
4. **`mag_sem_at_zero`**: Uncertainty in finding zero magnetization

### Trends to look for:

- Does `field_at_zero` change with position? → Field inhomogeneity
- Does `domain_mean_at_zero` vary? → Spatial domain structure variation
- Is `mag_sem_at_zero` consistent? → Shot-to-shot reproducibility

## Further Analysis

Results can be used for:

1. **Plotting magnetization profile vs position**
   - X-axis: `window_center_um`
   - Y-axis: `field_at_zero`
   - Shows spatial field dependence

2. **Domain nucleation analysis**
   - X-axis: `window_center_um`
   - Y-axis: `domain_mean_at_zero`
   - Track how many domains form at zero magnetization

3. **Uncertainties and stability**
   - Use `mag_sem_at_zero` for error bars
   - Use `domain_std_at_zero` for domain count uncertainties

4. **Comparison with model**
   - Predict expected `field_at_zero` from theory
   - Compare `domain_mean_at_zero` with model predictions

## Memory Notes

- JSON output includes full `field_summary` for all field values in each window (useful for detailed analysis)
- CSV output is more compact (summary only)
- For large datasets (many windows × many fields), JSON can be large

## Debugging

If windows show no data:

1. Check `DOMAIN_WALL_X_MIN` and `DOMAIN_WALL_X_MAX` bounds
2. Check `window_size` isn't larger than domain region
3. Verify domain info extraction worked (check domain_info.json)
4. Check field values in data using:
   ```python
   print(df[field_col].unique())
   ```

## Example Results Interpretation

```
Window 0 [0-80 μm]:
  Position: 40 μm
  At field 0.00125 T: mag ≈ 0, 3.5 ± 0.8 domains
  
Window 1 [5-85 μm]:
  Position: 45 μm
  At field 0.00124 T: mag ≈ 0, 3.5 ± 0.8 domains
  
→ Field for zero magnetization stable around 0.00124-0.00125 T across the sample
→ Domain count consistent at ~3.5 domains
```
