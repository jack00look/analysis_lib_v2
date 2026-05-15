# KZ Article Analysis Framework

A comprehensive framework for analyzing and reanalyzing KZ detuning scan datasets for publication quality control and parameter tuning.

## Structure

```
KZ_article/
├── config.yaml                          # Main config with dataset list
├── camera_settings.yaml                 # Camera configuration (shared across article)
├── defect_config.yaml                   # Domain wall defect-finding algorithm parameters
├── run_kz_article_analysis.py           # Interactive analysis script
├── batch_kz_article_analysis.py         # Batch (non-interactive) analysis script
├── README.md                            # This file
└── results/
    └── YYYY/MM/DD/
        ├── iteration_1/
        │   ├── config.yaml              # Configs used in this iteration
        │   ├── camera_settings.yaml
        │   ├── defect_config.yaml
        │   ├── metadata.json
        │   └── dataset_001/             # Per-dataset results
        │       ├── 01_waterfall.png
        │       ├── 02_domain_wall.png
        │       └── ...
        ├── iteration_2/
        └── ...
```

## Workflow

### Step 1: Setup Initial Configuration

1. **Edit `config.yaml`**:
   - Add your datasets with (year, month, day, sequences)
   - Set initial affine correction coefficients (usually 1.0, 0, 0 for defaults)
   - Set x_min/x_max integration (pixel range for domain wall analysis)
   - Set domain_balance_fit_window (x range for fit in micrometers)

2. **Edit `camera_settings.yaml`**:
   - Configure camera name, atom images, data_origin
   - This is typically shared across all datasets in the article

3. **Edit `defect_config.yaml`**:
   - Configure domain wall defect-finding algorithm parameters
   - Typically shared across all datasets

### Step 2: Interactive Tuning (First Run)

Run the interactive script to visually tune parameters for each dataset:

```bash
cd /home/rick/labscript-suite/userlib/analysislib/analysislib_v2/article_analysis/KZ_article

python run_kz_article_analysis.py config.yaml camera_settings.yaml defect_config.yaml
```

**Workflow**:
1. For each dataset:
   - Loads raw data and applies shot quality filters
   - Shows 5 random m1/m2 profiles from different detunings
   - Prompts to adjust affine_correction coefficients (a1, b1, c1, a2, b2, c2)
   - Prompts to adjust x_min_integration, x_max_integration
   - Runs waterfall_v2 KZ_det_scan analysis
   - Shows domain wall plots
   - Prompts to adjust domain_balance_fit_window
   - Asks for permission to continue
2. Saves iteration folder with all configs and plots

**Output**:
- All results saved to: `results/YYYY/MM/DD/iteration_1/`
- Saved files can be used exactly for reproducible analysis

### Step 3: Batch Reanalysis (Reproducible Runs)

Once you're satisfied with parameters, use batch script for automated reruns:

```bash
# Standard run (shows plots)
python batch_kz_article_analysis.py config.yaml camera_settings.yaml defect_config.yaml

# Silent mode (save plots without displaying)
python batch_kz_article_analysis.py config.yaml camera_settings.yaml defect_config.yaml --silent

# Rerun specific iteration
python batch_kz_article_analysis.py config.yaml camera_settings.yaml defect_config.yaml --iteration 3
```

**Features**:
- No user prompts (parameters from config.yaml)
- Saves all plots silently (use `--silent` flag)
- Can re-run any iteration exactly
- Perfect for "publish reproducible results"

## Shot Quality Filtering

All scripts apply shot quality filtering using multishot_lib:

```yaml
SHOT_FILTER_CONFIG:
  enabled: true
  apply_back_check: true              # Background contamination check
  apply_sat_check: true               # Saturation check
  apply_probe_norm_err_check: true    # Probe norm error check
  back_threshold: 0.001
  sat_threshold: 1.0
  probe_norm_err_threshold: 0.1
  ntot_threshold: 7.0e5               # Common n_tot threshold (always enabled)
```

The `ntot_threshold` is applied consistently across all datasets for uniformity.

## Output Structure

Each iteration saves:
- `config.yaml` - Analyzed datasets with final parameters
- `camera_settings.yaml` - Camera settings used
- `defect_config.yaml` - Defect algorithm settings
- `metadata.json` - Analysis metadata (timestamp, user, num_datasets)
- `dataset_XXX/` - Per-dataset folders with:
  - Waterfall magnitude/density plots
  - Domain wall evolution plots
  - Sigmoid fit analysis plots
  - All individual shot waterfall plots

## Reproducibility

To exactly reproduce a previous iteration:

```bash
# Copy iteration folder config files
cp results/YYYY/MM/DD/iteration_3/config.yaml config_backup.yaml
cp results/YYYY/MM/DD/iteration_3/camera_settings.yaml camera_backup.yaml
cp results/YYYY/MM/DD/iteration_3/defect_config.yaml defect_backup.yaml

# Run batch script with those configs
python batch_kz_article_analysis.py config_backup.yaml camera_backup.yaml defect_backup.yaml
```

This will produce identical results (same datasets, same parameters, same filtering, same algorithm settings).

## Key Parameters

### Affine Correction
Corrects for gain imbalances between m1 and m2 channels:
```
n1D_m1' = a1*n1D_m1 + b1*n1D_m2 + c1
n1D_m2' = a2*n1D_m2 + b2*n1D_m1 + c2
```

Default: `a1=1, b1=0, c1=0, a2=1, b2=0, c2=0` (no correction)

### Integration Limits (micrometers)
X-range where magnetization profiles are integrated and analyzed.
Typical values: 900-1200 μm depending on cloud size and imaging geometry.

### Domain Wall Fit Window (micrometers)
X-range used for linear fit of domain wall position vs time.
Used to calculate domain wall velocity.
Typical values: [910, 1150] μm

## Tips for Tuning

1. **Affine Correction**: Look for systematic offsets between m1/m2 profiles in the 5 sample plots
2. **Integration Limits**: Choose range that captures main domain wall region without noise
3. **Domain Fit Window**: Use range where domain wall is well-defined and moves monotonically
4. **Iteration Workflow**: 
   - Iteration 1: Interactive tuning
   - Iteration 2+: Batch reruns with minor parameter tweaks

## Common Issues

**No data after filtering**: Check that sequences and date are correct. Some shots may be rejected if background/saturation/probe_norm_err thresholds are too strict.

**Domain wall fit fails**: If domain wall velocity seems wrong, check `domain_balance_fit_window` - make sure it's centered on actual domain wall evolution.

**Affine correction causes issues**: If corrected profiles look worse, keep default values (a1=1, b1=0, c1=0, a2=1, b2=0, c2=0).

